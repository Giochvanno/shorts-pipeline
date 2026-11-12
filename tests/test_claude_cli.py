"""ClaudeCliLlm on a fake subprocess: command line, isolation, parsing, error mapping.

No real `claude` runs here (CI has neither the CLI nor a login). The live check is
tests/live/test_claude_cli_live.py, run by hand with `uv run pytest -m live`.
"""

import json
import random
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm import LlmRequest
from pipeline.adapters.llm.claude_cli import ClaudeCliLlm, classify_failure
from pipeline.adapters.llm.fake import fake_script
from pipeline.adapters.registry import Adapters
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.tts.fake import FakeTts
from pipeline.config import load_series
from pipeline.runner import DEFAULT_STEPS, RetryPolicy, Runner
from pipeline.state import StateStore, StepStatus
from pipeline.steps import script as script_step

MODEL = "model-from-config"
REQUEST = LlmRequest(
    task="script",
    system='Write a script.\nUse "quotes", newlines and café — all must survive.',
    prompt="Topic: octopus hearts",
)


def success(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": '{"structure": "x"}',
        "total_cost_usd": 0.0123,
        "usage": {
            "input_tokens": 12,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 50,
            "output_tokens": 40,
        },
    }
    data.update(overrides)
    return data


class FakeProcess:
    """Stands in for subprocess.run and records how it was called."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        raises: BaseException | None = None,
    ) -> None:
        self.stdout, self.stderr, self.returncode, self.raises = stdout, stderr, returncode, raises
        self.calls: list[dict[str, Any]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[bytes]":
        cwd = Path(kwargs["cwd"])
        self.calls.append(
            {"args": args, **kwargs, "cwd_existed": cwd.is_dir(), "cwd_files": list(cwd.iterdir())}
        )
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(
            args, self.returncode, self.stdout.encode(), self.stderr.encode()
        )


def _llm(process: FakeProcess, timeout_s: float = 30) -> ClaudeCliLlm:
    return ClaudeCliLlm(
        MODEL, timeout_s=timeout_s, run=process, which=lambda _name: "/usr/bin/claude"
    )


# --- how the CLI is called --------------------------------------------------------


def test_command_line_holds_only_fixed_flags() -> None:
    process = FakeProcess(stdout=json.dumps(success()))
    _llm(process).complete(REQUEST)

    (call,) = process.calls
    args = call["args"]
    assert args[:2] == ["/usr/bin/claude", "-p"]
    assert args[args.index("--output-format") + 1] == "json"
    assert args[args.index("--model") + 1] == MODEL
    assert args[args.index("--tools") + 1] == ""  # no tools: answer only
    assert args[args.index("--permission-prompts") + 1] == "none"  # never wait for a human
    assert "--no-session-persistence" in args
    assert "--safe-mode" in args
    assert "--bare" not in args  # --bare would ignore the subscription login
    joined = " ".join(args)
    assert "octopus" not in joined  # the prompt never goes on the command line
    assert "\n" not in joined


def test_prompt_goes_through_stdin_as_utf8() -> None:
    process = FakeProcess(stdout=json.dumps(success()))
    _llm(process).complete(REQUEST)

    stdin: bytes = process.calls[0]["input"]
    text = stdin.decode("utf-8")
    assert REQUEST.system in text
    assert REQUEST.prompt in text
    assert "café" in text


def test_runs_in_an_empty_temp_folder_that_is_removed(tmp_path: Path) -> None:
    process = FakeProcess(stdout=json.dumps(success()))
    _llm(process).complete(REQUEST)

    call = process.calls[0]
    cwd = Path(call["cwd"])
    assert call["cwd_existed"]
    assert call["cwd_files"] == []  # nothing for the agent to read
    assert Path.cwd() not in [cwd, *cwd.parents]  # not the project folder
    assert not cwd.exists()


def test_api_key_is_hidden_so_the_subscription_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    process = FakeProcess(stdout=json.dumps(success()))
    _llm(process).complete(REQUEST)

    assert "ANTHROPIC_API_KEY" not in process.calls[0]["env"]


def test_timeout_is_passed_to_the_process() -> None:
    process = FakeProcess(stdout=json.dumps(success()))
    _llm(process, timeout_s=42).complete(REQUEST)

    assert process.calls[0]["timeout"] == 42


# --- parsing ----------------------------------------------------------------------


def test_parses_text_tokens_and_cost() -> None:
    response = _llm(FakeProcess(stdout=json.dumps(success()))).complete(REQUEST)

    assert response.text == '{"structure": "x"}'
    assert response.provider == "claude_cli"
    assert response.model == MODEL
    assert response.input_tokens == 12 + 300 + 50  # cache tokens are input too
    assert response.output_tokens == 40
    assert response.cost_usd == pytest.approx(0.0123)


def test_tolerates_stray_lines_around_the_json() -> None:
    stdout = "warning: something minor\n" + json.dumps(success())
    response = _llm(FakeProcess(stdout=stdout)).complete(REQUEST)

    assert response.output_tokens == 40


def test_empty_result_is_transient() -> None:
    with pytest.raises(TransientAdapterError, match="empty"):
        _llm(FakeProcess(stdout=json.dumps(success(result="  ")))).complete(REQUEST)


# --- errors -----------------------------------------------------------------------


def test_timeout_is_transient() -> None:
    hung = FakeProcess(raises=subprocess.TimeoutExpired(cmd="claude", timeout=30))

    with pytest.raises(TransientAdapterError, match="timed out"):
        _llm(hung).complete(REQUEST)


def test_missing_cli_is_permanent() -> None:
    llm = ClaudeCliLlm(MODEL, run=FakeProcess(), which=lambda _name: None)

    with pytest.raises(PermanentAdapterError, match="not found in PATH"):
        llm.complete(REQUEST)


@pytest.mark.parametrize(
    ("message", "kind", "hint"),
    [
        ("Invalid API key · Please run /login", PermanentAdapterError, "not logged in"),
        ("OAuth token has expired", PermanentAdapterError, "not logged in"),
        (
            "Claude usage limit reached. Your limit will reset at 5pm",
            PermanentAdapterError,
            "limit",
        ),
        ("API Error: model: no-such-model not found", PermanentAdapterError, "model"),
        ("API Error: 529 Overloaded", TransientAdapterError, "Overloaded"),
        ("API Error: 429 rate_limit_error", TransientAdapterError, "429"),
        ("Request timed out", TransientAdapterError, "timed out"),
        ("something nobody has seen before", TransientAdapterError, "failed"),
    ],
)
def test_failures_are_sorted_for_the_runner(message: str, kind: type[Exception], hint: str) -> None:
    error = classify_failure(message)

    assert isinstance(error, kind)
    assert hint.lower() in str(error).lower()


def test_error_result_in_json_is_classified() -> None:
    data = success(is_error=True, subtype="success", result="Claude usage limit reached")

    with pytest.raises(PermanentAdapterError, match="usage limit"):
        _llm(FakeProcess(stdout=json.dumps(data), returncode=1)).complete(REQUEST)


def test_plain_stderr_failure_is_classified() -> None:
    process = FakeProcess(stderr="Error: Invalid API key · Please run /login", returncode=1)

    with pytest.raises(PermanentAdapterError, match="not logged in"):
        _llm(process).complete(REQUEST)


def test_garbage_output_is_transient() -> None:
    with pytest.raises(TransientAdapterError):
        _llm(FakeProcess(stdout="<html>proxy error</html>", returncode=0)).complete(REQUEST)


# --- end to end: the runner with claude_cli on a fake process -----------------------


def test_runner_records_claude_cli_calls_as_subscription(tmp_path: Path) -> None:
    topic = "how glass is made"
    structure = script_step.choose_structure(load_series("fact-of-day"), 1)
    script_json = fake_script(
        LlmRequest(
            task="script", system="", prompt="", tags={"topic": topic, "structure": structure}
        )
    )
    process = FakeProcess(stdout=json.dumps(success(result=script_json, total_cost_usd=0.02)))
    llm = _llm(process)
    adapters = Adapters(
        llm_heavy=llm, llm_light=llm, tts=FakeTts(), images=FakeImages(), stt=FakeTranscriber()
    )
    steps = list(DEFAULT_STEPS[:1])  # the script step only: the part this adapter serves

    with StateStore(tmp_path / "pipeline.db") as store:
        runner = Runner(
            store=store,
            adapters=adapters,
            runs_dir=tmp_path / "runs",
            policy=RetryPolicy(max_attempts=1),
            steps=steps,
            sleep=lambda _s: None,
            rng=random.Random(0),
        )
        video = runner.create_video("fact-of-day", topic)
        report = runner.run_video(video)
        (cost,) = store.costs(video)

    assert report.status == StepStatus.DONE
    assert (cost.provider, cost.billing, cost.unit) == ("claude_cli", "subscription", "tokens")
    assert cost.cost_usd == pytest.approx(0.02)
    assert report.cost_usd == pytest.approx(0.02)


# --- real recorded output (tests/fixtures/claude_cli, Claude Code 2.1.322) ----------

FIXTURES = Path(__file__).parent / "fixtures" / "claude_cli"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_real_success_output_is_parsed() -> None:
    response = _llm(FakeProcess(stdout=_fixture("success.json"))).complete(REQUEST)

    assert response.text == "ok"
    assert response.input_tokens == 2 + 10339 + 14517
    assert response.output_tokens == 4
    assert response.cost_usd == pytest.approx(0.00221517)


@pytest.mark.parametrize("banner_on", ["stderr", "stdout"])
def test_real_unknown_model_error_is_permanent(banner_on: str) -> None:
    banner, body = _fixture("error_unknown_model.banner"), _fixture("error_unknown_model.stdout")
    process = (
        FakeProcess(stdout=body, stderr=banner, returncode=1)
        if banner_on == "stderr"
        else FakeProcess(stdout=banner + body, returncode=1)
    )

    with pytest.raises(PermanentAdapterError, match="not available to this account"):
        _llm(process).complete(REQUEST)  # retrying a wrong model ID is pointless


@pytest.mark.parametrize(
    ("status", "message", "kind", "hint"),
    [
        (401, "authentication_error", PermanentAdapterError, "not logged in"),
        (403, "permission_error", PermanentAdapterError, "not logged in"),
        (404, "not_found_error", PermanentAdapterError, "model"),
        (400, "invalid_request_error", PermanentAdapterError, "rejected"),
        (529, "overloaded_error", TransientAdapterError, "overloaded"),
        (500, "api_error", TransientAdapterError, "api_error"),
        (429, "rate_limit_error", TransientAdapterError, "rate_limit"),
        (429, "Claude usage limit reached", PermanentAdapterError, "usage limit"),
    ],
)
def test_api_error_status_decides_before_text(
    status: int, message: str, kind: type[Exception], hint: str
) -> None:
    data = success(is_error=True, result=message, api_error_status=status)

    with pytest.raises(kind) as exc:
        _llm(FakeProcess(stdout=json.dumps(data), returncode=1)).complete(REQUEST)

    assert hint.lower() in str(exc.value).lower()
