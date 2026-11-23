"""Frame review through `claude -p`, on a fake process: command, message, answer, errors.

The output lines follow what Claude Code 2.1.325 printed for a stream-json run with
--json-schema: init (tools: StructuredOutput only), the assistant turns, then `result`.
"""

import base64
import json
import subprocess
from typing import Any

import pytest

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.images.fake import gradient_png
from pipeline.adapters.vision import FrameCheckRequest, FrameVerdict, media_type_of
from pipeline.adapters.vision.claude_cli import (
    MAX_IMAGE_BYTES,
    ClaudeCliReviewer,
    answer_schema,
    last_result,
)
from pipeline.adapters.vision.fake import FakeReviewer, clean_verdict

PNG = gradient_png(72, 128, (46, 64, 87), (232, 216, 176))
REQUEST = FrameCheckRequest(
    image=PNG,
    media_type="image/png",
    scene="A bare-nosed wombat standing on a grassy hillside",
    style="Clean digital natural history painting",
)
ANSWER: dict[str, Any] = {
    "subject_seen": "a brown wombat on grass",
    "right_subject": True,
    "extra_seen": "",
    "extra_object": False,
    "text_seen": "",
    "has_text": False,
    "subject_position": "middle",
    "style_matches": True,
    "unwanted_seen": "",
    "unwanted": False,
}


def stream(*events: dict[str, Any]) -> bytes:
    init = {"type": "system", "subtype": "init", "tools": ["StructuredOutput"]}
    return "\n".join(json.dumps(e) for e in (init, *events)).encode() + b"\n"


def result(**fields: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "",
        "structured_output": ANSWER,
        "total_cost_usd": 0.021,
    }
    event.update(fields)
    return event


class FakeRun:
    def __init__(self, stdout: bytes, returncode: int = 0, stderr: bytes = b"") -> None:
        self.reply = subprocess.CompletedProcess([], returncode, stdout, stderr)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append({"command": command, **kwargs})
        return self.reply


def reviewer(run: FakeRun) -> ClaudeCliReviewer:
    return ClaudeCliReviewer("test-model", timeout_s=42, run=run, which=lambda _: "/bin/claude")


# --- what is sent ----------------------------------------------------------------------


def test_the_command_streams_json_both_ways_with_the_schema_and_no_tools() -> None:
    run = FakeRun(stream(result()))
    reviewer(run).review(REQUEST)

    command = run.calls[0]["command"]
    flag = {command[i]: command[i + 1] for i in range(1, len(command) - 1)}
    assert flag["--input-format"] == "stream-json"
    assert flag["--output-format"] == "stream-json"  # 2.1.325 refuses stream-json in, json out
    assert "--verbose" in command
    assert json.loads(flag["--json-schema"])["required"] == list(ANSWER)
    assert " " not in flag["--json-schema"]
    assert flag["--tools"] == ""
    assert flag["--model"] == "test-model"
    assert {"--safe-mode", "--no-session-persistence"} <= set(command)
    assert run.calls[0]["timeout"] == 42


def test_the_frame_goes_through_stdin_as_an_image_block_then_the_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-would-bill-the-api")
    run = FakeRun(stream(result()))
    reviewer(run).review(REQUEST)

    stdin: bytes = run.calls[0]["input"]
    assert stdin.count(b"\n") == 1  # one stream-json line
    message = json.loads(stdin)
    image, text = message["message"]["content"]
    assert message["type"] == "user"
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == PNG
    assert REQUEST.scene in text["text"] and REQUEST.style in text["text"]
    assert "ANTHROPIC_API_KEY" not in run.calls[0]["env"]
    assert REQUEST.scene not in " ".join(run.calls[0]["command"])  # never on the command line


def test_the_schema_matches_the_fields_that_are_read() -> None:
    schema = json.loads(answer_schema())

    assert set(schema["properties"]) == set(ANSWER)
    assert schema["additionalProperties"] is False


# --- the answer ------------------------------------------------------------------------


def test_a_clean_answer_is_accepted_with_its_cost() -> None:
    verdict = reviewer(FakeRun(stream(result()))).review(REQUEST)

    assert verdict.accepted
    assert verdict.problems() == []
    assert verdict.subject_position == "middle"
    assert verdict.cost_usd == pytest.approx(0.021)
    assert (verdict.provider, verdict.model) == ("claude_cli", "test-model")


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"right_subject": False, "subject_seen": "a hamster"}, "wrong subject: a hamster"),
        ({"extra_object": True, "extra_seen": "a moon surface below"}, "extra object: a moon"),
        ({"has_text": True, "text_seen": "a signature, bottom left"}, "text in the frame"),
        ({"unwanted": True, "unwanted_seen": "red patches like blood"}, "unwanted content"),
        ({"style_matches": False}, "off-style"),
    ],
)
def test_each_problem_rejects_the_frame(changes: dict[str, Any], problem: str) -> None:
    answer = {**ANSWER, **changes}
    verdict = reviewer(FakeRun(stream(result(structured_output=answer)))).review(REQUEST)

    assert not verdict.accepted
    assert any(p.startswith(problem) for p in verdict.problems())


def test_where_the_subject_sits_only_informs() -> None:
    answer = {**ANSWER, "subject_position": "lower"}
    verdict = reviewer(FakeRun(stream(result(structured_output=answer)))).review(REQUEST)

    assert verdict.accepted


@pytest.mark.parametrize(
    "structured",
    [
        None,  # the run ended without valid structured output
        {**ANSWER, "has_text": "no"},  # a string where a boolean belongs
        {**ANSWER, "subject_position": "top"},
        {key: value for key, value in ANSWER.items() if key != "unwanted"},
        {**ANSWER, "mood": "calm"},
    ],
)
def test_an_answer_off_the_schema_is_transient(structured: dict[str, Any] | None) -> None:
    with pytest.raises(TransientAdapterError):
        reviewer(FakeRun(stream(result(structured_output=structured)))).review(REQUEST)


# --- failures --------------------------------------------------------------------------


def test_not_logged_in_is_permanent() -> None:
    # What 2.1.325 printed without a login: is_error true, subtype success.
    failed = result(
        is_error=True,
        structured_output=None,
        result="Authentication error · This may be a temporary network issue, please try again",
    )

    with pytest.raises(PermanentAdapterError, match="not logged in"):
        reviewer(FakeRun(stream(failed), returncode=1)).review(REQUEST)


def test_an_overloaded_api_is_transient() -> None:
    failed = result(
        is_error=True, structured_output=None, result="Overloaded", api_error_status=529
    )

    with pytest.raises(TransientAdapterError):
        reviewer(FakeRun(stream(failed), returncode=1)).review(REQUEST)


def test_a_crash_before_any_result_uses_stderr() -> None:
    run = FakeRun(b"", returncode=1, stderr=b"Error: unknown option '--tools'")

    with pytest.raises(TransientAdapterError, match="unknown option"):
        reviewer(run).review(REQUEST)


def test_a_missing_cli_or_a_huge_frame_is_refused_before_running() -> None:
    run = FakeRun(stream(result()))
    missing = ClaudeCliReviewer("m", run=run, which=lambda _: None)
    huge = FrameCheckRequest(b"\x89PNG\r\n\x1a\n" + b"\0" * MAX_IMAGE_BYTES, "image/png", "s", "s")

    with pytest.raises(PermanentAdapterError, match="not found"):
        missing.review(REQUEST)
    with pytest.raises(PermanentAdapterError, match="5 MB"):
        reviewer(run).review(huge)
    assert run.calls == []


def test_a_timeout_is_transient() -> None:
    def slow(command: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[bytes]":
        raise subprocess.TimeoutExpired(command, 42)

    with pytest.raises(TransientAdapterError, match="timed out"):
        ClaudeCliReviewer("m", run=slow, which=lambda _: "/bin/claude").review(REQUEST)


def test_the_last_result_wins_and_noise_is_skipped() -> None:
    text = 'noise\n{"type":"result","n":1}\n[1,2]\n{"type":"result","n":2}\n'

    assert last_result(text) == {"type": "result", "n": 2}
    assert last_result("nothing here") is None


# --- helpers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (PNG, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"\0" * 10, "image/jpeg"),
        (b"RIFF\0\0\0\0WEBPVP8 ", "image/webp"),
    ],
)
def test_the_media_type_comes_from_the_bytes(data: bytes, kind: str) -> None:
    assert media_type_of(data) == kind


def test_anything_else_is_not_an_image() -> None:
    with pytest.raises(ValueError, match="not a PNG"):
        media_type_of(b"GIF89a....")


def test_the_fake_accepts_and_records() -> None:
    fake = FakeReviewer()

    verdict = fake.review(REQUEST)

    assert isinstance(verdict, FrameVerdict) and verdict.accepted
    assert fake.calls == [REQUEST]
    assert clean_verdict(REQUEST).provider == "fake"
