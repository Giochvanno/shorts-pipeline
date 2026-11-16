"""Real LLM adapter: Claude through the Claude Code CLI (`claude -p`), on the Claude subscription.

One call = one `claude -p` process:

- Only fixed, safe tokens go on the command line. The prompt goes through stdin
  as UTF-8: Windows limits command-line length and mangles quotes and newlines.
- The process runs in an empty temporary folder, with no tools, no session saved,
  and `--safe-mode` (no CLAUDE.md, plugins, hooks or MCP from the user's setup):
  Claude Code is an agent, and here it must only answer the prompt.
- ANTHROPIC_API_KEY is removed from the child's environment: if it were set,
  the CLI would silently bill the API instead of the subscription.
- Errors are sorted for the runner: hiccups (timeout, overload, rate limit) are
  TransientAdapterError and get retried; things only a human can fix (not logged
  in, usage limit reached, unknown model, CLI missing) are PermanentAdapterError.

Flags were checked against `claude --help` of Claude Code 2.1.322.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.llm.base import LlmRequest, LlmResponse

PROVIDER = "claude_cli"

# Replaces Claude Code's default agent prompt. No quotes or newlines: safe in any shell.
SYSTEM_PROMPT = (
    "You are a careful writing assistant. Follow the instructions in the user message"
    " exactly and reply with the requested output only, with no preamble."
)

_NOT_LOGGED_IN = re.compile(
    r"not logged in|/login|log ?in again|authenticat|unauthori[sz]ed|\b401\b|oauth", re.I
)
_USAGE_LIMIT = re.compile(
    r"usage limit|limit reached|limit will reset|out of (credits|usage)|credit balance|quota",
    re.I,
)
_BAD_MODEL = re.compile(
    r"issue with the selected model|unrecognized_model"
    r"|model.{0,40}(not found|not available|invalid|does not exist|unknown)"
    r"|(invalid|unknown) model|no such model",
    re.I,
)
_LIMIT_HINT = "Claude subscription usage limit reached: wait for the reset"
_LOGIN_HINT = "Claude Code is not logged in: run `claude` once and sign in"
_MODEL_HINT = "the model in config/channel.yaml is not available to this account"
# HTTP statuses the CLI reports in `api_error_status`.
_TRANSIENT_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_TRANSIENT = re.compile(
    r"overloaded|\b529\b|\b5\d\d\b|rate.?limit|\b429\b|timeout|timed out|econnreset|"
    r"connection|network|temporar|try again",
    re.I,
)

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]


def classify_failure(message: str, status: int | None = None) -> Exception:
    """Turn a CLI failure into the error type the runner understands.

    The HTTP status from the JSON (`api_error_status`) decides first; the text is
    the fallback for failures that carry no status (CLI crashes, login problems).
    The usage limit is checked before 429, because both can look like "too many".
    """
    detail = _short(message)
    if _USAGE_LIMIT.search(message):
        return PermanentAdapterError(f"{_LIMIT_HINT} ({detail})")
    if status in {401, 403} or _NOT_LOGGED_IN.search(message):
        return PermanentAdapterError(f"{_LOGIN_HINT} ({detail})")
    if status == 404 or _BAD_MODEL.search(message):
        return PermanentAdapterError(f"{_MODEL_HINT} ({detail})")
    if status in _TRANSIENT_STATUSES or _TRANSIENT.search(message):
        return TransientAdapterError(f"claude -p: {detail}")
    if status == 400:
        return PermanentAdapterError(f"the request was rejected as invalid ({detail})")
    # Unknown: retry a few times; if it persists, the runner fails with this text.
    return TransientAdapterError(f"claude -p failed: {detail}")


def _short(text: str, limit: int = 300) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


class ClaudeCliLlm:
    def __init__(
        self,
        model: str,
        *,
        timeout_s: float = 180,
        executable: str = "claude",
        run: Runner = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
        popen: Callable[..., "subprocess.Popen[bytes]"] = subprocess.Popen,
    ) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._executable = executable
        self._run = run
        self._which = which
        self._popen = popen

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def model(self) -> str:
        return self._model

    def command(self, executable: str, *, output_format: str = "json") -> list[str]:
        return [
            executable,
            "-p",
            "--output-format", output_format,
            "--model", self._model,
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",
            "--permission-prompts", "none",
            "--no-session-persistence",
            "--safe-mode",
        ]  # fmt: skip

    def _executable_path(self) -> str:
        executable = self._which(self._executable)
        if executable is None:
            raise PermanentAdapterError(
                "Claude Code CLI not found in PATH: install it with"
                " `irm https://claude.ai/install.ps1 | iex` and sign in"
            )
        return executable

    @staticmethod
    def _stdin(request: LlmRequest) -> bytes:
        return f"{request.system.strip()}\n\n---\n\n{request.prompt.strip()}\n".encode()

    @staticmethod
    def _env() -> dict[str, str]:
        # Never the API key: the CLI would bill the API instead of the subscription.
        return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    def complete(self, request: LlmRequest) -> LlmResponse:
        executable = self._executable_path()
        stdin = self._stdin(request)
        env = self._env()
        with tempfile.TemporaryDirectory(prefix="claude_", ignore_cleanup_errors=True) as empty:
            try:
                proc = self._run(
                    self.command(executable),
                    input=stdin,
                    capture_output=True,
                    cwd=Path(empty),
                    env=env,
                    timeout=self._timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise TransientAdapterError(
                    f"claude -p timed out after {self._timeout_s:.0f} s"
                ) from e
            except OSError as e:
                raise TransientAdapterError(f"could not start claude: {e}") from e
        return self._parse(proc)

    def stream(
        self,
        request: LlmRequest,
        *,
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        """The same call as `complete`, shown while it is written (for pipeline.watch).

        `--output-format stream-json --verbose --include-partial-messages` prints one JSON
        event per line; text arrives as `stream_event` / `text_delta`, and the last line is
        the same `result` object `complete` reads. The pipeline itself uses `complete`.
        """
        executable = self._executable_path()
        command = [
            *self.command(executable, output_format="stream-json"),
            "--verbose",
            "--include-partial-messages",
        ]
        with tempfile.TemporaryDirectory(prefix="claude_", ignore_cleanup_errors=True) as empty:
            try:
                proc = self._popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=Path(empty),
                    env=self._env(),
                )
            except OSError as e:
                raise TransientAdapterError(f"could not start claude: {e}") from e
            stdin, stdout, stderr = proc.stdin, proc.stdout, proc.stderr
            assert stdin is not None and stdout is not None and stderr is not None  # all PIPE
            timer = threading.Timer(self._timeout_s, proc.kill)
            errors: list[bytes] = []
            drain = threading.Thread(target=lambda: errors.append(stderr.read()), daemon=True)
            timer.start()
            drain.start()
            try:
                stdin.write(self._stdin(request))
                stdin.close()
                result = _read_stream(stdout, on_text, on_thinking)
                returncode = proc.wait()
            finally:
                timed_out = not timer.is_alive()
                timer.cancel()
                drain.join(timeout=5)
        if timed_out and result is None:
            raise TransientAdapterError(f"claude -p timed out after {self._timeout_s:.0f} s")
        final = json.dumps(result).encode() if result is not None else b""
        return self._parse(
            subprocess.CompletedProcess(command, returncode, final, b"".join(errors))
        )

    def _parse(self, proc: "subprocess.CompletedProcess[bytes]") -> LlmResponse:
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        data = _json_object(stdout)
        if data is None:
            # No JSON at all: the CLI failed before answering (or printed garbage).
            raise classify_failure(stderr or stdout or f"exit code {proc.returncode}")
        failed = proc.returncode != 0 or bool(data.get("is_error"))
        if failed or data.get("subtype") not in {None, "success"}:
            # Seen in 2.1.322: is_error=true with subtype "success" and the reason in `result`.
            detail = str(data.get("result") or data.get("error") or data.get("subtype") or "")
            status = data.get("api_error_status")
            raise classify_failure(
                " ".join(filter(None, [detail, stderr])) or stdout,
                status if isinstance(status, int) else None,
            )
        text = data.get("result")
        if not isinstance(text, str) or not text.strip():
            raise TransientAdapterError("claude -p returned an empty result")
        usage: dict[str, Any] = data.get("usage") or {}
        input_tokens = sum(
            int(usage.get(key) or 0)
            for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        return LlmResponse(
            text=text,
            provider=PROVIDER,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens") or 0),
            cost_usd=float(data.get("total_cost_usd") or 0.0),
        )


def _read_stream(
    lines: IO[bytes],
    on_text: Callable[[str], None],
    on_thinking: Callable[[str], None] | None,
) -> dict[str, Any] | None:
    """Hand text and thinking chunks to the callbacks; return the final `result` event."""
    result: dict[str, Any] | None = None
    for raw in lines:
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue  # a stray line: the CLI's own noise, never part of the answer
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            result = event
        elif event.get("type") == "stream_event":
            inner = event.get("event")
            delta = inner.get("delta") if isinstance(inner, dict) else None
            if not isinstance(delta, dict):
                continue
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                on_text(delta["text"])
            elif delta.get("type") == "thinking_delta" and on_thinking is not None:
                thinking = delta.get("thinking")
                if isinstance(thinking, str):
                    on_thinking(thinking)
    return result


def _json_object(text: str) -> dict[str, Any] | None:
    """The CLI prints one JSON object; tolerate stray lines around it."""
    candidates: Sequence[str] = [text, *reversed(text.splitlines())]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
