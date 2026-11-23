"""Frame review through the Claude Code CLI, on the Claude subscription.

    claude -p --input-format stream-json --output-format stream-json --verbose
           --json-schema <schema> --model <model> --system-prompt <fixed text>
           --tools "" --permission-prompts none --no-session-persistence --safe-mode
           --max-turns 3

- The frame and the instructions go through stdin as one stream-json user message:
  an image block (base64) and a text block, the shape the Agent SDK uses for images.
  The command line holds fixed tokens and the schema only (compact JSON, no spaces).
- Claude Code 2.1.325 refuses stream-json input with json output ("requires
  output-format=stream-json"), although the CLI reference shows that pair. So the
  output is stream-json too, and the answer is the last `result` event.
- --json-schema puts the answer in `structured_output`. Claude Code adds a single
  StructuredOutput tool for it; every built-in tool is off. The answer is checked
  again here, field by field.
- As for text: an empty temporary folder as cwd, no ANTHROPIC_API_KEY (the CLI would
  bill the API), --safe-mode, errors sorted for the runner by `classify_failure`.

Checked against the CLI reference, the headless and Agent SDK streaming-input docs
(November 2026) and `claude` 2.1.325, up to the login step. The live test
(tests/live/test_claude_vision_live.py) is what proves a model sees the image.
"""

import base64
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.llm.claude_cli import PROVIDER, Runner, classify_failure
from pipeline.adapters.vision.base import FrameCheckRequest, FrameVerdict

# The API takes images up to 5 MB; a 720x1280 PNG is 1-2 MB.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TURNS = 3  # the answer is one StructuredOutput call; room for one retry by the CLI
SYSTEM_PROMPT = (
    "You review frames for an educational video channel. Look at the whole image,"
    " corners and edges included, and answer every field exactly and strictly."
)

INSTRUCTIONS = """\
Review this frame. It is the first frame of a short animated clip. The image model was asked for:

Scene: {scene}
Style: {style}

Fill every field:
- subject_seen: the main subject you see, in a few plain words.
- right_subject: true only if it is recognisably the subject the scene asks for. An animal \
must look like that species, not a similar one; a shape the scene names (a cube, a spiral) \
must have that shape.
- extra_seen: an eye-catching object the scene does not mention, one a viewer would notice \
as a thing of its own, such as a second animal or a brightly lit landscape. Dim or blurred \
shapes in the background do not count, nor do details that belong to the setting the scene \
names (people on a street, furniture in a room, a hanger on a hook). Empty if there is none.
- extra_object: true if there is such an object.
- text_seen: every letter, number, signature, handwriting-like scribble, watermark, logo or \
label anywhere in the frame. Look closely at the corners and edges. Empty if there is none.
- has_text: true if there is any such mark.
- subject_position: where the centre of the main subject is: upper, middle or lower third of \
the frame; none if there is no clear subject.
- style_matches: true if the frame looks like the style line. A photo, a 3D render, or a \
picture inside a frame or border is false for a painting style.
- unwanted_seen: anything a viewer could read as blood, a wound, injury, gore or cut flesh, \
including red patches on an animal that look like blood. A clean organ that the scene \
itself asks for is fine. Empty if there is none.
- unwanted: true if there is any such thing."""


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    subject_seen: str = Field(max_length=300)
    right_subject: bool
    extra_seen: str = Field(max_length=300)
    extra_object: bool
    text_seen: str = Field(max_length=300)
    has_text: bool
    subject_position: Literal["upper", "middle", "lower", "none"]
    style_matches: bool
    unwanted_seen: str = Field(max_length=300)
    unwanted: bool


def answer_schema() -> str:
    """The JSON Schema of the answer, compact: no spaces for the shell to split on."""
    schema = {
        "type": "object",
        "properties": {
            "subject_seen": {"type": "string"},
            "right_subject": {"type": "boolean"},
            "extra_seen": {"type": "string"},
            "extra_object": {"type": "boolean"},
            "text_seen": {"type": "string"},
            "has_text": {"type": "boolean"},
            "subject_position": {"type": "string", "enum": ["upper", "middle", "lower", "none"]},
            "style_matches": {"type": "boolean"},
            "unwanted_seen": {"type": "string"},
            "unwanted": {"type": "boolean"},
        },
        "required": list(_Answer.model_fields),
        "additionalProperties": False,
    }
    return json.dumps(schema, separators=(",", ":"))


class ClaudeCliReviewer:
    def __init__(
        self,
        model: str,
        *,
        timeout_s: float = 180,
        executable: str = "claude",
        run: Runner = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._model = model
        self._timeout_s = timeout_s
        self._executable = executable
        self._run = run
        self._which = which

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def model(self) -> str:
        return self._model

    def command(self, executable: str) -> list[str]:
        return [
            executable,
            "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--json-schema", answer_schema(),
            "--model", self._model,
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",
            "--permission-prompts", "none",
            "--no-session-persistence",
            "--safe-mode",
            "--max-turns", str(MAX_TURNS),
        ]  # fmt: skip

    @staticmethod
    def message(request: FrameCheckRequest) -> bytes:
        """One stream-json line: the image first, then what to look for."""
        text = INSTRUCTIONS.format(scene=request.scene.strip(), style=request.style.strip())
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": request.media_type,
                    "data": base64.b64encode(request.image).decode("ascii"),
                },
            },
            {"type": "text", "text": text},
        ]
        line = {"type": "user", "message": {"role": "user", "content": content}}
        return (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")

    def review(self, request: FrameCheckRequest) -> FrameVerdict:
        if not request.image:
            raise ValueError("nothing to review: empty image")
        if len(request.image) > MAX_IMAGE_BYTES:
            raise PermanentAdapterError(
                f"frame is {len(request.image) / 1e6:.1f} MB: the review takes up to 5 MB"
            )
        executable = self._which(self._executable)
        if executable is None:
            raise PermanentAdapterError(
                "Claude Code CLI not found in PATH: install it with"
                " `irm https://claude.ai/install.ps1 | iex` and sign in"
            )
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with tempfile.TemporaryDirectory(prefix="claude_", ignore_cleanup_errors=True) as empty:
            try:
                proc = self._run(
                    self.command(executable),
                    input=self.message(request),
                    capture_output=True,
                    cwd=Path(empty),
                    env=env,
                    timeout=self._timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise TransientAdapterError(
                    f"claude -p timed out after {self._timeout_s:.0f} s reviewing a frame"
                ) from e
            except OSError as e:
                raise TransientAdapterError(f"could not start claude: {e}") from e
        return self._parse(proc)

    def _parse(self, proc: "subprocess.CompletedProcess[bytes]") -> FrameVerdict:
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        result = last_result(stdout)
        if result is None:
            raise classify_failure(stderr or stdout.strip() or f"exit code {proc.returncode}")
        if proc.returncode != 0 or result.get("is_error") or result.get("subtype") != "success":
            detail = str(result.get("result") or result.get("subtype") or "")
            status = result.get("api_error_status")
            raise classify_failure(
                " ".join(filter(None, [detail, stderr])) or "claude -p failed",
                status if isinstance(status, int) else None,
            )
        structured = result.get("structured_output")
        if not isinstance(structured, dict):
            raise TransientAdapterError("claude -p gave no structured answer for the frame")
        try:
            answer = _Answer.model_validate(structured)
        except ValidationError as e:
            raise TransientAdapterError(f"claude -p answered off the schema: {e}") from e
        return FrameVerdict(
            **answer.model_dump(),
            provider=PROVIDER,
            model=self._model,
            cost_usd=float(result.get("total_cost_usd") or 0.0),
        )


def last_result(stdout: str) -> dict[str, Any] | None:
    """The last `result` event of a stream-json run; stray lines are skipped."""
    found: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            found = event
    return found
