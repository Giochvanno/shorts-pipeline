"""Live check: does `claude -p` really see a frame sent as a stream-json image block?

    uv run pytest tests/live/test_claude_vision_live.py -m live -s

On your subscription, no money. The heavy model from config, as the pipeline will use.

1. A plain blue frame: the answer must name the colour, or the image never arrived.
2. The same frame with a word drawn on it: the review must report text.
3. If the image-model comparison of video 10 is on disk: four of its frames that were
   scored by eye (a signature, a signature plus an extra animal, blood-like patches,
   a clean wombat). Printed side by side with the eye scores and saved to
   runs/10/compare/claude_review.json: how far the automatic check can be trusted.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from pipeline.adapters.images.fake import gradient_png
from pipeline.adapters.vision import FrameCheckRequest, FrameVerdict, media_type_of
from pipeline.adapters.vision.claude_cli import ClaudeCliReviewer
from pipeline.artifacts import read_meta
from pipeline.config import load_channel, load_series
from pipeline.media import run_ffmpeg
from pipeline.steps.render import font_path

pytestmark = pytest.mark.live

FIELDS = (
    "subject_seen",
    "right_subject",
    "text_seen",
    "has_text",
    "subject_position",
    "style_matches",
    "unwanted_seen",
    "unwanted",
)

REPO = Path(__file__).resolve().parents[2]
COMPARE = REPO / "runs" / "10" / "compare"
# (folder, scene index, what the eye saw in the first comparison)
SCORED = [
    ("fal-ai__flux__dev", 5, "text: signature bottom left"),
    ("fal-ai__flux-2", 1, "text: signature bottom right; an extra wombat"),
    ("fal-ai__flux-2-pro", 0, "unwanted: red blood-like patches on the chest"),
    ("fal-ai__nano-banana", 0, "clean: a clear wombat"),
]


@pytest.fixture
def reviewer() -> ClaudeCliReviewer:
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI is not installed")
    models = load_channel().models.llm
    return ClaudeCliReviewer(models.heavy.primary.model, timeout_s=models.timeout_s)


def _show(label: str, verdict: FrameVerdict) -> None:
    print(
        f"\n{label}\n  sees: {verdict.subject_seen!r}  right={verdict.right_subject}"
        f"  position={verdict.subject_position}  style={verdict.style_matches}"
        f"\n  text={verdict.has_text} {verdict.text_seen!r}"
        f"  unwanted={verdict.unwanted} {verdict.unwanted_seen!r}"
        f"\n  -> {'ACCEPTED' if verdict.accepted else 'REJECTED: ' + '; '.join(verdict.problems())}"
    )


def test_a_plain_frame_arrives_and_is_seen(reviewer: ClaudeCliReviewer) -> None:
    blue = gradient_png(720, 1280, (30, 60, 160), (30, 60, 160))

    verdict = reviewer.review(
        FrameCheckRequest(blue, "image/png", "A plain flat blue rectangle", "Flat colour")
    )

    _show("plain blue frame", verdict)
    assert "blue" in verdict.subject_seen.lower()
    assert not verdict.has_text


@pytest.mark.usefixtures("ffmpeg")
def test_a_word_in_the_frame_is_reported(reviewer: ClaudeCliReviewer) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        (work / "font.ttf").write_bytes(font_path("Montserrat ExtraBold").read_bytes())
        run_ffmpeg(
            [
                "-f", "lavfi", "-i", "color=c=0x1E3CA0:s=720x1280:d=1",
                "-vf", "drawtext=fontfile=font.ttf:text=SALE:fontsize=120:fontcolor=white"
                ":x=(w-text_w)/2:y=(h-text_h)/2",
                "-frames:v", "1", "-update", "1", "word.png",
            ],
            cwd=work,
        )  # fmt: skip
        image = (work / "word.png").read_bytes()

    verdict = reviewer.review(
        FrameCheckRequest(image, media_type_of(image), "A plain flat blue rectangle", "Flat colour")
    )

    _show("blue frame with a word", verdict)
    assert verdict.has_text
    assert not verdict.accepted


def test_frames_scored_by_eye(reviewer: ClaudeCliReviewer) -> None:
    if not COMPARE.exists():
        pytest.skip("no image-model comparison for video 10 on disk")
    style = load_series("fact-of-day").visual_bible.style_prefix
    saved = []
    for folder, index, by_eye in SCORED:
        frame = COMPARE / folder / f"scene_{index:02d}.png"
        if not frame.exists():
            continue
        image = frame.read_bytes()
        prompt = str((read_meta(frame) or {}).get("prompt") or "")
        verdict = reviewer.review(FrameCheckRequest(image, media_type_of(image), prompt, style))
        _show(f"{folder} scene {index + 1}  (by eye: {by_eye})", verdict)
        saved.append(
            {
                "frame": f"{folder}/{frame.name}",
                "by_eye": by_eye,
                "accepted": verdict.accepted,
                "problems": verdict.problems(),
                "verdict": {k: getattr(verdict, k) for k in FIELDS},
            }
        )
    out = COMPARE / "claude_review.json"
    out.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nsaved: {out}")
    assert saved
