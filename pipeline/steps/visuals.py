"""Step `visuals` v0: one image per scene = style bible + scene description.

v0 covers: prompt building, one seed per video, caching, source and licence
recorded per frame. v1 adds frame checks (brightness, Claude review), up to
`max_attempts_per_scene` tries and the stock fallback.

The bible's forbidden list is NOT put into the image prompt: naming "text" or
"hands" tends to make diffusion models draw exactly that. The list is enforced
upstream (the script prompt keeps scene descriptions clean) and downstream
(the frame check in v1).
"""

import hashlib
from pathlib import Path

from pipeline.adapters.errors import TransientAdapterError
from pipeline.adapters.images import ImageRequest, ImageResult
from pipeline.artifacts import input_hash, is_cached, write_output
from pipeline.config import VisualBible
from pipeline.logs import get_logger
from pipeline.steps import script as script_step
from pipeline.steps.base import StepContext, StepError, StepResult

log = get_logger()

FRAMES_DIR = "frames"
# Generation size, close to 9:16. Render scales and crops to 1080x1920.
WIDTH, HEIGHT = 768, 1344
_IMAGE_SUFFIXES = {".png", ".jpg", ".webp"}
_SIGNATURES = {"png": (b"\x89PNG\r\n\x1a\n",), "jpg": (b"\xff\xd8\xff",), "webp": (b"RIFF",)}
_MIN_BYTES = 1024  # anything smaller is not a usable 768x1344 frame


def looks_like_image(data: bytes, fmt: str) -> bool:
    """Cheap first gate: right magic bytes and a plausible size.

    Decoding, brightness and the Claude review are the frame checks of visuals v1.
    """
    return len(data) >= _MIN_BYTES and data.startswith(_SIGNATURES.get(fmt, (b"\x00\xff",)))


def build_image_prompt(bible: VisualBible, scene_visual: str) -> str:
    parts = [
        bible.style_prefix.strip().rstrip("."),
        scene_visual.strip().rstrip("."),
        f"lighting: {bible.lighting}",
        f"camera: {bible.camera}",
        "colour palette " + ", ".join(bible.palette),
    ]
    return ". ".join(parts) + "."


def video_seed(video_id: int, revision: int) -> int:
    """One seed per video (and per revision), stable across runs and machines."""
    digest = hashlib.sha256(f"seed:{video_id}:{revision}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _frame_candidates(frames_dir: Path, index: int) -> list[Path]:
    return sorted(p for p in frames_dir.glob(f"scene_{index:02d}.*") if p.suffix in _IMAGE_SUFFIXES)


def _generate_scene(ctx: StepContext, index: int, prompt: str, seed: int) -> ImageResult:
    """Up to `max_attempts_per_scene` tries for one scene; every rejection is logged with a reason.

    Retrying per scene (not the whole step) keeps one bad image from failing five good ones.
    In v1 the last resort after these attempts is a stock image instead of an error.
    """
    tries = ctx.series.visual_bible.max_attempts_per_scene
    reason = ""
    for attempt in range(1, tries + 1):
        try:
            result = ctx.images.generate(
                ImageRequest(prompt=prompt, seed=seed, width=WIDTH, height=HEIGHT)
            )
        except TransientAdapterError as e:
            reason = str(e)
        else:
            if looks_like_image(result.image, result.format):
                return result
            reason = "invalid image data"
        log.warning(
            "frame_rejected",
            extra={"video_id": ctx.video_id, "scene": index, "attempt": attempt, "reason": reason},
        )
    raise StepError(
        f"scene {index}: no usable image after {tries} attempts (last: {reason})", retryable=True
    )


def run(ctx: StepContext) -> StepResult:
    scenes = script_step.load(ctx.run_dir).scenes
    frames_dir = ctx.run_dir / FRAMES_DIR
    seed = video_seed(ctx.video_id, ctx.revision)
    outputs: list[Path] = []
    cost = 0.0
    called = False

    for index, scene in enumerate(scenes):
        prompt = build_image_prompt(ctx.series.visual_bible, scene.visual)
        key = input_hash(
            {
                "step": "visuals",
                "prompt": prompt,
                "seed": seed,
                "size": [WIDTH, HEIGHT],
                "provider": ctx.images.provider,
                "model": ctx.images.model,
            }
        )
        cached = next((p for p in _frame_candidates(frames_dir, index) if is_cached(p, key)), None)
        if cached is not None:
            outputs.append(cached)
            continue

        result = _generate_scene(ctx, index, prompt, seed)
        called = True
        cost += result.cost_usd
        path = frames_dir / f"scene_{index:02d}.{result.format}"
        for stale in _frame_candidates(frames_dir, index):  # e.g. an old .jpg next to a new .png
            if stale != path:
                stale.unlink()
        write_output(
            path,
            result.image,
            key,
            {
                "scene": index,
                "prompt": prompt,
                "seed": result.seed,
                "provider": result.provider,
                "model": result.model,
                "cost_usd": result.cost_usd,
                "source": "generated",
                "licence": result.licence,
            },
        )
        outputs.append(path)

    return StepResult(outputs=tuple(outputs), cost_usd=cost, cached=not called)


def frame_paths(run_dir: Path, count: int) -> list[Path]:
    """The frame for each scene 0..count-1 (used by render)."""
    frames: list[Path] = []
    for index in range(count):
        candidates = _frame_candidates(run_dir / FRAMES_DIR, index)
        if not candidates:
            raise StepError(
                f"no frame for scene {index}: run the visuals step first", retryable=False
            )
        frames.append(candidates[0])
    return frames
