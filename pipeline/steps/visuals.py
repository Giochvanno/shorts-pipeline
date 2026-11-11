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

from pipeline.adapters.images import ImageRequest
from pipeline.artifacts import input_hash, is_cached, write_output
from pipeline.config import VisualBible
from pipeline.steps import script as script_step
from pipeline.steps.base import StepContext, StepError, StepResult

FRAMES_DIR = "frames"
# Generation size, close to 9:16. Render scales and crops to 1080x1920.
WIDTH, HEIGHT = 768, 1344
_IMAGE_SUFFIXES = {".png", ".jpg", ".webp"}


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

        result = ctx.images.generate(
            ImageRequest(prompt=prompt, seed=seed, width=WIDTH, height=HEIGHT)
        )
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
