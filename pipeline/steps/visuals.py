"""Step `visuals`: one frame per scene = style bible + scene + its light.

Each new frame goes to the frame review (the vision port: Claude in real mode)
before it is kept. A frame turned down for a wrong subject, text, blood-like marks
or an off-style look is set aside in `rejected_frames/` with the reasons, and the
same model draws the scene again with the next seed, up to
`max_attempts_per_scene` frames. One model per video: never another model for the
retry, its style would differ.

If the review turns down every try, the step fails without a retry: the image
model is deterministic (docs/decisions/image-model.md), so the same seeds would
draw the same frames and charge for them again. A person looks at the rejected
frames; the image limit per video stops a loop either way.

A drawn frame is paid for, so it is cached before the review: its meta says
review "pending". If the review cannot run (the CLI is down, a limit pauses the
video), the next run reviews that frame instead of drawing and paying again.
Reviewed frames are not reviewed again: the cache key is the request (prompt,
seed, model), and the meta records the review the frame passed.

The bible's forbidden list is NOT put into the image prompt: naming "text" or
"hands" tends to make diffusion models draw exactly that. The list is enforced
upstream (the script prompt keeps scene descriptions clean) and downstream
(the frame review).
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.adapters.errors import TransientAdapterError
from pipeline.adapters.images import ImageRequest, ImageResult
from pipeline.adapters.vision import FrameCheckRequest, FrameVerdict, media_type_of
from pipeline.artifacts import (
    input_hash,
    invalidate,
    is_cached,
    read_meta,
    write_meta,
    write_output,
)
from pipeline.config import VisualBible
from pipeline.logs import get_logger
from pipeline.steps import script as script_step
from pipeline.steps.base import StepContext, StepError, StepResult

log = get_logger()

FRAMES_DIR = "frames"
REJECTED_FRAMES = "rejected_frames"
# A review that fails (a CLI timeout) is asked once more: the frame is paid already.
REVIEW_TRIES = 2
PENDING = "pending"  # meta["review"] of a frame drawn and paid for, not reviewed yet
# Generation size: exactly 9:16 and 720p, the size image-to-video models start from.
# 0.92 megapixels: billed as one megapixel however a provider counts. Render scales to
# 1080x1920.
WIDTH, HEIGHT = 720, 1280
_IMAGE_SUFFIXES = {".png", ".jpg", ".webp"}
_SIGNATURES = {"png": (b"\x89PNG\r\n\x1a\n",), "jpg": (b"\xff\xd8\xff",), "webp": (b"RIFF",)}
_MIN_BYTES = 1024  # anything smaller is not a usable 720x1280 frame


def looks_like_image(data: bytes, fmt: str) -> bool:
    """Cheap first gate, before the review: right magic bytes and a plausible size."""
    return len(data) >= _MIN_BYTES and data.startswith(_SIGNATURES.get(fmt, (b"\x00\xff",)))


def build_image_prompt(bible: VisualBible, scene_visual: str, lighting: str | None = None) -> str:
    """Style + the scene + its light (the bible's default if the scene names none) + camera
    + colours (in words when the bible has them: hex codes are text to an image model)."""
    colours = (
        f"colours: {bible.colours.strip()}"
        if bible.colours
        else "colour palette " + ", ".join(bible.palette)  # videos up to 10: keeps their cache
    )
    parts = [
        bible.style_prefix.strip().rstrip("."),
        scene_visual.strip().rstrip("."),
        f"lighting: {(lighting or bible.lighting).strip()}",
        f"camera: {bible.camera}",
        colours,
    ]
    return ". ".join(parts) + "."


def video_seed(video_id: int, revision: int) -> int:
    """One seed per video (and per revision), stable across runs and machines."""
    digest = hashlib.sha256(f"seed:{video_id}:{revision}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _frame_candidates(frames_dir: Path, index: int) -> list[Path]:
    return sorted(p for p in frames_dir.glob(f"scene_{index:02d}.*") if p.suffix in _IMAGE_SUFFIXES)


@dataclass(frozen=True, slots=True)
class _Drawn:
    path: Path
    attempt: int
    turned_down: tuple[str, ...]  # why the review rejected the earlier tries
    paid_usd: float  # every frame drawn for this scene in this run, rejected ones too


def review_text(bible: VisualBible, scene_visual: str, lighting: str | None) -> str:
    """What the reviewer compares the frame with: the scene and its light."""
    return f"{scene_visual.strip().rstrip('.')}. Light: {(lighting or bible.lighting).strip()}"


def _review(ctx: StepContext, scene: str, image: bytes) -> FrameVerdict | None:
    if ctx.vision is None:
        return None
    request = FrameCheckRequest(
        image=image,
        media_type=media_type_of(image),
        scene=scene,
        style=ctx.series.visual_bible.style_prefix.strip(),
    )
    for tried in range(1, REVIEW_TRIES + 1):
        try:
            return ctx.vision.review(request)
        except TransientAdapterError as e:
            if tried == REVIEW_TRIES:
                # The frame stays cached as pending: the next run reviews it, unpaid.
                raise StepError(f"the frame review failed: {e}", retryable=True) from e
    raise AssertionError("unreachable")  # pragma: no cover


def _review_meta(verdict: FrameVerdict | None) -> dict[str, Any] | None:
    if verdict is None:
        return None
    return {
        "accepted": verdict.accepted,
        "problems": verdict.problems(),
        "subject_seen": verdict.subject_seen,
        "subject_position": verdict.subject_position,
        "text_seen": verdict.text_seen,
        "unwanted_seen": verdict.unwanted_seen,
        "style_matches": verdict.style_matches,
        "provider": verdict.provider,
        "model": verdict.model,
    }


def _keep_rejected(
    ctx: StepContext, frame: Path, meta: dict[str, Any], verdict: FrameVerdict
) -> None:
    """Set a turned-down frame aside with the reasons, for a person to judge the review.

    Inside the video's run folder (git ignores runs/), capped at
    `qa.rejected_frames_max_mb`: the oldest go first, the newest always stays.
    """
    folder = ctx.run_dir / REJECTED_FRAMES
    folder.mkdir(exist_ok=True)
    stem = f"scene_{int(meta['scene']):02d}_try{int(meta['attempt'])}"
    (folder / f"{stem}{frame.suffix}").write_bytes(frame.read_bytes())
    report = {**meta, "review": _review_meta(verdict)}
    (folder / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _cap(folder, int(ctx.series.qa.rejected_frames_max_mb * 1024 * 1024))
    invalidate(frame)  # meta first: a crash in between leaves no meta, never a stale pair
    frame.unlink(missing_ok=True)


def _cap(folder: Path, max_bytes: int) -> None:
    groups: dict[str, list[Path]] = {}
    for path in folder.iterdir():
        groups.setdefault(path.name.split(".")[0], []).append(path)
    order = sorted(groups, key=lambda stem: max(p.stat().st_mtime_ns for p in groups[stem]))
    total = sum(p.stat().st_size for paths in groups.values() for p in paths)
    while len(order) > 1 and total > max_bytes:
        for path in groups[order.pop(0)]:
            total -= path.stat().st_size
            path.unlink()


def _settle(ctx: StepContext, frame: Path, key: str, scene: str) -> FrameVerdict | None:
    """Review a cached frame; accepted: the verdict goes into its meta, rejected: set aside.

    Returns the verdict (None without a reviewer: the frame is kept as it is).
    """
    meta = read_meta(frame) or {}
    verdict = _review(ctx, scene, frame.read_bytes())
    if verdict is None or verdict.accepted:
        write_meta(
            frame,
            key,
            {k: v for k, v in meta.items() if k != "input_hash"}
            | {"review": _review_meta(verdict)},
        )
    else:
        _keep_rejected(ctx, frame, meta, verdict)
    return verdict


def _generate_scene(
    ctx: StepContext,
    index: int,
    prompt: str,
    review: str,
    seed: int,
    key: str,
    first_attempt: int = 1,
    turned_down: tuple[str, ...] = (),
) -> _Drawn:
    """Up to `max_attempts_per_scene` frames for one scene; every rejection logged with a reason.

    Retrying per scene (not the whole step) keeps one bad frame from failing five good ones.
    """
    frames_dir = ctx.run_dir / FRAMES_DIR
    tries = ctx.series.visual_bible.max_attempts_per_scene
    reasons = list(turned_down)
    reason = reasons[-1].split(": ", 1)[1] if reasons else ""
    paid = 0.0
    for attempt in range(first_attempt, tries + 1):
        # Same seed, same prompt, same image: a retry with the video's seed would repeat a
        # blanked or rejected frame. The first try keeps the video's seed (one look per video).
        attempt_seed = (seed + attempt - 1) % 2**32
        try:
            result = ctx.images.generate(
                ImageRequest(prompt=prompt, seed=attempt_seed, width=WIDTH, height=HEIGHT)
            )
        except TransientAdapterError as e:
            reason = str(e)
        else:
            paid += result.cost_usd
            if not looks_like_image(result.image, result.format):
                reason = "invalid image data"
            else:
                path = _store_pending(frames_dir, index, key, prompt, result, attempt)
                verdict = _settle(ctx, path, key, review)
                if verdict is None or verdict.accepted:
                    return _Drawn(path, attempt, tuple(reasons), paid)
                reason = "; ".join(verdict.problems())
                reasons.append(f"try {attempt}: {reason}")
        log.warning(
            "frame_rejected",
            extra={"video_id": ctx.video_id, "scene": index, "attempt": attempt, "reason": reason},
        )
    if reasons:
        raise StepError(
            f"scene {index + 1}: the frame review turned down {len(reasons)} of {tries}"
            f" frames (last: {reason}); see {REJECTED_FRAMES}/ in the run folder",
            retryable=False,  # the same seeds would draw the same frames again
        )
    raise StepError(
        f"scene {index + 1}: no usable image after {tries} attempts (last: {reason})",
        retryable=True,
    )


def _store_pending(
    frames_dir: Path, index: int, key: str, prompt: str, result: ImageResult, attempt: int
) -> Path:
    """Cache a paid frame before its review, marked pending."""
    path = frames_dir / f"scene_{index:02d}.{result.format}"
    for stale in _frame_candidates(frames_dir, index):  # e.g. an old .jpg next to a new .png
        if stale != path:
            invalidate(stale)
            stale.unlink()
    write_output(
        path,
        result.image,
        key,
        {
            "scene": index,
            "prompt": prompt,
            "seed": result.seed,
            "attempt": attempt,
            "provider": result.provider,
            "model": result.model,
            "cost_usd": result.cost_usd,
            "source": "generated",
            "licence": result.licence,
            "review": PENDING,
        },
    )
    return path


def run(ctx: StepContext) -> StepResult:
    scenes = script_step.load(ctx.run_dir).scenes
    bible = ctx.series.visual_bible
    frames_dir = ctx.run_dir / FRAMES_DIR
    seed = video_seed(ctx.video_id, ctx.revision)
    outputs: list[Path] = []
    notes: list[str] = []
    cost = 0.0
    called = False

    for index, scene in enumerate(scenes):
        prompt = build_image_prompt(bible, scene.visual, scene.lighting)
        review = review_text(bible, scene.visual, scene.lighting)
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
        first_attempt, turned_down = 1, tuple[str, ...]()
        if cached is not None:
            meta = read_meta(cached) or {}
            if meta.get("review") != PENDING or ctx.vision is None:
                outputs.append(cached)
                continue
            # Drawn and paid for in an earlier run, never reviewed: review it, do not redraw.
            called = True
            attempt = int(meta.get("attempt", 1))
            verdict = _settle(ctx, cached, key, review)
            if verdict is None or verdict.accepted:
                outputs.append(cached)
                continue
            first_attempt = attempt + 1
            turned_down = (f"try {attempt}: {'; '.join(verdict.problems())}",)

        drawn = _generate_scene(ctx, index, prompt, review, seed, key, first_attempt, turned_down)
        called = True
        cost += drawn.paid_usd
        if drawn.turned_down:
            notes.append(
                f"scene {index + 1}: the review turned down {len(drawn.turned_down)} frame(s),"
                f" try {drawn.attempt} kept ({drawn.turned_down[-1].split(': ', 1)[1]})"
            )
        outputs.append(drawn.path)

    return StepResult(outputs=tuple(outputs), cost_usd=cost, cached=not called, notes=tuple(notes))


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
