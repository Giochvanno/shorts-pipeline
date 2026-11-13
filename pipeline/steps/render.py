"""Step `render`: frames + voice (+ music) + hook + captions -> runs/<id>/video.mp4, 1080x1920.

Static frames timed by each scene's share of words, the hook drawn over scene 1,
word-by-word captions (current word highlighted) from the captions step, optional
music ducked under the voice. Still to come: Ken Burns motion, loudness normalisation.

Captions skip the hook's words: the hook is already on screen in large type.

Text is drawn by FFmpeg, never by the image model. FFmpeg runs in a scratch
folder with the font and text files next to it, so filter arguments only hold
plain relative names: no Windows path escaping ("C\\:/...") anywhere.
"""

import shutil
import tempfile
import textwrap
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

from pipeline.artifacts import file_hash, input_hash, invalidate, is_cached, write_meta
from pipeline.ass import CaptionStyle, build_ass
from pipeline.media import MediaError, check_image, has_filter, run_ffmpeg
from pipeline.steps import captions as captions_step
from pipeline.steps import script as script_step
from pipeline.steps import visuals as visuals_step
from pipeline.steps import voice as voice_step
from pipeline.steps.base import StepContext, StepError, StepResult

OUTPUT = "video.mp4"
FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"

WIDTH, HEIGHT, FPS = 1080, 1920, 30
CRF = 20  # visually clean, reasonable size
# Bump when the filter graph changes, so cached videos are rebuilt.
RENDER_VERSION = 2

CAPTION_FONT_SIZE = 96  # readable on a phone; three words still fit one line
CAPTION_OUTLINE = 6
CAPTION_MARGIN_H = 90

HOOK_FONT_SIZE = 84
HOOK_LINE_CHARS = 16  # ~800 px at this size: safe margins at 1080 px wide
HOOK_TOP_RATIO = 0.27  # upper third, clear of YouTube's bottom UI
HOOK_LINE_SPACING = 1.25


def font_path(font_name: str) -> Path:
    """'Montserrat ExtraBold' -> assets/fonts/Montserrat-ExtraBold.ttf"""
    path = FONTS_DIR / f"{font_name.replace(' ', '-')}.ttf"
    if not path.exists():
        raise StepError(f"font {font_name!r} not found at {path}", retryable=False)
    return path


def frame_counts(weights: Sequence[int], total_s: float, fps: int = FPS) -> list[int]:
    """Split the video's frames between scenes by weight (words per scene).

    Boundaries are rounded once, cumulatively, so the counts always add up to
    exactly the video length; every scene gets at least one frame.
    """
    if not weights or any(w <= 0 for w in weights):
        raise ValueError("every scene needs a positive weight")
    total_frames = max(round(total_s * fps), len(weights))
    total_weight = sum(weights)
    bounds = [0]
    acc = 0
    for w in weights:
        acc += w
        bounds.append(round(total_frames * acc / total_weight))
    counts = [b - a for a, b in pairwise(bounds)]
    # Rounding can leave a very short scene with 0 frames: borrow from the longest one.
    for i, count in enumerate(counts):
        if count == 0:
            donor = counts.index(max(counts))
            counts[donor] -= 1
            counts[i] = 1
    return counts


def wrap_hook(text: str) -> list[str]:
    return textwrap.wrap(text.strip(), width=HOOK_LINE_CHARS, break_long_words=False) or [""]


def _video_filters(
    count: int, hook_lines: Sequence[str], hook_end_s: float, *, subtitles: bool
) -> list[str]:
    fit = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},setsar=1,fps={FPS},format=yuv420p"
    )
    parts = [f"[{i}:v]{fit}[v{i}]" for i in range(count)]
    parts.append("".join(f"[v{i}]" for i in range(count)) + f"concat=n={count}:v=1:a=0[vcat]")

    line_height = HOOK_FONT_SIZE * HOOK_LINE_SPACING
    top = HEIGHT * HOOK_TOP_RATIO
    draws = [
        f"drawtext=fontfile=fonts/font.ttf:textfile=hook_{n}.txt:expansion=none"
        f":fontsize={HOOK_FONT_SIZE}:fontcolor=white:borderw=6:bordercolor=black@0.85"
        f":x=(w-text_w)/2:y={top + n * line_height:.0f}"
        f":enable='lt(t,{hook_end_s:.3f})'"
        for n in range(len(hook_lines))
    ]
    if subtitles:
        # libass loads every file in fontsdir as a font: keep only the font in there.
        draws.append("ass=captions.ass:fontsdir=fonts")
    parts.append("[vcat]" + ",".join(draws) + "[vout]")
    return parts


def _audio_filters(voice_input: int, music_input: int | None, duck_db: float) -> list[str]:
    stereo = "aformat=sample_rates=48000:channel_layouts=stereo"
    if music_input is None:
        return [f"[{voice_input}:a]{stereo}[aout]"]
    return [
        f"[{voice_input}:a]{stereo}[voice]",
        f"[{music_input}:a]{stereo},volume=-{duck_db:g}dB[music]",
        "[voice][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]",
    ]


def render_video(
    *,
    frames: Sequence[Path],
    scene_weights: Sequence[int],
    voice: Path,
    duration_s: float,
    hook: str,
    font: Path,
    output: Path,
    music: Path | None = None,
    duck_db: float = 18.0,
    subtitles: str | None = None,
) -> None:
    """Pure FFmpeg assembly; knows nothing about steps, configs or caching.

    `subtitles` is the text of an ASS file (see pipeline/ass.py).
    """
    if subtitles is not None and not has_filter("ass"):
        raise MediaError(
            "this FFmpeg has no libass (the `ass` filter) for captions: install a full build"
            " (on Windows: the 'full' build from gyan.dev)"
        )
    if len(frames) != len(scene_weights):
        raise ValueError("one frame per scene is required")
    counts = frame_counts(scene_weights, duration_s)
    hook_lines = wrap_hook(hook)

    args: list[str] = []
    for frame, count in zip(frames, counts, strict=True):
        args += ["-loop", "1", "-framerate", str(FPS), "-t", f"{count / FPS:.6f}"]
        args += ["-i", str(frame.resolve())]
    voice_input = len(frames)
    args += ["-i", str(voice.resolve())]
    music_input: int | None = None
    if music is not None:
        music_input = voice_input + 1
        args += ["-stream_loop", "-1", "-i", str(music.resolve())]

    graph = _video_filters(len(frames), hook_lines, counts[0] / FPS, subtitles=bool(subtitles))
    graph += _audio_filters(voice_input, music_input, duck_db)
    tmp = output.with_name(output.stem + ".tmp" + output.suffix)  # FFmpeg needs the .mp4 suffix
    args += [
        "-filter_complex", ";".join(graph),
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-t", f"{sum(counts) / FPS:.6f}",
        "-movflags", "+faststart",
        str(tmp.resolve()),
    ]  # fmt: skip

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="render_") as scratch_name:
        scratch = Path(scratch_name)
        (scratch / "fonts").mkdir()
        shutil.copyfile(font, scratch / "fonts" / "font.ttf")
        if subtitles is not None:
            (scratch / "captions.ass").write_text(subtitles, encoding="utf-8")
        for n, line in enumerate(hook_lines):
            (scratch / f"hook_{n}.txt").write_text(line, encoding="utf-8")
        try:
            # A 31 s Short renders in ~8 s; 3x its length is a generous backstop.
            run_ffmpeg(args, cwd=scratch, timeout_s=max(60.0, 3 * duration_s))
        except MediaError:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(output)


def run(ctx: StepContext) -> StepResult:
    scenes = script_step.load(ctx.run_dir).scenes
    frames = visuals_step.frame_paths(ctx.run_dir, len(scenes))
    voice, duration = voice_step.voice_file(ctx.run_dir)
    font = font_path(ctx.series.captions.font)
    weights = [max(len(s.narration.split()), 1) for s in scenes]
    hook = scenes[0].narration
    output = ctx.run_dir / OUTPUT
    words, chunks = captions_step.load(ctx.run_dir)
    style = ctx.series.captions
    subtitles = build_ass(
        words,
        chunks,
        CaptionStyle(
            font=style.font,
            size=CAPTION_FONT_SIZE,
            color="#FFFFFF",
            highlight=style.highlight_color,
            outline=CAPTION_OUTLINE,
            margin_v=round(style.bottom_margin_ratio * HEIGHT),
            margin_h=CAPTION_MARGIN_H,
            width=WIDTH,
            height=HEIGHT,
        ),
        first_word=len(hook.split()),
    )

    key = input_hash(
        {
            "step": "render",
            "version": RENDER_VERSION,
            "frames": [file_hash(f) for f in frames],
            "voice": file_hash(voice),
            "font": file_hash(font),
            "weights": weights,
            "hook": hook,
            "subtitles": subtitles,  # holds every caption time, word and style value
            "duration_s": duration,
            "settings": [WIDTH, HEIGHT, FPS, CRF],
        }
    )
    if is_cached(output, key):
        return StepResult(outputs=(output,), cost_usd=0.0, cached=True)

    invalidate(output)
    try:
        for frame in frames:
            check_image(frame)
        render_video(
            frames=frames,
            scene_weights=weights,
            voice=voice,
            duration_s=duration,
            hook=hook,
            font=font,
            output=output,
            subtitles=subtitles,
        )
    except MediaError as e:
        # FFmpeg errors are deterministic for the same inputs: retrying will not help.
        raise StepError(f"render failed: {e}", retryable=False) from e

    write_meta(
        output,
        key,
        {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "duration_s": round(sum(frame_counts(weights, duration)) / FPS, 3),
            "scenes": len(scenes),
            "cost_usd": 0.0,
        },
    )
    return StepResult(outputs=(output,), cost_usd=0.0, cached=False)
