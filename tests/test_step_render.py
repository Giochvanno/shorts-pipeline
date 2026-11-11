"""Reference tests for render v0 with real FFmpeg: format, timing, text, audio, failures."""

import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.adapters.images.fake import gradient_png
from pipeline.artifacts import read_meta
from pipeline.media import MediaError, probe
from pipeline.steps import render, script, visuals, voice
from pipeline.steps.base import StepContext, StepError

pytestmark = pytest.mark.usefixtures("ffmpeg")

NAVY = (46, 64, 87)
RED = (193, 102, 107)


def _solid(path: Path, colour: tuple[int, int, int]) -> Path:
    path.write_bytes(gradient_png(96, 168, colour, colour))
    return path


def _tone(path: Path, seconds: float, *, gaps: bool) -> Path:
    """Beeps with silent gaps (like the fake voice) or a continuous tone (like music)."""
    source = f"sine=f=330:d={seconds}"
    graph = "volume='if(lt(mod(t,0.4),0.25),1,0)':eval=frame" if gaps else "anull"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-af",
            graph,
            str(path),
        ],
        check=True,
    )
    return path


def _frame(video: Path, at_s: float) -> bytes:
    """One decoded frame, downscaled to 108x192 RGB."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{at_s}", "-i", str(video), "-frames:v", "1",
         "-vf", "scale=108:192", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    return result.stdout


def _mean(frame: bytes) -> tuple[float, float, float]:
    n = len(frame) // 3
    return (sum(frame[0::3]) / n, sum(frame[1::3]) / n, sum(frame[2::3]) / n)


def _near(a: tuple[float, ...], b: tuple[int, ...], tol: float = 8) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def _white_pixels(frame: bytes) -> int:
    return sum(1 for i in range(0, len(frame), 3) if min(frame[i : i + 3]) > 230)


def _silences(media: Path) -> int:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(media),
         "-af", "silencedetect=n=-45dB:d=0.1", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return len(re.findall(r"silence_start", result.stderr))


@pytest.fixture
def two_scenes(tmp_path: Path) -> dict[str, Path]:
    return {
        "a": _solid(tmp_path / "a.png", NAVY),
        "b": _solid(tmp_path / "b.png", RED),
        "voice": _tone(tmp_path / "voice.wav", 4.0, gaps=True),
        "font": render.font_path("Montserrat ExtraBold"),
    }


def _render(files: dict[str, Path], out: Path, *, music: Path | None = None) -> Path:
    render.render_video(
        frames=[files["a"], files["b"]],
        scene_weights=[1, 1],
        voice=files["voice"],
        duration_s=4.0,
        hook="This one sounds made up.",
        font=files["font"],
        output=out,
        music=music,
    )
    return out


def test_output_format_is_shorts_ready(two_scenes: dict[str, Path], tmp_path: Path) -> None:
    info = probe(_render(two_scenes, tmp_path / "out.mp4"))

    assert (info.width, info.height) == (1080, 1920)
    assert info.fps == pytest.approx(30)
    assert info.video_codec == "h264"
    assert info.pix_fmt == "yuv420p"
    assert info.audio_codec == "aac"
    assert info.sample_rate == 48000
    assert info.channels == 2
    assert info.duration_s == pytest.approx(4.0, abs=0.07)
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_scenes_switch_on_time(two_scenes: dict[str, Path], tmp_path: Path) -> None:
    video = _render(two_scenes, tmp_path / "out.mp4")

    assert _near(_mean(_frame(video, 1.9)), NAVY, tol=25)  # scene 1 (with hook text)
    assert _near(_mean(_frame(video, 3.0)), RED)  # scene 2, plain


def test_hook_is_drawn_only_over_the_first_scene(
    two_scenes: dict[str, Path], tmp_path: Path
) -> None:
    video = _render(two_scenes, tmp_path / "out.mp4")

    assert _white_pixels(_frame(video, 0.5)) > 50
    assert _white_pixels(_frame(video, 3.0)) == 0


def test_music_fills_voice_gaps_but_stays_under_the_voice(
    two_scenes: dict[str, Path], tmp_path: Path
) -> None:
    music = _tone(tmp_path / "music.wav", 2.0, gaps=False)  # shorter than video: must loop
    voice_only = _render(two_scenes, tmp_path / "voice_only.mp4")
    with_music = _render(two_scenes, tmp_path / "with_music.mp4", music=music)

    assert _silences(voice_only) > 3  # gaps between fake words
    assert _silences(with_music) == 0  # music under the whole video
    assert probe(with_music).duration_s == pytest.approx(4.0, abs=0.07)


def test_broken_input_fails_fast_and_leaves_nothing(
    two_scenes: dict[str, Path], tmp_path: Path
) -> None:
    two_scenes["b"].write_bytes(b"definitely not a png")

    started = time.monotonic()
    with pytest.raises(MediaError, match="ffmpeg failed"):
        _render(two_scenes, tmp_path / "out.mp4")

    assert time.monotonic() - started < 30  # without -xerror this hung until the timeout

    assert not (tmp_path / "out.mp4").exists()
    assert not list(tmp_path.glob("*.tmp.mp4"))


# --- the step, end to end on fakes ------------------------------------------------


@pytest.fixture
def prepared(ctx: StepContext) -> StepContext:
    script.run(ctx)
    voice.run(ctx)
    visuals.run(ctx)
    return ctx


def test_step_renders_full_fake_video(prepared: StepContext) -> None:
    result = render.run(prepared)

    (video,) = result.outputs
    info = probe(video)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration_s == pytest.approx(voice.duration_s(prepared.run_dir), abs=0.07)
    meta = read_meta(video)
    assert meta is not None
    assert meta["scenes"] == len(script.load(prepared.run_dir).scenes)


def test_step_second_run_is_cached(prepared: StepContext) -> None:
    render.run(prepared)
    first = (prepared.run_dir / render.OUTPUT).stat().st_mtime_ns

    again = render.run(prepared)

    assert again.cached
    assert (prepared.run_dir / render.OUTPUT).stat().st_mtime_ns == first  # not re-encoded


def test_step_rerenders_when_a_frame_changes(prepared: StepContext) -> None:
    render.run(prepared)
    _solid(visuals.frame_paths(prepared.run_dir, 1)[0], RED)

    assert not render.run(prepared).cached


def test_step_reports_ffmpeg_failure_as_non_retryable(prepared: StepContext) -> None:
    visuals.frame_paths(prepared.run_dir, 1)[0].write_bytes(b"broken")

    with pytest.raises(StepError, match=r"scene_00\.png is not a readable image") as exc:
        render.run(prepared)

    assert not exc.value.retryable
    assert read_meta(prepared.run_dir / render.OUTPUT) is None


def test_step_reports_missing_ffmpeg(
    prepared: StepContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pipeline.media.shutil.which", lambda _name: None)

    with pytest.raises(StepError, match="not found in PATH"):
        render.run(replace(prepared))
