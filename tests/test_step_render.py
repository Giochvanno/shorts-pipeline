"""Reference tests for render v0 with real FFmpeg: format, timing, text, audio, failures."""

import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from pipeline.adapters.images.fake import gradient_png
from pipeline.artifacts import read_meta
from pipeline.ass import CaptionStyle, build_ass
from pipeline.media import MediaError, probe
from pipeline.steps import captions, render, script, visuals, voice
from pipeline.steps.base import StepContext, StepError
from pipeline.steps.captions import CaptionWord
from pipeline.steps.captions import chunk as caption_chunks
from pipeline.testing import fake_context

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
# One full fake video is rendered once per test session; each test gets its own copy.


@pytest.fixture(scope="session")
def rendered_template(ffmpeg: None, tmp_path_factory: pytest.TempPathFactory) -> Path:
    ctx = fake_context(tmp_path_factory.mktemp("template") / "run")
    for step in (script, voice, visuals, captions, render):
        step.run(ctx)
    return ctx.run_dir


@pytest.fixture
def rendered(rendered_template: Path, tmp_path: Path) -> StepContext:
    run_dir = Path(shutil.copytree(rendered_template, tmp_path / "run"))
    return fake_context(run_dir)


def test_step_renders_full_fake_video(rendered: StepContext) -> None:
    video = rendered.run_dir / render.OUTPUT
    info = probe(video)

    assert (info.width, info.height) == (1080, 1920)
    assert info.duration_s == pytest.approx(voice.duration_s(rendered.run_dir), abs=0.07)
    meta = read_meta(video)
    assert meta is not None
    assert meta["scenes"] == len(script.load(rendered.run_dir).scenes)


def test_step_second_run_is_cached(rendered: StepContext) -> None:
    video = rendered.run_dir / render.OUTPUT
    before = video.stat().st_mtime_ns

    again = render.run(rendered)

    assert again.cached
    assert video.stat().st_mtime_ns == before  # not re-encoded


def test_step_rerenders_when_a_frame_changes(rendered: StepContext) -> None:
    _solid(visuals.frame_paths(rendered.run_dir, 1)[0], RED)

    assert not render.run(rendered).cached


def test_step_reports_broken_frame_as_non_retryable(rendered: StepContext) -> None:
    visuals.frame_paths(rendered.run_dir, 1)[0].write_bytes(b"broken")

    with pytest.raises(StepError, match=r"scene_00\.png is not a readable image") as exc:
        render.run(rendered)

    assert not exc.value.retryable
    assert read_meta(rendered.run_dir / render.OUTPUT) is None


def test_step_reports_missing_ffmpeg(
    rendered: StepContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _solid(visuals.frame_paths(rendered.run_dir, 1)[0], RED)  # force a real render attempt
    monkeypatch.setattr("pipeline.media.shutil.which", lambda _name: None)

    with pytest.raises(StepError, match="not found in PATH"):
        render.run(rendered)


# --- captions drawn by libass -----------------------------------------------------

GRAY = (60, 60, 60)
YELLOW = (244, 211, 94)
W, H = 270, 480  # decoded frame size for caption checks (1/4 of 1080x1920)
CAPTION_STYLE = CaptionStyle(
    font="Montserrat ExtraBold",
    size=render.CAPTION_FONT_SIZE,
    color="#FFFFFF",
    highlight="#F4D35E",
    outline=render.CAPTION_OUTLINE,
    margin_v=422,
    margin_h=render.CAPTION_MARGIN_H,
)
BAND = (0, 330, W, 380)  # where a one-line caption sits with margin_v=422
BOTTOM_ZONE = (0, 395, W, H)  # covered by YouTube's own buttons: must stay empty
SIDES = [(0, 300, 12, H), (W - 12, 300, W, H)]


def _frame_rgb(video: Path, at_s: float) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{at_s}", "-i", str(video),
         "-frames:v", "1", "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    return result.stdout


def _count(frame: bytes, region: tuple[int, int, int, int], match: Callable[..., bool]) -> int:
    x0, y0, x1, y1 = region
    total = 0
    for y in range(y0, y1):
        row = frame[(y * W + x0) * 3 : (y * W + x1) * 3]
        total += sum(1 for i in range(0, len(row), 3) if match(*row[i : i + 3]))
    return total


def _white(r: int, g: int, b: int) -> bool:
    return min(r, g, b) > 235


def _yellow(r: int, g: int, b: int) -> bool:
    return abs(r - YELLOW[0]) < 40 and abs(g - YELLOW[1]) < 40 and abs(b - YELLOW[2]) < 40


@pytest.fixture
def captioned(tmp_path: Path, ffmpeg: None) -> Path:
    words = [
        CaptionWord("Honey", 2.0, 2.3),
        CaptionWord("outlives", 2.4, 2.7),
        CaptionWord("pharaohs.", 2.8, 3.1),
    ]
    subtitles = build_ass(words, caption_chunks(words, 3), CAPTION_STYLE)
    out = tmp_path / "captioned.mp4"
    render.render_video(
        frames=[_solid(tmp_path / "a.png", GRAY), _solid(tmp_path / "b.png", GRAY)],
        scene_weights=[1, 1],
        voice=_tone(tmp_path / "voice.wav", 4.0, gaps=True),
        duration_s=4.0,
        hook="This one sounds made up.",
        font=render.font_path("Montserrat ExtraBold"),
        output=out,
        subtitles=subtitles,
    )
    return out


def test_caption_shows_with_the_spoken_word_highlighted(captioned: Path) -> None:
    frame = _frame_rgb(captioned, 2.5)  # "outlives" is being spoken

    assert _count(frame, BAND, _white) > 30  # the other words of the chunk
    assert _count(frame, BAND, _yellow) > 30  # the current word


def test_no_caption_before_the_first_word_or_after_it_lingers(captioned: Path) -> None:
    assert _count(_frame_rgb(captioned, 1.0), BAND, _white) == 0
    assert _count(_frame_rgb(captioned, 3.8), BAND, _white) == 0  # 3.1 s + 0.3 s linger


def test_caption_stays_out_of_youtube_buttons_and_off_the_edges(captioned: Path) -> None:
    frame = _frame_rgb(captioned, 2.5)

    def ink(r: int, g: int, b: int) -> bool:
        return _white(r, g, b) or _yellow(r, g, b)

    assert _count(frame, BOTTOM_ZONE, ink) == 0
    assert all(_count(frame, side, ink) == 0 for side in SIDES)


def test_step_draws_captions_but_not_over_the_hook(rendered: StepContext) -> None:
    words, _ = captions.load(rendered.run_dir)
    hook_words = len(script.load(rendered.run_dir).scenes[0].narration.split())
    video = rendered.run_dir / render.OUTPUT
    during_hook = words[1].start + 0.1
    after_hook = words[hook_words + 1].start + 0.1

    assert _count(_frame_rgb(video, during_hook), BAND, _white) == 0
    assert _count(_frame_rgb(video, after_hook), BAND, _white) > 30


def test_step_rerenders_when_captions_change(rendered: StepContext) -> None:
    path = rendered.run_dir / captions.OUTPUT
    data = json.loads(path.read_text(encoding="utf-8"))
    data["words"][-1]["end"] -= 0.1
    path.write_text(json.dumps(data), encoding="utf-8")

    assert not render.run(rendered).cached


def test_ffmpeg_without_libass_is_a_clear_error(
    rendered: StepContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _solid(visuals.frame_paths(rendered.run_dir, 1)[0], RED)  # force a real render attempt
    monkeypatch.setattr("pipeline.steps.render.has_filter", lambda _name: False)

    with pytest.raises(StepError, match="libass") as exc:
        render.run(rendered)

    assert not exc.value.retryable
