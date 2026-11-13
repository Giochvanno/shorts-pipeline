"""Step `voice` on the fake TTS: audio file, duration, caching, guards."""

import json
from dataclasses import replace

import pytest

from pipeline.adapters.tts import Alignment, TtsRequest, TtsResult
from pipeline.adapters.tts.fake import FakeTts
from pipeline.artifacts import meta_path
from pipeline.steps import script, voice
from pipeline.steps.base import StepContext, StepError


def test_writes_audio_matching_series_length(ctx: StepContext) -> None:
    script.run(ctx)
    result = voice.run(ctx)

    (audio,) = result.outputs
    assert audio.name == "voice.wav"
    assert audio.stat().st_size > 0
    fmt = ctx.series.format.duration_s
    assert fmt.min <= voice.duration_s(ctx.run_dir) <= fmt.max


def test_second_run_is_cached_and_free(ctx: StepContext, fake_tts: FakeTts) -> None:
    script.run(ctx)
    voice.run(ctx)
    again = voice.run(ctx)

    assert again.cached
    assert len(fake_tts.calls) == 1


def test_changed_script_text_invalidates_voice_cache(ctx: StepContext, fake_tts: FakeTts) -> None:
    script.run(ctx)
    voice.run(ctx)

    script.run(replace(ctx, topic="how honey never spoils", revision=1))  # new text on disk
    redo = voice.run(ctx)

    assert not redo.cached
    assert len(fake_tts.calls) == 2


def test_identical_script_text_reuses_voice(ctx: StepContext, fake_tts: FakeTts) -> None:
    # The cache is keyed by the text itself: a "redo" that yields the same words is free.
    script.run(ctx)
    voice.run(ctx)

    script.run(replace(ctx, revision=1))  # the deterministic fake writes the same text
    again = voice.run(ctx)

    assert again.cached
    assert len(fake_tts.calls) == 1


def test_needs_a_script_first(ctx: StepContext) -> None:
    with pytest.raises(StepError, match="script step first") as exc:
        voice.run(ctx)

    assert not exc.value.retryable  # retrying cannot fix a missing input


def test_real_provider_refuses_placeholder_voice(ctx: StepContext) -> None:
    class PretendRealTts(FakeTts):
        @property
        def provider(self) -> str:
            return "elevenlabs"

    placeholder = ctx.series.voice.model_copy(update={"voice_id": voice.PLACEHOLDER_VOICE})
    ctx = replace(ctx, series=ctx.series.model_copy(update={"voice": placeholder}))
    script.run(ctx)

    with pytest.raises(StepError, match="voice_id") as exc:
        voice.run(replace(ctx, tts=PretendRealTts()))

    assert not exc.value.retryable


class TimedMp3Tts:
    """Stands in for ElevenLabs: MP3 bytes plus per-character timings."""

    provider = "timed"
    model = "timed-1"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, request: TtsRequest) -> TtsResult:
        self.calls += 1
        chars = tuple(request.text)
        return TtsResult(
            audio=b"ID3-not-really-mp3",
            format="mp3",
            duration_s=0.05 * len(chars) + 0.2,
            characters=len(request.text),
            provider=self.provider,
            model=self.model,
            cost_usd=0.01,
            alignment=Alignment(
                characters=chars,
                starts_s=tuple(0.05 * i for i in range(len(chars))),
                ends_s=tuple(0.05 * (i + 1) for i in range(len(chars))),
            ),
        )


@pytest.fixture
def real_voice(ctx: StepContext) -> StepContext:
    """A voice_id a real provider accepts (the repo config may still hold the placeholder)."""
    chosen = ctx.series.voice.model_copy(update={"voice_id": "test-voice"})
    return replace(ctx, series=ctx.series.model_copy(update={"voice": chosen}))


def test_provider_timings_are_kept_next_to_the_voice(real_voice: StepContext) -> None:
    ctx = real_voice
    script.run(ctx)
    voice.run(replace(ctx, tts=TimedMp3Tts()))

    path, _ = voice.voice_file(ctx.run_dir)
    timings = voice.alignment(ctx.run_dir)
    assert path.name == "voice.mp3"
    assert timings is not None
    assert "".join(timings.characters) == script.load(ctx.run_dir).narration
    assert timings.ends_s[0] == 0.05


def test_cached_voice_keeps_its_timings(real_voice: StepContext) -> None:
    ctx = real_voice
    script.run(ctx)
    tts = TimedMp3Tts()
    voice.run(replace(ctx, tts=tts))
    again = voice.run(replace(ctx, tts=tts))

    assert again.cached
    assert tts.calls == 1
    assert voice.alignment(ctx.run_dir) is not None


def test_fake_voice_has_no_timings(ctx: StepContext) -> None:
    script.run(ctx)
    voice.run(ctx)

    assert voice.alignment(ctx.run_dir) is None


def test_new_voice_replaces_old_file_of_other_format_and_its_timings(
    real_voice: StepContext,
) -> None:
    ctx = real_voice
    # Switching provider in one run dir (resume with --real tts and back) must never leave
    # the old file to be picked up by captions or render.
    script.run(ctx)
    voice.run(replace(ctx, tts=TimedMp3Tts()))
    voice.run(ctx)  # fake TTS: WAV, no timings

    path, _ = voice.voice_file(ctx.run_dir)
    assert path.name == "voice.wav"
    assert not (ctx.run_dir / "voice.mp3").exists()
    assert not (ctx.run_dir / voice.ALIGNMENT).exists()
    assert voice.alignment(ctx.run_dir) is None

    voice.run(replace(ctx, tts=TimedMp3Tts()))

    path, _ = voice.voice_file(ctx.run_dir)
    assert path.name == "voice.mp3"
    assert not (ctx.run_dir / "voice.wav").exists()
    assert voice.alignment(ctx.run_dir) is not None


def test_timings_from_another_voice_are_ignored(real_voice: StepContext) -> None:
    ctx = real_voice
    script.run(ctx)
    voice.run(replace(ctx, tts=TimedMp3Tts()))
    meta = meta_path(ctx.run_dir / voice.ALIGNMENT)
    data = json.loads(meta.read_text(encoding="utf-8"))
    meta.write_text(json.dumps({**data, "input_hash": "from-an-older-voice"}), encoding="utf-8")

    assert voice.alignment(ctx.run_dir) is None
