"""Step `voice` on the fake TTS: audio file, duration, caching, guards."""

from dataclasses import replace

import pytest

from pipeline.adapters.tts.fake import FakeTts
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
