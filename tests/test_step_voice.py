"""Step `voice` on the fake TTS: audio file, duration, caching, guards."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.adapters.stt import Transcript, WordTiming
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.tts.fake import FakeTts
from pipeline.artifacts import meta_path
from pipeline.steps import script, voice
from pipeline.steps.base import StepContext, StepError
from pipeline.testing import TimedFakeTts
from pipeline.text import spoken_tokens


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


@pytest.fixture
def real_voice(ctx: StepContext) -> StepContext:
    """A voice_id a real provider accepts (the repo config may still hold the placeholder)."""
    chosen = ctx.series.voice.model_copy(update={"voice_id": "test-voice"})
    return replace(ctx, series=ctx.series.model_copy(update={"voice": chosen}))


# --- provider timings ------------------------------------------------------------------


def test_provider_timings_are_kept_next_to_the_voice(real_voice: StepContext) -> None:
    ctx = real_voice
    script.run(ctx)
    voice.run(replace(ctx, tts=TimedFakeTts(audio_format="mp3")))

    path, _ = voice.voice_file(ctx.run_dir)
    timings = voice.alignment(ctx.run_dir)
    assert path.name == "voice.mp3"
    assert timings is not None
    assert "".join(timings.characters) == script.load(ctx.run_dir).narration


def test_cached_voice_keeps_its_timings_and_its_check(real_voice: StepContext) -> None:
    ctx = real_voice
    script.run(ctx)
    tts, stt = TimedFakeTts(), FakeTranscriber()
    voice.run(replace(ctx, tts=tts, stt=stt))
    again = voice.run(replace(ctx, tts=tts, stt=stt))

    assert again.cached
    assert len(tts.calls) == 1
    assert len(stt.calls) == 1  # what was heard is saved with the voice
    assert voice.alignment(ctx.run_dir) is not None
    assert again.notes[0] == "whisper heard 100.0% of the script"


def test_fake_voice_has_no_timings(ctx: StepContext) -> None:
    script.run(ctx)
    voice.run(ctx)

    assert voice.alignment(ctx.run_dir) is None


def test_new_voice_replaces_old_file_of_other_format_and_its_timings(
    real_voice: StepContext,
) -> None:
    # Switching provider in one run dir (resume with --real tts and back) must never leave
    # the old file to be picked up by captions or render.
    ctx = real_voice
    script.run(ctx)
    voice.run(replace(ctx, tts=TimedFakeTts(audio_format="mp3")))
    voice.run(ctx)  # fake TTS: WAV, no timings

    path, _ = voice.voice_file(ctx.run_dir)
    assert path.name == "voice.wav"
    assert not (ctx.run_dir / "voice.mp3").exists()
    assert not (ctx.run_dir / voice.ALIGNMENT).exists()
    assert voice.alignment(ctx.run_dir) is None

    voice.run(replace(ctx, tts=TimedFakeTts(audio_format="mp3")))

    path, _ = voice.voice_file(ctx.run_dir)
    assert path.name == "voice.mp3"
    assert not (ctx.run_dir / "voice.wav").exists()
    assert voice.alignment(ctx.run_dir) is not None


def test_timings_from_another_voice_are_ignored(real_voice: StepContext) -> None:
    ctx = real_voice
    script.run(ctx)
    voice.run(replace(ctx, tts=TimedFakeTts()))
    meta = meta_path(ctx.run_dir / voice.ALIGNMENT)
    data = json.loads(meta.read_text(encoding="utf-8"))
    meta.write_text(json.dumps({**data, "input_hash": "from-an-older-voice"}), encoding="utf-8")

    assert voice.alignment(ctx.run_dir) is None


FIXTURES = Path(__file__).parent / "fixtures" / "whisper"


# --- the voice check: listen, compare, redo --------------------------------------------


class HearsWrongWords(FakeTranscriber):
    """Hears a beep per word but not the words: like a voice that garbled the script."""

    def __init__(self, bad_takes: int = 99) -> None:
        super().__init__()
        self.bad_takes = bad_takes

    def transcribe(
        self, audio: Path, *, language: str, hint: str | None = None, expect: str | None = None
    ) -> Transcript:
        good = len(self.calls) >= self.bad_takes
        return super().transcribe(audio, language=language, hint=hint if good else None)


def test_voice_that_never_matches_fails_after_the_allowed_redos(ctx: StepContext) -> None:
    script.run(ctx)
    tts = FakeTts()

    with pytest.raises(StepError, match="after 3 takes") as exc:
        voice.run(replace(ctx, tts=tts, stt=HearsWrongWords()))

    assert not exc.value.retryable  # the runner stops and alerts
    first_words = ", ".join(spoken_tokens(script.load(ctx.run_dir).narration)[:3])
    assert f"missed in the last take: {first_words}" in str(exc.value)  # names the words
    assert len(tts.calls) == 1 + ctx.series.qa.voice_max_redos
    assert list(ctx.run_dir.glob("voice.*")) == []  # a rejected take is never cached
    kept = sorted(p.name for p in (ctx.run_dir / voice.REJECTED).iterdir())
    assert kept == [f"take{n}.{ext}" for n in (1, 2, 3) for ext in ("json", "wav")]


def test_a_bad_take_is_redone_and_the_good_one_kept(ctx: StepContext) -> None:
    script.run(ctx)
    tts = FakeTts()

    result = voice.run(replace(ctx, tts=tts, stt=HearsWrongWords(bad_takes=1)))

    assert len(tts.calls) == 2
    assert not result.cached
    assert result.notes[0] == "whisper heard 100.0% of the script (take 2)"
    assert voice.voice_file(ctx.run_dir)[0].exists()


def test_no_redos_allowed_means_one_take(ctx: StepContext) -> None:
    strict = ctx.series.qa.model_copy(update={"voice_max_redos": 0})
    ctx = replace(ctx, series=ctx.series.model_copy(update={"qa": strict}))
    script.run(ctx)
    tts = FakeTts()

    with pytest.raises(StepError, match="after 1 takes"):
        voice.run(replace(ctx, tts=tts, stt=HearsWrongWords()))
    assert len(tts.calls) == 1


def test_numbers_in_the_script_match_numbers_said_as_words() -> None:
    heard = tuple(
        WordTiming(w, 0.0, 0.0)
        for w in ["a", "full", "spin", "takes", "two", "hundred", "forty-three", "Earth", "days"]
    )

    assert voice.voice_match("A full spin takes 243 Earth days.", heard) == 1.0


def test_unsure_common_words_count_but_unsure_prompt_words_do_not() -> None:
    # "On" and "fifty" were heard right but unsure (videos 7, 8); "oxidase" is a prompt word,
    # the kind whisper echoes when it cannot understand the speech.
    script = "On hot days glucose oxidase works. Fifty more."
    heard = tuple(
        WordTiming(w, 0.0, 0.0, p)
        for w, p in [("On", 0.01), ("hot", 0.9), ("days", 0.9), ("glucose", 0.9),
                     ("oxidase", 0.2), ("works", 0.9), ("Fifty", 0.08), ("more", 0.9)]
    )  # fmt: skip

    result = voice.compare(script, heard)

    assert result.missed == ["oxidase"]
    assert result.match == pytest.approx(7 / 8)


def test_an_echo_of_the_prompt_is_not_a_match() -> None:
    script = "Bees add an enzyme called glucose oxidase to the nectar they bring back home."
    echo = tuple(WordTiming(w, 0.0, 0.0, 0.1) for w in ["glucose", "oxidase"])

    assert voice.voice_match(script, echo) == 0.0


@pytest.mark.parametrize(("video", "expected"), [(7, 66 / 67), (8, 1.0)])
def test_real_checks_pass_with_the_new_rule(video: int, expected: float) -> None:
    # Failed takes in videos 5 and 6 sat at 90-94 % for the same reason.
    data = json.loads((FIXTURES / f"video{video}.json").read_text(encoding="utf-8"))
    heard = tuple(WordTiming(*w) for w in data["heard"])

    assert voice.voice_match(data["script"], heard) == pytest.approx(expected)


# --- voice clock vs whisper clock -------------------------------------------------------


def _timed(*words: tuple[str, float]) -> tuple[WordTiming, ...]:
    return tuple(WordTiming(w, s, s + 0.2) for w, s in words)


def test_drift_counts_words_both_sides_agree_on() -> None:
    provider = _timed(("Venus", 0.0), ("spins", 0.5), ("243", 1.0), ("days", 2.0))
    heard = _timed(("venus", 0.0), ("spins", 0.25), ("243", 0.4), ("days", 2.05))

    drift = voice.timing_drift(provider, heard)

    assert drift.compared == 4
    assert drift.over_notice == 2  # spins 0.25 s, 243 0.6 s
    assert drift.over_limit == 1
    assert drift.worst_s == pytest.approx(0.6)
    assert drift.too_much  # 1 of 4 words > 15 %


def test_matching_clocks_give_a_quiet_report(real_voice: StepContext) -> None:
    ctx = real_voice
    script.run(ctx)
    result = voice.run(replace(ctx, tts=TimedFakeTts()))

    assert result.notes[1].startswith("timings voice vs whisper: 0/")
    assert not any(n.startswith("warning") for n in result.notes)


def test_a_provider_clock_off_by_more_than_the_limit_is_a_warning_not_a_failure(
    real_voice: StepContext,
) -> None:
    ctx = real_voice
    script.run(ctx)
    result = voice.run(replace(ctx, tts=TimedFakeTts(offset_s=0.6)))

    assert voice.voice_file(ctx.run_dir)[0].exists()
    assert result.notes[-1].startswith("warning: over 15% of word starts differ")


# --- second listen inside the step ------------------------------------------------------


class SkipsWordsOnce(FakeTranscriber):
    """Whole-take listen drops the given script words (like whisper on video 6); a clip
    listen with no prompt hears every beep of the clip as the script words in it."""

    def __init__(self, skip: set[int], script_words: list[str]) -> None:
        super().__init__()
        self.skip, self.script_words = skip, script_words

    def transcribe(
        self, audio: Path, *, language: str, hint: str | None = None, expect: str | None = None
    ) -> Transcript:
        full = super().transcribe(audio, language=language, hint=hint, expect=expect)
        if hint is not None:  # the whole take
            kept = tuple(w for i, w in enumerate(full.words) if i not in self.skip)
            return replace(full, words=kept)
        # A clip: label its beeps with the script words around the skipped ones.
        first = max(min(self.skip) - 1, 0)
        names = self.script_words[first : first + len(full.words)]
        return replace(
            full, words=tuple(replace(w, word=n) for w, n in zip(full.words, names, strict=True))
        )


@pytest.mark.usefixtures("ffmpeg")
def test_words_whisper_skipped_are_found_again_and_the_take_kept(
    real_voice: StepContext,
) -> None:
    ctx = real_voice
    script.run(ctx)
    words = script.load(ctx.run_dir).narration.split()
    tts = TimedFakeTts()
    stt = SkipsWordsOnce({3, 4, 5, 6, 7}, words)  # ~7 % of the script: under the 10 % limit

    result = voice.run(replace(ctx, tts=tts, stt=stt))

    assert len(tts.calls) == 1  # no new take was paid for
    assert "the second listen found" in result.notes[0]
    assert result.notes[0].endswith(": 100.0%")


@pytest.mark.usefixtures("ffmpeg")
def test_rejected_takes_keep_their_timings_and_the_folder_is_capped(
    real_voice: StepContext,
) -> None:
    tiny = real_voice.series.qa.model_copy(update={"rejected_takes_max_mb": 0.001})
    ctx = replace(real_voice, series=real_voice.series.model_copy(update={"qa": tiny}))
    script.run(ctx)

    with pytest.raises(StepError):
        voice.run(replace(ctx, tts=TimedFakeTts(), stt=HearsWrongWords()))

    kept = sorted(p.name for p in (ctx.run_dir / voice.REJECTED).iterdir())
    assert kept == ["take3.alignment.json", "take3.json", "take3.wav"]  # newest only
    report = json.loads((ctx.run_dir / voice.REJECTED / "take3.json").read_text())
    assert report["match"] == 0.0


@pytest.mark.usefixtures("ffmpeg")
def test_recheck_listens_to_every_kept_take_and_changes_nothing(real_voice: StepContext) -> None:
    ctx = real_voice
    script.run(ctx)
    with pytest.raises(StepError):
        voice.run(replace(ctx, tts=TimedFakeTts(), stt=HearsWrongWords()))
    before = sorted((p.name, p.stat().st_mtime_ns) for p in ctx.run_dir.rglob("*"))

    results = voice.recheck(ctx.run_dir, ctx.series, FakeTranscriber(), "en")

    assert [name for name, _ in results] == [
        "rejected_voice/take1.wav",
        "rejected_voice/take2.wav",
        "rejected_voice/take3.wav",
    ]
    assert all(result.match == 1.0 for _, result in results)  # a listener that hears right
    assert sorted((p.name, p.stat().st_mtime_ns) for p in ctx.run_dir.rglob("*")) == before
