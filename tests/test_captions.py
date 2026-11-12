"""Captions: fake transcriber, word alignment, chunking, the step, the whisper adapter."""

import itertools
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.stt import Transcript, WordTiming
from pipeline.adapters.stt.fake import FakeTranscriber, find_beeps
from pipeline.adapters.stt.whisper import WhisperTranscriber, decode_for_whisper, vocabulary
from pipeline.adapters.tts import TtsRequest
from pipeline.adapters.tts.fake import FakeTts
from pipeline.media import decode_pcm16
from pipeline.steps import captions, script, voice
from pipeline.steps.base import StepContext, StepError


def _wav(tmp_path: Path, text: str) -> Path:
    result = FakeTts().synthesize(
        TtsRequest(text=text, voice_id="v", stability=0.5, similarity_boost=0.5, speed=1.0)
    )
    path = tmp_path / "voice.wav"
    path.write_bytes(result.audio)
    return path


def _heard(*words: tuple[str, float, float]) -> tuple[WordTiming, ...]:
    return tuple(WordTiming(w, s, e) for w, s, e in words)


# --- the fake really listens ------------------------------------------------------


def test_fake_transcriber_finds_one_beep_per_word(tmp_path: Path) -> None:
    beeps, duration = find_beeps(_wav(tmp_path, "one two three four five"))

    assert len(beeps) == 5
    assert duration == pytest.approx(2.0, abs=0.01)  # 5 words x 0.4 s
    starts = [start for start, _ in beeps]
    assert starts == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6], abs=0.02)


def test_fake_transcriber_takes_words_from_the_hint(tmp_path: Path) -> None:
    transcript = FakeTranscriber().transcribe(
        _wav(tmp_path, "a b c"), language="en", hint="Honey never spoils"
    )

    assert [w.word for w in transcript.words] == ["Honey", "never", "spoils"]


# --- alignment --------------------------------------------------------------------


def test_exact_match_takes_heard_times() -> None:
    heard = _heard(("Honey", 0.0, 0.3), ("never", 0.4, 0.7), ("spoils.", 0.8, 1.2))

    words, match = captions.align(["Honey", "never", "spoils."], heard, 1.5)

    assert match == 1.0
    assert [(w.text, w.start, w.end) for w in words] == [
        ("Honey", 0.0, 0.3),
        ("never", 0.4, 0.7),
        ("spoils.", 0.8, 1.2),
    ]


def test_misheard_word_keeps_its_slot_but_script_spelling() -> None:
    heard = _heard(("enzyme", 0.0, 0.4), ("gloo", 0.5, 0.9), ("oxidase", 1.0, 1.5))

    words, match = captions.align(["enzyme", "glucose", "oxidase"], heard, 2.0)

    assert [w.text for w in words] == ["enzyme", "glucose", "oxidase"]  # what viewers read
    assert (words[1].start, words[1].end) == (0.5, 0.9)  # when it was said
    assert match == pytest.approx(2 / 3)


def test_missed_words_share_the_gap_between_neighbours() -> None:
    heard = _heard(("one", 0.0, 0.5), ("four", 3.0, 3.5))

    words, _ = captions.align(["one", "two", "three", "four"], heard, 4.0)

    assert [w.start for w in words] == pytest.approx([0.0, 0.5, 1.75, 3.0])
    assert all(a.end <= b.start + 1e-9 for a, b in itertools.pairwise(words))


def test_case_and_punctuation_do_not_matter_and_extra_heard_words_are_ignored() -> None:
    heard = _heard(("uh", 0.0, 0.1), ("HERE'S", 0.2, 0.5), ("how", 0.6, 0.9))

    words, match = captions.align(["Here's", "how."], heard, 1.0)

    assert match == 1.0
    assert words[0].start == 0.2


def test_times_never_run_backwards() -> None:
    heard = _heard(("a", 0.0, 1.0), ("b", 0.5, 0.8))  # whisper overlap

    words, _ = captions.align(["a", "b"], heard, 2.0)

    assert words[1].start >= words[0].end


# --- chunking ---------------------------------------------------------------------


def test_chunks_respect_size_and_sentence_ends() -> None:
    text = ["Honey", "outlives", "pharaohs.", "Here's", "how", "it", "works", "today"]
    words = [captions.CaptionWord(w, i * 0.4, i * 0.4 + 0.3) for i, w in enumerate(text)]

    chunks = captions.chunk(words, per_chunk=3)

    assert [c.text for c in chunks] == ["Honey outlives pharaohs.", "Here's how it", "works today"]
    assert (chunks[1].first, chunks[1].last) == (3, 5)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == words[-1].end


# --- the step ---------------------------------------------------------------------


@pytest.fixture
def voiced(ctx: StepContext) -> StepContext:
    script.run(ctx)
    voice.run(ctx)
    return ctx


def test_step_times_every_script_word(voiced: StepContext) -> None:
    result = captions.run(voiced)

    words, chunks = captions.load(voiced.run_dir)
    narration = script.load(voiced.run_dir).narration.split()
    assert result.outputs == (voiced.run_dir / captions.OUTPUT,)
    assert [w.text for w in words] == narration
    assert words[-1].end <= voice.duration_s(voiced.run_dir) + 0.01
    per_chunk = voiced.series.captions.words_per_chunk
    assert all(c.last - c.first + 1 <= per_chunk for c in chunks)


def test_step_is_cached(voiced: StepContext) -> None:
    assert isinstance(voiced.stt, FakeTranscriber)
    captions.run(voiced)

    assert captions.run(voiced).cached
    assert len(voiced.stt.calls) == 1


def test_step_refuses_a_voice_that_does_not_match_the_script(voiced: StepContext) -> None:
    class WrongVoice(FakeTranscriber):
        def transcribe(self, audio: Path, *, language: str, hint: str | None = None) -> Transcript:
            heard = super().transcribe(audio, language=language, hint=None)  # "word1", "word2"...
            return heard

    with pytest.raises(StepError, match="voice does not match") as exc:
        captions.run(replace(voiced, stt=WrongVoice()))

    assert not exc.value.retryable  # the voice must be redone, not the captions
    assert not (voiced.run_dir / captions.OUTPUT).exists()


def test_step_needs_the_voice_first(ctx: StepContext) -> None:
    script.run(ctx)

    with pytest.raises(StepError, match="voice step first"):
        captions.run(ctx)


# --- whisper adapter on a fake model ----------------------------------------------


class FakeWhisperModel:
    def __init__(self, words: list[tuple[str, float, float]], fail: Exception | None = None):
        self.words, self.fail = words, fail
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
        self.calls.append({"audio": audio, **kwargs})

        def segments() -> Any:
            if self.fail is not None:
                raise self.fail  # faster-whisper decodes lazily, while iterating
            yield SimpleNamespace(
                words=[SimpleNamespace(word=f" {w}", start=s, end=e, probability=0.9)
                       for w, s, e in self.words]
            )  # fmt: skip

        return segments(), SimpleNamespace(duration=2.5)


def test_whisper_adapter_maps_words_and_loads_the_model_once(tmp_path: Path) -> None:
    model = FakeWhisperModel([("Honey", 0.0, 0.4), ("never", 0.5, 0.8)])
    loads: list[str] = []

    def load(name: str) -> FakeWhisperModel:
        loads.append(name)
        return model

    stt = WhisperTranscriber("small.en", load=load, decode=lambda _path: "samples")
    first = stt.transcribe(tmp_path / "v.wav", language="en", hint="Honey never spoils")
    stt.transcribe(tmp_path / "v.wav", language="en")

    assert loads == ["small.en"]  # heavy model is loaded once per run
    assert [w.word for w in first.words] == ["Honey", "never"]  # leading spaces stripped
    assert first.duration_s == 2.5
    call = model.calls[0]
    assert call["language"] == "en"
    assert call["word_timestamps"] is True
    assert call["vad_filter"] is False
    assert call["condition_on_previous_text"] is False


def test_whisper_prompt_is_vocabulary_not_the_whole_script(tmp_path: Path) -> None:
    model = FakeWhisperModel([])
    stt = WhisperTranscriber("small.en", load=lambda _name: model, decode=lambda _path: "samples")
    hint = "Bees add an enzyme called glucose oxidase. Honey never spoils."

    stt.transcribe(tmp_path / "v.wav", language="en", hint=hint)

    prompt = model.calls[0]["initial_prompt"]
    assert "oxidase" in prompt
    assert "Honey" in prompt
    assert "add an" not in prompt  # not the script itself, or whisper would echo it
    assert vocabulary("a " * 2000) == ""  # common words never enter the prompt


def test_whisper_failure_while_decoding_is_transient(tmp_path: Path) -> None:
    model = FakeWhisperModel([], fail=RuntimeError("decoder blew up"))
    stt = WhisperTranscriber("small.en", load=lambda _name: model, decode=lambda _path: "samples")

    with pytest.raises(TransientAdapterError, match="decoder blew up"):
        stt.transcribe(tmp_path / "v.wav", language="en")


def test_whisper_not_installed_is_permanent(tmp_path: Path) -> None:
    def missing(_name: str) -> FakeWhisperModel:
        raise PermanentAdapterError("faster-whisper is not installed: run `uv sync --extra stt`")

    with pytest.raises(PermanentAdapterError, match="--extra stt"):
        WhisperTranscriber("small.en", load=missing).transcribe(tmp_path / "v.wav", language="en")


def test_whisper_gets_samples_never_a_file_path(tmp_path: Path) -> None:
    # Regression: faster-whisper 1.2.1 decodes paths through PyAV and breaks on PyAV 19
    # (`av.open() got an unexpected keyword argument 'metadata_errors'`). We decode with
    # FFmpeg and pass samples, so that code path is never used.
    model = FakeWhisperModel([("ok", 0.0, 0.2)])
    stt = WhisperTranscriber("small.en", load=lambda _name: model, decode=lambda _path: "samples")

    stt.transcribe(tmp_path / "voice.mp3", language="en")

    assert model.calls[0]["audio"] == "samples"


@pytest.mark.usefixtures("ffmpeg")
def test_ffmpeg_decodes_voice_to_16khz_mono_pcm(tmp_path: Path) -> None:
    audio = _wav(tmp_path, "one two three four five")  # 2.0 s at 22.05 kHz

    pcm = decode_pcm16(audio, sample_rate=16_000)

    assert len(pcm) == pytest.approx(2.0 * 16_000 * 2, rel=0.01)  # 2 bytes per sample


@pytest.mark.usefixtures("ffmpeg")
def test_decode_for_whisper_gives_float_samples(tmp_path: Path) -> None:
    pytest.importorskip("numpy", reason="installed with the `stt` extra")

    samples = decode_for_whisper(_wav(tmp_path, "one two"))

    assert samples.dtype.name == "float32"
    assert -1.0 <= float(samples.min()) <= float(samples.max()) <= 1.0


def test_broken_voice_file_is_permanent(tmp_path: Path) -> None:
    broken = tmp_path / "voice.wav"
    broken.write_bytes(b"not audio")

    with pytest.raises(PermanentAdapterError):
        decode_for_whisper(broken)


def test_prompt_echo_with_low_confidence_does_not_count_as_heard() -> None:
    # Recorded live (faster-whisper small.en, 2026-11-13): whisper could not understand
    # the speech and echoed the vocabulary prompt back. Before the probability filter
    # this matched 60 % of the script and passed the check with invented timings.
    script_words = [
        "Honey",
        "outlives",
        "pharaohs.",
        "Bees",
        "add",
        "an",
        "enzyme",
        "called",
        "glucose",
        "oxidase.",
    ]
    echo = (
        WordTiming("Honey,", 0.00, 0.52, 0.01),
        WordTiming("outlives,", 0.74, 1.38, 0.11),
        WordTiming("pharaohs,", 1.58, 2.74, 0.34),
        WordTiming("Bees,", 2.80, 3.02, 0.00),
        WordTiming("glucose,", 3.22, 3.88, 0.00),
        WordTiming("oxidase", 5.24, 6.42, 0.02),
    )

    _, match = captions.align(script_words, echo, 6.64)

    assert match == 0.0
    assert match < captions.MIN_MATCH  # the step refuses these captions


def test_hallucinated_outro_does_not_match() -> None:
    # Also recorded live: without a prompt, whisper "heard" a YouTube outro.
    script_words = [
        "Honey",
        "outlives",
        "pharaohs.",
        "Bees",
        "add",
        "an",
        "enzyme",
        "called",
        "glucose",
        "oxidase.",
    ]
    outro = tuple(
        WordTiming(w, i * 0.4, i * 0.4 + 0.3, 0.6)
        for i, w in enumerate(
            [
                "I",
                "hope",
                "you",
                "enjoy",
                "this",
                "video.",
                "Please",
                "like,",
                "share",
                "and",
                "subscribe!",
            ]
        )
    )

    _, match = captions.align(script_words, outro, 6.64)

    assert match == 0.0


def test_merged_words_share_the_span_where_they_were_heard() -> None:
    # Recorded live (no prompt): "glucose oxidase" came back as one word "glucosoxidase"
    # at 3.64-4.84 s, while the audio runs to 5.65 s. Both words must stay in 3.64-4.84.
    script_words = ["enzyme", "called", "glucose", "oxidase."]
    heard = (
        WordTiming("enzyme", 2.94, 3.22, 0.84),
        WordTiming("called", 3.22, 3.64, 1.00),
        WordTiming("glucosoxidase.", 3.64, 4.84, 0.86),
    )

    words, match = captions.align(script_words, heard, 5.65)

    assert match == pytest.approx(0.5)
    assert (words[2].start, words[2].end) == pytest.approx((3.64, 4.24))
    assert (words[3].start, words[3].end) == pytest.approx((4.24, 4.84))  # not 5.65
