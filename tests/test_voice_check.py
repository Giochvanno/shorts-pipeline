"""The voice check: compare, locate missed words, and listen again without inventing words.

The negative tests (silence or another word, with the expected text as prompt)
are the ones that matter most: a second listen that can be talked into hearing
its prompt would quietly let real voice errors through.
"""

import array
import json
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest

from pipeline.adapters.stt import Transcript, WordTiming
from pipeline.adapters.tts import TtsRequest
from pipeline.steps.voice import alignment_words
from pipeline.testing import TimedFakeTts
from pipeline.voice_check import (
    MAX_NO_SPEECH,
    PAD_S,
    accepts,
    check,
    compare,
    relisten,
    speech_needed,
    stretches,
)

FIXTURES = Path(__file__).parent / "fixtures" / "whisper"
EXPECTED = ("number", "three")
CONTEXT = frozenset({"feet", "number", "three", "fungal"})


def said(*words: str, p: float = 0.95, no_speech: float = 0.05) -> Transcript:
    timed = tuple(WordTiming(w, 0.1 * i, 0.1 * i + 0.1, p) for i, w in enumerate(words))
    return Transcript(timed, 1.0, "test", "test", no_speech_prob=no_speech if words else 1.0)


# --- the decision for one stretch -------------------------------------------------------


def test_heard_without_a_prompt_counts() -> None:
    assert accepts(EXPECTED, CONTEXT, 0.6, said("feet", "Number", "three", "Fungal"), None)


def test_digits_heard_without_a_prompt_count() -> None:
    assert accepts(EXPECTED, CONTEXT, 0.6, said("Number", "3."), None)


def test_heard_with_the_prompt_counts_when_whisper_is_sure() -> None:
    # The video 6 shape: the plain listen hears only the neighbours again.
    plain = said("feet", "Fungal")
    assert accepts(EXPECTED, CONTEXT, 0.6, plain, said("Number", "three", no_speech=0.1))


def test_silence_with_the_prompt_never_counts() -> None:
    # REQUIRED: whisper may "hear" its prompt in silence, confidently.
    echo = said("Number", "three", p=0.99, no_speech=0.0)

    assert not accepts(EXPECTED, CONTEXT, 0.0, said(), echo)


def test_another_word_with_the_prompt_never_counts() -> None:
    # REQUIRED: speech is there, but whisper plainly hears something else; the prompted
    # listen then "hears" the prompt over it.
    other = said("cat", "nap")
    echo = said("Number", "three", p=0.99, no_speech=0.0)

    assert not accepts(EXPECTED, CONTEXT, 0.6, other, echo)


def test_too_little_voice_for_the_words_never_counts() -> None:
    # 0.06 s of sound cannot hold "number three" (0.66 s in the real take), whatever
    # whisper says: a sound leaking in from a neighbour is not the words.
    assert speech_needed(EXPECTED) == pytest.approx(0.11)
    assert not accepts(EXPECTED, CONTEXT, 0.06, said("Number", "three"), None)


def test_prompted_result_needs_whisper_sure_there_is_speech() -> None:
    unsure_speech = said("Number", "three", no_speech=MAX_NO_SPEECH + 0.1)

    assert not accepts(EXPECTED, CONTEXT, 0.6, said(), unsure_speech)


def test_prompted_result_needs_every_word_sure() -> None:
    assert not accepts(EXPECTED, CONTEXT, 0.6, said(), said("Number", "three", p=0.2))


# --- comparing ----------------------------------------------------------------------------


def _fixture(name: str) -> tuple[str, tuple[WordTiming, ...]]:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data["script"], tuple(WordTiming(*w) for w in data["heard"])


def test_percent_written_apart_now_matches() -> None:
    # Video 6: whisper wrote "85 %" and "40 %"; the script says "percent".
    script, heard = _fixture("video6_take1.json")

    missed = compare(script, heard).missed

    assert "percent" not in missed
    assert missed == ["number", "three", "number", "two", "number", "one"]


def test_missed_runs_are_grouped_and_keep_their_script_words() -> None:
    words = ["trees", "trade", "fungal", "threads"]
    heard = tuple(WordTiming(w, i * 0.4, i * 0.4 + 0.3) for i, w in enumerate(words))
    result = compare("Trees trade. Number three. Fungal threads.", heard)

    assert result.runs() == [(2, 3)]
    assert result.words[2:4] == (2, 3)


# --- where to listen again ----------------------------------------------------------------


def test_real_take_windows_cover_the_skipped_phrases() -> None:
    # No provider timings were kept for this take: windows come from whisper's neighbours.
    script, heard = _fixture("video6_take1.json")
    result = compare(script, heard)

    windows = stretches(result, heard, None, duration_s=42.66)

    assert [result.tokens[w.first : w.last + 1] for w in windows] == [
        ("number", "three"),
        ("number", "two"),
        ("number", "one"),
    ]
    speech = [(2.6, 3.3), (15.6, 16.3), (24.1, 24.7)]  # measured in the take's audio
    for window, (start, end) in zip(windows, speech, strict=True):
        assert window.start_s <= start and end <= window.end_s
        assert window.end_s - window.start_s < 4  # a clip, not the whole take
        assert not window.exact  # so the prompt is never used on this take


def test_provider_timings_give_tight_windows() -> None:
    request = TtsRequest(text="Trees trade. Number three. Fungal threads.", voice_id="v",
                         stability=0.5, similarity_boost=0.5, speed=1)  # fmt: skip
    timings = TimedFakeTts().synthesize(request).alignment
    assert timings is not None
    provider = alignment_words(timings)
    heard = tuple(w for i, w in enumerate(provider) if i not in (2, 3))
    result = compare(request.text, heard)

    (window,) = stretches(result, heard, provider, duration_s=3.0)

    assert window.start_s == pytest.approx(provider[2].start - PAD_S)
    assert window.end_s == pytest.approx(provider[3].end + PAD_S)
    assert (window.core_start_s, window.core_end_s) == (provider[2].start, provider[3].end)
    assert {"number", "three"} <= window.context
    assert window.exact


# --- listening again on real audio (FakeTts beeps) --------------------------------------


class Scripted:
    """A transcriber that answers from a list, and records how it was asked."""

    provider, model = "scripted", "test"

    def __init__(self, *answers: Transcript, echo_prompt: bool = False) -> None:
        self.answers = list(answers)
        self.echo_prompt = echo_prompt
        self.asked: list[str | None] = []

    def transcribe(
        self, audio: Path, *, language: str, hint: str | None = None, expect: str | None = None
    ) -> Transcript:
        self.asked.append(expect)
        if expect is not None and self.echo_prompt:  # the hallucination: says the prompt back
            return said(*expect.split(), p=0.99, no_speech=0.0)
        return self.answers.pop(0)


TEXT = "Trees trade sugar. Number three. Fungal threads tangle."


def _take(
    tmp_path: Path, *, silence_words: Sequence[int] = ()
) -> tuple[Path, tuple[WordTiming, ...]]:
    """A beep take of TEXT and the provider's word timings; chosen words cut to silence."""
    request = TtsRequest(text=TEXT, voice_id="v", stability=0.5, similarity_boost=0.5, speed=1)
    result = TimedFakeTts().synthesize(request)
    assert result.alignment is not None
    provider = alignment_words(result.alignment)
    path = tmp_path / "voice.wav"
    path.write_bytes(result.audio)
    if silence_words:
        with wave.open(str(path), "rb") as wav:
            params, rate = wav.getparams(), wav.getframerate()
            samples = array.array("h", wav.readframes(wav.getnframes()))
        for i in silence_words:
            for n in range(int(provider[i].start * rate), int(provider[i].end * rate)):
                samples[n] = 0
        with wave.open(str(path), "wb") as wav:
            wav.setparams(params)
            wav.writeframes(samples.tobytes())
    return path, provider


def _missing(provider: Sequence[WordTiming], *skip: int) -> tuple[WordTiming, ...]:
    return tuple(w for i, w in enumerate(provider) if i not in skip)


@pytest.mark.usefixtures("ffmpeg")
def test_relisten_finds_words_whisper_skipped(tmp_path: Path) -> None:
    audio, provider = _take(tmp_path)
    heard = _missing(provider, 3, 4)  # "Number three." not heard in the whole take
    stt = Scripted(said("sugar", "Number", "three", "Fungal"))

    rescue = relisten(
        stt, audio, compare(TEXT, heard), heard, provider, language="en", duration_s=4.0
    )

    assert [compare(TEXT, heard).tokens[i] for i in rescue.found] == ["number", "three"]
    assert stt.asked == [None]  # heard plainly: the prompt was never needed


@pytest.mark.usefixtures("ffmpeg")
def test_relisten_never_rescues_words_the_voice_did_not_say(tmp_path: Path) -> None:
    # REQUIRED, end to end: the voice skipped "Number three." (silence there) and the
    # transcriber echoes any prompt confidently. Nothing may be rescued.
    audio, provider = _take(tmp_path, silence_words=(3, 4))
    heard = _missing(provider, 3, 4)
    stt = Scripted(said("sugar", "Fungal"), echo_prompt=True)

    rescue = relisten(
        stt, audio, compare(TEXT, heard), heard, provider, language="en", duration_s=4.0
    )

    assert rescue.found == ()
    assert stt.asked == []  # silence is caught before whisper is even asked


@pytest.mark.usefixtures("ffmpeg")
def test_relisten_never_rescues_over_another_word(tmp_path: Path) -> None:
    # REQUIRED, end to end: there is sound, but whisper plainly hears other words in it.
    audio, provider = _take(tmp_path)
    heard = _missing(provider, 3, 4)
    stt = Scripted(said("sugar", "cat", "nap", "Fungal"), echo_prompt=True)

    rescue = relisten(
        stt, audio, compare(TEXT, heard), heard, provider, language="en", duration_s=4.0
    )

    assert rescue.found == ()
    assert stt.asked == [None, "number three"]  # it was asked, and the echo was refused


@pytest.mark.usefixtures("ffmpeg")
def test_without_provider_timings_the_prompt_is_never_used(tmp_path: Path) -> None:
    # The words cannot be isolated from their neighbours, so silence where the voice skipped
    # them would look like speech. Only a plain listen (nothing to echo) may rescue them.
    audio, provider = _take(tmp_path, silence_words=(3, 4))
    heard = _missing(provider, 3, 4)
    stt = Scripted(said("sugar", "Fungal"), echo_prompt=True)

    rescue = relisten(stt, audio, compare(TEXT, heard), heard, None, language="en", duration_s=4.0)

    assert rescue.found == ()
    assert stt.asked == [None]


@pytest.mark.usefixtures("ffmpeg")
def test_too_many_missed_words_are_not_listened_to_again(tmp_path: Path) -> None:
    audio, provider = _take(tmp_path)
    heard = _missing(provider, 0, 1, 2, 3, 4)  # half the script
    stt = Scripted()

    result = check(stt, audio, TEXT, heard, provider, language="en", duration_s=4.0,
                   min_match=0.95, max_missed=0.10)  # fmt: skip

    assert not result.relistened
    assert stt.asked == []
    assert result.match == result.first_match < 0.95


@pytest.mark.usefixtures("ffmpeg")
def test_check_reports_what_the_second_listen_found(tmp_path: Path) -> None:
    long_text = " ".join([TEXT] * 3)  # 2 of 30 words missed: under the 10 % limit
    request = TtsRequest(text=long_text, voice_id="v", stability=0.5, similarity_boost=0.5, speed=1)
    voice = TimedFakeTts().synthesize(request)
    assert voice.alignment is not None
    provider = alignment_words(voice.alignment)
    audio = tmp_path / "voice.wav"
    audio.write_bytes(voice.audio)
    heard = _missing(provider, 3, 4)
    stt = Scripted(said("sugar", "Number", "three", "Fungal"))

    result = check(stt, audio, long_text, heard, provider, language="en",
                   duration_s=voice.duration_s, min_match=0.95, max_missed=0.10)  # fmt: skip

    assert result.relistened
    assert result.first_match < 0.95 <= result.match == 1.0
    assert result.rescued == ("number", "three")
    assert result.missed == ()
