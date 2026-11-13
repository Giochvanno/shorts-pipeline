"""Live checks against the real ElevenLabs API with your key. Never run in CI.

    uv run pytest tests/live/test_elevenlabs_live.py -m live -s

- A wrong key and an unknown voice are refused before any synthesis: free.
- One short sentence is synthesized once with the series voice: about 45 credits
  (~$0.01 on Starter). The MP3 goes to runs/live/elevenlabs.mp3 to listen to.
- If whisper is installed (`uv sync --extra stt`), the same MP3 is transcribed
  locally ($0) and the word start times of both are printed side by side: the
  data for choosing caption timings (ElevenLabs or whisper).

It prints the raw HTTP status and error bodies, so docs and reality can be compared.
"""

import difflib
import importlib.util
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from pipeline.adapters.errors import PermanentAdapterError
from pipeline.adapters.http import HttpRequest, HttpResponse, urllib_transport
from pipeline.adapters.stt.whisper import WhisperTranscriber  # loads faster-whisper lazily
from pipeline.adapters.tts import Alignment, TtsRequest, TtsResult
from pipeline.adapters.tts.elevenlabs import API_KEY_ENV, CREDIT_HEADERS, ElevenLabsTts
from pipeline.config import load_channel, load_series
from pipeline.env import load_env
from pipeline.steps.voice import PLACEHOLDER_VOICE

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[2]
SENTENCE = "Octopuses have three hearts and blue blood."
PUBLIC_VOICE = "21m00Tcm4TlvDq8ikWAM"  # a premade voice from the ElevenLabs docs examples


class Recorder:
    """The real transport, keeping the last answer for inspection."""

    def __init__(self) -> None:
        self.last: HttpResponse | None = None

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.last = urllib_transport(request)
        return self.last


def _tts(api_key: str, transport: Recorder) -> ElevenLabsTts:
    chain = load_channel().models.tts
    return ElevenLabsTts(
        chain.primary.model,
        api_key=api_key,
        usd_per_credit=chain.usd_per_credit,
        timeout_s=chain.timeout_s,
        transport=transport,
    )


def _request(voice_id: str) -> TtsRequest:
    voice = load_series("fact-of-day").voice
    return TtsRequest(
        text=SENTENCE,
        voice_id=voice_id,
        stability=voice.stability,
        similarity_boost=voice.similarity_boost,
        speed=voice.speed,
    )


def _show(recorder: Recorder, error: Exception) -> None:
    assert recorder.last is not None
    print(f"\nHTTP {recorder.last.status}: {recorder.last.body[:300]!r}\n-> {error}")


@pytest.fixture(scope="module")
def api_key() -> str:
    load_env(REPO / ".env")
    key = os.environ.get(API_KEY_ENV)
    if not key:
        pytest.skip(f"{API_KEY_ENV} is not set: add it to .env")
    return key


@pytest.mark.usefixtures("ffmpeg")
def test_wrong_key_is_permanent_and_free() -> None:
    recorder = Recorder()

    with pytest.raises(PermanentAdapterError, match="API key") as exc:
        _tts("sk_definitely_not_a_real_key", recorder).synthesize(_request(PUBLIC_VOICE))
    _show(recorder, exc.value)


@pytest.mark.usefixtures("ffmpeg")
def test_unknown_voice_is_permanent_and_free(api_key: str) -> None:
    recorder = Recorder()

    with pytest.raises(PermanentAdapterError, match="voice") as exc:
        _tts(api_key, recorder).synthesize(_request("ThisVoiceDoesNotExist"))
    _show(recorder, exc.value)


@dataclass(frozen=True)
class Spoken:
    result: TtsResult
    path: Path
    headers: dict[str, str | None]


@pytest.fixture(scope="module")
def spoken(api_key: str, ffmpeg: None) -> Iterator[Spoken]:
    voice_id = load_series("fact-of-day").voice.voice_id
    if voice_id == PLACEHOLDER_VOICE:
        pytest.skip("set voice.voice_id in config/series/fact-of-day.yaml")
    recorder = Recorder()
    result = _tts(api_key, recorder).synthesize(_request(voice_id))
    assert recorder.last is not None
    path = REPO / "runs" / "live" / "elevenlabs.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.audio)
    names = (*CREDIT_HEADERS, "request-id")
    yield Spoken(result, path, {name: recorder.last.header(name) for name in names})


def _words(alignment: Alignment) -> list[tuple[str, float]]:
    """(word, start) from per-character timings; a word starts at its first character."""
    words: list[tuple[str, float]] = []
    current, start = "", 0.0
    for char, char_start in zip(alignment.characters, alignment.starts_s, strict=True):
        if char.isspace():
            if current:
                words.append((current, start))
            current = ""
            continue
        if not current:
            start = char_start
        current += char
    if current:
        words.append((current, start))
    return words


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def test_series_voice_speaks_with_timings_and_reports_credits(spoken: Spoken) -> None:
    result = spoken.result
    credits = result.cost_usd / load_channel().models.tts.usd_per_credit
    print(
        f"\nlive elevenlabs: {result.model}, {result.duration_s:.2f} s, "
        f"{result.characters} chars, ~{credits:.0f} credits, ${result.cost_usd:.4f}"
        f"\nheaders: {spoken.headers}\nsaved: {spoken.path}"
    )

    assert result.format == "mp3"
    assert 1.0 < result.duration_s < 15
    assert any(spoken.headers[name] is not None for name in CREDIT_HEADERS), (
        "no credits header in the answer: cost falls back to 1 credit per character"
    )
    assert result.alignment is not None
    assert "".join(result.alignment.characters) == SENTENCE
    assert result.alignment.ends_s[-1] <= result.duration_s + 0.1


def test_timings_agree_with_whisper(spoken: Spoken) -> None:
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("whisper is not installed: uv sync --extra stt")
    assert spoken.result.alignment is not None
    eleven = _words(spoken.result.alignment)
    stt_model = load_channel().models.stt.primary.model
    heard = WhisperTranscriber(stt_model).transcribe(spoken.path, language="en", hint=SENTENCE)
    whisper = [(w.word.strip(), w.start) for w in heard.words]

    matcher = difflib.SequenceMatcher(
        a=[_norm(w) for w, _ in eleven], b=[_norm(w) for w, _ in whisper], autojunk=False
    )
    diffs: list[float] = []
    print(f"\n{'word':<12}{'elevenlabs':>12}{'whisper':>10}{'diff':>8}")
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            word, el_start = eleven[block.a + k]
            _, wh_start = whisper[block.b + k]
            diffs.append(wh_start - el_start)
            print(f"{word:<12}{el_start:>12.2f}{wh_start:>10.2f}{wh_start - el_start:>+8.2f}")
    worst = max((abs(d) for d in diffs), default=0.0)
    print(f"matched {len(diffs)}/{len(eleven)} words, worst start difference {worst:.2f} s")

    assert len(diffs) >= len(eleven) * 0.8
    assert worst < 0.5  # beyond this, one of them would visibly miss the word
