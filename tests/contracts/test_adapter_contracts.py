"""Contract tests: every implementation of a port must pass the same checks.

Today only the fakes are listed. Real adapters join these lists when they land
(marked to run only with API keys), so fake and real can never drift apart.
"""

import base64
import io
import itertools
import json
import subprocess
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pipeline.adapters.http import HttpRequest, HttpResponse
from pipeline.adapters.images import ImageClient, ImageRequest
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm import LlmClient, LlmRequest
from pipeline.adapters.llm.claude_cli import ClaudeCliLlm
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.stt import Transcriber
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.stt.whisper import WhisperTranscriber
from pipeline.adapters.tts import TtsClient, TtsRequest
from pipeline.adapters.tts.elevenlabs import ElevenLabsTts
from pipeline.adapters.tts.fake import FakeTts


def ClaudeCliOnFakeProcess() -> LlmClient:
    """The real adapter with subprocess replaced: CI has no CLI and no login."""
    answer = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": '{"ok": true}',
        "total_cost_usd": 0.001,
        "usage": {"input_tokens": 20, "output_tokens": 5},
    }

    def run(args: list[str], **_kwargs: object) -> "subprocess.CompletedProcess[bytes]":
        return subprocess.CompletedProcess(args, 0, json.dumps(answer).encode(), b"")

    return ClaudeCliLlm("model-from-config", run=run, which=lambda _name: "claude")


def ElevenLabsOnFakeHttp() -> TtsClient:
    """The real adapter with the network replaced: CI has no key and must not pay."""

    def transport(_request: HttpRequest) -> HttpResponse:
        answer = {"audio_base64": base64.b64encode(b"ID3audio").decode(), "alignment": None}
        return HttpResponse(200, json.dumps(answer).encode(), {"character-cost": "43"})

    return ElevenLabsTts(
        "model-from-config",
        api_key="test-key",
        usd_per_credit=0.0002,
        transport=transport,
        measure=lambda _audio: 2.5,
    )


LLM_IMPLEMENTATIONS: list[Callable[[], LlmClient]] = [FakeLlm, ClaudeCliOnFakeProcess]
TTS_IMPLEMENTATIONS: list[Callable[[], TtsClient]] = [FakeTts, ElevenLabsOnFakeHttp]

TTS_REQUEST = TtsRequest(
    text="Octopuses have three hearts and blue blood.",
    voice_id="test-voice",
    stability=0.5,
    similarity_boost=0.75,
    speed=1.0,
)


@pytest.fixture(params=LLM_IMPLEMENTATIONS, ids=lambda f: f.__name__)
def llm(request: pytest.FixtureRequest) -> LlmClient:
    factory: Callable[[], LlmClient] = request.param
    return factory()


@pytest.fixture(params=TTS_IMPLEMENTATIONS, ids=lambda f: f.__name__)
def tts(request: pytest.FixtureRequest) -> TtsClient:
    factory: Callable[[], TtsClient] = request.param
    return factory()


def test_llm_returns_text_usage_and_cost(llm: LlmClient) -> None:
    response = llm.complete(
        LlmRequest(task="script", system="Answer with JSON.", prompt="Topic: octopus")
    )

    assert response.text.strip()
    assert response.provider == llm.provider
    assert response.model == llm.model
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.cost_usd >= 0


def test_tts_returns_playable_audio(tts: TtsClient) -> None:
    result = tts.synthesize(TTS_REQUEST)

    assert result.audio
    assert result.format in {"mp3", "wav"}
    assert result.duration_s > 0
    assert result.characters == len(TTS_REQUEST.text)
    assert result.cost_usd >= 0
    if result.format == "wav":
        with wave.open(io.BytesIO(result.audio)) as wav:
            assert wav.getnframes() / wav.getframerate() == pytest.approx(result.duration_s)


def test_tts_rejects_empty_text(tts: TtsClient) -> None:
    empty = TtsRequest(text="  ", voice_id="v", stability=0.5, similarity_boost=0.5, speed=1.0)

    with pytest.raises(ValueError, match="empty"):
        tts.synthesize(empty)


# --- images ---------------------------------------------------------------------

IMAGE_IMPLEMENTATIONS: list[Callable[[], ImageClient]] = [FakeImages]

IMAGE_REQUEST = ImageRequest(prompt="a lighthouse at dusk", seed=7, width=64, height=112)

_SIGNATURES = {"png": b"\x89PNG", "jpg": b"\xff\xd8\xff", "webp": b"RIFF"}


@pytest.fixture(params=IMAGE_IMPLEMENTATIONS, ids=lambda f: f.__name__)
def images(request: pytest.FixtureRequest) -> ImageClient:
    factory: Callable[[], ImageClient] = request.param
    return factory()


def test_images_return_a_real_image_with_seed_and_licence(images: ImageClient) -> None:
    result = images.generate(IMAGE_REQUEST)

    assert result.image.startswith(_SIGNATURES[result.format])
    assert result.seed == IMAGE_REQUEST.seed
    assert result.width > 0
    assert result.height > 0
    assert result.licence.strip()
    assert result.cost_usd >= 0


# --- speech-to-text ---------------------------------------------------------------


def WhisperOnFakeModel() -> Transcriber:
    """The real adapter with the model replaced: CI has no faster-whisper and no network."""

    class Model:
        def transcribe(self, audio: Any, **_kwargs: Any) -> tuple[Iterable[Any], Any]:
            words = [
                SimpleNamespace(word=f" {w}", start=i * 0.4, end=i * 0.4 + 0.3, probability=0.9)
                for i, w in enumerate(["octopuses", "have", "three", "hearts"])
            ]
            return iter([SimpleNamespace(words=words)]), SimpleNamespace(duration=1.6)

    return WhisperTranscriber("small.en", load=lambda _name: Model(), decode=lambda _p: "samples")


TRANSCRIBER_IMPLEMENTATIONS: list[Callable[[], Transcriber]] = [FakeTranscriber, WhisperOnFakeModel]


@pytest.fixture(params=TRANSCRIBER_IMPLEMENTATIONS, ids=lambda f: f.__name__)
def stt(request: pytest.FixtureRequest) -> Transcriber:
    factory: Callable[[], Transcriber] = request.param
    return factory()


def test_transcriber_returns_ordered_words_inside_the_audio(
    stt: Transcriber, tmp_path: Path
) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(FakeTts().synthesize(TTS_REQUEST).audio)

    transcript = stt.transcribe(audio, language="en", hint=TTS_REQUEST.text)

    assert transcript.words
    assert all(w.word.strip() == w.word and w.word for w in transcript.words)
    assert all(0 <= w.start <= w.end <= transcript.duration_s + 0.25 for w in transcript.words)
    assert all(a.start <= b.start for a, b in itertools.pairwise(transcript.words))
    assert transcript.provider == stt.provider
    assert transcript.model == stt.model
    assert transcript.cost_usd >= 0
    assert 0 <= transcript.no_speech_prob <= 1


def test_transcriber_accepts_expected_words_for_a_clip(stt: Transcriber, tmp_path: Path) -> None:
    audio = tmp_path / "voice.wav"
    audio.write_bytes(FakeTts().synthesize(TTS_REQUEST).audio)

    transcript = stt.transcribe(audio, language="en", expect=TTS_REQUEST.text)

    assert 0 <= transcript.no_speech_prob <= 1
