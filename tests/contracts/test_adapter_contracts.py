"""Contract tests: every implementation of a port must pass the same checks.

Today only the fakes are listed. Real adapters join these lists when they land
(marked to run only with API keys), so fake and real can never drift apart.
"""

import io
import wave
from collections.abc import Callable

import pytest

from pipeline.adapters.llm import LlmClient, LlmRequest
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.tts import TtsClient, TtsRequest
from pipeline.adapters.tts.fake import FakeTts

LLM_IMPLEMENTATIONS: list[Callable[[], LlmClient]] = [FakeLlm]
TTS_IMPLEMENTATIONS: list[Callable[[], TtsClient]] = [FakeTts]

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
