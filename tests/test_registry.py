"""Adapter registry: fakes on demand, real adapters per port."""

import pytest

from pipeline.adapters.chaos import ChaosConfig, ChaosImages, ChaosTts
from pipeline.adapters.errors import PermanentAdapterError
from pipeline.adapters.llm.claude_cli import ClaudeCliLlm
from pipeline.adapters.registry import ALL_FAKE, build_adapters
from pipeline.adapters.stt.whisper import WhisperTranscriber
from pipeline.adapters.tts import TtsRequest
from pipeline.adapters.tts.elevenlabs import API_KEY_ENV, ElevenLabsTts
from pipeline.config import load_channel


def test_fake_adapters_are_offline_and_free() -> None:
    adapters = build_adapters(load_channel().models, fake=True)

    assert {
        adapters.llm_heavy.provider,
        adapters.llm_light.provider,
        adapters.tts.provider,
        adapters.images.provider,
    } == {"fake"}


def test_ports_without_a_real_adapter_are_refused() -> None:
    mode = ALL_FAKE.model_copy(update={"images": "real"})

    with pytest.raises(NotImplementedError, match="not connected yet"):
        build_adapters(load_channel().models, mode=mode)


def test_real_llm_uses_claude_cli_with_models_from_config() -> None:
    models = load_channel().models
    adapters = build_adapters(models, mode=ALL_FAKE.model_copy(update={"llm": "real"}))

    assert isinstance(adapters.llm_heavy, ClaudeCliLlm)
    assert adapters.llm_heavy.model == models.llm.heavy.primary.model
    assert adapters.llm_light.model == models.llm.light.primary.model
    assert adapters.tts.provider == "fake"  # other ports stay fake


def test_api_fallback_is_refused_until_its_adapter_exists() -> None:
    models = load_channel().models
    with_fallback = models.model_copy(
        update={"llm": models.llm.model_copy(update={"api_fallback": True})}
    )

    with pytest.raises(NotImplementedError, match="anthropic_api"):
        build_adapters(with_fallback, mode=ALL_FAKE.model_copy(update={"llm": "real"}))


def test_real_stt_is_local_whisper_and_loads_nothing_up_front() -> None:
    models = load_channel().models
    adapters = build_adapters(models, mode=ALL_FAKE.model_copy(update={"stt": "real"}))

    assert isinstance(adapters.stt, WhisperTranscriber)
    assert adapters.stt.model == models.stt.primary.model  # model loads on first use only


def test_real_tts_is_elevenlabs_with_model_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    models = load_channel().models
    adapters = build_adapters(models, mode=ALL_FAKE.model_copy(update={"tts": "real"}))

    assert isinstance(adapters.tts, ElevenLabsTts)
    assert adapters.tts.model == models.tts.primary.model
    assert "test-key" not in repr(adapters.tts)  # the key never shows up in logs
    assert adapters.llm_heavy.provider == "fake"


def test_real_tts_without_a_key_fails_on_first_call_not_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    adapters = build_adapters(
        load_channel().models, mode=ALL_FAKE.model_copy(update={"tts": "real"})
    )
    request = TtsRequest(text="Hi.", voice_id="v", stability=0.5, similarity_boost=0.75, speed=1)

    with pytest.raises(PermanentAdapterError, match=API_KEY_ENV):
        adapters.tts.synthesize(request)


def test_chaos_never_wraps_a_real_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    chaos = ChaosConfig(fail_rate=0.5, garbage_rate=0.0, max_latency_s=0.0, seed=1)
    adapters = build_adapters(
        load_channel().models, mode=ALL_FAKE.model_copy(update={"tts": "real"}), chaos=chaos
    )

    assert not isinstance(adapters.tts, ChaosTts)
    assert isinstance(adapters.images, ChaosImages)
