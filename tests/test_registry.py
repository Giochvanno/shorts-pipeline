"""Adapter registry: fakes on demand, real adapters not wired yet."""

import pytest

from pipeline.adapters.llm.claude_cli import ClaudeCliLlm
from pipeline.adapters.registry import ALL_FAKE, build_adapters
from pipeline.config import load_channel


def test_fake_adapters_are_offline_and_free() -> None:
    adapters = build_adapters(load_channel().models, fake=True)

    assert {
        adapters.llm_heavy.provider,
        adapters.llm_light.provider,
        adapters.tts.provider,
        adapters.images.provider,
    } == {"fake"}


@pytest.mark.parametrize("port", ["tts", "images"])
def test_ports_without_a_real_adapter_are_refused(port: str) -> None:
    mode = ALL_FAKE.model_copy(update={port: "real"})

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
