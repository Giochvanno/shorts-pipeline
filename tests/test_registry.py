"""Adapter registry: fakes on demand, real adapters not wired yet."""

import pytest

from pipeline.adapters.registry import build_adapters
from pipeline.config import load_channel


def test_fake_adapters_are_offline_and_free() -> None:
    adapters = build_adapters(load_channel().models, fake=True)

    assert {
        adapters.llm_heavy.provider,
        adapters.llm_light.provider,
        adapters.tts.provider,
        adapters.images.provider,
    } == {"fake"}


def test_real_adapters_are_not_connected_yet() -> None:
    with pytest.raises(NotImplementedError):
        build_adapters(load_channel().models, fake=False)
