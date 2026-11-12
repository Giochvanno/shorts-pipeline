"""Builds the adapter set from config. The only module that knows concrete adapters.

Steps receive an `Adapters` object and talk to ports, so swapping a provider
is a config change, not a code change in the steps. Each port is fake or real
on its own, so real providers are switched on one at a time.
"""

import time
from dataclasses import dataclass

from pipeline.adapters.chaos import ChaosConfig, ChaosImages, ChaosLlm, ChaosTts, Sleep
from pipeline.adapters.images import ImageClient
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm import LlmClient
from pipeline.adapters.llm.claude_cli import ClaudeCliLlm
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.stt import Transcriber
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.stt.whisper import WhisperTranscriber
from pipeline.adapters.tts import TtsClient
from pipeline.adapters.tts.fake import FakeTts
from pipeline.config import LlmChain, ModeConfig, ModelsConfig

ALL_FAKE = ModeConfig(llm="fake", tts="fake", images="fake", stt="fake")


@dataclass(frozen=True, slots=True)
class Adapters:
    llm_heavy: LlmClient  # script, critic, frame check
    llm_light: LlmClient  # titles, tags, stock queries
    tts: TtsClient
    images: ImageClient  # FLUX for scenes
    stt: Transcriber  # word timings for captions


def _real_llm(chain: LlmChain, models: ModelsConfig) -> LlmClient:
    if models.llm.api_fallback:
        raise NotImplementedError("the anthropic_api fallback adapter is not connected yet")
    if chain.primary.provider != "claude_cli":
        raise NotImplementedError(f"LLM provider {chain.primary.provider!r} is not connected yet")
    return ClaudeCliLlm(chain.primary.model, timeout_s=models.llm.timeout_s)


def build_adapters(
    models: ModelsConfig,
    *,
    fake: bool = False,
    mode: ModeConfig = ALL_FAKE,
    chaos: ChaosConfig | None = None,
    sleep: Sleep = time.sleep,
) -> Adapters:
    """`fake=True` forces every port to its fake (tests, demos, `--mode fake`)."""
    mode = ALL_FAKE if fake else mode
    if mode.tts == "real":
        raise NotImplementedError("the real TTS adapter (ElevenLabs) is not connected yet")
    if mode.images == "real":
        raise NotImplementedError("the real images adapter (FLUX) is not connected yet")

    heavy: LlmClient
    light: LlmClient
    if mode.llm == "real":
        heavy, light = _real_llm(models.llm.heavy, models), _real_llm(models.llm.light, models)
    else:
        heavy = light = FakeLlm()
    tts: TtsClient = FakeTts()
    images: ImageClient = FakeImages()
    stt: Transcriber = (
        WhisperTranscriber(models.stt.primary.model) if mode.stt == "real" else FakeTranscriber()
    )

    if chaos is not None:  # chaos only ever wraps fakes
        if mode.llm == "fake":
            heavy = light = ChaosLlm(heavy, chaos, sleep)
        tts = ChaosTts(tts, chaos, sleep)
        images = ChaosImages(images, chaos, sleep)
    return Adapters(llm_heavy=heavy, llm_light=light, tts=tts, images=images, stt=stt)
