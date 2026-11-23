"""Builds the adapter set from config. The only module that knows concrete adapters.

Steps receive an `Adapters` object and talk to ports, so swapping a provider
is a config change, not a code change in the steps. Each port is fake or real
on its own, so real providers are switched on one at a time.
"""

import os
import time
from dataclasses import dataclass, field

from pipeline.adapters.chaos import (
    ChaosConfig,
    ChaosImages,
    ChaosLlm,
    ChaosReviewer,
    ChaosTts,
    Sleep,
)
from pipeline.adapters.http import Transport, urllib_transport
from pipeline.adapters.images import ImageClient
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.images.fal import API_KEY_ENV as FAL_KEY_ENV
from pipeline.adapters.images.fal import FalImages
from pipeline.adapters.llm import LlmClient
from pipeline.adapters.llm.claude_cli import ClaudeCliLlm
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.stt import Transcriber
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.stt.whisper import WhisperTranscriber
from pipeline.adapters.tts import TtsClient
from pipeline.adapters.tts.elevenlabs import API_KEY_ENV, ElevenLabsTts
from pipeline.adapters.tts.fake import FakeTts
from pipeline.adapters.vision import FrameReviewer
from pipeline.adapters.vision.claude_cli import ClaudeCliReviewer
from pipeline.adapters.vision.fake import FakeReviewer
from pipeline.config import ImageChain, LlmChain, ModeConfig, ModelsConfig, TtsChain

ALL_FAKE = ModeConfig(llm="fake", tts="fake", images="fake", stt="fake", vision="fake")


@dataclass(frozen=True, slots=True)
class Adapters:
    llm_heavy: LlmClient  # script, critic
    llm_light: LlmClient  # titles, tags, stock queries
    tts: TtsClient
    images: ImageClient  # scene frames
    stt: Transcriber  # word timings for captions
    vision: FrameReviewer = field(default_factory=FakeReviewer)  # the frame review


def _real_llm(chain: LlmChain, models: ModelsConfig) -> LlmClient:
    if models.llm.api_fallback:
        raise NotImplementedError("the anthropic_api fallback adapter is not connected yet")
    if chain.primary.provider != "claude_cli":
        raise NotImplementedError(f"LLM provider {chain.primary.provider!r} is not connected yet")
    return ClaudeCliLlm(chain.primary.model, timeout_s=models.llm.timeout_s)


def _real_tts(chain: TtsChain) -> TtsClient:
    # The key is read here, at startup; a missing key fails on the first call, before paying.
    return ElevenLabsTts(
        chain.primary.model,
        api_key=os.environ.get(API_KEY_ENV),
        usd_per_credit=chain.usd_per_credit,
        timeout_s=chain.timeout_s,
    )


def fal_images(
    chain: ImageChain, model: str, *, transport: Transport = urllib_transport
) -> FalImages:
    """A fal client for any model in `fal_models`: the primary, or one under comparison."""
    profile = chain.fal_models[model]
    return FalImages(
        model,
        api_key=os.environ.get(FAL_KEY_ENV),
        licence=profile.licence,
        usd_per_megapixel=profile.usd_per_megapixel,
        usd_per_image=profile.usd_per_image,
        size=profile.size,
        params=profile.params,
        takes_seed=profile.takes_seed,
        timeout_s=chain.timeout_s,
        transport=transport,
    )


def _real_vision(models: ModelsConfig) -> FrameReviewer:
    """The frame review runs on the heavy model, through the same CLI as the script."""
    chain = models.llm.heavy
    if chain.primary.provider != "claude_cli":
        raise NotImplementedError("the frame review runs through claude_cli only")
    return ClaudeCliReviewer(chain.primary.model, timeout_s=models.llm.timeout_s)


def _real_images(chain: ImageChain) -> ImageClient:
    return fal_images(chain, chain.primary.model)


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
    if mode.images == "real" and models.images.primary.provider != "fal":
        raise NotImplementedError("only fal is connected for images")

    heavy: LlmClient
    light: LlmClient
    if mode.llm == "real":
        heavy, light = _real_llm(models.llm.heavy, models), _real_llm(models.llm.light, models)
    else:
        heavy = light = FakeLlm()
    tts: TtsClient = _real_tts(models.tts) if mode.tts == "real" else FakeTts()
    images: ImageClient = _real_images(models.images) if mode.images == "real" else FakeImages()
    stt: Transcriber = (
        WhisperTranscriber(models.stt.primary.model) if mode.stt == "real" else FakeTranscriber()
    )
    vision: FrameReviewer = _real_vision(models) if mode.vision == "real" else FakeReviewer()

    if chaos is not None:  # chaos only ever wraps fakes
        if mode.llm == "fake":
            heavy = light = ChaosLlm(heavy, chaos, sleep)
        if mode.tts == "fake":
            tts = ChaosTts(tts, chaos, sleep)
        if mode.images == "fake":
            images = ChaosImages(images, chaos, sleep)
        if mode.vision == "fake":
            vision = ChaosReviewer(vision, chaos, sleep)
    return Adapters(
        llm_heavy=heavy, llm_light=light, tts=tts, images=images, stt=stt, vision=vision
    )
