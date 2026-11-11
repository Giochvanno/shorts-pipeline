"""Builds the adapter set from config. The only module that knows concrete adapters.

Steps receive an `Adapters` object and talk to ports, so swapping a provider
is a config change, not a code change in the steps.
"""

import time
from dataclasses import dataclass

from pipeline.adapters.chaos import ChaosConfig, ChaosImages, ChaosLlm, ChaosTts, Sleep
from pipeline.adapters.images import ImageClient
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm import LlmClient
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.tts import TtsClient
from pipeline.adapters.tts.fake import FakeTts
from pipeline.config import ModelsConfig


@dataclass(frozen=True, slots=True)
class Adapters:
    llm_heavy: LlmClient  # script, critic, frame check
    llm_light: LlmClient  # titles, tags, stock queries
    tts: TtsClient
    images: ImageClient  # FLUX for scenes


def build_adapters(
    models: ModelsConfig,
    *,
    fake: bool,
    chaos: ChaosConfig | None = None,
    sleep: Sleep = time.sleep,
) -> Adapters:
    if not fake:
        # Real Claude, ElevenLabs and FLUX adapters are connected in stage 2.
        raise NotImplementedError("real adapters are not connected yet; use fake mode")
    del models  # fakes do not need model IDs; real adapters will read them from here
    llm: LlmClient = FakeLlm()
    tts: TtsClient = FakeTts()
    images: ImageClient = FakeImages()
    if chaos is not None:
        llm = ChaosLlm(llm, chaos, sleep)
        tts = ChaosTts(tts, chaos, sleep)
        images = ChaosImages(images, chaos, sleep)
    return Adapters(llm_heavy=llm, llm_light=llm, tts=tts, images=images)
