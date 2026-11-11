"""Builds the adapter set from config. The only module that knows concrete adapters.

Steps receive an `Adapters` object and talk to ports, so swapping a provider
is a config change, not a code change in the steps.
"""

from dataclasses import dataclass

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


def build_adapters(models: ModelsConfig, *, fake: bool) -> Adapters:
    if not fake:
        # Real Claude and ElevenLabs adapters are connected after stage 1.
        raise NotImplementedError("real adapters are not connected yet; use fake=True")
    del models  # fakes do not need model IDs; real adapters will read them from here
    return Adapters(llm_heavy=FakeLlm(), llm_light=FakeLlm(), tts=FakeTts())
