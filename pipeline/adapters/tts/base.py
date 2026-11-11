"""TTS port: what every text-to-speech adapter (fake, ElevenLabs, ...) must provide."""

from dataclasses import dataclass
from typing import Literal, Protocol

AudioFormat = Literal["mp3", "wav"]


@dataclass(frozen=True, slots=True)
class TtsRequest:
    text: str
    voice_id: str
    stability: float
    similarity_boost: float
    speed: float


@dataclass(frozen=True, slots=True)
class TtsResult:
    audio: bytes
    format: AudioFormat
    duration_s: float
    characters: int  # ElevenLabs bills by characters
    provider: str
    model: str
    cost_usd: float


class TtsClient(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def synthesize(self, request: TtsRequest) -> TtsResult: ...
