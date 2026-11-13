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
class Alignment:
    """When each character of the text is spoken, from providers that report it.

    Free with the audio (ElevenLabs `with-timestamps`); getting it later would
    mean paying for the voice again, so the voice step keeps it.
    """

    characters: tuple[str, ...]
    starts_s: tuple[float, ...]
    ends_s: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TtsResult:
    audio: bytes
    format: AudioFormat
    duration_s: float
    characters: int  # ElevenLabs bills by characters
    provider: str
    model: str
    cost_usd: float
    alignment: Alignment | None = None  # None: the provider gives no timings (fake, ...)


class TtsClient(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def synthesize(self, request: TtsRequest) -> TtsResult: ...
