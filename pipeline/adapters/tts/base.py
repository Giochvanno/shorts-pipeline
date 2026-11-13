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

    def edited(self, *, trim_start_s: float = 0.0, tempo: float = 1.0) -> "Alignment":
        """The same timings after an edit of the audio: the first `trim_start_s` seconds
        cut off, then played `tempo` times faster. Any edit of the voice audio must pass
        its timings through here, or captions drift by exactly that edit."""
        if trim_start_s < 0 or tempo <= 0:
            raise ValueError("trim_start_s must be >= 0 and tempo > 0")

        def move(t: float) -> float:
            return max(0.0, (t - trim_start_s) / tempo)

        return Alignment(
            characters=self.characters,
            starts_s=tuple(move(t) for t in self.starts_s),
            ends_s=tuple(move(t) for t in self.ends_s),
        )


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
