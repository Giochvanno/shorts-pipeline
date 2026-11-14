"""Speech-to-text port: audio in, words with timings out (used for captions)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WordTiming:
    word: str
    start: float  # seconds from the start of the audio
    end: float
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class Transcript:
    words: tuple[WordTiming, ...]
    duration_s: float
    provider: str
    model: str
    cost_usd: float = 0.0
    # How sure the recognizer is that there is NO speech (0..1, the worst segment).
    # 1.0 when it found nothing at all.
    no_speech_prob: float = 0.0


class Transcriber(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def transcribe(
        self, audio: Path, *, language: str, hint: str | None = None, expect: str | None = None
    ) -> Transcript:
        """`hint`: the script; the recognizer takes rare words from it as vocabulary.
        `expect`: the exact words a short clip should hold, given to the recognizer
        as they are. It can make the recognizer "hear" them in silence, so callers
        must not trust a result obtained with `expect` on its own."""
        ...
