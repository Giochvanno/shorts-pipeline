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


class Transcriber(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def transcribe(self, audio: Path, *, language: str, hint: str | None = None) -> Transcript:
        """`hint` is the text we expect to hear: it helps with names and rare words."""
        ...
