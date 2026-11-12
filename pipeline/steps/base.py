"""What every step receives and returns."""

from dataclasses import dataclass
from pathlib import Path

from pipeline.adapters.images import ImageClient
from pipeline.adapters.llm import LlmClient
from pipeline.adapters.stt import Transcriber
from pipeline.adapters.tts import TtsClient
from pipeline.config import SeriesConfig


class StepError(Exception):
    """A step failed. `retryable` tells the runner whether trying again can help."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class StepContext:
    video_id: int
    topic: str
    series: SeriesConfig
    run_dir: Path  # runs/<video_id>/
    llm_heavy: LlmClient
    llm_light: LlmClient
    tts: TtsClient
    images: ImageClient
    stt: Transcriber
    language: str = "en"  # of scripts, voice and captions (channel.yaml)
    # Bumped by "redo script" / "redo visuals": a new revision means a new cache key.
    revision: int = 0


@dataclass(frozen=True, slots=True)
class StepResult:
    outputs: tuple[Path, ...]
    cost_usd: float
    cached: bool  # True = nothing was called, nothing was paid
