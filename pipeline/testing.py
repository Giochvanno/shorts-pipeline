"""Helpers for tests, demos and soak runs. Fakes only: nothing here spends money."""

from pathlib import Path

from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.tts.fake import FakeTts
from pipeline.config import load_series
from pipeline.steps.base import StepContext

DEMO_TOPIC = "why octopuses have three hearts"


def fake_context(
    run_dir: Path, *, video_id: int = 1, topic: str = DEMO_TOPIC, series_id: str = "fact-of-day"
) -> StepContext:
    """A step context wired to fresh fakes and the real series config."""
    return StepContext(
        video_id=video_id,
        topic=topic,
        series=load_series(series_id),
        run_dir=run_dir,
        llm_heavy=FakeLlm(),
        llm_light=FakeLlm(),
        tts=FakeTts(),
        images=FakeImages(),
    )
