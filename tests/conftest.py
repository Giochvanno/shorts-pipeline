"""Shared fixtures: a step context wired to fakes and the real series config."""

import os
import shutil
from pathlib import Path

import pytest

from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.tts.fake import FakeTts
from pipeline.config import SeriesConfig, load_series
from pipeline.steps.base import StepContext


@pytest.fixture
def series() -> SeriesConfig:
    return load_series("fact-of-day")


@pytest.fixture
def fake_llm() -> FakeLlm:
    return FakeLlm()


@pytest.fixture
def fake_tts() -> FakeTts:
    return FakeTts()


@pytest.fixture
def fake_images() -> FakeImages:
    return FakeImages()


@pytest.fixture
def ctx(
    tmp_path: Path,
    series: SeriesConfig,
    fake_llm: FakeLlm,
    fake_tts: FakeTts,
    fake_images: FakeImages,
) -> StepContext:
    return StepContext(
        video_id=1,
        topic="why octopuses have three hearts",
        series=series,
        run_dir=tmp_path / "runs" / "1",
        llm_heavy=fake_llm,
        llm_light=fake_llm,
        tts=fake_tts,
        images=fake_images,
    )


@pytest.fixture
def ffmpeg() -> None:
    """Skip locally if FFmpeg is missing; in CI (REQUIRE_FFMPEG=1) fail instead of skipping."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if os.environ.get("REQUIRE_FFMPEG") == "1":
        pytest.fail("FFmpeg is required here but not found in PATH")
    pytest.skip("FFmpeg not installed")
