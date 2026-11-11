"""Step `visuals` v0 on fake images: one frame per scene, one seed, cache, licence."""

import json
import re
import struct
from dataclasses import replace

import pytest

from pipeline.adapters.images.fake import FakeImages
from pipeline.artifacts import read_meta
from pipeline.config import SeriesConfig
from pipeline.steps import script, visuals
from pipeline.steps.base import StepContext, StepError


def _png_size(data: bytes) -> tuple[int, int]:
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


@pytest.fixture
def scripted(ctx: StepContext) -> StepContext:
    script.run(ctx)
    return ctx


def test_one_frame_per_scene(scripted: StepContext) -> None:
    result = visuals.run(scripted)

    scenes = script.load(scripted.run_dir).scenes
    assert len(result.outputs) == len(scenes)
    for frame in result.outputs:
        assert _png_size(frame.read_bytes()) == (visuals.WIDTH, visuals.HEIGHT)
    assert visuals.frame_paths(scripted.run_dir, len(scenes)) == list(result.outputs)


def test_all_scenes_share_one_seed_per_video(
    scripted: StepContext, fake_images: FakeImages
) -> None:
    visuals.run(scripted)

    assert len({call.seed for call in fake_images.calls}) == 1
    assert visuals.video_seed(1, 0) != visuals.video_seed(2, 0)
    assert visuals.video_seed(1, 0) != visuals.video_seed(1, 1)  # "redo visuals" = new seed


def test_prompt_is_bible_plus_scene(series: SeriesConfig) -> None:
    prompt = visuals.build_image_prompt(series.visual_bible, "a lighthouse on a cliff")

    assert prompt.startswith(series.visual_bible.style_prefix.strip().rstrip("."))
    assert "a lighthouse on a cliff" in prompt
    assert series.visual_bible.palette[0] in prompt
    # Forbidden words stay out of the prompt: naming them invites the model to draw them.
    for word in ("text", "letters", "hands", "fingers"):
        assert not re.search(rf"\b{word}\b", prompt, re.IGNORECASE), word


def test_meta_records_source_licence_and_prompt(scripted: StepContext) -> None:
    (first, *_) = visuals.run(scripted).outputs
    meta = read_meta(first)

    assert meta is not None
    assert meta["source"] == "generated"
    assert meta["licence"]
    assert meta["seed"] == visuals.video_seed(scripted.video_id, scripted.revision)
    assert "prompt" in meta


def test_second_run_is_cached_and_free(scripted: StepContext, fake_images: FakeImages) -> None:
    visuals.run(scripted)
    calls = len(fake_images.calls)
    again = visuals.run(scripted)

    assert again.cached
    assert len(fake_images.calls) == calls


def test_only_changed_scenes_are_regenerated(
    scripted: StepContext, fake_images: FakeImages
) -> None:
    visuals.run(scripted)
    calls = len(fake_images.calls)
    path = scripted.run_dir / "script.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["scenes"][2]["visual"] = "a red kite over green hills"
    path.write_text(json.dumps(data), encoding="utf-8")

    again = visuals.run(scripted)

    assert not again.cached
    assert len(fake_images.calls) == calls + 1  # one scene changed -> one paid call


def test_style_change_regenerates_every_frame(
    scripted: StepContext, fake_images: FakeImages
) -> None:
    visuals.run(scripted)
    calls = len(fake_images.calls)
    bible = scripted.series.visual_bible.model_copy(update={"lighting": "golden hour"})
    series = scripted.series.model_copy(update={"visual_bible": bible})

    visuals.run(replace(scripted, series=series))

    assert len(fake_images.calls) == 2 * calls


def test_needs_a_script_first(ctx: StepContext) -> None:
    with pytest.raises(StepError, match="script step first"):
        visuals.run(ctx)


def test_render_gets_a_clear_error_for_missing_frames(scripted: StepContext) -> None:
    with pytest.raises(StepError, match="visuals step first") as exc:
        visuals.frame_paths(scripted.run_dir, 3)

    assert not exc.value.retryable
