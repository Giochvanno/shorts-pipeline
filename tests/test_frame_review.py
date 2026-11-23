"""The frame review inside the visuals step: what is reviewed, kept, set aside and redrawn.

The reviewer is the fake one, answering what each test needs. What a real reviewer
answers is covered by test_vision_claude (on a fake process) and the live test.
"""

import json
import os
import random
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pipeline.adapters.chaos import ChaosConfig, ChaosReviewer
from pipeline.adapters.errors import TransientAdapterError
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.registry import Adapters
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.tts.fake import FakeTts
from pipeline.adapters.vision import FrameCheckRequest, FrameVerdict
from pipeline.adapters.vision.fake import FakeReviewer, clean_verdict
from pipeline.artifacts import read_meta, write_meta
from pipeline.config import BudgetConfig
from pipeline.costs import Ledger, MeteredReviewer
from pipeline.runner import DEFAULT_STEPS, RetryPolicy, Runner, StepFn
from pipeline.state import StateStore, StepStatus
from pipeline.steps import script, visuals, voice
from pipeline.steps.base import StepContext, StepError, StepResult

TEXT = {"has_text": True, "text_seen": "scribbles on the wall, top right"}
WRONG = {"right_subject": False, "subject_seen": "a hamster"}


def turned_down(request: FrameCheckRequest, **changes: Any) -> FrameVerdict:
    return replace(clean_verdict(request), **changes)


class Picky(FakeReviewer):
    """Turns down the first `times` frames of the scene that mentions `needle`."""

    def __init__(self, needle: str, times: int, **changes: Any) -> None:
        self.rejected = 0

        def answer(request: FrameCheckRequest) -> FrameVerdict:
            if needle in request.scene and self.rejected < times:
                self.rejected += 1
                return turned_down(request, **changes)
            return clean_verdict(request)

        super().__init__(answer)


class Down(FakeReviewer):
    """The review fails (a CLI timeout) for the first `times` calls."""

    def __init__(self, times: int) -> None:
        super().__init__()
        self.failures_left = times

    def review(self, request: FrameCheckRequest) -> FrameVerdict:
        if self.failures_left > 0:
            self.failures_left -= 1
            self.calls.append(request)
            raise TransientAdapterError("claude -p timed out")
        return super().review(request)


@pytest.fixture
def scripted(ctx: StepContext) -> StepContext:
    script.run(ctx)
    return ctx


def _with(
    ctx: StepContext, reviewer: FakeReviewer, images: FakeImages | None = None
) -> StepContext:
    return replace(ctx, vision=reviewer, images=images or ctx.images)


def _first_scene(ctx: StepContext) -> str:
    """Scene 1's visual with its full stop: the fake's scene 6 starts with the same words."""
    return script.load(ctx.run_dir).scenes[0].visual.rstrip(".") + "."


def _scene_seeds(images: FakeImages, needle: str) -> list[int | None]:
    return [call.seed for call in images.calls if needle in call.prompt]


def _frame(ctx: StepContext) -> Path:
    return ctx.run_dir / visuals.FRAMES_DIR / "scene_00.png"


# --- every new frame is reviewed --------------------------------------------------------


def test_every_new_frame_is_reviewed_with_its_scene_light_and_style(
    scripted: StepContext,
) -> None:
    reviewer = FakeReviewer()
    result = visuals.run(_with(scripted, reviewer))

    scenes = script.load(scripted.run_dir).scenes
    assert len(reviewer.calls) == len(scenes)
    for scene, request, frame in zip(scenes, reviewer.calls, result.outputs, strict=True):
        assert request.scene.startswith(scene.visual.rstrip("."))
        assert f"Light: {scene.lighting}" in request.scene
        assert request.style == scripted.series.visual_bible.style_prefix.strip()
        assert request.media_type == "image/png"
        assert request.image == frame.read_bytes()
    assert result.notes == ()


def test_the_verdict_is_kept_in_the_frame_meta(scripted: StepContext) -> None:
    (first, *_) = visuals.run(_with(scripted, FakeReviewer())).outputs

    meta = read_meta(first)
    assert meta is not None
    assert meta["review"]["accepted"] is True
    assert meta["review"]["problems"] == []
    assert (meta["review"]["provider"], meta["review"]["model"]) == ("fake", "fake-vision")
    assert meta["review"]["subject_position"] == "middle"  # informational, never a rejection


def test_reviewed_frames_are_not_reviewed_again(scripted: StepContext) -> None:
    reviewer = FakeReviewer()
    visuals.run(_with(scripted, reviewer))
    reviews = len(reviewer.calls)

    again = visuals.run(_with(scripted, reviewer))

    assert again.cached
    assert len(reviewer.calls) == reviews


def test_frames_made_before_the_review_existed_stay_as_they_are(scripted: StepContext) -> None:
    """Videos up to 10: no review in the meta, and nobody pays to redo them."""
    (first, *_) = visuals.run(scripted).outputs  # no reviewer, like before
    meta = read_meta(first)
    assert meta is not None
    old = {k: v for k, v in meta.items() if k not in {"input_hash", "review"}}
    write_meta(first, meta["input_hash"], old)
    reviewer = FakeReviewer()

    assert visuals.run(_with(scripted, reviewer)).cached
    assert reviewer.calls == []


def test_without_a_reviewer_the_frame_is_kept_unreviewed(scripted: StepContext) -> None:
    (first, *_) = visuals.run(scripted).outputs

    meta = read_meta(first)
    assert meta is not None and meta["review"] is None


# --- a frame turned down ----------------------------------------------------------------


def test_a_turned_down_frame_is_set_aside_and_the_scene_drawn_with_the_next_seed(
    scripted: StepContext, fake_images: FakeImages
) -> None:
    needle = _first_scene(scripted)
    reviewer = Picky(needle, times=1, **TEXT)

    result = visuals.run(_with(scripted, reviewer))

    base = visuals.video_seed(scripted.video_id, scripted.revision)
    assert _scene_seeds(fake_images, needle) == [base, base + 1]
    kept = read_meta(_frame(scripted))
    assert kept is not None and (kept["attempt"], kept["seed"]) == (2, base + 1)
    rejected = scripted.run_dir / visuals.REJECTED_FRAMES
    assert sorted(p.name for p in rejected.iterdir()) == ["scene_00_try1.json", "scene_00_try1.png"]
    report = json.loads((rejected / "scene_00_try1.json").read_text(encoding="utf-8"))
    assert report["seed"] == base
    assert report["review"]["problems"] == ["text in the frame: scribbles on the wall, top right"]
    assert result.notes == (
        "scene 1: the review turned down 1 frame(s), try 2 kept"
        " (text in the frame: scribbles on the wall, top right)",
    )


def test_the_paid_rejected_frame_counts_in_the_step_cost(scripted: StepContext) -> None:
    class Priced(FakeImages):
        def generate(self, request: Any) -> Any:
            return replace(super().generate(request), cost_usd=0.05)

    scenes = len(script.load(scripted.run_dir).scenes)
    reviewer = Picky(_first_scene(scripted), times=1, **WRONG)

    result = visuals.run(_with(scripted, reviewer, Priced()))

    assert result.cost_usd == pytest.approx(0.05 * (scenes + 1))


def test_when_every_try_is_turned_down_the_video_stops_for_a_person(
    scripted: StepContext, fake_images: FakeImages
) -> None:
    needle = _first_scene(scripted)
    tries = scripted.series.visual_bible.max_attempts_per_scene

    with pytest.raises(StepError, match="rejected_frames/") as exc:
        visuals.run(_with(scripted, Picky(needle, times=99, **WRONG)))

    assert not exc.value.retryable  # the same seeds would draw and charge the same frames
    assert f"turned down {tries} of {tries}" in str(exc.value)
    assert len(_scene_seeds(fake_images, needle)) == tries
    rejected = scripted.run_dir / visuals.REJECTED_FRAMES
    assert len(list(rejected.glob("scene_00_try*.png"))) == tries
    assert not list((scripted.run_dir / visuals.FRAMES_DIR).glob("scene_00.*"))


# --- a paid frame is never paid twice ---------------------------------------------------


def test_a_review_that_fails_once_is_asked_again(
    scripted: StepContext, fake_images: FakeImages
) -> None:
    scenes = len(script.load(scripted.run_dir).scenes)

    visuals.run(_with(scripted, Down(times=1)))

    assert len(fake_images.calls) == scenes


def test_a_failed_review_keeps_the_paid_frame_for_the_next_run(
    scripted: StepContext, fake_images: FakeImages
) -> None:
    with pytest.raises(StepError, match="the frame review failed") as exc:
        visuals.run(_with(scripted, Down(times=visuals.REVIEW_TRIES)))

    assert exc.value.retryable
    meta = read_meta(_frame(scripted))
    assert meta is not None and meta["review"] == visuals.PENDING
    assert len(fake_images.calls) == 1

    reviewer = FakeReviewer()
    visuals.run(_with(scripted, reviewer))

    scenes = len(script.load(scripted.run_dir).scenes)
    assert len(fake_images.calls) == scenes  # scene 1 was reviewed, not drawn again
    assert len(reviewer.calls) == scenes
    meta = read_meta(_frame(scripted))
    assert meta is not None and meta["review"]["accepted"] is True


def test_a_pending_frame_turned_down_later_is_redrawn_with_the_next_seed(
    scripted: StepContext, fake_images: FakeImages
) -> None:
    needle = _first_scene(scripted)
    with pytest.raises(StepError):
        visuals.run(_with(scripted, Down(times=visuals.REVIEW_TRIES)))

    result = visuals.run(_with(scripted, Picky(needle, times=1, **TEXT)))

    base = visuals.video_seed(scripted.video_id, scripted.revision)
    assert _scene_seeds(fake_images, needle) == [base, base + 1]
    assert (scripted.run_dir / visuals.REJECTED_FRAMES / "scene_00_try1.png").exists()
    assert result.notes[0].startswith("scene 1: the review turned down 1 frame(s), try 2 kept")


# --- the folder of rejected frames ------------------------------------------------------


def _group(folder: Path, stem: str, size: int, age: int) -> None:
    for suffix, data in ((".png", b"x" * size), (".json", b"{}")):
        path = folder / f"{stem}{suffix}"
        path.write_bytes(data)
        os.utime(path, ns=(age, age))


def test_rejected_frames_are_capped_oldest_first(tmp_path: Path) -> None:
    for age, stem in enumerate(("scene_00_try1", "scene_00_try2", "scene_03_try1"), start=1):
        _group(tmp_path, stem, size=1000, age=age * 10**9)

    visuals._cap(tmp_path, max_bytes=2100)

    assert sorted(p.stem for p in tmp_path.iterdir()) == [
        "scene_00_try2",
        "scene_00_try2",
        "scene_03_try1",
        "scene_03_try1",
    ]


def test_the_newest_rejected_frame_stays_even_over_the_cap(tmp_path: Path) -> None:
    _group(tmp_path, "scene_00_try1", size=1000, age=10**9)
    _group(tmp_path, "scene_00_try2", size=5000, age=2 * 10**9)

    visuals._cap(tmp_path, max_bytes=100)

    assert {p.stem for p in tmp_path.iterdir()} == {"scene_00_try2"}


def test_a_zero_cap_keeps_only_the_last_rejected_frame(scripted: StepContext) -> None:
    qa = scripted.series.qa.model_copy(update={"rejected_frames_max_mb": 0})
    series = scripted.series.model_copy(update={"qa": qa})
    tries = series.visual_bible.max_attempts_per_scene

    with pytest.raises(StepError):
        visuals.run(
            _with(replace(scripted, series=series), Picky(_first_scene(scripted), 99, **WRONG))
        )

    rejected = scripted.run_dir / visuals.REJECTED_FRAMES
    assert {p.name for p in rejected.iterdir()} == {
        f"scene_00_try{tries}.png",
        f"scene_00_try{tries}.json",
    }


# --- with the runner: budget pause and chaos --------------------------------------------


def stub_render(ctx: StepContext) -> StepResult:
    frames = visuals.frame_paths(ctx.run_dir, len(script.load(ctx.run_dir).scenes))
    voice.voice_file(ctx.run_dir)
    return StepResult(outputs=tuple(frames), cost_usd=0.0, cached=False)


STEPS: list[tuple[str, StepFn]] = [*DEFAULT_STEPS[:-1], ("render", stub_render)]


class CreditReviewer(FakeReviewer):
    """Bills like `claude -p`: subscription credit."""

    @property
    def provider(self) -> str:
        return "claude_cli"

    def review(self, request: FrameCheckRequest) -> FrameVerdict:
        verdict = super().review(request)
        return replace(verdict, provider="claude_cli", model="test-model", cost_usd=0.10)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[StateStore]:
    with StateStore(tmp_path / "pipeline.db") as s:
        yield s


def _runner(store: StateStore, tmp_path: Path, adapters: Adapters, credit: float = 10.0) -> Runner:
    budget = BudgetConfig(
        max_cash_per_video_usd=10,
        max_image_cash_per_video_usd=10,
        max_cash_monthly_usd=10,
        max_subscription_credit_monthly_usd=credit,
        subscription_credit_warn_at=0.8,
    )
    return Runner(
        store=store,
        adapters=adapters,
        runs_dir=tmp_path / "runs",
        policy=RetryPolicy(max_attempts=3, base_delay_s=0, jitter=0),
        steps=STEPS,
        budget=budget,
        sleep=lambda _s: None,
        rng=random.Random(0),
    )


def _adapters(images: FakeImages, reviewer: FakeReviewer | ChaosReviewer) -> Adapters:
    llm = FakeLlm()
    return Adapters(
        llm_heavy=llm,
        llm_light=llm,
        tts=FakeTts(),
        images=images,
        stt=FakeTranscriber(),
        vision=reviewer,
    )


def test_a_pause_at_the_review_keeps_the_paid_frame(store: StateStore, tmp_path: Path) -> None:
    images, reviewer = FakeImages(), CreditReviewer()
    adapters = _adapters(images, reviewer)
    first = _runner(store, tmp_path, adapters, credit=0.25)  # the 4th review is over the limit
    video = first.create_video("fact-of-day", "why octopuses have three hearts")

    report = first.run_video(video)

    assert report.status == StepStatus.PAUSED_BUDGET
    assert report.paused_step is not None and report.paused_step.step == "visuals"
    assert (len(images.calls), len(reviewer.calls)) == (4, 3)

    report = _runner(store, tmp_path, adapters).run_video(video)

    assert report.status == StepStatus.DONE
    scenes = len(script.load(tmp_path / "runs" / str(video)).scenes)
    assert len(images.calls) == scenes  # the frame drawn before the pause was reviewed, not redrawn
    reviews = [r for r in store.costs(video) if r.unit == "frames"]
    assert len(reviews) == scenes
    assert {r.billing for r in reviews} == {"subscription_credit"}


def test_a_flaky_review_never_draws_a_frame_twice(store: StateStore, tmp_path: Path) -> None:
    images = FakeImages()
    flaky = ChaosReviewer(FakeReviewer(), ChaosConfig(fail_rate=0.4, garbage_rate=0.2, seed=2), _no)
    runner = _runner(store, tmp_path, _adapters(images, flaky))
    video = runner.create_video("fact-of-day", "why octopuses have three hearts")

    report = runner.run_video(video)
    if report.status != StepStatus.DONE:  # out of step retries: a person types resume
        report = runner.run_video(video)

    assert report.status == StepStatus.DONE
    assert len(images.calls) == len(script.load(tmp_path / "runs" / str(video)).scenes)
    assert flaky.stats.failures + flaky.stats.garbage > 0


def _no(_seconds: float) -> None:
    return None


# --- the port's wrappers ----------------------------------------------------------------


REQUEST = FrameCheckRequest(image=b"\x89PNG", media_type="image/png", scene="a wombat", style="s")


@pytest.mark.parametrize(
    ("config", "outcome"),
    [
        (ChaosConfig(fail_rate=1.0, garbage_rate=0.0, seed=1), "simulated vision outage"),
        (ChaosConfig(fail_rate=0.0, garbage_rate=1.0, seed=1), "off the schema"),
    ],
)
def test_chaos_review_fails_as_a_transient_error(config: ChaosConfig, outcome: str) -> None:
    with pytest.raises(TransientAdapterError, match=outcome):
        ChaosReviewer(FakeReviewer(), config, _no).review(REQUEST)


def test_chaos_review_passes_a_clean_answer_through() -> None:
    inner = FakeReviewer()
    chaos = ChaosReviewer(inner, ChaosConfig(fail_rate=0.0, garbage_rate=0.0, seed=1), _no)

    assert chaos.review(REQUEST).accepted
    assert (chaos.provider, chaos.model) == (inner.provider, inner.model)


def test_each_review_is_recorded_as_one_frame_of_subscription_credit(store: StateStore) -> None:
    ledger = Ledger(store)
    video = store.create_video("fact-of-day", "t", ["visuals"])
    metered = MeteredReviewer(CreditReviewer(), ledger)

    with ledger.scope(video, "visuals"):
        metered.review(REQUEST)

    (row,) = store.costs(video)
    assert (row.provider, row.model, row.unit, row.units) == (
        "claude_cli",
        "test-model",
        "frames",
        1,
    )
    assert row.billing == "subscription_credit"
    assert row.cost_usd == pytest.approx(0.10)


def test_a_failed_review_is_not_billed(store: StateStore) -> None:
    ledger = Ledger(store)
    video = store.create_video("fact-of-day", "t", ["visuals"])
    metered = MeteredReviewer(Down(times=1), ledger)

    with ledger.scope(video, "visuals"), pytest.raises(TransientAdapterError):
        metered.review(REQUEST)

    assert store.costs(video) == []
