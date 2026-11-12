"""Runner: order, retries with backoff, alerts, crash recovery, resume, logs.

Most tests use scripted stub steps, so each failure mode is exact and fast.
The last test runs the real script/voice/visuals steps on fakes.
"""

import json
import logging
import random
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from pipeline.adapters.errors import TransientAdapterError
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.notify.fake import RecordingNotifier
from pipeline.adapters.registry import Adapters
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.tts.fake import FakeTts
from pipeline.logs import get_logger, log_to_file
from pipeline.runner import DEFAULT_STEPS, RetryPolicy, Runner, StepFn
from pipeline.state import INTERRUPTED, StateError, StateStore, StepStatus
from pipeline.steps import script, visuals, voice
from pipeline.steps.base import StepContext, StepError, StepResult

NO_JITTER = RetryPolicy(max_attempts=3, base_delay_s=1.0, max_delay_s=60.0, jitter=0.0)


class SimulatedCrash(BaseException):
    """Like the process being killed: not an Exception, so nothing in the runner catches it."""


class Stub:
    """A step that raises the queued errors first, then succeeds. Counts its calls."""

    def __init__(self, *errors: BaseException, cost: float = 0.0) -> None:
        self.errors = list(errors)
        self.cost = cost
        self.calls = 0

    def __call__(self, ctx: StepContext) -> StepResult:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return StepResult(outputs=(), cost_usd=self.cost, cached=False)


def _adapters() -> Adapters:
    return Adapters(
        llm_heavy=FakeLlm(),
        llm_light=FakeLlm(),
        tts=FakeTts(),
        images=FakeImages(),
        stt=FakeTranscriber(),
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[StateStore]:
    with StateStore(tmp_path / "pipeline.db") as s:
        yield s


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()


@pytest.fixture
def sleeps() -> list[float]:
    return []


def _runner(
    store: StateStore,
    tmp_path: Path,
    steps: Mapping[str, StepFn],
    *,
    notifier: RecordingNotifier | None = None,
    sleeps: list[float] | None = None,
    policy: RetryPolicy = NO_JITTER,
) -> Runner:
    return Runner(
        store=store,
        adapters=_adapters(),
        runs_dir=tmp_path / "runs",
        policy=policy,
        steps=list(steps.items()),
        notifier=notifier,
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
        rng=random.Random(0),
    )


# --- happy path and retries -------------------------------------------------------


def test_runs_all_steps_in_order(store: StateStore, tmp_path: Path) -> None:
    steps = {"a": Stub(cost=0.01), "b": Stub(cost=0.02), "c": Stub()}
    runner = _runner(store, tmp_path, dict(steps))
    video = runner.create_video("fact-of-day", "octopus")

    report = runner.run_video(video)

    assert report.status == StepStatus.DONE
    assert [o.step for o in report.steps] == ["a", "b", "c"]
    assert all(o.attempts == 1 for o in report.steps)
    assert sum(o.cost_usd for o in report.steps) == pytest.approx(0.03)  # what steps reported
    assert report.cost_usd == 0  # what the ledger saw: stubs call no adapter


def test_retryable_error_is_retried_with_growing_pauses(
    store: StateStore, tmp_path: Path, sleeps: list[float]
) -> None:
    flaky = Stub(StepError("bad JSON", retryable=True), TransientAdapterError("429 rate limited"))
    runner = _runner(store, tmp_path, {"a": flaky}, sleeps=sleeps)
    video = runner.create_video("fact-of-day", "octopus")

    report = runner.run_video(video)

    assert report.status == StepStatus.DONE
    assert report.steps[0].attempts == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff
    assert store.step(video, "a").attempts == 3


def test_unexpected_exception_is_retried_and_named(store: StateStore, tmp_path: Path) -> None:
    runner = _runner(store, tmp_path, {"a": Stub(ValueError("boom"), ValueError("boom"))})
    video = runner.create_video("fact-of-day", "octopus")

    assert runner.run_video(video).status == StepStatus.DONE


# --- failures and alerts ----------------------------------------------------------


def test_non_retryable_error_stops_at_once_and_alerts(
    store: StateStore, tmp_path: Path, notifier: RecordingNotifier, sleeps: list[float]
) -> None:
    later = Stub()
    steps = {"a": Stub(StepError("set voice_id", retryable=False)), "b": later}
    runner = _runner(store, tmp_path, steps, notifier=notifier, sleeps=sleeps)
    video = runner.create_video("fact-of-day", "octopus")

    report = runner.run_video(video)

    assert report.status == StepStatus.FAILED
    assert report.steps[0].attempts == 1
    assert sleeps == []  # no pointless waiting
    assert later.calls == 0
    assert store.step(video, "b").status == StepStatus.PENDING
    (alert,) = notifier.messages
    assert "failed at 'a'" in alert
    assert "set voice_id" in alert
    assert f"resume --video {video}" in alert


def test_out_of_attempts_fails_then_resume_continues_there(
    store: StateStore, tmp_path: Path, notifier: RecordingNotifier
) -> None:
    first = Stub()
    always_down = Stub(*[TransientAdapterError("503")] * 3)
    runner = _runner(store, tmp_path, {"a": first, "b": always_down}, notifier=notifier)
    video = runner.create_video("fact-of-day", "octopus")

    report = runner.run_video(video)
    assert report.status == StepStatus.FAILED
    assert report.failed_step is not None
    assert report.failed_step.attempts == 3
    assert len(notifier.messages) == 1

    # Next night the provider is back: the same video resumes at "b".
    first_again, healthy = Stub(), Stub()
    resumed = _runner(store, tmp_path, {"a": first_again, "b": healthy}).run_video(video)

    assert resumed.status == StepStatus.DONE
    assert first_again.calls == 0  # a finished step is never run (or paid) again
    assert healthy.calls == 1
    assert store.step(video, "b").attempts == 4  # history across runs is kept


# --- crash recovery ---------------------------------------------------------------


def test_crash_mid_step_is_resumed_by_the_next_runner(store: StateStore, tmp_path: Path) -> None:
    done_before, crashing = Stub(), Stub(SimulatedCrash())
    runner = _runner(store, tmp_path, {"a": done_before, "b": crashing})
    video = runner.create_video("fact-of-day", "octopus")

    with pytest.raises(SimulatedCrash):
        runner.run_video(video)
    assert store.step(video, "b").status == StepStatus.RUNNING  # what a dead process leaves

    a_again, b_again = Stub(), Stub()
    after_restart = _runner(store, tmp_path, {"a": a_again, "b": b_again})
    report = after_restart.run_video(video)

    assert after_restart.recovered == 1
    assert report.status == StepStatus.DONE
    assert a_again.calls == 0
    assert b_again.calls == 1
    assert store.step(video, "b").error is None  # cleared by the successful retry
    assert INTERRUPTED  # the reason recorded in between (see test_state)


# --- guards, policy, logs ---------------------------------------------------------


def test_step_list_mismatch_is_refused(store: StateStore, tmp_path: Path) -> None:
    video = _runner(store, tmp_path, {"a": Stub()}).create_video("fact-of-day", "x")

    with pytest.raises(StateError, match="has steps"):
        _runner(store, tmp_path, {"a": Stub(), "b": Stub()}).run_video(video)


def test_backoff_grows_is_capped_and_jittered() -> None:
    policy = RetryPolicy(base_delay_s=2.0, max_delay_s=10.0, jitter=0.25)
    rng = random.Random(42)

    delays = [policy.delay_s(attempt, rng) for attempt in range(1, 7)]

    assert 1.5 <= delays[0] <= 2.5
    assert 3.0 <= delays[1] <= 5.0
    assert all(d <= 10.0 for d in delays)
    assert delays[-1] == pytest.approx(10.0, abs=2.5)


def test_every_decision_is_logged_as_json(store: StateStore, tmp_path: Path) -> None:
    handler = log_to_file(tmp_path / "pipeline.log.jsonl")
    try:
        runner = _runner(store, tmp_path, {"a": Stub(TransientAdapterError("timeout"))})
        runner.run_video(runner.create_video("fact-of-day", "octopus"))
    finally:
        get_logger().removeHandler(handler)
        handler.close()

    lines = (tmp_path / "pipeline.log.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    names = [e["event"] for e in events]
    assert names == [
        "video_created",
        "step_failed",
        "retry_scheduled",
        "step_done",
        "video_finished",
    ]
    failed = events[1]
    assert failed["step"] == "a"
    assert failed["retryable"] is True
    assert failed["level"] == logging.getLevelName(logging.WARNING)


# --- real steps on fakes ----------------------------------------------------------


def test_real_steps_run_and_a_rerun_pays_nothing(store: StateStore, tmp_path: Path) -> None:
    def fake_render(ctx: StepContext) -> StepResult:
        # Real render is covered in test_step_render; here only its inputs matter.
        frames = visuals.frame_paths(ctx.run_dir, len(script.load(ctx.run_dir).scenes))
        voice.voice_file(ctx.run_dir)
        return StepResult(outputs=tuple(frames), cost_usd=0.0, cached=False)

    steps = dict(DEFAULT_STEPS)
    steps["render"] = fake_render
    adapters = _adapters()
    runner = Runner(
        store=store, adapters=adapters, runs_dir=tmp_path / "runs", steps=list(steps.items())
    )
    video = runner.create_video("fact-of-day", "why octopuses have three hearts")

    assert runner.run_video(video).status == StepStatus.DONE

    assert isinstance(adapters.llm_heavy, FakeLlm)
    assert isinstance(adapters.tts, FakeTts)
    calls_before = (len(adapters.llm_heavy.calls), len(adapters.tts.calls))
    store.reset_from(video, "script")  # e.g. a rerun after a crash or a manual reset
    rerun = runner.run_video(video)

    assert rerun.status == StepStatus.DONE
    assert all(o.cached for o in rerun.steps if o.step != "render")
    assert (len(adapters.llm_heavy.calls), len(adapters.tts.calls)) == calls_before
