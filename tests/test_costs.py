"""Cost ledger and budget guard: every call is recorded, limits stop paid calls."""

import random
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.adapters.chaos import ChaosConfig, ChaosLlm
from pipeline.adapters.errors import TransientAdapterError
from pipeline.adapters.images import ImageRequest, ImageResult
from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm import LlmRequest
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.notify.fake import RecordingNotifier
from pipeline.adapters.registry import Adapters
from pipeline.adapters.tts.fake import FakeTts
from pipeline.cli import DB_NAME, main
from pipeline.config import BudgetConfig
from pipeline.costs import Ledger, MeteredLlm, billing_for, month_start
from pipeline.runner import DEFAULT_STEPS, RetryPolicy, Runner, StepFn
from pipeline.state import MIGRATIONS, CostRecord, StateStore, StepStatus
from pipeline.steps import script, visuals, voice
from pipeline.steps.base import StepContext, StepResult

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
TOPIC = "why octopuses have three hearts"


def stub_render(ctx: StepContext) -> StepResult:
    frames = visuals.frame_paths(ctx.run_dir, len(script.load(ctx.run_dir).scenes))
    voice.voice_file(ctx.run_dir)
    return StepResult(outputs=tuple(frames), cost_usd=0.0, cached=False)


STEPS: list[tuple[str, StepFn]] = [*DEFAULT_STEPS[:-1], ("render", stub_render)]


class PricedImages(FakeImages):
    """Fake images that cost money, to exercise the budget."""

    def __init__(self, price: float) -> None:
        super().__init__()
        self.price = price

    def generate(self, request: ImageRequest) -> ImageResult:
        return replace(super().generate(request), cost_usd=self.price)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[StateStore]:
    with StateStore(tmp_path / DB_NAME) as s:
        yield s


def _adapters(images: FakeImages | None = None) -> Adapters:
    llm = FakeLlm()
    return Adapters(llm_heavy=llm, llm_light=llm, tts=FakeTts(), images=images or FakeImages())


def _runner(
    store: StateStore,
    tmp_path: Path,
    adapters: Adapters,
    budget: BudgetConfig | None = None,
    notifier: RecordingNotifier | None = None,
    now: datetime = NOW,
) -> Runner:
    return Runner(
        store=store,
        adapters=adapters,
        runs_dir=tmp_path / "runs",
        policy=RetryPolicy(max_attempts=3, base_delay_s=0, jitter=0),
        steps=STEPS,
        notifier=notifier,
        budget=budget,
        sleep=lambda _s: None,
        rng=random.Random(0),
        wall_clock=lambda: now,
    )


# --- recording ------------------------------------------------------------------


def test_every_adapter_call_is_recorded_per_step(store: StateStore, tmp_path: Path) -> None:
    runner = _runner(store, tmp_path, _adapters())
    video = runner.create_video("fact-of-day", TOPIC)

    assert runner.run_video(video).status == StepStatus.DONE

    rows = store.costs(video)
    scenes = len(script.load(tmp_path / "runs" / str(video)).scenes)
    by_step = {step: [r for r in rows if r.step == step] for step in ("script", "voice", "visuals")}
    assert [len(by_step[s]) for s in ("script", "voice", "visuals")] == [1, 1, scenes]
    assert {r.unit for r in by_step["script"]} == {"tokens"}
    assert {r.unit for r in by_step["voice"]} == {"characters"}
    assert {r.unit for r in by_step["visuals"]} == {"images"}
    assert {r.billing for r in rows} == {"fake"}
    assert all(r.units > 0 for r in rows)


def test_cached_rerun_records_nothing(store: StateStore, tmp_path: Path) -> None:
    runner = _runner(store, tmp_path, _adapters())
    video = runner.create_video("fact-of-day", TOPIC)
    runner.run_video(video)
    rows_before = len(store.costs(video))

    store.reset_from(video, "script")
    runner.run_video(video)

    assert len(store.costs(video)) == rows_before  # cache hits are free and leave no trace


def test_garbage_answers_are_billed_failed_calls_are_not(store: StateStore) -> None:
    ledger = Ledger(store, clock=lambda: NOW)
    video = store.create_video("fact-of-day", TOPIC, ["script"])
    request = LlmRequest(task="script", system="s", prompt="p")
    garbage = MeteredLlm(ChaosLlm(FakeLlm(), ChaosConfig(0.0, 1.0), lambda _s: None), ledger)
    outage = MeteredLlm(ChaosLlm(FakeLlm(), ChaosConfig(1.0, 0.0), lambda _s: None), ledger)

    with ledger.scope(video, "script"):
        garbage.complete(request)  # a malformed answer still costs money
        with pytest.raises(TransientAdapterError):
            outage.complete(request)  # no answer, no bill

    assert len(store.costs(video)) == 1


# --- limits ---------------------------------------------------------------------


def test_per_video_limit_stops_paid_calls_and_alerts(store: StateStore, tmp_path: Path) -> None:
    images = PricedImages(price=0.10)
    notifier = RecordingNotifier()
    budget = BudgetConfig(max_cost_per_video_usd=0.25, max_monthly_usd=10.0)
    runner = _runner(store, tmp_path, _adapters(images), budget=budget, notifier=notifier)
    video = runner.create_video("fact-of-day", TOPIC)

    report = runner.run_video(video)

    assert report.status == StepStatus.FAILED
    failed = report.failed_step
    assert failed is not None
    assert failed.step == "visuals"
    assert failed.attempts == 1  # never retried: waiting does not add money
    assert "budget" in (failed.error or "")
    # 0.00 -> call, 0.10 -> call, 0.20 -> call, 0.30 >= 0.25 -> stop: at most one call over.
    assert len(images.calls) == 3
    assert report.cost_usd == pytest.approx(0.30)
    (alert,) = notifier.messages
    assert "budget" in alert
    assert "$0.25" in alert


def test_monthly_limit_blocks_new_videos_but_last_month_does_not_count(
    store: StateStore, tmp_path: Path
) -> None:
    def old_spend(ts: str, usd: float) -> None:
        store.record_cost(CostRecord(None, None, "fal", "flux", "api", "images", 1, usd, ts))

    old_spend("2026-10-31T23:59:59+00:00", 99.0)  # last month: ignored
    budget = BudgetConfig(max_cost_per_video_usd=0.5, max_monthly_usd=1.0)
    llm = FakeLlm()
    adapters = Adapters(llm_heavy=llm, llm_light=llm, tts=FakeTts(), images=FakeImages())

    first = _runner(store, tmp_path, adapters, budget=budget)
    assert first.run_video(first.create_video("fact-of-day", TOPIC)).status == StepStatus.DONE

    old_spend("2026-11-05T10:00:00+00:00", 1.0)  # this month: the limit is now reached
    calls_before = len(llm.calls)
    second = _runner(store, tmp_path, adapters, budget=budget)
    report = second.run_video(second.create_video("fact-of-day", "how glass is made"))

    assert report.status == StepStatus.FAILED
    assert report.failed_step is not None
    assert "this month" in (report.failed_step.error or "")
    assert len(llm.calls) == calls_before  # not a single paid call was made


def test_no_budget_means_record_but_never_block(store: StateStore, tmp_path: Path) -> None:
    images = PricedImages(price=5.0)
    runner = _runner(store, tmp_path, _adapters(images), budget=None)

    assert runner.run_video(runner.create_video("fact-of-day", TOPIC)).status == StepStatus.DONE
    assert runner.ledger.spend().month_usd > 5.0


# --- helpers, config, migration, CLI ----------------------------------------------


def test_billing_kinds() -> None:
    assert billing_for("fake") == "fake"
    assert billing_for("claude_cli") == "subscription"
    assert billing_for("elevenlabs") == "api"


def test_month_starts_in_utc() -> None:
    late_seoul = datetime.fromisoformat("2026-12-01T05:00:00+09:00")  # still Nov 30 in UTC

    assert month_start(late_seoul) == datetime(2026, 11, 1, tzinfo=UTC)


def test_monthly_limit_must_cover_one_video() -> None:
    with pytest.raises(ValueError, match="at least"):
        BudgetConfig(max_cost_per_video_usd=1.0, max_monthly_usd=0.5)


def test_old_database_is_upgraded_in_place(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(f"BEGIN;\n{MIGRATIONS[0]}\nPRAGMA user_version = 1;\nCOMMIT;")
    conn.execute(
        "INSERT INTO videos (series_id, topic, created_at) VALUES ('fact-of-day', 'x', 'now')"
    )
    conn.commit()
    conn.close()

    with StateStore(db) as store:
        assert store.schema_version == len(MIGRATIONS)
        assert [v.topic for v in store.videos()] == ["x"]  # data survived the upgrade
        assert store.video_cost(1) == 0


def test_status_shows_spend(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with StateStore(tmp_path / DB_NAME) as store:
        runner = _runner(store, tmp_path, _adapters(PricedImages(price=0.01)))
        video = runner.create_video("fact-of-day", TOPIC)
        runner.run_video(video)
        paid = store.video_cost(video)

    assert main(["status", "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert paid > 0
    assert f"${paid:.3f}" in out  # per video
    assert "this month:" in out
    assert "calls" in out
