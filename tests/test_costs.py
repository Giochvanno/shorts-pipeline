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
from pipeline.adapters.llm import LlmRequest, LlmResponse
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.notify.fake import RecordingNotifier
from pipeline.adapters.registry import Adapters
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.tts.fake import FakeTts
from pipeline.cli import DB_NAME, main
from pipeline.config import BudgetConfig
from pipeline.costs import Ledger, MeteredLlm, billing_for, month_start
from pipeline.runner import DEFAULT_STEPS, RetryPolicy, Runner, StepFn
from pipeline.state import MIGRATIONS, CostRecord, StateStore, StepStatus
from pipeline.steps import script, visuals, voice
from pipeline.steps.base import StepContext, StepResult

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
NOW_ISO = NOW.isoformat()
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
    return Adapters(
        llm_heavy=llm,
        llm_light=llm,
        tts=FakeTts(),
        images=images or FakeImages(),
        stt=FakeTranscriber(),
    )


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
    steps = ("script", "voice", "visuals", "captions")
    by_step = {step: [r for r in rows if r.step == step] for step in steps}
    # voice = synthesis + the check that listens to it; visuals = a frame + its review
    # per scene; captions reuse what was heard.
    assert [len(by_step[s]) for s in steps] == [1, 2, 2 * scenes, 0]
    assert {r.unit for r in by_step["script"]} == {"tokens"}
    assert {r.unit for r in by_step["voice"]} == {"characters", "seconds"}
    assert {r.unit for r in by_step["visuals"]} == {"images", "frames"}
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


def budget(
    *,
    video: float = 10.0,
    images: float | None = None,
    month: float = 10.0,
    credit: float = 10.0,
    warn_at: float = 0.8,
) -> BudgetConfig:
    return BudgetConfig(
        max_cash_per_video_usd=video,
        max_image_cash_per_video_usd=video if images is None else images,
        max_cash_monthly_usd=month,
        max_subscription_credit_monthly_usd=credit,
        subscription_credit_warn_at=warn_at,
    )


class CreditLlm(FakeLlm):
    """A fake that bills like `claude -p`: subscription credit, not cash."""

    def __init__(self, price: float) -> None:
        super().__init__()
        self.price = price

    @property
    def provider(self) -> str:
        return "claude_cli"

    def complete(self, request: LlmRequest) -> LlmResponse:
        return replace(super().complete(request), provider="claude_cli", cost_usd=self.price)


def test_the_video_cash_limit_pauses_the_video_and_alerts_with_the_sum(
    store: StateStore, tmp_path: Path
) -> None:
    images = PricedImages(price=0.10)
    notifier = RecordingNotifier()
    runner = _runner(
        store, tmp_path, _adapters(images), budget=budget(video=0.25), notifier=notifier
    )
    video = runner.create_video("fact-of-day", TOPIC)

    report = runner.run_video(video)

    assert report.status == StepStatus.PAUSED_BUDGET  # not failed: nothing broke
    assert report.failed_step is None
    paused = report.paused_step
    assert paused is not None and paused.step == "visuals"
    assert paused.attempts == 1  # never retried: waiting does not add money
    # 0.00 -> call, 0.10 -> call, 0.20 -> call, 0.30 >= 0.25 -> stop: at most one call over.
    assert len(images.calls) == 3
    assert report.cash_usd == pytest.approx(0.30)
    assert store.step(video, "visuals").status == StepStatus.PAUSED_BUDGET
    assert store.video_status(video) == StepStatus.PAUSED_BUDGET
    (alert,) = notifier.messages
    assert "paused at 'visuals'" in alert
    assert "$0.300 cash of its $0.25 limit" in alert
    assert f"resume --video {video}" in alert


def test_the_image_cash_limit_stops_a_redraw_loop_before_the_video_limit(
    store: StateStore, tmp_path: Path
) -> None:
    images = PricedImages(price=0.10)
    runner = _runner(store, tmp_path, _adapters(images), budget=budget(video=5.0, images=0.15))
    video = runner.create_video("fact-of-day", TOPIC)

    report = runner.run_video(video)

    assert report.status == StepStatus.PAUSED_BUDGET
    assert report.paused_step is not None
    assert "on images of its $0.15 limit" in (report.paused_step.error or "")
    assert len(images.calls) == 2


def test_resume_after_raising_the_limit_continues_from_the_paused_step(
    store: StateStore, tmp_path: Path
) -> None:
    images = PricedImages(price=0.10)
    adapters = _adapters(images)
    first = _runner(store, tmp_path, adapters, budget=budget(video=0.25))
    video = first.create_video("fact-of-day", TOPIC)
    first.run_video(video)
    script_tries = store.step(video, "script").attempts
    drawn = len(images.calls)

    second = _runner(store, tmp_path, adapters, budget=budget(video=2.0))
    report = second.run_video(video)

    assert report.status == StepStatus.DONE
    assert store.step(video, "script").attempts == script_tries  # earlier steps not rerun
    scenes = len(script.load(tmp_path / "runs" / str(video)).scenes)
    assert len(images.calls) == scenes  # the frames drawn before the pause were kept
    assert len(images.calls) > drawn


def test_resume_without_raising_the_limit_pauses_again_for_free(
    store: StateStore, tmp_path: Path
) -> None:
    images = PricedImages(price=0.10)
    runner = _runner(store, tmp_path, _adapters(images), budget=budget(video=0.25))
    video = runner.create_video("fact-of-day", TOPIC)
    runner.run_video(video)
    calls, cash = len(images.calls), store.spent(("fake", "cash"), video_id=video)[0]

    report = runner.run_video(video)

    assert report.status == StepStatus.PAUSED_BUDGET
    assert len(images.calls) == calls  # the guard stops before any paid call
    assert store.spent(("fake", "cash"), video_id=video)[0] == cash


def test_subscription_credit_is_kept_apart_from_the_video_cash(
    store: StateStore, tmp_path: Path
) -> None:
    llm = CreditLlm(price=0.40)
    adapters = Adapters(
        llm_heavy=llm, llm_light=llm, tts=FakeTts(), images=FakeImages(), stt=FakeTranscriber()
    )
    runner = _runner(store, tmp_path, adapters, budget=budget(video=0.25))
    video = runner.create_video("fact-of-day", TOPIC)

    report = runner.run_video(video)

    assert report.status == StepStatus.DONE  # $0.40 of credit, but the cash limit is $0.25
    assert report.cash_usd == 0.0
    assert report.credit_usd == pytest.approx(0.40)
    assert report.cost_usd == pytest.approx(0.40)  # the full price counts both
    assert {r.billing for r in store.costs(video) if r.provider == "claude_cli"} == {
        "subscription_credit"
    }


def test_the_monthly_credit_limit_pauses_scripts(store: StateStore, tmp_path: Path) -> None:
    llm = CreditLlm(price=0.40)
    adapters = Adapters(
        llm_heavy=llm, llm_light=llm, tts=FakeTts(), images=FakeImages(), stt=FakeTranscriber()
    )
    store.record_cost(
        CostRecord(None, None, "claude_cli", "m", "subscription_credit", "tokens", 1, 1.0, NOW_ISO)
    )
    runner = _runner(store, tmp_path, adapters, budget=budget(credit=1.0))

    report = runner.run_video(runner.create_video("fact-of-day", TOPIC))

    assert report.status == StepStatus.PAUSED_BUDGET
    assert report.paused_step is not None and report.paused_step.step == "script"
    assert "subscription credit $1.00 reached the $1.00 limit" in (report.paused_step.error or "")
    assert llm.calls == []


def test_the_credit_warning_comes_once_when_80_percent_is_crossed(
    store: StateStore, tmp_path: Path
) -> None:
    llm = CreditLlm(price=0.40)
    notifier = RecordingNotifier()
    store.record_cost(
        CostRecord(None, None, "claude_cli", "m", "subscription_credit", "tokens", 1, 7.7, NOW_ISO)
    )
    ledger = Ledger(store, budget(credit=10.0), clock=lambda: NOW, warn=notifier.send)
    metered_llm = MeteredLlm(llm, ledger)
    request = LlmRequest(task="script", system="s", prompt="p")

    metered_llm.complete(request)  # 7.70 -> 8.10: crosses 8.00
    metered_llm.complete(request)  # 8.10 -> 8.50: already past it

    (warning,) = notifier.messages
    assert "subscription credit this month: $8.10 of $10.00 (81%)" in warning


def test_the_credit_warning_reaches_the_notifier_during_a_run(
    store: StateStore, tmp_path: Path
) -> None:
    """The runner logs the warning too: a log field named "message" used to crash the run."""
    llm = CreditLlm(price=0.40)
    notifier = RecordingNotifier()
    adapters = Adapters(
        llm_heavy=llm, llm_light=llm, tts=FakeTts(), images=FakeImages(), stt=FakeTranscriber()
    )
    runner = _runner(store, tmp_path, adapters, budget=budget(credit=0.45), notifier=notifier)

    report = runner.run_video(runner.create_video("fact-of-day", TOPIC))

    assert report.status == StepStatus.DONE
    assert len(llm.calls) == 1  # the paid script was not lost to a crash and redone
    (warning,) = notifier.messages
    assert "subscription credit this month: $0.40 of $0.45" in warning


def test_monthly_cash_limit_blocks_new_videos_but_last_month_does_not_count(
    store: StateStore, tmp_path: Path
) -> None:
    def old_spend(ts: str, usd: float) -> None:
        store.record_cost(CostRecord(None, None, "fal", "flux", "cash", "images", 1, usd, ts))

    old_spend("2026-10-31T23:59:59+00:00", 99.0)  # last month: ignored
    limits = budget(video=0.5, month=1.0)
    llm = FakeLlm()
    adapters = Adapters(
        llm_heavy=llm, llm_light=llm, tts=FakeTts(), images=FakeImages(), stt=FakeTranscriber()
    )

    first = _runner(store, tmp_path, adapters, budget=limits)
    assert first.run_video(first.create_video("fact-of-day", TOPIC)).status == StepStatus.DONE

    old_spend("2026-11-05T10:00:00+00:00", 1.0)  # this month: the limit is now reached
    calls_before = len(llm.calls)
    second = _runner(store, tmp_path, adapters, budget=limits)
    report = second.run_video(second.create_video("fact-of-day", "how glass is made"))

    assert report.status == StepStatus.PAUSED_BUDGET
    assert report.paused_step is not None
    assert "this month's cash" in (report.paused_step.error or "")
    assert len(llm.calls) == calls_before  # not a single paid call was made


def test_no_budget_means_record_but_never_block(store: StateStore, tmp_path: Path) -> None:
    images = PricedImages(price=5.0)
    runner = _runner(store, tmp_path, _adapters(images), budget=None)

    assert runner.run_video(runner.create_video("fact-of-day", TOPIC)).status == StepStatus.DONE
    assert runner.ledger.spend().month_cash > 5.0


# --- helpers, config, migration, CLI ----------------------------------------------


def test_billing_kinds() -> None:
    assert billing_for("fake") == "fake"
    assert billing_for("claude_cli") == "subscription_credit"
    assert billing_for("elevenlabs") == "cash"
    assert billing_for("fal") == "cash"


def test_month_starts_in_utc() -> None:
    late_seoul = datetime.fromisoformat("2026-12-01T05:00:00+09:00")  # still Nov 30 in UTC

    assert month_start(late_seoul) == datetime(2026, 11, 1, tzinfo=UTC)


def test_limits_must_nest() -> None:
    with pytest.raises(ValueError, match="at least"):
        budget(video=1.0, month=0.5)
    with pytest.raises(ValueError, match="cannot exceed"):
        budget(video=0.5, images=0.6)


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


def test_costs_and_steps_move_to_two_accounts_and_keep_their_data(tmp_path: Path) -> None:
    db = tmp_path / "v2.db"
    conn = sqlite3.connect(db)
    for number in range(2):
        conn.executescript(
            f"BEGIN;\n{MIGRATIONS[number]}\nPRAGMA user_version = {number + 1};\nCOMMIT;"
        )
    conn.executescript(
        "INSERT INTO videos (series_id, topic, created_at) VALUES ('fact-of-day', 'x', 'now');"
        "INSERT INTO steps (video_id, name, position, status, attempts, error)"
        " VALUES (1, 'script', 0, 'done', 1, NULL), (1, 'visuals', 1, 'failed', 3, 'boom');"
        "INSERT INTO costs (video_id, step, provider, model, billing, unit, units, cost_usd, ts)"
        " VALUES (1, 'visuals', 'fal', 'flux', 'api', 'images', 1, 0.025, 't'),"
        " (1, 'script', 'claude_cli', 'm', 'subscription', 'tokens', 9, 0.07, 't'),"
        " (NULL, NULL, 'fake', 'f', 'fake', 'images', 1, 0.0, 't');"
    )
    conn.close()

    with StateStore(db) as store:
        assert store.schema_version == len(MIGRATIONS)
        assert [c.billing for c in store.costs(1)] == ["cash", "subscription_credit"]
        assert store.spent(("cash",), video_id=1)[0] == pytest.approx(0.025)
        assert store.spent(("subscription_credit",), video_id=1)[0] == pytest.approx(0.07)
        assert store.spent(("fake",))[1] == 1
        visuals_step = store.step(1, "visuals")
        assert (visuals_step.status, visuals_step.attempts, visuals_step.error) == (
            StepStatus.FAILED,
            3,
            "boom",
        )
        store.start_step(1, "visuals")
        store.pause_step(1, "visuals", "budget: test")  # the new status is allowed
        assert store.video_status(1) == StepStatus.PAUSED_BUDGET


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
    for line in ("cash:", "subscription credit:", "full cost:", "calls in all"):
        assert line in out
    assert "cash limit $0.75, of it images $0.55" in out
