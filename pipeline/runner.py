"""Runner: drives one video through its steps, in order, via the state store.

For each step that is not done yet:

    start -> run -> done
               |
               +-- retryable error -> wait (exponential backoff + jitter) -> try again
               +-- non-retryable error, or out of attempts -> failed + alert, stop

What makes it survive failures:
- Done steps are skipped, and their outputs are cached, so a rerun never pays twice.
- A crash (power cut, Ctrl+C, killed process) is not caught on purpose: the step
  stays "running" in the database, and the next Runner turns it into "failed"
  at startup and resumes exactly there.
- Every decision is logged as a JSON line, which the soak report will aggregate.
"""

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.adapters.errors import TransientAdapterError
from pipeline.adapters.notify import Notifier
from pipeline.adapters.registry import Adapters
from pipeline.config import SeriesConfig, load_series
from pipeline.logs import get_logger
from pipeline.state import StateError, StateStore, StepStatus
from pipeline.steps import render, script, visuals, voice
from pipeline.steps.base import StepContext, StepError, StepResult

StepFn = Callable[[StepContext], StepResult]

# Stage 1 order. idea, captions, qa and publish join in stage 2.
DEFAULT_STEPS: tuple[tuple[str, StepFn], ...] = (
    ("script", script.run),
    ("voice", voice.run),
    ("visuals", visuals.run),
    ("render", render.run),
)

log = get_logger()


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0
    # +-25 %: retries from many videos do not hit a recovering API at the same instant.
    jitter: float = 0.25

    def delay_s(self, attempt: int, rng: random.Random) -> float:
        """Wait before retry number `attempt` + 1: 2 s, 4 s, 8 s, ... capped at max_delay_s."""
        raw = self.base_delay_s * 2.0 ** (attempt - 1)
        jittered = raw * (1 + self.jitter * (2 * rng.random() - 1))
        return min(self.max_delay_s, jittered)


@dataclass(frozen=True, slots=True)
class StepOutcome:
    step: str
    status: StepStatus
    attempts: int  # attempts made in this run (0 = already done before)
    cost_usd: float
    cached: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class VideoReport:
    video_id: int
    status: StepStatus
    steps: tuple[StepOutcome, ...]
    cost_usd: float
    elapsed_s: float
    output: Path | None

    @property
    def failed_step(self) -> StepOutcome | None:
        return next((s for s in self.steps if s.status == StepStatus.FAILED), None)


class Runner:
    def __init__(
        self,
        *,
        store: StateStore,
        adapters: Adapters,
        runs_dir: Path,
        policy: RetryPolicy | None = None,
        steps: Sequence[tuple[str, StepFn]] = DEFAULT_STEPS,
        notifier: Notifier | None = None,
        series_loader: Callable[[str], SeriesConfig] = load_series,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._adapters = adapters
        self._runs_dir = runs_dir
        self._policy = policy or RetryPolicy()
        self._steps = tuple(steps)
        self._notifier = notifier
        self._load_series = series_loader
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._clock = clock
        # Single worker: anything still "running" belongs to a process that died.
        self.recovered = store.recover_interrupted()
        if self.recovered:
            log.warning("recovered_interrupted_steps", extra={"count": self.recovered})

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._steps)

    def create_video(self, series_id: str, topic: str) -> int:
        video_id = self._store.create_video(series_id, topic, self.step_names)
        log.info("video_created", extra={"video_id": video_id, "series": series_id, "topic": topic})
        return video_id

    def run_video(self, video_id: int) -> VideoReport:
        started = self._clock()
        record = self._store.video(video_id)
        stored = tuple(s.name for s in self._store.steps(video_id))
        if stored != self.step_names:
            raise StateError(f"video {video_id} has steps {stored}, runner has {self.step_names}")

        ctx = StepContext(
            video_id=video_id,
            topic=record.topic,
            series=self._load_series(record.series_id),
            run_dir=self._runs_dir / str(video_id),
            llm_heavy=self._adapters.llm_heavy,
            llm_light=self._adapters.llm_light,
            tts=self._adapters.tts,
            images=self._adapters.images,
        )
        outcomes: list[StepOutcome] = []
        for name, fn in self._steps:
            if outcomes and outcomes[-1].status != StepStatus.DONE:
                outcomes.append(StepOutcome(name, StepStatus.PENDING, 0, 0.0, False))
                continue
            if self._store.step(video_id, name).status == StepStatus.DONE:
                outcomes.append(StepOutcome(name, StepStatus.DONE, 0, 0.0, True))
                continue
            outcomes.append(self._run_step(ctx, name, fn))

        status = self._store.video_status(video_id)
        output = ctx.run_dir / render.OUTPUT
        report = VideoReport(
            video_id=video_id,
            status=status,
            steps=tuple(outcomes),
            cost_usd=sum(o.cost_usd for o in outcomes),
            elapsed_s=self._clock() - started,
            output=output if status == StepStatus.DONE and output.exists() else None,
        )
        log.info(
            "video_finished",
            extra={"video_id": video_id, "status": str(status), "cost_usd": report.cost_usd},
        )
        failed = report.failed_step
        if failed is not None:
            self._alert(record.series_id, record.topic, video_id, failed)
        return report

    def _run_step(self, ctx: StepContext, name: str, fn: StepFn) -> StepOutcome:
        video_id = ctx.video_id
        for attempt in range(1, self._policy.max_attempts + 1):
            self._store.start_step(video_id, name)
            step_started = self._clock()
            try:
                result = fn(ctx)
            except StepError as e:
                error, retryable = str(e), e.retryable
            except TransientAdapterError as e:
                error, retryable = f"{type(e).__name__}: {e}", True
            except Exception as e:
                # Unknown errors are retried: with the cache, a retry costs little,
                # and a real bug still ends as "failed" with its type in the alert.
                log.exception("step_crashed", extra={"video_id": video_id, "step": name})
                error, retryable = f"unexpected {type(e).__name__}: {e}", True
            else:
                self._store.finish_step(video_id, name)
                log.info(
                    "step_done",
                    extra={
                        "video_id": video_id,
                        "step": name,
                        "attempt": attempt,
                        "cached": result.cached,
                        "cost_usd": result.cost_usd,
                        "elapsed_s": round(self._clock() - step_started, 3),
                    },
                )
                return StepOutcome(name, StepStatus.DONE, attempt, result.cost_usd, result.cached)

            self._store.fail_step(video_id, name, error)
            log.warning(
                "step_failed",
                extra={
                    "video_id": video_id,
                    "step": name,
                    "attempt": attempt,
                    "retryable": retryable,
                    "error": error,
                },
            )
            if not retryable or attempt == self._policy.max_attempts:
                return StepOutcome(name, StepStatus.FAILED, attempt, 0.0, False, error)
            delay = self._policy.delay_s(attempt, self._rng)
            log.info(
                "retry_scheduled",
                extra={"video_id": video_id, "step": name, "attempt": attempt, "delay_s": delay},
            )
            self._sleep(delay)
        raise AssertionError("unreachable: the loop always returns")

    def _alert(self, series_id: str, topic: str, video_id: int, failed: StepOutcome) -> None:
        if self._notifier is None:
            return
        self._notifier.send(
            f"video {video_id} ({series_id}: {topic!r}) failed at '{failed.step}' after "
            f"{failed.attempts} attempt(s): {failed.error}\n"
            f"resume: python -m pipeline resume --video {video_id}"
        )
