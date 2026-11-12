"""Cost ledger and budget guard.

Every completed adapter call becomes one row in the `costs` table (provider,
model, units, dollars). Before each call the guard checks the limits from
channel.yaml; if a limit is already reached, the call is not made and the
video stops with BudgetExceeded (not retryable: waiting does not add money).

Why wrappers around the adapters and not costs reported by steps: a step that
fails after three paid calls returns nothing, but those calls were still paid.
The wrappers see every call, including garbage answers that had to be retried.

A cached step makes no adapter call, so it writes no row: cache hits are free.
One call may overshoot a limit by its own cost (its price is known only after
it returns); at cents per call that is accepted.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from pipeline.adapters.images import ImageClient, ImageRequest, ImageResult
from pipeline.adapters.llm import LlmClient, LlmRequest, LlmResponse
from pipeline.adapters.registry import Adapters
from pipeline.adapters.tts import TtsClient, TtsRequest, TtsResult
from pipeline.config import BudgetConfig
from pipeline.state import CostRecord, StateStore

Clock = Callable[[], datetime]

# Billed through the Claude subscription, not per API call. Recorded and counted
# against the limits all the same, to stay on the safe side.
SUBSCRIPTION_PROVIDERS = frozenset({"claude_cli"})


class BudgetExceeded(Exception):
    """A spending limit is reached: the paid call was not made."""


def billing_for(provider: str) -> str:
    if provider == "fake":
        return "fake"
    return "subscription" if provider in SUBSCRIPTION_PROVIDERS else "api"


def month_start(now: datetime) -> datetime:
    """Start of the calendar month in UTC (no tz database needed on Windows)."""
    return now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


@dataclass(frozen=True, slots=True)
class Spend:
    video_usd: float
    month_usd: float
    month_calls: int


class Ledger:
    def __init__(
        self,
        store: StateStore,
        budget: BudgetConfig | None = None,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._budget = budget
        self._clock = clock
        self._video_id: int | None = None
        self._step: str | None = None

    @contextmanager
    def scope(self, video_id: int, step: str) -> Iterator[None]:
        """Attribute every call inside the block to this video and step."""
        previous = (self._video_id, self._step)
        self._video_id, self._step = video_id, step
        try:
            yield
        finally:
            self._video_id, self._step = previous

    def spend(self, video_id: int | None = None) -> Spend:
        month_usd, calls = self._store.cost_since(month_start(self._clock()).isoformat())
        video = video_id if video_id is not None else self._video_id
        video_usd = self._store.video_cost(video) if video is not None else 0.0
        return Spend(video_usd=video_usd, month_usd=month_usd, month_calls=calls)

    def check(self) -> None:
        """Raise BudgetExceeded if a limit is already reached. Called before each paid call."""
        if self._budget is None:
            return
        spend = self.spend()
        if self._video_id is not None and spend.video_usd >= self._budget.max_cost_per_video_usd:
            raise BudgetExceeded(
                f"budget: video {self._video_id} has spent ${spend.video_usd:.3f}"
                f" of its ${self._budget.max_cost_per_video_usd:.2f} limit"
            )
        if spend.month_usd >= self._budget.max_monthly_usd:
            raise BudgetExceeded(
                f"budget: this month's spend ${spend.month_usd:.2f}"
                f" reached the ${self._budget.max_monthly_usd:.2f} limit"
            )

    def record(
        self, *, provider: str, model: str, unit: str, units: float, cost_usd: float
    ) -> None:
        self._store.record_cost(
            CostRecord(
                video_id=self._video_id,
                step=self._step,
                provider=provider,
                model=model,
                billing=billing_for(provider),
                unit=unit,
                units=units,
                cost_usd=cost_usd,
                ts=self._clock().astimezone(UTC).isoformat(timespec="milliseconds"),
            )
        )


# --- metering wrappers: same port, plus check-before and record-after ---------------


class MeteredLlm:
    def __init__(self, inner: LlmClient, ledger: Ledger) -> None:
        self.inner, self._ledger = inner, ledger

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(self, request: LlmRequest) -> LlmResponse:
        self._ledger.check()
        response = self.inner.complete(request)
        self._ledger.record(
            provider=response.provider,
            model=response.model,
            unit="tokens",
            units=response.input_tokens + response.output_tokens,
            cost_usd=response.cost_usd,
        )
        return response


class MeteredTts:
    def __init__(self, inner: TtsClient, ledger: Ledger) -> None:
        self.inner, self._ledger = inner, ledger

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def synthesize(self, request: TtsRequest) -> TtsResult:
        self._ledger.check()
        result = self.inner.synthesize(request)
        self._ledger.record(
            provider=result.provider,
            model=result.model,
            unit="characters",
            units=result.characters,
            cost_usd=result.cost_usd,
        )
        return result


class MeteredImages:
    def __init__(self, inner: ImageClient, ledger: Ledger) -> None:
        self.inner, self._ledger = inner, ledger

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def generate(self, request: ImageRequest) -> ImageResult:
        self._ledger.check()
        result = self.inner.generate(request)
        self._ledger.record(
            provider=result.provider,
            model=result.model,
            unit="images",
            units=1,
            cost_usd=result.cost_usd,
        )
        return result


def metered(adapters: Adapters, ledger: Ledger) -> Adapters:
    """The same adapter set, with every call checked against the budget and recorded."""
    heavy = MeteredLlm(adapters.llm_heavy, ledger)
    light = (
        heavy
        if adapters.llm_light is adapters.llm_heavy
        else MeteredLlm(adapters.llm_light, ledger)
    )
    return Adapters(
        llm_heavy=heavy,
        llm_light=light,
        tts=MeteredTts(adapters.tts, ledger),
        images=MeteredImages(adapters.images, ledger),
    )
