"""Cost ledger and budget guard, in two accounts.

    cash                 money per call: fal, ElevenLabs, the Anthropic API (a fallback).
                         The video, image and month limits apply to it.
    subscription_credit  what `claude -p` reports a call would cost on the API. The Claude
                         subscription pays it from a monthly credit: not cash, but finite
                         (when it runs out, scripts stop), so it has its own monthly limit
                         and a warning at a share of it. Never part of a video's cash limit.
    fake                 offline adapters, counted like cash: the budget can be tested
                         without spending.

Cash + credit is the full price of a video, the figure payback is measured against.

Every completed adapter call becomes one row in the `costs` table (provider, model,
account, units, dollars). Before each call the guard checks the limits of the account
the call is billed to. A reached limit stops the call before it is made, and the runner
pauses the video (paused_budget, not failed: nothing broke, and waiting does not add
money); raise the limit, then resume.

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
from pathlib import Path

from pipeline.adapters.images import ImageClient, ImageRequest, ImageResult
from pipeline.adapters.llm import LlmClient, LlmRequest, LlmResponse
from pipeline.adapters.registry import Adapters
from pipeline.adapters.stt import Transcriber, Transcript
from pipeline.adapters.tts import TtsClient, TtsRequest, TtsResult
from pipeline.config import BudgetConfig
from pipeline.state import CostRecord, StateStore

Clock = Callable[[], datetime]
Warn = Callable[[str], None]

CASH, CREDIT, FAKE = "cash", "subscription_credit", "fake"
CASH_LIKE = (CASH, FAKE)
# Billed through the Claude subscription's credit, not per call in cash.
SUBSCRIPTION_PROVIDERS = frozenset({"claude_cli"})


class BudgetExceeded(Exception):
    """A spending limit is reached: the paid call was not made."""


def billing_for(provider: str) -> str:
    if provider == "fake":
        return FAKE
    return CREDIT if provider in SUBSCRIPTION_PROVIDERS else CASH


def month_start(now: datetime) -> datetime:
    """Start of the calendar month in UTC (no tz database needed on Windows)."""
    return now.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


@dataclass(frozen=True, slots=True)
class Spend:
    video_cash: float
    video_credit: float
    video_image_cash: float
    month_cash: float
    month_credit: float
    month_calls: int

    @property
    def video_full(self) -> float:
        return self.video_cash + self.video_credit

    @property
    def month_full(self) -> float:
        return self.month_cash + self.month_credit


class Ledger:
    def __init__(
        self,
        store: StateStore,
        budget: BudgetConfig | None = None,
        clock: Clock = lambda: datetime.now(UTC),
        warn: Warn | None = None,
    ) -> None:
        self._store = store
        self._budget = budget
        self._clock = clock
        self._warn = warn
        self._video_id: int | None = None
        self._step: str | None = None

    @contextmanager
    def scope(self, video_id: int | None, step: str) -> Iterator[None]:
        """Attribute every call inside the block to this video and step.

        No video (an experiment such as compare_images): the month's limits still apply,
        no video's limits do.
        """
        previous = (self._video_id, self._step)
        self._video_id, self._step = video_id, step
        try:
            yield
        finally:
            self._video_id, self._step = previous

    def spend(self, video_id: int | None = None) -> Spend:
        since = month_start(self._clock()).isoformat()
        month_cash, cash_calls = self._store.spent(CASH_LIKE, since=since)
        month_credit, credit_calls = self._store.spent((CREDIT,), since=since)
        video = video_id if video_id is not None else self._video_id
        video_cash = video_credit = video_image_cash = 0.0
        if video is not None:
            video_cash = self._store.spent(CASH_LIKE, video_id=video)[0]
            video_credit = self._store.spent((CREDIT,), video_id=video)[0]
            video_image_cash = self._store.spent(CASH_LIKE, video_id=video, unit="images")[0]
        return Spend(
            video_cash=video_cash,
            video_credit=video_credit,
            video_image_cash=video_image_cash,
            month_cash=month_cash,
            month_credit=month_credit,
            month_calls=cash_calls + credit_calls,
        )

    def check(self, provider: str, unit: str) -> None:
        """Raise BudgetExceeded if a limit of this call's account is reached.

        Called before each paid call, with the adapter's provider and the unit it bills in.
        """
        budget = self._budget
        if budget is None:
            return
        spend = self.spend()
        if billing_for(provider) == CREDIT:
            if spend.month_credit >= budget.max_subscription_credit_monthly_usd:
                raise BudgetExceeded(
                    f"budget: this month's subscription credit ${spend.month_credit:.2f}"
                    f" reached the ${budget.max_subscription_credit_monthly_usd:.2f} limit"
                )
            return
        video = self._video_id
        if video is not None and spend.video_cash >= budget.max_cash_per_video_usd:
            raise BudgetExceeded(
                f"budget: video {video} has spent ${spend.video_cash:.3f} cash"
                f" of its ${budget.max_cash_per_video_usd:.2f} limit"
            )
        if (
            video is not None
            and unit == "images"
            and spend.video_image_cash >= budget.max_image_cash_per_video_usd
        ):
            raise BudgetExceeded(
                f"budget: video {video} has spent ${spend.video_image_cash:.3f} on images"
                f" of its ${budget.max_image_cash_per_video_usd:.2f} limit"
            )
        if spend.month_cash >= budget.max_cash_monthly_usd:
            raise BudgetExceeded(
                f"budget: this month's cash ${spend.month_cash:.2f}"
                f" reached the ${budget.max_cash_monthly_usd:.2f} limit"
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
        if billing_for(provider) == CREDIT:
            self._warn_on_credit(cost_usd)

    def _warn_on_credit(self, cost_usd: float) -> None:
        """Warn once, from the call that crosses the warning share of the credit limit."""
        if self._budget is None or self._warn is None:
            return
        limit = self._budget.max_subscription_credit_monthly_usd
        line = limit * self._budget.subscription_credit_warn_at
        after = self.spend().month_credit
        if after - cost_usd < line <= after:
            self._warn(
                f"subscription credit this month: ${after:.2f} of ${limit:.2f}"
                f" ({after / limit:.0%}); at the limit scripts stop until next month"
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
        self._ledger.check(self.inner.provider, "tokens")
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
        self._ledger.check(self.inner.provider, "characters")
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
        self._ledger.check(self.inner.provider, "images")
        result = self.inner.generate(request)
        self._ledger.record(
            provider=result.provider,
            model=result.model,
            unit="images",
            units=1,
            cost_usd=result.cost_usd,
        )
        return result


class MeteredTranscriber:
    def __init__(self, inner: Transcriber, ledger: Ledger) -> None:
        self.inner, self._ledger = inner, ledger

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def transcribe(
        self, audio: Path, *, language: str, hint: str | None = None, expect: str | None = None
    ) -> Transcript:
        self._ledger.check(self.inner.provider, "seconds")
        result = self.inner.transcribe(audio, language=language, hint=hint, expect=expect)
        self._ledger.record(
            provider=result.provider,
            model=result.model,
            unit="seconds",
            units=result.duration_s,
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
        stt=MeteredTranscriber(adapters.stt, ledger),
    )
