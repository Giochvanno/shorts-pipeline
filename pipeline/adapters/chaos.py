"""Chaos mode: wrap any adapter so it randomly fails, stalls or returns garbage.

Real providers misbehave in exactly these three ways (timeouts and 5xx, slow
responses, malformed output). With a fixed seed the chaos is reproducible, so a
failing chaos test can be replayed exactly.

Each wrapper keeps stats; tests use `passed` (answers delivered intact) to prove
that every good answer is paid for once, however many retries happen around it.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from pipeline.adapters.errors import TransientAdapterError
from pipeline.adapters.images import ImageClient, ImageRequest, ImageResult
from pipeline.adapters.llm import LlmClient, LlmRequest, LlmResponse
from pipeline.adapters.tts import TtsClient, TtsRequest, TtsResult

Sleep = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class ChaosConfig:
    fail_rate: float = 0.2  # raise TransientAdapterError
    garbage_rate: float = 0.1  # return a malformed answer
    max_latency_s: float = 0.0  # random extra delay per call
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.fail_rate <= 1 and 0 <= self.garbage_rate <= 1):
            raise ValueError("rates must be between 0 and 1")
        if self.fail_rate + self.garbage_rate > 1:
            raise ValueError("fail_rate + garbage_rate must not exceed 1")


@dataclass(slots=True)
class ChaosStats:
    calls: int = 0
    failures: int = 0
    garbage: int = 0
    passed: int = 0  # answers delivered intact


class _Chaos:
    """Shared dice: decide per call whether to fail, corrupt or pass through."""

    def __init__(self, config: ChaosConfig, salt: int, sleep: Sleep) -> None:
        self.config = config
        self.stats = ChaosStats()
        self._rng = random.Random(config.seed * 1_000 + salt)
        self._sleep = sleep

    def roll(self, what: str) -> str:
        """Returns "fail", "garbage" or "pass"; raises for "fail"."""
        self.stats.calls += 1
        if self.config.max_latency_s > 0:
            self._sleep(self._rng.uniform(0, self.config.max_latency_s))
        dice = self._rng.random()
        if dice < self.config.fail_rate:
            self.stats.failures += 1
            raise TransientAdapterError(f"chaos: simulated {what} outage")
        if dice < self.config.fail_rate + self.config.garbage_rate:
            self.stats.garbage += 1
            return "garbage"
        self.stats.passed += 1
        return "pass"

    def choice(self, options: list[str]) -> str:
        return self._rng.choice(options)


class ChaosLlm(_Chaos):
    GARBAGE: ClassVar[list[str]] = [
        "I'm sorry, I can't help with that right now.",
        '{"structure": "question_reveal", "scenes": [{"narration": "cut off mid',
        '{"unexpected": "shape"}',
    ]

    def __init__(self, inner: LlmClient, config: ChaosConfig, sleep: Sleep) -> None:
        super().__init__(config, salt=1, sleep=sleep)
        self.inner = inner

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(self, request: LlmRequest) -> LlmResponse:
        outcome = self.roll("LLM")
        response = self.inner.complete(request)
        if outcome == "garbage":
            text = self.choice(self.GARBAGE)
            return LlmResponse(
                text=text,
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=len(text.split()),
                cost_usd=response.cost_usd,  # garbage is still billed
            )
        return response


class ChaosTts(_Chaos):
    def __init__(self, inner: TtsClient, config: ChaosConfig, sleep: Sleep) -> None:
        super().__init__(config, salt=2, sleep=sleep)
        self.inner = inner

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def synthesize(self, request: TtsRequest) -> TtsResult:
        outcome = self.roll("TTS")
        result = self.inner.synthesize(request)
        if outcome == "garbage":
            return TtsResult(
                audio=b"",
                format=result.format,
                duration_s=0.0,
                characters=result.characters,
                provider=result.provider,
                model=result.model,
                cost_usd=result.cost_usd,
            )
        return result


class ChaosImages(_Chaos):
    def __init__(self, inner: ImageClient, config: ChaosConfig, sleep: Sleep) -> None:
        super().__init__(config, salt=3, sleep=sleep)
        self.inner = inner

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def generate(self, request: ImageRequest) -> ImageResult:
        outcome = self.roll("images")
        result = self.inner.generate(request)
        if outcome == "garbage":
            return ImageResult(
                image=b"\x00" * 512,
                format=result.format,
                width=result.width,
                height=result.height,
                seed=result.seed,
                provider=result.provider,
                model=result.model,
                cost_usd=result.cost_usd,
                licence=result.licence,
            )
        return result
