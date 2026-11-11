"""LLM port: what every LLM adapter (fake, Claude CLI, Anthropic API) must provide."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LlmRequest:
    task: str  # "script", "critic", ... - used for logs, costs and by the fake
    system: str
    prompt: str
    max_tokens: int = 2000
    # Context such as topic or structure. Real adapters ignore it; the fake uses it.
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LlmResponse:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LlmClient(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: LlmRequest) -> LlmResponse: ...
