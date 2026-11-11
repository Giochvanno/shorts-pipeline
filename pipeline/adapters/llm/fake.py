"""Fake LLM: deterministic, offline, free. Same request -> same answer.

Each task has a responder that produces a plausible answer for that task.
Chaos options (random failures, latency, garbage) are added with the runner.
"""

import hashlib
import json
from collections.abc import Callable, Mapping

from pipeline.adapters.llm.base import LlmRequest, LlmResponse

Responder = Callable[[LlmRequest], str]


def _pick(options: list[str], seed: str) -> str:
    digest = int(hashlib.sha256(seed.encode()).hexdigest(), 16)
    return options[digest % len(options)]


def fake_script(request: LlmRequest) -> str:
    topic = request.tags.get("topic", "this fact")
    structure = request.tags.get("structure", "question_reveal")
    place = _pick(["a quiet museum hall", "an open field at dawn", "a small workshop"], topic)
    scenes = [
        ("This one sounds made up.", f"wide view of {place}"),
        (
            f"Today we look at {topic}, and the real answer is simpler than the myth.",
            "an old map spread on a wooden table",
        ),
        (
            "Scientists measured it again and again, and the numbers kept telling the same story.",
            "a row of glass jars catching soft light",
        ),
        (
            "The surprising part is how close to you this happens, almost every single day.",
            "a city street seen from a rooftop",
        ),
        (
            "Once you know it, you start noticing it everywhere you look around you.",
            "a window with rain drops and a calm sky",
        ),
        (
            "Next time you see it, you will know exactly what is going on there.",
            f"wide view of {place} at dusk",
        ),
    ]
    return json.dumps(
        {
            "structure": structure,
            "scenes": [{"narration": n, "visual": v} for n, v in scenes],
            "sources": ["Fake Encyclopedia, entry 42"],
        }
    )


DEFAULT_RESPONDERS: Mapping[str, Responder] = {"script": fake_script}


class FakeLlm:
    def __init__(
        self,
        model: str = "fake-llm",
        responders: Mapping[str, Responder] | None = None,
    ) -> None:
        self._model = model
        self._responders = dict(DEFAULT_RESPONDERS if responders is None else responders)
        self.calls: list[LlmRequest] = []  # tests assert on this: "was it called twice?"

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.calls.append(request)
        responder = self._responders.get(request.task)
        if responder is None:
            raise ValueError(f"FakeLlm has no responder for task {request.task!r}")
        text = responder(request)
        return LlmResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            input_tokens=len((request.system + request.prompt).split()),
            output_tokens=len(text.split()),
            cost_usd=0.0,
        )
