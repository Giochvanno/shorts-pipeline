"""pipeline.watch: the window onto the script step, live (fake streaming Claude) and replay."""

import io
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from pipeline.adapters.llm import LlmRequest, LlmResponse
from pipeline.adapters.llm.fake import fake_script
from pipeline.cli import main as pipeline_main
from pipeline.config import load_series
from pipeline.steps import script as script_step
from pipeline.watch import main, watch_live


class StreamingFake:
    """Answers like fake_script, handed over in small pieces as Claude streams."""

    model = "fake-streaming"

    def __init__(self, answer: Callable[[LlmRequest], str] = fake_script) -> None:
        self.answer = answer
        self.requests: list[LlmRequest] = []

    def stream(
        self,
        request: LlmRequest,
        *,
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None] | None = None,
    ) -> LlmResponse:
        self.requests.append(request)
        if on_thinking is not None:
            on_thinking("Hook first. ")
        text = self.answer(request)
        for start in range(0, len(text), 40):
            on_text(text[start : start + 40])
        return LlmResponse(text, "fake", self.model, 25_000, 600, 0.05)


def test_live_shows_the_request_the_writing_the_checks_and_the_image_prompts() -> None:
    out = io.StringIO()
    fake = StreamingFake()

    code = watch_live(out, fake, load_series("fact-of-day"), "why honey never spoils")

    text = out.getvalue()
    assert code == 0
    for part in ("1. What Claude gets", "2. Claude writes", "3. What the pipeline", "4. What"):
        assert part in text
    assert fake.requests[0].system in text  # the exact system prompt is shown
    series = load_series("fact-of-day")
    structure = script_step.choose_structure(series, 1)
    system, prompt = script_step.build_prompts(series, "why honey never spoils", structure)
    expected = script_step.script_request(system, prompt, "why honey never spoils", structure)
    assert fake.requests == [expected]  # the very request the script step sends
    assert "--- thinking ---" in text and "Hook first." in text
    assert "verdict: ACCEPTED" in text
    assert text.count("Clean digital natural history painting") >= 5  # one image prompt per scene
    assert "25,000 tokens in" in text


def test_live_explains_a_rejected_script() -> None:
    def too_short(request: LlmRequest) -> str:
        data = json.loads(fake_script(request))
        data["scenes"] = data["scenes"][:2]
        data["scenes"][1]["visual"] = "an empty room with no furniture"
        return json.dumps(data)

    out = io.StringIO()
    watch_live(out, StreamingFake(too_short), load_series("fact-of-day"), "why honey never spoils")

    text = out.getvalue()
    assert "verdict: REJECTED" in text
    assert "[!!] 2 scenes" in text
    assert "says what is absent ('no')" in text


def test_live_explains_an_answer_that_is_not_a_script() -> None:
    out = io.StringIO()
    watch_live(
        out, StreamingFake(lambda _r: "Sorry, here is a poem."), load_series("fact-of-day"), "x"
    )

    assert "REJECTED before any rule" in out.getvalue()


def test_main_live_uses_the_chosen_model(monkeypatch: pytest.MonkeyPatch) -> None:
    chosen: list[str] = []

    def factory(model: str) -> StreamingFake:
        chosen.append(model)
        return StreamingFake()

    out = io.StringIO()
    assert (
        main(
            ["--topic", "why honey never spoils", "--model", "light"], out=out, llm_factory=factory
        )
        == 0
    )
    assert chosen == ["light"]


@pytest.mark.usefixtures("ffmpeg")
def test_replay_shows_a_made_video_and_the_prompts_flux_really_got(tmp_path: Path) -> None:
    assert (
        pipeline_main(
            ["run", "--series", "fact-of-day", "--runs-dir", str(tmp_path), "--mode", "fake"]
        )
        == 0
    )
    before = sorted(p.stat().st_mtime_ns for p in tmp_path.rglob("*"))
    out = io.StringIO()

    code = main(["--video", "1", "--runs-dir", str(tmp_path)], out=out)

    text = out.getvalue()
    assert code == 0
    assert "2. What Claude answered (saved)" in text
    assert "verdict: ACCEPTED" in text
    assert "(sent, seed " in text  # from the frames' own records, not rebuilt
    assert sorted(p.stat().st_mtime_ns for p in tmp_path.rglob("*")) == before  # writes nothing


def test_replay_of_an_unknown_video_is_an_error(tmp_path: Path) -> None:
    out = io.StringIO()

    assert main(["--video", "7", "--runs-dir", str(tmp_path)], out=out) == 2
    assert "7" in out.getvalue()
