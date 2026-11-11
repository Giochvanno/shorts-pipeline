"""Step `script` on the fake LLM: output, validation and caching."""

import json
from dataclasses import replace

import pytest

from pipeline.adapters.llm import LlmRequest
from pipeline.adapters.llm.fake import FakeLlm, fake_script
from pipeline.config import SeriesConfig
from pipeline.steps import script
from pipeline.steps.base import StepContext, StepError


def test_writes_a_valid_script(ctx: StepContext) -> None:
    result = script.run(ctx)

    saved = script.load(ctx.run_dir)
    assert result.outputs == (ctx.run_dir / "script.json",)
    assert not result.cached
    assert script.check_script(saved, ctx.series, saved.structure) == []
    assert saved.structure in ctx.series.format.structures


def test_second_run_is_cached_and_free(ctx: StepContext, fake_llm: FakeLlm) -> None:
    script.run(ctx)
    again = script.run(ctx)

    assert again.cached
    assert again.cost_usd == 0
    assert len(fake_llm.calls) == 1  # the LLM was not called (or paid) twice


def test_new_revision_regenerates(ctx: StepContext, fake_llm: FakeLlm) -> None:
    script.run(ctx)
    redo = script.run(replace(ctx, revision=1))  # "redo script" button

    assert not redo.cached
    assert len(fake_llm.calls) == 2


def test_prompt_carries_series_rules(ctx: StepContext, fake_llm: FakeLlm) -> None:
    script.run(ctx)
    request = fake_llm.calls[0]

    assert ctx.series.script.tone.split()[0] in request.system
    assert "health advice" in request.system  # hard-banned topic reaches the model
    assert "hands" in request.system  # hard-forbidden visual reaches the model
    assert ctx.topic in request.prompt
    assert "$" not in request.system + request.prompt  # every placeholder was filled


def test_structures_rotate_between_videos(series: SeriesConfig) -> None:
    picked = [script.choose_structure(series, video_id) for video_id in range(4)]

    assert len(set(picked)) == len(series.format.structures)


def _llm_answering(text: str) -> FakeLlm:
    return FakeLlm(responders={"script": lambda _request: text})


def _fake_script_dict(ctx: StepContext) -> dict[str, object]:
    structure = script.choose_structure(ctx.series, ctx.video_id)
    request = LlmRequest(
        task="script", system="", prompt="", tags={"topic": ctx.topic, "structure": structure}
    )
    data: dict[str, object] = json.loads(fake_script(request))
    return data


def test_json_inside_code_fence_is_accepted(ctx: StepContext) -> None:
    fenced = "Here you go:\n```json\n" + json.dumps(_fake_script_dict(ctx)) + "\n```"

    result = script.run(replace(ctx, llm_heavy=_llm_answering(fenced)))

    assert not result.cached


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        pytest.param("Sorry, I cannot help with that.", "no JSON", id="prose"),
        pytest.param('{"structure": "x"}', "not a valid script", id="missing-fields"),
        pytest.param("{not json}", "not a valid script", id="broken-json"),
    ],
)
def test_garbage_answers_are_retryable_errors(ctx: StepContext, answer: str, message: str) -> None:
    with pytest.raises(StepError, match=message) as exc:
        script.run(replace(ctx, llm_heavy=_llm_answering(answer)))

    assert exc.value.retryable
    assert not (ctx.run_dir / "script.json").exists()  # nothing broken is saved


def test_script_breaking_series_rules_is_rejected(ctx: StepContext) -> None:
    data = _fake_script_dict(ctx)
    scenes = data["scenes"]
    assert isinstance(scenes, list)
    data["scenes"] = scenes[:2]  # too few scenes and far too short

    with pytest.raises(StepError, match="scenes") as exc:
        script.run(replace(ctx, llm_heavy=_llm_answering(json.dumps(data))))

    assert exc.value.retryable
    assert "expected" in str(exc.value)


def test_long_hook_is_rejected(ctx: StepContext) -> None:
    data = _fake_script_dict(ctx)
    scenes = data["scenes"]
    assert isinstance(scenes, list)
    scenes[0]["narration"] = "This is a very long opening line that no viewer will wait for."

    with pytest.raises(StepError, match="hook"):
        script.run(replace(ctx, llm_heavy=_llm_answering(json.dumps(data))))
