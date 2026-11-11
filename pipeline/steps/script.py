"""Step `script`: topic + series config -> runs/<id>/script.json, checked against the series rules.

The LLM is asked for JSON; the answer is parsed and validated here in code.
A bad answer (wrong length, too many scenes, long hook) is a retryable error,
so the runner can ask again instead of passing a broken script downstream.
"""

import json
import math
import re
from pathlib import Path
from string import Template

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pipeline.adapters.llm import LlmRequest
from pipeline.artifacts import input_hash, is_cached, write_output
from pipeline.config import SeriesConfig
from pipeline.steps.base import StepContext, StepError, StepResult

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
OUTPUT = "script.json"

# Rough speaking rate used to check length before paying for TTS.
# The real duration is measured after the voice step.
WORDS_PER_SECOND = 2.5


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    narration: str = Field(min_length=1)
    visual: str = Field(min_length=1)


class Script(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structure: str
    scenes: list[Scene] = Field(min_length=1)
    sources: list[str] = Field(min_length=1)

    @property
    def narration(self) -> str:
        return " ".join(scene.narration for scene in self.scenes)

    @property
    def word_count(self) -> int:
        return len(self.narration.split())

    @property
    def estimated_duration_s(self) -> float:
        return self.word_count / WORDS_PER_SECOND


def hook_word_limit(series: SeriesConfig) -> int:
    return math.floor(series.format.hook_max_s * WORDS_PER_SECOND) + 1


def choose_structure(series: SeriesConfig, video_id: int) -> str:
    """Rotate structures so consecutive videos never share one."""
    structures = series.format.structures
    return structures[video_id % len(structures)]


def build_prompts(series: SeriesConfig, topic: str, structure: str) -> tuple[str, str]:
    fmt = series.format
    system = Template((PROMPTS_DIR / "script_system.md").read_text(encoding="utf-8"))
    user = Template((PROMPTS_DIR / "script_user.md").read_text(encoding="utf-8"))
    values = {
        "series_title": series.title,
        "tone": series.script.tone,
        "angle": series.script.angle,
        "rules": "\n".join(f"  - {rule}" for rule in series.script.rules),
        "min_words": str(math.ceil(fmt.duration_s.min * WORDS_PER_SECOND)),
        "max_words": str(math.floor(fmt.duration_s.max * WORDS_PER_SECOND)),
        "min_s": f"{fmt.duration_s.min:g}",
        "max_s": f"{fmt.duration_s.max:g}",
        "min_scenes": f"{fmt.scenes.min:g}",
        "max_scenes": f"{fmt.scenes.max:g}",
        "hook_words": str(hook_word_limit(series)),
        "structure": structure,
        "banned_topics": ", ".join(series.script.banned_topics),
        "forbidden": ", ".join(series.visual_bible.forbidden),
        "topic": topic,
    }
    return system.substitute(values), user.substitute(values)


def parse_script(text: str) -> Script:
    """Accept bare JSON or JSON wrapped in prose/code fences; reject anything else."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise StepError("LLM answer contains no JSON object", retryable=True)
    try:
        return Script.model_validate(json.loads(match.group(0)))
    except json.JSONDecodeError as e:
        raise StepError(
            f"LLM answer is not a valid script: bad JSON ({e.msg})", retryable=True
        ) from e
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'root'}: {err['msg']}"
            for err in e.errors(include_url=False)
        )
        raise StepError(f"LLM answer is not a valid script: {problems}", retryable=True) from e


def check_script(script: Script, series: SeriesConfig, structure: str) -> list[str]:
    """Series rules a script must satisfy. Empty list = OK."""
    fmt = series.format
    problems: list[str] = []
    if script.structure != structure:
        problems.append(f"structure is {script.structure!r}, expected {structure!r}")
    if not fmt.scenes.min <= len(script.scenes) <= fmt.scenes.max:
        problems.append(
            f"{len(script.scenes)} scenes, expected {fmt.scenes.min:g}-{fmt.scenes.max:g}"
        )
    duration = script.estimated_duration_s
    if not fmt.duration_s.min <= duration <= fmt.duration_s.max:
        problems.append(
            f"~{duration:.0f} s ({script.word_count} words),"
            f" expected {fmt.duration_s.min:g}-{fmt.duration_s.max:g} s"
        )
    hook_words = len(script.scenes[0].narration.split())
    if hook_words > hook_word_limit(series):
        problems.append(f"hook has {hook_words} words, limit is {hook_word_limit(series)}")
    return problems


def run(ctx: StepContext) -> StepResult:
    output = ctx.run_dir / OUTPUT
    structure = choose_structure(ctx.series, ctx.video_id)
    system, prompt = build_prompts(ctx.series, ctx.topic, structure)
    key = input_hash(
        {
            "step": "script",
            "system": system,  # includes tone, rules, limits: any edit invalidates the cache
            "prompt": prompt,
            "provider": ctx.llm_heavy.provider,
            "model": ctx.llm_heavy.model,
            "revision": ctx.revision,
        }
    )
    if is_cached(output, key):
        return StepResult(outputs=(output,), cost_usd=0.0, cached=True)

    response = ctx.llm_heavy.complete(
        LlmRequest(
            task="script",
            system=system,
            prompt=prompt,
            tags={"topic": ctx.topic, "structure": structure},
        )
    )
    script = parse_script(response.text)
    problems = check_script(script, ctx.series, structure)
    if problems:
        raise StepError("script breaks series rules: " + "; ".join(problems), retryable=True)

    write_output(
        output,
        script.model_dump_json(indent=2).encode("utf-8"),
        key,
        {
            "provider": response.provider,
            "model": response.model,
            "cost_usd": response.cost_usd,
            "words": script.word_count,
            "estimated_duration_s": round(script.estimated_duration_s, 1),
        },
    )
    return StepResult(outputs=(output,), cost_usd=response.cost_usd, cached=False)


def load(run_dir: Path) -> Script:
    """Read the script written by `run` (used by later steps)."""
    path = run_dir / OUTPUT
    try:
        return Script.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise StepError(f"{path} not found: run the script step first", retryable=False) from e
