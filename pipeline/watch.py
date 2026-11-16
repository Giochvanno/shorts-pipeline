"""Watch Claude at work in the pipeline: what it is given, how it writes, what is done with it.

    uv run python -m pipeline.watch --topic "why octopuses have three hearts"
        Live: Claude writes one script while you watch (claude -p, on the subscription;
        it uses Pro limits like a real run, costs no money).
    uv run python -m pipeline.watch --video 10
        Replay: what Claude was given and answered for a video already made. Free.

Four parts, in pipeline order:
1. What Claude gets: the model, the system prompt (the series rules) and the request.
2. Claude writes: its answer as it arrives (live), or the saved one (replay).
3. What the pipeline checks: every rule the script step applies, and its verdict.
4. What goes to FLUX: the image prompt each scene's visual becomes.

Nothing is written: no video, no state, no files. A window onto the script step.
"""

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TextIO

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.llm import LlmRequest, LlmResponse
from pipeline.adapters.llm.claude_cli import PROVIDER as CLAUDE_CLI
from pipeline.adapters.llm.claude_cli import ClaudeCliLlm
from pipeline.artifacts import read_meta
from pipeline.config import SeriesConfig, load_channel, load_series
from pipeline.state import StateStore, UnknownVideoError
from pipeline.steps import script as script_step
from pipeline.steps import visuals
from pipeline.steps.base import StepError

DB_NAME = "pipeline.db"
WIDTH = 78


class Streaming(Protocol):
    """What watching needs from an LLM: the answer piece by piece (ClaudeCliLlm.stream)."""

    @property
    def model(self) -> str: ...

    def stream(
        self,
        request: LlmRequest,
        *,
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None] | None = None,
    ) -> LlmResponse: ...


def _rule(out: TextIO, title: str) -> None:
    out.write(f"\n== {title} {'=' * max(3, WIDTH - len(title) - 4)}\n")


def _block(out: TextIO, label: str, text: str) -> None:
    out.write(f"--- {label} ---\n{text.strip()}\n")


def show_request(out: TextIO, model: str, structure: str, system: str, prompt: str) -> None:
    _rule(out, "1. What Claude gets")
    out.write(f"model: {model}    structure for this video: {structure}\n")
    _block(out, f"system prompt: the series rules ({len(system):,} characters)", system)
    _block(out, "request", prompt)


def show_checks(
    out: TextIO, script: script_step.Script, series: SeriesConfig, structure: str
) -> bool:
    """Print the script as the pipeline reads it and every rule; True if it would be kept."""
    _rule(out, "3. What the pipeline checks")
    for number, scene in enumerate(script.scenes, start=1):
        words = len(scene.narration.split())
        hook = ", the hook" if number == 1 else ""
        out.write(f"scene {number} ({words} words{hook}): {scene.narration}\n")
        out.write(f"    visual: {scene.visual}\n")
    out.write(f"sources: {len(script.sources)}\n")
    for source in script.sources:
        out.write(f"    - {source}\n")

    fmt = series.format
    limit = script_step.hook_word_limit(series)
    hook_words = len(script.scenes[0].narration.split()) if script.scenes else 0
    duration = script.estimated_duration_s
    checks = [
        (script.structure == structure, f"structure {script.structure!r} (asked {structure!r})"),
        (
            fmt.scenes.min <= len(script.scenes) <= fmt.scenes.max,
            f"{len(script.scenes)} scenes (allowed {fmt.scenes.min:g}-{fmt.scenes.max:g})",
        ),
        (
            fmt.duration_s.min <= duration <= fmt.duration_s.max,
            f"~{duration:.0f} s at {script_step.WORDS_PER_SECOND} words/s, {script.word_count}"
            f" words (allowed {fmt.duration_s.min:g}-{fmt.duration_s.max:g} s)",
        ),
        (hook_words <= limit, f"hook {hook_words} words (limit {limit})"),
    ]
    for number, scene in enumerate(script.scenes, start=1):
        problems = script_step.visual_problems(scene.visual, series)
        checks.append((not problems, f"scene {number} visual: {'; '.join(problems) or 'clean'}"))
    out.write("checks:\n")
    for ok, text in checks:
        out.write(f"  [{'ok' if ok else '!!'}] {text}\n")
    problems = script_step.check_script(script, series, structure)
    if problems:
        out.write("verdict: REJECTED. The pipeline would ask Claude again, saying why:\n")
        for problem in problems:
            out.write(f"    {problem}\n")
    else:
        out.write("verdict: ACCEPTED. The pipeline would save this script and go on.\n")
    return not problems


def show_image_prompts(out: TextIO, prompts: Sequence[tuple[str, str]]) -> None:
    _rule(out, "4. What goes to FLUX")
    out.write("Style from the series bible + the scene's visual, word for word:\n")
    for label, prompt in prompts:
        out.write(f"{label}: {prompt}\n")


def watch_live(
    out: TextIO,
    llm: Streaming,
    series: SeriesConfig,
    topic: str,
    *,
    video_id: int = 1,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    structure = script_step.choose_structure(series, video_id)
    system, prompt = script_step.build_prompts(series, topic, structure)
    show_request(out, llm.model, structure, system, prompt)

    _rule(out, "2. Claude writes (live)")
    state = {"thinking": False, "text": False}

    def on_thinking(chunk: str) -> None:
        if not state["thinking"]:
            out.write("--- thinking ---\n")
            state["thinking"] = True
        out.write(chunk)
        out.flush()

    def on_text(chunk: str) -> None:
        if not state["text"]:
            out.write("\n--- answer ---\n" if state["thinking"] else "")
            state["text"] = True
        out.write(chunk)
        out.flush()

    started = clock()
    request = script_step.script_request(system, prompt, topic, structure)
    try:
        response = llm.stream(request, on_text=on_text, on_thinking=on_thinking)
    except (PermanentAdapterError, TransientAdapterError) as e:
        out.write(f"\nClaude did not answer: {e}\n")
        return 1
    elapsed = clock() - started
    out.write(
        f"\n\n({elapsed:.1f} s; {response.input_tokens:,} tokens in, incl. the CLI's own"
        f" context; {response.output_tokens:,} out; ${response.cost_usd:.3f} on the"
        " subscription: counted against the budget, not billed)\n"
    )
    try:
        script = script_step.parse_script(response.text)
    except StepError as e:
        _rule(out, "3. What the pipeline checks")
        out.write(f"verdict: REJECTED before any rule: {e}\nThe pipeline would ask again.\n")
        return 0
    show_checks(out, script, series, structure)
    show_image_prompts(
        out,
        [
            (f"scene {n}", visuals.build_image_prompt(series.visual_bible, scene.visual))
            for n, scene in enumerate(script.scenes, start=1)
        ],
    )
    return 0


def watch_replay(out: TextIO, runs_dir: Path, video_id: int) -> int:
    try:
        with StateStore(runs_dir / DB_NAME) as store:
            video = store.video(video_id)
    except UnknownVideoError as e:
        out.write(f"error  {e}\n")
        return 2
    run_dir = runs_dir / str(video.id)
    try:
        script = script_step.load(run_dir)
    except StepError as e:
        out.write(f"error  {e}\n")
        return 2
    series = load_series(video.series_id)
    structure = script_step.choose_structure(series, video.id)
    system, prompt = script_step.build_prompts(series, video.topic, structure)
    meta = read_meta(run_dir / script_step.OUTPUT) or {}
    out.write(
        f"video {video.id}  {video.series_id}  {video.topic!r}\n"
        "(the prompt below is rebuilt from today's config; the script was written with the"
        " config of that day)\n"
    )
    show_request(out, str(meta.get("model", "?")), structure, system, prompt)
    _rule(out, "2. What Claude answered (saved)")
    out.write(script.model_dump_json(indent=2) + "\n")
    out.write(
        f"({meta.get('provider', '?')}, {meta.get('model', '?')},"
        f" ${float(meta.get('cost_usd') or 0):.3f})\n"
    )
    show_checks(out, script, series, structure)
    prompts = []
    for n, scene in enumerate(script.scenes):
        frame_meta = None
        for frame in sorted((run_dir / visuals.FRAMES_DIR).glob(f"scene_{n:02d}.*")):
            frame_meta = read_meta(frame) or frame_meta
        if frame_meta and "prompt" in frame_meta:
            label = f"scene {n + 1} (sent, seed {frame_meta.get('seed')})"
            prompts.append((label, str(frame_meta["prompt"])))
        else:
            built = visuals.build_image_prompt(series.visual_bible, scene.visual)
            prompts.append((f"scene {n + 1} (not drawn yet)", built))
    show_image_prompts(out, prompts)
    return 0


def _llm(choice: str) -> ClaudeCliLlm:
    models = load_channel().models.llm
    chain = models.heavy if choice == "heavy" else models.light
    if chain.primary.provider != CLAUDE_CLI:
        raise SystemExit(
            "watching needs the claude_cli provider for the llm in config/channel.yaml"
        )
    return ClaudeCliLlm(chain.primary.model, timeout_s=models.timeout_s)


def main(
    argv: list[str] | None = None,
    *,
    out: TextIO | None = None,
    llm_factory: Callable[[str], Streaming] = _llm,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.watch", description=__doc__.split("\n")[0]
    )
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--topic", help="live: Claude writes a script for this topic")
    what.add_argument("--video", type=int, help="replay: a video already made")
    parser.add_argument("--series", default="fact-of-day")
    parser.add_argument("--model", choices=["heavy", "light"], default="heavy")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    stream = out or sys.stdout
    if args.video is not None:
        return watch_replay(stream, args.runs_dir, args.video)
    return watch_live(stream, llm_factory(args.model), load_series(args.series), args.topic)


if __name__ == "__main__":
    raise SystemExit(main())
