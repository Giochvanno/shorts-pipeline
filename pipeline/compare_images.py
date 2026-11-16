"""Draw one video's scenes with several image models and put them side by side.

    uv run python -m pipeline.compare_images --video 10 --style "..."
        The plan and its price. Free: nothing is drawn.
    uv run python -m pipeline.compare_images --video 10 --style "..." --pay
        Draws every scene with every model in `fal_models` (config/channel.yaml),
        or only those given with --models.

The video's own frames are the first row (the model and style it was made with);
each model gets a row below, all with the same prompts and the video's seed. With
--style the model rows use that style instead of the series' own, so FLUX.1 [dev]
in both rows shows what the style alone changes.

Written to runs/<video>/compare/:
    grid.png       rows: the video's frames, then one row per model; columns: scenes
    report.md      price, time and failures per model, and a score table to fill in
    results.json   every frame: model, prompt, seed, size, seconds, cost, error
    <model>/scene_NN.png   the frames, cached: a second run pays only for what is missing

A prompt variant (round 3) instead of a model comparison: --tag puts it in
runs/<video>/compare-<tag>/, --against compare adds the rows of that earlier run for
the same models (free, from disk), --colours sets the palette in words and
--lighting 2="cool moonlight" the light of one scene (for scripts written before
scenes had their own light).

Money: the plan's price must fit --max-usd (default $1) before the first call; each
call is checked again against what is left, and against the month's budget. Spend
goes into the ledger without a video: an experiment, not the cost of video N.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.images import ImageRequest, ImageResult
from pipeline.adapters.images.fal import API_KEY_ENV as FAL_KEY_ENV
from pipeline.adapters.images.fal import FalImages
from pipeline.adapters.registry import fal_images
from pipeline.artifacts import input_hash, is_cached, read_meta, write_output
from pipeline.config import ChannelConfig, ImageChain, VisualBible, load_channel, load_series
from pipeline.costs import BudgetExceeded, Ledger, MeteredImages
from pipeline.env import load_env
from pipeline.media import run_ffmpeg
from pipeline.state import StateStore, UnknownVideoError
from pipeline.steps import script as script_step
from pipeline.steps import visuals
from pipeline.steps.base import StepError
from pipeline.steps.render import font_path

DB_NAME = "pipeline.db"
COMPARE_DIR = "compare"
STEP = "compare_images"
DEFAULT_MAX_USD = 1.00
# Grid cells: a 720x1280 frame at 30 %, big enough to judge subject, text and layout.
CELL_W, CELL_H, GAP = 216, 384, 8
LABEL_H, HEADER_H = 48, 48
BACKGROUND = "0x1e1e1e"
_IMAGE_SUFFIXES = {".png", ".jpg", ".webp"}

ImagesFactory = Callable[[ImageChain, str], FalImages]


@dataclass(slots=True)
class Cell:
    scene: int
    path: Path | None = None
    cost_usd: float = 0.0
    seconds: float | None = None
    cached: bool = False
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    error: str | None = None


@dataclass(slots=True)
class Row:
    label: str
    model: str
    cells: list[Cell] = field(default_factory=list)
    price_usd: float = 0.0  # per frame, from config

    @property
    def spent_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.cells), 6)


_FOLDER = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")


def scene_lighting(items: Sequence[str], count: int) -> dict[int, str] | str:
    """--lighting "2=cool moonlight" -> {2: "cool moonlight"}; an error text if malformed."""
    found: dict[int, str] = {}
    for item in items:
        number, sep, text = item.partition("=")
        if not sep or not number.strip().isdigit() or not text.strip():
            return f"--lighting {item!r}: write it as N=light, e.g. 2=cool moonlight"
        n = int(number)
        if not 1 <= n <= count:
            return f"--lighting {item!r}: the video has scenes 1-{count}"
        found[n] = text.strip()
    return found


def earlier_row(folder: Path, model: str, label: str, tag: str, count: int) -> Row:
    """The same model's frames from an earlier run, read from disk (nothing is drawn)."""
    cells = []
    for n in range(count):
        found = sorted(
            p
            for p in (folder / model_dir(model)).glob(f"scene_{n:02d}.*")
            if p.suffix in _IMAGE_SUFFIXES
        )
        cell = Cell(scene=n, path=found[0] if found else None, cached=True)
        if not found:
            cell.error = f"no frame in {folder.name}"
        else:
            meta = read_meta(found[0]) or {}
            cell.width, cell.height, cell.seed = (
                meta.get("width"),
                meta.get("height"),
                meta.get("seed"),
            )
        cells.append(cell)
    return Row(label=f"{label} ({tag})", model=model, cells=cells)


def model_dir(model: str) -> str:
    """acme/flux/dev -> acme__flux__dev (one folder per model)."""
    return model.replace("/", "__")


def frame_key(prompt: str, seed: int, model: str, params: Mapping[str, object]) -> str:
    return input_hash(
        {
            "step": STEP,
            "prompt": prompt,
            "seed": seed,
            "size": [visuals.WIDTH, visuals.HEIGHT],
            "provider": "fal",
            "model": model,
            "params": dict(params),
        }
    )


def _own_frame(run_dir: Path, index: int) -> Path | None:
    found = sorted(
        p
        for p in (run_dir / visuals.FRAMES_DIR).glob(f"scene_{index:02d}.*")
        if p.suffix in _IMAGE_SUFFIXES
    )
    return found[0] if found else None


def baseline_row(run_dir: Path, count: int, chain: ImageChain) -> Row:
    cells = [Cell(scene=n, path=_own_frame(run_dir, n)) for n in range(count)]
    meta = next((read_meta(c.path) for c in cells if c.path is not None), None) or {}
    model = str(meta.get("model") or "?")
    known = chain.fal_models.get(model)
    for cell in cells:
        if cell.path is None:
            cell.error = "no frame in the video"
        else:
            cell.seed = (read_meta(cell.path) or {}).get("seed")
    return Row(
        label=f"as made: {known.label if known else model}, own style", model=model, cells=cells
    )


def _draw(
    images: MeteredImages, prompt: str, seed: int
) -> tuple[ImageResult | None, float, str | None]:
    """One frame, one retry with the next seed on a transient failure (as the visuals step)."""
    error: str | None = None
    for attempt in range(2):
        started = time.monotonic()
        try:
            result = images.generate(
                ImageRequest(
                    prompt=prompt,
                    seed=(seed + attempt) % 2**32,
                    width=visuals.WIDTH,
                    height=visuals.HEIGHT,
                )
            )
        except TransientAdapterError as e:
            error = str(e)
            continue
        except PermanentAdapterError as e:
            return None, time.monotonic() - started, str(e)
        seconds = time.monotonic() - started
        if not visuals.looks_like_image(result.image, result.format):
            error = "invalid image data"
            continue
        return result, seconds, None
    return None, 0.0, error


def draw_rows(
    out: TextIO,
    rows: Sequence[tuple[Row, FalImages, Mapping[str, object]]],
    prompts: Sequence[str],
    seed: int,
    compare_dir: Path,
    ledger: Ledger,
    max_usd: float,
) -> float:
    """Draw every missing frame, cheapest risk first: one model at a time, scene by scene."""
    spent = 0.0
    stop: str | None = None
    with ledger.scope(None, STEP):
        for row, client, params in rows:
            metered = MeteredImages(client, ledger)
            for index, prompt in enumerate(prompts):
                path = compare_dir / model_dir(row.model) / f"scene_{index:02d}.png"
                key = frame_key(prompt, seed, row.model, params)
                cell = Cell(scene=index)
                row.cells.append(cell)
                cached = next(
                    (p for p in path.parent.glob(f"scene_{index:02d}.*") if is_cached(p, key)),
                    None,
                )
                if cached is not None:
                    meta = read_meta(cached) or {}
                    cell.path, cell.cached = cached, True
                    cell.width, cell.height = meta.get("width"), meta.get("height")
                    cell.seed, cell.seconds = meta.get("seed"), meta.get("seconds")
                    out.write(f"  {row.label:<14} scene {index + 1}  cached\n")
                    continue
                if stop is None and spent + row.price_usd > max_usd + 1e-9:
                    stop = f"not drawn: the next frame would pass the ${max_usd:.2f} limit"
                if stop is not None:
                    cell.error = stop
                    continue
                try:
                    result, seconds, error = _draw(metered, prompt, seed)
                except BudgetExceeded as e:
                    stop = f"not drawn: {e}"
                    cell.error = stop
                    out.write(f"  {stop}\n")
                    continue
                if result is None:
                    cell.error = error
                    out.write(f"  {row.label:<14} scene {index + 1}  FAILED  {error}\n")
                    continue
                spent += result.cost_usd
                path = path.with_suffix(f".{result.format}")
                write_output(
                    path,
                    result.image,
                    key,
                    {
                        "scene": index,
                        "prompt": prompt,
                        "seed": result.seed,
                        "provider": result.provider,
                        "model": result.model,
                        "width": result.width,
                        "height": result.height,
                        "seconds": round(seconds, 2),
                        "cost_usd": result.cost_usd,
                        "licence": result.licence,
                    },
                )
                cell.path, cell.cost_usd, cell.seconds = path, result.cost_usd, seconds
                cell.width, cell.height, cell.seed = result.width, result.height, result.seed
                out.write(
                    f"  {row.label:<14} scene {index + 1}  ${result.cost_usd:.3f}"
                    f"  {seconds:4.1f} s  {result.width}x{result.height}\n"
                )
    return spent


# --- grid ------------------------------------------------------------------------------


def _cell_filter(source: str, name: str) -> str:
    fit = (
        f"scale={CELL_W}:{CELL_H}:force_original_aspect_ratio=increase,crop={CELL_W}:{CELL_H},"
        if source.startswith("[")
        else ""
    )
    return (
        f"{source}{fit}setsar=1,format=rgb24,"
        f"pad={CELL_W + GAP}:{CELL_H + GAP}:{GAP // 2}:{GAP // 2}:color={BACKGROUND}[{name}]"
    )


def grid_filters(rows: Sequence[Sequence[int | None]]) -> list[str]:
    """Filter graph for rows of cells; each cell is an input number or None (no image).

    Text comes from files next to the font (cwd = scratch dir): nothing to escape, on
    Windows either (the render step does the same).
    """
    parts: list[str] = []
    for r, cells in enumerate(rows):
        for c, source in enumerate(cells):
            name = f"r{r}c{c}"
            if source is None:
                blank = (
                    f"color=c=0x3a3a3a:s={CELL_W}x{CELL_H}:d=1,"
                    "drawtext=fontfile=fonts/font.ttf:textfile=missing.txt:expansion=none"
                    ":fontsize=26:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2,"
                )
                parts.append(_cell_filter(blank, name))
            else:
                parts.append(_cell_filter(f"[{source}:v]", name))
        joined = "".join(f"[r{r}c{c}]" for c in range(len(cells)))
        stack = f"hstack=inputs={len(cells)}" if len(cells) > 1 else "null"
        parts.append(
            f"{joined}{stack},pad=iw:ih+{LABEL_H}:0:{LABEL_H}:color={BACKGROUND},"
            f"drawtext=fontfile=fonts/font.ttf:textfile=label_{r}.txt:expansion=none"
            f":fontsize=28:fontcolor=white:x=12:y=({LABEL_H}-text_h)/2[row{r}]"
        )
    joined = "".join(f"[row{r}]" for r in range(len(rows)))
    stack = f"vstack=inputs={len(rows)}" if len(rows) > 1 else "null"
    numbers = ",".join(
        f"drawtext=fontfile=fonts/font.ttf:text={c + 1}:fontsize=30:fontcolor=white"
        f":x={c * (CELL_W + GAP) + (CELL_W + GAP) // 2}-text_w/2:y=({HEADER_H}-text_h)/2"
        for c in range(len(rows[0]))
    )
    parts.append(
        f"{joined}{stack},pad=iw:ih+{HEADER_H}:0:{HEADER_H}:color={BACKGROUND},{numbers}[grid]"
    )
    return parts


def build_grid(rows: Sequence[Row], output: Path, font: Path) -> None:
    args: list[str] = []
    sources: list[list[int | None]] = []
    count = 0
    for row in rows:
        line: list[int | None] = []
        for cell in row.cells:
            if cell.path is None:
                line.append(None)
            else:
                args += ["-i", str(cell.path.resolve())]
                line.append(count)
                count += 1
        sources.append(line)
    tmp = output.with_name(output.stem + ".tmp" + output.suffix)
    args += [
        "-filter_complex", ";".join(grid_filters(sources)),
        "-map", "[grid]", "-frames:v", "1", "-update", "1",
        str(tmp.resolve()),
    ]  # fmt: skip
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grid_") as scratch_name:
        scratch = Path(scratch_name)
        (scratch / "fonts").mkdir()
        shutil.copyfile(font, scratch / "fonts" / "font.ttf")
        (scratch / "missing.txt").write_text("no image", encoding="utf-8")
        for r, row in enumerate(rows):
            (scratch / f"label_{r}.txt").write_text(row.label, encoding="utf-8")
        run_ffmpeg(args, cwd=scratch, timeout_s=120)
    tmp.replace(output)


# --- report ----------------------------------------------------------------------------


def _seconds(row: Row) -> str:
    timed = [c.seconds for c in row.cells if c.seconds is not None]
    return f"{sum(timed) / len(timed):.1f}" if timed else "-"


SCORE_COLUMNS = (
    "Scene",
    "Model",
    "Right subject",
    "No text or caption",
    "Subject in upper 2/3",
    "Style matches",
    "Light matches the scene",
    "Good first frame for video",
    "Notes",
)
STYLE_COLUMNS = ("Row", "One style across the frames", "Notes")
Kept = dict[tuple[str, tuple[str, ...]], dict[str, str]]


def _header(columns: Sequence[str]) -> str:
    return "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns)


def _row(columns: Sequence[str], key: Sequence[str], filled: dict[str, str]) -> str:
    cells = [*key, *(filled.get(name, "") for name in columns[len(key) :])]
    return "|" + "|".join(f" {c} " if c else " " for c in cells) + "|"


def filled_cells(path: Path) -> Kept:
    """Cells already filled in a report.md, by table and row: a re-run keeps the scores.

    Matched by column name, so a table that gained a column keeps its old cells.
    """
    if not path.exists():
        return {}
    kept: Kept = {}
    header: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if not "".join(cells).strip("-: "):
            continue  # the |---|---| line
        width = 2 if header[0] == SCORE_COLUMNS[0] else 1
        values = {
            name: value for name, value in zip(header[width:], cells[width:], strict=False) if value
        }
        if values:
            kept.setdefault((header[0], tuple(cells[:width])), {}).update(values)
    return kept


def write_report(
    path: Path,
    *,
    video_id: int,
    topic: str,
    style: str,
    seed: int,
    rows: Sequence[Row],
    prompts: Sequence[str],
    spent_usd: float,
    max_usd: float,
    settings: Sequence[str] = (),
) -> None:
    baseline, models = rows[0], rows[1:]
    kept = filled_cells(path)
    lines = [
        f"# Image models on video {video_id}: {topic}",
        "",
        f"- First row: the video's own frames ({baseline.model}, the style it was made with).",
        f"- Other rows: style `{style}`, seed {seed} for every model, the same prompts.",
        *(f"- {line}" for line in settings),
        f"- Spent in this run: ${spent_usd:.3f} of the ${max_usd:.2f} limit.",
        "",
        "| Model | Price per frame | Frames | Failed | Spent (this run) | Seconds per frame |",
        "|---|---|---|---|---|---|",
    ]
    for row in models:
        made = sum(1 for c in row.cells if c.path is not None)
        failed = sum(1 for c in row.cells if c.error)
        lines.append(
            f"| {row.label} | ${row.price_usd:.3f} | {made} | {failed} | ${row.spent_usd:.3f}"
            f" | {_seconds(row)} |"
        )
    failures = [(row, c) for row in models for c in row.cells if c.error]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- {row.label}, scene {c.scene + 1}: {c.error}" for row, c in failures]
    lines += [
        "",
        "## Style across the video",
        "",
        "Per row: do its frames look like one video? yes / no / partly.",
        "",
        _header(STYLE_COLUMNS),
    ]
    lines += [
        _row(STYLE_COLUMNS, (row.label,), kept.get((STYLE_COLUMNS[0], (row.label,)), {}))
        for row in rows
    ]
    lines += [
        "",
        "## Scores",
        "",
        "Per frame: yes / no / partly. A good first frame: a clear subject, a readable"
        " foreground and background (a video model animates exactly this).",
        "",
        _header(SCORE_COLUMNS),
    ]
    for n in range(len(prompts)):
        for row in rows:
            key = (str(n + 1), row.label)
            lines.append(_row(SCORE_COLUMNS, key, kept.get((SCORE_COLUMNS[0], key), {})))
    lines += ["", "## Prompts (model rows)", ""]
    lines += [f"{n + 1}. {prompt}" for n, prompt in enumerate(prompts)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(path: Path, rows: Sequence[Row], prompts: Sequence[str], seed: int) -> None:
    data = [
        {
            "row": row.label,
            "model": row.model,
            "scene": c.scene + 1,
            "path": str(c.path) if c.path else None,
            "prompt": prompts[c.scene] if r else None,
            "seed": c.seed if c.seed is not None else (seed if r else None),
            "width": c.width,
            "height": c.height,
            "seconds": round(c.seconds, 2) if c.seconds is not None else None,
            "cost_usd": c.cost_usd,
            "cached": c.cached,
            "error": c.error,
        }
        for r, row in enumerate(rows)
        for c in row.cells
    ]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --- command ---------------------------------------------------------------------------


def _models(chain: ImageChain, wanted: str | None) -> list[str] | str:
    if not wanted:
        return list(chain.fal_models)
    chosen = [m.strip() for m in wanted.split(",") if m.strip()]
    unknown = [m for m in chosen if m not in chain.fal_models]
    if unknown:
        return f"not in fal_models (config/channel.yaml): {', '.join(unknown)}"
    return chosen


def _plan(
    out: TextIO,
    chain: ImageChain,
    clients: Sequence[tuple[str, FalImages]],
    prompts: Sequence[str],
    seed: int,
    compare_dir: Path,
) -> float:
    out.write(f"\n{'model':<16}{'per frame':>10}{'to draw':>9}{'cached':>8}{'price':>9}\n")
    total = 0.0
    for model, client in clients:
        params = chain.fal_models[model].params
        price = client.price(visuals.WIDTH, visuals.HEIGHT)
        folder = compare_dir / model_dir(model)
        cached = sum(
            1
            for n, prompt in enumerate(prompts)
            if any(
                is_cached(p, frame_key(prompt, seed, model, params))
                for p in folder.glob(f"scene_{n:02d}.*")
            )
        )
        cost = price * (len(prompts) - cached)
        total += cost
        out.write(
            f"{chain.fal_models[model].label:<16}{f'${price:.3f}':>10}"
            f"{len(prompts) - cached:>9}{cached:>8}{f'${cost:.3f}':>9}\n"
        )
    return round(total, 6)


def compare(
    out: TextIO,
    *,
    channel: ChannelConfig,
    runs_dir: Path,
    video_id: int,
    style: str | None,
    models: str | None,
    max_usd: float,
    pay: bool,
    images_factory: ImagesFactory,
    colours: str | None = None,
    lighting: Sequence[str] = (),
    tag: str | None = None,
    against: str | None = None,
) -> int:
    chain = channel.models.images
    chosen = _models(chain, models)
    if isinstance(chosen, str):
        out.write(f"error  {chosen}\n")
        return 2
    for name, value in (("--tag", tag), ("--against", against)):
        if value is not None and not _FOLDER.fullmatch(value):
            out.write(f"error  {name} {value!r}: use lowercase letters, digits and '-'\n")
            return 2
    try:
        with StateStore(runs_dir / DB_NAME) as store:
            video = store.video(video_id)
    except UnknownVideoError as e:
        out.write(f"error  {e}\n")
        return 2
    run_dir = runs_dir / str(video.id)
    try:
        scenes = script_step.load(run_dir).scenes
    except StepError as e:
        out.write(f"error  {e}\n")
        return 2
    relit = scene_lighting(lighting, len(scenes))
    if isinstance(relit, str):
        out.write(f"error  {relit}\n")
        return 2
    series = load_series(video.series_id)
    bible: VisualBible = series.visual_bible
    if style:
        bible = bible.model_copy(update={"style_prefix": style})
    if colours:
        bible = bible.model_copy(update={"colours": colours})
    lights = [relit.get(n) or scene.lighting for n, scene in enumerate(scenes, start=1)]
    prompts = [
        visuals.build_image_prompt(bible, scene.visual, light)
        for scene, light in zip(scenes, lights, strict=True)
    ]
    seed = visuals.video_seed(video.id, 0)
    compare_dir = run_dir / (f"{COMPARE_DIR}-{tag}" if tag else COMPARE_DIR)
    clients = [(model, images_factory(chain, model)) for model in chosen]
    settings = [
        f"Colours: {bible.colours.strip() if bible.colours else ', '.join(bible.palette)}.",
        "Lighting: "
        + "; ".join(
            f"scene {n}: {light or bible.lighting + ' (default)'}"
            for n, light in enumerate(lights, start=1)
        )
        + ".",
    ]
    if against:
        settings.append(f"Rows marked ({against}) are frames of that earlier run, for comparison.")

    out.write(
        f"video {video.id}  {video.series_id}  {video.topic!r}\n"
        f"style: {bible.style_prefix.strip()}\nseed:  {seed} (the video's, for every model)\n"
        + "".join(f"{line}\n" for line in settings)
    )
    total = _plan(out, chain, clients, prompts, seed, compare_dir)
    out.write(f"total ${total:.3f} of the ${max_usd:.2f} limit\n")
    if total > max_usd + 1e-9:
        out.write("error  the plan costs more than --max-usd: drop a model or raise the limit\n")
        return 2
    if not pay:
        out.write(f"\nnothing drawn: add --pay to draw (up to ${total:.3f})\n")
        return 0

    rows: list[Row] = [baseline_row(run_dir, len(scenes), chain)]
    work: list[tuple[Row, FalImages, Mapping[str, object]]] = []
    for model, client in clients:
        profile = chain.fal_models[model]
        price = client.price(visuals.WIDTH, visuals.HEIGHT)
        if against:
            rows.append(earlier_row(run_dir / against, model, profile.label, against, len(scenes)))
            rows[-1].price_usd = price
        row = Row(label=profile.label, model=model, price_usd=price)
        rows.append(row)
        work.append((row, client, profile.params))

    out.write("\n")
    with StateStore(runs_dir / DB_NAME) as store:
        spent = draw_rows(
            out, work, prompts, seed, compare_dir, Ledger(store, channel.budget), max_usd
        )
    compare_dir.mkdir(parents=True, exist_ok=True)
    grid = compare_dir / "grid.png"
    build_grid(rows, grid, font_path(series.captions.font))
    report = compare_dir / "report.md"
    write_report(
        report,
        video_id=video.id,
        topic=video.topic,
        style=bible.style_prefix.strip(),
        seed=seed,
        rows=rows,
        prompts=prompts,
        spent_usd=spent,
        max_usd=max_usd,
        settings=settings,
    )
    write_results(compare_dir / "results.json", rows, prompts, seed)
    failed = sum(1 for row in rows[1:] for c in row.cells if c.error)
    out.write(
        f"\nspent ${spent:.3f}" + (f", {failed} frame(s) not drawn" if failed else "") + "\n"
        f"grid:   {grid}\nreport: {report}\n"
    )
    return 0


def main(
    argv: list[str] | None = None,
    *,
    out: TextIO | None = None,
    images_factory: ImagesFactory = fal_images,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.compare_images", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--video", type=int, required=True, help="a video already made")
    parser.add_argument("--style", help="style for the model rows (default: the series' own)")
    parser.add_argument("--models", help="comma-separated model ids (default: all fal_models)")
    parser.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    parser.add_argument("--pay", action="store_true", help="draw: this spends money")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--colours", help="the palette in words for the prompts")
    parser.add_argument(
        "--lighting", action="append", default=[], help="N=light for scene N (repeatable)"
    )
    parser.add_argument("--tag", help="a prompt variant: writes to compare-<tag>/")
    parser.add_argument("--against", help="add the rows of an earlier run, e.g. compare")
    args = parser.parse_args(argv)
    stream = out or sys.stdout
    if args.pay:
        loaded = load_env(Path.cwd() / ".env")
        if loaded:
            stream.write(f"secrets from .env: {', '.join(loaded)}\n")
        if not os.environ.get(FAL_KEY_ENV):
            stream.write(f"error  {FAL_KEY_ENV} is not set: add it to .env\n")
            return 2
    return compare(
        stream,
        channel=load_channel(),
        runs_dir=args.runs_dir,
        video_id=args.video,
        style=args.style,
        models=args.models,
        max_usd=args.max_usd,
        pay=args.pay,
        images_factory=images_factory,
        colours=args.colours,
        lighting=args.lighting,
        tag=args.tag,
        against=args.against,
    )


if __name__ == "__main__":
    raise SystemExit(main())
