"""pipeline.compare_images on a fake fal: the plan, the money guards, the grid, the report.

The fal clients are the real adapter on a fake HTTP transport, built from the real
`fal_models` in config: request shapes and prices are the ones a paid run would use.
"""

import base64
import io
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pipeline.adapters.http import HttpRequest, HttpResponse
from pipeline.adapters.images.fake import gradient_png
from pipeline.adapters.images.fal import API_KEY_ENV as FAL_KEY_ENV
from pipeline.adapters.images.fal import FalImages
from pipeline.adapters.registry import fal_images
from pipeline.cli import main as pipeline_main
from pipeline.compare_images import (
    CELL_H,
    CELL_W,
    GAP,
    HEADER_H,
    LABEL_H,
    grid_filters,
    main,
    model_dir,
)
from pipeline.config import ImageChain, load_channel
from pipeline.costs import Ledger
from pipeline.media import probe
from pipeline.state import CostRecord, StateStore
from pipeline.steps import script as script_step

CHAIN = load_channel().models.images
MODELS = list(CHAIN.fal_models)
PNG_PIXELS = gradient_png(720, 1280, (46, 64, 87), (232, 216, 176))
PNG_GEMINI = gradient_png(768, 1344, (46, 64, 87), (232, 216, 176))
Reply = Callable[[HttpRequest, int], HttpResponse | None]
PAY = ("--pay", "--max-usd", "5")  # every model on every scene costs more than the $1 default


class FakeFal:
    """Answers like fal: a PNG of the asked size, or Gemini's 768x1344 for a ratio."""

    def __init__(self, reply: Reply | None = None) -> None:
        self.requests: list[HttpRequest] = []
        self.reply = reply

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if self.reply is not None:
            special = self.reply(request, len(self.requests))
            if special is not None:
                return special
        body = json.loads(request.body or b"")
        png = PNG_GEMINI if "aspect_ratio" in body else PNG_PIXELS
        image = {"url": "data:image/png;base64," + base64.b64encode(png).decode()}
        answer: dict[str, object] = {"images": [image]}
        if "seed" in body:
            answer["seed"] = body["seed"]
        return HttpResponse(200, json.dumps(answer).encode())

    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(r.body or b"") for r in self.requests]

    def model_of(self, request: HttpRequest) -> str:
        return request.url.removeprefix("https://fal.run/")


def factory(fake: FakeFal) -> Callable[[ImageChain, str], FalImages]:
    return lambda chain, model: fal_images(chain, model, transport=fake)


@pytest.fixture(scope="module")
def made_video(tmp_path_factory: pytest.TempPathFactory, ffmpeg: None) -> Path:
    """One fake video, made once; each test gets its own copy."""
    runs = tmp_path_factory.mktemp("made")
    assert pipeline_main(["run", "--series", "fact-of-day", "--runs-dir", str(runs)]) == 0
    return runs


@pytest.fixture
def runs(made_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # no real .env is ever read
    monkeypatch.setenv(FAL_KEY_ENV, "fal-test-key")
    copy = tmp_path / "runs"
    shutil.copytree(made_video, copy)
    return copy


def scenes(runs: Path) -> int:
    return len(script_step.load(runs / "1").scenes)


def run(runs: Path, fake: FakeFal, *extra: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(
        ["--video", "1", "--runs-dir", str(runs), *extra], out=out, images_factory=factory(fake)
    )
    return code, out.getvalue()


def plan_price(count: int) -> float:
    return sum(
        count * (p.usd_per_image or p.usd_per_megapixel or 0) for p in CHAIN.fal_models.values()
    )


# --- plan ------------------------------------------------------------------------------


def test_without_pay_it_shows_the_price_and_draws_nothing(runs: Path) -> None:
    fake = FakeFal()

    code, text = run(runs, fake, "--max-usd", "5")

    assert code == 0
    assert fake.requests == []
    assert not (runs / "1" / "compare").exists()
    assert f"total ${plan_price(scenes(runs)):.3f} of the $5.00 limit" in text
    assert "add --pay" in text
    for profile in CHAIN.fal_models.values():
        assert profile.label in text


def test_a_plan_over_the_limit_is_refused_before_any_call(runs: Path) -> None:
    fake = FakeFal()

    code, text = run(runs, fake, "--pay", "--max-usd", "0.05")

    assert code == 2
    assert fake.requests == []
    assert "more than --max-usd" in text


def test_unknown_models_and_videos_are_errors(runs: Path) -> None:
    assert run(runs, FakeFal(), "--models", "acme/nope")[0] == 2
    out = io.StringIO()
    assert main(["--video", "9", "--runs-dir", str(runs)], out=out) == 2
    assert "9" in out.getvalue()


def test_paying_needs_the_key(runs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FAL_KEY_ENV)
    fake = FakeFal()

    code, text = run(runs, fake, *PAY)

    assert code == 2
    assert FAL_KEY_ENV in text
    assert fake.requests == []


# --- paid run --------------------------------------------------------------------------


def test_every_model_draws_every_scene_with_the_same_prompt_and_seed(runs: Path) -> None:
    fake = FakeFal()
    count = scenes(runs)

    code, _ = run(runs, fake, *PAY, "--style", "Clean digital natural history painting")

    assert code == 0
    assert len(fake.requests) == count * len(MODELS)
    bodies = fake.bodies()
    assert {fake.model_of(r) for r in fake.requests} == set(MODELS)
    for n in range(count):
        same_scene = bodies[n::count]  # one model after another, scene by scene
        assert len({b["prompt"] for b in same_scene}) == 1
        assert len({b["seed"] for b in same_scene if "seed" in b}) == 1
    for request, body in zip(fake.requests, bodies, strict=True):
        assert ("seed" in body) == CHAIN.fal_models[fake.model_of(request)].takes_seed
    assert all(
        str(b["prompt"]).startswith("Clean digital natural history painting") for b in bodies
    )
    for model in MODELS:
        assert len(list((runs / "1" / "compare" / model_dir(model)).glob("scene_*.png"))) == count


def test_the_grid_has_a_row_per_model_under_the_videos_own_frames(runs: Path) -> None:
    count = scenes(runs)

    run(runs, FakeFal(), *PAY)

    info = probe(runs / "1" / "compare" / "grid.png")
    rows = 1 + len(MODELS)
    assert info.width == count * (CELL_W + GAP)
    assert info.height == HEADER_H + rows * (LABEL_H + CELL_H + GAP)


def test_report_and_results_hold_prices_and_an_empty_score_table(runs: Path) -> None:
    count = scenes(runs)

    run(runs, FakeFal(), *PAY)

    report = (runs / "1" / "compare" / "report.md").read_text(encoding="utf-8")
    for model, profile in CHAIN.fal_models.items():
        price = profile.usd_per_image or profile.usd_per_megapixel or 0
        assert f"| {profile.label} | ${price:.3f} | {count} | 0 | ${price * count:.3f} |" in report
        assert f"| 1 | {profile.label} | | | | | | |" in report
        assert f"| {profile.label} | | |" in report  # its style-across-the-video row
        assert model in json.dumps(
            json.loads((runs / "1" / "compare" / "results.json").read_text(encoding="utf-8"))
        )
    assert "as made: fake-images, own style" in report
    results = json.loads((runs / "1" / "compare" / "results.json").read_text(encoding="utf-8"))
    assert len(results) == count * (1 + len(MODELS))
    gemini = [r for r in results if r["model"] == "fal-ai/nano-banana"]
    assert {(r["width"], r["height"]) for r in gemini} == {(768, 1344)}  # read from the PNG


def test_scores_filled_in_by_hand_survive_a_re_run(runs: Path) -> None:
    run(runs, FakeFal(), *PAY)
    report = runs / "1" / "compare" / "report.md"
    label = CHAIN.fal_models[MODELS[0]].label
    old_layout = (  # an earlier report, before the first-frame column existed
        "| Scene | Model | Right subject | No text or caption | Notes |\n|---|---|---|---|---|\n"
        f"| 2 | {label} | partly | no | signature bottom right |\n"
        "\n| Row | One style across the frames | Notes |\n|---|---|---|\n"
        f"| {label} | yes | |\n"
    )
    report.write_text(old_layout, encoding="utf-8")

    run(runs, FakeFal(), *PAY)

    text = report.read_text(encoding="utf-8")
    assert f"| 2 | {label} | partly | no | | | | signature bottom right |" in text
    assert f"| {label} | yes | |" in text
    assert f"| 1 | {label} | | | | | | |" in text  # nothing invented for the rest


def test_spend_goes_to_the_month_not_to_the_video(runs: Path) -> None:
    count = scenes(runs)
    with StateStore(runs / "pipeline.db") as store:
        before = Ledger(store).spend(1)

    run(runs, FakeFal(), *PAY)

    with StateStore(runs / "pipeline.db") as store:
        after = Ledger(store).spend(1)
    assert after.video_usd == before.video_usd
    assert after.month_usd - before.month_usd == pytest.approx(plan_price(count))
    assert after.month_calls - before.month_calls == count * len(MODELS)


def test_a_second_run_pays_only_for_what_is_missing(runs: Path) -> None:
    run(runs, FakeFal(), *PAY)
    (runs / "1" / "compare" / model_dir(MODELS[1]) / "scene_00.png").unlink()
    fake = FakeFal()

    code, text = run(runs, fake, *PAY)

    assert code == 0
    assert [fake.model_of(r) for r in fake.requests] == [MODELS[1]]
    assert "cached" in text


def test_a_refused_frame_is_reported_and_the_others_are_drawn(runs: Path) -> None:
    refused = json.dumps({"detail": [{"type": "content_policy_violation", "msg": "no"}]})

    def policy(request: HttpRequest, _n: int) -> HttpResponse | None:
        body = json.loads(request.body or b"")
        first_prompt = script_step.load(runs / "1").scenes[0].visual.strip().rstrip(".")
        if request.url.endswith(MODELS[-1]) and first_prompt in str(body["prompt"]):
            return HttpResponse(422, refused.encode())
        return None

    fake = FakeFal(policy)
    count = scenes(runs)

    code, text = run(runs, fake, *PAY)

    assert code == 0
    assert len(fake.requests) == count * len(MODELS)  # a refusal is final: no retry
    assert "FAILED" in text
    report = (runs / "1" / "compare" / "report.md").read_text(encoding="utf-8")
    label = CHAIN.fal_models[MODELS[-1]].label
    assert f"- {label}, scene 1: " in report
    assert (runs / "1" / "compare" / "grid.png").exists()  # a grey cell in its place


def test_a_transient_failure_is_tried_once_more_with_the_next_seed(runs: Path) -> None:
    fake = FakeFal(lambda _r, n: HttpResponse(503, b"busy") if n == 1 else None)

    code, _ = run(runs, fake, *PAY, "--models", MODELS[0])

    assert code == 0
    first, second = fake.bodies()[:2]
    assert first["prompt"] == second["prompt"]
    assert second["seed"] == (int(str(first["seed"])) + 1) % 2**32
    assert len(fake.requests) == scenes(runs) + 1


def test_a_spent_month_stops_drawing(runs: Path) -> None:
    with StateStore(runs / "pipeline.db") as store:
        store.record_cost(
            CostRecord(
                video_id=None,
                step="elsewhere",
                provider="fal",
                model="x",
                billing="api",
                unit="images",
                units=1,
                cost_usd=load_channel().budget.max_monthly_usd,
                ts=datetime.now(UTC).isoformat(timespec="milliseconds"),
            )
        )
    fake = FakeFal()

    code, text = run(runs, fake, *PAY)

    assert code == 0
    assert fake.requests == []
    assert "not drawn: budget" in text


# --- grid graph ------------------------------------------------------------------------


def test_grid_filters_put_a_grey_cell_where_an_image_is_missing() -> None:
    graph = ";".join(grid_filters([[0, 1], [None, 2]]))

    assert "[0:v]" in graph and "[2:v]" in graph and "[1:v]" in graph
    assert graph.count("textfile=missing.txt") == 1
    assert graph.count("hstack=inputs=2") == 2
    assert "vstack=inputs=2" in graph
