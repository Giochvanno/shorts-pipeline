"""Live checks against fal with your key. Never run in CI.

    uv run pytest tests/live/test_fal_live.py -m live -s

- A wrong key is refused before anything runs: free.
- One frame in the series style, 720x1280: one megapixel, $0.025 for FLUX.1 [dev].
  Saved to runs/live/fal.png to look at; compare the charge on fal's billing page.

It prints the raw HTTP status and error body, so docs and reality can be compared.
"""

import os
from pathlib import Path

import pytest

from pipeline.adapters.errors import PermanentAdapterError
from pipeline.adapters.http import HttpRequest, HttpResponse, urllib_transport
from pipeline.adapters.images import ImageRequest
from pipeline.adapters.images.fal import API_KEY_ENV, FalImages, billed_megapixels
from pipeline.adapters.registry import fal_images
from pipeline.config import load_channel, load_series
from pipeline.env import load_env
from pipeline.steps.visuals import HEIGHT, WIDTH, build_image_prompt

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[2]
SCENE = "A fern frond slowly unrolling on a mossy forest floor in morning light"


class Recorder:
    """The real transport, keeping the last answer for inspection."""

    def __init__(self) -> None:
        self.last: HttpResponse | None = None

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.last = urllib_transport(request)
        return self.last


def _images(api_key: str, transport: Recorder, monkeypatch: pytest.MonkeyPatch) -> FalImages:
    monkeypatch.setenv(API_KEY_ENV, api_key)
    chain = load_channel().models.images
    return fal_images(chain, chain.primary.model, transport=transport)


def _request() -> ImageRequest:
    prompt = build_image_prompt(load_series("fact-of-day").visual_bible, SCENE)
    return ImageRequest(prompt=prompt, seed=1234, width=WIDTH, height=HEIGHT)


def test_wrong_key_is_permanent_and_free(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder()

    with pytest.raises(PermanentAdapterError, match="API key") as exc:
        _images("fal-definitely-not-a-real-key", recorder, monkeypatch).generate(_request())
    assert recorder.last is not None
    print(f"\nHTTP {recorder.last.status}: {recorder.last.body[:300]!r}\n-> {exc.value}")


def test_one_frame_in_the_series_style(monkeypatch: pytest.MonkeyPatch) -> None:
    load_env(REPO / ".env")
    key = os.environ.get(API_KEY_ENV)
    if not key:
        pytest.skip(f"{API_KEY_ENV} is not set: add it to .env")
    recorder = Recorder()

    result = _images(key, recorder, monkeypatch).generate(_request())

    path = REPO / "runs" / "live" / "fal.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.image)
    assert recorder.last is not None
    print(
        f"\nlive fal: {result.model}, {result.width}x{result.height},"
        f" {billed_megapixels(result.width, result.height)} MP, ${result.cost_usd:.3f},"
        f" seed {result.seed}, {len(result.image) // 1024} KB\n"
        f"request-id: {recorder.last.header('x-fal-request-id')}\nsaved: {path}"
    )
    assert result.format == "png"
    assert (result.width, result.height) == (WIDTH, HEIGHT)
    assert result.seed == 1234
