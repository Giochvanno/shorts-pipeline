"""FLUX on fal, on a fake HTTP transport: request shape, money, image data, errors.

Error bodies follow fal's error docs: model-level errors are a `detail` list with
a `type`, request-level ones a flat `detail` plus `error_type`. The live test
checks the real thing.
"""

import base64
import json
from typing import Any

import pytest

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.http import HttpRequest, HttpResponse
from pipeline.adapters.images import ImageRequest
from pipeline.adapters.images.fake import gradient_png
from pipeline.adapters.images.fal import (
    FalImages,
    aspect_ratio,
    billed_megapixels,
    classify_failure,
    png_size,
)

KEY = "fal-secret-key-123"
MODEL = "acme/flux-test"
PNG = gradient_png(720, 1280, (46, 64, 87), (232, 216, 176))
REQUEST = ImageRequest(prompt="a fern unrolling at dawn", seed=42, width=720, height=1280)


def answer(**overrides: Any) -> bytes:
    data: dict[str, Any] = {
        "images": [
            {
                "url": "data:image/png;base64," + base64.b64encode(PNG).decode(),
                "width": 720,
                "height": 1280,
                "content_type": "image/png",
            }
        ],
        "seed": 42,
        "has_nsfw_concepts": [False],
        "prompt": REQUEST.prompt,
    }
    data.update(overrides)
    return json.dumps(data).encode()


class FakeHttp:
    def __init__(self, *responses: HttpResponse | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[HttpRequest] = []

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def ok(body: bytes | None = None) -> HttpResponse:
    return HttpResponse(200, answer() if body is None else body)


def make(http: FakeHttp, *, api_key: str | None = KEY, **profile: Any) -> FalImages:
    settings: dict[str, Any] = {
        "usd_per_megapixel": 0.025,
        "params": {"enable_safety_checker": True},
        **profile,
    }
    return FalImages(
        MODEL,
        api_key=api_key,
        licence="test licence",
        timeout_s=33,
        base_url="https://fal.test",
        transport=http,
        **settings,
    )


# --- request ---------------------------------------------------------------------------


def test_request_carries_prompt_size_seed_and_key_in_header_only() -> None:
    http = FakeHttp(ok())
    make(http).generate(REQUEST)

    (sent,) = http.requests
    assert sent.method == "POST"
    assert sent.url == f"https://fal.test/{MODEL}"
    assert sent.headers["Authorization"] == f"Key {KEY}"
    assert KEY not in sent.url
    assert sent.timeout_s == 33
    assert json.loads(sent.body or b"") == {
        "prompt": REQUEST.prompt,
        "image_size": {"width": 720, "height": 1280},
        "seed": 42,
        "output_format": "png",
        "sync_mode": True,
        "enable_safety_checker": True,  # the model's own params, from config
    }


def test_an_aspect_ratio_model_gets_the_ratio_and_no_pixel_size() -> None:
    http = FakeHttp(ok())
    make(http, size="aspect_ratio", params={"safety_tolerance": "4"}).generate(REQUEST)

    body = json.loads(http.requests[0].body or b"")
    assert body["aspect_ratio"] == "9:16"
    assert "image_size" not in body
    assert body["safety_tolerance"] == "4"


def test_a_model_without_a_seed_gets_none_and_reports_none() -> None:
    http = FakeHttp(ok(answer(seed=None)))

    result = make(http, takes_seed=False).generate(REQUEST)

    assert "seed" not in json.loads(http.requests[0].body or b"")
    assert result.seed is None  # not the request's seed: nothing could repeat this frame


def test_model_params_cannot_replace_what_the_adapter_sets() -> None:
    http = FakeHttp(ok())
    make(http, params={"prompt": "something else", "sync_mode": False}).generate(REQUEST)

    body = json.loads(http.requests[0].body or b"")
    assert body["prompt"] == REQUEST.prompt
    assert body["sync_mode"] is True


@pytest.mark.parametrize(
    ("width", "height", "ratio"), [(720, 1280, "9:16"), (1080, 1920, "9:16"), (768, 1344, "4:7")]
)
def test_aspect_ratio_is_reduced(width: int, height: int, ratio: str) -> None:
    assert aspect_ratio(width, height) == ratio


def test_exactly_one_price() -> None:
    with pytest.raises(ValueError, match="exactly one price"):
        FalImages(MODEL, api_key=KEY, licence="l")
    with pytest.raises(ValueError, match="exactly one price"):
        FalImages(MODEL, api_key=KEY, licence="l", usd_per_megapixel=0.025, usd_per_image=0.04)


@pytest.mark.parametrize("model", ["", "flux", "../admin", "acme/flux?x=1", "acme/ flux", "Acme/X"])
def test_a_model_id_cannot_bend_the_url(model: str) -> None:
    with pytest.raises(ValueError, match="model id"):
        FalImages(model, api_key=KEY, usd_per_megapixel=0.025, licence="l")


# --- success ---------------------------------------------------------------------------


def test_returns_the_inline_image_its_size_seed_licence_and_cost() -> None:
    result = make(FakeHttp(ok())).generate(REQUEST)

    assert result.image == PNG
    assert result.format == "png"
    assert (result.width, result.height) == (720, 1280)
    assert result.seed == 42
    assert result.provider == "fal"
    assert result.model == MODEL
    assert result.licence == "test licence"
    assert result.cost_usd == pytest.approx(0.025)  # 0.92 MP -> one megapixel


@pytest.mark.parametrize(
    ("width", "height", "megapixels"),
    [(720, 1280, 1), (768, 1344, 2), (1024, 1024, 2), (1080, 1920, 3), (512, 512, 1)],
)
def test_megapixels_are_rounded_up_never_below_the_bill(
    width: int, height: int, megapixels: int
) -> None:
    assert billed_megapixels(width, height) == megapixels


def test_jpeg_data_is_recorded_as_jpg() -> None:
    jpeg = b"\xff\xd8\xff" + b"\x00" * 2000
    image = {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}
    result = make(FakeHttp(ok(answer(images=[image])))).generate(REQUEST)

    assert result.format == "jpg"
    assert result.image == jpeg


def test_a_cdn_link_is_downloaded_once_without_the_key() -> None:
    image = {"url": "https://cdn.fal.test/out.png", "content_type": "image/png"}
    http = FakeHttp(ok(answer(images=[image])), HttpResponse(200, PNG))

    result = make(http).generate(REQUEST)

    assert result.image == PNG
    download = http.requests[1]
    assert (download.method, download.url) == ("GET", "https://cdn.fal.test/out.png")
    assert "Authorization" not in download.headers


def test_missing_size_and_seed_fall_back_to_the_request() -> None:
    jpeg = b"\xff\xd8\xff" + b"\x00" * 2000  # no PNG header to read the size from
    image = {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}
    result = make(FakeHttp(ok(answer(images=[image], seed=None)))).generate(REQUEST)

    assert (result.width, result.height, result.seed) == (720, 1280, 42)


def test_the_size_comes_from_the_png_itself_and_a_per_image_price_ignores_it() -> None:
    tall = gradient_png(768, 1344, (0, 0, 0), (255, 255, 255))  # Gemini's "9:16"
    image = {"url": "data:image/png;base64," + base64.b64encode(tall).decode()}  # no sizes
    http = FakeHttp(ok(answer(images=[image])))

    result = make(http, size="aspect_ratio", usd_per_megapixel=None, usd_per_image=0.04).generate(
        REQUEST
    )

    assert (result.width, result.height) == (768, 1344)
    assert result.cost_usd == pytest.approx(0.04)  # 1.03 MP, still one price per image


def test_the_png_header_wins_over_sizes_in_the_json() -> None:
    image = {
        "url": "data:image/png;base64," + base64.b64encode(PNG).decode(),
        "width": 1024,
        "height": 1024,
    }
    result = make(FakeHttp(ok(answer(images=[image])))).generate(REQUEST)

    assert (result.width, result.height) == (720, 1280)


def test_png_size_reads_only_pngs() -> None:
    assert png_size(PNG) == (720, 1280)
    assert png_size(b"\xff\xd8\xff" + b"\x00" * 100) is None
    assert png_size(PNG[:20]) is None


# --- refused before paying -------------------------------------------------------------


@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_is_permanent_without_a_request(key: str | None) -> None:
    http = FakeHttp()

    with pytest.raises(PermanentAdapterError, match=r"FAL_KEY .*\.env"):
        make(http, api_key=key).generate(REQUEST)
    assert http.requests == []


def test_empty_prompt_is_refused_without_a_request() -> None:
    http = FakeHttp()

    with pytest.raises(ValueError, match="empty"):
        make(http).generate(ImageRequest(prompt=" ", seed=1, width=720, height=1280))
    assert http.requests == []


def test_key_never_shows_in_repr() -> None:
    assert KEY not in repr(make(FakeHttp()))


# --- errors ------------------------------------------------------------------------------


def model_error(type_: str, msg: str = "msg") -> bytes:
    return json.dumps(
        {"detail": [{"loc": ["body", "prompt"], "msg": msg, "type": type_, "url": "u"}]}
    ).encode()


def request_error(error_type: str, detail: str = "detail") -> bytes:
    return json.dumps({"detail": detail, "error_type": error_type}).encode()


PERMANENT = [
    (401, b'{"detail": "Unauthorized"}', "rejected the API key"),
    (403, b'{"detail": "User is locked. Reason: Exhausted balance."}', "balance"),
    (402, b"", "balance"),
    (404, b'{"detail": "Application not found"}', "does not know this model"),
    (422, model_error("content_policy_violation", "flagged"), "content filter"),
    (422, model_error("value_error"), "rejected the request"),
    (422, model_error("less_than_equal"), "rejected the request"),
    (400, request_error("bad_request"), "rejected the request"),
]


@pytest.mark.parametrize(("status", "body", "hint"), PERMANENT)
def test_errors_a_human_must_fix_are_permanent(status: int, body: bytes, hint: str) -> None:
    with pytest.raises(PermanentAdapterError, match=hint) as exc:
        make(FakeHttp(HttpResponse(status, body))).generate(REQUEST)

    assert f"HTTP {status}" in str(exc.value)
    assert KEY not in str(exc.value)


TRANSIENT = [
    (422, model_error("no_media_generated")),  # another seed may well work
    (429, b'{"detail": "Too many requests"}'),
    (500, model_error("internal_server_error")),
    (503, request_error("runner_scheduling_failure")),
    (504, request_error("request_timeout", "Request timed out")),
    (502, b"<html>Bad Gateway</html>"),
]


@pytest.mark.parametrize(("status", "body"), TRANSIENT)
def test_hiccups_are_transient(status: int, body: bytes) -> None:
    with pytest.raises(TransientAdapterError, match=f"HTTP {status}"):
        make(FakeHttp(HttpResponse(status, body))).generate(REQUEST)


def test_error_text_names_the_type_and_the_field() -> None:
    error = classify_failure(422, model_error("less_than_equal", "must be <= 2048"))

    assert "less_than_equal" in str(error)
    assert "body.prompt: must be <= 2048" in str(error)


@pytest.mark.parametrize(
    ("failure", "match"),
    [(ConnectionResetError("reset"), "network error"), (TimeoutError(), "no answer within 33 s")],
)
def test_network_failures_are_transient(failure: Exception, match: str) -> None:
    with pytest.raises(TransientAdapterError, match=match):
        make(FakeHttp(failure)).generate(REQUEST)


def test_a_frame_the_safety_checker_blanked_is_transient_not_an_image() -> None:
    with pytest.raises(TransientAdapterError, match="safety checker"):
        make(FakeHttp(ok(answer(has_nsfw_concepts=[True])))).generate(REQUEST)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (b"<html>not json</html>", "unreadable JSON"),
        (b"[1]", "without an object"),
        (answer(images=[]), "no image"),
        (answer(images=[{"width": 720}]), "without data"),
        (answer(images=[{"url": "data:image/png;base64,***"}]), "broken base64"),
        (answer(images=[{"url": "data:image/png;base64,"}]), "empty image"),
        (answer(images=[{"url": "http://insecure.test/x.png"}]), "neither data nor https"),
    ],
)
def test_a_broken_200_answer_is_transient(body: bytes, match: str) -> None:
    with pytest.raises(TransientAdapterError, match=match):
        make(FakeHttp(ok(body))).generate(REQUEST)


def test_a_failed_download_is_transient() -> None:
    image = {"url": "https://cdn.fal.test/out.png"}
    http = FakeHttp(ok(answer(images=[image])), HttpResponse(404, b"gone"))

    with pytest.raises(TransientAdapterError, match="download failed: HTTP 404"):
        make(http).generate(REQUEST)
