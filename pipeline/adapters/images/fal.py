"""Real images: FLUX on fal, `POST https://fal.run/{model}` (the synchronous API).

One call = one HTTPS request; FLUX at our size answers in seconds, so the queue
API (submit, poll, fetch) would only add calls. With `sync_mode` the image comes
back inside the JSON as a data URI: no second download, nothing kept in fal's
request history. A plain URL is still accepted (downloaded once) in case a
model ignores `sync_mode`.

- The key is read from FAL_KEY (put it in `.env`) and sent only in the
  `Authorization: Key ...` header. Error texts never include it.
- Money: fal bills per megapixel, rounded up to a whole megapixel. Cost here is
  ceil(width x height / 1,000,000) x `usd_per_megapixel` from config: with the
  decimal megapixel it is never below the bill. Frames are 720x1280 (0.92 MP),
  one megapixel whichever way fal counts.
- A frame the safety checker flagged comes back black: that is a transient
  failure (the visuals step retries with another seed), not an image.
- Errors are sorted for the runner: bad key, no balance, unknown model, a
  rejected prompt or request -> PermanentAdapterError (a human must act);
  429, 5xx, timeouts, a broken answer -> TransientAdapterError.

Checked against fal's API reference for FLUX.1 [dev] and its error docs
(November 2026). 401/402/403 are not in those docs: handled by status alone.
"""

import base64
import binascii
import json
import math
import re
from typing import Any

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.http import HttpRequest, HttpResponse, Transport, send, urllib_transport
from pipeline.adapters.images.base import ImageFormat, ImageRequest, ImageResult

PROVIDER = "fal"
API_KEY_ENV = "FAL_KEY"
BASE_URL = "https://fal.run"
OUTPUT_FORMAT = "png"  # lossless key frames: video models amplify compression artefacts

_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)+")
_DATA_URI = re.compile(r"data:image/(png|jpeg|jpg|webp);base64,(.*)", re.DOTALL)
_FORMATS: dict[str, ImageFormat] = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp"}

_KEY_HINT = f"fal rejected the API key: check {API_KEY_ENV} in .env"
_BALANCE_HINT = (
    "fal refused to run: check the balance at fal.ai/dashboard/billing and the key's scope"
)
_MODEL_HINT = "fal does not know this model: check models.images in config/channel.yaml"
_POLICY_HINT = "fal's content filter rejected this scene's prompt: the script must change"
# Model-level 422 types worth another try with a new seed (per fal's docs the same
# request would fail again, but the visuals step changes the seed on each attempt).
_SEED_DEPENDENT = frozenset({"no_media_generated"})
_TRANSIENT_STATUSES = frozenset({408, 409, 425, 429})


def _short(text: str, limit: int = 300) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def _error_detail(body: bytes) -> tuple[tuple[str, ...], str]:
    """(error types, message) from a fal error body.

    Model-level errors: `{"detail": [{"loc", "msg", "type", ...}]}`.
    Request-level errors: `{"detail": "Request timed out", "error_type": "request_timeout"}`.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return (), _short(body.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        return (), _short(str(data))
    detail = data.get("detail")
    if isinstance(detail, list):
        types, parts = [], []
        for item in detail:
            if isinstance(item, dict):
                types.append(str(item.get("type") or ""))
                where = ".".join(str(x) for x in item.get("loc") or [])
                parts.append(f"{where}: {item.get('msg', '')}".strip(": "))
        return tuple(t for t in types if t), _short("; ".join(parts))
    error_type = data.get("error_type")
    return ((str(error_type),) if error_type else ()), _short(str(detail or ""))


def classify_failure(status: int, body: bytes) -> Exception:
    """Turn a non-2xx answer into the error type the runner understands."""
    types, message = _error_detail(body)
    found = set(types)
    detail = " ".join(filter(None, [f"HTTP {status}", "/".join(types), message]))
    if status == 401:
        return PermanentAdapterError(f"{_KEY_HINT} ({detail})")
    if status in (402, 403):
        return PermanentAdapterError(f"{_BALANCE_HINT} ({detail})")
    if status == 404:
        return PermanentAdapterError(f"{_MODEL_HINT} ({detail})")
    if "content_policy_violation" in found:
        return PermanentAdapterError(f"{_POLICY_HINT} ({detail})")
    if found & _SEED_DEPENDENT:
        return TransientAdapterError(f"fal: {detail}")
    if status in _TRANSIENT_STATUSES or status >= 500:
        return TransientAdapterError(f"fal: {detail}")
    if 400 <= status < 500:
        return PermanentAdapterError(f"fal rejected the request ({detail})")
    return TransientAdapterError(f"fal: unexpected answer ({detail})")


def billed_megapixels(width: int, height: int) -> int:
    return max(1, math.ceil(width * height / 1_000_000))


class FalImages:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        usd_per_megapixel: float,
        licence: str,
        timeout_s: float = 120,
        base_url: str = BASE_URL,
        transport: Transport = urllib_transport,
    ) -> None:
        if not _MODEL_ID.fullmatch(model):
            raise ValueError(f"not a fal model id: {model!r}")  # it goes into the URL path
        self._model = model
        self._api_key = api_key
        self._usd_per_megapixel = usd_per_megapixel
        self._licence = licence
        self._timeout_s = timeout_s
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    def __repr__(self) -> str:  # never show the key, even in a debugger or a log line
        return f"FalImages(model={self._model!r})"

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def model(self) -> str:
        return self._model

    def request_for(self, request: ImageRequest) -> HttpRequest:
        body = {
            "prompt": request.prompt,
            "image_size": {"width": request.width, "height": request.height},
            "seed": request.seed,
            "num_images": 1,
            "output_format": OUTPUT_FORMAT,
            "sync_mode": True,
            "enable_safety_checker": True,
        }
        return HttpRequest(
            method="POST",
            url=f"{self._base_url}/{self._model}",
            headers={
                "Authorization": f"Key {self._api_key or ''}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            body=json.dumps(body).encode(),
            timeout_s=self._timeout_s,
        )

    def generate(self, request: ImageRequest) -> ImageResult:
        if not request.prompt.strip():
            raise ValueError("nothing to draw: empty prompt")
        if not self._api_key:
            raise PermanentAdapterError(
                f"{API_KEY_ENV} is not set: add `{API_KEY_ENV}=...` to .env in the repo root"
            )
        response = send(self._transport, self.request_for(request), what="fal")
        if not 200 <= response.status < 300:
            raise classify_failure(response.status, response.body)
        data = _json_object(response)
        flagged = data.get("has_nsfw_concepts")
        if isinstance(flagged, list) and any(flagged):
            raise TransientAdapterError("fal's safety checker blanked the image: try another seed")
        image = _first_image(data)
        data_bytes, fmt = self._image_bytes(image)
        width = _int(image.get("width"), request.width)
        height = _int(image.get("height"), request.height)
        seed = _int(data.get("seed"), request.seed)
        return ImageResult(
            image=data_bytes,
            format=fmt,
            width=width,
            height=height,
            seed=seed,
            provider=PROVIDER,
            model=self._model,
            cost_usd=round(billed_megapixels(width, height) * self._usd_per_megapixel, 6),
            licence=self._licence,
        )

    def _image_bytes(self, image: dict[str, Any]) -> tuple[bytes, ImageFormat]:
        url = image.get("url")
        if not isinstance(url, str) or not url:
            raise TransientAdapterError("fal returned an image without data")
        inline = _DATA_URI.fullmatch(url)
        if inline:
            try:
                data = base64.b64decode(inline.group(2), validate=True)
            except (binascii.Error, ValueError) as e:
                raise TransientAdapterError(f"fal returned broken base64 image: {e}") from e
            fmt = _FORMATS[inline.group(1)]
        elif url.startswith("https://"):
            # The CDN link needs no key: never send it to another host.
            got = send(
                self._transport,
                HttpRequest("GET", url, timeout_s=self._timeout_s),
                what="fal image download",
            )
            if got.status != 200:
                raise TransientAdapterError(f"fal image download failed: HTTP {got.status}")
            data = got.body
            kind = str(image.get("content_type") or got.header("content-type") or "")
            fmt = _FORMATS.get(kind.removeprefix("image/").split(";")[0], "png")
        else:
            raise TransientAdapterError("fal returned an image link that is neither data nor https")
        if not data:
            raise TransientAdapterError("fal returned an empty image")
        return data, fmt


def _json_object(response: HttpResponse) -> dict[str, Any]:
    try:
        data = json.loads(response.body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise TransientAdapterError(f"fal returned unreadable JSON: {e}") from e
    if not isinstance(data, dict):
        raise TransientAdapterError("fal returned JSON without an object")
    return data


def _first_image(data: dict[str, Any]) -> dict[str, Any]:
    images = data.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise TransientAdapterError("fal returned no image")
    return images[0]


def _int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and value > 0 else default
