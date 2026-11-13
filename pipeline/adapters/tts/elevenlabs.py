"""Real TTS: ElevenLabs, `POST /v1/text-to-speech/{voice_id}/with-timestamps`.

One call = one HTTPS request, answered with JSON:
`audio_base64` (MP3) plus `alignment`, the time each character is spoken.
The timings come free with the audio, so they are returned too (captions can
use them instead of, or to check, whisper).

- The key is read from ELEVENLABS_API_KEY (put it in `.env`, never in config or
  git) and sent only in the `xi-api-key` header. Error texts never include it.
- Money: ElevenLabs bills credits. The credits of each call come from the
  response header (`character-cost`; older docs name it `x-character-count`);
  if it is missing, 1 credit per character is assumed (the most a model costs
  in TTS). Dollars = credits x `usd_per_credit` from config (plan price /
  plan credits).
- Duration is measured with ffprobe, not trusted from the provider. ffprobe is
  checked before the paid request: a missing tool must not cost credits.
- Errors are sorted for the runner: bad key, no credits, unknown voice or
  model, invalid request -> PermanentAdapterError (a human must act);
  429, 5xx, network trouble, a broken answer -> TransientAdapterError.

Checked against the ElevenLabs API reference and error docs (November 2026).
"""

import base64
import binascii
import json
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.http import HttpRequest, HttpResponse, Transport, send, urllib_transport
from pipeline.adapters.tts.base import Alignment, TtsRequest, TtsResult
from pipeline.media import MediaError, find_tool, probe

PROVIDER = "elevenlabs"
API_KEY_ENV = "ELEVENLABS_API_KEY"
BASE_URL = "https://api.elevenlabs.io"
OUTPUT_FORMAT = "mp3_44100_128"  # the API default; available on every plan
CREDIT_HEADERS = ("character-cost", "x-character-count")

_KEY_HINT = f"ElevenLabs rejected the API key: check {API_KEY_ENV} in .env"
_CREDITS_HINT = "ElevenLabs credits are used up: wait for the monthly reset or upgrade the plan"
_VOICE_HINT = "ElevenLabs does not know this voice: check voice.voice_id in config/series"
_MODEL_HINT = "ElevenLabs does not offer this model to the account: check models.tts"
_ACCESS_HINT = "this ElevenLabs key may not use text-to-speech or this voice/model"

_QUOTA_CODES = frozenset({"quota_exceeded", "insufficient_credits", "payment_required"})
_VOICE_CODES = frozenset({"voice_not_found", "invalid_voice_id"})
_MODEL_CODES = frozenset({"model_not_found", "unsupported_model", "model_access_denied"})
_TRANSIENT_STATUSES = frozenset({408, 409, 425, 429})

Measure = Callable[[bytes], float]


def probe_mp3_duration(audio: bytes) -> float:
    """Exact duration of MP3 bytes, via ffprobe on a temporary file."""
    with tempfile.TemporaryDirectory(prefix="tts_", ignore_cleanup_errors=True) as tmp:
        path = Path(tmp) / "voice.mp3"
        path.write_bytes(audio)
        return probe(path).duration_s


def _error_detail(body: bytes) -> tuple[str, str]:
    """(code, message) from an ElevenLabs error body; tolerant of every shape seen in docs.

    `detail` is an object (`code`, legacy `status`, `type`, `message`), a list of
    validation errors (422), or a plain string. Non-JSON bodies (a proxy's HTML
    page) are shortened to one line.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "", _short(body.decode("utf-8", errors="replace"))
    detail = data.get("detail") if isinstance(data, dict) else data
    if isinstance(detail, dict):
        code = str(detail.get("code") or detail.get("status") or detail.get("type") or "")
        message = str(detail.get("message") or "")
        request_id = detail.get("request_id")
        if request_id:
            message = f"{message} [request_id {request_id}]"
        return code, _short(message)
    if isinstance(detail, list):  # 422: [{"loc": [...], "msg": "...", "type": "..."}]
        parts = []
        for item in detail:
            if isinstance(item, dict):
                where = ".".join(str(x) for x in item.get("loc") or [])
                parts.append(f"{where}: {item.get('msg', '')}".strip(": "))
        return "validation_error", _short("; ".join(parts))
    return "", _short(str(detail))


def classify_failure(status: int, body: bytes) -> Exception:
    """Turn a non-2xx answer into the error type the runner understands.

    The error code decides first (no credits can arrive as 401, 402 or 400);
    the HTTP status is the fallback.
    """
    code, message = _error_detail(body)
    detail = " ".join(filter(None, [f"HTTP {status}", code, message]))
    if code in _QUOTA_CODES or status == 402:
        return PermanentAdapterError(f"{_CREDITS_HINT} ({detail})")
    if status == 401:
        return PermanentAdapterError(f"{_KEY_HINT} ({detail})")
    if code in _VOICE_CODES:
        return PermanentAdapterError(f"{_VOICE_HINT} ({detail})")
    if code in _MODEL_CODES:
        return PermanentAdapterError(f"{_MODEL_HINT} ({detail})")
    if status == 403:
        return PermanentAdapterError(f"{_ACCESS_HINT} ({detail})")
    if status in _TRANSIENT_STATUSES or status >= 500:
        return TransientAdapterError(f"ElevenLabs: {detail}")
    if 400 <= status < 500:
        return PermanentAdapterError(f"ElevenLabs rejected the request ({detail})")
    # 1xx/3xx should never reach us: retry a few times, then fail with this text.
    return TransientAdapterError(f"ElevenLabs: unexpected answer ({detail})")


def _short(text: str, limit: int = 300) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def _alignment(value: Any) -> Alignment | None:
    """Parse one alignment object; None if missing or inconsistent (captions fall back)."""
    if not isinstance(value, dict):
        return None
    chars = value.get("characters")
    starts = value.get("character_start_times_seconds")
    ends = value.get("character_end_times_seconds")
    if not (isinstance(chars, list) and isinstance(starts, list) and isinstance(ends, list)):
        return None
    if not chars or not len(chars) == len(starts) == len(ends):
        return None
    try:
        start_s = tuple(float(x) for x in starts)
        end_s = tuple(float(x) for x in ends)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (*start_s, *end_s)):
        return None
    return Alignment(characters=tuple(str(c) for c in chars), starts_s=start_s, ends_s=end_s)


def _credits(response: HttpResponse) -> float | None:
    for name in CREDIT_HEADERS:
        value = response.header(name)
        if value is None:
            continue
        try:
            credits = float(value)
        except ValueError:
            continue
        if math.isfinite(credits) and credits >= 0:
            return credits
    return None


class ElevenLabsTts:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        usd_per_credit: float,
        timeout_s: float = 60,
        base_url: str = BASE_URL,
        transport: Transport = urllib_transport,
        measure: Measure | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._usd_per_credit = usd_per_credit
        self._timeout_s = timeout_s
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._measure = measure or probe_mp3_duration
        self._needs_ffprobe = measure is None

    def __repr__(self) -> str:  # never show the key, even in a debugger or a log line
        return f"ElevenLabsTts(model={self._model!r})"

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def model(self) -> str:
        return self._model

    def request_for(self, request: TtsRequest) -> HttpRequest:
        voice = quote(request.voice_id, safe="")  # the id goes into the path: no "/" or "?"
        body = {
            "text": request.text.strip(),
            "model_id": self._model,
            "voice_settings": {
                "stability": request.stability,
                "similarity_boost": request.similarity_boost,
                "speed": request.speed,
            },
        }
        return HttpRequest(
            method="POST",
            url=(
                f"{self._base_url}/v1/text-to-speech/{voice}/with-timestamps"
                f"?output_format={OUTPUT_FORMAT}"
            ),
            headers={
                "xi-api-key": self._api_key or "",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            body=json.dumps(body).encode(),
            timeout_s=self._timeout_s,
        )

    def synthesize(self, request: TtsRequest) -> TtsResult:
        if not request.text.strip():
            raise ValueError("nothing to synthesize: empty text")
        if not self._api_key:
            raise PermanentAdapterError(
                f"{API_KEY_ENV} is not set: add `{API_KEY_ENV}=...` to .env in the repo root"
            )
        if self._needs_ffprobe:
            try:
                find_tool("ffprobe")
            except MediaError as e:  # checked before paying, not after
                raise PermanentAdapterError(str(e)) from e

        response = send(self._transport, self.request_for(request), what="ElevenLabs")
        if not 200 <= response.status < 300:
            raise classify_failure(response.status, response.body)

        audio, alignment = self._parse(response.body)
        try:
            duration = self._measure(audio)
        except MediaError as e:
            raise TransientAdapterError(f"ElevenLabs returned unreadable audio: {e}") from e
        if duration <= 0:
            raise TransientAdapterError("ElevenLabs returned audio with no duration")

        credits = _credits(response)
        if credits is None:
            credits = float(len(request.text))
        return TtsResult(
            audio=audio,
            format="mp3",
            duration_s=duration,
            characters=len(request.text),
            provider=PROVIDER,
            model=self._model,
            cost_usd=round(credits * self._usd_per_credit, 6),
            alignment=alignment,
        )

    @staticmethod
    def _parse(body: bytes) -> tuple[bytes, Alignment | None]:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise TransientAdapterError(f"ElevenLabs returned unreadable JSON: {e}") from e
        if not isinstance(data, dict):
            raise TransientAdapterError("ElevenLabs returned JSON without an object")
        encoded = data.get("audio_base64")
        if not isinstance(encoded, str) or not encoded:
            raise TransientAdapterError("ElevenLabs returned no audio")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as e:
            raise TransientAdapterError(f"ElevenLabs returned broken base64 audio: {e}") from e
        if not audio:
            raise TransientAdapterError("ElevenLabs returned empty audio")
        # Timings for the text as sent; the normalized ones ("1984" read as words) are a fallback.
        alignment = _alignment(data.get("alignment")) or _alignment(
            data.get("normalized_alignment")
        )
        return audio, alignment
