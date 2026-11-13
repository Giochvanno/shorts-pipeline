"""ElevenLabs adapter on a fake HTTP transport: request shape, money, timings, errors.

Error bodies follow the ElevenLabs docs: the current shape (`detail.type/code/message`),
the legacy one (`detail.status`) and 422 validation lists. The live test checks
the real thing.
"""

import base64
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.http import HttpRequest, HttpResponse
from pipeline.adapters.tts import TtsRequest
from pipeline.adapters.tts.elevenlabs import ElevenLabsTts, classify_failure, probe_mp3_duration
from pipeline.media import MediaError

KEY = "sk_secret_key_123"
AUDIO = b"ID3fake-mp3-bytes"
TEXT = "Octopuses have three hearts."
REQUEST = TtsRequest(text=TEXT, voice_id="voice-1", stability=0.4, similarity_boost=0.8, speed=1.1)


def alignment_for(text: str, step: float = 0.05) -> dict[str, Any]:
    return {
        "characters": list(text),
        "character_start_times_seconds": [i * step for i in range(len(text))],
        "character_end_times_seconds": [(i + 1) * step for i in range(len(text))],
    }


def ok_body(**overrides: Any) -> bytes:
    data: dict[str, Any] = {
        "audio_base64": base64.b64encode(AUDIO).decode(),
        "alignment": alignment_for(TEXT),
        "normalized_alignment": alignment_for(TEXT),
    }
    data.update(overrides)
    return json.dumps(data).encode()


class FakeHttp:
    def __init__(self, *responses: HttpResponse | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[HttpRequest] = []

    def __call__(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        answer = self.responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def ok(headers: dict[str, str] | None = None, body: bytes | None = None) -> HttpResponse:
    return HttpResponse(200, ok_body() if body is None else body, headers or {})


def make(http: FakeHttp, *, api_key: str | None = KEY, duration: float = 2.5) -> ElevenLabsTts:
    return ElevenLabsTts(
        "model-from-config",
        api_key=api_key,
        usd_per_credit=0.0002,
        timeout_s=42,
        base_url="https://api.test",
        transport=http,
        measure=lambda _audio: duration,
    )


# --- request ----------------------------------------------------------------------


def test_request_carries_text_model_voice_settings_and_key_in_header_only() -> None:
    http = FakeHttp(ok())
    make(http).synthesize(REQUEST)

    (sent,) = http.requests
    url = urlsplit(sent.url)
    assert sent.method == "POST"
    assert url.path == "/v1/text-to-speech/voice-1/with-timestamps"
    assert url.query == "output_format=mp3_44100_128"
    assert sent.headers["xi-api-key"] == KEY
    assert KEY not in sent.url
    assert sent.timeout_s == 42
    assert json.loads(sent.body or b"") == {
        "text": TEXT,
        "model_id": "model-from-config",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "speed": 1.1},
    }


def test_voice_id_cannot_escape_the_url_path() -> None:
    http = FakeHttp(ok())
    tricky = TtsRequest(
        text=TEXT, voice_id="a/../b?x=1", stability=0.5, similarity_boost=0.5, speed=1
    )
    make(http).synthesize(tricky)

    path = urlsplit(http.requests[0].url).path
    assert path.count("/") == 4  # /v1/text-to-speech/<one segment>/with-timestamps
    assert unquote(path.split("/")[3]) == "a/../b?x=1"


# --- success ------------------------------------------------------------------------


def test_returns_mp3_measured_duration_timings_and_cost_from_credits_header() -> None:
    result = make(FakeHttp(ok({"character-cost": "28"}))).synthesize(REQUEST)

    assert result.audio == AUDIO
    assert result.format == "mp3"
    assert result.duration_s == 2.5
    assert result.characters == len(TEXT)
    assert result.provider == "elevenlabs"
    assert result.model == "model-from-config"
    assert result.cost_usd == pytest.approx(28 * 0.0002)
    assert result.alignment is not None
    assert "".join(result.alignment.characters) == TEXT
    assert result.alignment.starts_s[1] == pytest.approx(0.05)


def test_older_credit_header_name_is_read_too() -> None:
    result = make(FakeHttp(ok({"x-character-count": "14"}))).synthesize(REQUEST)

    assert result.cost_usd == pytest.approx(14 * 0.0002)  # e.g. Flash at 0.5 credit per char


@pytest.mark.parametrize("headers", [{}, {"character-cost": "n/a"}, {"character-cost": "-5"}])
def test_without_a_usable_credits_header_one_credit_per_character_is_assumed(
    headers: dict[str, str],
) -> None:
    result = make(FakeHttp(ok(headers))).synthesize(REQUEST)

    assert result.cost_usd == pytest.approx(len(TEXT) * 0.0002)


def test_normalized_timings_are_the_fallback() -> None:
    body = ok_body(alignment=None, normalized_alignment=alignment_for("three"))
    result = make(FakeHttp(ok(body=body))).synthesize(REQUEST)

    assert result.alignment is not None
    assert "".join(result.alignment.characters) == "three"


@pytest.mark.parametrize(
    "broken",
    [
        None,
        {"characters": ["a", "b"], "character_start_times_seconds": [0.0]},
        {
            "characters": ["a"],
            "character_start_times_seconds": ["x"],
            "character_end_times_seconds": [0.1],
        },
        {"characters": [], "character_start_times_seconds": [], "character_end_times_seconds": []},
    ],
)
def test_broken_timings_are_dropped_but_the_paid_audio_is_kept(broken: Any) -> None:
    body = ok_body(alignment=broken, normalized_alignment=None)
    result = make(FakeHttp(ok(body=body))).synthesize(REQUEST)

    assert result.audio == AUDIO
    assert result.alignment is None


# --- refused before paying ------------------------------------------------------------


def test_empty_text_is_refused_without_a_request() -> None:
    http = FakeHttp()
    empty = TtsRequest(text=" ", voice_id="v", stability=0.5, similarity_boost=0.5, speed=1)

    with pytest.raises(ValueError, match="empty"):
        make(http).synthesize(empty)
    assert http.requests == []


@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_is_permanent_without_a_request(key: str | None) -> None:
    http = FakeHttp()

    with pytest.raises(PermanentAdapterError, match=r"ELEVENLABS_API_KEY .*\.env"):
        make(http, api_key=key).synthesize(REQUEST)
    assert http.requests == []


def test_missing_ffprobe_is_permanent_and_checked_before_paying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pipeline.media.shutil.which", lambda _name: None)
    http = FakeHttp(ok())
    real_measure = ElevenLabsTts("m", api_key=KEY, usd_per_credit=0.0002, transport=http)

    with pytest.raises(PermanentAdapterError, match="ffprobe"):
        real_measure.synthesize(REQUEST)
    assert http.requests == []


def test_key_never_shows_in_repr() -> None:
    assert KEY not in repr(make(FakeHttp()))


# --- errors -----------------------------------------------------------------------------


def body(detail: Any) -> bytes:
    return json.dumps({"detail": detail}).encode()


def new_style(type_: str, code: str, message: str = "msg") -> bytes:
    return body({"type": type_, "code": code, "message": message, "request_id": "req-1"})


def legacy(status: str, message: str = "msg") -> bytes:
    return body({"status": status, "message": message})


PERMANENT_CASES = [
    (401, legacy("invalid_api_key", "Invalid API key"), "API key"),
    (401, new_style("authentication_error", "missing_api_key"), "API key"),
    (401, legacy("quota_exceeded", "You have insufficient quota"), "credits are used up"),
    (400, legacy("quota_exceeded"), "credits are used up"),
    (402, new_style("payment_required", "insufficient_credits"), "credits are used up"),
    (402, b"", "credits are used up"),
    (404, new_style("not_found", "voice_not_found"), "voice"),
    (400, legacy("invalid_voice_id"), "voice"),
    (404, new_style("not_found", "model_not_found"), "model"),
    (403, new_style("authorization_error", "model_access_denied"), "model"),
    (403, new_style("authorization_error", "forbidden"), "may not use"),
    (400, new_style("validation_error", "text_too_long"), "rejected"),
    (400, legacy("max_character_limit_exceeded"), "rejected"),
    (422, body([{"loc": ["body", "text"], "msg": "field required", "type": "missing"}]), "text"),
]


@pytest.mark.parametrize(("status", "error_body", "hint"), PERMANENT_CASES)
def test_errors_a_human_must_fix_are_permanent(status: int, error_body: bytes, hint: str) -> None:
    with pytest.raises(PermanentAdapterError, match=hint) as exc:
        make(FakeHttp(HttpResponse(status, error_body))).synthesize(REQUEST)

    assert f"HTTP {status}" in str(exc.value)
    assert KEY not in str(exc.value)


TRANSIENT_CASES = [
    (429, new_style("rate_limit_error", "rate_limit_exceeded")),
    (429, new_style("rate_limit_error", "concurrent_limit_exceeded")),
    (429, legacy("too_many_concurrent_requests")),
    (429, legacy("system_busy")),
    (500, new_style("internal_error", "internal_error")),
    (503, b"<html><body>502 Bad Gateway</body></html>"),
    (504, b""),
    (408, b""),
]


@pytest.mark.parametrize(("status", "error_body"), TRANSIENT_CASES)
def test_hiccups_are_transient(status: int, error_body: bytes) -> None:
    with pytest.raises(TransientAdapterError, match=f"HTTP {status}"):
        make(FakeHttp(HttpResponse(status, error_body))).synthesize(REQUEST)


def test_error_text_keeps_code_message_and_request_id_for_support() -> None:
    error = classify_failure(404, new_style("not_found", "voice_not_found", "Voice x not found"))

    assert "voice_not_found" in str(error)
    assert "Voice x not found" in str(error)
    assert "req-1" in str(error)


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        (ConnectionResetError("reset by peer"), "network error"),
        (TimeoutError("timed out"), "no answer within 42 s"),
    ],
)
def test_network_failures_are_transient(failure: Exception, match: str) -> None:
    with pytest.raises(TransientAdapterError, match=match):
        make(FakeHttp(failure)).synthesize(REQUEST)


@pytest.mark.parametrize(
    ("broken_body", "match"),
    [
        (b"<html>not json</html>", "unreadable JSON"),
        (b"[1, 2]", "without an object"),
        (ok_body(audio_base64=None), "no audio"),
        (ok_body(audio_base64=""), "no audio"),
        (ok_body(audio_base64="***not base64***"), "broken base64"),
    ],
)
def test_a_broken_200_answer_is_transient(broken_body: bytes, match: str) -> None:
    with pytest.raises(TransientAdapterError, match=match):
        make(FakeHttp(ok(body=broken_body))).synthesize(REQUEST)


def test_unreadable_or_silent_audio_is_transient() -> None:
    def broken(_audio: bytes) -> float:
        raise MediaError("Invalid data found when processing input")

    tts = ElevenLabsTts(
        "m", api_key=KEY, usd_per_credit=0, transport=FakeHttp(ok()), measure=broken
    )
    with pytest.raises(TransientAdapterError, match="unreadable audio"):
        tts.synthesize(REQUEST)

    with pytest.raises(TransientAdapterError, match="no duration"):
        make(FakeHttp(ok()), duration=0.0).synthesize(REQUEST)


# --- real duration measurement --------------------------------------------------------


@pytest.mark.usefixtures("ffmpeg")
def test_probe_measures_a_real_mp3(tmp_path: Path) -> None:
    mp3 = tmp_path / "tone.mp3"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5",
            "-c:a", "libmp3lame", "-b:a", "128k", str(mp3),
        ],
        check=True,
        timeout=60,
    )  # fmt: skip

    assert probe_mp3_duration(mp3.read_bytes()) == pytest.approx(1.5, abs=0.1)
