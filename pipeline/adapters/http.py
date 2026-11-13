"""Minimal HTTP for real adapters: the standard library, behind a swappable transport.

Adapters never call the network directly; they get a `Transport` (a function
from `HttpRequest` to `HttpResponse`). Production uses `urllib_transport`; tests
pass a fake that returns canned responses, so CI never touches the network.

Why not an HTTP library: two providers, a handful of JSON calls. The standard
library covers it with no extra dependency, and urllib honours HTTPS_PROXY and
the system certificate store on Windows.

Contract of a transport:
- any HTTP status comes back as an `HttpResponse` (4xx/5xx are not exceptions);
- a network failure (DNS, refused, reset, timeout) raises OSError or
  http.client.HTTPException; `send` turns those into TransientAdapterError.
"""

import http.client
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.message import Message

from pipeline.adapters.errors import TransientAdapterError

USER_AGENT = "shorts-pipeline/0.1"


@dataclass(frozen=True, slots=True)
class HttpRequest:
    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes | None = None
    timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)  # names in lower case

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


Transport = Callable[[HttpRequest], HttpResponse]


def _lower(headers: Message) -> dict[str, str]:
    return {name.lower(): value for name, value in headers.items()}


def urllib_transport(request: HttpRequest) -> HttpResponse:
    # A fresh opener per call reads the proxy settings at call time, not at import.
    opener = urllib.request.build_opener()
    raw = urllib.request.Request(
        request.url,
        data=request.body,
        headers={"User-Agent": USER_AGENT, **request.headers},
        method=request.method,
    )
    try:
        with opener.open(raw, timeout=request.timeout_s) as response:
            return HttpResponse(response.status, response.read(), _lower(response.headers))
    except urllib.error.HTTPError as e:  # 4xx/5xx: a normal answer for the adapter to read
        with e:
            return HttpResponse(e.code, e.read(), _lower(e.headers))


def send(transport: Transport, request: HttpRequest, *, what: str) -> HttpResponse:
    """Run one request; network failures become TransientAdapterError (worth a retry)."""
    try:
        return transport(request)
    except TimeoutError as e:
        raise TransientAdapterError(f"{what}: no answer within {request.timeout_s:.0f} s") from e
    except (OSError, http.client.HTTPException) as e:
        raise TransientAdapterError(f"{what}: network error: {e}") from e
