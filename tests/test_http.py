"""The urllib transport against a real local HTTP server (no internet, no proxy)."""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pipeline.adapters.errors import TransientAdapterError
from pipeline.adapters.http import HttpRequest, HttpResponse, send, urllib_transport


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        received = self.rfile.read(length)
        status = 404 if self.path.startswith("/missing") else 200
        answer = b'{"detail": {"code": "voice_not_found"}}' if status == 404 else received
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Character-Cost", "17")
        self.send_header("X-Echo-Agent", self.headers.get("User-Agent", ""))
        self.send_header("Content-Length", str(len(answer)))
        self.end_headers()
        self.wfile.write(answer)

    def log_message(self, *_args: object) -> None:  # keep pytest output clean
        pass


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")  # also skips a Windows system proxy
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_success_returns_body_and_lower_case_headers(server: str) -> None:
    request = HttpRequest("POST", f"{server}/ok", {"Content-Type": "application/json"}, b"{}", 10)
    response = urllib_transport(request)

    assert response.status == 200
    assert response.body == b"{}"
    assert response.header("character-cost") == "17"
    assert response.header("Character-Cost") == "17"
    assert response.header("x-echo-agent") == "shorts-pipeline/0.1"


def test_http_errors_are_answers_not_exceptions(server: str) -> None:
    response = urllib_transport(HttpRequest("POST", f"{server}/missing", body=b"{}", timeout_s=10))

    assert response.status == 404
    assert b"voice_not_found" in response.body


def test_send_turns_network_failures_into_transient_errors() -> None:
    def refused(_request: HttpRequest) -> HttpResponse:
        raise ConnectionRefusedError("connection refused")

    with pytest.raises(TransientAdapterError, match="Provider: network error"):
        send(refused, HttpRequest("GET", "https://x.test"), what="Provider")
