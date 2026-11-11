"""Fake images: a real PNG with a vertical gradient, no network, no cost.

Colours come from the prompt hash, so each scene looks different and the
same prompt + seed always gives the same picture. Written with zlib only.
"""

import hashlib
import struct
import zlib

from pipeline.adapters.images.base import ImageRequest, ImageResult

_PALETTE = (
    (46, 64, 87),
    (232, 216, 176),
    (193, 102, 107),
    (245, 241, 230),
    (84, 110, 122),
    (120, 144, 98),
)


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def gradient_png(
    width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]
) -> bytes:
    rows = bytearray()
    for y in range(height):
        t = y / max(height - 1, 1)
        pixel = bytes(round(a + (b - a) * t) for a, b in zip(top, bottom, strict=True))
        rows += b"\x00" + pixel * width  # filter byte 0 = no filter
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + _chunk(b"IEND", b"")
    )


class FakeImages:
    def __init__(self, model: str = "fake-images") -> None:
        self._model = model
        self.calls: list[ImageRequest] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def generate(self, request: ImageRequest) -> ImageResult:
        self.calls.append(request)
        digest = hashlib.sha256(f"{request.seed}:{request.prompt}".encode()).digest()
        top = _PALETTE[digest[0] % len(_PALETTE)]
        bottom = _PALETTE[(digest[0] + 1 + digest[1] % (len(_PALETTE) - 1)) % len(_PALETTE)]
        return ImageResult(
            image=gradient_png(request.width, request.height, top, bottom),
            format="png",
            width=request.width,
            height=request.height,
            seed=request.seed,
            provider=self.provider,
            model=self.model,
            cost_usd=0.0,
            licence="fake: test only, never publish",
        )
