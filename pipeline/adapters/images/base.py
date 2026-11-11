"""Images port: what every image adapter (fake, FLUX on fal, FLUX on Replicate) must provide."""

from dataclasses import dataclass
from typing import Literal, Protocol

ImageFormat = Literal["png", "jpg", "webp"]


@dataclass(frozen=True, slots=True)
class ImageRequest:
    prompt: str
    seed: int  # one seed per video keeps the style stable across its scenes
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ImageResult:
    image: bytes
    format: ImageFormat
    width: int
    height: int
    seed: int
    provider: str
    model: str
    cost_usd: float
    licence: str  # recorded per asset: we publish only what we may publish


class ImageClient(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def generate(self, request: ImageRequest) -> ImageResult: ...
