"""Vision port: a reviewer looks at one frame and says what is wrong with it.

A frame is the first frame of an animated clip: a wrong subject, a fake signature or
something that reads as blood would be animated and paid for again by the video
model. So these reject a frame; where the subject sits only informs (no image
model keeps to the camera line, captions get their own safe zone instead).
"""

from dataclasses import dataclass
from typing import Literal, Protocol

Position = Literal["upper", "middle", "lower", "none"]
ImageType = Literal["image/png", "image/jpeg", "image/webp"]


@dataclass(frozen=True, slots=True)
class FrameCheckRequest:
    image: bytes
    media_type: ImageType
    scene: str  # the scene's visual description, as the image model got it
    style: str  # the series' style line


@dataclass(frozen=True, slots=True)
class FrameVerdict:
    subject_seen: str  # what the reviewer sees, in its own words
    right_subject: bool
    text_seen: str  # letters, signatures, watermarks; empty if none
    has_text: bool
    subject_position: Position
    style_matches: bool
    unwanted_seen: str  # blood, wounds, gore-like marks; empty if none
    unwanted: bool
    provider: str
    model: str
    cost_usd: float

    def problems(self) -> list[str]:
        """Why the frame must be drawn again; empty when it can be animated."""
        found = []
        if not self.right_subject:
            found.append(f"wrong subject: {self.subject_seen or 'unclear'}")
        if self.has_text:
            found.append(f"text in the frame: {self.text_seen or 'unreadable marks'}")
        if self.unwanted:
            found.append(f"unwanted content: {self.unwanted_seen or 'unnamed'}")
        if not self.style_matches:
            found.append("off-style")
        return found

    @property
    def accepted(self) -> bool:
        return not self.problems()


class FrameReviewer(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def review(self, request: FrameCheckRequest) -> FrameVerdict: ...


def media_type_of(image: bytes) -> ImageType:
    """The type from the file's own first bytes, never from a name or a claim."""
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("not a PNG, JPEG or WebP image")
