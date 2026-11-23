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
# Why a frame is turned down, as codes for the statistics (status counts them).
Reason = Literal["wrong_subject", "extra_object", "text", "unwanted", "off_style"]
REASONS: tuple[Reason, ...] = ("wrong_subject", "extra_object", "text", "unwanted", "off_style")
_LABELS: dict[Reason, str] = {
    "wrong_subject": "wrong subject",
    "extra_object": "extra object",
    "text": "text in the frame",
    "unwanted": "unwanted content",
    "off_style": "off-style",
}


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
    extra_seen: str  # a prominent object the scene does not ask for; empty if none
    extra_object: bool
    text_seen: str  # letters, signatures, watermarks; empty if none
    has_text: bool
    subject_position: Position
    style_matches: bool
    unwanted_seen: str  # blood, wounds, gore-like marks; empty if none
    unwanted: bool
    provider: str
    model: str
    cost_usd: float

    def findings(self) -> list[tuple[Reason, str]]:
        """Why the frame must be drawn again, with what was seen; empty when it can be
        animated."""
        found: list[tuple[Reason, str]] = []
        if not self.right_subject:
            found.append(("wrong_subject", self.subject_seen or "unclear"))
        if self.extra_object:
            found.append(("extra_object", self.extra_seen or "unnamed"))
        if self.has_text:
            found.append(("text", self.text_seen or "unreadable marks"))
        if self.unwanted:
            found.append(("unwanted", self.unwanted_seen or "unnamed"))
        if not self.style_matches:
            found.append(("off_style", ""))
        return found

    def reasons(self) -> list[Reason]:
        return [reason for reason, _ in self.findings()]

    def problems(self) -> list[str]:
        """The findings in words, for the report and the rejected frame's json."""
        return [
            f"{_LABELS[reason]}: {seen}" if seen else _LABELS[reason]
            for reason, seen in self.findings()
        ]

    @property
    def accepted(self) -> bool:
        return not self.findings()


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
