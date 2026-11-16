"""Offline reviewer: accepts every frame, or answers with a verdict a test hands it."""

from collections.abc import Callable

from pipeline.adapters.vision.base import FrameCheckRequest, FrameVerdict


def clean_verdict(request: FrameCheckRequest) -> FrameVerdict:
    return FrameVerdict(
        subject_seen=request.scene[:60],
        right_subject=True,
        text_seen="",
        has_text=False,
        subject_position="middle",
        style_matches=True,
        unwanted_seen="",
        unwanted=False,
        provider="fake",
        model="fake-vision",
        cost_usd=0.0,
    )


class FakeReviewer:
    def __init__(self, answer: Callable[[FrameCheckRequest], FrameVerdict] = clean_verdict) -> None:
        self._answer = answer
        self.calls: list[FrameCheckRequest] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-vision"

    def review(self, request: FrameCheckRequest) -> FrameVerdict:
        self.calls.append(request)
        return self._answer(request)
