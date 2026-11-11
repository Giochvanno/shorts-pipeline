"""Render logic that needs no FFmpeg: timing, hook wrapping, fonts."""

import pytest

from pipeline.config import list_series, load_series
from pipeline.steps import render


@pytest.mark.parametrize(
    ("weights", "total_s"),
    [([10, 20, 30], 12.0), ([1] * 8, 31.196), ([50, 1, 1], 2.0), ([3, 3, 3], 0.05)],
)
def test_frame_counts_add_up_exactly(weights: list[int], total_s: float) -> None:
    counts = render.frame_counts(weights, total_s)

    assert sum(counts) == max(round(total_s * render.FPS), len(weights))
    assert all(c >= 1 for c in counts)


def test_frame_counts_follow_word_share() -> None:
    assert render.frame_counts([1, 2, 1], 4.0) == [30, 60, 30]


def test_frame_counts_reject_empty_scenes() -> None:
    with pytest.raises(ValueError, match="positive"):
        render.frame_counts([3, 0], 2.0)


def test_hook_wraps_into_short_lines() -> None:
    lines = render.wrap_hook("This one sounds completely made up.")

    assert len(lines) > 1
    assert all(len(line) <= render.HOOK_LINE_CHARS for line in lines)


def test_every_series_font_ships_with_the_repo() -> None:
    for series_id in list_series():
        font = render.font_path(load_series(series_id).captions.font)
        assert font.read_bytes()[:4] in {b"\x00\x01\x00\x00", b"OTTO", b"true"}


def test_font_licence_ships_with_the_font() -> None:
    assert (render.FONTS_DIR / "OFL.txt").exists()
