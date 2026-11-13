"""ASS subtitles: colours, times, one event per word, highlight, hook skipped, no flicker."""

import itertools
import re

import pytest

from pipeline.ass import LINGER_S, CaptionStyle, ass_color, ass_time, build_ass, escape
from pipeline.steps.captions import CaptionWord, Chunk, chunk

STYLE = CaptionStyle(
    font="Montserrat ExtraBold",
    size=84,
    color="#FFFFFF",
    highlight="#F4D35E",
    outline=6,
    margin_v=422,
    margin_h=90,
)


def _words(text: str, step: float = 0.4) -> list[CaptionWord]:
    return [CaptionWord(w, i * step, i * step + 0.3) for i, w in enumerate(text.split())]


def _events(ass: str) -> list[tuple[str, str, str]]:
    return re.findall(r"^Dialogue: 0,([^,]+),([^,]+),Default,,0,0,0,,(.*)$", ass, re.M)


def test_colour_is_alpha_blue_green_red() -> None:
    assert ass_color("#C1666B") == "&H006B66C1&"
    assert ass_color("#ffffff") == "&H00FFFFFF&"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0, "0:00:00.00"), (1.234, "0:00:01.23"), (61.5, "0:01:01.50"), (3725.01, "1:02:05.01")],
)
def test_time_format(seconds: float, text: str) -> None:
    assert ass_time(seconds) == text


def test_header_matches_the_frame_and_series_style() -> None:
    ass = build_ass([], [], STYLE)

    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "Style: Default,Montserrat ExtraBold,84,&H00FFFFFF&" in ass
    assert ass.rstrip().endswith("Text")  # no events without words
    assert ",2,90,90,422,1" in ass  # bottom-centre, side margins, bottom margin


def test_one_event_per_word_with_that_word_highlighted() -> None:
    words = _words("Honey outlives pharaohs.")
    ass = build_ass(words, chunk(words, 3), STYLE)

    events = _events(ass)
    assert [e[0] for e in events] == ["0:00:00.00", "0:00:00.40", "0:00:00.80"]
    on, off = "{\\1c&H005ED3F4&}", "{\\1c&H00FFFFFF&}"
    assert events[0][2] == f"{on}Honey{off} outlives pharaohs."
    assert events[1][2] == f"Honey {on}outlives{off} pharaohs."
    assert events[2][2] == f"Honey outlives {on}pharaohs.{off}"


def test_word_events_are_back_to_back_and_the_chunk_lingers() -> None:
    words = _words("one two three")
    events = _events(build_ass(words, chunk(words, 3), STYLE))

    assert [(e[0], e[1]) for e in events[:2]] == [
        ("0:00:00.00", "0:00:00.40"),
        ("0:00:00.40", "0:00:00.80"),
    ]
    assert events[-1][1] == ass_time(words[-1].end + LINGER_S)


def test_no_flicker_when_the_next_chunk_starts_soon() -> None:
    words = _words("one two three four five six")  # two chunks, 0.1 s gap between words
    events = _events(build_ass(words, chunk(words, 3), STYLE))

    assert events[2][1] == events[3][0]  # chunk 1 ends exactly when chunk 2 begins


def test_hook_words_are_left_out() -> None:
    words = _words("This sounds made up. Today we look at honey")
    chunks = chunk(words, 3)
    events = _events(build_ass(words, chunks, STYLE, first_word=4))

    assert len(events) == len(words) - 4
    assert all("This" not in text and "made" not in text for _, _, text in events)
    assert events[0][0] == ass_time(words[4].start)


def test_hook_ending_mid_chunk_keeps_only_the_rest_of_that_chunk() -> None:
    words = _words("alpha beta gamma delta")
    chunks = [Chunk("alpha beta gamma delta", 0.0, 1.5, 0, 3)]
    events = _events(build_ass(words, chunks, STYLE, first_word=2))

    assert [text.replace("{\\1c&H005ED3F4&}", "").replace("{\\1c&H00FFFFFF&}", "")
            for _, _, text in events] == ["gamma delta", "gamma delta"]  # fmt: skip


def test_override_characters_are_neutralised() -> None:
    assert escape("a{b}c\\N") == "a(b)c/N"


def _seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def test_events_never_overlap_even_when_a_word_has_zero_length() -> None:
    # Real case (video 4): whisper gave "Scientists" zero length at the start of "suspect".
    # Overlapping events make libass lift the caption a line for the whole word.
    words = [
        CaptionWord("Why", 18.0, 18.2),
        CaptionWord("slow?", 18.2, 18.6),
        CaptionWord("Scientists", 19.46, 19.46),
        CaptionWord("suspect", 19.46, 20.0),
        CaptionWord("solar", 20.0, 20.36),
        CaptionWord("tides", 20.36, 20.74),
    ]
    events = _events(build_ass(words, chunk(words, 3), STYLE))

    times = [(_seconds(start), _seconds(end)) for start, end, _ in events]
    assert len(times) == len(words)
    for (_, previous_end), (start, end) in itertools.pairwise(times):
        assert start >= previous_end
        assert end > start
    assert times[-1][0] == pytest.approx(20.36)  # the shift does not snowball
