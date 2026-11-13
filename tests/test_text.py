"""Spoken-form normalization used by the voice check."""

import pytest

from pipeline.text import number_words, spoken_tokens


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (7, "seven"),
        (13, "thirteen"),
        (40, "forty"),
        (117, "one hundred seventeen"),
        (243, "two hundred forty three"),
        (1000, "one thousand"),
        (2802, "two thousand eight hundred two"),
        (5_832_600, "five million eight hundred thirty two thousand six hundred"),
    ],
)
def test_number_words(n: int, words: str) -> None:
    assert " ".join(number_words(n)) == words


def test_script_digits_and_spoken_words_compare_equal() -> None:
    # The case from video 4: the script writes the number, the voice says it.
    script = spoken_tokens("A full spin takes 243 Earth days.")
    heard = spoken_tokens("a full spin takes two hundred forty-three earth days")

    assert script == heard


def test_whisper_writing_digits_also_compares_equal() -> None:
    assert spoken_tokens("243 Earth days") == spoken_tokens("two hundred forty-three Earth days")


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("Honey outlives pharaohs.", ["honey", "outlives", "pharaohs"]),
        ("Here's how!", ["here's", "how"]),
        ("It\u2019s 2.5 times", ["it's", "two", "point", "five", "times"]),
        ("About 50% of it", ["about", "fifty", "percent", "of", "it"]),
        ("the 21st and 3rd", ["the", "twenty", "first", "and", "third"]),
        ("the 40th, 12th, 8th", ["the", "fortieth", "twelfth", "eighth"]),
        ("1,000 years", ["one", "thousand", "years"]),
        ("rocky\u2014volcanic", ["rocky", "volcanic"]),
        ("\u201cQuoted\u201d (aside)\u2026", ["quoted", "aside"]),
    ],
)
def test_spoken_tokens(text: str, tokens: list[str]) -> None:
    assert spoken_tokens(text) == tokens
