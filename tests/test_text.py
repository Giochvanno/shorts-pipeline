"""Spoken-form normalization used by the voice check."""

import pytest

from pipeline.text import (
    CURRENCIES,
    SYMBOLS,
    number_words,
    spoken_tokens,
    spoken_words,
    vocabulary_words,
)


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


def test_vocabulary_is_long_words_and_names_not_sentence_starters() -> None:
    text = "The tower grows. On hot days the Eiffel Tower expands. Not much. Ask Simard."

    assert vocabulary_words(text) == ["Eiffel", "Tower", "expands", "Simard"]


def test_vocabulary_lists_each_word_once() -> None:
    assert vocabulary_words("Bees love glucose. Honey has glucose and Bees.") == [
        "glucose",
        "Bees",
    ]


# --- the normalization table: every rule, both ways it is written ---------------------

RULES = [
    # (script or whisper text, spoken tokens)
    ("85%", "eighty five percent"),
    ("85 %", "eighty five percent"),  # whisper's way (video 6)
    ("85 percent", "eighty five percent"),
    ("salt & pepper", "salt and pepper"),
    ("5+ years", "five plus years"),
    ("50\u00b0", "fifty degrees"),
    ("50 \u00b0F", "fifty degrees fahrenheit"),
    ("50\u00b0F", "fifty degrees fahrenheit"),
    ("-40\u00b0C", "forty degrees celsius"),
    ("3.5 km", "three point five kilometers"),
    ("3.5km", "three point five kilometers"),
    ("120 km/h", "one hundred twenty kilometers per hour"),
    ("60 mph", "sixty miles per hour"),
    ("15 cm", "fifteen centimeters"),
    ("15 centimetres", "fifteen centimeters"),  # whisper's British spelling (video 7)
    ("2 mm", "two millimeters"),
    ("70kg", "seventy kilograms"),
    ("5 lbs", "five pounds"),
    ("1,000 ft", "one thousand feet"),
    ("$20", "twenty dollars"),
    ("$1.5 million", "one point five million dollars"),
    ("\u20ac5", "five euros"),
    ("\u00a33", "three pounds"),
    ("1,000", "one thousand"),
    ("the 1990s", "the 1990s"),  # digits glued to letters stay a word, on both sides
]


@pytest.mark.parametrize(("text", "tokens"), RULES, ids=[text for text, _ in RULES])
def test_normalization_rule(text: str, tokens: str) -> None:
    assert spoken_tokens(text) == tokens.split()


def test_every_symbol_and_currency_has_a_rule_test() -> None:
    tested = " ".join(text.lower() for text, _ in RULES)
    for key in [*SYMBOLS, *CURRENCIES]:
        assert key in tested, f"no test for {key!r}"


def test_tokens_remember_their_script_word() -> None:
    assert spoken_words("Up to 40% of fine-root carbon") == [
        ("up", 0),
        ("to", 1),
        ("forty", 2),
        ("percent", 2),
        ("of", 3),
        ("fine", 4),
        ("root", 4),
        ("carbon", 5),
    ]
