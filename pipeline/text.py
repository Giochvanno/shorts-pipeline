"""Text as it is spoken, for comparing a script with what speech recognition heard.

The script says "243 Earth days"; the voice says "two hundred forty-three";
whisper may write either. Both sides go through `spoken_tokens` before they are
compared, so the "voice matches script" check does not fail on a number.

    spoken_tokens("243 Earth-days, 50%!")
    -> ["two", "hundred", "forty", "three", "earth", "days", "fifty", "percent"]

`vocabulary_words` picks the words speech recognition may misspell, for its
prompt and for telling a real but unsure word from an echo of that prompt.

Covered: integers (with thousands commas), decimals, percentages, ordinals
("21st"), hyphens, case, punctuation. Not covered: years read in pairs
("1984" as "nineteen eighty-four"), currencies, Roman numerals; if a script
leans on those, the match drops by a word or two, well above the threshold.
"""

import re

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_ORDINAL_WORDS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)(%|st|nd|rd|th)?")
_SEPARATORS = re.compile(r"[\s\-\u2010-\u2015/]+")  # spaces, hyphens, dashes, slashes
_EDGE_PUNCTUATION = '.,!?;:"()[]\u201c\u201d\u2018\u2019\u2026'  # curly quotes, ellipsis


def _below_thousand(n: int) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words += [_ONES[hundreds], "hundred"]
    if rest >= 20:
        tens, ones = divmod(rest, 10)
        words.append(_TENS[tens])
        if ones:
            words.append(_ONES[ones])
    elif rest or not words:
        words.append(_ONES[rest])
    return words


def number_words(n: int) -> list[str]:
    """Cardinal number as words, American style (no "and"): 243 -> two hundred forty three."""
    if n < 0:
        return ["minus", *number_words(-n)]
    if n < 1000:
        return _below_thousand(n)
    words: list[str] = []
    for size, name in _SCALES:
        if n >= size:
            count, n = divmod(n, size)
            words += [*number_words(count), name]
    if n:
        words += _below_thousand(n)
    return words


def _ordinal(words: list[str]) -> list[str]:
    last = words[-1]
    if last in _ORDINAL_WORDS:
        last = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return [*words[:-1], last]


def _number_token(number: str, suffix: str | None) -> list[str]:
    whole, _, fraction = number.replace(",", "").partition(".")
    words = number_words(int(whole))
    if fraction:
        words += ["point", *(_ONES[int(d)] for d in fraction)]
    if suffix == "%":
        words.append("percent")
    elif suffix and not fraction:
        words = _ordinal(words)
    return words


def spoken_tokens(text: str) -> list[str]:
    """Lower-case words with numbers spelled out and punctuation dropped."""
    tokens: list[str] = []
    for raw in re.split(_SEPARATORS, text.lower()):
        word = raw.strip(_EDGE_PUNCTUATION)
        match = _NUMBER.fullmatch(word)
        if match:
            tokens += _number_token(match.group(1), match.group(2))
            continue
        cleaned = re.sub(r"[^\w']", "", word.replace("\u2019", "'")).strip("'")
        if cleaned:
            tokens.append(cleaned)
    return tokens


_SENTENCE_END = (".", "!", "?", '."', '!"', '?"')


def vocabulary_words(text: str) -> list[str]:
    """Words recognition may misspell: long words and names (capitalised mid-sentence).

    Words that are capitalised only because they start a sentence ("The", "On",
    "Not") are left out. Seen live (videos 5-8): with them in a comma-separated
    prompt, whisper wrote most words with a trailing comma and was unsure of
    exactly those sentence starters, so correct words failed the voice check.
    """
    seen: dict[str, None] = {}
    sentence_start = True
    for raw in text.split():
        word = re.sub(r"[^\w'-]", "", raw)
        name = word[:1].isupper() and len(word) > 1 and not sentence_start
        if len(word) >= 7 or name:
            seen.setdefault(word, None)
        sentence_start = raw.endswith(_SENTENCE_END)
    return list(seen)
