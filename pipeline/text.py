"""Text as it is spoken, for comparing a script with what speech recognition heard.

The script says "243 Earth days"; the voice says "two hundred forty-three";
whisper may write either. Both sides go through `spoken_tokens` before they are
compared, so the "voice matches script" check does not fail on a number.

    spoken_tokens("243 Earth-days, 50%!")
    -> ["two", "hundred", "forty", "three", "earth", "days", "fifty", "percent"]

The rules are tables (SYMBOLS, CURRENCIES), each entry tested: a symbol reads the
same alone ("85 %", as whisper writes it) or glued to a number ("85%"). Also:
thousands commas, decimals, ordinals ("21st"), "$1.5 million" read as "... million
dollars", British "-tre" spellings ("centimetres", as whisper writes them).

`vocabulary_words` picks the words speech recognition may misspell, for its
prompt and for telling a real but unsure word from an echo of that prompt.

Not covered: years read in pairs ("1984" as "nineteen eighty-four"), Roman
numerals, units that are also words ("m", "g", "in"); each costs a word or two
of the match, and the second listen can still find them.
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
# Symbols and abbreviations as said aloud. Keys are lower case; each reads the same
# alone ("85 %", "3.5 km") or glued to a number ("85%", "3.5km", "50\u00b0F").
SYMBOLS: dict[str, tuple[str, ...]] = {
    "%": ("percent",),
    "&": ("and",),
    "+": ("plus",),
    "\u00b0": ("degrees",),
    "\u00b0f": ("degrees", "fahrenheit"),
    "\u00b0c": ("degrees", "celsius"),
    "km": ("kilometers",),
    "km/h": ("kilometers", "per", "hour"),
    "mph": ("miles", "per", "hour"),
    "cm": ("centimeters",),
    "mm": ("millimeters",),
    "kg": ("kilograms",),
    "lb": ("pounds",),
    "lbs": ("pounds",),
    "ft": ("feet",),
}
# Said after the number: "$20" -> "twenty dollars".
CURRENCIES: dict[str, str] = {"$": "dollars", "\u20ac": "euros", "\u00a3": "pounds"}
_SCALE_WORDS = frozenset({"thousand", "million", "billion", "trillion"})
_ORDINAL_SUFFIXES = frozenset({"st", "nd", "rd", "th"})

_NUMBER = re.compile(r"([$\u20ac\u00a3]?)(\d[\d,]*(?:\.\d+)?)(.*)")
_PARTS = re.compile(r"[\-\u2010-\u2015/]+")  # hyphens, dashes, slashes inside a word
_EDGE_PUNCTUATION = '.,!?;:"()[]\u201c\u201d\u2018\u2019\u2026'  # curly quotes, ellipsis
_BRITISH_TRE = re.compile(r"(?<=[a-z])tre(s?)$")  # centimetre(s) -> centimeter(s)


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


def _number_words(number: str) -> list[str]:
    whole, _, fraction = number.replace(",", "").partition(".")
    words = number_words(int(whole))
    if fraction:
        words += ["point", *(_ONES[int(d)] for d in fraction)]
    return words


def _clean(word: str) -> str:
    cleaned = re.sub(r"[^\w']", "", word.replace("\u2019", "'")).strip("'")
    return _BRITISH_TRE.sub(r"ter\1", cleaned)


def _part_tokens(part: str) -> list[str] | None:
    """Tokens of a number-like part ("$20", "85%", "3.5km", "21st"), or None if not one."""
    match = _NUMBER.fullmatch(part)
    if not match:
        return None
    currency, number, suffix = match.groups()
    words = _number_words(number)
    if suffix in SYMBOLS:
        words += SYMBOLS[suffix]
    elif suffix in _ORDINAL_SUFFIXES and "." not in number:
        words = _ordinal(words)
    elif suffix:
        return None  # "1990s", "3d": a word with digits, compared as written
    if currency:
        words.append(CURRENCIES[currency])
    return words


def spoken_words(text: str) -> list[tuple[str, int]]:
    """(token, index of the whitespace-separated word it came from), in spoken order.

    The index lets a missed token be found in time again (per-word timings).
    """
    words = [raw.lower().strip(_EDGE_PUNCTUATION) for raw in text.split()]
    out: list[tuple[str, int]] = []
    skip = -1
    for i, word in enumerate(words):
        if i == skip:
            continue
        if word in SYMBOLS:
            out += [(token, i) for token in SYMBOLS[word]]
            continue
        for part in _PARTS.split(word):
            part = part.strip(_EDGE_PUNCTUATION)
            tokens = _part_tokens(part)
            if tokens is None:
                cleaned = _clean(part)
                out += [(cleaned, i)] if cleaned else []
                continue
            scaled = i + 1 < len(words) and words[i + 1] in _SCALE_WORDS
            if part[:1] in CURRENCIES and scaled:  # "$1.5 million" -> "... million dollars"
                tokens.insert(-1, words[i + 1])
                skip = i + 1
            out += [(token, i) for token in tokens]
    return out


def spoken_tokens(text: str) -> list[str]:
    """Lower-case words with numbers and symbols spelled out, punctuation dropped."""
    return [token for token, _ in spoken_words(text)]


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
