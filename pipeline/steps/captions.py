"""Step `captions`: voice + script -> runs/<id>/captions.json with a timing for every word.

The words on screen come from the script (exact spelling, no "gloo coast" for
"glucose"); only the timings come from speech recognition. The two word lists
are lined up with a diff:

    script:  honey  outlives  pharaohs  here's  how
    heard:   honey  outlives  pharaoh's here's  how     <- misheard, same slot: take its time
    heard:   honey  outlives            here's  how     <- missed: time shared with neighbours
    heard:   ...    glucosoxidase                       <- merged: split the heard span

The share of script words found in the audio is the "voice matches script"
check: below MIN_MATCH the voice skipped or garbled the text, and captions would
lie. That needs the voice redone, so the error is not retryable here.

Words whisper is unsure about (probability below MIN_WORD_PROBABILITY) do not
count as heard. Seen live: on speech it could not understand, whisper echoed
the vocabulary prompt back with p = 0.00-0.34, which matched 60 % of the script
and would have passed the check with made-up timings.
"""

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.adapters.stt import WordTiming
from pipeline.artifacts import file_hash, input_hash, is_cached, write_output
from pipeline.steps import script as script_step
from pipeline.steps import voice as voice_step
from pipeline.steps.base import StepContext, StepError, StepResult

OUTPUT = "captions.json"
CAPTIONS_VERSION = 1
MIN_MATCH = 0.6
MIN_WORD_PROBABILITY = 0.4
_SENTENCE_END = (".", "!", "?")
_TOLERANCE_S = 0.25


@dataclass(frozen=True, slots=True)
class CaptionWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    start: float
    end: float
    first: int  # index of its first word, for highlighting the current word
    last: int


def normalize(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower()).strip("'")


def align(
    script_words: list[str], heard: tuple[WordTiming, ...], duration_s: float
) -> tuple[list[CaptionWord], float]:
    """Give every script word a time. Returns the words and the share matched exactly."""
    if not script_words:
        return [], 0.0
    heard = tuple(
        h for h in heard if h.probability is None or h.probability >= MIN_WORD_PROBABILITY
    )
    a = [normalize(w) for w in script_words]
    b = [normalize(h.word) for h in heard]
    times: list[tuple[float, float] | None] = [None] * len(a)
    matched = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            matched += i2 - i1
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                times[i1 + k] = (heard[j1 + k].start, heard[j1 + k].end)
        elif tag == "replace":
            # Whisper merged or split words ("glucose oxidase" -> "glucosoxidase"):
            # share the span where it heard them, instead of stretching to the next word.
            span_start, span_end = heard[j1].start, heard[j2 - 1].end
            step = max(span_end - span_start, 0.0) / (i2 - i1)
            for k in range(i2 - i1):
                times[i1 + k] = (span_start + k * step, span_start + (k + 1) * step)
    _fill_gaps(times, duration_s)
    words: list[CaptionWord] = []
    previous_end = 0.0
    for text, slot in zip(script_words, times, strict=True):
        assert slot is not None
        start = max(slot[0], previous_end)  # never run backwards
        end = max(slot[1], start)
        words.append(CaptionWord(text, round(start, 3), round(end, 3)))
        previous_end = end
    return words, matched / len(a)


def _fill_gaps(times: list[tuple[float, float] | None], duration_s: float) -> None:
    """Spread words nobody heard evenly between their heard neighbours."""
    i = 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        before = times[i - 1][1] if i > 0 and times[i - 1] is not None else 0.0  # type: ignore[index]
        after = times[j][0] if j < len(times) and times[j] is not None else duration_s  # type: ignore[index]
        step = max(after - before, 0.0) / (j - i)
        for k in range(i, j):
            times[k] = (before + (k - i) * step, before + (k - i + 1) * step)
        i = j


def chunk(words: list[CaptionWord], per_chunk: int) -> list[Chunk]:
    """Short groups shown together; a sentence end always closes a group."""
    chunks: list[Chunk] = []
    first = 0
    for i, word in enumerate(words):
        if i - first + 1 == per_chunk or word.text.endswith(_SENTENCE_END) or i == len(words) - 1:
            group = words[first : i + 1]
            chunks.append(
                Chunk(" ".join(w.text for w in group), group[0].start, group[-1].end, first, i)
            )
            first = i + 1
    return chunks


def run(ctx: StepContext) -> StepResult:
    narration = script_step.load(ctx.run_dir).narration
    audio, duration = voice_step.voice_file(ctx.run_dir)
    per_chunk = ctx.series.captions.words_per_chunk
    output = ctx.run_dir / OUTPUT
    key = input_hash(
        {
            "step": "captions",
            "version": CAPTIONS_VERSION,
            "voice": file_hash(audio),
            "text": narration,
            "words_per_chunk": per_chunk,
            "language": ctx.language,
            "provider": ctx.stt.provider,
            "model": ctx.stt.model,
        }
    )
    if is_cached(output, key):
        return StepResult(outputs=(output,), cost_usd=0.0, cached=True)

    transcript = ctx.stt.transcribe(audio, language=ctx.language, hint=narration)
    words, match = align(narration.split(), transcript.words, duration)
    if match < MIN_MATCH:
        raise StepError(
            f"voice does not match the script: only {match:.0%} of the words were heard"
            " (redo the voice)",
            retryable=False,
        )
    late = [w.text for w in words if w.end > duration + _TOLERANCE_S]
    if late:
        raise StepError(f"captions run past the end of the voice: {late[:3]}", retryable=False)

    chunks = chunk(words, per_chunk)
    data = {
        "match_ratio": round(match, 3),
        "words": [asdict(w) for w in words],
        "chunks": [asdict(c) for c in chunks],
    }
    write_output(
        output,
        json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
        key,
        {
            "provider": transcript.provider,
            "model": transcript.model,
            "cost_usd": transcript.cost_usd,
            "match_ratio": round(match, 3),
            "words": len(words),
            "chunks": len(chunks),
        },
    )
    return StepResult(outputs=(output,), cost_usd=transcript.cost_usd, cached=False)


def load(run_dir: Path) -> tuple[list[CaptionWord], list[Chunk]]:
    """Read captions written by `run` (used by render)."""
    path = run_dir / OUTPUT
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise StepError(f"{path} not found: run the captions step first", retryable=False) from e
    return [CaptionWord(**w) for w in data["words"]], [Chunk(**c) for c in data["chunks"]]
