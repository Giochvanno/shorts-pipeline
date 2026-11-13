"""Step `captions`: voice + script -> runs/<id>/captions.json with a timing for every word.

The words on screen come from the script (exact spelling, no "gloo coast" for
"glucose"); only the timings come from elsewhere, per `captions.timing_source`:

- tts (default): when the voice provider says it spoke each word. Exact for the
  text it was given; checked live against the audio (video 4: whisper started
  words up to 0.95 s early after pauses, the provider never did).
- whisper: what the voice check heard (voice_transcript.json).

A provider without timings (a future fallback TTS), or timings that do not fit
the audio (an edit that forgot `Alignment.edited`), fall back to whisper with a
note in the video report. Either source is lined up with the script with a diff:

    script:  honey  outlives  pharaohs  here's  how
    heard:   honey  outlives  pharaoh's here's  how     <- misheard, same slot: take its time
    heard:   honey  outlives            here's  how     <- missed: time shared with neighbours
    heard:   ...    glucosoxidase                       <- merged: split the heard span

Whether the voice says the script is checked by the voice step (whisper, with
a threshold and new takes). Here MIN_MATCH is only a sanity floor for the
line-up itself; below it the captions would lie, so the step refuses.

Words whisper is unsure about (probability below MIN_WORD_PROBABILITY) do not
count as heard. Seen live: on speech it could not understand, whisper echoed
the vocabulary prompt back with p = 0.00-0.34, which matched 60 % of the script
and would have passed the check with made-up timings.
"""

import difflib
import itertools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.adapters.stt import WordTiming
from pipeline.adapters.tts import Alignment
from pipeline.artifacts import file_hash, input_hash, is_cached, write_output
from pipeline.steps import script as script_step
from pipeline.steps import voice as voice_step
from pipeline.steps.base import StepContext, StepError, StepResult

OUTPUT = "captions.json"
CAPTIONS_VERSION = 2
MIN_MATCH = 0.6
MIN_WORD_PROBABILITY = voice_step.MIN_WORD_PROBABILITY
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


def fits(timings: Alignment, duration_s: float) -> bool:
    """Provider timings that can belong to this audio: inside it, never running backwards."""
    starts, ends = timings.starts_s, timings.ends_s
    return (
        bool(starts)
        and starts[0] >= 0
        and ends[-1] <= duration_s + _TOLERANCE_S
        and all(a <= b for a, b in itertools.pairwise(starts))
    )


def _timings(
    ctx: StepContext, audio: Path, duration_s: float, narration: str
) -> tuple[tuple[WordTiming, ...], str, float, tuple[str, ...]]:
    """(word timings, source used, cost, notes)."""
    if ctx.series.captions.timing_source == "tts":
        timings = voice_step.alignment(ctx.run_dir)
        if timings is not None and fits(timings, duration_s):
            return voice_step.alignment_words(timings), "tts", 0.0, ()
        note = (
            "timings from whisper: the voice provider reports none"
            if timings is None
            else "warning: the voice provider's timings do not fit the audio"
            " (edited without Alignment.edited?): timings from whisper"
        )
    else:
        note = "timings from whisper (captions.timing_source)"
    heard = voice_step.transcript(ctx.run_dir)
    if heard is not None:
        return heard, "whisper", 0.0, (note,)
    # A voice made before the voice check existed: listen now.
    transcript = ctx.stt.transcribe(audio, language=ctx.language, hint=narration)
    return transcript.words, "whisper", transcript.cost_usd, (note,)


def run(ctx: StepContext) -> StepResult:
    narration = script_step.load(ctx.run_dir).narration
    audio, duration = voice_step.voice_file(ctx.run_dir)
    per_chunk = ctx.series.captions.words_per_chunk
    output = ctx.run_dir / OUTPUT
    heard, source, cost, notes = _timings(ctx, audio, duration, narration)
    key = input_hash(
        {
            "step": "captions",
            "version": CAPTIONS_VERSION,
            "voice": file_hash(audio),
            "text": narration,
            "words_per_chunk": per_chunk,
            "source": source,
            "heard": [[w.word, w.start, w.end, w.probability] for w in heard],
        }
    )
    if is_cached(output, key):
        return StepResult(outputs=(output,), cost_usd=cost, cached=cost == 0, notes=notes)

    words, match = align(narration.split(), heard, duration)
    if match < MIN_MATCH:
        raise StepError(
            f"captions do not line up with the script: only {match:.0%} of the words"
            f" found in the {source} timings",
            retryable=False,
        )
    late = [w.text for w in words if w.end > duration + _TOLERANCE_S]
    if late:
        raise StepError(f"captions run past the end of the voice: {late[:3]}", retryable=False)

    chunks = chunk(words, per_chunk)
    data = {
        "timing_source": source,
        "match_ratio": round(match, 3),
        "words": [asdict(w) for w in words],
        "chunks": [asdict(c) for c in chunks],
    }
    write_output(
        output,
        json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
        key,
        {
            "timing_source": source,
            "cost_usd": cost,
            "match_ratio": round(match, 3),
            "words": len(words),
            "chunks": len(chunks),
        },
    )
    return StepResult(outputs=(output,), cost_usd=cost, cached=False, notes=notes)


def load(run_dir: Path) -> tuple[list[CaptionWord], list[Chunk]]:
    """Read captions written by `run` (used by render)."""
    path = run_dir / OUTPUT
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise StepError(f"{path} not found: run the captions step first", retryable=False) from e
    return [CaptionWord(**w) for w in data["words"]], [Chunk(**c) for c in data["chunks"]]
