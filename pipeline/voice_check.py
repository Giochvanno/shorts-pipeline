"""Did the voice say the script? Listen, compare, and listen again where words were missed.

The voice step checks every take. Whisper hears the whole take once (with the
script's rare words as vocabulary) and the result is compared with the script in
spoken form. Missed words are either the voice's fault (a skipped or garbled
phrase: a new take is needed, and paid) or whisper's: on video 6 it skipped
"Number three." between two pauses although the voice clearly said it, and two
good takes were thrown away. So before paying for a new take, each missed
stretch is cut out and heard again on its own. Guards keep that second listen
from inventing words:

1. Only when few words are missing (`qa.voice_relisten_max_missed`): a voice
   that lost more did not say the script, and no listening will fix that.
2. Where: the voice provider's timings for those words (exact), else between
   the neighbouring words whisper did hear. The clip whisper hears is padded by
   PAD_S on both sides, so it holds parts of the neighbours too.
3. Speech must be there: voiced audio (loud against the take's own level) of at
   least MIN_SPEECH_PER_LETTER_S per letter of the missed words, measured on their
   own span, never on the padding (the neighbours are loud). Silence never passes.
4. Whisper must hear the words: first with no prompt at all. Only if that fails,
   with the expected words as its prompt. A prompt can make whisper "hear" its
   own text in silence or over another word, so that result counts only if
   - the provider's timings isolate the words (without them the prompt is
     never used: there is no telling silence from a neighbour's sound),
   - whisper is sure there is speech (no_speech_prob <= MAX_NO_SPEECH),
   - every expected word comes out sure (probability >= MIN_WORD_PROBABILITY),
   - and the plain listen did not clearly hear other words in that clip.
"""

import array
import difflib
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pipeline.adapters.stt import Transcriber, Transcript, WordTiming
from pipeline.media import decode_pcm16
from pipeline.text import spoken_tokens, spoken_words, vocabulary_words

# On speech it could not understand, whisper echoed its vocabulary prompt back with
# p = 0.00-0.34. So an unsure word counts as heard only if it is not a prompt word:
# a real but unsure "On" or "fifty" counts, an echoed "oxidase" does not.
MIN_WORD_PROBABILITY = 0.4

PAD_S = 0.3  # around a missed stretch: catches a word whose timing is a little off
SAMPLE_RATE = 16_000  # what whisper listens at
FRAME_S = 0.03
VOICED_LEVEL = 0.1  # a frame is speech above 10 % (-20 dB) of the take's loud level
# Measured on 198 real words (videos 4 and 6): 1 % had under 0.015 s of voice per letter,
# all short ones ("in", "of", "to"); the median was 0.048. The gate sits below that: it
# tells silence from speech, it does not judge pronunciation.
MIN_SPEECH_PER_LETTER_S = 0.01
MAX_NO_SPEECH = 0.3  # a prompted result counts only below this


def sure(word: WordTiming) -> bool:
    return word.probability is None or word.probability >= MIN_WORD_PROBABILITY


# --- comparing a transcript with the script ---------------------------------------------


@dataclass(frozen=True, slots=True)
class Comparison:
    tokens: tuple[str, ...]  # the script in spoken form
    words: tuple[int, ...]  # for each token, the index of its script word
    found: tuple[bool, ...]
    heard_at: tuple[int | None, ...]  # for each found token, the heard word it matched

    @property
    def match(self) -> float:
        return sum(self.found) / len(self.tokens) if self.tokens else 0.0

    @property
    def missed(self) -> list[str]:
        return [t for t, ok in zip(self.tokens, self.found, strict=True) if not ok]

    def runs(self) -> list[tuple[int, int]]:
        """Consecutive missed tokens as (first, last) indices."""
        runs: list[tuple[int, int]] = []
        for i, ok in enumerate(self.found):
            if ok:
                continue
            if runs and runs[-1][1] == i - 1:
                runs[-1] = (runs[-1][0], i)
            else:
                runs.append((i, i))
        return runs


def compare(script: str, heard: Sequence[WordTiming]) -> Comparison:
    """Which spoken words of the script appear, in order, in what was heard."""
    expected = spoken_words(script)
    echo_prone = {token for word in vocabulary_words(script) for token in spoken_tokens(word)}
    got = [
        (token, index)
        for index, w in enumerate(heard)
        for token in spoken_tokens(w.word)
        if sure(w) or token not in echo_prone
    ]
    tokens = [token for token, _ in expected]
    found = [False] * len(tokens)
    heard_at: list[int | None] = [None] * len(tokens)
    matcher = difflib.SequenceMatcher(a=tokens, b=[t for t, _ in got], autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            found[block.a + k] = True
            heard_at[block.a + k] = got[block.b + k][1]
    return Comparison(
        tokens=tuple(tokens),
        words=tuple(word for _, word in expected),
        found=tuple(found),
        heard_at=tuple(heard_at),
    )


# --- where the missed words are -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Stretch:
    first: int  # token indices, inclusive
    last: int
    start_s: float  # the clip to listen to again, padding included
    end_s: float
    core_start_s: float  # where the missed words themselves are: speech is measured here
    core_end_s: float
    context: frozenset[str]  # every token the script puts inside the clip
    exact: bool  # the core comes from the provider's timings (the prompt may be used)


def stretches(
    comparison: Comparison,
    heard: Sequence[WordTiming],
    provider_words: Sequence[WordTiming] | None,
    duration_s: float,
) -> list[Stretch]:
    word_count = max(comparison.words, default=-1) + 1
    exact = provider_words is not None and len(provider_words) == word_count
    out = []
    for first, last in comparison.runs():
        if exact:
            assert provider_words is not None
            core_start = provider_words[comparison.words[first]].start
            core_end = provider_words[comparison.words[last]].end
            start, end = core_start - PAD_S, core_end + PAD_S
            context = frozenset(
                token
                for token, word in zip(comparison.tokens, comparison.words, strict=True)
                if provider_words[word].end > start and provider_words[word].start < end
            )
        else:
            # Whisper may stretch a neighbour over the missed words (video 6: "Fungal" ran
            # 1.84-4.46 s over "Number three"), so take the neighbours whole.
            before = max((i for i in range(first) if comparison.found[i]), default=None)
            after = min(
                (i for i in range(last + 1, len(comparison.tokens)) if comparison.found[i]),
                default=None,
            )
            start = _heard(heard, comparison, before).start - PAD_S if before is not None else 0.0
            end = _heard(heard, comparison, after).end + PAD_S if after is not None else duration_s
            low = before if before is not None else 0
            high = after if after is not None else len(comparison.tokens) - 1
            context = frozenset(comparison.tokens[low : high + 1])
            core_start, core_end = start, end  # cannot be told apart from the neighbours
        start, end = max(0.0, start), min(duration_s, end)
        out.append(
            Stretch(
                first,
                last,
                start,
                end,
                max(start, core_start),
                min(end, core_end),
                context,
                exact,
            )
        )
    return out


def _heard(heard: Sequence[WordTiming], comparison: Comparison, token: int) -> WordTiming:
    index = comparison.heard_at[token]
    assert index is not None  # only called for found tokens
    return heard[index]


# --- is there speech --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Loudness:
    """Frame levels of a whole take, to measure speech in any part of it."""

    levels: tuple[float, ...]
    reference: float  # a loud frame of this take: speech is judged against the take itself

    @classmethod
    def of(cls, pcm: bytes) -> "Loudness":
        samples = array.array("h", pcm)
        step = int(SAMPLE_RATE * FRAME_S)
        levels = tuple(
            (sum(x * x for x in samples[i : i + step]) / step) ** 0.5
            for i in range(0, len(samples) - step + 1, step)
        )
        ranked = sorted(levels)
        reference = ranked[int(len(ranked) * 0.9)] if ranked else 0.0
        return cls(levels, reference)

    def voiced_s(self, start_s: float, end_s: float) -> float:
        if self.reference <= 0:
            return 0.0
        first, last = int(start_s / FRAME_S), int(end_s / FRAME_S)
        voiced = sum(level >= VOICED_LEVEL * self.reference for level in self.levels[first:last])
        return voiced * FRAME_S


# --- listening again --------------------------------------------------------------------


def speech_needed(expected: Sequence[str]) -> float:
    return MIN_SPEECH_PER_LETTER_S * sum(len(token) for token in expected)


def _in_order(expected: Sequence[str], got: Sequence[str]) -> bool:
    position = 0
    for token in got:
        if position < len(expected) and token == expected[position]:
            position += 1
    return position == len(expected)


def accepts(
    expected: Sequence[str],
    context: frozenset[str],
    speech_s: float,
    plain: Transcript,
    prompted: Transcript | None,
) -> bool:
    """The guarded decision for one missed stretch (see the module docstring)."""
    if speech_s <= 0 or speech_s < speech_needed(expected):
        return False  # silence, or too little voice for these words
    if _in_order(expected, [t for w in plain.words for t in spoken_tokens(w.word)]):
        return True  # heard with no prompt: nothing to echo
    if prompted is None or prompted.no_speech_prob > MAX_NO_SPEECH:
        return False
    other = [t for w in plain.words if sure(w) for t in spoken_tokens(w.word) if t not in context]
    if other:
        return False  # whisper clearly heard different words here
    return _in_order(
        expected, [t for w in prompted.words if sure(w) for t in spoken_tokens(w.word)]
    )


@dataclass(frozen=True, slots=True)
class Rescue:
    found: tuple[int, ...]  # token indices heard on the second listen
    tried: int  # stretches listened to again


def relisten(
    stt: Transcriber,
    audio: Path,
    comparison: Comparison,
    heard: Sequence[WordTiming],
    provider_words: Sequence[WordTiming] | None,
    *,
    language: str,
    duration_s: float,
) -> Rescue:
    pcm = decode_pcm16(audio, sample_rate=SAMPLE_RATE)
    loudness = Loudness.of(pcm)
    found: list[int] = []
    todo = stretches(comparison, heard, provider_words, duration_s)
    with tempfile.TemporaryDirectory(prefix="relisten_", ignore_cleanup_errors=True) as tmp:
        for n, stretch in enumerate(todo):
            expected = comparison.tokens[stretch.first : stretch.last + 1]
            speech_s = loudness.voiced_s(stretch.core_start_s, stretch.core_end_s)
            if speech_s <= 0 or speech_s < speech_needed(expected):
                continue  # no need to ask whisper about silence
            clip = Path(tmp) / f"clip{n}.wav"
            _write_clip(pcm, stretch.start_s, stretch.end_s, clip)
            plain = stt.transcribe(clip, language=language)
            prompted = None
            heard_plainly = accepts(expected, stretch.context, speech_s, plain, None)
            if not heard_plainly and stretch.exact:
                prompted = stt.transcribe(clip, language=language, expect=" ".join(expected))
            if accepts(expected, stretch.context, speech_s, plain, prompted):
                found += range(stretch.first, stretch.last + 1)
    return Rescue(found=tuple(found), tried=len(todo))


def _write_clip(pcm: bytes, start_s: float, end_s: float, path: Path) -> None:
    first, last = int(start_s * SAMPLE_RATE) * 2, int(end_s * SAMPLE_RATE) * 2
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(SAMPLE_RATE)
        clip.writeframes(pcm[first:last])


# --- the whole check ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VoiceCheck:
    first_match: float  # the whole-take listen
    match: float  # after the second listen
    missed: tuple[str, ...]  # still missed
    rescued: tuple[str, ...]  # found on the second listen
    relistened: bool


def check(
    stt: Transcriber,
    audio: Path,
    script: str,
    heard: Sequence[WordTiming],
    provider_words: Sequence[WordTiming] | None,
    *,
    language: str,
    duration_s: float,
    min_match: float,
    max_missed: float,
) -> VoiceCheck:
    comparison = compare(script, heard)
    first = comparison.match
    if first >= min_match or 1 - first > max_missed or not comparison.tokens:
        missed = tuple(comparison.missed)
        return VoiceCheck(first, first, missed, (), relistened=False)
    rescue = relisten(
        stt, audio, comparison, heard, provider_words, language=language, duration_s=duration_s
    )
    found = list(comparison.found)
    for i in rescue.found:
        found[i] = True
    tokens = comparison.tokens
    return VoiceCheck(
        first_match=first,
        match=sum(found) / len(tokens),
        missed=tuple(t for t, ok in zip(tokens, found, strict=True) if not ok),
        rescued=tuple(tokens[i] for i in rescue.found),
        relistened=True,
    )


# --- the provider's clock against whisper's -------------------------------------------

NOTICE_DRIFT_S = 0.2  # counted in the report
DRIFT_S = 0.5  # a word this far off is visibly early or late on screen
DRIFT_SHARE = 0.15  # more such words than this share -> warning


@dataclass(frozen=True, slots=True)
class Drift:
    """How far the provider's word starts are from whisper's."""

    compared: int
    over_notice: int
    over_limit: int
    worst_s: float

    @property
    def too_much(self) -> bool:
        return self.compared > 0 and self.over_limit / self.compared > DRIFT_SHARE


def timing_drift(voice: Sequence[WordTiming], heard: Sequence[WordTiming]) -> Drift:
    """Start differences for the words both sides agree on."""
    confident = [w for w in heard if sure(w)]
    a = [" ".join(spoken_tokens(w.word)) for w in voice]
    b = [" ".join(spoken_tokens(w.word)) for w in confident]
    diffs = [
        abs(confident[block.b + k].start - voice[block.a + k].start)
        for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks()
        for k in range(block.size)
    ]
    return Drift(
        compared=len(diffs),
        over_notice=sum(d > NOTICE_DRIFT_S for d in diffs),
        over_limit=sum(d > DRIFT_S for d in diffs),
        worst_s=max(diffs, default=0.0),
    )
