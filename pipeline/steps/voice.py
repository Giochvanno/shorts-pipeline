"""Step `voice`: script narration -> runs/<id>/voice.<mp3|wav>, checked by listening to it.

1. Synthesize (TTS port). Cached by text and voice settings: a rerun never pays twice.
2. Listen (STT port: whisper for a real voice, the beep counter for the fake one)
   and compare with the script in spoken form ("243" == "two hundred forty-three").
   Below `qa.voice_min_match` the take is thrown away and a new one is made, at
   most `qa.voice_max_redos` times; then the step fails and the runner alerts.
3. Keep, tied to this exact voice file, what was heard (voice_transcript.json) and
   the provider's per-character timings if it reports them (voice_alignment.json).
   Captions take their word timings from one of the two, with no new call.
4. Compare the provider's word starts with whisper's. If more than DRIFT_SHARE of
   the words differ by more than DRIFT_S, the video report gets a warning: one of
   the two clocks is off, and captions may be too.
"""

import difflib
import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.adapters.stt import WordTiming
from pipeline.adapters.tts import Alignment, TtsRequest
from pipeline.artifacts import (
    file_hash,
    input_hash,
    invalidate,
    is_cached,
    read_meta,
    write_output,
)
from pipeline.steps import script as script_step
from pipeline.steps.base import StepContext, StepError, StepResult
from pipeline.text import spoken_tokens

PLACEHOLDER_VOICE = "CHANGE_ME"
AUDIO_SUFFIXES = (".mp3", ".wav")
ALIGNMENT = "voice_alignment.json"
TRANSCRIPT = "voice_transcript.json"

# Words whisper is unsure about do not count as heard. Seen live: on speech it could
# not understand, whisper echoed its vocabulary prompt back with p = 0.00-0.34.
MIN_WORD_PROBABILITY = 0.4
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


def run(ctx: StepContext) -> StepResult:
    narration = script_step.load(ctx.run_dir).narration
    voice, qa = ctx.series.voice, ctx.series.qa
    if ctx.tts.provider != "fake" and voice.voice_id == PLACEHOLDER_VOICE:
        raise StepError("set voice.voice_id in the series config", retryable=False)

    request = TtsRequest(
        text=narration,
        voice_id=voice.voice_id,
        stability=voice.stability,
        similarity_boost=voice.similarity_boost,
        speed=voice.speed,
    )
    key = input_hash(
        {
            "step": "voice",
            "text": narration,
            "voice": voice.model_dump(),
            "provider": ctx.tts.provider,
            "model": ctx.tts.model,
            "revision": ctx.revision,
        }
    )
    audio = next((p for p in _audio_files(ctx.run_dir) if is_cached(p, key)), None)
    from_cache = audio is not None
    takes, cost = 0, 0.0
    rejected: list[str] = []
    while True:
        if audio is None:
            if takes > qa.voice_max_redos:
                raise StepError(
                    f"the voice does not match the script after {takes} takes: whisper heard"
                    f" {', '.join(rejected)} of the words, {qa.voice_min_match:.0%} needed",
                    retryable=False,
                )
            audio, paid = _synthesize(ctx, request, key)
            takes, cost = takes + 1, cost + paid
        heard, listen_cost = _listen(ctx, audio, narration)
        cost += listen_cost
        match = voice_match(narration, heard)
        if match >= qa.voice_min_match:
            break
        rejected.append(f"{match:.0%}")
        _discard(ctx.run_dir)
        audio = None

    notes = [f"whisper heard {match:.1%} of the script" + (f" (take {takes})" if takes > 1 else "")]
    timings = alignment(ctx.run_dir)
    if timings is not None:
        drift = timing_drift(alignment_words(timings), heard)
        notes.append(
            f"timings voice vs whisper: {drift.over_notice}/{drift.compared} words"
            f" > {NOTICE_DRIFT_S} s, {drift.over_limit} > {DRIFT_S} s, worst {drift.worst_s:.2f} s"
        )
        if drift.too_much:
            notes.append(
                f"warning: over {DRIFT_SHARE:.0%} of word starts differ by > {DRIFT_S} s"
                " between the voice provider and whisper: check the captions"
            )
    return StepResult(
        outputs=(audio,), cost_usd=cost, cached=from_cache and takes == 0, notes=tuple(notes)
    )


def _synthesize(ctx: StepContext, request: TtsRequest, key: str) -> tuple[Path, float]:
    result = ctx.tts.synthesize(request)
    if not result.audio or result.duration_s <= 0:
        # Seen in the wild as empty bodies on a 200 OK: retry, never save it.
        raise StepError("TTS returned empty audio", retryable=True)
    output = ctx.run_dir / f"voice.{result.format}"
    _discard(ctx.run_dir)  # e.g. an old voice.wav next to a new voice.mp3
    write_output(
        output,
        result.audio,
        key,
        {
            "provider": result.provider,
            "model": result.model,
            "cost_usd": result.cost_usd,
            "characters": result.characters,
            "duration_s": round(result.duration_s, 3),
        },
    )
    if result.alignment is not None:
        timings = {
            "characters": list(result.alignment.characters),
            "starts_s": list(result.alignment.starts_s),
            "ends_s": list(result.alignment.ends_s),
        }
        write_output(
            ctx.run_dir / ALIGNMENT,
            json.dumps(timings, ensure_ascii=False).encode(),
            key,
            {"provider": result.provider, "model": result.model},
        )
    return output, result.cost_usd


def _listen(ctx: StepContext, audio: Path, narration: str) -> tuple[tuple[WordTiming, ...], float]:
    """What the STT heard in this voice file; transcribed once, then read from disk."""
    path = ctx.run_dir / TRANSCRIPT
    voice_hash = file_hash(audio)
    key = input_hash(
        {
            "step": "voice-check",
            "voice": voice_hash,
            "hint": narration,
            "language": ctx.language,
            "provider": ctx.stt.provider,
            "model": ctx.stt.model,
        }
    )
    if is_cached(path, key):
        return _read_words(path), 0.0
    transcript = ctx.stt.transcribe(audio, language=ctx.language, hint=narration)
    data = [[w.word, w.start, w.end, w.probability] for w in transcript.words]
    write_output(
        path,
        json.dumps(data, ensure_ascii=False).encode(),
        key,
        {"provider": transcript.provider, "model": transcript.model, "voice_hash": voice_hash},
    )
    return transcript.words, transcript.cost_usd


def _read_words(path: Path) -> tuple[WordTiming, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        WordTiming(word, start, end, probability) for word, start, end, probability in data
    )


def voice_match(script: str, heard: tuple[WordTiming, ...]) -> float:
    """Share of the script's spoken words found, in order, in what was heard."""
    expected = spoken_tokens(script)
    if not expected:
        return 0.0
    got = [
        token
        for w in heard
        if w.probability is None or w.probability >= MIN_WORD_PROBABILITY
        for token in spoken_tokens(w.word)
    ]
    blocks = difflib.SequenceMatcher(a=expected, b=got, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(expected)


def alignment_words(timings: Alignment) -> tuple[WordTiming, ...]:
    """Per-character timings grouped into words: a word spans its first to last character."""
    words: list[WordTiming] = []
    text, start, end = "", 0.0, 0.0
    for char, char_start, char_end in zip(
        timings.characters, timings.starts_s, timings.ends_s, strict=True
    ):
        if char.isspace():
            if text:
                words.append(WordTiming(text, start, end))
            text = ""
            continue
        if not text:
            start = char_start
        text, end = text + char, char_end
    if text:
        words.append(WordTiming(text, start, end))
    return tuple(words)


def timing_drift(voice: tuple[WordTiming, ...], heard: tuple[WordTiming, ...]) -> Drift:
    """Start differences for the words both sides agree on."""
    sure = [w for w in heard if w.probability is None or w.probability >= MIN_WORD_PROBABILITY]
    a = [" ".join(spoken_tokens(w.word)) for w in voice]
    b = [" ".join(spoken_tokens(w.word)) for w in sure]
    diffs = [
        abs(sure[block.b + k].start - voice[block.a + k].start)
        for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks()
        for k in range(block.size)
    ]
    return Drift(
        compared=len(diffs),
        over_notice=sum(d > NOTICE_DRIFT_S for d in diffs),
        over_limit=sum(d > DRIFT_S for d in diffs),
        worst_s=max(diffs, default=0.0),
    )


def _audio_files(run_dir: Path) -> list[Path]:
    return [p for p in sorted(run_dir.glob("voice.*")) if p.suffix in AUDIO_SUFFIXES]


def _remove(path: Path) -> None:
    invalidate(path)  # meta first: a crash in between leaves no meta, never a stale pair
    path.unlink(missing_ok=True)


def _discard(run_dir: Path) -> None:
    """Drop the voice and everything derived from it (a rejected take, or a new voice)."""
    for path in [*_audio_files(run_dir), run_dir / ALIGNMENT, run_dir / TRANSCRIPT]:
        _remove(path)


def voice_file(run_dir: Path) -> tuple[Path, float]:
    """The voice file written by `run` and its measured duration (used by captions and render)."""
    for existing in _audio_files(run_dir):
        meta = read_meta(existing)
        if meta is not None:
            return existing, float(meta["duration_s"])
    raise StepError("no voice file: run the voice step first", retryable=False)


def alignment(run_dir: Path) -> Alignment | None:
    """Provider timings for the current voice file, or None (fake TTS, stale or missing)."""
    voice, _ = voice_file(run_dir)
    path = run_dir / ALIGNMENT
    voice_meta, meta = read_meta(voice), read_meta(path)
    if voice_meta is None or meta is None or not path.exists():
        return None
    if meta.get("input_hash") != voice_meta.get("input_hash"):
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Alignment(
        characters=tuple(data["characters"]),
        starts_s=tuple(float(x) for x in data["starts_s"]),
        ends_s=tuple(float(x) for x in data["ends_s"]),
    )


def transcript(run_dir: Path) -> tuple[WordTiming, ...] | None:
    """What the voice check heard in the current voice file, or None if not for this file."""
    voice, _ = voice_file(run_dir)
    path = run_dir / TRANSCRIPT
    meta = read_meta(path)
    if meta is None or not path.exists() or meta.get("voice_hash") != file_hash(voice):
        return None
    return _read_words(path)


def duration_s(run_dir: Path) -> float:
    return voice_file(run_dir)[1]
