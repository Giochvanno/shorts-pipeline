"""Step `voice`: script narration -> runs/<id>/voice.<mp3|wav>, checked by listening to it.

1. Synthesize (TTS port). Cached by text and voice settings: a rerun never pays twice.
2. Check it (pipeline.voice_check): whisper hears the whole take, the result is
   compared with the script in spoken form, and missed stretches are heard again
   on their own, with guards, before a take is blamed. Below `qa.voice_min_match`
   the take is set aside (rejected_voice/, to listen to) and a new one is made, at
   most `qa.voice_max_redos` times; then the step fails, naming the missed words.
3. Keep, tied to this exact voice file, what was heard (voice_transcript.json) and
   the provider's per-character timings if it reports them (voice_alignment.json).
   Captions take their word timings from one of the two, with no new call.
4. Compare the provider's word starts with whisper's: if many differ by more than
   DRIFT_S, the video report gets a warning (one of the two clocks is off).
"""

import json
import shutil
from pathlib import Path

from pipeline.adapters.stt import Transcriber, WordTiming
from pipeline.adapters.tts import Alignment, TtsRequest
from pipeline.artifacts import (
    file_hash,
    input_hash,
    invalidate,
    is_cached,
    read_meta,
    write_output,
)
from pipeline.config import SeriesConfig
from pipeline.media import probe
from pipeline.steps import script as script_step
from pipeline.steps.base import StepContext, StepError, StepResult
from pipeline.voice_check import (
    DRIFT_S,
    DRIFT_SHARE,
    MIN_WORD_PROBABILITY,
    NOTICE_DRIFT_S,
    VoiceCheck,
    check,
    compare,
    timing_drift,
)

__all__ = ["MIN_WORD_PROBABILITY", "compare", "timing_drift"]  # re-exported for callers

PLACEHOLDER_VOICE = "CHANGE_ME"
AUDIO_SUFFIXES = (".mp3", ".wav")
ALIGNMENT = "voice_alignment.json"
TRANSCRIPT = "voice_transcript.json"

REJECTED = "rejected_voice"  # rejected takes, kept to listen to; nothing reads them


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
    result: VoiceCheck | None = None
    while True:
        if audio is None:
            if takes > qa.voice_max_redos:
                assert result is not None
                missed = list(result.missed)
                shown = ", ".join(missed[:10]) + (" ..." if len(missed) > 10 else "")
                raise StepError(
                    f"the voice does not match the script after {takes} takes: whisper heard"
                    f" {', '.join(rejected)} of the words, {qa.voice_min_match:.0%} needed;"
                    f" missed in the last take: {shown} (takes kept in {REJECTED}/)",
                    retryable=False,
                )
            audio, paid = _synthesize(ctx, request, key)
            takes, cost = takes + 1, cost + paid
        heard, listen_cost = _listen(ctx, audio, narration)
        cost += listen_cost
        timings = alignment(ctx.run_dir)
        result = check(
            ctx.stt,
            audio,
            narration,
            heard,
            alignment_words(timings) if timings is not None else None,
            language=ctx.language,
            duration_s=voice_file(ctx.run_dir)[1],
            min_match=qa.voice_min_match,
            max_missed=qa.voice_relisten_max_missed,
        )
        if result.match >= qa.voice_min_match:
            break
        rejected.append(f"{result.match:.0%}")
        _keep_rejected(ctx, audio, heard, result)
        _discard(ctx.run_dir)
        audio = None

    return StepResult(
        outputs=(audio,),
        cost_usd=cost,
        cached=from_cache and takes == 0,
        notes=_notes(result, takes, heard, alignment(ctx.run_dir)),
    )


def _notes(
    result: VoiceCheck, takes: int, heard: tuple[WordTiming, ...], timings: Alignment | None
) -> tuple[str, ...]:
    take = f" (take {takes})" if takes > 1 else ""
    if result.rescued:
        notes = [
            f"whisper heard {result.first_match:.1%} of the script; the second listen found"
            f" {len(result.rescued)} more ({', '.join(result.rescued)}): {result.match:.1%}{take}"
        ]
    else:
        notes = [f"whisper heard {result.match:.1%} of the script{take}"]
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
    return tuple(notes)


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


def voice_match(script: str, heard: tuple[WordTiming, ...]) -> float:
    """Share of the script heard in order, before any second listen."""
    return compare(script, heard).match


def _keep_rejected(
    ctx: StepContext, audio: Path, heard: tuple[WordTiming, ...], result: VoiceCheck
) -> None:
    """Copy a rejected take aside, so a human can hear whether the voice or whisper was wrong.

    Kept inside the video's run folder (git ignores runs/), so it goes with the video.
    The folder is capped at `qa.rejected_takes_max_mb`: the oldest takes go first,
    the newest always stays.
    """
    folder = ctx.run_dir / REJECTED
    folder.mkdir(exist_ok=True)
    numbers = [int(p.stem[4:]) for p in folder.glob("take*.json") if p.stem[4:].isdigit()]
    number = max(numbers, default=0) + 1
    shutil.copyfile(audio, folder / f"take{number}{audio.suffix}")
    timings = ctx.run_dir / ALIGNMENT
    if timings.exists():
        shutil.copyfile(timings, folder / f"take{number}.alignment.json")
    report = {
        "match": round(result.match, 3),
        "first_match": round(result.first_match, 3),
        "missed": list(result.missed),
        "rescued": list(result.rescued),
        "heard": [[w.word, w.start, w.end, w.probability] for w in heard],
    }
    (folder / f"take{number}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    _cap(folder, int(ctx.series.qa.rejected_takes_max_mb * 1024 * 1024))


def _cap(folder: Path, max_bytes: int) -> None:
    def take_of(path: Path) -> int:
        digits = path.name[4:].split(".")[0]
        return int(digits) if digits.isdigit() else 0

    files = sorted(folder.glob("take*"), key=take_of)
    total = sum(p.stat().st_size for p in files)
    # The newest take always stays: it is the one a human will want to hear.
    while len({take_of(p) for p in files}) > 1 and total > max_bytes:
        oldest = take_of(files[0])
        for path in [p for p in files if take_of(p) == oldest]:
            total -= path.stat().st_size
            path.unlink()
            files.remove(path)


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
    return read_alignment(path)


def read_alignment(path: Path) -> Alignment:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Alignment(
        characters=tuple(data["characters"]),
        starts_s=tuple(float(x) for x in data["starts_s"]),
        ends_s=tuple(float(x) for x in data["ends_s"]),
    )


def recheck(
    run_dir: Path, series: SeriesConfig, stt: Transcriber, language: str
) -> list[tuple[str, VoiceCheck]]:
    """Run today's voice check on every take kept for a video: rejected ones and the voice.

    Only listens (local, free): nothing is synthesized and nothing in run_dir changes.
    """
    narration = script_step.load(run_dir).narration
    takes: list[tuple[Path, Alignment | None]] = []
    folder = run_dir / REJECTED
    for audio in sorted(folder.glob("take*")) if folder.is_dir() else []:
        if audio.suffix in AUDIO_SUFFIXES:
            saved = folder / f"{audio.stem}.alignment.json"
            takes.append((audio, read_alignment(saved) if saved.exists() else None))
    if _audio_files(run_dir):
        takes.append((voice_file(run_dir)[0], alignment(run_dir)))
    results = []
    for audio, timings in takes:
        heard = stt.transcribe(audio, language=language, hint=narration).words
        result = check(
            stt,
            audio,
            narration,
            heard,
            alignment_words(timings) if timings is not None else None,
            language=language,
            duration_s=probe(audio).duration_s,
            min_match=series.qa.voice_min_match,
            max_missed=series.qa.voice_relisten_max_missed,
        )
        results.append((audio.relative_to(run_dir).as_posix(), result))
    return results


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
