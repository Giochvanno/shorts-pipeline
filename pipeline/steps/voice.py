"""Step `voice`: script narration -> runs/<id>/voice.<mp3|wav> via the TTS port.

If the provider reports when each character is spoken (ElevenLabs does), the
timings go to `voice_alignment.json` with the same cache key as the audio:
they are free now and would cost a second synthesis later.
"""

import json
from pathlib import Path

from pipeline.adapters.tts import Alignment, TtsRequest
from pipeline.artifacts import input_hash, invalidate, is_cached, read_meta, write_output
from pipeline.steps import script as script_step
from pipeline.steps.base import StepContext, StepError, StepResult

PLACEHOLDER_VOICE = "CHANGE_ME"
AUDIO_SUFFIXES = (".mp3", ".wav")
ALIGNMENT = "voice_alignment.json"


def run(ctx: StepContext) -> StepResult:
    narration = script_step.load(ctx.run_dir).narration
    voice = ctx.series.voice
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
    for existing in _audio_files(ctx.run_dir):
        if is_cached(existing, key):
            return StepResult(outputs=(existing,), cost_usd=0.0, cached=True)

    result = ctx.tts.synthesize(request)
    if not result.audio or result.duration_s <= 0:
        # Seen in the wild as empty bodies on a 200 OK: retry, never save it.
        raise StepError("TTS returned empty audio", retryable=True)
    output = ctx.run_dir / f"voice.{result.format}"
    for stale in _audio_files(ctx.run_dir):  # e.g. an old voice.wav next to a new voice.mp3
        if stale != output:
            _remove(stale)
    _remove(ctx.run_dir / ALIGNMENT)
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
    return StepResult(outputs=(output,), cost_usd=result.cost_usd, cached=False)


def _audio_files(run_dir: Path) -> list[Path]:
    return [p for p in sorted(run_dir.glob("voice.*")) if p.suffix in AUDIO_SUFFIXES]


def _remove(path: Path) -> None:
    invalidate(path)  # meta first: a crash in between leaves no meta, never a stale pair
    path.unlink(missing_ok=True)


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
    if voice_meta is None or meta is None or meta.get("input_hash") != voice_meta.get("input_hash"):
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Alignment(
        characters=tuple(data["characters"]),
        starts_s=tuple(float(x) for x in data["starts_s"]),
        ends_s=tuple(float(x) for x in data["ends_s"]),
    )


def duration_s(run_dir: Path) -> float:
    return voice_file(run_dir)[1]
