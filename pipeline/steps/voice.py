"""Step `voice`: script narration -> runs/<id>/voice.<mp3|wav> via the TTS port."""

from pathlib import Path

from pipeline.adapters.tts import TtsRequest
from pipeline.artifacts import input_hash, is_cached, read_meta, write_output
from pipeline.steps import script as script_step
from pipeline.steps.base import StepContext, StepError, StepResult

PLACEHOLDER_VOICE = "CHANGE_ME"


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
    for existing in sorted(ctx.run_dir.glob("voice.*")):
        if existing.suffix in {".mp3", ".wav"} and is_cached(existing, key):
            return StepResult(outputs=(existing,), cost_usd=0.0, cached=True)

    result = ctx.tts.synthesize(request)
    output = ctx.run_dir / f"voice.{result.format}"
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
    return StepResult(outputs=(output,), cost_usd=result.cost_usd, cached=False)


def voice_file(run_dir: Path) -> tuple[Path, float]:
    """The voice file written by `run` and its measured duration (used by captions and render)."""
    for existing in sorted(run_dir.glob("voice.*")):
        meta = read_meta(existing)
        if meta is not None and existing.suffix in {".mp3", ".wav"}:
            return existing, float(meta["duration_s"])
    raise StepError("no voice file: run the voice step first", retryable=False)


def duration_s(run_dir: Path) -> float:
    return voice_file(run_dir)[1]
