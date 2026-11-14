"""The voice check on the real takes of video 6, with real whisper. Never run in CI.

    uv run pytest tests/live/test_voice_recheck_live.py -m live -s

Video 6 lost two good takes: whisper skipped "Number three." / "Number two." /
"Number one." between pauses and wrote "85 %" apart. This replays today's check
on the takes kept in runs/6/rejected_voice/ (local and free: nothing is
synthesized, nothing in runs/6 changes) and prints what each take scores.
Same as `python -m pipeline recheck --video 6`.
"""

import importlib.util
from pathlib import Path

import pytest

from pipeline.adapters.stt.whisper import WhisperTranscriber
from pipeline.config import load_channel, load_series
from pipeline.steps import voice

pytestmark = pytest.mark.live

RUN = Path(__file__).resolve().parents[2] / "runs" / "6"


@pytest.mark.usefixtures("ffmpeg")
def test_rejected_takes_of_video_6_pass_today() -> None:
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("whisper is not installed: uv sync --extra stt")
    if not (RUN / voice.REJECTED / "take1.mp3").exists():
        pytest.skip("runs/6/rejected_voice/take1.mp3 is not on this machine")
    channel = load_channel()
    stt = WhisperTranscriber(channel.models.stt.primary.model)

    results = dict(voice.recheck(RUN, load_series("fact-of-day"), stt, "en"))

    for name, result in results.items():
        found = f", second listen found {', '.join(result.rescued)}" if result.rescued else ""
        print(f"\n{name}: {result.first_match:.1%} -> {result.match:.1%}{found}")
        print(f"  still missed: {', '.join(result.missed) or '-'}")
    take1 = results["rejected_voice/take1.mp3"]
    assert take1.match >= 0.95
    assert "three" not in take1.missed and "two" not in take1.missed
