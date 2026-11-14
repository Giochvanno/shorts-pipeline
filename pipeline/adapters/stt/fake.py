"""Fake transcriber: really listens to the audio, but only for beeps.

FakeTts writes one beep per word, so the fake finds the beeps by loudness and
returns one timing per beep. Words come from the hint (the script), so captions
built on fakes are exact. Real speech needs the real adapter (faster-whisper).
"""

import array
import wave
from pathlib import Path

from pipeline.adapters.stt.base import Transcript, WordTiming

WINDOW_S = 0.01
LOUD = 1000  # 16-bit amplitude: the fake beep peaks at 8000, silence is 0


def find_beeps(audio: Path) -> tuple[list[tuple[float, float]], float]:
    """(start, end) of every loud stretch in a 16-bit mono WAV, and its duration."""
    with wave.open(str(audio), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError("FakeTranscriber reads 16-bit mono WAV only (what FakeTts writes)")
        rate = wav.getframerate()
        samples = array.array("h", wav.readframes(wav.getnframes()))
    step = max(int(rate * WINDOW_S), 1)
    beeps: list[tuple[float, float]] = []
    start: float | None = None
    for i in range(0, len(samples), step):
        loud = max((abs(s) for s in samples[i : i + step]), default=0) >= LOUD
        t = i / rate
        if loud and start is None:
            start = t
        elif not loud and start is not None:
            beeps.append((start, t))
            start = None
    duration = len(samples) / rate
    if start is not None:
        beeps.append((start, duration))
    return beeps, duration


class FakeTranscriber:
    def __init__(self, model: str = "fake-stt") -> None:
        self._model = model
        self.calls: list[Path] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def transcribe(
        self, audio: Path, *, language: str, hint: str | None = None, expect: str | None = None
    ) -> Transcript:
        self.calls.append(audio)
        beeps, duration = find_beeps(audio)
        expected = (expect or hint or "").split()
        words = tuple(
            WordTiming(
                word=expected[i] if i < len(expected) else f"word{i + 1}",
                start=round(start, 3),
                end=round(end, 3),
                probability=1.0,
            )
            for i, (start, end) in enumerate(beeps)
        )
        return Transcript(
            words=words,
            duration_s=duration,
            provider=self.provider,
            model=self.model,
            no_speech_prob=0.0 if words else 1.0,
        )
