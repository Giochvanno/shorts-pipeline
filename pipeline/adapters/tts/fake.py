"""Fake TTS: a real, playable WAV with one short beep per word.

The timing matches a normal speaking rate, so durations and the render step
behave like they will with real speech.
"""

import io
import math
import struct
import wave

from pipeline.adapters.tts.base import TtsRequest, TtsResult

SAMPLE_RATE = 22_050
BEEP_S = 0.25
GAP_S = 0.15  # BEEP_S + GAP_S = 0.4 s per word = 2.5 words per second
TONE_HZ = 440.0


def _beep_and_gap() -> bytes:
    beep = bytearray()
    for i in range(int(SAMPLE_RATE * BEEP_S)):
        value = int(8000 * math.sin(2 * math.pi * TONE_HZ * i / SAMPLE_RATE))
        beep += struct.pack("<h", value)
    silence = b"\x00\x00" * int(SAMPLE_RATE * GAP_S)
    return bytes(beep) + silence


_WORD = _beep_and_gap()


class FakeTts:
    def __init__(self, model: str = "fake-tts") -> None:
        self._model = model
        self.calls: list[TtsRequest] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def synthesize(self, request: TtsRequest) -> TtsResult:
        self.calls.append(request)
        words = len(request.text.split())
        if words == 0:
            raise ValueError("nothing to synthesize: empty text")
        frames = _WORD * words
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(frames)
        return TtsResult(
            audio=buffer.getvalue(),
            format="wav",
            duration_s=len(frames) / 2 / SAMPLE_RATE,
            characters=len(request.text),
            provider=self.provider,
            model=self.model,
            cost_usd=0.0,
        )
