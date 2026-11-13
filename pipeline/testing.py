"""Helpers for tests, demos and soak runs. Fakes only: nothing here spends money."""

from dataclasses import replace
from pathlib import Path

from pipeline.adapters.images.fake import FakeImages
from pipeline.adapters.llm.fake import FakeLlm
from pipeline.adapters.stt.fake import FakeTranscriber
from pipeline.adapters.tts import Alignment, AudioFormat, TtsRequest, TtsResult
from pipeline.adapters.tts.fake import BEEP_S, GAP_S, FakeTts
from pipeline.config import load_series
from pipeline.steps.base import StepContext

DEMO_TOPIC = "why octopuses have three hearts"


def fake_context(
    run_dir: Path, *, video_id: int = 1, topic: str = DEMO_TOPIC, series_id: str = "fact-of-day"
) -> StepContext:
    """A step context wired to fresh fakes and the real series config."""
    return StepContext(
        video_id=video_id,
        topic=topic,
        series=load_series(series_id),
        run_dir=run_dir,
        llm_heavy=FakeLlm(),
        llm_light=FakeLlm(),
        tts=FakeTts(),
        images=FakeImages(),
        stt=FakeTranscriber(),
    )


class TimedFakeTts(FakeTts):
    """FakeTts that also reports per-character timings, like ElevenLabs does.

    The timings match its beeps exactly (each word's characters spread over the
    word's beep), shifted by `offset_s` to simulate a provider whose clock is off.
    `audio_format="mp3"` only renames the file: the bytes stay WAV, which the fake
    transcriber reads by content, so the voice check still hears the beeps.
    """

    def __init__(
        self, *, offset_s: float = 0.0, audio_format: AudioFormat = "wav", provider: str = "timed"
    ) -> None:
        super().__init__()
        self.offset_s = offset_s
        self.audio_format: AudioFormat = audio_format
        self._provider = provider

    @property
    def provider(self) -> str:
        return self._provider

    def synthesize(self, request: TtsRequest) -> TtsResult:
        result = super().synthesize(request)
        text = request.text.strip()
        chars, starts, ends = [], [], []
        word, position, length = 0, 0, 0
        for i, char in enumerate(text):
            if char.isspace():
                if position:  # first space after a word: the word is over
                    word, position = word + 1, 0
                begin = word * (BEEP_S + GAP_S) - GAP_S
                start, end = begin, begin + GAP_S
            else:
                if position == 0:
                    length = len(text[i:].split(maxsplit=1)[0])
                begin = word * (BEEP_S + GAP_S)
                start = begin + BEEP_S * position / length
                end = begin + BEEP_S * (position + 1) / length
                position += 1
            chars.append(char)
            starts.append(round(start + self.offset_s, 4))
            ends.append(round(end + self.offset_s, 4))
        return replace(
            result,
            format=self.audio_format,
            provider=self._provider,
            alignment=Alignment(tuple(chars), tuple(starts), tuple(ends)),
        )
