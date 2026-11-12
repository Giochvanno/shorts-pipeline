"""Real speech-to-text: faster-whisper on the CPU, locally, $0.

- The model named in config (`models.stt.primary.model`, e.g. small.en) is
  downloaded from Hugging Face on first use and cached; loading takes seconds,
  so one adapter keeps it in memory for the whole run.
- faster-whisper is an optional dependency (`uv sync --extra stt`) and is
  imported only when the model is first needed: CI runs on the fake.
- Only rare words from the script go into whisper's prompt, as vocabulary.
  The whole script would make whisper echo it back even where the voice
  skipped words, and then the "voice matches script" check would be blind.
- Audio is decoded by FFmpeg (already required and tested in CI) and handed
  over as samples. faster-whisper's own decoder goes through PyAV, and
  faster-whisper 1.2.1 + PyAV 19 break on `av.open(metadata_errors=...)`;
  passing samples never touches that code.

API checked against faster-whisper 1.2.1.
"""

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from pipeline.adapters.errors import PermanentAdapterError, TransientAdapterError
from pipeline.adapters.stt.base import Transcript, WordTiming
from pipeline.media import MediaError, decode_pcm16

PROVIDER = "faster_whisper"
SAMPLE_RATE = 16_000  # what whisper models expect
_MAX_PROMPT_CHARS = 600  # whisper's prompt window is ~224 tokens


class WhisperModelLike(Protocol):
    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Iterable[Any], Any]: ...


def decode_for_whisper(path: Path) -> Any:
    """Mono float32 samples in [-1, 1] at 16 kHz, decoded by FFmpeg."""
    try:
        pcm = decode_pcm16(path, sample_rate=SAMPLE_RATE)
    except MediaError as e:  # missing FFmpeg or a broken voice file: retrying will not help
        raise PermanentAdapterError(str(e)) from e
    import numpy as np  # installed with the `stt` extra, together with faster-whisper

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def load_faster_whisper(name: str) -> WhisperModelLike:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise PermanentAdapterError(
            "faster-whisper is not installed: run `uv sync --extra stt`"
        ) from e
    try:
        model: WhisperModelLike = WhisperModel(name, device="cpu", compute_type="int8")
    except ValueError as e:  # unknown model size or bad option
        raise PermanentAdapterError(f"whisper model {name!r} cannot be loaded: {e}") from e
    except Exception as e:  # download or disk problems: worth another try later
        raise TransientAdapterError(f"whisper model {name!r} failed to load: {e}") from e
    return model


def vocabulary(text: str) -> str:
    """Rare-looking words (capitalised or long), as a hint for spelling, not as a script."""
    seen: dict[str, None] = {}
    for raw in text.split():
        word = re.sub(r"[^\w'-]", "", raw)
        if len(word) >= 7 or (word[:1].isupper() and len(word) > 1):
            seen.setdefault(word, None)
    return ", ".join(seen)[:_MAX_PROMPT_CHARS]


class WhisperTranscriber:
    def __init__(
        self,
        model: str,
        *,
        load: Callable[[str], WhisperModelLike] = load_faster_whisper,
        decode: Callable[[Path], Any] = decode_for_whisper,
    ) -> None:
        self._model_name = model
        self._load = load
        self._decode = decode
        self._loaded: WhisperModelLike | None = None

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def model(self) -> str:
        return self._model_name

    def transcribe(self, audio: Path, *, language: str, hint: str | None = None) -> Transcript:
        if self._loaded is None:
            self._loaded = self._load(self._model_name)
        samples = self._decode(audio)
        try:
            segments, info = self._loaded.transcribe(
                samples,
                language=language,
                word_timestamps=True,
                initial_prompt=vocabulary(hint) if hint else None,
                condition_on_previous_text=False,  # avoids repetition loops
                vad_filter=False,  # keep word times relative to the original audio
                beam_size=5,
            )
            # Segments are a generator: decoding happens while iterating.
            words = tuple(
                WordTiming(
                    word=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(w.probability),
                )
                for segment in segments
                for w in (segment.words or [])
                if w.word.strip()
            )
        except (PermanentAdapterError, TransientAdapterError):
            raise
        except Exception as e:
            raise TransientAdapterError(f"whisper failed on {audio.name}: {e}") from e
        return Transcript(
            words=words,
            duration_s=float(info.duration),
            provider=PROVIDER,
            model=self._model_name,
        )
