"""Live check of faster-whisper on real speech. Never runs in CI.

    uv sync --extra stt
    uv run pytest -m live -s

Real speech comes from the speech synthesizer built into Windows (System.Speech),
or espeak-ng on Linux, so this needs no ElevenLabs key. The first run downloads
the whisper model from config (small.en is ~480 MB) and caches it.

It prints what whisper heard, with and without the vocabulary prompt, so a low
match can be diagnosed instead of guessed.
"""

import array
import math
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

from pipeline.adapters.stt import Transcript
from pipeline.adapters.stt.whisper import WhisperTranscriber, decode_for_whisper, vocabulary
from pipeline.config import load_channel
from pipeline.steps import captions

pytestmark = pytest.mark.live

TEXT = "Honey outlives pharaohs. Bees add an enzyme called glucose oxidase."


def _speak(text: str, out: Path) -> None:
    if sys.platform == "win32":
        # The default voice follows the Windows language: a Korean or Russian voice reading
        # English is not English speech for an English-only model. Pick an English voice.
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$all = $s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo }; "
            "$all | ForEach-Object "
            "{ Write-Output ('voice: ' + $_.Name + ' / ' + $_.Culture.Name) }; "
            "$en = $all | Where-Object { $_.Culture.Name -like 'en-*' } | Select-Object -First 1; "
            "if (-not $en) { Write-Output 'NO_ENGLISH_VOICE'; exit 0 }; "
            "$s.SelectVoice($en.Name); Write-Output ('using: ' + $en.Name); "
            f"$s.SetOutputToWaveFile('{out}'); $s.Speak('{text}'); $s.Dispose()"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, check=True, timeout=60,
        )  # fmt: skip
        print("\n" + result.stdout.strip())
        if "NO_ENGLISH_VOICE" in result.stdout:
            pytest.skip(
                "no English Windows voice:"
                " Settings > Time & language > Speech > Add voices > English"
            )
    elif shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-w", str(out), text], check=True, timeout=60)
    else:
        pytest.skip("no speech synthesizer (Windows System.Speech or espeak-ng)")


def _report(label: str, transcript: Transcript, elapsed: float) -> float:
    words, match = captions.align(TEXT.split(), transcript.words, transcript.duration_s)
    print(f"\n[{label}] {elapsed:.1f} s, match {match:.0%}, audio {transcript.duration_s:.2f} s")
    print("  heard:  " + " | ".join(
        f"{w.word} {w.start:.2f}-{w.end:.2f} p={w.probability or 0:.2f}" for w in transcript.words
    ))  # fmt: skip
    print("  script: " + " | ".join(f"{w.text} {w.start:.2f}-{w.end:.2f}" for w in words))
    return match


def _check_audio(audio: Path) -> None:
    """Is there sound in the file at all, and does our decoder see the same sound?"""
    with wave.open(str(audio), "rb") as wav:
        rate, channels, width, frames = (
            wav.getframerate(),
            wav.getnchannels(),
            wav.getsampwidth(),
            wav.getnframes(),
        )
        raw = wav.readframes(frames)
    print(
        f"\nwav: {rate} Hz, {channels} ch, {width * 8}-bit, {frames / rate:.2f} s, {len(raw)} bytes"
    )
    if width == 2:
        pcm = array.array("h", raw)
        peak = max((abs(x) for x in pcm), default=0) / 32768
        rms = math.sqrt(sum(x * x for x in pcm) / max(len(pcm), 1)) / 32768
        print(f"python wave:    peak {peak:.3f}, rms {rms:.4f}")
    samples = decode_for_whisper(audio)
    peak = float(abs(samples).max()) if len(samples) else 0.0
    rms = float((samples**2).mean() ** 0.5) if len(samples) else 0.0
    print(f"ffmpeg decoder: peak {peak:.3f}, rms {rms:.4f}, {len(samples) / 16000:.2f} s at 16 kHz")
    keep = Path("runs") / "live" / "speech.wav"
    keep.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(audio, keep)
    print(f"listen to it: {keep.resolve()}")


def test_real_whisper_times_real_speech(tmp_path: Path) -> None:
    pytest.importorskip("faster_whisper", reason="run `uv sync --extra stt` first")
    audio = tmp_path / "speech.wav"
    _speak(TEXT, audio)
    _check_audio(audio)
    stt = WhisperTranscriber(load_channel().models.stt.primary.model)
    print(f"\nmodel {stt.model}; vocabulary prompt: {vocabulary(TEXT)!r}")

    started = time.monotonic()
    with_prompt = stt.transcribe(audio, language="en", hint=TEXT)
    first = time.monotonic() - started  # includes loading the model
    started = time.monotonic()
    without_prompt = stt.transcribe(audio, language="en", hint=None)
    second = time.monotonic() - started

    match_prompt = _report("with vocabulary prompt", with_prompt, first)
    match_plain = _report("no prompt", without_prompt, second)

    assert max(match_prompt, match_plain) >= 0.8
    assert all(w.start <= w.end for w in with_prompt.words)