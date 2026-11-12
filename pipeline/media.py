"""Thin, testable wrappers around FFmpeg and ffprobe.

FFmpeg is called with an argument list (never a shell string), so paths with
spaces or Cyrillic letters on Windows need no quoting.
"""

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MediaError(Exception):
    """FFmpeg/ffprobe is missing or failed."""


@dataclass(frozen=True, slots=True)
class MediaInfo:
    duration_s: float
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    pix_fmt: str | None
    audio_codec: str | None
    sample_rate: int | None
    channels: int | None


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise MediaError(f"{name} not found in PATH: install FFmpeg (it ships ffprobe too)")
    return path


def run_ffmpeg(args: Sequence[str], *, cwd: Path | None = None, timeout_s: float = 600) -> None:
    # -xerror: stop at the first decode error. Without it, a broken looped image makes
    # FFmpeg retry forever and the run hangs until the timeout.
    command = [
        find_tool("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror", "-y",
        *args,
    ]  # fmt: skip
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"ffmpeg timed out after {timeout_s:.0f} s") from e
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-8:])
        raise MediaError(f"ffmpeg failed (exit {result.returncode}):\n{tail}")


def _rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    num, _, den = value.partition("/")
    return float(num) / float(den or 1)


def probe(path: Path) -> MediaInfo:
    command = [
        find_tool("ffprobe"),
        "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]  # fmt: skip
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    data: dict[str, Any] = json.loads(result.stdout)
    streams: list[dict[str, Any]] = data.get("streams", [])
    video: dict[str, Any] = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio: dict[str, Any] = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return MediaInfo(
        duration_s=float(data.get("format", {}).get("duration") or 0.0),  # images have none
        width=video.get("width"),
        height=video.get("height"),
        fps=_rate(video.get("r_frame_rate")),
        video_codec=video.get("codec_name"),
        pix_fmt=video.get("pix_fmt"),
        audio_codec=audio.get("codec_name"),
        sample_rate=int(audio["sample_rate"]) if "sample_rate" in audio else None,
        channels=audio.get("channels"),
    )


def check_image(path: Path) -> None:
    """Fail fast on an unreadable image instead of letting the render choke on it."""
    try:
        info = probe(path)
    except (MediaError, KeyError, ValueError) as e:
        raise MediaError(f"{path.name} is not a readable image: {e}") from e
    if not info.width or not info.height:
        raise MediaError(f"{path.name} is not a readable image")


def decode_pcm16(path: Path, *, sample_rate: int) -> bytes:
    """Any audio FFmpeg can read -> raw 16-bit little-endian mono PCM at `sample_rate`.

    Used to feed speech recognition, so the whole project decodes audio one way.
    """
    command = [
        find_tool("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]  # fmt: skip
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, timeout=120, check=False
        )
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"ffmpeg timed out decoding {path.name}") from e
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:]
        raise MediaError(f"ffmpeg could not decode {path.name}: {' '.join(detail)}")
    return result.stdout
