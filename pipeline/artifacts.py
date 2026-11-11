"""Step outputs on disk: atomic writes and a cache keyed by the hash of the inputs.

Each output file gets a sidecar ``<file>.meta.json`` with the hash of everything
that produced it (topic, config, model, prompt, revision). If a step runs again
with the same inputs, it finds a matching hash and skips the paid call.
Change any input and the hash changes, so the cache can never serve stale output.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def input_hash(inputs: Mapping[str, Any]) -> str:
    """Stable hash: the same inputs give the same hash on any machine and run."""
    canonical = json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def meta_path(output: Path) -> Path:
    return output.with_name(output.name + ".meta.json")


def write_atomic(path: Path, data: bytes) -> None:
    """Write to a temp file, then rename: readers see the old file or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)  # atomic on both Windows and POSIX


def write_output(path: Path, data: bytes, key: str, info: Mapping[str, Any]) -> None:
    """Write the output first, the meta last: meta present means the output is complete."""
    invalidate(path)
    write_atomic(path, data)
    write_meta(path, key, info)


def file_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks (videos can be large)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invalidate(output: Path) -> None:
    """Drop the meta before overwriting an output, so a crash mid-write can never pair
    the old meta (old inputs) with new bytes."""
    meta_path(output).unlink(missing_ok=True)


def write_meta(output: Path, key: str, info: Mapping[str, Any]) -> None:
    """Mark an already written output (e.g. by FFmpeg) as complete for these inputs."""
    meta = {"input_hash": key, **info}
    write_atomic(meta_path(output), json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))


def read_meta(output: Path) -> dict[str, Any] | None:
    try:
        meta: dict[str, Any] = json.loads(meta_path(output).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return meta


def is_cached(output: Path, key: str) -> bool:
    meta = read_meta(output)
    return output.exists() and meta is not None and meta.get("input_hash") == key
