"""Secrets from a `.env` file, without a dependency.

Keys (ELEVENLABS_API_KEY, ...) live in `.env` at the repo root, which git
ignores. Format: one `NAME=value` per line; `#` comments, blank lines,
`export NAME=value` and quoted values are accepted.

Variables already set in the environment win over the file, so CI or a
one-off `$env:NAME = "..."` in PowerShell can override it.

Windows PowerShell 5.1 writes files as UTF-16 (`echo ... > .env`) and Notepad
may add a UTF-8 BOM; both are read correctly.
"""

import os
from collections.abc import MutableMapping
from pathlib import Path


def _decode(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def load_env(path: Path, environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Copy variables from `path` into the environment. Returns the names set (never values)."""
    target = os.environ if environ is None else environ
    if not path.is_file():
        return []
    loaded = []
    for name, value in parse_env(_decode(path.read_bytes())).items():
        if name not in target:
            target[name] = value
            loaded.append(name)
    return loaded
