"""`.env` loading: the format people actually write, and the encodings Windows produces."""

from pathlib import Path

import pytest

from pipeline.env import load_env, parse_env


def test_parses_comments_export_quotes_and_spaces() -> None:
    text = """
# ElevenLabs
ELEVENLABS_API_KEY = sk_abc123
export FAL_KEY="fal:with=equals"
SINGLE='quoted value'
EMPTY=
not a line
=no_name
BAD-NAME=x
"""
    assert parse_env(text) == {
        "ELEVENLABS_API_KEY": "sk_abc123",
        "FAL_KEY": "fal:with=equals",
        "SINGLE": "quoted value",
        "EMPTY": "",
    }


def test_existing_variables_win_and_only_names_are_returned(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=from_file\nB=from_file\n", encoding="utf-8")
    environ = {"A": "from_shell"}

    loaded = load_env(env_file, environ)

    assert loaded == ["B"]
    assert environ == {"A": "from_shell", "B": "from_file"}


def test_missing_file_is_fine(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    assert load_env(tmp_path / ".env", environ) == []
    assert environ == {}


@pytest.mark.parametrize("encoding", ["utf-16", "utf-8-sig", "utf-8"])
def test_reads_what_powershell_and_notepad_write(tmp_path: Path, encoding: str) -> None:
    # Windows PowerShell 5.1 `echo KEY=... > .env` writes UTF-16 with a BOM.
    env_file = tmp_path / ".env"
    env_file.write_bytes("ELEVENLABS_API_KEY=sk_abc\r\n".encode(encoding))
    environ: dict[str, str] = {}

    load_env(env_file, environ)

    assert environ == {"ELEVENLABS_API_KEY": "sk_abc"}
