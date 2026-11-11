"""Cache keys and atomic writes."""

from pathlib import Path

import pytest

from pipeline.artifacts import input_hash, is_cached, meta_path, write_output


def test_hash_ignores_key_order_but_not_values() -> None:
    assert input_hash({"a": 1, "b": "x"}) == input_hash({"b": "x", "a": 1})
    assert input_hash({"a": 1}) != input_hash({"a": 2})


def test_output_is_cached_only_for_the_same_inputs(tmp_path: Path) -> None:
    out = tmp_path / "script.json"
    write_output(out, b"{}", "key-1", {"cost_usd": 0.01})

    assert is_cached(out, "key-1")
    assert not is_cached(out, "key-2")


def test_output_without_meta_is_not_cached(tmp_path: Path) -> None:
    # A crash between the two writes leaves an output without meta: must be redone.
    out = tmp_path / "voice.wav"
    write_output(out, b"RIFF", "key", {})
    meta_path(out).unlink()

    assert not is_cached(out, "key")


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    write_output(tmp_path / "a.json", b"{}", "k", {})

    assert not list(tmp_path.glob("*.tmp"))


def test_overwrite_drops_old_meta_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "script.json"
    write_output(out, b"old", "old-key", {})

    def crash(*_args: object) -> None:
        raise RuntimeError("power cut while writing meta")

    monkeypatch.setattr("pipeline.artifacts.write_meta", crash)
    with pytest.raises(RuntimeError):
        write_output(out, b"new", "new-key", {})

    assert out.read_bytes() == b"new"
    assert not is_cached(out, "old-key")  # old meta must not vouch for new bytes
    assert not is_cached(out, "new-key")
