"""Cache keys and atomic writes."""

from pathlib import Path

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
