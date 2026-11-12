"""The stage 1 goal, end to end: `python -m pipeline run --series fact-of-day` -> mp4."""

from pathlib import Path

import pytest

from pipeline.cli import DB_NAME, LOG_NAME, main, pick_topic
from pipeline.media import probe
from pipeline.runner import DEFAULT_STEPS
from pipeline.state import StateStore


@pytest.mark.usefixtures("ffmpeg")
def test_run_makes_a_shorts_video_even_under_chaos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    code = main(
        ["run", "--series", "fact-of-day", "--runs-dir", str(runs),
         "--chaos", "0.2", "--seed", "3", "--retry-delay", "0"]
    )  # fmt: skip

    out = capsys.readouterr().out
    assert code == 0, out
    video = runs / "1" / "video.mp4"
    info = probe(video)
    assert (info.width, info.height) == (1080, 1920)
    assert "done in" in out
    assert "$0.000" in out
    log = (runs / LOG_NAME).read_text(encoding="utf-8")
    assert log.count('"event": "step_done"') == len(DEFAULT_STEPS)

    assert main(["status", "--runs-dir", str(runs)]) == 0
    assert "done" in capsys.readouterr().out


def test_run_takes_the_next_unused_topic(tmp_path: Path) -> None:
    topics = ["comets", "glass", "honey"]
    with StateStore(tmp_path / DB_NAME) as store:
        store.create_video("fact-of-day", "Comets", ["script"])

        assert pick_topic(store, "fact-of-day", topics) == "glass"
        store.create_video("fact-of-day", "glass", ["script"])
        store.create_video("fact-of-day", "honey", ["script"])
        assert pick_topic(store, "fact-of-day", topics) is None


def test_ports_without_a_real_adapter_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["run", "--series", "fact-of-day", "--runs-dir", str(tmp_path), "--real", "tts"])

    assert code == 2
    assert "not connected yet" in capsys.readouterr().err
    with StateStore(tmp_path / DB_NAME) as store:
        assert store.videos() == []  # nothing was created


def test_chaos_is_refused_with_real_ports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["run", "--series", "fact-of-day", "--runs-dir", str(tmp_path)]
    code = main([*args, "--real", "llm", "--chaos", "0.2"])

    assert code == 2
    assert "fake ports only" in capsys.readouterr().err


def test_resume_of_unknown_video_is_a_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["resume", "--video", "42", "--runs-dir", str(tmp_path)]) == 2
    assert "no video with id 42" in capsys.readouterr().err


def test_status_with_no_videos(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", "--runs-dir", str(tmp_path)]) == 0
    assert "no videos yet" in capsys.readouterr().out
