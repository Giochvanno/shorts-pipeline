"""The CLI starts and reports its version, in-process and as ``python -m pipeline``."""

import subprocess
import sys

import pytest

from pipeline.cli import main


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("pipeline 0.")


def test_module_is_runnable_with_python_dash_m() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pipeline", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("pipeline ")
