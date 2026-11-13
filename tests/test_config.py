"""Config loading: the real files are valid, and every broken variant fails loudly."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.cli import main
from pipeline.config import (
    DEFAULT_CONFIG_DIR,
    HARD_BANNED_TOPICS,
    HARD_FORBIDDEN_VISUALS,
    ConfigError,
    list_series,
    load_channel,
    load_series,
    load_topics,
)

SERIES = "fact-of-day"

Mutator = Callable[[dict[str, Any]], None]


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A private copy of the real config/ that a test may break."""
    return Path(shutil.copytree(DEFAULT_CONFIG_DIR, tmp_path / "config"))


def _edit(path: Path, mutate: Mutator) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def _edit_series(config_dir: Path, mutate: Mutator) -> None:
    _edit(config_dir / "series" / f"{SERIES}.yaml", mutate)


# --- the real files ---------------------------------------------------------------


def test_real_channel_config_is_valid() -> None:
    channel = load_channel()

    assert channel.mode.real_ports == ()  # in the repo nothing may spend money by default
    assert channel.models.llm.api_fallback is False  # separate money: off unless enabled
    assert channel.upload.privacy == "private"  # test phase: never public by accident
    assert channel.budget.max_cost_per_video_usd == pytest.approx(0.5)


def test_every_real_series_is_valid() -> None:
    ids = list_series()

    assert SERIES in ids
    for series_id in ids:
        assert load_series(series_id).id == series_id


def test_hard_rules_cannot_be_removed_by_a_series(config_dir: Path) -> None:
    def drop_extras(d: dict[str, Any]) -> None:
        d["visual_bible"]["extra_forbidden"] = []
        d["script"]["extra_banned_topics"] = []

    _edit_series(config_dir, drop_extras)
    series = load_series(SERIES, config_dir)

    assert set(HARD_FORBIDDEN_VISUALS) <= set(series.visual_bible.forbidden)
    assert set(HARD_BANNED_TOPICS) <= set(series.script.banned_topics)


def test_series_extras_are_added_to_hard_rules(config_dir: Path) -> None:
    _edit_series(config_dir, lambda d: d["visual_bible"].update(extra_forbidden=["neon"]))

    assert "neon" in load_series(SERIES, config_dir).visual_bible.forbidden


# --- broken variants ----------------------------------------------------------------


def _set(section: str, key: str, value: Any) -> Mutator:
    return lambda d: d[section].update({key: value})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(_set("voice", "stabilty", 0.5), "stabilty", id="typo-in-key"),
        pytest.param(_set("visual_bible", "palette", ["#12345"]), "palette", id="bad-hex"),
        pytest.param(_set("format", "duration_s", {"min": 50, "max": 30}), "min", id="min-gt-max"),
        pytest.param(_set("format", "duration_s", {"min": 15, "max": 90}), "15-60", id="too-long"),
        pytest.param(_set("format", "structures", ["a", "b"]), "structures", id="few-structures"),
        pytest.param(_set("format", "structures", ["a", "a", "b"]), "unique", id="dup-structures"),
        pytest.param(
            _set("visual_bible", "max_attempts_per_scene", 5), "max_attempts", id="too-many-tries"
        ),
        pytest.param(_set("voice", "stability", 1.5), "stability", id="unit-out-of-range"),
        pytest.param(
            _set("captions", "highlight_color", "#123456"), "palette", id="caption-off-palette"
        ),
    ],
)
def test_broken_series_is_rejected(config_dir: Path, mutate: Mutator, message: str) -> None:
    _edit_series(config_dir, mutate)

    with pytest.raises(ConfigError, match=message) as exc:
        load_series(SERIES, config_dir)

    assert f"{SERIES}.yaml" in str(exc.value)  # the error names the file to fix


def test_unknown_provider_is_rejected(config_dir: Path) -> None:
    def typo(d: dict[str, Any]) -> None:
        d["models"]["tts"]["primary"]["provider"] = "elevenlab"

    _edit(config_dir / "channel.yaml", typo)

    with pytest.raises(ConfigError, match="provider"):
        load_channel(config_dir)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["mode"].update(tts="real"), "switched together"),
        (lambda d: d["mode"].update(stt="real"), "switched together"),
        (lambda d: d["models"]["tts"].update(usd_per_credit=-1), "usd_per_credit"),
        (lambda d: d["models"]["tts"].update(usd_per_credit=0.2), "usd_per_credit"),
        (lambda d: d["models"]["tts"].pop("timeout_s"), "timeout_s"),
    ],
    ids=[
        "real-voice-fake-check",
        "fake-voice-real-check",
        "negative-price",
        "price-typo",
        "no-timeout",
    ],
)
def test_broken_channel_is_rejected(config_dir: Path, mutate: Mutator, message: str) -> None:
    _edit(config_dir / "channel.yaml", mutate)

    with pytest.raises(ConfigError, match=message):
        load_channel(config_dir)


def test_series_id_must_match_file_name(config_dir: Path) -> None:
    _edit_series(config_dir, lambda d: d.update(id="other-series"))

    with pytest.raises(ConfigError, match="file name"):
        load_series(SERIES, config_dir)


def test_invalid_yaml_is_a_config_error(config_dir: Path) -> None:
    (config_dir / "channel.yaml").write_text("channel: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_channel(config_dir)


def test_missing_series_is_a_config_error(config_dir: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_series("no-such-series", config_dir)


# --- CLI ----------------------------------------------------------------------------


def test_check_config_passes_on_real_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-config"]) == 0
    assert f"ok  series/{SERIES}.yaml" in capsys.readouterr().out


def test_check_config_fails_on_broken_config(
    config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _edit_series(config_dir, _set("voice", "speed", 3.0))

    assert main(["check-config", "--config-dir", str(config_dir)]) == 1
    assert "speed" in capsys.readouterr().err


def test_topics_are_twenty_unique_entries() -> None:
    topics = load_topics(SERIES)

    assert len(topics) == 20
    assert len({t.lower() for t in topics}) == 20


def test_duplicate_topics_are_rejected(config_dir: Path) -> None:
    (config_dir / "topics" / f"{SERIES}.yaml").write_text(
        "topics: [comets, Comets]", encoding="utf-8"
    )

    with pytest.raises(ConfigError, match="unique"):
        load_topics(SERIES, config_dir)
