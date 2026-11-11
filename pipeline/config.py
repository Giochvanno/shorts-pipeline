"""Typed, validated configuration: config/channel.yaml and config/series/*.yaml.

Rules from the plan that must never be switched off live here as constants.
A series can add to them, but cannot remove them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Never in a generated image: text is drawn by FFmpeg, faces and hands break most often.
HARD_FORBIDDEN_VISUALS: tuple[str, ...] = (
    "text",
    "letters",
    "numbers",
    "close-up faces",
    "hands",
    "fingers",
)

# YouTube rule from the plan: no AI "experts" giving this kind of advice.
HARD_BANNED_TOPICS: tuple[str, ...] = (
    "health advice",
    "financial advice",
    "legal advice",
    "politics",
)

HexColor = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]

_SERIES_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class ConfigError(Exception):
    """A config file is missing, is not valid YAML, or breaks the schema."""


class _Strict(BaseModel):
    # extra="forbid": a typo like `stabilty:` is an error, not a silently ignored key.
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- channel.yaml ---------------------------------------------------------------


class ChannelInfo(_Strict):
    name: str = Field(min_length=1)
    language: str = Field(pattern=r"^[a-z]{2}$")
    timezone: str = Field(min_length=1)


class UploadConfig(_Strict):
    privacy: Literal["private", "unlisted", "public"]
    category_id: str = Field(pattern=r"^\d+$")
    made_for_kids: bool
    synthetic_media: bool


class BudgetConfig(_Strict):
    max_cost_per_video_usd: float = Field(gt=0)


class ModelRef[P: str](_Strict):
    provider: P
    model: str = Field(min_length=1)


LlmProvider = Literal["claude_cli", "anthropic_api"]
TtsProvider = Literal["elevenlabs"]
ImageProvider = Literal["fal", "replicate"]


class LlmChain(_Strict):
    primary: ModelRef[LlmProvider]
    fallback: ModelRef[LlmProvider] | None


class TtsChain(_Strict):
    primary: ModelRef[TtsProvider]
    fallback: ModelRef[TtsProvider] | None


class ImageChain(_Strict):
    primary: ModelRef[ImageProvider]
    fallback: ModelRef[ImageProvider] | None


class LlmModels(_Strict):
    heavy: LlmChain
    light: LlmChain


class ModelsConfig(_Strict):
    llm: LlmModels
    tts: TtsChain
    images: ImageChain


class ChannelConfig(_Strict):
    channel: ChannelInfo
    upload: UploadConfig
    budget: BudgetConfig
    models: ModelsConfig


# --- series/*.yaml --------------------------------------------------------------


class Range(_Strict):
    min: float = Field(gt=0)
    max: float = Field(gt=0)

    @model_validator(mode="after")
    def _min_not_above_max(self) -> Self:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must not be greater than max ({self.max})")
        return self


class FormatConfig(_Strict):
    duration_s: Range
    hook_max_s: float = Field(gt=0, le=3)
    scenes: Range
    structures: list[str] = Field(min_length=3)

    @field_validator("duration_s")
    @classmethod
    def _shorts_length(cls, v: Range) -> Range:
        if v.min < 15 or v.max > 60:
            raise ValueError("Shorts v1 must be 15-60 s long")
        return v

    @field_validator("structures")
    @classmethod
    def _unique_structures(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("structures must be unique")
        return v


class ScriptConfig(_Strict):
    tone: str = Field(min_length=1)
    angle: str = Field(min_length=1)
    rules: list[str]
    extra_banned_topics: list[str]

    @property
    def banned_topics(self) -> tuple[str, ...]:
        return HARD_BANNED_TOPICS + tuple(self.extra_banned_topics)


class VoiceConfig(_Strict):
    voice_id: str = Field(min_length=1)
    stability: Unit
    similarity_boost: Unit
    speed: float = Field(ge=0.7, le=1.2)


class VisualBible(_Strict):
    style_prefix: str = Field(min_length=1)
    palette: list[HexColor] = Field(min_length=2, max_length=6)
    lighting: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    extra_forbidden: list[str]
    max_attempts_per_scene: int = Field(ge=1, le=3)

    @property
    def forbidden(self) -> tuple[str, ...]:
        return HARD_FORBIDDEN_VISUALS + tuple(self.extra_forbidden)


class CaptionsConfig(_Strict):
    font: str = Field(min_length=1)
    words_per_chunk: int = Field(ge=1, le=6)
    highlight_color: HexColor
    bottom_margin_ratio: float = Field(ge=0.05, le=0.5)


class MusicConfig(_Strict):
    moods: list[str] = Field(min_length=1)
    duck_below_voice_db: float = Field(ge=6, le=40)


class SeriesConfig(_Strict):
    id: str
    title: str = Field(min_length=1)
    format: FormatConfig
    script: ScriptConfig
    voice: VoiceConfig
    visual_bible: VisualBible
    captions: CaptionsConfig
    music: MusicConfig

    @field_validator("id")
    @classmethod
    def _kebab_case_id(cls, v: str) -> str:
        if not _SERIES_ID.match(v):
            raise ValueError("id must be kebab-case, e.g. fact-of-day")
        return v


# --- loading --------------------------------------------------------------------


def _read_yaml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ConfigError(f"{path}: file not found") from e
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e


def _validate[M: BaseModel](model: type[M], path: Path) -> M:
    data = _read_yaml(path)
    try:
        return model.model_validate(data)
    except ValidationError as e:
        lines = [
            f"  {'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in e.errors(include_url=False)
        ]
        raise ConfigError(f"{path}: {len(lines)} error(s)\n" + "\n".join(lines)) from e


def load_channel(config_dir: Path = DEFAULT_CONFIG_DIR) -> ChannelConfig:
    return _validate(ChannelConfig, config_dir / "channel.yaml")


def list_series(config_dir: Path = DEFAULT_CONFIG_DIR) -> list[str]:
    return sorted(p.stem for p in (config_dir / "series").glob("*.yaml"))


def load_series(series_id: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> SeriesConfig:
    path = config_dir / "series" / f"{series_id}.yaml"
    series = _validate(SeriesConfig, path)
    if series.id != series_id:
        raise ConfigError(f"{path}: id is {series.id!r}, but the file name says {series_id!r}")
    return series
