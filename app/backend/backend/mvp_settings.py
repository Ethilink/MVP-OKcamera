"""Load and validate the operator-facing ``app/mvp.toml`` configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from orc_model.pipelines.config import (
    DetectorConfig,
    LinkerConfig,
    MatcherConfig,
    OCSortConfig,
    TrackerConfig,
)

DEFAULT_MVP_CONFIG_PATH = Path(__file__).resolve().parents[2] / "mvp.toml"


@dataclass(frozen=True)
class SessionSettings:
    off_debounce_s: float
    on_debounce_s: float

    def __post_init__(self) -> None:
        if self.off_debounce_s < 0 or self.on_debounce_s < 0:
            raise ValueError("session debounce durations cannot be negative")


@dataclass(frozen=True)
class CaptureSettings:
    frame_size: tuple[int, int]
    stale_after_s: float
    fake_fps: float

    def __post_init__(self) -> None:
        if any(dimension < 1 for dimension in self.frame_size):
            raise ValueError("capture frame dimensions must be positive")
        if self.stale_after_s <= 0 or self.fake_fps <= 0:
            raise ValueError("capture timings must be positive")


@dataclass(frozen=True)
class MVPSettings:
    tracker: TrackerConfig
    session: SessionSettings
    capture: CaptureSettings


ConfigType = TypeVar("ConfigType")


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value: Any = data
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing [{name}] configuration table")
        value = value[part]
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a configuration table")
    return dict(value)


def _construct(config_type: type[ConfigType], section: str, values: dict) -> ConfigType:
    try:
        return config_type(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid [{section}] configuration: {exc}") from exc


def _load_tracker(data: dict[str, Any], config_dir: Path) -> TrackerConfig:
    values = _table(data, "tracker")
    try:
        galleries_enabled = values.pop("persistent_galleries_enabled")
        instruments_dir_value = values.pop("instruments_dir")
    except KeyError as exc:
        raise ValueError(f"missing [tracker] setting: {exc.args[0]}") from exc

    if not isinstance(galleries_enabled, bool):
        raise ValueError("[tracker] persistent_galleries_enabled must be true or false")
    instruments_dir = None
    if galleries_enabled:
        instruments_dir = (config_dir / str(instruments_dir_value)).resolve()

    detector_values = values.pop("detector", {})
    workspace_enabled = detector_values.pop("workspace_filter_enabled", None)
    if not isinstance(workspace_enabled, bool):
        raise ValueError("[tracker.detector] workspace_filter_enabled must be true or false")
    if not workspace_enabled:
        detector_values["workspace_max_center_y_ratio"] = None

    values.update(
        instruments_dir=instruments_dir,
        detector=_construct(DetectorConfig, "tracker.detector", detector_values),
        oc_sort=_construct(OCSortConfig, "tracker.oc_sort", values.pop("oc_sort", {})),
        matcher=_construct(MatcherConfig, "tracker.matcher", values.pop("matcher", {})),
        linker=_construct(LinkerConfig, "tracker.linker", values.pop("linker", {})),
    )
    return _construct(TrackerConfig, "tracker", values)


def _load_capture(data: dict[str, Any]) -> CaptureSettings:
    values = _table(data, "capture")
    try:
        frame_size = (values.pop("frame_width"), values.pop("frame_height"))
    except KeyError as exc:
        raise ValueError(f"missing [capture] setting: {exc.args[0]}") from exc
    values["frame_size"] = frame_size
    return _construct(CaptureSettings, "capture", values)


def load_mvp_settings(path: str | Path = DEFAULT_MVP_CONFIG_PATH) -> MVPSettings:
    """Read one TOML file and return validated, immutable runtime settings."""
    config_path = Path(path).resolve()
    try:
        data = tomllib.loads(config_path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot load MVP config {config_path}: {exc}") from exc

    extra_sections = set(data) - {"tracker", "session", "capture"}
    if extra_sections:
        raise ValueError(f"unknown MVP config section(s): {sorted(extra_sections)}")

    return MVPSettings(
        tracker=_load_tracker(data, config_path.parent),
        session=_construct(SessionSettings, "session", _table(data, "session")),
        capture=_load_capture(data),
    )
