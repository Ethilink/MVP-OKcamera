from pathlib import Path

import pytest

from backend.mvp_settings import (
    DEFAULT_MVP_CONFIG_PATH,
    MVPSettings,
    load_mvp_settings,
)
from orc_model.pipelines.config import TrackerConfig


def test_app_root_toml_loads_into_the_typed_settings_model() -> None:
    settings = load_mvp_settings()

    assert DEFAULT_MVP_CONFIG_PATH.name == "mvp.toml"
    assert DEFAULT_MVP_CONFIG_PATH.parent.name == "app"
    assert isinstance(settings, MVPSettings)
    assert isinstance(settings.tracker, TrackerConfig)
    assert settings.tracker.expected_processing_fps == 3.0
    assert settings.session.off_debounce_s > settings.session.on_debounce_s
    assert settings.capture.frame_size == (1920, 1080)


def test_relative_gallery_paths_are_resolved_from_the_config_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        DEFAULT_MVP_CONFIG_PATH.read_text().replace(
            'instruments_dir = "../model/data/instruments"',
            'instruments_dir = "specimens"',
        )
    )

    settings = load_mvp_settings(config_path)

    assert settings.tracker.instruments_dir == tmp_path / "specimens"


def test_a_misspelled_setting_fails_at_startup(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        DEFAULT_MVP_CONFIG_PATH.read_text().replace("confidence =", "confidnce =", 1)
    )

    with pytest.raises(ValueError, match=r"tracker\.detector"):
        load_mvp_settings(config_path)
