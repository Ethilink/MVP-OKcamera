import pytest

from orc_model.pipelines import matching, session_linker, tracking
from orc_model.pipelines.config import (
    DetectorConfig,
    LinkerConfig,
    MatcherConfig,
    OCSortConfig,
    TrackerConfig,
)
from orc_model.pipelines.tracking import load_tracker


def test_tracker_config_groups_every_tunable_pipeline_stage() -> None:
    config = TrackerConfig()

    assert isinstance(config.detector, DetectorConfig)
    assert isinstance(config.oc_sort, OCSortConfig)
    assert isinstance(config.matcher, MatcherConfig)
    assert isinstance(config.linker, LinkerConfig)


def test_tracker_config_rejects_a_linker_that_kills_before_oc_sort_stops_coasting() -> None:
    with pytest.raises(ValueError, match="absent_death_s.*max_age_seconds"):
        TrackerConfig(
            oc_sort=OCSortConfig(max_age_seconds=2.0),
            linker=LinkerConfig(absent_death_s=1.0),
        )


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_detector_config_rejects_invalid_workspace_boundaries(ratio: float) -> None:
    with pytest.raises(ValueError, match="workspace_max_center_y_ratio"):
        DetectorConfig(workspace_max_center_y_ratio=ratio)


def test_load_tracker_applies_the_grouped_stage_configuration(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMatcher:
        def __init__(self, **kwargs) -> None:
            captured["matcher"] = kwargs

    class FakeLinker:
        def __init__(self, matcher, **kwargs) -> None:
            captured["linker"] = kwargs

    monkeypatch.setattr(matching, "ChampionMethod", FakeMatcher)
    monkeypatch.setattr(session_linker, "SessionLinker", FakeLinker)
    monkeypatch.setattr(
        tracking,
        "_build_detector",
        lambda weights, confidence, top_k: captured.update(
            detector=(confidence, top_k)
        )
        or object(),
    )
    monkeypatch.setattr(
        tracking,
        "_RealInstrumentTracker",
        lambda **kwargs: captured.update(real_tracker=kwargs) or "configured-tracker",
    )

    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"weights")
    config = TrackerConfig(
        expected_processing_fps=4.0,
        instruments_dir=None,
        detector=DetectorConfig(confidence=0.61, top_k=42),
        oc_sort=OCSortConfig(max_age_seconds=1.25, min_hits=5),
        matcher=MatcherConfig(acceptance_threshold=0.41, winner_margin=0.07),
        linker=LinkerConfig(
            bind_threshold=0.33,
            absent_death_s=1.5,
            unknown_recheck_cooldown_s=0.9,
        ),
    )

    tracker = load_tracker(weights, config=config)

    assert tracker == "configured-tracker"
    assert captured["detector"] == (0.61, 42)
    assert captured["matcher"]["tau"] == 0.41
    assert captured["matcher"]["margin"] == 0.07
    assert captured["linker"]["fps"] == 4.0
    assert captured["linker"]["bind_tau"] == 0.33
    assert captured["linker"]["unknown_recheck_cooldown_s"] == 0.9
    assert captured["real_tracker"]["oc_sort_config"] == config.oc_sort


# ==========================================================================
# T11 SPEC-M — catalog-only knobs + fail-closed wiring (M2 / test 8)
# ==========================================================================


def _stub_load_tracker_boundaries(monkeypatch, *, galleries):
    """Stub every heavyweight boundary load_tracker touches (detector, matcher,
    linker, real tracker) and force `load_persistent_galleries` to return
    `galleries`, so only the catalog-only fail-closed gate is exercised."""

    class _FakeMatcher:
        def __init__(self, **kwargs) -> None:
            pass

    class _FakeLinker:
        def __init__(self, matcher, **kwargs) -> None:
            pass

    monkeypatch.setattr(matching, "ChampionMethod", _FakeMatcher)
    monkeypatch.setattr(session_linker, "SessionLinker", _FakeLinker)
    monkeypatch.setattr(
        tracking, "_build_detector", lambda weights, confidence, top_k: object()
    )
    monkeypatch.setattr(
        tracking, "_RealInstrumentTracker", lambda **kwargs: "configured-tracker"
    )
    monkeypatch.setattr(
        tracking,
        "load_persistent_galleries",
        lambda matcher, instruments_dir: galleries,
    )


def test_load_tracker_catalog_only_missing_galleries_fails_fast(tmp_path, monkeypatch) -> None:
    # Nothing usable loads ({} -> None after `or None`): catalog-only mode must
    # refuse the session-only fallback it exists to replace, naming the directory.
    _stub_load_tracker_boundaries(monkeypatch, galleries={})
    instruments_dir = tmp_path / "instruments"
    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"weights")
    config = TrackerConfig(
        instruments_dir=instruments_dir,
        linker=LinkerConfig(catalog_only_enrolment=True, expected_catalog_size=8),
    )

    with pytest.raises(ValueError) as excinfo:
        load_tracker(weights, config=config)
    message = str(excinfo.value)
    assert "catalog_only_enrolment" in message
    assert str(instruments_dir) in message, "the failure must name the directory it searched"


def test_load_tracker_catalog_only_wrong_sized_catalog_fails_fast(tmp_path, monkeypatch) -> None:
    # Seven galleries load but eight are expected: a size mismatch is a
    # safety failure and must be reported with expected vs loaded counts.
    seven = {n: object() for n in range(1, 8)}
    _stub_load_tracker_boundaries(monkeypatch, galleries=seven)
    instruments_dir = tmp_path / "instruments"
    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"weights")
    config = TrackerConfig(
        instruments_dir=instruments_dir,
        linker=LinkerConfig(catalog_only_enrolment=True, expected_catalog_size=8),
    )

    with pytest.raises(ValueError) as excinfo:
        load_tracker(weights, config=config)
    message = str(excinfo.value)
    assert "expected 8" in message and "loaded 7" in message


def test_load_tracker_catalog_only_rejects_specimen_ids_reaching_the_unknown_offset(
    tmp_path, monkeypatch
) -> None:
    # Eight galleries load (the count gate passes) but one specimen id reaches the
    # unknown_id_offset namespace. An unbound raw id `k` emits `k + offset`, so a
    # roster id >= offset could not be told apart from a foreign Unknown -- the
    # safety mode must reject the catalog outright, not merely count it.
    galleries = {n: object() for n in range(1, 8)}
    galleries[1001] = object()  # 8 galleries total, but 1001 >= unknown_id_offset
    _stub_load_tracker_boundaries(monkeypatch, galleries=galleries)
    instruments_dir = tmp_path / "instruments"
    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"weights")
    config = TrackerConfig(
        instruments_dir=instruments_dir,
        linker=LinkerConfig(
            catalog_only_enrolment=True,
            expected_catalog_size=8,
            unknown_id_offset=1000,
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        load_tracker(weights, config=config)
    message = str(excinfo.value)
    assert "1001" in message and "1000" in message


def test_linker_config_rejects_nonpositive_expected_catalog_size() -> None:
    with pytest.raises(ValueError, match="expected_catalog_size must be positive"):
        LinkerConfig(expected_catalog_size=0)
