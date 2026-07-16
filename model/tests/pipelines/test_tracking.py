import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv
from PIL import Image
from pycocotools import mask as coco_mask

from orc_model.pipelines import persistent_gallery, tracking
from orc_model.pipelines.session_linker import SessionLinker
from orc_model.pipelines.tracking import (
    FakeInstrumentTracker,
    InstrumentTracker,
    _filter_workspace,
    _RealInstrumentTracker,
    load_tracker,
)


def test_workspace_filter_removes_detections_below_the_configured_table_boundary():
    detections = sv.Detections(
        xyxy=np.array(
            [
                [10.0, 20.0, 30.0, 40.0],
                [10.0, 91.0, 30.0, 99.0],
            ],
            dtype=np.float32,
        ),
        mask=np.zeros((2, 100, 100), dtype=bool),
        confidence=np.array([0.8, 0.9], dtype=np.float32),
        class_id=np.array([0, 0]),
        data={"class_name": np.array(["surgical_instrument", "surgical_instrument"])},
    )

    filtered = _filter_workspace(detections, frame_height=100, max_center_y_ratio=0.88)

    assert len(filtered) == 1
    assert np.array_equal(filtered.xyxy[0], detections.xyxy[0])
    assert filtered.confidence[0] == detections.confidence[0]


def test_workspace_filter_can_be_disabled():
    detections = sv.Detections(
        xyxy=np.array([[10.0, 91.0, 30.0, 99.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )

    assert _filter_workspace(detections, frame_height=100, max_center_y_ratio=None) is detections


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.1])
def test_workspace_filter_rejects_invalid_boundaries_even_when_detections_are_empty(ratio):
    with pytest.raises(ValueError, match="workspace_max_center_y_ratio"):
        _filter_workspace(sv.Detections.empty(), frame_height=100, max_center_y_ratio=ratio)


def test_load_tracker_rejects_invalid_workspace_boundary_before_loading_models(tmp_path):
    with pytest.raises(ValueError, match="workspace_max_center_y_ratio"):
        load_tracker(tmp_path / "missing.onnx", workspace_max_center_y_ratio=1.1)


# ==========================================================================
# T08 — SPEC §B-R3: the roster crosses the seam
#
# Contract: docs/wayfinder/session-linker/assets/T08-T10/T08-SPEC.md.
# Route (b) of the T10 grilling: the linker already holds the roster, so the
# seam exposes it rather than making the app re-derive it from the Start
# snapshot (whose instant differs from the enrolment freeze by ~2 frames).
# ==========================================================================


def test_b_r3_fake_instrument_tracker_roster_matches_the_ids_it_emits():
    tracker = FakeInstrumentTracker(n_instruments=3)

    assert tracker.roster == frozenset({0, 1, 2}), (
        "the fake enrols instantly and emits tracker_ids 0..n-1, so that IS its roster"
    )


def test_b_r3_fake_instrument_tracker_roster_follows_its_instrument_count():
    assert FakeInstrumentTracker(n_instruments=5).roster == frozenset({0, 1, 2, 3, 4})


def test_b_r3_both_trackers_still_satisfy_the_runtime_checkable_protocol():
    assert isinstance(FakeInstrumentTracker(), InstrumentTracker)


class _StubMatcher:
    """Enough of the ChampionMethod duck-type for an enrolment freeze."""

    def __init__(self):
        self.build_gallery_calls: list[dict] = []

    def build_gallery(self, crops, masks, meta):
        self.build_gallery_calls.append(dict(meta))
        return SimpleNamespace(
            views=np.zeros((max(len(crops), 1), 4), dtype=np.float32),
            identity=meta.get("identity"),
        )


def _one_instrument_call(tracker_id=7):
    """A single quality detection: inset box, mask well above the area gate."""
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    mask = np.zeros((1, 64, 64), dtype=bool)
    mask[0, 4:24, 4:24] = True
    frame[4:24, 4:24] = (10, 20, 30)
    detections = sv.Detections(
        xyxy=np.array([[4.0, 4.0, 24.0, 24.0]], dtype=np.float32),
        mask=mask,
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
        tracker_id=np.array([tracker_id]),
    )
    return frame, detections


def test_b_r3_real_tracker_reports_the_session_linkers_roster():
    linker = SessionLinker(
        _StubMatcher(),
        fps=4.0,
        unknown_id_offset=5000,
        enrolment_window_s=0.25,  # freezes on the first call
        evidence_window_s=10.0,
        evidence_frames=1,
        absent_death_s=10.0,
        min_mask_area_px=100,
    )
    tracker = _RealInstrumentTracker(
        detector=object(),
        confidence=0.5,
        fps=4.0,
        max_age_seconds=1.0,
        session_linker=linker,
        model_version="stub-0",
        workspace_max_center_y_ratio=None,
    )

    assert tracker.roster == frozenset(), "empty before the enrolment freeze"

    frame, detections = _one_instrument_call()
    linker.update(detections, frame)
    assert linker.roster, "sanity: the linker froze a roster"

    assert tracker.roster == linker.roster, (
        "the seam must report the linker's roster, not a fabricated one"
    )

    tracker.reset()
    assert tracker.roster == frozenset(), "empty again immediately after reset()"


# ==========================================================================
# T08 — SPEC §B-R2: load_tracker composes the persistent galleries
# ==========================================================================


def _write_specimen(instruments_dir: Path, n: int) -> None:
    """The smallest COCO specimen folder `load_persistent_galleries` accepts.

    Deliberately a local, minimal copy of test_persistent_gallery.py's richer
    synthesiser: this file only needs "one loadable specimen exists", and the
    two suites are better off independently readable than sharing a fixture.
    """
    root = instruments_dir / f"instrument{n}"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[8:40, 16:40] = (200, 40, 60)
    Image.fromarray(arr).save(root / "images" / "frame_00001.jpg", quality=100)

    m = np.zeros((64, 64), dtype=np.uint8)
    m[8:40, 16:40] = 1
    encoded = coco_mask.encode(np.asfortranarray(m))
    (root / "annotations" / "annotations.json").write_text(
        json.dumps(
            {
                "info": {},
                "categories": [{"id": 1, "name": "surgical_instrument"}],
                "images": [
                    {"id": 1, "width": 64, "height": 64, "file_name": "frame_00001.jpg"}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [16.0, 8.0, 24.0, 32.0],
                        "area": 768,
                        "iscrowd": 0,
                        "segmentation": {
                            "size": [64, 64],
                            "counts": encoded["counts"].decode("ascii"),
                        },
                    }
                ],
            }
        )
    )


@pytest.fixture
def stub_models(monkeypatch):
    """Replace the two heavyweight boundaries load_tracker() reaches for — the
    ONNX detector session and ChampionMethod's DINOv2-B — with stubs, so the
    composition itself can be exercised in milliseconds. Yields the list of
    matchers load_tracker() constructed."""
    from orc_model.pipelines import matching

    created: list[_StubMatcher] = []

    class _FakeChampion(_StubMatcher):
        def __init__(self, **kwargs):
            super().__init__()
            self.init_kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(matching, "ChampionMethod", _FakeChampion)
    monkeypatch.setattr(
        tracking,
        "_build_detector",
        lambda weights_path, confidence, top_k: object(),
    )
    return created


@pytest.fixture
def weights(tmp_path) -> Path:
    path = tmp_path / "checkpoint_best_regular.onnx"
    path.write_bytes(b"stub weights")  # _model_version() only hashes the bytes
    return path


def test_b_r2_load_tracker_embeds_the_persistent_galleries_before_returning(
    tmp_path, weights, stub_models
):
    instruments_dir = tmp_path / "instruments"
    _write_specimen(instruments_dir, 3)

    tracker = load_tracker(weights, instruments_dir=instruments_dir)

    assert len(stub_models) == 1, "sanity: load_tracker built exactly one matcher"
    matcher = stub_models[0]
    assert [c.get("identity") for c in matcher.build_gallery_calls] == [3], (
        "load_tracker must load and embed the persistent galleries eagerly — once, before "
        "returning — not lazily on the first tracked frame (Trap 5)"
    )

    calls_after_load = len(matcher.build_gallery_calls)
    tracker.reset()
    assert len(matcher.build_gallery_calls) == calls_after_load, (
        "reset() must not re-load or re-embed the persistent galleries"
    )


def test_b_r2_load_tracker_with_instruments_dir_none_disables_binding(
    tmp_path, weights, stub_models
):
    instruments_dir = tmp_path / "instruments"
    _write_specimen(instruments_dir, 3)  # present, but explicitly not asked for

    load_tracker(weights, instruments_dir=None)

    assert stub_models[0].build_gallery_calls == [], (
        "instruments_dir=None -> binding disabled, no galleries loaded"
    )


def test_b_r2_load_tracker_never_raises_on_a_missing_instruments_dir(
    tmp_path, weights, stub_models
):
    tracker = load_tracker(weights, instruments_dir=tmp_path / "does-not-exist")

    assert tracker is not None, "a missing dir must degrade to session-only, never raise"
    assert stub_models[0].build_gallery_calls == []


def test_b_r2_a_bare_load_tracker_call_still_loads_the_shipped_galleries(
    weights, stub_models
):
    """The backward-compatible bare call gets the shipped specimen photos,
    rather than silently disabling persistent binding."""
    if not persistent_gallery.DEFAULT_INSTRUMENTS_DIR.exists():
        pytest.skip("shipped specimen photos are not present in this checkout")

    load_tracker(weights)

    assert stub_models[0].build_gallery_calls, (
        "load_tracker(weights) with no instruments_dir must still load the persistent galleries"
    )


def test_fake_instrument_tracker_emits_an_all_settled_resolving_flag():
    # The fake enrols instantly and never re-identifies, so every track it emits
    # is a settled roster id -- data["resolving"] is present, row-aligned, and
    # all False (the seam's wait-state flag; see tracker-interface.md).
    tracker = FakeInstrumentTracker(n_instruments=3, confidence=0.0)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    dets = tracker.update(frame)

    assert len(dets) > 0, "sanity: confidence=0 keeps every instrument in-frame"
    assert "resolving" in dets.data
    assert len(dets.data["resolving"]) == len(dets)
    assert not dets.data["resolving"].any(), "the fake enrols instantly; nothing is resolving"
