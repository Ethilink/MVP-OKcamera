"""AC2-AC6 for T01: `ScenarioTracker` + `FakeCaptureSource`.

All tests pass `fps=None` to `FakeCaptureSource` (instant frames) and drive
`ScenarioTracker` timing purely by frame count — no real sleeps anywhere in
this suite (see the T01 task spec's FakeCaptureSource pacing note).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import supervision as sv
from orc_model.pipelines.tracking import InstrumentTracker

from backend.fakes import DEFAULT_SCENARIO, FakeCaptureSource, ScenarioEvent, ScenarioTracker

# A small frame for tests that only care about tracker_id presence/absence —
# ScenarioTracker allocates a full-frame-sized bool mask per detection per
# call, and several tests below drive hundreds of frames; a tiny frame keeps
# the suite fast without weakening those assertions. The one test that checks
# mask *shape* uses a distinctly-sized frame instead, so it can't pass by
# accident from a hardcoded shape in the implementation.
FRAME = np.zeros((48, 64, 3), dtype=np.uint8)

# Real cv2 CAP_PROP_* values, inlined so this test suite has no dependency on
# opencv-python being installed (it isn't declared as a dependency here).
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_BUFFERSIZE = 38


def _update_until(tracker: ScenarioTracker, t: float) -> sv.Detections:
    """Drive a freshly-constructed `tracker` frame-by-frame so its LAST
    `update()` call lands at exactly scenario-time `t` (given `tracker.fps`),
    and return that call's result. No wall clock involved."""
    n_calls = round(t * tracker.fps) + 1
    detections = sv.Detections.empty()
    for _ in range(n_calls):
        detections = tracker.update(FRAME)
    return detections


def _tracker_ids(detections: sv.Detections) -> set[int]:
    if detections.tracker_id is None:
        return set()
    return set(detections.tracker_id.tolist())


class TestScenarioTrackerIsInstrumentTracker:
    """AC2: ScenarioTracker satisfies isinstance(t, InstrumentTracker)."""

    def test_scenario_tracker_is_an_instrument_tracker(self) -> None:
        tracker = ScenarioTracker()

        assert isinstance(tracker, InstrumentTracker)


class TestDefaultScenarioPresenceWindows:
    """AC3: at fps=10, instrument 1 absent for t in (20, 35) and present
    again after; instrument 3 absent for all t > 50; ids 2/4/5 present
    throughout."""

    def test_instrument_1_present_at_leave_boundary(self) -> None:
        detections = _update_until(ScenarioTracker(), t=20.0)

        assert 1 in _tracker_ids(detections)

    def test_instrument_1_absent_strictly_inside_the_window(self) -> None:
        detections = _update_until(ScenarioTracker(), t=27.0)

        assert 1 not in _tracker_ids(detections)

    def test_instrument_1_present_at_return_boundary(self) -> None:
        detections = _update_until(ScenarioTracker(), t=35.0)

        assert 1 in _tracker_ids(detections)

    def test_instrument_1_present_again_well_after_return(self) -> None:
        detections = _update_until(ScenarioTracker(), t=40.0)

        assert 1 in _tracker_ids(detections)

    def test_instrument_3_present_at_leave_boundary(self) -> None:
        detections = _update_until(ScenarioTracker(), t=50.0)

        assert 3 in _tracker_ids(detections)

    def test_instrument_3_absent_just_after_leaving(self) -> None:
        detections = _update_until(ScenarioTracker(), t=50.1)

        assert 3 not in _tracker_ids(detections)

    def test_instrument_3_absent_long_after_leaving(self) -> None:
        detections = _update_until(ScenarioTracker(), t=80.0)

        assert 3 not in _tracker_ids(detections)

    def test_other_instruments_present_throughout(self) -> None:
        for t in (0.0, 20.0, 27.0, 35.0, 50.0, 80.0):
            ids = _tracker_ids(_update_until(ScenarioTracker(), t=t))

            assert {2, 4, 5}.issubset(ids)


class TestDeterminismAndReset:
    """AC4: two ScenarioTrackers fed the same frame count return identical
    tracker_id sets; reset() restores frame-0 behaviour."""

    def test_same_frame_count_yields_identical_tracker_id_sets(self) -> None:
        tracker_a = ScenarioTracker()
        tracker_b = ScenarioTracker()

        for _ in range(301):
            detections_a = tracker_a.update(FRAME)
            detections_b = tracker_b.update(FRAME)

        assert _tracker_ids(detections_a) == _tracker_ids(detections_b)

    def test_reset_restores_frame_zero_behaviour(self) -> None:
        tracker = ScenarioTracker()
        frame_zero_ids = _tracker_ids(tracker.update(FRAME))
        for _ in range(300):
            tracker.update(FRAME)

        tracker.reset()

        assert _tracker_ids(tracker.update(FRAME)) == frame_zero_ids


class TestTrackerContractFields:
    """AC5: update() honours the tracker contract (xyxy/confidence/class_id/
    tracker_id/mask populated), including the sv.Detections.empty() branch
    via a custom all-gone scenario."""

    def test_present_detections_populate_every_guaranteed_field(self) -> None:
        # A frame shape distinct from every other constant in this file, so
        # this assertion can't pass by coincidence against a hardcoded shape.
        height, width = 90, 130
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        detections = ScenarioTracker().update(frame)
        n = len(detections.tracker_id)

        assert n > 0
        assert detections.xyxy.shape == (n, 4)
        assert detections.xyxy.dtype == np.float32
        assert detections.confidence.shape == (n,)
        assert detections.confidence.dtype == np.float32
        assert detections.class_id.shape == (n,)
        assert np.issubdtype(detections.class_id.dtype, np.integer)
        assert detections.tracker_id.shape == (n,)
        assert np.issubdtype(detections.tracker_id.dtype, np.integer)
        assert detections.mask.shape == (n, height, width)
        assert detections.mask.dtype == bool

    def test_default_scenario_never_fully_empties_the_table(self) -> None:
        tracker = ScenarioTracker()

        for _ in range(1000):
            detections = tracker.update(FRAME)
            assert len(detections.tracker_id) > 0

    def test_all_gone_scenario_returns_sv_detections_empty(self) -> None:
        all_gone = (
            ScenarioEvent(tracker_id=1, leave_s=0.0, return_s=None),
            ScenarioEvent(tracker_id=2, leave_s=0.0, return_s=None),
        )
        tracker = ScenarioTracker(n_instruments=2, events=all_gone)
        tracker.update(FRAME)  # t=0.0: still present (boundary-frame convention)

        detections = tracker.update(FRAME)  # t=0.1: both gone

        expected = sv.Detections.empty()
        assert len(detections) == 0
        assert detections.mask is None
        assert detections.tracker_id is None
        assert np.array_equal(detections.xyxy, expected.xyxy)
        assert np.array_equal(detections.confidence, expected.confidence)
        assert np.array_equal(detections.class_id, expected.class_id)


class TestFakeCaptureSource:
    """AC6: read() returns (True, frame) with the requested size; set/get
    record and echo CAP_PROP_* values."""

    def test_read_returns_true_and_a_frame_of_the_requested_size(self) -> None:
        cap = FakeCaptureSource(size=(640, 480), fps=None)

        ok, frame = cap.read()

        assert ok is True
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

    def test_default_size_is_1920x1080(self) -> None:
        cap = FakeCaptureSource(fps=None)

        _, frame = cap.read()

        assert frame.shape == (1080, 1920, 3)

    def test_is_opened_until_released(self) -> None:
        cap = FakeCaptureSource(fps=None)

        assert cap.isOpened() is True
        cap.release()
        assert cap.isOpened() is False

    def test_set_then_get_echoes_the_same_prop_value(self) -> None:
        cap = FakeCaptureSource(fps=None)

        assert cap.set(CAP_PROP_BUFFERSIZE, 1) is True
        assert cap.get(CAP_PROP_BUFFERSIZE) == 1

    def test_get_of_an_unset_prop_defaults_to_zero(self) -> None:
        cap = FakeCaptureSource(fps=None)

        assert cap.get(CAP_PROP_FRAME_WIDTH) == 0.0

    def test_fps_none_reads_are_instant_no_sleep(self) -> None:
        cap = FakeCaptureSource(fps=None)

        start = time.monotonic()
        for _ in range(200):
            cap.read()
        elapsed = time.monotonic() - start

        assert elapsed < 0.5

    def test_fps_set_paces_reads_via_sleep(self) -> None:
        """Verifies the pacing branch is exercised, via a mocked sleep — never
        a real one (per the T01 gate: no real sleeps in the test suite)."""
        cap = FakeCaptureSource(fps=10.0)

        with patch("backend.fakes.time.sleep") as mock_sleep:
            cap.read()

        mock_sleep.assert_called_once_with(0.1)


class TestDefaultScenarioConstant:
    def test_matches_the_frozen_public_interface(self) -> None:
        assert DEFAULT_SCENARIO == (
            ScenarioEvent(tracker_id=1, leave_s=20.0, return_s=35.0),
            ScenarioEvent(tracker_id=3, leave_s=50.0, return_s=None),
        )
