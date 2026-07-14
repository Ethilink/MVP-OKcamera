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

from backend.fakes import (
    DEFAULT_SCENARIO,
    FakeCaptureSource,
    ScenarioEvent,
    ScenarioState,
    ScenarioTracker,
)

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
    """Drive a `tracker` frame-by-frame so its LAST `update()` call lands at
    exactly scenario-time `t` (given `tracker.fps`), and return that result.
    `reset()` first: the scripted `events` (id-1/id-3 pickups) belong to the
    RECORDING phase, which `reset()` (a `Start`) enters — before any reset the
    tracker is in setup-churn mode, a different scenario. No wall clock."""
    tracker.reset()
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
        tracker.reset()  # enter the recording scenario the events describe
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


def _drive_setup_to(tracker: ScenarioTracker, t: float) -> sv.Detections:
    """Drive a FRESH (un-reset) tracker to setup-time `t`. Unlike
    `_update_until`, this stays in the setup-churn phase (no `reset()`)."""
    n_calls = round(t * tracker.fps) + 1
    detections = sv.Detections.empty()
    for _ in range(n_calls):
        detections = tracker.update(FRAME)
    return detections


class TestSetupChurn:
    """Before any `reset()` (i.e. on the setup screen) the count breathes: a
    stable core of 5 plus extra ids that come and go, so the frontend can be
    exercised against a changing count and changing tiles. `reset()` switches to
    the clean scripted recording scenario."""

    def test_full_tray_present_at_the_start_of_a_cycle(self) -> None:
        ids = _tracker_ids(_drive_setup_to(ScenarioTracker(), t=0.0))

        assert ids == {1, 2, 3, 4, 5, 6, 7, 8}

    def test_count_dips_to_the_core_when_all_extras_are_away(self) -> None:
        # t=5.0 sits inside every extra's absence window (id6 [2,7), id7 [3,8),
        # id8 [4,9)) → only the core 1-5 remain.
        ids = _tracker_ids(_drive_setup_to(ScenarioTracker(), t=5.0))

        assert ids == {1, 2, 3, 4, 5}

    def test_core_ids_never_churn(self) -> None:
        for t in (0.0, 2.5, 5.0, 8.5, 11.0, 20.0):
            ids = _tracker_ids(_drive_setup_to(ScenarioTracker(), t=t))

            assert {1, 2, 3, 4, 5}.issubset(ids), f"core dropped at t={t}"

    def test_count_actually_varies_across_a_cycle(self) -> None:
        counts = {
            len(_tracker_ids(_drive_setup_to(ScenarioTracker(), t=t)))
            for t in (0.0, 2.5, 5.0, 8.5)
        }

        assert counts == {8, 7, 5}  # 8 (full), 7 (one extra away), 5 (core only)
        assert max(counts) == 8 and min(counts) == 5

    def test_churn_is_deterministic(self) -> None:
        a = _tracker_ids(_drive_setup_to(ScenarioTracker(), t=5.0))
        b = _tracker_ids(_drive_setup_to(ScenarioTracker(), t=5.0))

        assert a == b

    def test_reset_switches_from_churn_to_the_scripted_scenario(self) -> None:
        tracker = ScenarioTracker()
        # In setup churn, id 1 is core → present even deep in a cycle.
        assert 1 in _tracker_ids(_drive_setup_to(tracker, t=27.0))

        # After reset (= Start), the recording scenario governs: id 1 is away in
        # (20, 35), so it is absent at t=27.
        assert 1 not in _tracker_ids(_update_until(ScenarioTracker(), t=27.0))


class TestSharedEpochAlignment:
    """The Codex-flagged bug: a `Start` (`reset()`) must not let the drawn frame
    drift from the detections. With ONE shared `ScenarioState`, the pixels the
    capture source draws always sit on the boxes the tracker reports — before
    AND after a reset."""

    @staticmethod
    def _detected_boxes_are_drawn(frame: np.ndarray, detections: sv.Detections) -> bool:
        if detections.xyxy is None:
            return True
        for x1, y1, x2, y2 in detections.xyxy:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if int(frame[cy, cx].max()) == 0:  # black => nothing drawn here
                return False
        return True

    def _one_tick(self, source: FakeCaptureSource, tracker: ScenarioTracker):
        ok, frame = source.read()          # draws the CURRENT frame (no advance)
        detections = tracker.update(frame)  # reports it, then advances the clock
        return ok, frame, detections

    def test_drawn_shapes_sit_on_the_detected_boxes_each_tick(self) -> None:
        state = ScenarioState(fps=10.0)  # one state, shared by tracker + source
        tracker = ScenarioTracker(state=state)
        source = FakeCaptureSource(size=(320, 240), fps=None, scenario=state)

        for _ in range(15):
            _, frame, detections = self._one_tick(source, tracker)
            assert frame.max() > 0  # something was drawn
            assert self._detected_boxes_are_drawn(frame, detections)

    def test_alignment_survives_a_reset(self) -> None:
        state = ScenarioState(fps=10.0)
        tracker = ScenarioTracker(state=state)
        source = FakeCaptureSource(size=(320, 240), fps=None, scenario=state)

        for _ in range(30):  # run a while in setup churn
            self._one_tick(source, tracker)

        tracker.reset()  # Start: rewinds the SHARED clock for both fakes

        for _ in range(15):  # recording — draw + detect must still line up
            _, frame, detections = self._one_tick(source, tracker)
            assert self._detected_boxes_are_drawn(frame, detections)

    def test_read_does_not_advance_the_clock_only_update_does(self) -> None:
        state = ScenarioState(fps=10.0)
        tracker = ScenarioTracker(state=state)
        source = FakeCaptureSource(size=(320, 240), fps=None, scenario=state)

        source.read()
        source.read()
        assert state.time_s == 0.0  # reads alone never move the clock

        tracker.update(FRAME)
        assert state.time_s == 1.0 / state.fps  # exactly one frame consumed
