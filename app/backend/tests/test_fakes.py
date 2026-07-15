"""AC2-AC6 for T01 (`ScenarioTracker` + `FakeCaptureSource`) and T10's B-F.

All tests pass `fps=None` to `FakeCaptureSource` (instant frames) and drive
`ScenarioTracker` timing purely by frame count — no real sleeps anywhere in
this suite (see the T01 task spec's FakeCaptureSource pacing note).

T10 (B-F) gives the fakes a foreign object so `--fake` mode can demo the whole
Unknown story without a camera. The scripted `DEFAULT_FOREIGN` *timings* are a
tunable, so the tests below derive their instants from the constant instead of
hard-coding 40/48 — what they pin is the relationship the demo depends on (the
window lands between instrument 1's return and instrument 3's loss).
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import supervision as sv
from orc_model.pipelines.tracking import InstrumentTracker

from backend.fakes import (
    DEFAULT_FOREIGN,
    DEFAULT_SCENARIO,
    FakeCaptureSource,
    ForeignWindow,
    ScenarioEvent,
    ScenarioState,
    ScenarioTracker,
)
from backend.session import Session

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


# --- T10 B-F: the fakes demo the whole Unknown story --------------------------

_FOREIGN_ID = 9
"""The id the B-F tests script as a foreign object. It must sit OUTSIDE the
roster (B-F2) yet still inside `ScenarioState._box`'s geometry, which spreads
ids across the frame by `id / (n_instruments + 1)` — so with the default tray
of 8, id 9 is the natural (and the spec's) choice."""


def _drive_state_to(state: ScenarioState, t: float) -> ScenarioState:
    """Advance a state's frame clock so `state.time_s` lands exactly on `t`.
    Frame count only — no wall clock, mirroring `_update_until` above."""
    for _ in range(round(t * state.fps)):
        state.advance()
    return state


def _recording_state_at(t: float, **kwargs) -> ScenarioState:
    """A state in the RECORDING phase (i.e. past a `Start`) at scenario-time
    `t`. Foreign windows belong to recording; `reset()` is what enters it."""
    state = ScenarioState(fps=10.0, **kwargs)
    state.reset()
    return _drive_state_to(state, t)


def _box_centre(state: ScenarioState, tracker_id: int, width: int, height: int):
    x1, y1, x2, y2 = state.box(tracker_id, width, height)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _box_mean(frame: np.ndarray, box, inset: int = 4) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    region = frame[y1 + inset : y2 - inset, x1 + inset : x2 - inset]
    return region.reshape(-1, 3).astype(np.float64).mean(axis=0)


def _boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class TestBF1ForeignObjectsAppearDuringRecording:
    """B-F1: during RECORDING, `present_ids()` additionally includes any
    `ForeignWindow` id whose window contains `t`. During SETUP churn foreign
    windows are ignored — nobody puts a phone on the tray while arranging it."""

    WINDOW = ForeignWindow(tracker_id=_FOREIGN_ID, appear_s=5.0, disappear_s=8.0)

    def test_b_f1_recording_present_ids_include_a_foreign_object_in_its_window(
        self,
    ) -> None:
        state = _recording_state_at(6.0, foreign=(self.WINDOW,))

        assert _FOREIGN_ID in state.present_ids()

    def test_b_f1_recording_present_ids_exclude_a_foreign_object_outside_its_window(
        self,
    ) -> None:
        for t in (0.0, 4.0, 9.0, 20.0):
            state = _recording_state_at(t, foreign=(self.WINDOW,))

            assert _FOREIGN_ID not in state.present_ids(), f"still there at t={t}"

    def test_b_f1_the_foreign_window_is_half_open_on_both_ends(self) -> None:
        # The interface spells the window out as `appear_s <= t < disappear_s`.
        present_at = lambda t: _recording_state_at(t, foreign=(self.WINDOW,)).present_ids()

        assert _FOREIGN_ID not in present_at(4.9)
        assert _FOREIGN_ID in present_at(5.0)  # present AT appear_s
        assert _FOREIGN_ID in present_at(7.9)
        assert _FOREIGN_ID not in present_at(8.0)  # ... gone AT disappear_s

    def test_b_f1_setup_churn_ignores_foreign_windows(self) -> None:
        state = ScenarioState(fps=10.0, foreign=(self.WINDOW,))  # no reset: still SETUP
        _drive_state_to(state, 6.0)

        assert _FOREIGN_ID not in state.present_ids()

    def test_b_f1_a_foreign_object_joins_the_instruments_rather_than_displacing_them(
        self,
    ) -> None:
        state = _recording_state_at(6.0, events=(), foreign=(self.WINDOW,))

        assert set(state.present_ids()) == {1, 2, 3, 4, 5, 6, 7, 8, _FOREIGN_ID}

    def test_b_f1_no_foreign_windows_means_no_foreign_ids(self) -> None:
        state = _recording_state_at(6.0, events=(), foreign=())

        assert set(state.present_ids()) == {1, 2, 3, 4, 5, 6, 7, 8}


class TestBF2ScenarioTrackerHasARoster:
    """B-F2: `ScenarioTracker.roster` is `frozenset(range(1, n_instruments+1))`,
    always — the fake "enrols" instantly. ForeignWindow ids must be chosen
    outside it."""

    def test_b_f2_roster_is_the_full_instrument_range(self) -> None:
        assert ScenarioTracker(n_instruments=5).roster == frozenset({1, 2, 3, 4, 5})

    def test_b_f2_roster_members_are_builtin_ints(self) -> None:
        roster = ScenarioTracker(n_instruments=3).roster

        assert roster == frozenset({1, 2, 3})
        assert all(type(member) is int for member in roster)

    def test_b_f2_roster_is_stable_across_updates_and_a_reset(self) -> None:
        tracker = ScenarioTracker()
        expected = frozenset(range(1, tracker.n_instruments + 1))

        assert tracker.roster == expected
        for _ in range(5):
            tracker.update(FRAME)
        tracker.reset()  # a Start rewinds the script; the tray is still the tray
        for _ in range(5):
            tracker.update(FRAME)

        assert tracker.roster == expected

    def test_b_f2_the_default_foreign_ids_sit_outside_the_roster(self) -> None:
        roster = ScenarioTracker().roster

        assert roster  # sanity: the fake enrols instantly
        assert DEFAULT_FOREIGN  # sanity: there is a foreign object to place
        for window in DEFAULT_FOREIGN:
            assert window.tracker_id not in roster


class TestBF3ForeignObjectsAreDrawn:
    """B-F3: a present foreign object is drawn as a visibly different shape at
    `ScenarioState.box(9, ...)` geometry, so detections and pixels stay aligned
    (the same invariant `TestSharedEpochAlignment` guards for instruments)."""

    WINDOW = ForeignWindow(tracker_id=_FOREIGN_ID, appear_s=5.0, disappear_s=8.0)
    SIZE = (320, 240)

    def _frame_at(self, t: float) -> tuple[np.ndarray, ScenarioState]:
        state = _recording_state_at(t, foreign=(self.WINDOW,))
        source = FakeCaptureSource(size=self.SIZE, fps=None, scenario=state)
        _, frame = source.read()
        return frame, state

    def test_b_f3_a_present_foreign_object_is_drawn_at_its_detection_box(self) -> None:
        frame, state = self._frame_at(6.0)

        assert _FOREIGN_ID in state.present_ids()  # sanity: it should be there
        cx, cy = _box_centre(state, _FOREIGN_ID, *self.SIZE)
        assert int(frame[cy, cx].max()) > 0  # ... and drawn where it is detected

    def test_b_f3_an_absent_foreign_object_is_not_drawn(self) -> None:
        frame, state = self._frame_at(9.0)  # past disappear_s

        cx, cy = _box_centre(state, _FOREIGN_ID, *self.SIZE)
        assert int(frame[cy, cx].max()) == 0

    def test_b_f3_the_foreign_object_does_not_look_like_an_instrument(self) -> None:
        frame, state = self._frame_at(6.0)
        width, height = self.SIZE
        foreign_box = state.box(_FOREIGN_ID, width, height)
        foreign = _box_mean(frame, foreign_box)

        assert _FOREIGN_ID in state.present_ids()  # sanity: it should be there
        assert foreign.max() > 0  # ... and something was drawn for it
        for tracker_id in state.present_ids():
            box = state.box(tracker_id, width, height)
            if tracker_id == _FOREIGN_ID or _boxes_overlap(box, foreign_box):
                continue  # an overlapping neighbour would pollute the sample
            assert not np.allclose(foreign, _box_mean(frame, box), atol=30), tracker_id


class TestBF4DefaultForeignTellsTheDemoStory:
    """B-F4: the default window sits AFTER instrument 1's return and BEFORE
    instrument 3's loss, so a `--fake` run reads as pickup/return -> foreign
    object goes gray -> real loss. The instants themselves are a tunable, so
    everything here is derived from the constants rather than spelled out."""

    def test_b_f4_default_foreign_is_a_single_well_formed_window(self) -> None:
        assert len(DEFAULT_FOREIGN) == 1
        window = DEFAULT_FOREIGN[0]
        assert window.appear_s < window.disappear_s

    def test_b_f4_the_window_lands_between_the_return_and_the_loss(self) -> None:
        window = DEFAULT_FOREIGN[0]
        returns = [e.return_s for e in DEFAULT_SCENARIO if e.return_s is not None]
        losses = [e.leave_s for e in DEFAULT_SCENARIO if e.return_s is None]

        assert max(returns) < window.appear_s  # after instrument 1 is back
        assert window.disappear_s < min(losses)  # before instrument 3 is lost

    def test_b_f4_the_default_tracker_shows_the_foreign_object_mid_story(self) -> None:
        window = DEFAULT_FOREIGN[0]
        mid_window = (window.appear_s + window.disappear_s) / 2

        ids = _tracker_ids(_update_until(ScenarioTracker(), t=mid_window))

        assert window.tracker_id in ids  # the foreign object is on the table
        assert 1 in ids  # ... instrument 1 has already come back
        assert 3 in ids  # ... and instrument 3 is not lost yet


class TestBF5TheForeignWindowDoesNotPerturbTheReport:
    """B-F5: with the default scenario the final report still contains exactly
    ids 1-8 with instrument 3 missing. The foreign object is loud on the video
    and invisible to the paperwork."""

    def test_b_f5_the_default_fake_run_still_reports_ids_1_to_8_with_3_missing(
        self,
    ) -> None:
        tracker = ScenarioTracker(fps=10.0)
        session = Session()
        session.start(0.0)
        tracker.reset()  # Start: the scripted recording begins at t=0
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        foreign_id = DEFAULT_FOREIGN[0].tracker_id
        foreign_seen = False

        for step in range(1, 601):  # 60s of the scripted story at 10 fps
            ids = _tracker_ids(tracker.update(frame))
            foreign_seen = foreign_seen or foreign_id in ids
            session.observe(step / 10.0, frozenset(ids), tracker.roster)
        report = session.stop(60.0)

        assert tracker.roster == frozenset(range(1, 9))  # sanity: a real roster
        assert foreign_seen, "sanity: the scripted foreign object never appeared"
        by_id = {ir.tracker_id: ir for ir in report.instruments}
        assert set(by_id) == set(range(1, 9))  # ... and no 9th "instrument"
        assert [tid for tid, ir in by_id.items() if ir.completeness == "missing"] == [3]


class TestBFResolvingFlag:
    """The wait-state fix (2026-07-16): the fake emits the seam's
    `data["resolving"]` flag so `--fake` mode reproduces the real
    spinner-then-settle beat. A foreign object resolves for its first
    `resolving_window_s`, then settles; roster instruments never resolve."""

    WINDOW = ForeignWindow(tracker_id=_FOREIGN_ID, appear_s=5.0, disappear_s=8.0)

    def test_a_foreign_object_resolves_only_at_the_start_of_its_window(self) -> None:
        early = _recording_state_at(5.2, foreign=(self.WINDOW,), resolving_window_s=1.0)
        assert early.is_resolving(_FOREIGN_ID) is True

        late = _recording_state_at(6.5, foreign=(self.WINDOW,), resolving_window_s=1.0)
        assert late.is_resolving(_FOREIGN_ID) is False  # window elapsed -> settled
        assert _FOREIGN_ID in late.present_ids(), "sanity: still present, just settled"

    def test_roster_instruments_never_resolve(self) -> None:
        state = _recording_state_at(5.2, foreign=(self.WINDOW,), resolving_window_s=1.0)

        for member in range(1, state.n_instruments + 1):
            assert state.is_resolving(member) is False

    def test_nothing_resolves_during_setup(self) -> None:
        state = ScenarioState(fps=10.0, foreign=(self.WINDOW,), resolving_window_s=1.0)  # no reset
        _drive_state_to(state, 5.2)

        assert state.is_resolving(_FOREIGN_ID) is False

    def test_tracker_update_emits_a_row_aligned_resolving_flag(self) -> None:
        state = ScenarioState(fps=10.0, events=(), foreign=(self.WINDOW,), resolving_window_s=1.0)
        state.reset()
        tracker = ScenarioTracker(state=state)
        _drive_state_to(state, 5.2)  # inside the foreign object's resolving window

        dets = tracker.update(FRAME)

        ids = [int(i) for i in dets.tracker_id]
        resolving = dets.data["resolving"]
        assert len(resolving) == len(ids), "the flag must be one-per-detection (row-aligned)"
        assert bool(resolving[ids.index(_FOREIGN_ID)]) is True
        assert not any(
            bool(resolving[i]) for i, tid in enumerate(ids) if tid != _FOREIGN_ID
        ), "only the foreign object resolves; every roster id is settled"
