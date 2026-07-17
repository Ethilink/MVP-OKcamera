"""Fakes: `ScenarioTracker` (scripted instrument pickups) + `FakeCaptureSource`
(synthetic camera frames). These let T02-T04 be built and tested without a
camera or the real weights (DESIGN D10) — `model/` is not touched; the real
`InstrumentTracker` seam is `orc_model.pipelines.tracking`.

Two phases, one shared clock. `ScenarioState` is the single source of truth for
"which instruments are on the table right now, and where":

- **Setup (before recording):** the operator is still arranging the tray, so
  the detected count *churns* — a stable core of instruments plus a few "extra"
  ones that come and go, making the count breathe (e.g. 8 → 5 → 8). This is what
  lets the setup screen be exercised against a changing count and changing tiles.
- **Recording (after `begin_recording()`):** the clean scripted story from `events` — all
  instruments present except the scripted pickups (`DEFAULT_SCENARIO`: id 1
  leaves at t=20 s and returns at t=35 s; id 3 leaves at t=50 s and never
  returns). No churn, so the report stays crisp.

`FakeCaptureSource` draws the present instruments as coloured shapes at the SAME
boxes the tracker reports, reading the SAME `ScenarioState`. Because both share
one state (one frame clock and one scheduled recording epoch), the pixels line up
with the detections — so the `/status` thumbnail crops show real shapes, and a
when recording starts without requiring a tracker reset.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import supervision as sv

_SURGICAL_INSTRUMENT_CLASS_ID = 0
_CONFIDENCE_VALUE = 0.9
"""Every ScenarioTracker detection reports this fixed confidence — presence
and absence here are scripted by `events`/`churn`, never by confidence flicker
(that kind of noise is `FakeInstrumentTracker`'s job, in `orc_model`)."""


@dataclass(frozen=True)
class ScenarioEvent:
    """One scripted pickup: instrument `tracker_id` leaves the table at
    `leave_s` and returns at `return_s` (`None` = never returns — "missing")."""

    tracker_id: int
    leave_s: float
    return_s: float | None


@dataclass(frozen=True)
class ChurnWindow:
    """A setup-phase absence: instrument `tracker_id` is off the table while the
    time-into-the-current-cycle is in `[absent_from_s, absent_to_s)`. Cycles
    repeat every `churn_period_s`, so the count keeps breathing for as long as
    the operator lingers on the setup screen."""

    tracker_id: int
    absent_from_s: float
    absent_to_s: float


DEFAULT_SCENARIO: tuple[ScenarioEvent, ...] = (
    ScenarioEvent(tracker_id=1, leave_s=20.0, return_s=35.0),
    ScenarioEvent(tracker_id=3, leave_s=50.0, return_s=None),
)


@dataclass(frozen=True)
class ForeignWindow:
    """A scripted foreign object during recording: `tracker_id` (NOT in the
    roster) is present for appear_s <= t < disappear_s (T10)."""

    tracker_id: int
    appear_s: float
    disappear_s: float


DEFAULT_FOREIGN: tuple[ForeignWindow, ...] = (
    ForeignWindow(tracker_id=9, appear_s=40.0, disappear_s=48.0),
)

_DEFAULT_N_INSTRUMENTS = 8
_DEFAULT_CHURN_PERIOD_S = 12.0
# How long a foreign object reads as "resolving" (spinner) after it appears,
# before it settles to gray "Unknown". The real linker holds a track pending for
# its evidence window before deciding; the fake reproduces just that beat so
# `--fake` mode demos spinner-then-settle. TUNABLE (a demo-feel knob only).
_DEFAULT_RESOLVING_WINDOW_S = 1.0
DEFAULT_SETUP_CHURN: tuple[ChurnWindow, ...] = (
    # The "extra" tray items (ids 6-8) drop out in overlapping windows so the
    # count eases 8 → 7 → 6 → 5 → 6 → 7 → 8 across each 12 s cycle and starts a
    # cycle with the whole tray present (t=0 → nothing absent). Core ids 1-5
    # have no windows, so they never churn — the count never falls below 5.
    ChurnWindow(tracker_id=6, absent_from_s=2.0, absent_to_s=7.0),
    ChurnWindow(tracker_id=7, absent_from_s=3.0, absent_to_s=8.0),
    ChurnWindow(tracker_id=8, absent_from_s=4.0, absent_to_s=9.0),
)


class ScenarioState:
    """The shared frame clock + presence/geometry logic behind BOTH fakes.

    Deterministic: everything is a function of the integer frame count (no wall
    clock), so `ScenarioTracker.update()` and `FakeCaptureSource.read()` agree
    frame-for-frame as long as they hold the same instance. Recording begins at
    one shared future frame boundary, so the two fakes re-align without
    conflating the lifecycle signal with tracker.reset().
    """

    def __init__(
        self,
        n_instruments: int = _DEFAULT_N_INSTRUMENTS,
        fps: float = 10.0,
        events: Sequence[ScenarioEvent] = DEFAULT_SCENARIO,
        churn: Sequence[ChurnWindow] = DEFAULT_SETUP_CHURN,
        churn_period_s: float = _DEFAULT_CHURN_PERIOD_S,
        foreign: Sequence[ForeignWindow] = DEFAULT_FOREIGN,
        resolving_window_s: float = _DEFAULT_RESOLVING_WINDOW_S,
    ) -> None:
        self.n_instruments = n_instruments
        self.fps = fps
        self.events = tuple(events)
        self.churn = tuple(churn)
        self.churn_period_s = churn_period_s
        self.foreign = tuple(foreign)
        self.resolving_window_s = resolving_window_s
        self._frame_count = 0
        self._recording_start_frame: int | None = None
        self._recording_end_frame: int | None = None

    @property
    def recording(self) -> bool:
        return (
            self._recording_start_frame is not None
            and self._frame_count >= self._recording_start_frame
            and (
                self._recording_end_frame is None
                or self._frame_count < self._recording_end_frame
            )
        )

    @property
    def time_s(self) -> float:
        if self.recording:
            assert self._recording_start_frame is not None
            return (self._frame_count - self._recording_start_frame) / self.fps
        return self._frame_count / self.fps

    def reset(self) -> None:
        """Begin a fresh setup/enrolment pass at t=0.

        Runtime confidence changes call tracker.reset(); recording Start does
        not. The fixed fake catalog/roster remains available throughout."""
        self._frame_count = 0
        self._recording_start_frame = None
        self._recording_end_frame = None

    def begin_recording(self) -> None:
        """Schedule the scripted recording story on the next shared frame.

        Both FakeCaptureSource and ScenarioTracker judge `recording` from the
        same counter. Scheduling `current + 1` means an HTTP call landing between
        a source read and tracker update cannot split that in-flight frame across
        setup and recording modes."""
        self._recording_start_frame = self._frame_count + 1
        self._recording_end_frame = None

    def end_recording(self) -> None:
        """Return the fake tray to setup on the next shared frame boundary."""
        self._recording_end_frame = self._frame_count + 1

    def advance(self) -> None:
        """Consume the current frame. Only the tracker advances the clock; the
        capture source READS the current frame (drawing it) without advancing,
        so both see the same frame within one capture tick."""
        self._frame_count += 1

    def present_ids(self) -> list[int]:
        t = self.time_s
        absent = self._scenario_absent if self.recording else self._churn_absent
        ids = [
            tracker_id
            for tracker_id in range(1, self.n_instruments + 1)
            if not absent(tracker_id, t)
        ]
        # Foreign objects only land on the table during recording — nobody drops
        # a phone on the tray while it is still being arranged (T10/B-F1).
        if self.recording:
            ids += [window.tracker_id for window in self.foreign_present(t)]
        return ids

    def foreign_present(self, t: float) -> tuple[ForeignWindow, ...]:
        """The foreign windows open at `t` — shared by the tracker (which reports
        their ids) and the drawing (which gives them a non-instrument shape), so
        detections and pixels cannot disagree."""
        return tuple(w for w in self.foreign if w.appear_s <= t < w.disappear_s)

    def is_resolving(self, tracker_id: int) -> bool:
        """True while a present foreign object is still in its simulated re-id
        window — the first `resolving_window_s` after it appears — so the fake
        emits the spinner-then-settle beat the real linker produces for an
        undecided track. Roster instruments never resolve: the fake enrols
        instantly and re-emits their ids unchanged, so they are always settled."""
        if not self.recording:
            return False
        t = self.time_s
        return any(
            window.tracker_id == tracker_id
            and window.appear_s <= t < window.appear_s + self.resolving_window_s
            and t < window.disappear_s
            for window in self.foreign
        )

    def box(self, tracker_id: int, width: int, height: int) -> list[float]:
        return self._box(tracker_id, self.time_s, width, height)

    def _scenario_absent(self, tracker_id: int, t: float) -> bool:
        for event in self.events:
            if event.tracker_id != tracker_id:
                continue
            if event.return_s is None:
                if t > event.leave_s:
                    return True
            elif event.leave_s < t < event.return_s:
                return True
        return False

    def _churn_absent(self, tracker_id: int, t: float) -> bool:
        phase = t % self.churn_period_s
        for window in self.churn:
            if window.tracker_id == tracker_id and window.absent_from_s <= phase < window.absent_to_s:
                return True
        return False

    def _box(self, tracker_id: int, t: float, width: int, height: int) -> list[float]:
        phase = t * 0.2 + tracker_id
        cx = (0.15 + 0.7 * tracker_id / (self.n_instruments + 1)) * width
        cx += math.sin(phase) * 0.02 * width
        cy = (0.5 + 0.15 * math.sin(phase * 0.6)) * height
        box_w, box_h = 0.10 * width, 0.08 * height
        x1 = max(0.0, cx - box_w / 2)
        y1 = max(0.0, cy - box_h / 2)
        x2 = min(float(width), cx + box_w / 2)
        y2 = min(float(height), cy + box_h / 2)
        return [x1, y1, x2, y2]


class ScenarioTracker:
    """Implements `InstrumentTracker` (model/docs/tracker-interface.md) on top of
    a shared `ScenarioState`.

    Constructed standalone it owns a fresh state (the unit-test path); in
    `--fake` mode it shares its state with `FakeCaptureSource` so drawn frames
    and detections line up. `reset()` starts a fresh setup pass;
    `begin_recording()` independently anchors the scripted recording story.

    Emits the seam's `data["resolving"]` flag per detection (see
    `tracker-interface.md`): a foreign object reads resolving for its first
    `resolving_window_s`, then settled; roster ids are always settled.

    Boundary-frame convention (so T02 windows land on exact seconds): during
    recording an instrument is ABSENT for frames where `leave_s < t < return_s`
    (`return_s` None => absent for all t > leave_s). So it is PRESENT at exactly
    t=leave_s (its last-seen frame) and PRESENT again at exactly t=return_s. At
    fps=10 that means present at t=20.0, absent from t=20.1 through t=34.9,
    present at t=35.0.
    """

    def __init__(
        self,
        n_instruments: int = _DEFAULT_N_INSTRUMENTS,
        fps: float = 10.0,
        events: Sequence[ScenarioEvent] = DEFAULT_SCENARIO,
        confidence: float = 0.5,
        churn: Sequence[ChurnWindow] = DEFAULT_SETUP_CHURN,
        churn_period_s: float = _DEFAULT_CHURN_PERIOD_S,
        foreign: Sequence[ForeignWindow] = DEFAULT_FOREIGN,
        resolving_window_s: float = _DEFAULT_RESOLVING_WINDOW_S,
        state: ScenarioState | None = None,
    ) -> None:
        self.confidence = confidence
        self._state = state if state is not None else ScenarioState(
            n_instruments=n_instruments,
            fps=fps,
            events=events,
            churn=churn,
            churn_period_s=churn_period_s,
            foreign=foreign,
            resolving_window_s=resolving_window_s,
        )

    @property
    def state(self) -> ScenarioState:
        return self._state

    @property
    def fps(self) -> float:
        return self._state.fps

    @property
    def n_instruments(self) -> int:
        return self._state.n_instruments

    @property
    def events(self) -> tuple[ScenarioEvent, ...]:
        return self._state.events

    @property
    def class_names(self) -> dict[int, str]:
        return {_SURGICAL_INSTRUMENT_CLASS_ID: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "scenario-0.1"

    @property
    def roster(self) -> frozenset[int]:
        """The fake enrols instantly and never revises: the tray IS the roster.
        ForeignWindow ids live outside it (the default uses 9)."""
        return frozenset(range(1, self._state.n_instruments + 1))

    @property
    def catalog(self) -> frozenset[int]:
        """The fake's fixed catalog == its roster: the tray is fully known and
        fully enrolled. (T11 seam widening; constant across reset().)"""
        return frozenset(range(1, self._state.n_instruments + 1))

    @property
    def match_debug(self) -> dict[int, object]:
        # No real matcher runs in fake mode -- deliberately blank
        # (feat/matching-tests).
        return {}

    def reset(self) -> None:
        self._state.reset()

    def begin_recording(self) -> None:
        """Start the fake pickup story without mutating tracker identity state."""
        self._state.begin_recording()

    def end_recording(self) -> None:
        """Return the fake scene to its setup choreography after Stop."""
        self._state.end_recording()

    def update(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]
        present_ids = self._state.present_ids()
        if not present_ids:
            self._state.advance()
            return sv.Detections.empty()

        boxes = [self._state.box(tracker_id, width, height) for tracker_id in present_ids]
        masks = [_mask(box, width, height) for box in boxes]
        # Computed before advance() so it reads THIS frame's clock, exactly like
        # present_ids/box above. A foreign object is resolving for its first
        # second, then settles; roster ids are always settled.
        resolving = np.array(
            [self._state.is_resolving(tracker_id) for tracker_id in present_ids], dtype=bool
        )
        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            mask=np.stack(masks),
            confidence=np.full(len(present_ids), _CONFIDENCE_VALUE, dtype=np.float32),
            class_id=np.full(len(present_ids), _SURGICAL_INSTRUMENT_CLASS_ID, dtype=int),
            tracker_id=np.array(present_ids, dtype=int),
            data={"resolving": resolving},
        )
        self._state.advance()
        return detections


def _mask(box: list[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), dtype=bool)
    mask[int(y1) : int(y2), int(x1) : int(x2)] = True
    return mask


# A small, fixed BGR palette so each tracker_id draws (and therefore crops) as a
# visibly distinct coloured shape. Cycles for ids beyond its length.
_SHAPE_COLORS: tuple[tuple[int, int, int], ...] = (
    (66, 133, 244),   # blue
    (52, 168, 83),    # green
    (0, 165, 255),    # orange
    (180, 120, 60),   # steel
    (128, 62, 178),   # purple
    (60, 180, 200),   # teal
    (80, 80, 220),    # red
    (200, 160, 40),   # cyan-ish
)


def _color_for(tracker_id: int) -> tuple[int, int, int]:
    return _SHAPE_COLORS[(tracker_id - 1) % len(_SHAPE_COLORS)]


def _draw_scenario(frame: np.ndarray, state: ScenarioState) -> None:
    """Draw each present instrument as a filled coloured block with a light
    'shaft' — enough for the `/status` crop to read as a distinct instrument
    rather than a black square. A present foreign object draws as a dark slab
    instead, so the demo's Unknown is visibly not an instrument (T10/B-F3). Uses
    the SAME boxes the tracker will report for this frame (state is not advanced
    by reading)."""
    height, width = frame.shape[:2]
    foreign_ids = {window.tracker_id for window in state.foreign}
    for tracker_id in state.present_ids():
        x1, y1, x2, y2 = (int(round(v)) for v in state.box(tracker_id, width, height))
        if x2 <= x1 or y2 <= y1:
            continue
        if tracker_id in foreign_ids:
            _draw_foreign(frame, x1, y1, x2, y2)
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), _color_for(tracker_id), thickness=-1)
        cx = (x1 + x2) // 2
        cv2.line(frame, (cx, y1 + 3), (cx, y2 - 3), (255, 255, 255), thickness=2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), thickness=2)


def _draw_foreign(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """A dark slab with a pale screen — reads as a phone dropped on the tray,
    not as a 9th instrument."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (45, 40, 40), thickness=-1)
    cv2.rectangle(frame, (x1 + 8, y1 + 8), (x2 - 8, y2 - 8), (130, 125, 120), thickness=-1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), thickness=2)


class FakeCaptureSource:
    """`cv2.VideoCapture`-compatible (isOpened/read/set/get/release) emitting
    1920x1080 BGR synthetic frames. For `--fake` mode AND as the tests' capture
    source. Note: it does NOT take a camera index — T03's `CaptureLoop` calls
    `cap_factory(camera_index)`, so wrap it as `lambda _idx: FakeCaptureSource()`
    rather than passing the class directly.

    Pass a shared `scenario` (the tracker's `ScenarioState`) to draw the present
    instruments as coloured shapes aligned with the detections; without one the
    frames are blank (the plain test double). Reading NEVER advances the shared
    clock — the tracker's `update()` does, once per capture tick, so the drawn
    frame and the detections describe the same instant.

    **Pacing (critical for --fake correctness).** When `fps` is not None,
    `read()` blocks so frames are emitted at ~`fps` in WALL time — this makes
    scenario seconds ~= wall seconds, so `ScenarioTracker(fps=10)`'s scripted
    t=20/35/50 line up with `Session`'s wall-clock debounce (1.5 s). Without
    pacing the loop free-runs at encode speed and the 15 s scripted pickup
    collapses below the debounce and never registers. Demo/`--fake` mode uses
    `fps=10.0` (matching the ScenarioTracker fps); UNIT TESTS pass `fps=None`
    for instant frames and drive timing by other means (injected clock /
    frame counts), never real sleeps.
    """

    def __init__(
        self,
        size: tuple[int, int] = (1920, 1080),
        fps: float | None = 10.0,
        scenario: ScenarioState | None = None,
    ) -> None:
        self._width, self._height = size
        self._fps = fps
        self._scenario = scenario
        self._frame_count = 0
        self._props: dict[int, float] = {}
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        if self._fps is not None:
            time.sleep(1.0 / self._fps)

        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        if self._scenario is not None:
            _draw_scenario(frame, self._scenario)
        frame[0, 0] = (
            self._frame_count & 0xFF,
            (self._frame_count >> 8) & 0xFF,
            (self._frame_count >> 16) & 0xFF,
        )
        self._frame_count += 1
        return True, frame

    def set(self, prop_id: int, value: float) -> bool:
        self._props[prop_id] = value
        return True

    def get(self, prop_id: int) -> float:
        return self._props.get(prop_id, 0.0)

    def release(self) -> None:
        self._opened = False
