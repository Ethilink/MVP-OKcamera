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
- **Recording (after `reset()`):** the clean scripted story from `events` — all
  instruments present except the scripted pickups (`DEFAULT_SCENARIO`: id 1
  leaves at t=20 s and returns at t=35 s; id 3 leaves at t=50 s and never
  returns). No churn, so the report stays crisp.

`FakeCaptureSource` draws the present instruments as coloured shapes at the SAME
boxes the tracker reports, reading the SAME `ScenarioState`. Because both share
one state (one frame clock, reset together), the drawn pixels always line up
with the detections — so the `/status` thumbnail crops show real shapes, and a
recording `Start` (which resets the shared clock) can't drift the two apart.
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

_DEFAULT_N_INSTRUMENTS = 8
_DEFAULT_CHURN_PERIOD_S = 12.0
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
    frame-for-frame as long as they hold the same instance. `reset()` marks the
    transition into recording AND rewinds the clock, so the two fakes re-align
    on `Start` (the bug a naive two-counter design would have introduced).
    """

    def __init__(
        self,
        n_instruments: int = _DEFAULT_N_INSTRUMENTS,
        fps: float = 10.0,
        events: Sequence[ScenarioEvent] = DEFAULT_SCENARIO,
        churn: Sequence[ChurnWindow] = DEFAULT_SETUP_CHURN,
        churn_period_s: float = _DEFAULT_CHURN_PERIOD_S,
    ) -> None:
        self.n_instruments = n_instruments
        self.fps = fps
        self.events = tuple(events)
        self.churn = tuple(churn)
        self.churn_period_s = churn_period_s
        self._frame_count = 0
        self._recording = False

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def time_s(self) -> float:
        return self._frame_count / self.fps

    def reset(self) -> None:
        """Enter recording and rewind the clock to t=0. Called on `Start`."""
        self._frame_count = 0
        self._recording = True

    def advance(self) -> None:
        """Consume the current frame. Only the tracker advances the clock; the
        capture source READS the current frame (drawing it) without advancing,
        so both see the same frame within one capture tick."""
        self._frame_count += 1

    def present_ids(self) -> list[int]:
        t = self.time_s
        absent = self._scenario_absent if self._recording else self._churn_absent
        return [
            tracker_id
            for tracker_id in range(1, self.n_instruments + 1)
            if not absent(tracker_id, t)
        ]

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
    and detections line up. `reset()` restarts the script at t=0 and switches
    the state from setup churn to the recording scenario.

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
        state: ScenarioState | None = None,
    ) -> None:
        self.confidence = confidence
        self._state = state if state is not None else ScenarioState(
            n_instruments=n_instruments,
            fps=fps,
            events=events,
            churn=churn,
            churn_period_s=churn_period_s,
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

    def reset(self) -> None:
        self._state.reset()

    def update(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]
        present_ids = self._state.present_ids()
        if not present_ids:
            self._state.advance()
            return sv.Detections.empty()

        boxes = [self._state.box(tracker_id, width, height) for tracker_id in present_ids]
        masks = [_mask(box, width, height) for box in boxes]
        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            mask=np.stack(masks),
            confidence=np.full(len(present_ids), _CONFIDENCE_VALUE, dtype=np.float32),
            class_id=np.full(len(present_ids), _SURGICAL_INSTRUMENT_CLASS_ID, dtype=int),
            tracker_id=np.array(present_ids, dtype=int),
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
    rather than a black square. Uses the SAME boxes the tracker will report for
    this frame (state is not advanced by reading)."""
    height, width = frame.shape[:2]
    for tracker_id in state.present_ids():
        x1, y1, x2, y2 = (int(round(v)) for v in state.box(tracker_id, width, height))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), _color_for(tracker_id), thickness=-1)
        cx = (x1 + x2) // 2
        cv2.line(frame, (cx, y1 + 3), (cx, y2 - 3), (255, 255, 255), thickness=2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), thickness=2)


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
