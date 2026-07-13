"""Fakes: `ScenarioTracker` (scripted instrument pickups) + `FakeCaptureSource`
(synthetic camera frames). These let T02-T04 be built and tested without a
camera or the real weights (DESIGN D10) — `model/` is not touched; the real
`InstrumentTracker` seam is `orc_model.pipelines.tracking`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import supervision as sv

_SURGICAL_INSTRUMENT_CLASS_ID = 0
_CONFIDENCE_VALUE = 0.9
"""Every ScenarioTracker detection reports this fixed confidence — presence
and absence here are scripted by `events`, never by confidence flicker (that
kind of noise is `FakeInstrumentTracker`'s job, in `orc_model`)."""


@dataclass(frozen=True)
class ScenarioEvent:
    """One scripted pickup: instrument `tracker_id` leaves the table at
    `leave_s` and returns at `return_s` (`None` = never returns — "missing")."""

    tracker_id: int
    leave_s: float
    return_s: float | None


DEFAULT_SCENARIO: tuple[ScenarioEvent, ...] = (
    ScenarioEvent(tracker_id=1, leave_s=20.0, return_s=35.0),
    ScenarioEvent(tracker_id=3, leave_s=50.0, return_s=None),
)


class ScenarioTracker:
    """Implements `InstrumentTracker` (model/docs/tracker-interface.md).

    Instrument *positions* drift smoothly; `events` script pickups. Time =
    frame_count/fps (deterministic — no wall clock). `tracker_id`s are the
    integers `1..n_instruments` (1-based; `DEFAULT_SCENARIO`'s ids 1 and 3 are
    within {1..5}). `reset()` restarts the script at t=0.

    Boundary-frame convention (so T02 windows land on exact seconds): an
    instrument is ABSENT from `update()` for frames where
    `leave_s < t < return_s` (`return_s` None => absent for all t > leave_s).
    So it is PRESENT at exactly t=leave_s (its last-seen frame) and PRESENT
    again at exactly t=return_s. At fps=10 that means present at t=20.0,
    absent from t=20.1 through t=34.9, present at t=35.0.
    """

    def __init__(
        self,
        n_instruments: int = 5,
        fps: float = 10.0,
        events: Sequence[ScenarioEvent] = DEFAULT_SCENARIO,
        confidence: float = 0.5,
    ) -> None:
        self.confidence = confidence
        self.n_instruments = n_instruments
        self.fps = fps
        self.events = events
        self._frame_count = 0

    @property
    def class_names(self) -> dict[int, str]:
        return {_SURGICAL_INSTRUMENT_CLASS_ID: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "scenario-0.1"

    def reset(self) -> None:
        self._frame_count = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]
        t = self._frame_count / self.fps
        self._frame_count += 1

        present_ids = [
            tracker_id
            for tracker_id in range(1, self.n_instruments + 1)
            if not self._is_absent(tracker_id, t)
        ]
        if not present_ids:
            return sv.Detections.empty()

        boxes = [self._box(tracker_id, t, width, height) for tracker_id in present_ids]
        masks = [self._mask(box, width, height) for box in boxes]

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            mask=np.stack(masks),
            confidence=np.full(len(present_ids), _CONFIDENCE_VALUE, dtype=np.float32),
            class_id=np.full(len(present_ids), _SURGICAL_INSTRUMENT_CLASS_ID, dtype=int),
            tracker_id=np.array(present_ids, dtype=int),
        )

    def _is_absent(self, tracker_id: int, t: float) -> bool:
        for event in self.events:
            if event.tracker_id != tracker_id:
                continue
            if event.return_s is None:
                if t > event.leave_s:
                    return True
            elif event.leave_s < t < event.return_s:
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

    @staticmethod
    def _mask(box: list[float], width: int, height: int) -> np.ndarray:
        x1, y1, x2, y2 = box
        mask = np.zeros((height, width), dtype=bool)
        mask[int(y1) : int(y2), int(x1) : int(x2)] = True
        return mask


class FakeCaptureSource:
    """`cv2.VideoCapture`-compatible (isOpened/read/set/get/release) emitting
    1920x1080 BGR synthetic frames (frame index burned in). For `--fake` mode
    AND as the tests' capture source. Note: it does NOT take a camera index —
    T03's `CaptureLoop` calls `cap_factory(camera_index)`, so wrap it as
    `lambda _idx: FakeCaptureSource()` rather than passing the class directly.

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

    def __init__(self, size: tuple[int, int] = (1920, 1080), fps: float | None = 10.0) -> None:
        self._width, self._height = size
        self._fps = fps
        self._frame_count = 0
        self._props: dict[int, float] = {}
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        if self._fps is not None:
            time.sleep(1.0 / self._fps)

        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
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
