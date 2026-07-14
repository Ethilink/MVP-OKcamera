"""The seam between the model (RF-DETR detector + tracker) and its consumers.

A *consumer* (the data-collection dashboard, later the demo backend) depends ONLY
on the `InstrumentTracker` interface below — never on the detector internals. The
producer is free to retrain the detector, swap the ONNX export, retune the tracker,
etc.; as long as `update()` returns the `sv.Detections` shape documented here,
nothing reaches the consumer.

See `docs/tracker-interface.md` for the full contract and rationale.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
import supervision as sv


@runtime_checkable
class InstrumentTracker(Protocol):
    """Stateful, one instance per recording. Feed frames in capture order.

    `update(frame)` returns the tracked detections for THIS frame as an
    `sv.Detections` with these fields guaranteed populated:

        xyxy        float32 (N, 4)      pixel coords in the frame's OWN space
        confidence  float32 (N,)        per detection
        class_id    int     (N,)        key into `class_names`
        tracker_id  int     (N,)        stable & unique per instance this session
        mask        bool    (N, H, W)   full-frame instance masks

    Treat `frame` as read-only: do not mutate its pixels or retain a mutable
    reference. Every output field is row-aligned for this exact input frame;
    boxes contain finite, ordered coordinates in the frame's pixel space and
    may extend outside its boundary (consumers clamp before pixel indexing).

    Only detections at or above `confidence` are returned, and every returned
    detection carries a real `tracker_id` (there is no untracked/sentinel case).
    A frame with nothing to report returns `sv.Detections.empty()`. `update()`
    never returns None and never raises on a valid frame.
    """

    confidence: float
    """Read/write. The live detection threshold — set at startup, changeable
    between frames (e.g. from the dashboard's slider). Only detections >= this
    are returned. Changing it mid-recording may perturb track ids; that's fine,
    each captured frame is an independent image."""

    def update(self, frame: np.ndarray) -> sv.Detections:
        """One read-only BGR frame (H, W, 3 uint8) -> detections for THIS frame."""
        ...

    def reset(self) -> None:
        """Drop all track state. Called once when a new recording starts, so
        each recording is its own `tracker_id` namespace."""
        ...

    @property
    def class_names(self) -> dict[int, str]:
        """Maps `class_id` -> label, e.g. `{0: "surgical_instrument"}`. The
        consumer builds its category table from this."""
        ...

    @property
    def model_version(self) -> str:
        """Provenance string (weights tag/hash). The consumer stashes it with
        every saved frame so a dataset is traceable to the model that made it."""
        ...


class FakeInstrumentTracker:
    """A dependency-free stand-in that honours the `InstrumentTracker` contract,
    so the dashboard can be built and demoed before the real weights exist.

    Emits a few instruments that drift smoothly across the frame with
    oscillating confidence — so some flicker below the threshold (exercising the
    "instrument the model missed" case) and occasionally a frame comes back
    empty (a false negative). Deterministic: same frame count -> same output.
    """

    def __init__(self, confidence: float = 0.5, n_instruments: int = 3) -> None:
        self.confidence = confidence
        self.n_instruments = n_instruments
        self._frame = 0

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "fake-0.1"

    def reset(self) -> None:
        self._frame = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]
        t = self._frame
        self._frame += 1

        boxes: list[list[float]] = []
        masks: list[np.ndarray] = []
        confidences: list[float] = []
        tracker_ids: list[int] = []

        for i in range(self.n_instruments):
            phase = t * 0.05 + i
            cx = (0.2 + 0.6 * (i + 1) / (self.n_instruments + 1)) * width
            cx += math.sin(phase) * 0.03 * width
            cy = (0.5 + 0.2 * math.sin(phase * 0.7 + i)) * height
            box_w, box_h = 0.12 * width, 0.10 * height

            x1 = max(0.0, cx - box_w / 2)
            y1 = max(0.0, cy - box_h / 2)
            x2 = min(float(width), cx + box_w / 2)
            y2 = min(float(height), cy + box_h / 2)

            confidence = 0.55 + 0.40 * math.sin(phase * 1.3 + i * 2.0)
            confidence = min(0.99, max(0.01, confidence))
            if confidence < self.confidence:
                continue  # flickered below threshold -> simulates a miss

            mask = np.zeros((height, width), dtype=bool)
            mask[int(y1) : int(y2), int(x1) : int(x2)] = True

            boxes.append([x1, y1, x2, y2])
            masks.append(mask)
            confidences.append(confidence)
            tracker_ids.append(i)

        if not boxes:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            mask=np.stack(masks),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
            tracker_id=np.array(tracker_ids, dtype=int),
            data={"class_name": np.array(["surgical_instrument"] * len(boxes))},
        )
