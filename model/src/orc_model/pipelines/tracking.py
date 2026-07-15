"""The seam between the model (RF-DETR detector + tracker) and its consumers.

A *consumer* (the data-collection dashboard, later the demo backend) depends ONLY
on the `InstrumentTracker` interface below — never on the detector internals. The
producer is free to retrain the detector, swap the ONNX export, retune the tracker,
etc.; as long as `update()` returns the `sv.Detections` shape documented here,
nothing reaches the consumer.

See `docs/tracker-interface.md` for the full contract and rationale.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import supervision as sv

from orc_model.components.detector.detector import Detector
from orc_model.pipelines.deep_ocsort.tracker import DeepOCSortTracker
from orc_model.pipelines.matching import ChampionMethod
from orc_model.pipelines.session_linker import SessionLinker


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


def load_tracker(
    weights_path: str | Path,
    confidence: float = 0.5,
    *,
    fps: float = 12.0,
    max_age_seconds: float = 1.0,
    absent_death_s: float | None = None,
) -> InstrumentTracker:
    """Compose the real `InstrumentTracker`: RF-DETR detector -> Deep OC-SORT
    -> SessionLinker, per `model/docs/linker-design.md` "Where it composes".
    This is the concrete unblock for `app/backend/backend/main.py`'s
    `load_tracker(args.weights)` call site.

    `weights_path` is the only required argument -- every other knob has a
    workable default, so `load_tracker(weights_path)` alone (the app's call
    site) just works.

    `fps` is an HONEST ESTIMATE, not a measurement: live inference is the
    bottleneck and has historically run ~10-15 fps end-to-end on this
    hardware, so 12.0 is a conservative midpoint. TODO(W9, T05 brief): replace
    this default with the value the first live-camera run actually measures.
    NEVER let this default drift to 30 -- `DeepOCSortTracker.frame_rate`
    defaults to 30 internally, which is exactly why `fps` is threaded through
    explicitly to both the tracker and the linker below (T05 brief C4): a
    wrong fps here silently stretches the tracker's "1.0 s" coast window to
    ~2.5 s of wall time at real fps, breaching the tracker/linker boundary
    described in linker-design.md §9.

    `max_age_seconds` is Deep OC-SORT's coasting window before it frees a raw
    id. `absent_death_s` is the linker's window before it declares a
    *session* identity Missing. These must satisfy
    `absent_death_s >= max_age_seconds` -- otherwise the linker could bury an
    identity while the tracker is still coasting its raw id, and a coasted
    reactivation would silently resurrect a "dead" identity's raw id without
    ever crossing the linker's re-id gate. By default `absent_death_s` is
    pinned to `max_age_seconds`; pass it explicitly only to widen it further,
    a narrower value raises `ValueError`.

    Constructs the SRC matcher (`ChampionMethod`) eagerly, which loads
    DINOv2-B from the local HF cache -- so this call fails fast at startup on
    a missing/mismatched cache instead of on the first tracked frame.
    """
    if absent_death_s is None:
        absent_death_s = max_age_seconds
    elif absent_death_s < max_age_seconds:
        raise ValueError(
            f"absent_death_s ({absent_death_s}) must be >= max_age_seconds "
            f"({max_age_seconds}): the linker must never declare an identity "
            "dead before the tracker itself has given up coasting its raw id."
        )

    weights_path = Path(weights_path)
    detector = Detector(weights_path, confidence_threshold=confidence)
    matcher = ChampionMethod()  # loads DINOv2-B eagerly -- fail fast, not on first frame
    session_linker = SessionLinker(matcher, fps=fps, absent_death_s=absent_death_s)

    return _RealInstrumentTracker(
        detector=detector,
        confidence=confidence,
        fps=fps,
        max_age_seconds=max_age_seconds,
        session_linker=session_linker,
        model_version=_model_version(weights_path),
    )


def _model_version(weights_path: Path) -> str:
    """Weights filename stem + first 8 hex chars of the file's sha256 -- a
    cheap one-time provenance stamp, hashed once at load."""
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:8]
    return f"{weights_path.stem}-{digest}"


class _RealInstrumentTracker:
    """The real `InstrumentTracker`: RF-DETR detector -> Deep OC-SORT ->
    SessionLinker. Built by `load_tracker()`; not meant to be constructed
    directly.

    `update()` runs the three-stage composition from
    `model/docs/linker-design.md` "Where it composes" every call. `reset()`
    re-creates the Deep OC-SORT tracker (it has no `reset()` of its own -- a
    fresh instance IS the reset) and resets the linker's session state, but
    keeps the already-loaded detector session and matcher/DINOv2-B model, so
    repeated recordings don't pay the load cost twice.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        confidence: float,
        fps: float,
        max_age_seconds: float,
        session_linker: SessionLinker,
        model_version: str,
    ) -> None:
        self.confidence = confidence
        self._detector = detector
        self._fps = fps
        self._max_age_seconds = max_age_seconds
        self._session_linker = session_linker
        self._model_version = model_version
        self._deep_ocsort = self._new_deep_ocsort()

    def _new_deep_ocsort(self) -> DeepOCSortTracker:
        # det_thresh stays fixed at the startup confidence (T05 brief C8):
        # OC-SORT's internal gate, distinct from the per-call detector
        # threshold below, which does track `self.confidence` live.
        return DeepOCSortTracker(
            det_thresh=self.confidence,
            frame_rate=self._fps,
            max_age_seconds=self._max_age_seconds,
        )

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return self._model_version

    def reset(self) -> None:
        self._deep_ocsort = self._new_deep_ocsort()
        self._session_linker.reset()

    def update(self, frame: np.ndarray) -> sv.Detections:
        detections = self._detector.predict(frame, confidence_threshold=self.confidence)
        detections = self._deep_ocsort.update(detections, frame)
        detections = self._session_linker.update(detections, frame)
        return detections
