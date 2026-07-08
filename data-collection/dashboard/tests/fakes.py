"""Shared test vocabulary for the dashboard (T01).

The fixtures in ``conftest.py`` are thin wrappers over these builders/classes so
tests can use either style:

    def test_x(fake_dets): dets = fake_dets([[0, 0, 10, 10]])
    from tests.fakes import FakeDetector, FakeCapture

Everything here is camera-free and onnx-free — the fakes mimic the exact
``sv.Detections`` shape the real ``Detector`` emits (see the T01 AC7 contract
test), so blind-TDD tests for T02/T03 never touch hardware or weights.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import supervision as sv


def make_fake_frame(w: int, h: int) -> np.ndarray:
    """Deterministic BGR uint8 frame (H, W, 3): a smooth two-axis gradient.

    Deterministic so pixel-level assertions (e.g. "the saved JPEG is the raw
    frame") are reproducible. Blue ramps along x, green along y, red is their
    average — no random noise.
    """
    xs = np.linspace(0, 255, w, dtype=np.float32)
    ys = np.linspace(0, 255, h, dtype=np.float32)
    b = np.broadcast_to(xs[np.newaxis, :], (h, w))
    g = np.broadcast_to(ys[:, np.newaxis], (h, w))
    r = (b + g) / 2.0
    return np.stack([b, g, r], axis=2).astype(np.uint8)


def make_fake_dets(
    boxes,
    masks=None,
    confidences=None,
    size: tuple[int, int] = (1920, 1080),
) -> sv.Detections:
    """Build a real ``sv.Detections`` mimicking the detector's output shape.

    Args:
        boxes: iterable of ``[x1, y1, x2, y2]`` in frame pixel coords. Empty →
            the empty detection set (``len == 0``, ``mask is None``).
        masks: optional ``(n, H, W)`` bool array. Default: each box filled as a
            solid rectangle over a zeroed ``(n, H, W)`` canvas (H, W = size).
        confidences: optional length-n floats. Default: 0.9 each.
        size: ``(W, H)`` the masks live in — the same resolution as the frame
            the detector ran on (one resolution end to end).

    Matches real output: ``xyxy`` float32, bool full-frame masks, ``confidence``
    float32, ``class_id`` all 0.
    """
    boxes = [list(map(float, b)) for b in boxes]
    n = len(boxes)
    w, h = size

    if n == 0:
        # Match the REAL detector: decode_predictions returns an empty (0, H, W)
        # bool mask, NOT None, when nothing survives thresholding. (The spec's
        # "mask is None on an empty frame" comment is inaccurate; harmless because
        # every consumer guards mask access with `if len(dets)`. Guarded by the
        # AC7 contract test.)
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            mask=np.empty((0, h, w), dtype=bool),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=int),
        )

    xyxy = np.asarray(boxes, dtype=np.float32).reshape(n, 4)

    if masks is None:
        masks = np.zeros((n, h, w), dtype=bool)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
            xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
            if xi2 > xi1 and yi2 > yi1:
                masks[i, yi1:yi2, xi1:xi2] = True
    else:
        masks = np.asarray(masks, dtype=bool)

    if confidences is None:
        confidences = np.full(n, 0.9, dtype=np.float32)
    else:
        confidences = np.asarray(confidences, dtype=np.float32).reshape(n)

    return sv.Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=confidences,
        class_id=np.zeros(n, dtype=int),
    )


class FakeDetector:
    """Stand-in for ``orc_model.components.detector.Detector``.

    - ``confidence_threshold`` is a plain mutable attribute (the real detector
      exposes it the same way; the capture loop snapshots it at predict time).
    - ``predict(frame)`` pops the next programmed ``sv.Detections`` off a queue
      (falls back to the empty set when drained) and records the frame it saw
      and the threshold value in force at that call.
    """

    def __init__(self, predictions=None, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self._queue = list(predictions or [])
        self.seen_frames: list[np.ndarray] = []
        self.seen_thresholds: list[float] = []
        self._lock = threading.Lock()

    def queue(self, dets: sv.Detections) -> None:
        """Append one detection set to the predict queue."""
        with self._lock:
            self._queue.append(dets)

    def predict(self, frame: np.ndarray) -> sv.Detections:
        with self._lock:
            self.seen_frames.append(frame)
            self.seen_thresholds.append(self.confidence_threshold)
            if self._queue:
                return self._queue.pop(0)
        return sv.Detections.empty()


class FakeCapture:
    """Stand-in for ``cv2.VideoCapture``.

    - ``set(prop, value)`` records every call in ``set_calls`` (so tests assert
      the requested width/height + BUFFERSIZE=1 requests).
    - ``read()`` pops the next programmed frame. When the queue drains it repeats
      the last frame (a steady stream for the capture loop) unless
      ``on_empty="fail"``. ``fail_reads=True`` forces ``(False, None)`` — the
      dead-camera case.
    - ``get(prop)`` reports the last frame's actual dimensions for the WIDTH /
      HEIGHT props so a loop can assert the negotiated resolution.
    """

    def __init__(
        self,
        frames=None,
        index: int = 0,
        fail_reads: bool = False,
        on_empty: str = "repeat",
    ) -> None:
        self.index = index
        self.fail_reads = fail_reads
        self.on_empty = on_empty
        self.set_calls: list[tuple[int, float]] = []
        self.read_count = 0
        self.released = False
        self._queue = list(frames or [])
        self._last = self._queue[-1] if self._queue else None
        self._lock = threading.Lock()

    def set(self, prop: int, value) -> bool:
        with self._lock:
            self.set_calls.append((prop, value))
        return True

    def get(self, prop: int):
        with self._lock:
            last = self._last
        if last is None:
            return 0.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(last.shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(last.shape[0])
        return 0.0

    def read(self):
        with self._lock:
            self.read_count += 1
            if self.fail_reads:
                return (False, None)
            if self._queue:
                frame = self._queue.pop(0)
                self._last = frame
                return (True, frame)
            if self.on_empty == "repeat" and self._last is not None:
                return (True, self._last)
            return (False, None)

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API name
        with self._lock:
            return not self.released

    def release(self) -> None:
        with self._lock:
            self.released = True
