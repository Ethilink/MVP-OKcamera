"""T01 AC7: guard fake-vs-real drift.

Runs the REAL Detector on one synthetic frame and asserts its sv.Detections has
the exact shape FakeDetector/fake_dets mimic — xyxy float32, bool full-frame
masks (or None), confidence float32, class_id all 0. Skipped on weight-less
machines/CI; runs on Bram's Mac where the ONNX export lives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# model/weights/checkpoint_best_regular.onnx relative to this repo.
_WEIGHTS = (
    Path(__file__).resolve().parents[3]
    / "model"
    / "weights"
    / "checkpoint_best_regular.onnx"
)

pytestmark = pytest.mark.skipif(
    not _WEIGHTS.exists(),
    reason=f"real ONNX weights absent ({_WEIGHTS}) — contract test runs only where weights exist",
)


def test_ac07_real_detector_matches_fake_shape(fake_frame):
    from orc_model.components.detector import Detector

    detector = Detector(_WEIGHTS)
    frame = fake_frame(1920, 1080)  # native 1080p frame, like the capture loop feeds
    dets = detector.predict(frame)

    # xyxy: float32, (n, 4)
    assert dets.xyxy.dtype == np.float32
    assert dets.xyxy.ndim == 2 and dets.xyxy.shape[1] == 4
    n = len(dets)

    # confidence: float32, length n
    assert dets.confidence is not None
    assert dets.confidence.dtype == np.float32
    assert dets.confidence.shape == (n,)

    # class_id: all zeros (single class)
    assert dets.class_id is not None
    assert np.array_equal(dets.class_id, np.zeros(n, dtype=dets.class_id.dtype))

    # mask: bool full-frame (n, H, W) matching the input frame. On an empty
    # frame the real detector returns an empty (0, H, W) array (NOT None) — the
    # fake mimics exactly this. Both cases: dtype bool, trailing dims == frame.
    assert dets.mask is not None
    assert dets.mask.dtype == bool
    assert dets.mask.shape == (n, 1080, 1920)
