"""T01 AC5/AC6: the fake fixtures match real detector output and record camera state."""

from __future__ import annotations

import cv2
import numpy as np


def test_ac05_fake_dets_shape_roundtrip(fake_dets):
    """AC5: fake_dets round-trips len/.xyxy/.mask/.confidence like real output."""
    dets = fake_dets(
        [[10, 20, 110, 220], [0, 0, 50, 50]],
        confidences=[0.8, 0.6],
        size=(1920, 1080),
    )
    assert len(dets) == 2
    assert dets.xyxy.dtype == np.float32
    assert dets.xyxy.shape == (2, 4)
    assert dets.mask.dtype == bool
    assert dets.mask.shape == (2, 1080, 1920)  # (n, H, W)
    assert dets.confidence.dtype == np.float32
    np.testing.assert_allclose(dets.confidence, [0.8, 0.6], atol=1e-6)
    assert np.array_equal(dets.class_id, np.zeros(2, dtype=int))


def test_ac05_fake_dets_empty_variant(fake_dets):
    """AC5: empty variant has len 0 and the real detector's empty (0, H, W) mask.

    (The real Detector returns an empty (0, H, W) bool array on an empty frame,
    NOT None — proven by the AC7 contract test. The fake matches it.)
    """
    dets = fake_dets([], size=(1920, 1080))
    assert len(dets) == 0
    assert dets.mask.shape == (0, 1080, 1920)


def test_ac05_default_mask_covers_box(fake_dets):
    """The default mask is a solid rectangle inside the box (non-empty → RLE-able)."""
    dets = fake_dets([[100, 100, 200, 300]], size=(1920, 1080))
    mask = dets.mask[0]
    assert mask[150, 150]  # inside the box
    assert not mask[0, 0]  # outside the box
    assert mask.sum() == (200 - 100) * (300 - 100)


def test_ac06_fake_capture_records_sets_and_fails_reads(FakeCapture, fake_frame):
    """AC6: FakeCapture records set() calls and can be told to fail reads."""
    frame = fake_frame(1920, 1080)
    cap = FakeCapture(frames=[frame])

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1920) in cap.set_calls
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in cap.set_calls

    ok, out = cap.read()
    assert ok and out.shape == (1080, 1920, 3)

    dead = FakeCapture(fail_reads=True)
    ok, out = dead.read()
    assert ok is False and out is None
