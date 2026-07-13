"""T04 — overlay renderer tests, one per acceptance criterion.

Uses the shared ``make_fake_dets`` / ``make_fake_frame`` builders (tests.fakes),
which mimic the real ``sv.Detections`` shape, so no camera or weights are needed.
"""

from __future__ import annotations

import time

import numpy as np

from backend.render import render
from tests.fakes import make_fake_dets, make_fake_frame


def test_ac1_returns_same_array_object():
    """AC1 — in-place contract: the returned array *is* the input array."""
    frame = make_fake_frame(320, 240)
    dets = make_fake_dets([[10, 10, 100, 100]], size=(320, 240))
    out = render(frame, dets)
    assert out is frame


def test_ac2_zero_detections_leaves_frame_unchanged():
    """AC2 — no detections (empty set, mask is (0,H,W)) → untouched, no raise."""
    frame = make_fake_frame(320, 240)
    original = frame.copy()
    out = render(frame, make_fake_dets([], size=(320, 240)))
    assert out is frame
    assert np.array_equal(out, original)


def test_ac3_two_detections_get_distinct_colors():
    """AC3 — two instances differ from each other and from the background."""
    w, h = 320, 240
    frame = make_fake_frame(w, h)
    before = frame.copy()
    dets = make_fake_dets(
        [[20, 20, 80, 80], [200, 140, 280, 210]], size=(w, h)
    )
    render(frame, dets)

    p1 = frame[50, 50].astype(int)      # inside mask 0
    p2 = frame[175, 240].astype(int)    # inside mask 1
    bg1 = before[50, 50].astype(int)
    bg2 = before[175, 240].astype(int)

    assert not np.array_equal(p1, p2)          # instances differ
    assert not np.array_equal(p1, bg1)         # each differs from background
    assert not np.array_equal(p2, bg2)


def test_ac4_box_edges_and_mask_fill_both_drawn():
    """AC4 — a pixel on the box border and a pixel inside the mask both change."""
    w, h = 320, 240
    frame = make_fake_frame(w, h)
    before = frame.copy()
    box = [40, 40, 160, 160]
    render(frame, make_fake_dets([box], size=(w, h)))

    x1, y1, x2, y2 = box
    # Mask fill: a pixel well inside the box interior.
    inside = (y1 + y2) // 2, (x1 + x2) // 2
    assert not np.array_equal(frame[inside], before[inside])

    # Box edge: a pixel on the top border differs somewhere along the edge.
    top_edge_changed = np.any(
        frame[y1, x1:x2] != before[y1, x1:x2]
    )
    assert top_edge_changed


def test_ac5_box_with_all_false_mask_still_renders_box():
    """AC5 — box + all-False mask: box drawn, thresholded independently of mask."""
    w, h = 320, 240
    frame = make_fake_frame(w, h)
    before = frame.copy()
    box = [40, 40, 160, 160]
    empty_mask = np.zeros((1, h, w), dtype=bool)
    dets = make_fake_dets([box], masks=empty_mask, size=(w, h))

    render(frame, dets)

    x1, y1, x2, y2 = box
    # Interior is untouched (no mask fill) ...
    interior = (y1 + y2) // 2, (x1 + x2) // 2
    assert np.array_equal(frame[interior], before[interior])
    # ... but the box edge is still drawn.
    assert np.any(frame[y1, x1:x2] != before[y1, x1:x2])


def test_ac6_fast_enough_on_1080p_with_five_masks():
    """AC6 — p50 < 15 ms on a 1920×1080 frame with 5 masks (rough guard)."""
    w, h = 1920, 1080
    boxes = [
        [100 + i * 300, 100, 300 + i * 300, 400] for i in range(5)
    ]
    dets = make_fake_dets(boxes, size=(w, h))

    timings = []
    for _ in range(11):
        frame = make_fake_frame(w, h)
        t0 = time.perf_counter()
        render(frame, dets)
        timings.append(time.perf_counter() - t0)

    timings.sort()
    p50 = timings[len(timings) // 2]
    assert p50 < 0.015, f"p50 render {p50 * 1000:.1f} ms exceeds 15 ms"
