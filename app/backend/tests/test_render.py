"""AC10 for T03: `render(frame, dets) -> np.ndarray` — the overlay drawer.

Only AC10 is numbered against `render.py` itself (the frozen interface's
other prose — the exact label text and where "a corner" the on-table count
is burned — isn't independently numbered and isn't pinned down precisely
enough to test without coupling to one implementation's choice of corner /
font, so it is only exercised structurally here, not verified byte-for-byte).

No threads, no clock, no camera here — `render` is a pure function of
(frame, dets). Frames are small so mask/box math stays cheap.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from backend.render import render


def _blank_frame(height: int = 90, width: int = 120) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _boxed_detections(
    boxes: list[tuple[int, int, int, int]],
    tracker_ids: list[int],
    frame_shape: tuple[int, int],
) -> sv.Detections:
    """Builds a minimal, valid `sv.Detections` with a full-frame boolean mask
    per box (True only inside that box) — enough to exercise both the box
    and mask drawing paths without depending on any tracker."""
    height, width = frame_shape
    n = len(boxes)
    mask = np.zeros((n, height, width), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        mask[i, y1:y2, x1:x2] = True
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        mask=mask,
        confidence=np.full(n, 0.9, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
    )


class TestAC10RenderInPlaceContract:
    """AC10 (part 1): render mutates the frame where dets exist, and returns
    the SAME array (identity, size, dtype) — never a fresh allocation."""

    def test_ac10_render_mutates_in_place_and_preserves_shape_and_dtype(self) -> None:
        frame = _blank_frame()
        before = frame.copy()
        dets = _boxed_detections([(10, 10, 50, 50)], [7], frame.shape[:2])

        result = render(frame, dets)

        assert result is frame
        assert result.shape == before.shape
        assert result.dtype == before.dtype
        assert not np.array_equal(frame, before)


class TestAC10RenderColoursByTrackerId:
    """AC10 (part 2): two dets with different tracker_id get different
    colours — checked by comparing the mean colour inside each det's box,
    which is robust to whatever exact palette/lookup scheme is used."""

    def test_ac10_different_tracker_ids_get_different_colours(self) -> None:
        frame = _blank_frame()
        dets = _boxed_detections(
            [(10, 10, 40, 40), (80, 60, 110, 85)], [1, 2], frame.shape[:2]
        )

        result = render(frame, dets)

        color_a = result[10:40, 10:40].reshape(-1, 3).astype(np.float64).mean(axis=0)
        color_b = result[60:85, 80:110].reshape(-1, 3).astype(np.float64).mean(axis=0)
        assert not np.allclose(color_a, color_b, atol=5)
