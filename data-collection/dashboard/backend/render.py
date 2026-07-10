"""Live overlay renderer (T04).

Draws the detector's boxes **and** masks on a 1080p frame, coloured by
instance index so the operator can judge mask quality live (a box can look right
while the mask is garbage). Small, pure, fast — the expensive RLE encoding stays
flag-time only (T02); this only paints pixels.

The caller passes a *copy* it owns (the capture loop does: ``render(frame.copy(),
dets)``); we draw **in place** and return that same array, so the frame ``/flag``
saves is never touched.

**Why the mask fill is hand-rolled** (T04 suggested ``sv.MaskAnnotator``): its
``annotate`` composites over the *whole* frame — a full-frame copy plus a
boolean-index colour write per mask. On a 1920×1080 frame with 5 masks that is
~30 ms here, twice AC6's 15 ms budget. Each mask is only True inside its own
bounding box, so we blend within that crop instead (~4 ms). Boxes and labels
still go through ``sv`` (cheap), and the mask fill reuses ``sv``'s default
palette by index so a mask and its box share one colour.
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

# Colour by instance INDEX, not class: the detector is single-class (every
# class_id == 0), so the default CLASS lookup would paint every instrument the
# same colour and the operator couldn't tell two overlapping masks apart.
_COLOR_LOOKUP = sv.ColorLookup.INDEX
_PALETTE = sv.ColorPalette.DEFAULT

_box_annotator = sv.BoxAnnotator(color_lookup=_COLOR_LOOKUP)
_label_annotator = sv.LabelAnnotator(color_lookup=_COLOR_LOOKUP)

_MASK_OPACITY = 0.5


def _fill_masks(frame: np.ndarray, dets: "sv.Detections") -> None:
    """Blend each mask's colour into the frame, within its bounding box only.

    Painted in descending-area order so smaller masks land on top of larger
    ones (matching ``sv.MaskAnnotator``). Colours come from ``sv``'s palette by
    instance index, so each mask matches its box edge.
    """
    if dets.mask is None:
        return
    h, w = frame.shape[:2]
    for i in np.flip(np.argsort(dets.area)):
        x1, y1, x2, y2 = dets.xyxy[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        sub = frame[y1:y2, x1:x2]
        mask_crop = dets.mask[i][y1:y2, x1:x2].view(np.uint8)
        if not mask_crop.any():
            continue
        solid = np.empty_like(sub)
        solid[:] = _PALETTE.by_idx(int(i)).as_bgr()
        blended = cv2.addWeighted(sub, 1 - _MASK_OPACITY, solid, _MASK_OPACITY, 0)
        cv2.copyTo(blended, mask_crop, sub)


def render(frame: np.ndarray, dets: "sv.Detections") -> np.ndarray:
    """Draws IN PLACE on ``frame`` (caller passes a copy) and returns it.

    Masks first (fill), then boxes (edges on top), then per-instance confidence
    labels. With no detections the frame is returned untouched.
    """
    if len(dets) == 0:
        return frame

    _fill_masks(frame, dets)
    _box_annotator.annotate(frame, dets)

    if dets.confidence is not None:
        labels = [f"{c:.2f}" for c in dets.confidence]
        _label_annotator.annotate(frame, dets, labels=labels)

    return frame
