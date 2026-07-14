"""Consumer-side detection thumbnails for `GET /status` setup previews.

The model returns frame-aligned `xyxy` boxes, not encoded crops. The app backend
owns the camera frame, so it derives the small per-instrument previews here — a
focused, OpenCV-only, FastAPI-free module so the crop geometry and encoding are
unit-testable in isolation and never run under the capture/session lock.

See `app/docs/BACKEND-TODO-setup-detections.md`, `app/docs/api-contract.md`
§/status, and `model/docs/tracker-interface.md` §Consumer-generated crops.
"""

from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from backend.capture import DetectionBox

logger = logging.getLogger(__name__)

_THUMBNAIL_SIZE = 96      # px — the ~96×96 letterboxed canvas per tile
_BBOX_PAD = 0.15          # expand each box ~15 % so instrument ends aren't clipped
_JPEG_QUALITY = 80
_MIN_BOX_PX = 2.0         # boxes narrower/shorter than this are degenerate → skip


@dataclass(frozen=True)
class Detection:
    """One setup detection as the API exposes it: a stable id, a label, and a
    data-URI thumbnail (or `None` when that single crop could not be produced —
    a bad crop must never fail the whole `/status` response)."""

    tracker_id: int
    label: str
    thumbnail: str | None


def build_detections(
    frame_bgr: np.ndarray | None,
    boxes: tuple[DetectionBox, ...],
    *,
    size: int = _THUMBNAIL_SIZE,
    pad: float = _BBOX_PAD,
) -> list[Detection]:
    """Turn a capture snapshot's `(tracker_id, xyxy)` boxes into API detections,
    sorted by `tracker_id` for stable tile ordering across polls. Each crop is
    isolated: a failure yields `thumbnail=None`, never an exception."""
    return [
        Detection(
            tracker_id=int(tracker_id),
            label=f"Instrument {int(tracker_id)}",
            thumbnail=crop_thumbnail(frame_bgr, bbox, size=size, pad=pad),
        )
        for tracker_id, bbox in sorted(boxes, key=lambda item: item[0])
    ]


def crop_thumbnail(
    frame_bgr: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    *,
    size: int = _THUMBNAIL_SIZE,
    pad: float = _BBOX_PAD,
) -> str | None:
    """Crop `bbox` out of `frame_bgr`, letterbox it into a `size`×`size` canvas,
    and return a `data:image/jpeg;base64,…` URI — or `None` if the box is
    unusable or encoding fails. Defensive by contract: every failure path
    returns `None` (logged), so one bad tile can't 500 the endpoint."""
    try:
        return _crop_thumbnail(frame_bgr, bbox, size, pad)
    except Exception:
        logger.exception("thumbnail crop failed for bbox %r", bbox)
        return None


def _crop_thumbnail(
    frame_bgr: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    size: int,
    pad: float,
) -> str | None:
    if frame_bgr is None or frame_bgr.ndim != 3:
        return None
    height, width = frame_bgr.shape[:2]
    if height == 0 or width == 0:
        return None

    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < _MIN_BOX_PX or y2 - y1 < _MIN_BOX_PX:
        return None  # collapsed/degenerate box — no meaningful crop to show
    if x2 <= 0 or x1 >= width or y2 <= 0 or y1 >= height:
        # The box lies entirely outside the frame. Test this on the RAW box,
        # before padding — otherwise the 15 % pad + clamp can pull in a strip of
        # unrelated edge pixels and encode a garbage tile instead of null.
        return None

    pad_x = (x2 - x1) * pad
    pad_y = (y2 - y1) * pad
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None  # the (padded, clamped) box fell entirely off the frame

    crop = frame_bgr[top:bottom, left:right]
    canvas = _letterbox(crop, size)
    ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _letterbox(crop: np.ndarray, size: int) -> np.ndarray:
    """Resize `crop` to fit inside `size`×`size` preserving aspect ratio, padded
    (letterboxed) onto a black square — long instruments keep their proportions
    instead of being squashed into a square."""
    crop_h, crop_w = crop.shape[:2]
    scale = min(size / crop_w, size / crop_h)
    new_w = max(1, min(size, int(round(crop_w * scale))))
    new_h = max(1, min(size, int(round(crop_h * scale))))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    off_x = (size - new_w) // 2
    off_y = (size - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas
