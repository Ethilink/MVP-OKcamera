"""Consumer-side detection cutouts for `GET /status` previews.

The model returns frame-aligned boxes and masks, not encoded UI assets. The app
backend owns the camera frame, so it derives small transparent PNG cutouts here.
A rectangular JPEG crop is the defensive fallback for a producer without a
usable mask. This stays OpenCV-only and FastAPI-free so geometry and encoding
are unit-testable and never run under the capture/session lock.

See ``app/docs/api-contract.md`` and ``model/docs/tracker-interface.md``.
"""

from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from backend.capture import DetectionBox
from backend.render import classify_detection

logger = logging.getLogger(__name__)

_THUMBNAIL_SIZE = 96      # px — the ~96×96 letterboxed canvas per tile
_BBOX_PAD = 0.15          # expand each box ~15 % so instrument ends aren't clipped
_JPEG_QUALITY = 80
_PNG_COMPRESSION = 3
_MIN_BOX_PX = 2.0         # boxes narrower/shorter than this are degenerate → skip


@dataclass(frozen=True)
class Detection:
    """One setup detection as the API exposes it: a stable id, its identity
    `state`/`label`/`colour` (from the single policy in `render.classify_detection`
    so the tile and the video mask can never disagree — R1/R3), and a data-URI
    thumbnail (or `None` when that preview could not be produced — a bad crop
    must never fail the whole `/status` response)."""

    tracker_id: int
    state: str
    label: str
    colour: str
    thumbnail: str | None
    # Experimental (feat/matching-tests) — testing aid, not part of the frozen
    # api-contract. detector_confidence: this frame's raw detector score, or
    # `None` when the tracker didn't report one (e.g. fake mode). The rest come
    # from the tracker's `match_debug` (blank in fake mode): the matcher's last
    # score for this id, whichever event produced it (bind at Start or a later
    # re-id decision), the threshold it was gated against, the closest-scoring
    # candidate whether or not it was accepted, and whether that last score won.
    detector_confidence: float | None = None
    matcher_score: float | None = None
    matcher_tau: float | None = None
    matcher_closest_id: int | None = None
    matcher_accepted: bool | None = None


def build_detections(
    frame_bgr: np.ndarray | None,
    boxes: tuple[DetectionBox, ...],
    roster: frozenset[int],
    catalog: frozenset[int],
    *,
    size: int = _THUMBNAIL_SIZE,
    pad: float = _BBOX_PAD,
    match_debug: dict | None = None,
) -> list[Detection]:
    """Turn a capture snapshot's `DetectionBox` boxes into API detections,
    sorted by `tracker_id` for stable tile ordering across polls. Each tile's
    `state`/`label`/`colour` comes from `render.classify_detection` against the
    SAME-tick `roster`/`catalog` and the box's `resolving` flag (R3), so a
    recognised tile shows its catalog colour + `Instrument N`, a resolving tile is
    gray with no name, and a settled non-roster tile is a gray `Unknown` — exactly
    what the overlay draws for that detection. Each preview is isolated: a failure
    yields `thumbnail=None`, never an exception.

    `match_debug` (experimental, feat/matching-tests) is the tracker's last
    matcher score per emitted id, sampled the same tick; looked up here by
    `tracker_id` and left `None` for any id the matcher hasn't scored yet."""
    debug = match_debug or {}
    detections: list[Detection] = []
    for box in sorted(boxes, key=lambda item: item.tracker_id):
        state, label, colour = classify_detection(box.tracker_id, roster, catalog, box.resolving)
        entry = debug.get(box.tracker_id)
        detections.append(
            Detection(
                tracker_id=box.tracker_id,
                state=state,
                label=label,
                colour=colour,
                thumbnail=crop_thumbnail(
                    frame_bgr,
                    box.xyxy,
                    mask=box.mask,
                    size=size,
                    pad=pad,
                ),
                detector_confidence=box.confidence,
                matcher_score=entry.score if entry is not None else None,
                matcher_tau=entry.tau if entry is not None else None,
                matcher_closest_id=entry.closest_id if entry is not None else None,
                matcher_accepted=entry.accepted if entry is not None else None,
            )
        )
    return detections


def crop_thumbnail(
    frame_bgr: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    *,
    mask: np.ndarray | None = None,
    size: int = _THUMBNAIL_SIZE,
    pad: float = _BBOX_PAD,
) -> str | None:
    """Return a transparent PNG cutout, or a JPEG crop when no mask is usable."""
    try:
        return _crop_thumbnail(frame_bgr, bbox, mask, size, pad)
    except Exception:
        logger.exception("thumbnail crop failed for bbox %r", bbox)
        return None


def _crop_thumbnail(
    frame_bgr: np.ndarray | None,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray | None,
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

    raw_left = max(0, int(math.floor(x1)))
    raw_top = max(0, int(math.floor(y1)))
    raw_right = min(width, int(math.ceil(x2)))
    raw_bottom = min(height, int(math.ceil(y2)))

    pad_x = (x2 - x1) * pad
    pad_y = (y2 - y1) * pad
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None  # the (padded, clamped) box fell entirely off the frame

    crop = frame_bgr[top:bottom, left:right]
    cutout = _cutout_thumbnail(
        crop,
        mask,
        raw_bounds=(raw_left, raw_top, raw_right, raw_bottom),
        padded_origin=(left, top),
        size=size,
    )
    if cutout is not None:
        ok, buffer = cv2.imencode(
            ".png", cutout, [int(cv2.IMWRITE_PNG_COMPRESSION), _PNG_COMPRESSION]
        )
        if ok:
            return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")

    canvas = _letterbox(crop, size)
    ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _cutout_thumbnail(
    crop: np.ndarray,
    mask: np.ndarray | None,
    *,
    raw_bounds: tuple[int, int, int, int],
    padded_origin: tuple[int, int],
    size: int,
) -> np.ndarray | None:
    """Compose a bbox-local mask into the padded crop and letterbox with alpha."""
    if mask is None or mask.ndim != 2 or not np.any(mask):
        return None
    raw_left, raw_top, raw_right, raw_bottom = raw_bounds
    expected_shape = (raw_bottom - raw_top, raw_right - raw_left)
    if mask.shape != expected_shape:
        return None

    alpha = np.zeros(crop.shape[:2], dtype=np.uint8)
    left, top = padded_origin
    offset_x = raw_left - left
    offset_y = raw_top - top
    alpha[
        offset_y : offset_y + mask.shape[0],
        offset_x : offset_x + mask.shape[1],
    ] = mask.astype(np.uint8) * 255
    return _letterbox_cutout(crop, alpha, size)


def _letterbox_cutout(crop: np.ndarray, alpha: np.ndarray, size: int) -> np.ndarray:
    """Resize premultiplied colour + alpha, avoiding dark fringes on cutouts."""
    crop_h, crop_w = crop.shape[:2]
    scale = min(size / crop_w, size / crop_h)
    new_size = (
        max(1, min(size, int(round(crop_w * scale)))),
        max(1, min(size, int(round(crop_h * scale)))),
    )
    opacity = alpha.astype(np.float32) / 255.0
    premultiplied = crop.astype(np.float32) * opacity[:, :, None]
    resized_alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_AREA)
    resized_colour = cv2.resize(premultiplied, new_size, interpolation=cv2.INTER_AREA)
    divisor = resized_alpha.astype(np.float32)[:, :, None] / 255.0
    colour = np.divide(
        resized_colour,
        divisor,
        out=np.zeros_like(resized_colour),
        where=divisor > 0,
    )
    tile = np.dstack((np.clip(colour, 0, 255).astype(np.uint8), resized_alpha))
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    off_x = (size - new_size[0]) // 2
    off_y = (size - new_size[1]) // 2
    canvas[off_y : off_y + new_size[1], off_x : off_x + new_size[0]] = tile
    return canvas


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
