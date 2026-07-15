"""Shared probes for the T10 overlay tests (`test_render.py`, `test_api.py`).

The overlay may be drawn with supervision's annotators OR with hand-rolled cv2
calls — T10-BACKEND-SPEC's "Dependencies / reuse" sanctions both — so these
probes read the RESULT (pixels, and the text that reached a text-drawing call)
rather than any one drawing API.

**No probe here pins a tunable.** The spec lists the 8 palette hex values and
the gray placeholder hex as tunables, so everything below is phrased as
*distinctness*, *stability*, or *gray-vs-coloured*. (The resolving vs settled
split is no longer a renderer timer but the linker's per-detection
`data["resolving"]` flag — `boxed_detections` sets it so a test can drive either
state directly.)

- `chroma_ratio` answers "is this pixel region GRAY (achromatic) or a
  saturated instrument colour?" without naming either colour. Measured
  separation is wide: today's overlay colours for the ids used in these tests
  score 0.81-1.00, the spec's gray scores 0.11, and the ratio is invariant to
  the mask's (unknown) blend opacity.
- `colour_similarity` compares a drawn region to an expected hex by DIRECTION,
  so it survives any mask opacity — a blend of `alpha * C` over a black frame
  keeps C's direction exactly. A degenerate (zero/black) expected colour scores
  -1.0 rather than raising, so a stubbed-out palette fails an assertion instead
  of erroring.
"""

from __future__ import annotations

import contextlib

import cv2
import numpy as np
import supervision as sv

# A region whose channels sit within this fraction of its brightest channel
# reads as gray. Comfortably between the spec's gray (0.11) and the overlay's
# instrument colours (>= 0.37 for every entry of today's palette).
GRAY_CHROMA_MAX = 0.25

# Cosine agreement required before a drawn region counts as "this hex".
COLOUR_SIMILARITY_MIN = 0.98


def blank_frame(height: int = 200, width: int = 260) -> np.ndarray:
    """A black frame. Black matters: a mask blended over it keeps the mask
    colour's direction whatever the opacity, which is what `colour_similarity`
    reads."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def boxed_detections(
    boxes: list[tuple[int, int, int, int]],
    tracker_ids: list[int],
    frame_shape: tuple[int, int],
    resolving: list[bool] | None = None,
) -> sv.Detections:
    """A minimal, valid `sv.Detections` with a full-frame boolean mask per box
    (True only inside that box) — exercises both the box and the mask drawing
    paths without depending on any tracker.

    `resolving` (one bool per box) sets the seam's `data["resolving"]` flag the
    overlay reads to tell a resolving track (spinner) from a settled one
    ("Unknown"). Omit it to leave the key off entirely — the state a bare
    producer emits, which the renderer treats as settled."""
    height, width = frame_shape
    n = len(boxes)
    mask = np.zeros((n, height, width), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        mask[i, y1:y2, x1:x2] = True
    data = {} if resolving is None else {"resolving": np.array(resolving, dtype=bool)}
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        mask=mask,
        confidence=np.full(n, 0.9, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
        data=data,
    )


def box_mean_bgr(
    frame: np.ndarray, box: tuple[int, int, int, int], inset: int = 10
) -> np.ndarray:
    """Mean BGR of a box's INTERIOR. The inset skips the box outline (drawn on
    the edge) and any label (drawn above the box), so the sample is the mask
    fill — the thing whose colour the spec talks about."""
    x1, y1, x2, y2 = box
    region = frame[y1 + inset : y2 - inset, x1 + inset : x2 - inset]
    return region.reshape(-1, 3).astype(np.float64).mean(axis=0)


def chroma_ratio(bgr: np.ndarray) -> float:
    """How far a colour is from gray, as a fraction of its brightest channel:
    0.0 is a perfect gray, 1.0 is fully saturated. NaN for pure black (nothing
    drawn) so a blank region can never be mistaken for a gray one — assert
    `mean.max() > 0` first for a readable failure."""
    brightest = float(bgr.max())
    if brightest <= 0.0:
        return float("nan")
    return (brightest - float(bgr.min())) / brightest


def hex_to_bgr(colour: str) -> np.ndarray:
    """'#4285f4' -> array([244., 133., 66.]) (BGR, the order cv2 frames use)."""
    text = colour.lstrip("#")
    r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return np.array([b, g, r], dtype=np.float64)


def colour_similarity(drawn_bgr: np.ndarray, expected_hex: str) -> float:
    """Cosine agreement in BGR space between a drawn region and a hex colour —
    1.0 when the region is that colour at ANY blend opacity. Returns -1.0 (no
    agreement) when either side is degenerate/black, so a stubbed palette makes
    callers fail an assertion rather than raise ZeroDivisionError."""
    drawn = np.asarray(drawn_bgr, dtype=np.float64)
    expected = hex_to_bgr(expected_hex)
    drawn_norm = float(np.linalg.norm(drawn))
    expected_norm = float(np.linalg.norm(expected))
    if drawn_norm == 0.0 or expected_norm == 0.0:
        return -1.0
    return float(drawn @ expected / (drawn_norm * expected_norm))


@contextlib.contextmanager
def label_spy():
    """Collect every string the overlay draws as text, through EITHER sanctioned
    path: supervision's `LabelAnnotator.annotate(labels=[...])` or a direct
    `cv2.putText`. (supervision's own annotator funnels into cv2.putText, so a
    label can be collected twice — every assertion built on this asks about
    membership, never counts.)

    Yields the list, which fills as the overlay is drawn.
    """
    texts: list[str] = []

    real_annotate = sv.LabelAnnotator.annotate

    def spy_annotate(self, scene, detections, labels=None, *args, **kwargs):
        if labels is not None:
            texts.extend(str(label) for label in labels)
        return real_annotate(self, scene, detections, labels, *args, **kwargs)

    real_put_text = cv2.putText

    def spy_put_text(img, text, *args, **kwargs):
        texts.append(str(text))
        return real_put_text(img, text, *args, **kwargs)

    sv.LabelAnnotator.annotate = spy_annotate
    cv2.putText = spy_put_text
    try:
        yield texts
    finally:
        sv.LabelAnnotator.annotate = real_annotate
        cv2.putText = real_put_text


def drew_any_instrument_label(texts: list[str]) -> bool:
    return any("Instrument" in text for text in texts)
