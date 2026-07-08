# T04 — Overlay renderer

status: done (claude)
depends-on: T01
blocks: T05
spec: [DASHBOARD.md](../DASHBOARD.md) §Scope (Watch), §Runtime ("Why draw masks live")

## Goal

Draw the detector's boxes **and masks** on a 1080p frame, coloured by
instance index, so the operator can judge mask quality live. Small, pure, fast.

## In scope

`backend/render.py` + `tests/test_render.py`. Supervision's annotators
(`sv.BoxAnnotator` / `sv.MaskAnnotator`) are fine — don't hand-roll drawing.

## Out of scope

RLE encoding (flag-time only, T02), JPEG encoding (T03), any text/HUD beyond
per-instance confidence labels (the live count is shown by the UI via `/status`,
not burned into pixels).

## Public interface (frozen)

```python
# backend/render.py
def render(frame: np.ndarray, dets: "sv.Detections") -> np.ndarray:
    """Draws IN PLACE on frame (caller passes a copy — T03 does) and returns it."""
```

## Acceptance criteria

- [x] **AC1** Returns the same array object it was given (in-place contract).
- [x] **AC2** With 0 detections (incl. `mask is None`) the frame is returned
  unchanged, no exception.
- [x] **AC3** Two detections get visibly distinct colors (sample pixels inside each
  mask region differ from each other and from the background).
- [x] **AC4** Both box edges and mask fill are drawn (pixels on the box border and
  inside the mask differ from the input frame).
- [x] **AC5** A detection with a box but an all-`False` mask still renders its box
  (box/mask thresholded independently — spec).
- [x] **AC6** Fast enough to be loop-invisible: p50 < 15 ms on a 1920×1080 frame
  with 5 masks (rough guard, not a benchmark).

## Log

- 2026-07-07 (claude) Shipped `backend/render.py` + `tests/test_render.py`;
  6/6 ACs green. Boxes + confidence labels go through `sv.BoxAnnotator` /
  `sv.LabelAnnotator` with `ColorLookup.INDEX` (single-class detector → INDEX,
  not the default CLASS lookup, so instances get distinct colours — AC3).
  **Deviation from "don't hand-roll drawing":** `sv.MaskAnnotator` composites
  over the whole frame (full copy + per-mask full-frame boolean write) and
  measured ~30 ms on a 1080p/5-mask frame here — 2× AC6's budget. Replaced only
  the mask fill with a bbox-cropped cv2 blend (~4 ms), reusing `sv`'s palette by
  index so a mask matches its box colour. Boxes/labels unchanged. Flagged for
  Bram in case the "use sv annotators" note was a hard constraint.
