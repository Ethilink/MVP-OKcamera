"""Draw boxes and masks under ONE identity policy in every phase (T11/R1).

The on-table count is intentionally NOT burned into the frame: the operator
reads it from the app chrome (the setup constellation / recording list), so the
video stays a clean feed with just the coloured detection boxes.

Since T11/R1 there is no separate setup appearance. In setup, recording, and the
run-2 setup view the overlay is roster-aware: a roster instrument keeps a fixed
palette colour derived from its id AND the fixed catalog (`catalog_colour`, D5) —
so a returned instrument regains its colour with no renderer memory, and a
partial roster can never shift another instrument's colour — while anything NOT in
the roster draws gray. Whether such a track shows the resolving spinner or the
settled "Unknown" label is read from the linker's per-detection
``data["resolving"]`` flag (the seam's wait-state, grilled 2026-07-16): the
renderer keeps NO first-seen clock of its own, so the spinner follows the linker's
actual decision instead of a timer that could clear before a deferred track is
really settled. A not-in-roster track is never "Instrument N".
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

_INDEX = sv.ColorLookup.INDEX

# 8 distinct hues, one per CATALOG slot (indexed by position in sorted(catalog),
# D5). TUNABLE: the hexes themselves carry no meaning beyond being mutually
# distinct, stable, and distinct from the gray.
CATALOG_PALETTE: tuple[str, ...] = (
    "#4285f4",  # blue
    "#34a853",  # green
    "#fbbc04",  # yellow
    "#ea4335",  # red
    "#a142f4",  # purple
    "#24c1e0",  # cyan
    "#ff6d01",  # orange
    "#f538a0",  # pink
)

# Not in the roster => not an instrument => no hue of its own.
UNKNOWN_COLOUR = "#9ca3af"

_UNKNOWN_LABEL = "Unknown"
# The pending indicator sweeps 300 deg/s: over any gap shorter than the pending
# window it lands on a different angle, so it visibly turns rather than sitting.
_SPINNER_DEGREES_PER_S = 300.0
_SPINNER_ARC_DEGREES = 270


def catalog_colour(catalog: frozenset[int], specimen_id: int) -> str:
    """Fixed hex colour for a catalog id: palette[index of id in sorted(catalog),
    mod 8] (D5). Pure; stable for the tracker's whole life because the catalog is
    constant — which is what lets the panel swatch and the mask agree without
    either side remembering anything, AND keeps a partial roster from shifting an
    already-known instrument's colour (the roster is NOT the palette basis). An id
    outside the catalog is not an instrument and gets the gray."""
    if specimen_id not in catalog:
        return UNKNOWN_COLOUR
    return CATALOG_PALETTE[sorted(catalog).index(specimen_id) % len(CATALOG_PALETTE)]


def classify_detection(
    tracker_id: int, roster: frozenset[int], catalog: frozenset[int], resolving: bool
) -> tuple[str, str, str]:
    """The single identity policy (D4/D5), returning ``(state, label, colour)``.

    Shared by the overlay renderer AND the setup thumbnail builder so the video
    mask and the setup tile for the same detection can never disagree (R1/R3).

    - recognised (id in the same-tick roster): ``Instrument {id}`` + its catalog
      colour;
    - recognising (not in roster, still resolving): NO name (empty label) + gray —
      the caller shows a spinner, never the raw id;
    - unknown (not in roster, settled): ``Unknown`` + gray.

    A raw tracker id is never exposed as a label (D4)."""
    if tracker_id in roster:
        return "recognised", f"Instrument {tracker_id}", catalog_colour(catalog, tracker_id)
    if resolving:
        return "recognising", "", UNKNOWN_COLOUR
    return "unknown", _UNKNOWN_LABEL, UNKNOWN_COLOUR


class OverlayRenderer:
    """Stateless overlay renderer — CaptureLoop's default render_fn (replaces the
    old `render` free function). Since T11/R1 the identity policy is phase-
    independent — roster-aware in setup, recording, and finished alike (D4) — so
    the setup frame before Track and the first recording frame after Track are
    visually identical for unchanged detections; there is no recording flag.

    Whether a not-in-roster track spins (resolving) or draws the settled "Unknown"
    label is read from the linker's per-detection `data["resolving"]` flag, NOT
    from a renderer-side timer. The renderer therefore holds no first-seen map: it
    cannot latch, cannot grow, and cannot disagree with the linker about when a
    track is settled (the wait-state flicker a local timer had — grilled
    2026-07-16). A producer that omits the flag (a bare test double) has every
    not-in-roster track read as settled Unknown."""

    def __call__(
        self,
        frame: np.ndarray,
        dets: sv.Detections,
        roster: frozenset[int],
        catalog: frozenset[int],
        t: float,
    ) -> np.ndarray:
        if dets.tracker_id is None or len(dets) == 0:
            return frame

        resolving = dets.data.get("resolving") if dets.data is not None else None
        colours: list[str] = []
        labels: list[str | None] = []
        for row, tracker_id in enumerate(int(i) for i in dets.tracker_id):
            is_resolving = bool(resolving[row]) if resolving is not None else False
            state, label, colour = classify_detection(tracker_id, roster, catalog, is_resolving)
            colours.append(colour)
            # recognising -> spinner, no label; recognised/unknown -> draw label.
            labels.append(None if state == "recognising" else label)

        palette = _palette(colours)
        sv.MaskAnnotator(color=palette, color_lookup=_INDEX).annotate(scene=frame, detections=dets)
        sv.BoxAnnotator(color=palette, color_lookup=_INDEX).annotate(scene=frame, detections=dets)

        labelled = [row for row, label in enumerate(labels) if label is not None]
        if labelled:
            sv.LabelAnnotator(
                color=_palette([colours[row] for row in labelled]), color_lookup=_INDEX
            ).annotate(
                scene=frame,
                detections=dets[labelled],
                labels=[labels[row] for row in labelled],
            )
        for row, label in enumerate(labels):
            if label is None:
                _draw_spinner(frame, dets.xyxy[row], t)
        return frame


def _palette(colours: list[str]) -> sv.ColorPalette:
    """A palette whose slot i IS row i's colour — paired with ColorLookup.INDEX
    this colours each detection individually."""
    return sv.ColorPalette(colors=[sv.Color.from_hex(colour) for colour in colours])


def _draw_spinner(frame: np.ndarray, box: np.ndarray, t: float) -> None:
    """A turning arc in the box's top-right corner: this track is still being
    resolved, not a 9th instrument."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    radius = max(3, min(14, (x2 - x1) // 6, (y2 - y1) // 6))
    centre = (x2 - radius - 4, y1 + radius + 4)
    start = (t * _SPINNER_DEGREES_PER_S) % 360.0
    cv2.ellipse(
        frame,
        centre,
        (radius, radius),
        0.0,
        start,
        start + _SPINNER_ARC_DEGREES,
        sv.Color.from_hex(UNKNOWN_COLOUR).as_bgr(),
        thickness=2,
    )
