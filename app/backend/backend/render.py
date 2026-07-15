"""Draw boxes and masks coloured by ``tracker_id`` (DESIGN D5/D6).

The on-table count is intentionally NOT burned into the frame: the operator
reads it from the app chrome (the setup constellation / recording list), so the
video stays a clean feed with just the coloured detection boxes.

While RECORDING the overlay is roster-aware (T10/D8a): a roster instrument keeps
a fixed palette colour derived from its id — so a returned instrument regains
its colour with no renderer memory — and anything NOT in the roster draws gray.
Whether such a track shows the resolving spinner or the settled "Unknown" label
is read from the linker's per-detection ``data["resolving"]`` flag (the seam's
wait-state, grilled 2026-07-16): the renderer keeps NO first-seen clock of its
own, so the spinner follows the linker's actual decision instead of a timer that
could clear before a deferred track is really settled. It is never "Instrument N".
"""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

_TRACK = sv.ColorLookup.TRACK
_MASK_ANNOTATOR = sv.MaskAnnotator(color_lookup=_TRACK)
_BOX_ANNOTATOR = sv.BoxAnnotator(color_lookup=_TRACK)
_LABEL_ANNOTATOR = sv.LabelAnnotator(color_lookup=_TRACK)

_INDEX = sv.ColorLookup.INDEX

# 8 distinct hues, one per roster slot. TUNABLE: the hexes themselves carry no
# meaning beyond being mutually distinct, stable, and distinct from the gray.
ROSTER_PALETTE: tuple[str, ...] = (
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


def render(frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
    """Draw the overlay IN PLACE on `frame` and return it: boxes + masks
    coloured by tracker_id, label 'Instrument {tracker_id}'. Copy ownership:
    the CAPTURE LOOP passes a frame it owns (a copy of the camera buffer if that
    buffer may be reused), so render is free to mutate. render does NOT copy —
    single owner, no double-copy."""
    if dets.tracker_id is not None and len(dets) > 0:
        labels = [f"Instrument {int(tracker_id)}" for tracker_id in dets.tracker_id]
        _MASK_ANNOTATOR.annotate(scene=frame, detections=dets)
        _BOX_ANNOTATOR.annotate(scene=frame, detections=dets)
        _LABEL_ANNOTATOR.annotate(scene=frame, detections=dets, labels=labels)

    return frame


def roster_colour(roster: frozenset[int], session_id: int) -> str:
    """Fixed hex colour for a roster id: palette[index of id in sorted(roster),
    mod 8]. Pure; stable for the whole recording because the roster is frozen —
    which is what lets the panel swatch and the mask agree without either side
    remembering anything. An id outside the roster is not an instrument and gets
    the gray (the same answer as an empty/unknown roster)."""
    if session_id not in roster:
        return UNKNOWN_COLOUR
    return ROSTER_PALETTE[sorted(roster).index(session_id) % len(ROSTER_PALETTE)]


class OverlayRenderer:
    """Stateful only in its recording flag (capture thread reads it; the HTTP
    start/stop handlers flip it via `set_recording`). Replaces the plain
    `render` function as CaptureLoop's default render_fn.

    Whether a not-in-roster track spins (resolving) or draws the settled
    "Unknown" label is read from the linker's per-detection `data["resolving"]`
    flag, NOT from a renderer-side timer. The renderer therefore holds no
    first-seen map: it cannot latch, cannot grow, and cannot disagree with the
    linker about when a track is settled (the wait-state flicker a local timer
    had — grilled 2026-07-16). A producer that omits the flag (a bare test
    double) has every not-in-roster track read as settled Unknown."""

    def __init__(self) -> None:
        self._recording = False

    def set_recording(self, recording: bool) -> None:
        """Thread-safe flag flip; called by the start/stop HTTP handlers. No
        per-id state to reset -- resolving is the linker's per-frame flag, and a
        new recording gets a freshly-reset linker (its own `_pending`)."""
        self._recording = recording

    def __call__(
        self, frame: np.ndarray, dets: sv.Detections, roster: frozenset[int], t: float
    ) -> np.ndarray:
        if dets.tracker_id is None or len(dets) == 0:
            return frame
        if not self._recording:
            # Setup / finished: the roster is stale by design, so draw exactly
            # what the operator saw before T10.
            return render(frame, dets)

        resolving = dets.data.get("resolving") if dets.data is not None else None
        colours: list[str] = []
        labels: list[str | None] = []
        for row, tracker_id in enumerate(int(i) for i in dets.tracker_id):
            if tracker_id in roster:
                colours.append(roster_colour(roster, tracker_id))
                labels.append(f"Instrument {tracker_id}")
                continue
            colours.append(UNKNOWN_COLOUR)
            is_resolving = bool(resolving[row]) if resolving is not None else False
            labels.append(None if is_resolving else _UNKNOWN_LABEL)

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
