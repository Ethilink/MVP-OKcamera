"""T03: overlay renderer — draws boxes + masks coloured by `tracker_id`,
label 'Instrument {tracker_id}' (DESIGN D5/D6). See
`app/docs/tasks/T03-capture.md` for the frozen signature and ACs.

The on-table count is intentionally NOT burned into the frame: the operator
reads it from the app chrome (the setup constellation / recording list), so the
video stays a clean feed with just the coloured detection boxes.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

_TRACK = sv.ColorLookup.TRACK
_MASK_ANNOTATOR = sv.MaskAnnotator(color_lookup=_TRACK)
_BOX_ANNOTATOR = sv.BoxAnnotator(color_lookup=_TRACK)
_LABEL_ANNOTATOR = sv.LabelAnnotator(color_lookup=_TRACK)


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
