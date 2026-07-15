"""Typed port of GTA-Link's `Tracklet` (github.com/sjc042/gta-link, MIT).

Deviations from upstream:
  - Dataclass with type-annotated fields, instead of a plain `__init__` that
    duck-types each argument as "a list, or a single scalar to be wrapped in
    a list, or None".  Callers here always build complete tracklets from
    tracker output, so that flexibility isn't needed.
  - Bounding boxes are `xyxy` (`x1, y1, x2, y2`), matching this project's
    tracker convention (`sv.Detections.xyxy`), not upstream's MOT-style
    `xywh` top-left box. Every geometry-touching helper in `refine.py` (the
    spatial-gate functions) computes centers from `xyxy` accordingly.
  - Drops `append_det`/`append_feat`, upstream's incremental per-frame
    accumulation helpers used by `generate_tracklets.py` (its OSNet feature
    extraction pass, out of scope here -- see module docstring in
    `__init__.py`). Build a `Tracklet` from complete per-detection lists
    instead.
  - `parent_id` is repurposed slightly: upstream sets it once at
    construction and never uses it to link a fragment back to the tracklet
    it was split from. Here, `refine.split_tracklets` explicitly sets a
    fragment's `parent_id` to its pre-split origin id, so `refine_tracklets`
    can compose an id-mapping across split + merge. See `refine.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tracklet:
    """One track's detections across frames, as parallel per-detection lists.

    `times[i]`, `scores[i]`, `bboxes[i]`, and (if present) `features[i]` all
    describe the same detection; `bboxes[i]` is `(x1, y1, x2, y2)`.
    """

    track_id: int
    times: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    bboxes: list[np.ndarray] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)
    parent_id: int | None = None

    def __post_init__(self) -> None:
        if self.parent_id is None:
            self.parent_id = self.track_id
        if not (len(self.times) == len(self.scores) == len(self.bboxes)):
            raise ValueError("times/scores/bboxes must have the same length")
        if self.features and len(self.features) != len(self.times):
            raise ValueError("features, if provided, must match times length")

    def __len__(self) -> int:
        return len(self.times)

    def extract(self, start: int, end: int) -> Tracklet:
        """Return a new `Tracklet` covering indices `[start, end]` inclusive.

        Ported from upstream `Tracklet.extract`; used by the spatial-gate
        helpers (`refine.check_spatial_constraints`) to pull out
        consecutive-frame segments. Keeps `track_id`/`parent_id` unchanged
        (this is a view into the same track, not a new one).
        """
        sl = slice(start, end + 1)
        return Tracklet(
            track_id=self.track_id,
            times=self.times[sl],
            scores=self.scores[sl],
            bboxes=self.bboxes[sl],
            features=self.features[sl] if self.features else [],
            parent_id=self.parent_id,
        )
