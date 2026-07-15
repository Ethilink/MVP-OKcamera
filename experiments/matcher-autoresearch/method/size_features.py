#!/usr/bin/env python
"""
MUTABLE — mask-size features for the champion's size-fusion cue (program.md §6
"multi-cue fusion", stage-1 `size_probe.py`/`fuse.py`). Demo facts that license
this (program.md §2): fixed overhead camera, instruments flat on the table, so
pixel mask-size is a valid physical-size proxy.

`area` = mask pixel count. `length` = PCA major-axis extent (orientation-
invariant physical length), same method as stage-1's `size_probe.py`.

Normalization constants below are FIXED, derived once (not per-run, not from
query data) from the CV/selection pool only (frozen/holdout.py's
`instrument_cv_selection_pool` — order 0..11, i.e. never the locked holdout
QUERY frames order>=12) via `runs/baseline/derive_size_stats.py`. Freezing them
avoids two problems: (a) a z-score computed live from a single gallery call
can't see other identities' scale range, which is exactly the signal
size-fusion needs; (b) recomputing from "everything visible so far" would risk
quietly drifting between CV and holdout runs. Provenance: see
runs/baseline/derive_size_stats.log.
"""
from __future__ import annotations

import numpy as np

# log(area), log(length) mean/std over the CV/selection pool (8 instruments x
# 12 frames [order 0..11] = 96 crops). See runs/baseline/derive_size_stats.log.
LOG_AREA_MEAN = 8.7201
LOG_AREA_STD = 0.4651
LOG_LENGTH_MEAN = 5.5297
LOG_LENGTH_STD = 0.4733


def mask_extent(mask: np.ndarray) -> tuple[float, float]:
    """(area_px, major_axis_length_px) via PCA on mask foreground pixels."""
    ys, xs = np.nonzero(mask)
    area = float(len(xs))
    if area < 2:
        return area, 0.0
    pts = np.stack([xs, ys], 1).astype(float)
    pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt.T
    length = float(proj[:, 0].max() - proj[:, 0].min())
    return area, length


def size_feature(mask: np.ndarray) -> np.ndarray:
    """z-scored [log(area), log(length)] using the FIXED constants above.
    log1p-guarded against degenerate (empty) masks."""
    area, length = mask_extent(mask)
    log_area = np.log(max(area, 1.0))
    log_length = np.log(max(length, 1.0))
    return np.array([
        (log_area - LOG_AREA_MEAN) / LOG_AREA_STD,
        (log_length - LOG_LENGTH_MEAN) / LOG_LENGTH_STD,
    ], dtype=np.float32)
