#!/usr/bin/env python
"""
Mask-size features for the champion's size-fusion cue. Demo facts that license
this (fixed overhead camera, instruments flat on the table — linker-design.md):
pixel mask-size is a valid physical-size proxy.

`area` = mask pixel count. `length` = PCA major-axis extent (orientation-
invariant physical length).

Normalization constants below are fixed values derived from the development
pool (never the locked holdout). Re-derive them if the reference instrument
set or camera geometry changes materially.
"""
from __future__ import annotations

import numpy as np

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
    """z-scored [log(area), log(length)] using the FIXED constants above."""
    area, length = mask_extent(mask)
    log_area = np.log(max(area, 1.0))
    log_length = np.log(max(length, 1.0))
    return np.array([
        (log_area - LOG_AREA_MEAN) / LOG_AREA_STD,
        (log_length - LOG_LENGTH_MEAN) / LOG_LENGTH_STD,
    ], dtype=np.float32)
