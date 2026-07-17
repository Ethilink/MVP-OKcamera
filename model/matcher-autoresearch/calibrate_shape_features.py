#!/usr/bin/env python
"""Derive (mean, std) z-score constants for shape features, same discipline as
the champion's own LOG_AREA_MEAN/STD (see TRIED.md Round 0; also reused by
r2-c2/r6-c2's feature-stat scripts): fit ONLY over the CV/selection pool
(`frozen.holdout.instrument_cv_selection_pool` -- 8 instruments x order 0..11
= 96 crops from `model/data/instruments/`, never the locked holdout, never
`other_objects`), population std (`ddof=0`).

The original `runs/baseline/derive_size_stats.py` this mirrors no longer
exists in the repo; this script reproduces LOG_AREA_MEAN/STD/LOG_LENGTH_*
bit-for-bit against today's data (verified), so its numbers for the
remaining shape features are derived the same way and are just as
reproducible.

Run from `model/`: `uv run python matcher-autoresearch/calibrate_shape_features.py`

Whether to log-transform each feature before z-scoring is decided here by
skewness: a heavy-tailed/right-skewed raw distribution gets logged (mirrors
why log_area/log_length are logged in the first place), a distribution
already roughly symmetric is left alone. Printed skew is the raw-vs-logged
comparison that motivated each choice below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from frozen.holdout import instrument_cv_selection_pool  # noqa: E402
from frozen.loader import load_instruments  # noqa: E402
from orc_model.pipelines.matching.size_features import SHAPE_FEATURES  # noqa: E402


def _skew(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0, ddof=0)
    return ((values - values.mean(axis=0)) ** 3).mean(axis=0) / np.where(std > 0, std, 1.0) ** 3


# name in SHAPE_FEATURES -> whether to np.log(max(value, floor)) before fitting.
# floor=1.0 for pixel-scale quantities (area/length/width, consistent with the
# existing log_area/log_length convention); floor=1e-6 for fraction-valued
# quantities in (0, 1] (circularity/extent), where a 1.0 floor would clamp
# every real value to log(1.0)=0.
LOG_TRANSFORM = {
    "aspect_ratio": 1.0,
    "circularity": 1e-6,
    "extent": 1e-6,
    "width_profile": 1.0,
}


def derive(name: str) -> None:
    fn, _dim, _norm = SHAPE_FEATURES[name]
    pool = instrument_cv_selection_pool(load_instruments())
    raw = np.stack([np.atleast_1d(fn(item.mask)) for item in pool]).astype(np.float64)

    floor = LOG_TRANSFORM.get(name)
    fitted = np.log(np.maximum(raw, floor)) if floor is not None else raw

    print(f"{name}  (n={len(pool)}, log={floor is not None})")
    print(f"  raw    skew={np.round(_skew(raw), 3)}")
    if floor is not None:
        print(f"  logged skew={np.round(_skew(fitted), 3)}")
    print(f"  mean = {np.round(fitted.mean(axis=0), 4).tolist()}")
    print(f"  std  = {np.round(fitted.std(axis=0, ddof=0), 4).tolist()}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(SHAPE_FEATURES)
    for name in names:
        derive(name)
        print()
