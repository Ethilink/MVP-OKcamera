#!/usr/bin/env python
"""
Mask-size/shape features for the champion's size-fusion cue. Demo facts that
license this (fixed overhead camera, instruments flat on the table —
linker-design.md): pixel mask-size is a valid physical-size proxy.

`area` = mask pixel count. `length` = PCA major-axis extent (orientation-
invariant physical length). These are the only two features currently fused
into the matcher (see ACTIVE_FEATURES below).

Every feature below carries a real (mean, std), derived the same way as the
original log_area/log_length: fit ONLY over the development/CV pool, never
the locked holdout -- 96 crops, 8 instruments x order-0..11 frames from
`model/data/instruments/`, the exact population
`frozen.holdout.instrument_cv_selection_pool()` defines in
`model/matcher-autoresearch/`. Skewed raw distributions (aspect_ratio,
circularity, extent, width_profile) are log-transformed before fitting, same
rationale as log_area/log_length; eccentricity/solidity/hu_moments are fit
directly (already roughly symmetric, or -- hu_moments -- already log-scaled
internally). Re-derive all of these with
`model/matcher-autoresearch/calibrate_shape_features.py` if the reference
instrument set or camera geometry changes materially -- the `_UNCALIBRATED`
placeholder below is now unused but kept for any future feature added here.

To try a different feature mix: edit ACTIVE_FEATURES. Nothing else needs to
change. Note that changing the fused vector's composition/dimensionality
means size_alpha and the SRC sweep (PARAMS.md / linker-design.md) should be
re-tuned against the new vector.
"""
from __future__ import annotations

import cv2
import numpy as np

LOG_AREA_MEAN = 8.7201
LOG_AREA_STD = 0.4651
LOG_LENGTH_MEAN = 5.5297
LOG_LENGTH_STD = 0.4733

# Derived by calibrate_shape_features.py over the same 96-crop CV pool.
LOG_ASPECT_RATIO_MEAN = 1.1914
LOG_ASPECT_RATIO_STD = 0.6300
ECCENTRICITY_MEAN = 0.9369
ECCENTRICITY_STD = 0.1077
SOLIDITY_MEAN = 0.5527
SOLIDITY_STD = 0.2313
LOG_CIRCULARITY_MEAN = -1.8208
LOG_CIRCULARITY_STD = 0.7031
LOG_EXTENT_MEAN = -1.6355
LOG_EXTENT_STD = 0.6266
# Per-component (7,): cv2.HuMoments order, already log-scaled inside
# hu_moments() itself, so these are fit directly (no further log here).
HU_MOMENTS_MEAN = np.array(
    [-0.0500, -0.3139, -0.7813, -1.2182, -1.0806, -1.2855, 0.6469]
)
HU_MOMENTS_STD = np.array(
    [0.2760, 0.6703, 1.1334, 1.2239, 3.0625, 1.6151, 4.7045]
)
# Per-component (7,): one per width_profile() sample bin, log-transformed.
LOG_WIDTH_PROFILE_MEAN = np.array(
    [3.7779, 3.6706, 3.4030, 3.4053, 3.3046, 3.3204, 3.5004]
)
LOG_WIDTH_PROFILE_STD = np.array(
    [0.7245, 0.7750, 0.6038, 0.4428, 0.6512, 0.7940, 0.7537]
)

# Placeholder norm for any future feature not yet calibrated against a dev
# pool — a no-op (mean 0, std 1) until real constants are derived.
_UNCALIBRATED = (0.0, 1.0)


def _pca(mask: np.ndarray) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """area, points projected onto PCA axes (descending eigenvalue order),
    singular values. proj/singular values are None when area < 2."""
    ys, xs = np.nonzero(mask)
    area = float(len(xs))
    if area < 2:
        return area, None, None
    pts = np.stack([xs, ys], 1).astype(float)
    pts -= pts.mean(0)
    _, s, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt.T
    return area, proj, s


def mask_extent(mask: np.ndarray) -> tuple[float, float]:
    """(area_px, major_axis_length_px) via PCA on mask foreground pixels."""
    area, proj, _ = _pca(mask)
    if proj is None:
        return area, 0.0
    length = float(proj[:, 0].max() - proj[:, 0].min())
    return area, length


def minor_axis_length(mask: np.ndarray) -> float:
    """PCA minor-axis extent, px — orthogonal to mask_extent's major axis."""
    _, proj, _ = _pca(mask)
    if proj is None:
        return 0.0
    return float(proj[:, 1].max() - proj[:, 1].min())


def aspect_ratio(mask: np.ndarray) -> float:
    """major_length / minor_length — elongation, scale-invariant. Separates
    long thin instruments (scissors, needle holders) from squat ones."""
    _, length = mask_extent(mask)
    minor = minor_axis_length(mask)
    return float(length / max(minor, 1.0))


def eccentricity(mask: np.ndarray) -> float:
    """sqrt(1 - (s2/s1)^2) from PCA singular values — a variance-based
    elongation measure, smoother/less outlier-sensitive than aspect_ratio."""
    _, proj, s = _pca(mask)
    if proj is None or s[0] < 1e-8:
        return 0.0
    ratio = s[1] / s[0]
    return float(np.sqrt(max(0.0, 1.0 - ratio ** 2)))


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def solidity(mask: np.ndarray) -> float:
    """mask_area / convex_hull_area — 1.0 = fully convex, lower = concave or
    hollow (e.g. open jaws, rings) — something area/length can't see."""
    c = _largest_contour(mask)
    if c is None:
        return 0.0
    hull_area = cv2.contourArea(cv2.convexHull(c))
    if hull_area < 1e-8:
        return 0.0
    return float(cv2.contourArea(c) / hull_area)


def circularity(mask: np.ndarray) -> float:
    """4*pi*area / perimeter^2 — 1.0 for a perfect circle, ->0 for thin or
    jagged shapes."""
    c = _largest_contour(mask)
    if c is None:
        return 0.0
    perimeter = cv2.arcLength(c, closed=True)
    if perimeter < 1e-8:
        return 0.0
    area = cv2.contourArea(c)
    return float(4.0 * np.pi * area / (perimeter ** 2))


def extent(mask: np.ndarray) -> float:
    """mask_area / bounding_box_area — cheaper, cruder cousin of solidity."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    bbox_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    if bbox_area < 1e-8:
        return 0.0
    return float(len(xs) / bbox_area)


def hu_moments(mask: np.ndarray) -> np.ndarray:
    """7 log-scaled Hu invariant moments — classic rotation/scale/
    translation-invariant shape descriptors, more expressive than a couple
    of scalars but less interpretable."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    hu = cv2.HuMoments(m).flatten()
    sign = np.sign(hu)
    log_hu = sign * np.log10(np.abs(hu) + 1e-30)
    return log_hu.astype(np.float64)


def width_profile(mask: np.ndarray, n_samples: int = 7) -> np.ndarray:
    """Mask width sampled at n_samples points along the PCA major axis —
    captures tapering shapes (e.g. a needle holder: wide handle, thin tip)
    that a single aspect-ratio scalar collapses away."""
    _, proj, _ = _pca(mask)
    if proj is None:
        return np.zeros(n_samples, dtype=np.float64)
    major, minor = proj[:, 0], proj[:, 1]
    lo, hi = major.min(), major.max()
    if hi - lo < 1e-8:
        return np.zeros(n_samples, dtype=np.float64)
    edges = np.linspace(lo, hi, n_samples + 1)
    widths = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        in_bin = (major >= edges[i]) & (major <= edges[i + 1])
        if in_bin.any():
            widths[i] = float(minor[in_bin].max() - minor[in_bin].min())
    return widths


def _log_area(mask: np.ndarray) -> float:
    area, _ = mask_extent(mask)
    return float(np.log(max(area, 1.0)))


def _log_length(mask: np.ndarray) -> float:
    _, length = mask_extent(mask)
    return float(np.log(max(length, 1.0)))


def _log_aspect_ratio(mask: np.ndarray) -> float:
    return float(np.log(max(aspect_ratio(mask), 1.0)))


def _log_circularity(mask: np.ndarray) -> float:
    # circularity is a fraction in (0, 1], so the pixel-scale 1.0 floor used
    # above would clamp every real value to log(1.0)=0 -- use a near-zero
    # floor instead.
    return float(np.log(max(circularity(mask), 1e-6)))


def _log_extent(mask: np.ndarray) -> float:
    return float(np.log(max(extent(mask), 1e-6)))


def _log_width_profile(mask: np.ndarray, n_samples: int = 7) -> np.ndarray:
    return np.log(np.maximum(width_profile(mask, n_samples), 1.0))


# name -> (compute_fn(mask) -> scalar | array, output_dim, (mean, std))
# `mean`/`std` broadcast against the raw output, so a scalar norm works for
# both 1-d and n-d features and a (7,) array norm works for the two
# per-component features (hu_moments, width_profile).
SHAPE_FEATURES: dict[str, tuple] = {
    "log_area": (_log_area, 1, (LOG_AREA_MEAN, LOG_AREA_STD)),
    "log_length": (_log_length, 1, (LOG_LENGTH_MEAN, LOG_LENGTH_STD)),
    # Registry key stays "aspect_ratio" (not "log_aspect_ratio") though the
    # fit is on log(aspect_ratio) -- keeps ACTIVE_FEATURES/mvp.toml entries
    # stable regardless of which transform is used internally.
    "aspect_ratio": (_log_aspect_ratio, 1, (LOG_ASPECT_RATIO_MEAN, LOG_ASPECT_RATIO_STD)),
    "eccentricity": (eccentricity, 1, (ECCENTRICITY_MEAN, ECCENTRICITY_STD)),
    "solidity": (solidity, 1, (SOLIDITY_MEAN, SOLIDITY_STD)),
    "circularity": (_log_circularity, 1, (LOG_CIRCULARITY_MEAN, LOG_CIRCULARITY_STD)),
    "extent": (_log_extent, 1, (LOG_EXTENT_MEAN, LOG_EXTENT_STD)),
    "hu_moments": (hu_moments, 7, (HU_MOMENTS_MEAN, HU_MOMENTS_STD)),
    "width_profile": (_log_width_profile, 7, (LOG_WIDTH_PROFILE_MEAN, LOG_WIDTH_PROFILE_STD)),
}

# Which SHAPE_FEATURES entries size_feature() actually fuses today, in order.
# Add/remove a name here to bring a registered feature into (or out of) the
# matcher — no other code needs to change.
ACTIVE_FEATURES: tuple[str, ...] = ("log_area", "log_length")


def size_feature(
    mask: np.ndarray, active_features: tuple[str, ...] = ACTIVE_FEATURES
) -> np.ndarray:
    """z-scored, concatenated vector of every feature named in
    `active_features`, in that order (defaults to module-level
    ACTIVE_FEATURES). Caller is expected to have already validated the names
    against SHAPE_FEATURES -- see ChampionMethod.__init__."""
    if not active_features:
        return np.zeros(0, dtype=np.float32)
    parts = []
    for name in active_features:
        fn, _dim, (mean, std) = SHAPE_FEATURES[name]
        raw = np.atleast_1d(fn(mask)).astype(np.float64)
        safe_std = np.where(np.asarray(std) != 0, std, 1.0)
        parts.append((raw - mean) / safe_std)
    return np.concatenate(parts).astype(np.float32)
