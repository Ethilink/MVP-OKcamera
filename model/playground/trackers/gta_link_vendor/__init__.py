"""GTA-Link, vendored and adapted from https://github.com/sjc042/gta-link
(MIT license).

Upstream is an offline tracklet-refinement algorithm for sports-MOT
benchmarks (SportsMOT/SoccerNet): given a finished tracker's output as
`dict[track_id, Tracklet]`, it (1) **splits** any tracklet that looks like it
silently swallowed an id switch (DBSCAN over its per-detection appearance
embeddings finds multiple clusters), then (2) **connects** tracklets that are
probably the same identity reappearing after a gap (closest-pair
agglomerative merge on mean cosine distance between embeddings, vetoed
whenever the pair overlaps in time).

This module ports the split + connect algorithm only -- not upstream's
OSNet/torchreid person-ReID feature extraction (`generate_tracklets.py`),
which doesn't apply to surgical instruments. It's deliberately
**tracker-agnostic**: the only input is `dict[int, Tracklet]`, so any
tracker notebook in this repo can use it if it can supply per-detection
embeddings, not just Deep-OC-SORT.

Deviations from upstream (see `refine.py`/`tracklet.py` docstrings for the
full detail on each):
  - Stripped: CLI/argparse, dataset-directory walking, `.pkl` I/O, MOT-format
    `save_results`, debug plotting, torchreid/OSNet imports, hard-coded
    `.cuda()`.
  - `Tracklet` bboxes are `xyxy` (this project's convention), not upstream's
    `xywh`; spatial-gate geometry is adjusted accordingly.
  - The spatial gate (`use_spatial_gate`) defaults to **off** -- surgical
    instruments relocate arbitrarily within a frame, unlike sports players,
    so exit/entry position is only weakly informative here.
  - `merge_tracklets` keeps the **lower (older) track id** of a merged pair,
    not upstream's "always fold the second track into the first" -- keeps
    ids stable relative to first appearance.
  - Cosine-distance math runs in torch on the best available device
    (MPS > CUDA > CPU), with a numpy fallback, instead of upstream's
    `cuda`-or-`cpu` hard-coding.
  - New top-level `refine_tracklets(tracklets, config) -> (refined, id_map)`
    entry point (upstream only has a CLI driver) that also returns an
    id-mapping so callers can relabel already-produced output without
    re-tracking.

See `docs/plan-gta-link-tracklet-refinement.md` for the full design
rationale.
"""

from .refine import (
    RefineConfig,
    check_spatial_constraints,
    detect_id_switch,
    find_consecutive_segments,
    get_distance,
    get_distance_matrix,
    get_spatial_constraints,
    merge_tracklets,
    refine_tracklets,
    split_tracklets,
)
from .tracklet import Tracklet

__all__ = [
    "Tracklet",
    "RefineConfig",
    "refine_tracklets",
    "split_tracklets",
    "merge_tracklets",
    "get_distance",
    "get_distance_matrix",
    "detect_id_switch",
    "find_consecutive_segments",
    "get_spatial_constraints",
    "check_spatial_constraints",
]
