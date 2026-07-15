"""Split + connect tracklet refinement, ported from GTA-Link's
`refine_tracklets.py` (github.com/sjc042/gta-link, MIT).

Stripped from upstream: CLI/argparse (`parse_args`, `main`), dataset-directory
walking, `.pkl` I/O, MOT-format `save_results`, `matplotlib`/`seaborn` debug
plotting (`display_Dist`), torchreid/OSNet imports, hard-coded `.cuda()`, and
the unused `get_subtrack` helper (dead code upstream -- `query_subtracks`
uses `Tracklet.extract` instead).

Other deviations from upstream, beyond what's noted per-function below:
  - Bbox geometry (`get_spatial_constraints`, `check_spatial_constraints`)
    is computed from `xyxy` boxes, not upstream's `xywh` -- see
    `tracklet.py`.
  - The temporal-overlap veto in `get_distance` is unconditional, exactly as
    upstream (not a config option).
  - The spatial gate (`use_spatial_gate`) defaults to *off*: this
    application tracks surgical instruments that can be picked up and set
    down anywhere, unlike upstream's sports players, so exit/entry position
    is only weakly informative here (see `docs/plan-gta-link-tracklet-
    refinement.md` §1.3).
  - `merge_tracklets` keeps the **lower (older) track id** of a merged pair,
    not upstream's "always fold track2 into track1" -- this keeps ids
    stable relative to first appearance, which matters more here than
    upstream's benchmark-submission id numbering.
  - Cosine-distance math runs in torch on the best available device
    (MPS > CUDA > CPU, mirroring `deep_ocsort_vendor/embedder.py`'s
    `select_device()`), with a numpy fallback if torch can't be imported
    (torch is a dev-only dependency in this project's `pyproject.toml`, not
    guaranteed present everywhere this module might run). Upstream
    hard-codes `torch.device("cuda" if torch.cuda.is_available() else
    "cpu")`.
  - `refine_tracklets` is a new top-level entry point (not in upstream,
    which only has the `main()` CLI driver) that runs split (optional) then
    connect, and returns an id-mapping alongside the refined dict -- see its
    docstring for the split+merge id-mapping semantics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .tracklet import Tracklet


@dataclass
class RefineConfig:
    """Config for `refine_tracklets`. Defaults are this application's
    starting point per `docs/plan-gta-link-tracklet-refinement.md` §2.4, not
    upstream's sports-MOT defaults verbatim.
    """

    use_split: bool = False
    """Off by default: our generic (non-OSNet) embedder risks DBSCAN
    over-splitting a clean tracklet, and the observed failure mode here is
    fragmentation (needs *connect*), not identity impurity (needs *split*)."""

    eps: float = 0.7
    """DBSCAN neighborhood radius for `split_tracklets`. Upstream's OSNet-
    calibrated value, kept only as a placeholder -- recalibrate per
    plan §2.4/step 5 for whatever embedder produces `Tracklet.features`."""

    min_samples: int = 10
    max_k: int = 3
    len_thres: int = 100

    use_connect: bool = True
    """Always on by default -- this is the actual fix for reappearance-
    after-occlusion id fragmentation."""

    merge_dist_thres: float = 0.4
    """Same calibration caveat as `eps`."""

    use_spatial_gate: bool = False
    """Off by default: instruments relocate arbitrarily within a frame,
    unlike upstream's sports players (plan §1.3). Only meaningful when
    `use_connect=True`."""

    spatial_factor: float = 1.0
    """Scales the max x/y range computed by `get_spatial_constraints`.
    Only used when `use_spatial_gate=True`; plan §1.3 suggests a large
    factor (>=3) if the gate is ever enabled for this application."""


def select_device():
    """Mirrors `deep_ocsort_vendor/embedder.py`'s `select_device()`
    (MPS > CUDA > CPU) -- duplicated rather than imported, since this module
    must stay tracker-agnostic (zero imports from `deep_ocsort_vendor`).
    """
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pairwise_cosine_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """`(len(feats_a), len(feats_b))` matrix of cosine distances (`1 - cos_sim`).

    Runs in torch on `select_device()`; falls back to plain numpy if torch
    isn't importable.
    """
    try:
        import torch
    except ImportError:
        a_norm = feats_a / np.linalg.norm(feats_a, axis=1, keepdims=True)
        b_norm = feats_b / np.linalg.norm(feats_b, axis=1, keepdims=True)
        return 1.0 - a_norm @ b_norm.T

    device = select_device()
    a = torch.as_tensor(feats_a, dtype=torch.float32, device=device)
    b = torch.as_tensor(feats_b, dtype=torch.float32, device=device)
    a_norm = torch.nn.functional.normalize(a, dim=1)
    b_norm = torch.nn.functional.normalize(b, dim=1)
    cos_sim = a_norm @ b_norm.T
    return (1.0 - cos_sim).cpu().numpy()


def get_distance(track1: Tracklet, track2: Tracklet) -> float:
    """Mean cosine distance across all embedding pairs of two tracklets,
    forced to `1.0` (maximally distant / unmergeable) if the tracklets share
    any frame index. This temporal-overlap veto is exact and unconditional,
    exactly as upstream -- not gated by any config flag.

    Deviation: drops upstream's redundant `track1_id`/`track2_id` id
    arguments (used there only for a debug `assert` against
    `track.track_id`) -- `Tracklet` already carries its own id.
    """
    if track1.track_id != track2.track_id and set(track1.times) & set(track2.times):
        return 1.0

    feats1 = np.stack(track1.features).astype(np.float32)
    feats2 = np.stack(track2.features).astype(np.float32)
    return float(_pairwise_cosine_distance(feats1, feats2).mean())


def get_distance_matrix(tracklets: dict[int, Tracklet]) -> tuple[np.ndarray, list[int]]:
    """Symmetric pairwise `get_distance` matrix over `tracklets`, plus the
    `list[int]` of track ids giving each row/column's identity (row `i` /
    column `i` is `ids[i]`).
    """
    ids = list(tracklets.keys())
    n = len(ids)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i, n):
            d = get_distance(tracklets[ids[i]], tracklets[ids[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist, ids


def detect_id_switch(
    features: list[np.ndarray],
    eps: float,
    min_samples: int = 10,
    max_clusters: int | None = None,
) -> tuple[bool, np.ndarray]:
    """DBSCAN (cosine metric) over one tracklet's per-detection embeddings.

    Returns `(id_switch_detected, labels)`: `labels[i]` is the cluster id
    assigned to detection `i` (DBSCAN noise points are reassigned to their
    nearest cluster center, and clusters beyond `max_clusters` are
    iteratively merged pairwise) -- ported near-verbatim from upstream,
    including the pre-DBSCAN `StandardScaler` step.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import cdist

    embs = np.stack(features)
    if len(embs) > 15000:
        embs = embs[1::2]

    embs_scaled = StandardScaler().fit_transform(embs)

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embs_scaled).labels_
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]

    if -1 in labels and len(unique_labels) > 1:
        cluster_centers = np.array([embs_scaled[labels == lbl].mean(axis=0) for lbl in unique_labels])
        for idx in np.where(labels == -1)[0]:
            distances = cdist([embs_scaled[idx]], cluster_centers, metric="cosine")
            labels[idx] = unique_labels[np.argmin(distances)]

    n_clusters = len(unique_labels)
    if max_clusters and n_clusters > max_clusters:
        while n_clusters > max_clusters:
            cluster_centers = np.array([embs_scaled[labels == lbl].mean(axis=0) for lbl in unique_labels])
            dist_matrix = cdist(cluster_centers, cluster_centers, metric="cosine")
            np.fill_diagonal(dist_matrix, np.inf)
            i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            labels[labels == unique_labels[j]] = unique_labels[i]
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != -1]
            n_clusters = len(unique_labels)

    return n_clusters > 1, labels


def split_tracklets(
    tracklets: dict[int, Tracklet],
    eps: float,
    min_samples: int = 10,
    max_k: int = 3,
    len_thres: int = 100,
) -> dict[int, Tracklet]:
    """Fragment each tracklet with a detected id switch into one new
    tracklet per DBSCAN cluster; tracklets shorter than `len_thres` or
    without a detected switch pass through unchanged. Matches upstream:
    fragments get brand-new ids (`max(tracklets) + 1, ...`), the original id
    is dropped entirely.

    Deviation: each fragment's `parent_id` is set to the *pre-split* origin
    id (upstream leaves `parent_id` defaulted to the fragment's own new id,
    losing the split lineage). `refine_tracklets` uses this to compose an
    id-mapping across split + merge.
    """
    next_id = max(tracklets.keys()) + 1
    result: dict[int, Tracklet] = {}

    for tid in sorted(tracklets.keys()):
        trk = tracklets[tid]
        if len(trk.times) < len_thres:
            result[tid] = trk
            continue

        id_switch, labels = detect_id_switch(trk.features, eps=eps, min_samples=min_samples, max_clusters=max_k)
        if not id_switch:
            result[tid] = trk
            continue

        times = np.array(trk.times)
        scores = np.array(trk.scores)
        bboxes = np.array(trk.bboxes)
        feats = np.array(trk.features)
        for label in sorted(set(labels)):
            if label == -1:
                continue  # unreachable in practice: detect_id_switch reassigns noise, but skip defensively
            mask = labels == label
            result[next_id] = Tracklet(
                track_id=next_id,
                times=times[mask].tolist(),
                scores=scores[mask].tolist(),
                bboxes=list(bboxes[mask]),
                features=list(feats[mask]),
                parent_id=tid,
            )
            next_id += 1

    return result


def find_consecutive_segments(times: list[int]) -> list[tuple[int, int]]:
    """Index ranges `[start, end]` (inclusive) of maximal runs of unit-step
    consecutive frame numbers in `times`.
    """
    segments: list[tuple[int, int]] = []
    start = end = 0
    for i in range(1, len(times)):
        if times[i] == times[end] + 1:
            end = i
        else:
            segments.append((start, end))
            start = end = i
    segments.append((start, end))
    return segments


def _query_subtracks(
    seg1: list[tuple[int, int]],
    seg2: list[tuple[int, int]],
    track1: Tracklet,
    track2: Tracklet,
) -> list[Tracklet]:
    """Interleave two tracklets' consecutive-frame segments into temporal
    order, dropping any trailing leftover segment shorter than 30 frames
    (ported inline threshold). Internal helper for `check_spatial_constraints`.
    """
    seg1, seg2 = list(seg1), list(seg2)
    subtracks: list[Tracklet] = []

    while seg1 and seg2:
        s1_start, s1_end = seg1[0]
        s2_start, s2_end = seg2[0]
        subtrack_1 = track1.extract(s1_start, s1_end)
        subtrack_2 = track2.extract(s2_start, s2_end)

        if track1.times[s1_start] < track2.times[s2_start]:
            subtracks.append(subtrack_1)
            subtracks.append(subtrack_2)
        else:
            subtracks.append(subtrack_2)
            subtracks.append(subtrack_1)
        seg1.pop(0)
        seg2.pop(0)

    seg_remain, track_remain = (seg1, track1) if seg1 else (seg2, track2)
    while seg_remain:
        s_start, s_end = seg_remain[0]
        if (s_end - s_start) < 30:
            seg_remain.pop(0)
            continue
        subtracks.append(track_remain.extract(s_start, s_end))
        seg_remain.pop(0)

    return subtracks


def get_spatial_constraints(tracklets: dict[int, Tracklet], factor: float) -> tuple[float, float]:
    """Max x/y center-to-center range across every bbox in every tracklet,
    scaled by `factor`. Only meaningful when `use_spatial_gate=True`.

    Deviation: centers are computed from `xyxy` bboxes (`(x1+x2)/2,
    (y1+y2)/2`), not upstream's `xywh` (`x + w/2, y + h/2`) -- see
    `tracklet.py`.
    """
    xs: list[float] = []
    ys: list[float] = []
    for trk in tracklets.values():
        for x1, y1, x2, y2 in trk.bboxes:
            xs.append((x1 + x2) / 2)
            ys.append((y1 + y2) / 2)
    return (max(xs) - min(xs)) * factor, (max(ys) - min(ys)) * factor


def check_spatial_constraints(track1: Tracklet, track2: Tracklet, max_x_range: float, max_y_range: float) -> bool:
    """True if, across `track1`/`track2`'s temporally-interleaved
    consecutive segments, every exit-point-to-next-entry-point jump stays
    within `max_x_range`/`max_y_range`. Ported from upstream nearly
    unchanged; centers computed from `xyxy` (see `get_spatial_constraints`).
    Only called when `use_spatial_gate=True`.
    """
    seg1 = find_consecutive_segments(track1.times)
    seg2 = find_consecutive_segments(track2.times)
    subtracks = _query_subtracks(seg1, seg2, track1, track2)
    if not subtracks:
        return True

    prev = subtracks.pop(0)
    while subtracks:
        cur = subtracks.pop(0)
        if prev.parent_id == cur.parent_id:
            prev = cur
            continue

        x1, y1, x2, y2 = prev.bboxes[-1]
        exit_x, exit_y = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = cur.bboxes[0]
        entry_x, entry_y = (x1 + x2) / 2, (y1 + y2) / 2

        if abs(exit_x - entry_x) > max_x_range or abs(exit_y - entry_y) > max_y_range:
            return False
        prev = cur

    return True


def _merge_pair(keep: Tracklet, drop: Tracklet) -> Tracklet:
    """Concatenate `drop`'s detections onto `keep`, preserving `keep.track_id`
    -- the caller (`merge_tracklets`) ensures `keep` is the lower/older id.
    Matches upstream's plain-concatenation (not time-sorted) semantics.
    """
    return Tracklet(
        track_id=keep.track_id,
        times=keep.times + drop.times,
        scores=keep.scores + drop.scores,
        bboxes=keep.bboxes + drop.bboxes,
        features=keep.features + drop.features,
        parent_id=keep.parent_id,
    )


def merge_tracklets(
    tracklets: dict[int, Tracklet],
    merge_dist_thres: float,
    use_spatial_gate: bool = False,
    max_x_range: float | None = None,
    max_y_range: float | None = None,
) -> tuple[dict[int, Tracklet], dict[int, int]]:
    """Hierarchical agglomerative merge: repeatedly fold the closest pair of
    tracklets (mean cosine distance below `merge_dist_thres`) together,
    subject to `check_spatial_constraints` if `use_spatial_gate=True`, until
    no pair remains below threshold.

    Returns `(merged_tracklets, id_map)`. `id_map` maps every input id to
    the id it ends up under (self-mapped if never merged); when two
    tracklets merge, **the surviving id is the lower (numerically older) of
    the pair** -- a deviation from upstream, which always folds "track2"
    into "track1" regardless of id order, to keep ids stable relative to
    first appearance.
    """
    if use_spatial_gate and (max_x_range is None or max_y_range is None):
        raise ValueError("use_spatial_gate=True requires max_x_range/max_y_range")

    tracklets = dict(tracklets)
    id_map: dict[int, int] = {tid: tid for tid in tracklets}

    if len(tracklets) < 2:
        return tracklets, id_map

    dist, idx2tid = get_distance_matrix(tracklets)
    non_diagonal = ~np.eye(dist.shape[0], dtype=bool)

    while np.any(dist[non_diagonal] < merge_dist_thres):
        masked = np.where(non_diagonal, dist, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        tid_a, tid_b = idx2tid[i], idx2tid[j]
        keep_id, drop_id = (tid_a, tid_b) if tid_a < tid_b else (tid_b, tid_a)

        if use_spatial_gate and not check_spatial_constraints(tracklets[keep_id], tracklets[drop_id], max_x_range, max_y_range):
            dist[i, j] = dist[j, i] = merge_dist_thres
            continue

        tracklets[keep_id] = _merge_pair(tracklets[keep_id], tracklets[drop_id])
        del tracklets[drop_id]
        for old_id, mapped_id in id_map.items():
            if mapped_id == drop_id:
                id_map[old_id] = keep_id

        drop_idx = i if idx2tid[i] == drop_id else j
        dist = np.delete(dist, drop_idx, axis=0)
        dist = np.delete(dist, drop_idx, axis=1)
        idx2tid = {idx: tid for idx, tid in enumerate(tracklets.keys())}
        keep_idx = next(idx for idx, tid in idx2tid.items() if tid == keep_id)

        for idx in range(dist.shape[0]):
            d = get_distance(tracklets[keep_id], tracklets[idx2tid[idx]])
            dist[keep_idx, idx] = d
            dist[idx, keep_idx] = d

        non_diagonal = ~np.eye(dist.shape[0], dtype=bool)

    return tracklets, id_map


def refine_tracklets(tracklets: dict[int, Tracklet], config: RefineConfig | None = None) -> tuple[dict[int, Tracklet], dict[int, int]]:
    """Run split (if `config.use_split`) then connect (if `config.use_connect`)
    over `tracklets`. Returns `(refined_tracklets, id_map)`: `id_map` maps
    every original input id to the id it now lives under, including a
    self-mapping for unmerged/unsplit tracklets, so callers can relabel
    already-rendered structures (e.g. per-frame track ids) without re-running
    the tracker.

    Split+merge id-mapping caveat: splitting fragments one tracklet into
    several *new* ids (upstream drops the original id entirely -- see
    `split_tracklets`), which is a genuine one-to-many relationship a flat
    `dict[int, int]` can't represent exactly. When `use_split=True` and a
    tracklet is actually fragmented, its original id is mapped to whichever
    surviving fragment holds the *majority* of its original detections --
    exact for the default `use_split=False` path (this function's primary,
    tested use case), an approximation otherwise.
    """
    config = config or RefineConfig()
    if not tracklets:
        return {}, {}

    working = dict(tracklets)

    if config.use_split:
        working = split_tracklets(
            working,
            eps=config.eps,
            min_samples=config.min_samples,
            max_k=config.max_k,
            len_thres=config.len_thres,
        )
        fragments_by_origin: dict[int, list[Tracklet]] = defaultdict(list)
        for frag in working.values():
            fragments_by_origin[frag.parent_id].append(frag)
        primary_fragment_for = {
            tid: max(fragments_by_origin[tid], key=len).track_id for tid in tracklets if tid in fragments_by_origin
        }
    else:
        primary_fragment_for = {tid: tid for tid in tracklets}

    if config.use_connect:
        max_x_range = max_y_range = None
        if config.use_spatial_gate:
            max_x_range, max_y_range = get_spatial_constraints(working, config.spatial_factor)
        merged, merge_map = merge_tracklets(
            working,
            merge_dist_thres=config.merge_dist_thres,
            use_spatial_gate=config.use_spatial_gate,
            max_x_range=max_x_range,
            max_y_range=max_y_range,
        )
    else:
        merged, merge_map = working, {tid: tid for tid in working}

    id_map = {tid: merge_map[primary_fragment_for[tid]] for tid in tracklets if tid in primary_fragment_for}
    return merged, id_map
