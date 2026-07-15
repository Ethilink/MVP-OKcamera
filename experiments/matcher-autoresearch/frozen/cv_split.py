#!/usr/bin/env python
"""
FROZEN — leave-frame-group-out CV split + return-window construction.
NEVER edit from a method/ variant.

Protocol (program.md §4): "leave-frame-group-out cross-validation over the
crops for selection (keep stage-1's 5-seed averaging)". Splits are done by
`group_id` (source frame), not by crop, so a future multi-crop-per-frame
dataset still can't leak one frame's crops across gallery/query.

Multi-frame voting (the champion baseline's 5th ingredient) needs the query
side chunked into ordered "return windows" rather than scored one crop at a
time — `make_windows` does that, always in stable within-identity frame order
(never shuffled — a window is meant to emulate a short real time span).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .loader import RawItem


def group_split(items: list[RawItem], n_gallery_groups: int, seed: int) -> tuple[list[RawItem], list[RawItem]]:
    """Per-identity split: `n_gallery_groups` groups become the enrolment
    gallery, the rest become the query pool. Returns (gallery_items, query_items)
    pooled across all identities present in `items`."""
    by_identity: dict[str, list[RawItem]] = defaultdict(list)
    for it in items:
        by_identity[it.identity].append(it)

    rng = np.random.default_rng(seed)
    gallery: list[RawItem] = []
    query: list[RawItem] = []
    for ident, its in by_identity.items():
        groups = sorted({it.group_id for it in its})  # stable order before shuffling
        perm = rng.permutation(len(groups))
        gallery_groups = {groups[i] for i in perm[:n_gallery_groups]}
        for it in its:
            (gallery if it.group_id in gallery_groups else query).append(it)
    return gallery, query


def restrict_pool(items: list[RawItem], excluded_group_ids: set[str]) -> list[RawItem]:
    """Drop every crop whose group_id is in `excluded_group_ids`. Used to keep
    the CV-selection pool strictly disjoint from the locked holdout's frames —
    CV never even sees the holdout groups, not just re-shuffled away from them."""
    return [it for it in items if it.group_id not in excluded_group_ids]


def make_windows(items: list[RawItem], window: int) -> list[list[RawItem]]:
    """Chunk a query pool into ordered, non-overlapping windows (per identity,
    stable frame order) of up to `window` crops each — the multi-frame evidence
    buffer a returning instrument would present over its first ~0.5s. The last
    window per identity may be shorter than `window`."""
    by_identity: dict[str | None, list[RawItem]] = defaultdict(list)
    for it in items:
        by_identity[it.identity].append(it)
    windows: list[list[RawItem]] = []
    for ident, its in by_identity.items():
        its = sorted(its, key=lambda it: (it.order, it.item_id))
        for i in range(0, len(its), window):
            chunk = its[i:i + window]
            if chunk:
                windows.append(chunk)
    return windows
