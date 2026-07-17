#!/usr/bin/env python
"""
FROZEN — the locked held-out split. NEVER edit from a method/ variant, and
NEVER call this module during CV/parameter selection (only `runs/*/run_eval.py`
"champion report" step may import it, and only after all hyperparameters are
already fixed from the CV pool).

Deterministic, NOT seed-shuffled — a fixed reserved slice of frames, per
program.md §4 ("a reserved instrument-frame split ... scored only for the
champion, never optimized against"):

- Instruments: each instrument has frames order 0..14 (file names
  frame_00001.jpg .. frame_00015.jpg, identical pattern across all 8 dirs —
  verified at authoring time). The locked holdout QUERY set is the last 3
  frames (order >= 12, i.e. frame_00013/14/15) of every instrument — CV never
  even sees these, at any order, so no amount of hyperparameter search can
  leak into them (stronger than re-shuffling around a fixed test set). The
  locked holdout GALLERY is the first 5 (order < 5); CV/selection is free to
  ALSO draw on order < 5 as ordinary gallery/query material (order 0..11,
  12 frames/instrument) — reusing those specimens as CV material doesn't leak
  the holdout QUERY answers (no learnable per-crop state is fit here, only a
  handful of global scalar hyperparameters), it just means CV isn't needlessly
  starved down to 7 frames/instrument.
- Foreign objects: `other_objects` has 5 source images (frame_00001..5.jpg,
  ~12 crops each). The locked holdout foreign pool is the LAST source image
  (order == max); the CV/selection foreign pool is the rest.
"""
from __future__ import annotations

from .loader import RawItem

HOLDOUT_QUERY_MIN_ORDER = 12    # frame_00013.jpg onward (0-indexed order) — locked, never in CV
CV_GALLERY_MAX_ORDER = 5          # frame_00001..00005.jpg — the FIXED locked-report gallery


def instrument_holdout_query_group_ids(items: list[RawItem]) -> set[str]:
    """Frame file names reserved as the locked holdout QUERY set (last 3 frames
    of every instrument). Never present in the CV/selection pool."""
    return {it.group_id for it in items if it.order >= HOLDOUT_QUERY_MIN_ORDER}


def instrument_locked_gallery_items(items: list[RawItem]) -> list[RawItem]:
    """The fixed (non-random) gallery used ONLY for the one-time locked-holdout
    champion report: the first 5 frames per instrument."""
    return [it for it in items if it.order < CV_GALLERY_MAX_ORDER]


def instrument_locked_query_items(items: list[RawItem]) -> list[RawItem]:
    """The fixed (non-random) query pool used ONLY for the one-time locked-
    holdout champion report: the last 3 frames per instrument."""
    return [it for it in items if it.order >= HOLDOUT_QUERY_MIN_ORDER]


def instrument_cv_selection_pool(items: list[RawItem]) -> list[RawItem]:
    """Everything CV/parameter-selection is allowed to touch: frames order
    0..11 (12 per instrument) — strictly disjoint from the locked holdout QUERY
    set (order >= 12), which CV never sees under any seed/split."""
    return [it for it in items if it.order < HOLDOUT_QUERY_MIN_ORDER]


def foreign_holdout_split(items: list[RawItem]) -> tuple[list[RawItem], list[RawItem]]:
    """(cv_selection_pool, locked_holdout_pool) for the foreign-object negatives,
    split by SOURCE IMAGE (group_id), last image reserved for the locked report."""
    orders = sorted({it.order for it in items})
    if not orders:
        return [], []
    holdout_order = orders[-1]
    cv_pool = [it for it in items if it.order != holdout_order]
    holdout_pool = [it for it in items if it.order == holdout_order]
    return cv_pool, holdout_pool
