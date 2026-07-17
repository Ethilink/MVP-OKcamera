#!/usr/bin/env python
"""
FROZEN — the guarded eval driver. NEVER edit from a method/ variant.

Calls a method ONLY through the fixed interface (frozen/interface.py):
build_gallery / score / accept. Computes the guarded, held-out metrics from
program.md §4:

  - re-ID top-1 (CV-averaged, ±seed std)
  - instrument1<->2 twin-error count
  - foreign-reject rate + true-accept rate at the method's own operating point
  - per-instrument breakdown

Two entry points:
  - `run_cv(...)`   — 5-seed leave-frame-group-out CV over the CV/selection
                       pool ONLY (never the locked holdout). Use this for
                       hyperparameter/family selection.
  - `run_locked_holdout(...)` — ONE-TIME report on the deterministic locked
                       split. Call this only after every hyperparameter is
                       already fixed from run_cv — never loop/search against it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .cv_split import group_split, make_windows
from .holdout import (
    foreign_holdout_split,
    instrument_cv_selection_pool,
    instrument_locked_gallery_items,
    instrument_locked_query_items,
)
from .interface import REJECT, MatcherMethod, check_decision
from .loader import RawItem

TWIN_PAIR = ("instrument1", "instrument2")


def _crops_masks(items: list[RawItem]):
    return [it.image for it in items], [it.mask for it in items]


def _build_galleries(method: MatcherMethod, gallery_items: list[RawItem]) -> dict[str, Any]:
    by_identity: dict[str, list[RawItem]] = defaultdict(list)
    for it in gallery_items:
        by_identity[it.identity].append(it)
    galleries = {}
    for ident, its in by_identity.items():
        crops, masks = _crops_masks(its)
        galleries[ident] = method.build_gallery(crops, masks, {"identity": ident})
    return galleries


def _score_windows(method: MatcherMethod, windows: list[list[RawItem]], galleries: dict[str, Any]):
    """Runs score()+accept() on each window WITHOUT ever exposing the window's
    true identity to the method. Returns list of (true_identity_or_None, decision)."""
    results = []
    for win in windows:
        true_identity = win[0].identity  # ground truth kept eval-side only
        crops, masks = _crops_masks(win)
        # meta given to the method carries no identity/label information
        meta = {"n_frames": len(win)}
        scores = method.score(crops, masks, meta, galleries)
        decision = method.accept(scores)
        check_decision(decision, scores)
        results.append((true_identity, decision))
    return results


def _metrics_from_results(genuine_results, foreign_results) -> dict:
    n_genuine = len(genuine_results)
    correct = sum(1 for true, dec in genuine_results if dec == true)
    rejected_genuine = sum(1 for _, dec in genuine_results if dec == REJECT)
    wrong_link = n_genuine - correct - rejected_genuine

    twin_errors = sum(
        1 for true, dec in genuine_results
        if true in TWIN_PAIR and dec in TWIN_PAIR and dec != true
    )

    per_instrument: dict[str, dict] = {}
    by_true: dict[str, list] = defaultdict(list)
    for true, dec in genuine_results:
        by_true[true].append(dec)
    for ident, decs in by_true.items():
        n = len(decs)
        per_instrument[ident] = {
            "n_windows": n,
            "top1": round(sum(1 for d in decs if d == ident) / n, 3) if n else None,
        }

    n_foreign = len(foreign_results)
    foreign_rejected = sum(1 for _, dec in foreign_results if dec == REJECT)

    return {
        "n_genuine_windows": n_genuine,
        "reid_top1": round(correct / n_genuine, 4) if n_genuine else None,
        "true_accept": round(correct / n_genuine, 4) if n_genuine else None,
        "false_reject_rate": round(rejected_genuine / n_genuine, 4) if n_genuine else None,
        "wrong_link_rate": round(wrong_link / n_genuine, 4) if n_genuine else None,
        "twin_errors_1v2": twin_errors,
        "n_foreign_windows": n_foreign,
        "foreign_reject": round(foreign_rejected / n_foreign, 4) if n_foreign else None,
        "per_instrument": per_instrument,
    }


@dataclass
class CVReport:
    per_seed: list = field(default_factory=list)
    reid_top1_mean: float = 0.0
    reid_top1_std: float = 0.0
    foreign_reject_mean: float = 0.0
    foreign_reject_std: float = 0.0
    true_accept_mean: float = 0.0
    twin_errors_total: int = 0
    n_seeds: int = 0

    def as_dict(self):
        return {
            "per_seed": self.per_seed,
            "reid_top1_mean": round(self.reid_top1_mean, 4),
            "reid_top1_std": round(self.reid_top1_std, 4),
            "foreign_reject_mean": round(self.foreign_reject_mean, 4),
            "foreign_reject_std": round(self.foreign_reject_std, 4),
            "true_accept_mean": round(self.true_accept_mean, 4),
            "twin_errors_total": self.twin_errors_total,
            "n_seeds": self.n_seeds,
        }


def run_cv(
    method: MatcherMethod,
    instrument_items: list[RawItem],
    foreign_items: list[RawItem],
    n_gallery_groups: int = 5,
    window: int = 3,
    seeds: tuple = (0, 1, 2, 3, 4),
) -> CVReport:
    """5-seed leave-frame-group-out CV, restricted to the CV/selection pool
    (order 5..11 per instrument; never the locked holdout frames order>=12 or
    the locked-gallery-only frames order<5). Foreign objects use their own
    CV/selection pool (holds out the last source image)."""
    cv_instrument_pool = instrument_cv_selection_pool(instrument_items)
    foreign_cv_pool, _foreign_holdout = foreign_holdout_split(foreign_items)

    report = CVReport()
    seed_reids, seed_rejects = [], []
    for seed in seeds:
        gallery_items, query_items = group_split(cv_instrument_pool, n_gallery_groups, seed)
        galleries = _build_galleries(method, gallery_items)

        genuine_windows = make_windows(query_items, window)
        genuine_results = _score_windows(method, genuine_windows, galleries)

        foreign_windows = make_windows(foreign_cv_pool, window)
        foreign_results = _score_windows(method, foreign_windows, galleries)

        m = _metrics_from_results(genuine_results, foreign_results)
        m["seed"] = seed
        report.per_seed.append(m)
        seed_reids.append(m["reid_top1"] or 0.0)
        seed_rejects.append(m["foreign_reject"] or 0.0)
        report.twin_errors_total += m["twin_errors_1v2"]

    report.reid_top1_mean = float(np.mean(seed_reids))
    report.reid_top1_std = float(np.std(seed_reids))
    report.foreign_reject_mean = float(np.mean(seed_rejects))
    report.foreign_reject_std = float(np.std(seed_rejects))
    report.true_accept_mean = report.reid_top1_mean  # by construction (correct AND not rejected)
    report.n_seeds = len(seeds)
    return report


def run_locked_holdout(
    method: MatcherMethod,
    instrument_items: list[RawItem],
    foreign_items: list[RawItem],
    window: int = 3,
) -> dict:
    """ONE-TIME report on the deterministic locked split. Call only after every
    hyperparameter is already fixed via run_cv — never search against this."""
    gallery_items = instrument_locked_gallery_items(instrument_items)
    query_items = instrument_locked_query_items(instrument_items)
    _foreign_cv, foreign_holdout_items = foreign_holdout_split(foreign_items)

    galleries = _build_galleries(method, gallery_items)

    genuine_windows = make_windows(query_items, window)
    genuine_results = _score_windows(method, genuine_windows, galleries)

    foreign_windows = make_windows(foreign_holdout_items, window)
    foreign_results = _score_windows(method, foreign_windows, galleries)

    return _metrics_from_results(genuine_results, foreign_results)
