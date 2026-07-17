#!/usr/bin/env python
"""Re-run the champion's CV hyperparameter sweep (program.md's guarded
metric) for the CURRENT active_features composition (see app/mvp.toml):
log_area/log_length/aspect_ratio (round 3b). Two later attempts to add more
registry features on top -- round 3c (solidity/extent/width_profile,
swapped in for aspect_ratio) and round 3d (eccentricity/hu_moments, added
alongside aspect_ratio) -- both regressed re-ID and were reverted; see
PARAMS.md. Kept here so re-running this script always sweeps whatever
composition is actually deployed.

Faithful to the frozen harness (frozen/cv_split.py, frozen/holdout.py,
frozen/eval.py) -- same CV/selection pool, same 5-seed leave-frame-group-out
split, same windowing -- but decouples the two halves of a grid point:

  - alpha, size_alpha: change the embedding/dictionary and the Lasso solve.
    Expensive; solved by running the ACTUAL production `ChampionMethod`
    through `score()` (never `accept()`) for every CV window, once per
    (alpha, size_alpha) pair, with DINO embeddings memoized across the
    whole sweep (fixed active_features/mask_dilate_px/model_id -> a crop's
    embedding never changes across grid points).
  - tau, margin: pure post-hoc thresholding of the scores already computed
    above -- reimplemented here identically to `ChampionMethod.accept()`
    (see champion.py) so sweeping 10x3 threshold combinations per
    (alpha, size_alpha) point is instant arithmetic, not 30 more Lasso
    re-solves. This is what makes a 630-point-equivalent grid tractable.

Selection rule (mirrors PARAMS.md): maximize CV re-ID top-1 subject to
foreign-reject staying >= the round-3 champion's own CV floor (0.9733).

Run from `model/`: `uv run python matcher-autoresearch/sweep_hyperparams.py`
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from frozen.cv_split import group_split, make_windows  # noqa: E402
from frozen.holdout import foreign_holdout_split, instrument_cv_selection_pool  # noqa: E402
from frozen.loader import RawItem, load_foreign, load_instruments  # noqa: E402
from orc_model.pipelines.matching.champion import ChampionMethod  # noqa: E402

# -- current live composition (app/mvp.toml [tracker.matcher] active_features)
ACTIVE_FEATURES = ("log_area", "log_length", "aspect_ratio")

# -- grid: alpha/size_alpha need a real Lasso solve; kept modest (7x6=42
#    points) since each is a full 5-seed CV pass. tau/margin are free
#    (evaluated below from cached scores), so given a generous 12x5 grid.
#    size_alpha's low end widened to 0.1/0.15 after round 3c's optimum
#    landed on the previous grid's own boundary (0.25) -- don't repeat that.
ALPHA_GRID = (0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012)
SIZE_ALPHA_GRID = (0.1, 0.15, 0.25, 0.5, 0.75, 1.0)
TAU_GRID = tuple(round(0.15 + 0.05 * i, 2) for i in range(10))  # 0.15..0.60
MARGIN_GRID = (0.01, 0.02, 0.03, 0.05, 0.08)

SEEDS = (0, 1, 2, 3, 4)
N_GALLERY_GROUPS = 5
WINDOW = 3
TWIN_PAIR = ("instrument1", "instrument2")

# round-3 champion's own guarded CV floor (PARAMS.md) -- the new operating
# point "may only go up" from here, not regress the reject rate.
BASELINE_FOREIGN_REJECT_FLOOR = 0.9733


def enable_embed_cache(embedder) -> None:
    """Memoize DINO forward passes by exact pixel content. Safe here because
    active_features/mask_dilate_px/model_id are fixed for this whole sweep --
    only alpha/size_alpha/tau/margin vary, none of which affect the DINO
    embedding of a given crop."""
    if getattr(embedder, "_sweep_cached", False):
        return
    orig_embed = embedder.embed
    cache: dict[bytes, np.ndarray] = {}

    def cached_embed(images, batch_size=8):
        keys = [np.asarray(im).tobytes() for im in images]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            fresh = orig_embed([images[i] for i in missing], batch_size=batch_size)
            for idx, vec in zip(missing, fresh):
                cache[keys[idx]] = vec
        return np.stack([cache[k] for k in keys])

    embedder.embed = cached_embed
    embedder._sweep_cached = True


def _crops_masks(items: list[RawItem]):
    return [it.image for it in items], [it.mask for it in items]


@dataclass
class WindowScores:
    true_identity: str | None
    scores: dict[str, float]


def _score_all_windows(
    method: ChampionMethod, instrument_items: list[RawItem], foreign_items: list[RawItem]
) -> list[list[WindowScores]]:
    """Per seed: build galleries + score() every genuine/foreign CV window.
    accept() is deliberately never called -- thresholding happens later."""
    cv_pool = instrument_cv_selection_pool(instrument_items)
    foreign_cv_pool, _holdout = foreign_holdout_split(foreign_items)

    per_seed: list[list[WindowScores]] = []
    for seed in SEEDS:
        gallery_items, query_items = group_split(cv_pool, N_GALLERY_GROUPS, seed)
        by_identity: dict[str, list[RawItem]] = defaultdict(list)
        for it in gallery_items:
            by_identity[it.identity].append(it)
        galleries = {
            ident: method.build_gallery(*_crops_masks(its), {"identity": ident})
            for ident, its in by_identity.items()
        }

        seed_results: list[WindowScores] = []
        for win in make_windows(query_items, WINDOW):
            crops, masks = _crops_masks(win)
            scores = method.score(crops, masks, {"n_frames": len(win)}, galleries)
            seed_results.append(WindowScores(win[0].identity, scores))
        for win in make_windows(foreign_cv_pool, WINDOW):
            crops, masks = _crops_masks(win)
            scores = method.score(crops, masks, {"n_frames": len(win)}, galleries)
            seed_results.append(WindowScores(None, scores))
        per_seed.append(seed_results)
    return per_seed


def _accept(scores: dict[str, float], tau: float, margin: float) -> str:
    """Verbatim copy of ChampionMethod.accept()'s decision rule (champion.py)
    so thresholding here stays faithful to production."""
    if not scores:
        return "REJECT"
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_score = ranked[0]
    if best_score < tau:
        return "REJECT"
    if len(ranked) > 1 and (best_score - ranked[1][1]) < margin:
        return "REJECT"
    return best_id


def _metrics(per_seed: list[list[WindowScores]], tau: float, margin: float) -> dict:
    seed_reids, seed_rejects = [], []
    twin_errors = 0
    for seed_results in per_seed:
        genuine = [(w.true_identity, _accept(w.scores, tau, margin))
                   for w in seed_results if w.true_identity is not None]
        foreign = [_accept(w.scores, tau, margin)
                   for w in seed_results if w.true_identity is None]

        n_genuine = len(genuine)
        correct = sum(1 for true, dec in genuine if dec == true)
        seed_reids.append(correct / n_genuine if n_genuine else 0.0)

        n_foreign = len(foreign)
        rejected = sum(1 for dec in foreign if dec == "REJECT")
        seed_rejects.append(rejected / n_foreign if n_foreign else 0.0)

        twin_errors += sum(
            1 for true, dec in genuine
            if true in TWIN_PAIR and dec in TWIN_PAIR and dec != true
        )
    return {
        "reid_top1_mean": float(np.mean(seed_reids)),
        "reid_top1_std": float(np.std(seed_reids)),
        "foreign_reject_mean": float(np.mean(seed_rejects)),
        "foreign_reject_std": float(np.std(seed_rejects)),
        "twin_errors_total": twin_errors,
    }


def main() -> None:
    instrument_items = load_instruments()
    foreign_items = load_foreign()

    print(f"active_features = {ACTIVE_FEATURES}")
    print(f"grid: {len(ALPHA_GRID)} alpha x {len(SIZE_ALPHA_GRID)} size_alpha "
          f"(Lasso re-solves) x {len(TAU_GRID)} tau x {len(MARGIN_GRID)} margin "
          f"(free) = {len(ALPHA_GRID) * len(SIZE_ALPHA_GRID) * len(TAU_GRID) * len(MARGIN_GRID)} points\n")

    results = []
    for alpha in ALPHA_GRID:
        for size_alpha in SIZE_ALPHA_GRID:
            method = ChampionMethod(alpha=alpha, size_alpha=size_alpha, active_features=ACTIVE_FEATURES)
            enable_embed_cache(method._embedder)
            per_seed = _score_all_windows(method, instrument_items, foreign_items)
            for tau in TAU_GRID:
                for margin in MARGIN_GRID:
                    m = _metrics(per_seed, tau, margin)
                    results.append({"alpha": alpha, "size_alpha": size_alpha, "tau": tau, "margin": margin, **m})
            print(f"alpha={alpha:.5f} size_alpha={size_alpha:.2f} -- done ({len(TAU_GRID) * len(MARGIN_GRID)} threshold points)")

    # selection rule: maximize re-ID top-1 subject to reject >= baseline floor
    feasible = [r for r in results if r["foreign_reject_mean"] >= BASELINE_FOREIGN_REJECT_FLOOR]
    pool = feasible if feasible else results
    if not feasible:
        print(f"\nWARNING: no grid point reached the {BASELINE_FOREIGN_REJECT_FLOOR} "
              f"reject floor -- reporting the best point found anyway (constraint relaxed).")
    best = max(pool, key=lambda r: (r["reid_top1_mean"], r["foreign_reject_mean"]))

    print("\n=== BEST OPERATING POINT ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    top5 = sorted(pool, key=lambda r: (r["reid_top1_mean"], r["foreign_reject_mean"]), reverse=True)[:5]
    print("\n=== TOP 5 (for context) ===")
    for r in top5:
        print(f"  alpha={r['alpha']:.5f} size_alpha={r['size_alpha']:.2f} tau={r['tau']:.2f} "
              f"margin={r['margin']:.2f} -> reid={r['reid_top1_mean']:.4f}"
              f"±{r['reid_top1_std']:.4f} reject={r['foreign_reject_mean']:.4f}"
              f"±{r['foreign_reject_std']:.4f} twins={r['twin_errors_total']}")


if __name__ == "__main__":
    main()
