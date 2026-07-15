#!/usr/bin/env python
"""
CHAMPION — round 3, matcher-autoresearch.

Lives here at `model/src/orc_model/pipelines/matching/champion.py`; ported
near-verbatim from `experiments/matcher-autoresearch/champion/champion.py`
(the matcher-autoresearch loop's winning module) — see that directory's
LEADERBOARD.md/PARAMS.md for the full selection history this docstring
summarizes below.

  SPARSE-REPRESENTATION-BASED CLASSIFICATION (SRC): a query return-window's
  DINOv2-B CLS embeddings (same masked-crop representation as the round-0
  baseline this method superseded) are jointly reconstructed as a SPARSE
  linear combination of every candidate gallery's atoms, concatenated into
  one dictionary and solved as a SINGLE L1-penalized multi-task least-squares
  problem (`MultiTaskLasso`) shared across the whole window at once — "Joint
  Sparse Representation" (Wright et al. 2009's SRC, extended from one frame
  to a window-joint code). The per-candidate score is a RECONSTRUCTION
  quality term (inverse residual of that candidate's atoms alone against the
  window) times the SPARSITY CONCENTRATION INDEX (SCI, Wright et al. 2009
  eq. 12): how much of the solved code's L1 mass sits on that one candidate's
  atoms versus spread thinly across every candidate — the open-set rejection
  signal. No crop-to-crop similarity is ever computed anywhere in this
  method; matching is convex reconstruction over a shared dictionary, not
  nearest-neighbour search.

Superseded the round-0 baseline (DINOv2-B CLS + moderate size-fusion +
top-3-mean + multi-frame voting): CV re-ID 0.9333 vs 0.850 (+0.083, clears
the round-0 seed-variance band of 0.043), CV foreign-reject 0.9733 vs 0.9467
(both metrics up), 0 twin errors either way. Cleared a Codex leak-check
(clean=true) before promotion — see
`experiments/matcher-autoresearch/TRIED.md` "Round 3 leak-check verdict" for
the full file:line audit.

Implements the build_gallery/score/accept interface (interface.py, mirrors
linker-design.md §6). Hyperparameters below are the winners of the 630-point
CV grid sweep in `experiments/matcher-autoresearch/runs/r3-c2/run_eval.py` —
see PARAMS.md for the full table + the guarded held-out numbers + ablations.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.linear_model import MultiTaskLasso

from .embedder import get_embedder
from .interface import REJECT, Gallery
from .size_features import size_feature

FAMILY = (
    "sparse-representation-based classification (SRC): joint dictionary "
    "reconstruction over concatenated candidate galleries via L1-penalized "
    "multi-task (window-joint) least squares, open-set rejection via the "
    "Sparsity Concentration Index (SCI)"
)

# Winning hyperparameters from the CV sweep (runs/r3-c2/cv_sweep.json).
DEFAULT_ALPHA = 0.0003
DEFAULT_SIZE_ALPHA = 0.5
DEFAULT_TAU = 0.30
DEFAULT_MARGIN = 0.02
DEFAULT_MASK_DILATE_PX = 2
DEFAULT_MAX_ITER = 2000


class ChampionGallery(Gallery):
    __slots__ = ("views", "identity")

    def __init__(self, views: np.ndarray, identity: str | None):
        self.views = views  # (n_views, D) fused feature matrix — dictionary atoms
        self.identity = identity


def _apply_mask(image: np.ndarray, mask: np.ndarray, dilate_px: int) -> Image.Image:
    m = mask.astype(bool)
    if dilate_px > 0:
        struct = np.ones((3, 3), dtype=bool)
        m = binary_dilation(m, structure=struct, iterations=dilate_px)
    masked = image * m[:, :, None].astype(image.dtype)
    return Image.fromarray(masked)


class ChampionMethod:
    """CHAMPION — round 3, SRC via joint dictionary reconstruction + SCI. See
    module docstring + PARAMS.md."""

    family = FAMILY

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        size_alpha: float = DEFAULT_SIZE_ALPHA,
        tau: float = DEFAULT_TAU,
        margin: float = DEFAULT_MARGIN,
        mask_dilate_px: int = DEFAULT_MASK_DILATE_PX,
        max_iter: int = DEFAULT_MAX_ITER,
        model_id: str = "facebook/dinov2-base",
    ):
        self.alpha = alpha
        self.size_alpha = size_alpha
        self.tau = tau
        self.margin = margin
        self.mask_dilate_px = mask_dilate_px
        self.max_iter = max_iter
        self._embedder = get_embedder(model_id)

    # -- shared feature extraction (same fused-vector convention as the
    #    round-0 baseline: L2-normalized DINO CLS + z-scored size cue) -------
    def _embed_fused(self, crops: list, masks: list) -> np.ndarray:
        imgs = [_apply_mask(c, m, self.mask_dilate_px) for c, m in zip(crops, masks)]
        E = self._embedder.embed(imgs)
        En = E / np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-8)
        S = np.stack([size_feature(m) for m in masks], axis=0)
        fused = np.concatenate([En, self.size_alpha * S], axis=1)
        return fused.astype(np.float64)  # MultiTaskLasso wants float64

    # -- build_gallery/score/accept interface (linker-design.md §6) ----------
    def build_gallery(self, crops: list, masks: list, meta: dict) -> ChampionGallery:
        views = self._embed_fused(crops, masks)
        return ChampionGallery(views=views, identity=meta.get("identity"))

    def score(self, query_crops: list, query_masks: list, meta: dict,
              galleries: dict[str, ChampionGallery]) -> dict[str, float]:
        if not galleries:
            return {}
        candidate_ids = list(galleries.keys())
        K = len(candidate_ids)

        # -- assemble the joint dictionary: columns = every candidate's
        #    atoms, unit-norm (classic SRC preprocessing).
        atom_blocks = [galleries[cid].views.T for cid in candidate_ids]  # each (D, n_views_i)
        D_mat = np.concatenate(atom_blocks, axis=1)  # (D, N_atoms)
        owner = np.concatenate([
            np.full(atom_blocks[j].shape[1], j, dtype=int) for j in range(K)
        ])
        col_norms = np.linalg.norm(D_mat, axis=0)
        col_norms = np.where(col_norms < 1e-8, 1.0, col_norms)
        D_norm = D_mat / col_norms

        Fq = self._embed_fused(query_crops, query_masks)  # (n_frames, D)
        Y = Fq.T  # (D, n_frames) — treat embedding coords as "samples", atoms as "features"
        n_frames = Y.shape[1]

        # -- ONE joint/multi-task sparse code shared across the whole window
        #    (row-sparsity across frames: MultiTaskLasso) — "Joint Sparse
        #    Representation," the family's central hypothesis.
        solver = MultiTaskLasso(alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter)
        solver.fit(D_norm, Y)
        W = solver.coef_  # sklearn multi-task convention: (n_targets, n_features)
        X = np.atleast_2d(W)
        X = X.T if X.shape[0] == n_frames else X  # normalize to (N_atoms, n_frames)

        total_l1 = np.abs(X).sum()
        scores: dict[str, float] = {}
        for j, cid in enumerate(candidate_ids):
            idx = np.where(owner == j)[0]
            Xc = X[idx, :]
            Dc = D_norm[:, idx]
            recon_c = Dc @ Xc
            resid = float(np.linalg.norm(Y - recon_c) / max(1, n_frames) ** 0.5)
            sim = 1.0 / (1.0 + resid)  # bounded (0, 1], monotonic decreasing in residual

            if K > 1 and total_l1 > 1e-12:
                # Sparsity Concentration Index (Wright et al. 2009, eq. 12):
                # 1.0 = all coefficient mass on this one candidate's atoms,
                # 0.0 = mass spread perfectly evenly across all K candidates.
                cand_l1 = float(np.abs(Xc).sum())
                sci = (K * cand_l1 / total_l1 - 1.0) / (K - 1.0)
                sci = float(np.clip(sci, 0.0, 1.0))
            else:
                sci = 1.0  # single-candidate one-missing prior: vacuous
            scores[cid] = sim * sci
        return scores

    def accept(self, scores: dict[str, float]) -> str:
        if not scores:
            return REJECT
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best_score = ranked[0]
        if best_score < self.tau:
            return REJECT
        if len(ranked) > 1:
            second_score = ranked[1][1]
            if (best_score - second_score) < self.margin:
                return REJECT
        # one-missing prior (linker-design.md §6.6): with a single candidate the
        # margin test is vacuous by construction above — tau still gates it.
        return best_id
