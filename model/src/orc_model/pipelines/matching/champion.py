#!/usr/bin/env python
"""Session re-identification using joint sparse representation.

A query window's masked DINOv2-B embeddings are jointly reconstructed from
the complete frozen-roster gallery dictionary with ``MultiTaskLasso``. Each
identity receives a reconstruction-similarity score multiplied by the
Sparsity Concentration Index (SCI), which measures whether the sparse code is
concentrated on that identity instead of spread across the roster.

The selected parameters and guarded benchmark numbers are retained in
``model/docs/linker-design.md``. Production must score against the complete
roster: SCI is relative and loses its open-set meaning when callers reduce the
dictionary to only the currently Missing identities.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.linear_model import MultiTaskLasso

from .embedder import get_embedder
from .interface import REJECT, Gallery
from .size_features import size_feature

_log = logging.getLogger("orc_model.matching")

FAMILY = (
    "sparse-representation-based classification (SRC): joint dictionary "
    "reconstruction over concatenated candidate galleries via L1-penalized "
    "multi-task (window-joint) least squares, open-set rejection via the "
    "Sparsity Concentration Index (SCI)"
)

# Winners of the recorded 630-point CV sweep; see model/docs/linker-design.md.
DEFAULT_ALPHA = 0.0003
DEFAULT_SIZE_ALPHA = 0.5
DEFAULT_TAU = 0.30
DEFAULT_MARGIN = 0.02
DEFAULT_MASK_DILATE_PX = 2
DEFAULT_MAX_ITER = 2000

# Absolute fallback for a genuinely one-gallery roster. Calibrated on the
# 8x15 captured instrument stills: 0.60 gave 93.5% genuine accept and 98.6%
# foreign reject. Normal demo decisions retain the complete eight-gallery
# roster and therefore use SCI instead.
DEFAULT_COS_TAU = 0.60


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
    """SRC via joint dictionary reconstruction and SCI open-set rejection."""

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
        cos_tau: float = DEFAULT_COS_TAU,
    ):
        """``cos_tau`` is the only parameter not selected by the SRC sweep.

        Why it exists: SCI, this method's open-set rejection signal, is a
        RELATIVE statistic — "how much of the sparse code's L1 mass sits on this
        candidate versus the others". At K=1 there are no others: the formula
        `(K*cand_l1/total_l1 - 1)/(K-1)` is literally 0/0, and it used to be
        hardcoded to 1.0 ("vacuous"). That left `score = sim = 1/(1+residual)`
        as the whole decision, whose FLOOR for a ~unit-norm query is ~0.47 —
        a gallery reconstructing NOTHING still scores 0.47, above `tau`=0.30.
        So the previous K=1 branch could not reject anything: all seven foreign
        instruments force-linked into the sole candidate in the smoke test.

        Production now preserves the full frozen-roster dictionary, so this
        absolute nearest-view cosine gate is only a fail-closed fallback for a
        deployment whose entire usable roster really contains one gallery.
        """
        self.alpha = alpha
        self.size_alpha = size_alpha
        self.tau = tau
        self.margin = margin
        self.cos_tau = cos_tau
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

        # Absolute (K-free) appearance similarity, for the K=1 gate below.
        # D_norm's columns are already unit-norm, so this is a plain cosine of
        # each query view against every atom. Costs one matmul on vectors we
        # already have — no extra embed.
        Yn = Y / np.linalg.norm(Y, axis=0, keepdims=True).clip(min=1e-8)
        cos_all = Yn.T @ D_norm  # (n_frames, N_atoms)

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
                # K == 1: SCI is 0/0 — there is no "other candidate" to
                # concentrate against, so it carries no information and used to
                # be hardcoded 1.0, leaving `sim` (floor ~0.47) as the only
                # gate and `tau`=0.30 unreachable. Gate on an ABSOLUTE cosine
                # instead: it needs no rival candidates to mean something.
                # Binary by design — with one candidate there is nothing to
                # RANK, only an admit/reject to make, and passing the gate
                # leaves the score exactly what tau/margin were tuned against.
                # See __init__'s `cos_tau` docstring for the full story.
                cos = float(cos_all[:, idx].max(axis=1).mean())  # window-mean of per-view best
                sci = 1.0 if cos >= self.cos_tau else 0.0
                # Log the absolute value so replay validation can recalibrate
                # cos_tau without changing the score contract.
                _log.debug(
                    "K=1 gate: candidate=%s cos=%.4f cos_tau=%.2f -> %s (sim=%.4f)",
                    cid, cos, self.cos_tau, "admit" if sci else "REJECT", sim,
                )
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
        # A one-gallery roster has no second candidate, so its margin is
        # vacuous; the absolute cosine fallback in score() still gates it.
        return best_id
