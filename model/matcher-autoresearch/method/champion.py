#!/usr/bin/env python
"""
MUTABLE — the CHAMPION BASELINE (program.md §5):

  DINOv2-B (ViT-B/14) CLS embedding on masked crops (mild dilation, neutral
  fill), fused with a MODERATE weight of mask-size cues (z-scored log area +
  log major-axis-length — stage-1's `fuse.py` finding: size resolves the
  instrument1<->2 near-twin but >=2x weight re-collides other pairs), TOP-3-MEAN
  gallery aggregation (per-frame score = mean of the 3 highest-cosine gallery
  views), and MULTI-FRAME WINDOW VOTING (per-frame scores aggregated across the
  query's return-window, discounted by how often each candidate was any single
  frame's own top pick — this is the concrete form of linker-design.md §6.3's
  "argmax required stable across the window frames").

Implements the frozen build_gallery/score/accept interface
(frozen/interface.py, mirrors linker-design.md §6). Stateless across calls
(all state lives in the Gallery objects the eval passes back in) except for the
shared embedder singleton.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.metrics.pairwise import cosine_similarity

from frozen.interface import REJECT, Gallery

from .embedder import get_embedder
from .size_features import size_feature

FAMILY = (
    "baseline: DINOv2-B CLS + masked-crop + moderate size-fusion "
    "+ top3-mean gallery aggregation + multi-frame window voting"
)


class ChampionGallery(Gallery):
    __slots__ = ("views", "identity")

    def __init__(self, views: np.ndarray, identity: str | None):
        self.views = views  # (n_views, D) fused feature matrix
        self.identity = identity


def _apply_mask(image: np.ndarray, mask: np.ndarray, dilate_px: int) -> Image.Image:
    m = mask.astype(bool)
    if dilate_px > 0:
        struct = np.ones((3, 3), dtype=bool)
        m = binary_dilation(m, structure=struct, iterations=dilate_px)
    masked = (image * m[:, :, None].astype(image.dtype))
    return Image.fromarray(masked)


class ChampionMethod:
    """CHAMPION BASELINE. See module docstring; hyperparameters are chosen via
    CV in runs/baseline/run_eval.py and then frozen into champion/PARAMS.md."""

    family = FAMILY

    def __init__(
        self,
        size_alpha: float = 0.75,
        top_k: int = 3,
        tau: float = 0.60,
        margin: float = 0.05,
        mask_dilate_px: int = 2,
        model_id: str = "facebook/dinov2-base",
    ):
        self.size_alpha = size_alpha
        self.top_k = top_k
        self.tau = tau
        self.margin = margin
        self.mask_dilate_px = mask_dilate_px
        self._embedder = get_embedder(model_id)

    # -- shared feature extraction -------------------------------------------
    def _embed_fused(self, crops: list, masks: list) -> np.ndarray:
        imgs = [_apply_mask(c, m, self.mask_dilate_px) for c, m in zip(crops, masks)]
        E = self._embedder.embed(imgs)
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        S = np.stack([size_feature(m) for m in masks], axis=0)
        return np.concatenate([En, self.size_alpha * S], axis=1)

    # -- frozen interface -----------------------------------------------------
    def build_gallery(self, crops: list, masks: list, meta: dict) -> ChampionGallery:
        views = self._embed_fused(crops, masks)
        return ChampionGallery(views=views, identity=meta.get("identity"))

    def score(self, query_crops: list, query_masks: list, meta: dict,
              galleries: dict[str, ChampionGallery]) -> dict[str, float]:
        if not galleries:
            return {}
        Fq = self._embed_fused(query_crops, query_masks)  # (n_frames, D)
        candidate_ids = list(galleries.keys())
        n_frames, n_cand = Fq.shape[0], len(candidate_ids)
        per_frame = np.zeros((n_frames, n_cand), dtype=np.float64)
        for j, cid in enumerate(candidate_ids):
            views = galleries[cid].views
            sims = cosine_similarity(Fq, views)          # (n_frames, n_views)
            k = min(self.top_k, sims.shape[1])
            top_k_vals = np.sort(sims, axis=1)[:, -k:]
            per_frame[:, j] = top_k_vals.mean(axis=1)

        # multi-frame window voting: discount each candidate's mean score by
        # how often it was any single frame's OWN top pick (argmax stability).
        frame_argmax = per_frame.argmax(axis=1)
        mean_score = per_frame.mean(axis=0)
        consistency = np.array([(frame_argmax == j).mean() for j in range(n_cand)])
        fused = mean_score * consistency
        return {cid: float(fused[j]) for j, cid in enumerate(candidate_ids)}

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
