#!/usr/bin/env python
"""
DINOv2-B (ViT-B/14) embedder, offline-forced, process-wide singleton so
repeated build_gallery/score calls don't reload the model.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

_SINGLETONS: dict[str, "Dinov2Embedder"] = {}


class Dinov2Embedder:
    def __init__(self, model_id: str = "facebook/dinov2-base", device: str | None = None):
        self.model_id = model_id
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def embed(self, images: list, batch_size: int = 8) -> np.ndarray:
        """images: list of PIL.Image (or anything the HF processor accepts).
        Returns (N, D) float32 CLS-token embeddings, NOT normalized."""
        out = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inp = self.proc(images=batch, return_tensors="pt").to(self.device)
            out.append(self.model(**inp).last_hidden_state[:, 0].float().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)


def get_embedder(model_id: str = "facebook/dinov2-base", device: str | None = None) -> Dinov2Embedder:
    key = f"{model_id}::{device}"
    if key not in _SINGLETONS:
        _SINGLETONS[key] = Dinov2Embedder(model_id, device)
    return _SINGLETONS[key]
