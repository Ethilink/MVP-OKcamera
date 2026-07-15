"""Appearance embedder for Deep-OC-SORT's embedding-fused association.

Upstream uses FastReID/torchreid checkpoints trained specifically for
MOT17/MOT20/DanceTrack person re-identification, fetched from external
Google Drive links and hard-coded to `.cuda()`. That's not available here, so
this uses a generic ImageNet-pretrained torchvision backbone (frozen,
global-average-pooled, L2-normalized) as a CPU/MPS-friendly stand-in --
a weaker appearance cue than a real ReID model, but enough to demonstrate
the algorithm's embedding-fused association.
"""

import cv2
import numpy as np
import torch
import torchvision


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TorchvisionEmbedder:
    def __init__(
        self,
        device: torch.device | None = None,
        crop_size=(128, 256),
        batch_size=64,
        mask_crop: bool = False,
    ):
        self.device = device or select_device()
        self.crop_size = crop_size  # (width, height)
        self.batch_size = batch_size
        self.mask_crop = mask_crop

        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        self.model = torch.nn.Sequential(backbone.features, torch.nn.AdaptiveAvgPool2d(1)).to(self.device).eval()

        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, crop_size[1], crop_size[0], device=self.device)
            self.embedding_dim = self.model(dummy).flatten(1).shape[1]

        # the value a background pixel should hold so that, after the same
        # /255 + normalize below, it lands at ~0 instead of registering as a
        # sharp, out-of-distribution edge against the object
        mean_rgb = self._mean.reshape(3).cpu().numpy() * 255
        self._fill_bgr = mean_rgb[::-1].astype(np.uint8)

    def _suppress_background(self, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Replace pixels outside `mask` with a neutral fill, in the crop's
        original resolution -- masking before the resize below lets bilinear
        interpolation feather the boundary instead of leaving a hard edge.
        """
        if mask.sum() == 0:
            return crop  # degenerate/empty mask -- keep the raw crop rather than blanking it entirely
        out = crop.copy()
        out[~mask] = self._fill_bgr
        return out

    @torch.no_grad()
    def compute_embedding(
        self,
        frame_bgr: np.ndarray,
        bboxes_xyxy: np.ndarray,
        masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """`masks`, if given, is one full-frame boolean array per detection
        (`sv.Detections.mask`'s shape) -- only used when `mask_crop=True`.
        """
        if len(bboxes_xyxy) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        crops = []
        for i, (x1, y1, x2, y2) in enumerate(np.round(bboxes_xyxy).astype(int)):
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8)
            elif self.mask_crop and masks is not None:
                crop = self._suppress_background(crop, masks[i, y1:y2, x1:x2])
            crop = cv2.cvtColor(cv2.resize(crop, self.crop_size), cv2.COLOR_BGR2RGB)
            crops.append(crop)

        batch = torch.from_numpy(np.stack(crops)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - self._mean) / self._std

        embeddings = []
        for start in range(0, batch.shape[0], self.batch_size):
            feats = self.model(batch[start : start + self.batch_size]).flatten(1)
            embeddings.append(feats)
        embeddings = torch.nn.functional.normalize(torch.cat(embeddings, dim=0), dim=-1)
        return embeddings.cpu().numpy().astype(np.float32)
