"""Appearance embedder for Deep-OC-SORT's embedding-fused association.

Upstream uses FastReID/torchreid checkpoints trained specifically for
MOT17/MOT20/DanceTrack person re-identification, fetched from external
Google Drive links and hard-coded to `.cuda()`. That's not available here,
and it wouldn't be the right tool anyway: those checkpoints (like gta-link's
OSNet) are supervised person-ReID models, tuned to discriminate *people*
they've never seen -- not surgical instruments.

Instead this uses `facebook/dinov2-small` (ViT-S/14, 384-dim), a general-
purpose *self-supervised* backbone (frozen, CLS-token output, L2-normalized).
DINOv2 was never trained to classify or re-identify anything in particular,
which is exactly what makes it transfer well to fine-grained instance
discrimination of never-seen-at-pretraining object categories -- unlike an
ImageNet-classification backbone (the previous `TorchvisionEmbedder`
MobileNetV3-Small stand-in), whose features are optimized to *collapse*
intra-class appearance differences rather than preserve them.

Loaded via `transformers` (`AutoModel`/`AutoImageProcessor`), not
`torch.hub` -- pinned, cacheable through the project's normal dependency
management instead of a GitHub-code download at runtime.
"""

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Dinov2Embedder:
    """DINOv2-S (ViT-S/14) appearance embedder.

    Crop size defaults to 224x224 (16x14 by 16x14) -- DINOv2's standard
    pretraining resolution, and a clean multiple of the patch size (14),
    which ViT-S/14 *requires* (crop height/width must both be divisible by
    14; the old (128, 256) MobileNet crop size was not and won't work
    here). A taller crop closer to the old 1:2 aspect ratio (e.g. 126x252
    = 9x14 by 18x14) would better match an elongated instrument's silhouette,
    but 224x224 is the resolution the backbone was actually pretrained at --
    preferring the well-tested default over guessing at an aspect ratio the
    model has never seen. Revisit with the 1:2 crop if step 5's calibration
    shows the embedder struggling on elongated instruments.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        model_name: str = "facebook/dinov2-small",
        crop_size=(224, 224),
        # DINOv2 ViT-S/14 is heavier than MobileNetV3-Small -- the old
        # default of 64 crops/batch risked stalling the loop on CPU/MPS, so
        # this defaults lower. Raise it back up if profiling shows headroom.
        batch_size=16,
        mask_crop: bool = False,
    ):
        self.device = device or select_device()
        self.crop_size = crop_size  # (width, height)
        if self.crop_size[0] % 14 != 0 or self.crop_size[1] % 14 != 0:
            raise ValueError(f"crop_size {self.crop_size} must be divisible by DINOv2's 14x14 patch size")
        self.batch_size = batch_size
        self.mask_crop = mask_crop

        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        # Pull normalization stats from the model's own image processor
        # rather than assuming ImageNet's -- they happen to coincide
        # (verified: both (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)),
        # but sourcing them from the processor keeps that an observed fact
        # rather than a baked-in assumption if the model is ever swapped
        # (e.g. for `dinov2-base` per the plan's fallback option).
        processor = AutoImageProcessor.from_pretrained(model_name)
        self._mean = torch.tensor(processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(processor.image_std, device=self.device).view(1, 3, 1, 1)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, crop_size[1], crop_size[0], device=self.device)
            self.embedding_dim = self.model(pixel_values=dummy).pooler_output.shape[1]

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
            feats = self.model(pixel_values=batch[start : start + self.batch_size]).pooler_output
            embeddings.append(feats)
        embeddings = torch.nn.functional.normalize(torch.cat(embeddings, dim=0), dim=-1)
        return embeddings.cpu().numpy().astype(np.float32)
