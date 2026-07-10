"""Shared sv.Detections -> COCO annotation helper.

Extracted verbatim from ``DatasetWriter.flag``'s per-detection block so image
mode and video mode share one serialization path: clip to bounds, drop
degenerate boxes, RLE-encode the mask (empty mask -> box-only), cast np
scalars with float()/int(). Callers merge in their own per-mode ``extra``
fields (image mode: confidence/model_version/confidence_threshold; video
mode: video_id/track_id) after the shared geometry.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

CATEGORIES = [{"id": 1, "name": "surgical_instrument", "supercategory": ""}]


def build_annotation(dets, i, W, H, *, ann_id: int, image_id: int, extra: dict | None = None) -> dict | None:
    x1, y1, x2, y2 = dets.xyxy[i]
    x1 = min(max(float(x1), 0.0), W)
    x2 = min(max(float(x2), 0.0), W)
    y1 = min(max(float(y1), 0.0), H)
    y2 = min(max(float(y2), 0.0), H)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    bbox = [x1, y1, bw, bh]

    rle = mask_utils.encode(np.asfortranarray(dets.mask[i].astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    mask_area = int(mask_utils.area(rle))

    ann = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
        "iscrowd": 0,
    }
    if mask_area > 0:
        ann["area"] = mask_area
        ann["segmentation"] = {"size": rle["size"], "counts": rle["counts"]}
    else:
        ann["area"] = int(bw * bh)

    if extra is not None:
        ann.update(extra)

    return ann
