"""Pre/post-processing helpers for the RF-DETR ONNX model.

Internal to `components/` — nothing outside this package should import this
module directly; go through `Detector` instead.

See `docs/plan-first-detections.md` ("RF-DETR ONNX contract") for the verified
contract this implements.
"""

import cv2
import numpy as np
import supervision as sv

_INPUT_SIZE = 768
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 (any H,W) -> float32 (1,3,768,768) NCHW, RGB, ImageNet-normalized."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD

    chw = normalized.transpose(2, 0, 1)
    return chw[np.newaxis, ...].astype(np.float32)


def decode_predictions(
    dets: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    image_width: int,
    image_height: int,
    confidence_threshold: float,
    top_k: int,
) -> sv.Detections:
    """(1,300,4) cxcywh + (1,300,2) raw logits + (1,300,192,192) raw mask
    logits -> `sv.Detections` in the *original* image's pixel coordinates.

    Order of operations (see "RF-DETR ONNX contract" in
    `docs/plan-first-detections.md` for why each step is here):
    1. `sigmoid(labels)` -> per-(query, class) probability.
    2. Flatten the (query, class) grid into one length-`300*num_classes` axis.
    3. Take the top-`top_k` flat indices by probability, descending. Recover
       `query_idx = flat_idx // num_classes`, `class_idx = flat_idx % num_classes`.
    4. Gather `dets[query_idx]` and `masks[query_idx]` using that SAME
       `query_idx` — the critical alignment step; get this wrong and a box
       silently pairs with the wrong query's mask.
    5. Keep only rows where `class_idx == 0` (the real/trained detection
       channel — `class_idx == 1` is an untrained channel that must never be
       treated as a real detection, however high its raw score happens to be).
    6. Among the survivors, keep only `prob >= confidence_threshold`.
    7. cxcywh -> xyxy in normalized [0,1] space, then scale directly to the
       *original* image's pixel width/height (NOT the square 768x768 model
       input — the normalized box coords were relative to that square input,
       but RF-DETR's own postprocessing rescales them directly against the
       target/original image size).
    8. Resize each surviving query's 192x192 mask logits (bilinear) to
       `(image_height, image_width)`, then threshold `> 0` (equivalent to
       sigmoid-then->0.5, since sigmoid is monotonic and sigmoid(0) == 0.5).
    9. Build the `sv.Detections` (empty-but-valid when nothing survives).
    """
    dets = np.squeeze(dets, axis=0)  # (300, 4)
    labels = np.squeeze(labels, axis=0)  # (300, num_classes)
    masks = np.squeeze(masks, axis=0)  # (300, 192, 192)

    num_classes = labels.shape[1]
    prob = 1.0 / (1.0 + np.exp(-labels))
    flat_prob = prob.reshape(-1)

    k = min(top_k, flat_prob.size)
    top_flat_idx = np.argsort(-flat_prob)[:k]

    query_idx = top_flat_idx // num_classes
    class_idx = top_flat_idx % num_classes
    scores = flat_prob[top_flat_idx]

    keep = (class_idx == 0) & (scores >= confidence_threshold)
    query_idx = query_idx[keep]
    scores = scores[keep].astype(np.float32)
    n = len(query_idx)

    # Shared-index gather: both `dets` and `masks` are indexed by the exact
    # same `query_idx` array, so box i and mask i always come from the same
    # query.
    selected_dets = dets[query_idx]  # (n, 4) cxcywh, normalized to 768x768 input
    cx, cy, w, h = (
        selected_dets[:, 0],
        selected_dets[:, 1],
        selected_dets[:, 2],
        selected_dets[:, 3],
    )
    x1 = (cx - w / 2.0) * image_width
    y1 = (cy - h / 2.0) * image_height
    x2 = (cx + w / 2.0) * image_width
    y2 = (cy + h / 2.0) * image_height
    xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    selected_mask_logits = masks[query_idx]  # (n, 192, 192)
    resized_masks = np.empty((n, image_height, image_width), dtype=bool)
    for i in range(n):
        resized = cv2.resize(
            selected_mask_logits[i],
            (image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_masks[i] = resized > 0

    class_id = np.zeros(n, dtype=int)
    class_names = np.full(n, "surgical_instrument")

    return sv.Detections(
        xyxy=xyxy,
        mask=resized_masks,
        confidence=scores,
        class_id=class_id,
        data={"class_name": class_names},
    )
