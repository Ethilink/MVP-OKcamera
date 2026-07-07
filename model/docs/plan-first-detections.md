# Plan: first detections (RF-DETR)

Design rationale for the first detection component (`components/detector.py`,
`components/_rfdetr_postprocess.py`), wrapping the pre-trained RF-DETR
instance-segmentation ONNX export.

## Phase 2: detector.py

`Detector` owns the ONNX session and the `preprocess -> session.run ->
decode_predictions` pipeline. It takes a single BGR image (e.g. from
`cv2.imread` or `Frame.load_image()`) and returns `sv.Detections` in that
image's own pixel coordinate space — callers never see the model's internal
768x768 input space or raw logits.

Output tensors are looked up by name (`dets`, `labels`, `masks`) rather than
by position, since ONNX does not guarantee output order across export runs.

## RF-DETR ONNX contract

`_rfdetr_postprocess.py` is internal to `components/` — nothing outside this
package should import it directly; go through `Detector` instead.

`preprocess`: BGR HWC uint8 (any H, W) -> float32 (1,3,768,768) NCHW, RGB,
ImageNet-normalized. RF-DETR was trained on square 768x768 RGB input
normalized with standard ImageNet mean/std.

`decode_predictions` takes the raw ONNX outputs — `dets` (1,300,4) cxcywh,
`labels` (1,300,2) raw logits, `masks` (1,300,192,192) raw mask logits — and
turns them into `sv.Detections` in the *original* image's pixel coordinates.
Order of operations, and why each step is there:

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

This contract (tensor shapes, the two-class label layout, the cxcywh
normalization target) was verified against the actual ONNX export rather
than assumed from RF-DETR's PyTorch-side docs, since ONNX exports can drop
or reorder postprocessing that the PyTorch model applies internally.
