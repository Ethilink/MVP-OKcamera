# RF-DETR detector

Runtime contract for the detection component
(`components/detector/detector.py`, `components/detector/_rfdetr_postprocess.py`),
wrapping the pre-trained RF-DETR instance-segmentation ONNX export.

## Detector

`Detector` owns the ONNX session and the `preprocess -> session.run ->
decode_predictions` pipeline. It takes a single BGR image (e.g. from
`cv2.imread` or `Frame.load_image()`) and returns `sv.Detections` in that
image's own pixel coordinate space — callers never see the model's internal
768x768 input space or raw logits.

The input and required output names (`dets`, `labels`, `masks`) are validated
once when the ONNX session loads. Prediction explicitly requests those outputs
in contract order; no graph metadata is re-read per frame.

## RF-DETR ONNX contract

`_rfdetr_postprocess.py` is internal to `components/detector/` — nothing
outside this package should import it directly; go through `Detector` instead.

`preprocess`: BGR HWC uint8 (any H, W) -> float32 (1,3,768,768) NCHW, RGB,
ImageNet-normalized. RF-DETR was trained on square 768x768 RGB input
normalized with standard ImageNet mean/std.

`decode_predictions` takes the raw ONNX outputs — `dets` (1,300,4) cxcywh,
`labels` (1,300,2) raw logits, `masks` (1,300,192,192) raw mask logits — and
turns them into `sv.Detections` in the *original* image's pixel coordinates.
Order of operations, and why each step is there:

1. Select channel 0, the only trained detection channel. Channel 1 is removed
   before ranking so it cannot consume the top-k budget.
2. Apply sigmoid and take the top-`top_k` query indices by probability.
3. Gather `dets[query_idx]` and `masks[query_idx]` using that SAME
   `query_idx` — the critical alignment step; get this wrong and a box
   silently pairs with the wrong query's mask.
4. Keep only `prob >= confidence_threshold`.
5. cxcywh -> xyxy in normalized [0,1] space, then scale directly to the
   *original* image's pixel width/height (NOT the square 768x768 model
   input — the normalized box coords were relative to that square input,
   but RF-DETR's own postprocessing rescales them directly against the
   target/original image size).
6. Resize each surviving query's 192x192 mask logits (bilinear) to
   `(image_height, image_width)`, then threshold `> 0` (equivalent to
   sigmoid-then->0.5, since sigmoid is monotonic and sigmoid(0) == 0.5).
7. Build the `sv.Detections` (empty-but-valid when nothing survives).

This contract (tensor shapes, the two-class label layout, the cxcywh
normalization target) was verified against the actual ONNX export rather
than assumed from RF-DETR's PyTorch-side docs, since ONNX exports can drop
or reorder postprocessing that the PyTorch model applies internally.

## Apple Silicon execution

`load_tracker()` uses ONNX Runtime's CoreML execution provider when available,
with MLProgram format, all compute units, static input shapes, a persistent
compiled-model cache, and CPU fallback. The current M3 Max measurement is
approximately 0.33 s/frame versus 0.84 s/frame on CPU for a 1920×1080 input.
