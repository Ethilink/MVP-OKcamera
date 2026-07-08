# ORC data-collection dashboard

Live capture tool: stream Camo → run the RF-DETR detector → operator snapshots
bad frames → save each still into a COCO-VID dataset (`review_status: "pending"`)
for correction in the separate annotation dashboard. Spec:
[`../docs/DASHBOARD.md`](../docs/DASHBOARD.md).

## Setup

```bash
cd data-collection/dashboard
uv sync            # resolves deps + the ../../model (orc_model) path dep
```

## Run

```bash
uv run python -m backend.main \
    --weights ../../model/weights/checkpoint_best_regular.onnx \
    --model-version rfdetr-2026-07-07 \
    [--camera-index 0]
```

Both `--weights` and `--model-version` are required (no silent default tag).
Open http://127.0.0.1:8000/.

## Test

```bash
uv run pytest
```

The detector-contract test (`tests/test_detector_contract.py`) is skipped unless
`model/weights/*.onnx` is present — it runs the real `Detector` and asserts its
`sv.Detections` shape still matches the test fakes.

## Layout

```
backend/     app factory + startup (main.py); T02 dataset_writer.py, T03 capture.py, …
static/      index.html — the live operator UI (T06)
tests/       conftest.py fixtures (fake_frame, fake_dets, FakeDetector, FakeCapture)
```
