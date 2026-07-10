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
Open http://127.0.0.1:8000/. Recording mode adds `--capture-fps` (default 30) —
see below before touching it.

### Finding the camera index (Camo)

Camo only registers while the Camo app is running, and its OpenCV index is **not
stable** — it shifts when the set of connected cameras changes (the dashboard
was seen at index 3 with 4 devices present, then at index 0 with 3). Do not
hard-code it. Probe and eyeball each run:

```bash
uv run python -m scripts.find_camera        # prints index -> shape + a preview JPEG per index
```

Multiple devices report 1080p (Camo *and* iPhone Continuity), so pick the index
whose preview shows the phone's view of the instruments, then pass it as
`--camera-index`.

### Recording mode

```bash
uv run python -m backend.main \
    --weights ../../model/weights/checkpoint_best_regular.onnx \
    --model-version rfdetr-2026-07-07 \
    --camera-index <n>            # from find_camera above
```

Record → name the entry → SPACE marks keyframes (each keeps the live detection
shown at mark time) → Stop → the finished MP4 plus one JPEG + annotations per
marked keyframe are written **synchronously** to a video-project entry under the
Settings output path. No offline post-pass over every frame (ADR-0002), so Stop
returns in well under a second regardless of clip length. **Capture rate:**
camera-limited to **30 fps** (Camo does not deliver 1080p60 — measured; see
[`../docs/RECORDING.md`](../docs/RECORDING.md) §FPS).

### FPS spike

```bash
uv run python -m scripts.spike_fps --camera-index <n> --fps 60   # stop the dashboard first (frees the camera)
```

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
