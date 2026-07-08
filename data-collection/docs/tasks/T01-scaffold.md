# T01 — Scaffold + shared test fixtures

status: done
depends-on: —
blocks: T02, T03, T04, T05, T06
spec: [DASHBOARD.md](../DASHBOARD.md) §Stack, §Model / config

## Goal

An empty-but-running `data-collection/dashboard/` uv project that every other
task can drop code into: FastAPI app boots, `orc_model` imports, pytest runs,
and the fake-detector/fake-camera fixtures that T02/T03/T05's blind-TDD tests
depend on exist.

## In scope

- `data-collection/dashboard/` layout: `backend/`, `static/`, `tests/`,
  `pyproject.toml` (uv), `README.md` (run instructions).
- Dependencies: `fastapi`, `uvicorn`, `opencv-python`, `supervision`,
  `pycocotools`, `numpy`, path dep on `../model` (`orc_model`), dev: `pytest`,
  `httpx` (TestClient).
- `backend/main.py`: app factory + startup config — **weights path and
  `model_version` tag come from CLI args/env at startup, not the UI**
  (spec §Model / config; tag format `rfdetr-YYYY-MM-DD`).
- Placeholder `static/index.html` ("dashboard here") served at `GET /`.
- `tests/conftest.py` fixtures (the shared test vocabulary):
  - `fake_frame(w, h)` → deterministic BGR `np.uint8` array (e.g. gradient).
  - `fake_dets(boxes, masks=..., confidences=..., size=(1920,1080))` →
    real `sv.Detections` with `xyxy` float32, bool masks at work size,
    `class_id` 0. Must also support the empty case (`len == 0`, `mask is None`).
  - `FakeDetector` — has a mutable `confidence_threshold` attribute and a
    `predict(frame)` you can program with a queue of `sv.Detections`; records
    the frames and threshold values it saw.
  - `FakeCapture` — `VideoCapture` stand-in: programmable frame queue,
    records every `cap.set(...)` call, can simulate `read() → (False, None)`.

## Out of scope

Any real endpoint logic, dataset writing, camera code. Stubs only.

## Public interface (frozen)

```python
# backend/main.py
def create_app(detector, writer_factory, capture) -> FastAPI   # T05 fills the routes
# CLI: uv run python -m backend.main --weights <path.onnx> --model-version rfdetr-2026-07-07 [--camera-index N]
```

## Acceptance criteria

- **AC1** `uv run pytest` passes from `data-collection/dashboard/` (≥1 trivial test).
- **AC2** `uv run python -c "import orc_model"` succeeds (path dep resolves).
- **AC3** App boots and `GET /` returns the placeholder HTML (TestClient).
- **AC4** Starting without `--weights`/`--model-version` fails loudly with a
  clear message (no silent default tag).
- **AC5** `fake_dets` round-trips through `len()`, `.xyxy`, `.mask`,
  `.confidence` exactly like real detector output; empty variant has `mask is None`.
- **AC6** `FakeCapture` records `set()` calls and can be told to fail reads.
- **AC7** Contract test guarding fake-vs-real drift: one test marked
  `@pytest.mark.skipif` (no `model/weights/*.onnx` present) that runs the real
  `Detector` on one sample image and asserts its `sv.Detections` has the exact
  shape `FakeDetector` mimics — xyxy `float32`, bool full-frame masks or `None`,
  `confidence float32`, `class_id 0`. Skipped on weight-less machines/CI, runs
  on Bram's Mac.

## Log

- 2026-07-07 — Built `data-collection/dashboard/` (uv). `uv sync` resolves
  `orc-model` path dep + `trackers` git dep; `uv run pytest` = 9 passed. All AC1–AC7
  green. **AC7 caught a real fake-vs-real drift:** the RF-DETR export returns
  `dets.mask` as an empty `(0, H, W)` bool array on an empty frame — **not `None`**,
  contradicting the DASHBOARD.md mapping comment and this file's original AC5
  wording. Consumers all guard with `if len(dets):` so it never bites, but the
  fake now mimics the real empty shape and AC5/AC7 assert `(0, H, W)`. Flag for
  DASHBOARD.md §mapping: the "`dets.mask is None on an empty frame`" comment is
  inaccurate (behaviorally irrelevant under the len guard).
