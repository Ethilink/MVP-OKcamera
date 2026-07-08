# T05 — FastAPI layer

status: done
depends-on: T02, T03, T04
blocks: T07
spec: [DASHBOARD.md](../DASHBOARD.md) §Runtime (endpoint table), §Threading rules, §Failure-mode guardrails

## Goal

Wire `CaptureLoop`, `DatasetWriter`, and `render` into the six endpoints of the
spec's Runtime table, with the threading rules that bite: `/flag` as plain
`def` on the threadpool, `dataset_lock` serialization, paced MJPEG streaming.

## In scope

`backend/app.py` (+ finishing `backend/main.py` startup) +
`tests/test_api.py` (TestClient + T01 fakes — no real camera/model).

## Out of scope

Frontend files (T06), the internals of writer/capture/render (consume their
frozen interfaces).

## Endpoint contract (frozen — T06 builds against this)

| method · path | request | response |
|---|---|---|
| `GET /` | — | `static/index.html` |
| `GET /stream` | — | `multipart/x-mixed-replace; boundary=frame`, JPEG parts |
| `POST /confidence` | `{"value": 0.0–1.0}` | `200 {"confidence": v}` · `422` outside range |
| `POST /settings` | `{"camera_index"?: int, "output_path": str, "dataset_name": str}` | `200 {"ok": true}` · `409` collision · `422` invalid name |
| `POST /flag` | — | `200 {"image_id", "n_annotations", "n_flagged"}` · `409` no valid target |
| `POST /validate` | — | `200 {"errors": [...], "warnings": [...]}` · `409` no target |
| `GET /status` | — | `200 {"count", "confidence", "dataset_name", "output_path", "n_flagged", "capture_health"}` |

## Acceptance criteria

- [x] **AC1** `/flag` and `/validate` are **plain `def`** endpoints
  (`not inspect.iscoroutinefunction`) — blocking work stays off the event loop.
- [x] **AC2** `/flag` grabs the tuple ref under the `latest` lock, then works on
  that stable snapshot; it serializes on `dataset_lock`: N concurrent flags
  (threads) yield N sequential `image_id`s and a valid `annotations.json` —
  no duplicates, no torn file.
- [x] **AC3** `/flag` with no configured target → `409` with a human-readable
  `detail`. Never a silent no-op.
- [x] **AC4** `/settings` collision (existing folder) → `409`; invalid
  `dataset_name` → `422`; success swaps in a fresh `DatasetWriter` (counters
  reset) and, if `camera_index` present, calls `capture.set_camera`. The
  dataset-mutating part holds `dataset_lock`.
- [x] **AC5** `/confidence` sets `detector.confidence_threshold` live; out-of-range
  rejected by Pydantic (`422`).
- [x] **AC6** `/stream` sends each overlay **once**: with a paused fake capture the
  generator does not repeat frames (pace on `capture.generation`), and it
  yields a new part promptly after a new `Latest` is published.
- [x] **AC7** `/status` returns exactly the spec's fields; `capture_health` reflects
  `CaptureLoop.health` (drive `FakeCapture` read-failures to prove `"stale"`).
- [x] **AC8** `/validate` calls `validate()` from `scripts/validate_import.py`
  **in-process** and returns its typed `(errors, warnings)` — no subprocess.
- [x] **AC9** Provenance end-to-end: a flag stores the threshold **from the
  snapshot**, not the live slider value — move `/confidence` between the fake
  predict and the flag and assert the saved annotation keeps the old value.

## Log

- 2026-07-07 (claude) — Claimed and shipped. Routes live in `backend/app.py`
  (`create_app` re-exported from `backend/main.py` to keep the frozen import path);
  `main.py` now builds the real `Detector` + `CaptureLoop` and starts/stops it
  around `uvicorn.run`. `/validate` imports its validator from
  `scripts/validate_import.py`, a thin shim that loads the canonical
  `docs/scripts/validate_import.py` by path (single source of truth — no copy to
  drift). Tests in `tests/test_api.py` cover AC1–AC9 (real `CaptureLoop`+`FakeCapture`
  for health, deterministic `StubCapture` for stream pacing / provenance). Full
  suite green: **61 passed** (`uv run pytest`). Frozen `/`-serves-index scaffold
  test still green. No interfaces changed; T07 unblocked.
