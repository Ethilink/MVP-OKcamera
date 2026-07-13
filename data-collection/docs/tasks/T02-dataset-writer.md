# T02 — COCO-VID DatasetWriter

status: done
depends-on: T01
blocks: T05
spec: [DASHBOARD.md](../DASHBOARD.md) §`sv.Detections → COCO-VID` mapping, §Dataset layout & config · [IMPORT_FORMAT.md](../IMPORT_FORMAT.md)

## Goal

A pure, camera-free, FastAPI-free module that turns one `(frame, dets,
threshold)` snapshot into a correct on-disk COCO-VID dataset increment — the
crux of the whole tool, and the most test-friendly piece.

## In scope

`backend/dataset_writer.py` + `tests/test_dataset_writer.py`. Everything in the
spec's mapping section: clipping (single-resolution, no scaling), RLE, box-only
fallback, ID counters, atomic rewrite, lazy folder creation.

## Out of scope

Threading/locking (caller serializes — see spec `dataset_lock` rule), HTTP,
camera, rendering. The writer is **not thread-safe by design**.

## Public interface (frozen)

```python
# backend/dataset_writer.py
@dataclass
class FlagResult:
    image_id: int
    n_annotations: int
    file_name: str            # "frame_00001.jpg"

class DatasetWriter:
    def __init__(self, output_path: Path, dataset_name: str, model_version: str):
        """ValueError: dataset_name not a single path component or starts with '.'
           FileExistsError: <output_path>/<dataset_name> already exists (collision → reject).
           Creates NO folders (lazy — first flag does)."""
    def flag(self, frame: np.ndarray, dets: sv.Detections, threshold: float) -> FlagResult: ...
    @property
    def n_flagged(self) -> int: ...
    @property
    def dataset_dir(self) -> Path: ...
```

## Acceptance criteria

Follow the spec's mapping code verbatim; these pin the behavior:

- **AC1** Constructor: bad names (`a/b`, `.hidden`, empty) → `ValueError`;
  existing folder → `FileExistsError`; valid name → nothing on disk yet.
- **AC2** First `flag()` creates `<out>/<name>/images/` and
  `annotations/annotations.json`; `n` starts at 1.
- **AC3** Saved `frame_00001.jpg` has the frame's dimensions and is the
  raw frame (no overlay) — decode and compare shape + sampled pixels.
- **AC4** Image record: `width`/`height` = frame dims, `review_status: "pending"`,
  `id == n`.
- **AC5** No scaling: with dets and masks in the frame's own coords (e.g. 1920×1080
  masks on a 1920×1080 `frame`), bboxes are written unscaled (±1e-6) and
  `segmentation.size == [Hf, Wf]`; decoding the RLE reproduces the input mask exactly.
- **AC6** Clipping happens against the frame (no scaling step): an off-frame box is
  clamped to `[0,W]/[0,H]`; a fully out-of-frame box
  (`bw<=0 or bh<=0`) produces **no annotation**.
- **AC7** All-zero mask → annotation has **no** `segmentation` key and
  `area == int(bw*bh)` (box-only SAM3-fill path).
- **AC8** `annotations.json` loads with stdlib `json`; no numpy scalars anywhere
  (bbox entries are `float`, ids `int`).
- **AC9** Counter behavior across flags 3-dets → 0-dets → 2-dets:
  `image_id` 1,2,3; `ann_id` 1..5 continuous; empty frame gets an image record
  with zero annotations.
- **AC10** Rewrite is atomic: after every flag the file parses, and no
  `*.tmp` file remains (`os.replace` pattern).
- **AC11** Per-annotation provenance: `confidence` (float), `model_version`
  (constructor arg), `confidence_threshold` (the `flag()` arg, untouched).
- **AC12** `categories` is exactly
  `[{"id": 1, "name": "surgical_instrument", "supercategory": ""}]`; `info` has
  `description == dataset_name` and iso8601 `date_created`.
- **AC13** A produced dataset passes `validate()` from
  [`scripts/validate_import.py`](../scripts/validate_import.py) with zero errors.

## Log

- 2026-07-07 — Built `backend/dataset_writer.py` via blind-TDD (SPEC at
  `backend/SPEC-dataset-writer.md`). Test-writer + coder ran blind in parallel
  (coder never saw tests; test-writer mutation-checked its own suite outside the
  repo — 5/5 injected bugs caught). Dual review vs SPEC: Opus approved (0 ❌);
  Codex raised 1 ❌ — `annotations.json.tmp` could linger if `json.dump` raised
  before `os.replace`. Fixed: `_write_annotations` now wraps write+replace in
  `try/except BaseException` → unlink tmp (`missing_ok=True`) + re-raise. Codex
  re-review: resolved, consensus reached. `uv run pytest` = 23/23 for T02 (AC1–AC13),
  32/32 whole suite. Done.
