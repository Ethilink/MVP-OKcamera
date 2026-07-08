# TR3 — Shared COCO helper + `VideoEntryWriter`

status: done
depends-on: —
blocks: TR4, TR5
spec: [RECORDING.md](../RECORDING.md) §Post-pass (steps 1–2), §Post-pass specifics · [IMPORT_FORMAT_VIDEO.md](../IMPORT_FORMAT_VIDEO.md) §§2–5

## Goal

Two things, tightly linked. **(1)** Extract the per-detection
`sv.Detections → COCO` logic that currently lives inline in `DatasetWriter.flag`
into a shared helper, and re-point `DatasetWriter` at it — so "keyframes look
exactly like image-mode output" is guaranteed by *one code path*, not a copy that
drifts. **(2)** Build `VideoEntryWriter`, which turns a finished recording (MP4
already on disk) + a keyframe list + per-frame detections into the **four
artifacts** of the video-project import contract: `annotations.json` (COCO-VID,
keyframes only), keyframe JPEGs, `selected_frames.json`, and the all-frames
`full_frame_detections.json` sidecar.

## In scope

- `backend/coco.py` — **new**: the shared `build_annotation` helper +
  `CATEGORIES` constant.
- `backend/dataset_writer.py` — **edit**: replace the inline per-detection block
  in `flag` with a call to `coco.build_annotation`. Image-mode output must stay
  structurally identical (T02's `tests/test_dataset_writer.py` is the regression
  gate — **keep it green; do not edit it**).
- `backend/video_writer.py` — **new**: `VideoEntryWriter`.
- `tests/test_coco.py`, `tests/test_video_writer.py` — **new** test files.

## Out of scope

- Running the detector / opening the MP4 / iterating frames — TR4 does that and
  feeds decoded frames + detections into `add_frame`. `VideoEntryWriter` never
  decodes video itself; it receives pixels.
- The encoder / probe (TR2), the API (TR5). `VideoEntryWriter` takes a plain
  `video` **dict** (not TR2's `VideoProbe`) so it has no code dependency on TR2.

## Public interface (frozen)

```python
# backend/coco.py
CATEGORIES = [{"id": 1, "name": "surgical_instrument", "supercategory": ""}]

def build_annotation(dets, i, W, H, *, ann_id: int, image_id: int, extra: dict | None = None) -> dict | None:
    """The per-detection sv.Detections[i] → COCO annotation, extracted verbatim
    from DatasetWriter.flag: clip xyxy to [0,W]/[0,H] → drop degenerate box
    (bw<=0 or bh<=0) → RLE-encode the mask (empty mask ⇒ box-only, area = bw*bh,
    no `segmentation` key) → all np scalars cast with float()/int(). Returns the
    annotation dict, or None for a degenerate box. Core keys, in order:
    id, image_id, category_id (==1), bbox, iscrowd (==0), area, [segmentation].
    `extra` is shallow-merged AFTER the core keys — the per-mode fields:
      image mode  -> {"confidence", "model_version", "confidence_threshold"}
      video mode  -> {"video_id", "track_id"}
    Geometry (bbox, area, segmentation, category_id, iscrowd) is therefore
    byte-identical across both modes for the same dets[i]."""
```

```python
# backend/video_writer.py
class VideoEntryWriter:
    def __init__(
        self,
        entry_dir: Path,               # <output_path>/<entry_name>
        entry_name: str,               # project name; MP4 is entry_dir/video/<entry_name>.mp4
        model_version: str,            # sidecar model.version (startup provenance tag)
        *,
        video: dict,                   # {"file_name","width","height","fps","frame_count"}
        keyframes,                     # iterable of 0-based frame numbers (deduped+sorted internally)
        operator_threshold: float,     # keyframe annotations filter: conf >= this
        mining_threshold: float,       # sidecar model.conf_threshold (provenance only)
        model_name: str = "nf-detector",
    ): ...

    def write_selected_frames(self) -> None:
        """Write ONLY annotations/metadata/selected_frames.json (manual_review:
        true, fps + selected_frames_with_time from `video`). Called first by the
        post-pass so a job that dies before finalize still leaves a valid PARTIAL
        project (video + selected_frames.json, no annotations.json). Idempotent."""

    def add_frame(self, frame_number: int, frame: np.ndarray, dets: "sv.Detections") -> None:
        """`dets` are the MINING-threshold detections for this frame. ALWAYS
        appended to the in-memory sidecar as {bbox:[x,y,w,h], score, label} — no
        masks, every frame incl. empty. If frame_number is a keyframe: write its
        JPEG (images/<entry>_f{frame_number:06d}.jpg, quality ~95, straight from
        `frame`) and stage its COCO keyframe records — dets filtered to
        conf >= operator_threshold, each via coco.build_annotation with
        extra={"video_id":1, "track_id":<unique running counter>}."""

    def finalize(self) -> None:
        """Rebuild-from-scratch (never append) all outputs: annotations.json
        (COCO-VID: top-level `video` block with id=1 + source_type="video";
        images[] keyframes only, each with video_id + frame_number;
        annotations[] with video_id + unique track_id; review_status "pending"),
        full_frame_detections.json (schema_version 1, video block copy, model
        block, frames[] for every processed frame incl. empty), the keyframe JPEGs
        (already written by add_frame), and selected_frames.json. Idempotent: a
        second full pass reproduces byte-identical JSON (stable ordering)."""

    @property
    def entry_dir(self) -> Path: ...
```

## Acceptance criteria

No real video decode — feed `add_frame` synthetic frames + `make_fake_dets`
(T01). No detector, no camera.

- **AC1 (helper equivalence — the "one code path" guarantee).** For the same
  `sv.Detections[i]`, `coco.build_annotation`'s geometry keys (`bbox`, `area`,
  `segmentation`, `category_id`, `iscrowd`) equal what image mode produced before
  the refactor: clip, drop-degenerate, RLE, box-only fallback, float/int casts all
  match the spec's mapping (DASHBOARD.md §mapping). Degenerate box → `None`.
- **AC2 (image-mode regression).** `DatasetWriter` still produces the same
  `annotations.json` structure as T02 — T02's full `tests/test_dataset_writer.py`
  passes unchanged, and a direct diff of one annotation dict (image mode) shows
  the image-mode `extra` fields (`confidence`, `model_version`,
  `confidence_threshold`) still present alongside the shared geometry.
- **AC3 (byte-identical keyframe structure — spec AC5).** For one `sv.Detections`
  fed to BOTH a `DatasetWriter.flag` and a `VideoEntryWriter` keyframe, the
  geometry keys of the resulting annotations are equal; the video annotation
  additionally carries `video_id` and a unique `track_id` and no image-mode
  provenance keys, exactly per IMPORT_FORMAT_VIDEO §2.
- **AC4 (folder layout).** After `finalize`, the entry has `annotations/
  annotations.json`, `annotations/metadata/selected_frames.json`,
  `annotations/metadata/full_frame_detections.json`, and one
  `images/<entry>_f{n:06d}.jpg` per keyframe — names matching the §3 filename
  contract exactly.
- **AC5 (keyframes only + fields).** `annotations.json.images[]` contains exactly
  the keyframes (one record each), each with `video_id==1`, the right
  `frame_number`, `width`/`height` == `video` dims, `review_status:"pending"`;
  the top-level `video` block equals the passed dict plus `id:1`,
  `source_type:"video"`.
- **AC6 (unique track_id).** Every annotation across all keyframes has a distinct
  `track_id` (running counter), and every annotation carries `video_id==1` — image
  mode still omits `track_id`.
- **AC7 (operator threshold gate).** Only dets with `confidence >=
  operator_threshold` become keyframe annotations; a keyframe whose dets are all
  below threshold yields an image record with **zero** annotations (a first-class
  empty keyframe, not an error).
- **AC8 (sidecar completeness — spec AC4).** `full_frame_detections.json.frames`
  has one entry per processed frame_number `0..frame_count-1`, empty frames
  included (empty `detections` list); each detection is `{bbox:[x,y,w,h] abs px,
  score, label}` with **no** `segmentation`/mask; `schema_version==1`; `model`
  block = `{name, version==model_version, conf_threshold==mining_threshold}`;
  `video` block copies file_name/width/height/fps/frame_count.
- **AC9 (both files, no dedup — import doc §7).** A keyframe's detections appear
  in BOTH `annotations.json` (filtered at operator threshold) and the sidecar
  (all mining-threshold dets) — assert the keyframe frame_number is present in
  both, not deduplicated away.
- **AC10 (JPEG == pixels — spec AC2).** The JPEG written for a keyframe decodes to
  the exact `frame` array handed to `add_frame` (shape + sampled pixels), at
  `video` dims, no resize.
- **AC11 (selected_frames).** `selected_frames.json` has `manual_review: true`,
  `selected_frames` == the sorted keyframe set == the set of `frame_number`s in
  `annotations.json`, `fps` from `video`, and `selected_frames_with_time` mapping
  each frame to `frame/fps` seconds.
- **AC12 (partial project).** After only `write_selected_frames()` (no
  `finalize`), the entry has `selected_frames.json` but **no** `annotations.json`
  — the "needs detection" partial state.
- **AC13 (idempotent rebuild — spec AC7).** Running a full pass twice (fresh
  writer each time, same inputs) yields byte-identical `annotations.json` and
  `full_frame_detections.json` and the same JPEGs — no appended/duplicated records.

## Log

- 2026-07-08 — Brief created (recording-mode decomposition of RECORDING.md, task
  cut T-R3).
- 2026-07-08 — **Done** (blind-TDD). `backend/coco.py` (shared `build_annotation`
  extracted verbatim from `DatasetWriter.flag`), `backend/dataset_writer.py`
  (flag re-pointed at the helper), `backend/video_writer.py` (`VideoEntryWriter`),
  + `tests/test_coco.py` + `tests/test_video_writer.py` (15 tests, AC1–AC13 each
  covered). Blind cycle: assertion-level red vs Phase-0 stubs → test review CLEAN
  (13/13 ACs mapped) + 3 tightenings applied (AC8 sidecar-carries-non-keyframe-
  dets, AC5 image-record file_name, AC11 direct cross-check) → blind Sonnet coder
  → **Opus review CLEAN, zero blockers**; Codex first-pass read all 3 modules,
  surfaced no blocker, then its CLI bridge stalled before emitting a verdict
  (known caveat) — consensus closed on Opus re-verify + the objective test gate.
  15 green first pass, no fix loop. AC13 byte-identical rebuild guaranteed via a
  fixed `_FIXED_TIMESTAMP` (no `datetime.now()`) + deterministic id/track_id
  ordering. Full suite 112 green (97 + 15); T02's `test_dataset_writer.py`
  regression gate green unchanged (image-mode output structurally identical).
  Followup nit (non-blocking, deferred): `dataset_writer._CATEGORIES` still
  duplicates `coco.CATEGORIES` — geometry path is shared (the AC1/AC3 requirement);
  fold the constant too if we ever touch that file.
