# First detections on sample data — Phase 1 (ground-truth loading) + Phase 2 (ONNX prediction)

## Context

`model/` currently only has empty scaffolding (`components/`, `pipelines/` are empty packages) plus real assets added in a prior session: `model/data/` (3 annotated clips — IMG_2047, IMG_2076, IMG_2081) and `model/weights/rf-detr-seg-2xl-v1.onnx` (an RF-DETR instance-segmentation ONNX export, single class `surgical_instrument`). `supervision` and `onnxruntime` are already dependencies.

The immediate goal is to get to a real, visually-verified prediction on a real image — nothing about the eventual Tracker/pipeline/Completeness-Usage goal is in scope here. The user wants this built in two small, notebook-checkpointed phases, each one visually verified before moving to the next: first prove ground-truth data loads and displays correctly, then load the ONNX model and compare its predictions against that same ground truth on the same image.

Two new `src/orc_model/` subpackages get created; no `inference/`/`visualization/`/`scripts/` yet — see rationale in each phase below.

---

## Phase 1 — `src/orc_model/data/` (ground-truth loading)

New files:
```
src/orc_model/data/
├── __init__.py   # re-exports: BBox, InstrumentAnnotation, Frame, Clip, ClipDataset
├── models.py      # BBox, InstrumentAnnotation, Frame, Clip — plain frozen dataclasses
└── dataset.py     # ClipDataset (collection) + directory discovery
```

**Why a new subpackage, not `components/`**: `components/` is for wrappers around pretrained weights (Detector/Tracker/Classifier — things that run inference). Ground-truth loading does no inference; it's I/O/schema code that both the Phase 2 comparison and future Tracker-eval work will reuse. Symmetric top-level package alongside `components/`/`pipelines/`. Named `data/` (not `datasets/`) to match the folder it loads from (`model/data/`).

**No pydantic.** Plain `@dataclass(frozen=True)` throughout — these objects are always constructed by our own code from data that's already been parsed by `pycocotools` (see below), never from untrusted/arbitrary input, so there's no validation boundary here that would justify pydantic's overhead (type coercion, `PrivateAttr` friction for private fields, etc.). Frozen dataclasses give the same immutability with less machinery.

**Use `pycocotools` for COCO parsing, not a hand-rolled schema or `supervision.DetectionDataset.from_coco`.** Verified two things directly before committing to this:
1. `pycocotools` resolves cleanly against this project's Python 3.13 venv (`uv pip install pycocotools` → `pycocotools==2.0.11`, a prebuilt `cp312-abi3` wheel — no build-from-source risk).
2. `pycocotools.coco.COCO.annToRLE` (and thus `.annToMask`) dispatches on the **type/shape of the `segmentation` field itself** (list → polygon, dict with list `counts` → uncompressed RLE, dict with string `counts` → compressed RLE) — not on the `iscrowd` flag. This is the correct, general dispatch, and it's what avoids a real bug found in `supervision.dataset.formats.coco.coco_annotations_to_masks`: that function only takes the RLE decode path when `iscrowd` is truthy, but our data's `segmentation` is genuine compressed RLE (`{"size":[H,W],"counts":<str>}`) with `iscrowd` always `0` — so `from_coco(..., force_masks=True)` would silently return **empty masks** for every instance, no crash, just wrong. Using `pycocotools` (the actual reference implementation the COCO format was defined around) sidesteps this entirely, correctly, and generally — no custom workaround needed.

**Verified schema facts driving this implementation** (confirmed directly against all 3 clips' `annotations.json`, not assumed):
- `categories` always exactly one entry (`surgical_instrument`, id 1). `iscrowd` and `category_id` are constant (`0`, `1`) on every annotation in every clip. Assert single-category at load time (`assert len(coco.getCatIds()) == 1`) — a real assumption baked throughout the rest of the code, worth failing loudly on if a future export ever changes it.
- `segmentation` is genuine COCO **compressed RLE** — `{"size": [H, W], "counts": <str>}` — never polygon, in this data. (`pycocotools` also correctly handles polygon if that ever changes, for free.)
- `track_id` (clip-local small int), `instance_id` (`= video.id * 10000 + track_id`, globally unique), `gemini_description` (free-text, e.g. "stainless steel bone rongeur") are present on every annotation; `attributes` (dict) is optional/absent on some (`{"manually_edited": True}` or `{"propagated_from": <int>}`, never both). `gemini_bbox`/`gemini_points` are in a different, unreliable coordinate space — ignore them entirely, never surface them past the raw COCO dict.
- Not every image has ≥1 annotation (one frame in IMG_2081 has zero) — must produce a `Frame` with an empty `annotations` list for that case, not crash.
- Annotation `id` is only unique within one clip's file; image `id` is already globally unique across clips.

### `models.py`

```python
@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]: ...


@dataclass(frozen=True)
class InstrumentAnnotation:
    track_id: int
    instance_id: int
    description: str            # from gemini_description — human-readable label
    bbox: BBox
    manually_edited: bool
    propagated_from: int | None
    segmentation: dict          # raw COCO segmentation dict, kept as-is for decode

    def to_mask(self) -> np.ndarray:
        """maskUtils.decode reads width/height off the RLE's own 'size' field —
        no separate width/height needed here, no W/H-ordering footgun."""
        return maskUtils.decode(self.segmentation).astype(bool)


@dataclass(frozen=True)
class Frame:
    frame_number: int
    file_path: Path
    width: int
    height: int
    annotations: list[InstrumentAnnotation]

    def load_image(self) -> np.ndarray:
        """Lazy — cv2.imread, not an eagerly-loaded field."""
        ...

    def to_detections(self) -> sv.Detections:
        """Bridge to supervision, for visualization only (annotators/plot_image
        consume sv.Detections). Builds .xyxy/.mask/.data['description'] from
        self.annotations; masks via InstrumentAnnotation.to_mask()."""
        ...


@dataclass(frozen=True)
class Clip:
    name: str                      # "IMG_2047"
    video_path: Path
    fps: float
    frame_count: int
    resolution: tuple[int, int]    # (width, height)
    frames: list[Frame]            # only the sampled/annotated frames, not every video frame

    @classmethod
    def from_directory(cls, clip_dir: Path) -> "Clip":
        """Parse clip_dir/annotations/annotations.json via pycocotools.coco.COCO:
        assert single category, iterate coco.getImgIds() -> loadImgs/getAnnIds/loadAnns,
        build InstrumentAnnotation per raw annotation dict, group into Frame per image,
        sort frames by frame_number."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Frame: ...
```

### `dataset.py`

```python
@dataclass(frozen=True)
class ClipDataset:
    clips: list[Clip]

    @classmethod
    def from_data_dir(cls, data_dir: Path | str | None = None) -> "ClipDataset":
        """Discover clip subdirectories under data_dir (default model/data — resolved
        relative to this file's location on disk; acceptable since data/ is dev/eval-only
        code, never reaches artifacts/). Skip (don't crash on) incomplete clip dirs, since
        data/ is gitignored/machine-local. Clip.from_directory() each valid one."""

    def __len__(self) -> int: ...
    def __iter__(self): ...
    def get_clip(self, name: str) -> Clip: ...
    def __getitem__(self, key: str | int) -> Clip: ...
```

Naming: `ClipDataset` (not `SurgicalInstrumentDataset`) — kept generic, doesn't hardcode the current single category name into the class itself.

`__init__.py` re-exports `BBox`, `InstrumentAnnotation`, `Frame`, `Clip`, `ClipDataset`.

### Notebook — `playground/01_explore_ground_truth.ipynb`

1. Imports: `orc_model.data`, `supervision as sv`.
2. **Parameters cell** (clearly marked, e.g. `# --- parameters, edit me ---`): `CLIP_NAME = "IMG_2081"`, `IMAGE_INDEX = 0` — this is the "easily vary and play with it" surface; re-run from here down after editing.
3. `ClipDataset.from_data_dir()` → print available clip names/frame counts, so the user sees what's pickable before setting `CLIP_NAME`.
4. `clip = dataset.get_clip(CLIP_NAME)` → print `clip.fps`, `clip.resolution`, `len(clip)`.
5. `frame = clip[IMAGE_INDEX]` → `image = frame.load_image()` → print `image.shape`, `len(frame.annotations)`, and one annotation's `track_id`/`description` (proves the parse worked end-to-end).
6. Build `sv.MaskAnnotator(opacity=0.5, color_lookup=sv.ColorLookup.INDEX)`, `sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)`, `sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)` (use `INDEX` not the default `CLASS` — this dataset is single-class, `CLASS` would paint every instrument identically). `detections = frame.to_detections()`, labels from `detections.data["description"]` (human-readable) rather than class name. Annotate mask→box→label onto a copy of `image`, `sv.plot_image(annotated, size=(12,12))`.
7. Optional: `sv.plot_images_grid` over the whole clip (box+label only, skip masks for speed) to eyeball all frames and help pick a good shared sample image for Phase 2.

No promotion into `src/visualization/` yet — this annotate-and-display block is ~6 lines and will be lightly duplicated once more in notebook 02; two uses isn't the "promote once reusable" trigger per `playground/README.md`. Flag: a third use (e.g. a future Tracker-eval notebook) is the trigger to extract it.

---

## Phase 2 — `Detector` component + prediction notebook

New files:
```
src/orc_model/components/
├── detector.py               # Detector class: public API
└── _rfdetr_postprocess.py    # pure functions: preprocess(), decode_predictions() — no onnxruntime/cv2 beyond numpy
```

### RF-DETR ONNX contract (verified against the actual `.onnx` file's I/O signature and the official RF-DETR repo's `PostProcess.forward`, `rfdetr/models/lwdetr.py`)

- Input `input`: `[1,3,768,768]` float32. Preprocess: resize to 768×768 (plain resize, no letterbox), scale to `[0,1]`, ImageNet-normalize (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`), HWC→CHW, add batch dim.
- Outputs: `dets [1,300,4]` (cxcywh, normalized `[0,1]` **relative to the 768×768 input**, not original image), `labels [1,300,2]` (raw logits — index 0 is an unused/background channel since only category id 1 was trained, index 1 = `surgical_instrument`), `masks [1,300,192,192]` (raw per-query mask logits, matched 1:1 by query index to `dets`/`labels`).
- **Correct decode** (matches official `PostProcess`, NOT the naive per-query-independent-argmax some third-party ONNX wrappers use, which can leak background rows and mismatch box↔mask pairing): `prob = sigmoid(labels)` → flatten to `(300*2,)` → top-k over the **flattened** array → recover `query_idx = flat_idx // 2`, `class_idx = flat_idx % 2` → gather `dets[query_idx]` and `masks[query_idx]` by that **same** `query_idx` → drop rows where `class_idx == 0` → apply confidence threshold on `prob` at the selected index → cxcywh→xyxy, scale from `[0,1]` to the **original** image's pixel width/height (not 768×768) → resize each selected query's 192×192 mask logits to original H,W (bilinear) → threshold `> 0` for the boolean mask.

### `_rfdetr_postprocess.py`

```python
def preprocess(image: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 (any H,W) -> float32 (1,3,768,768) NCHW, RGB, ImageNet-normalized."""

def decode_predictions(
    dets: np.ndarray, labels: np.ndarray, masks: np.ndarray,
    image_width: int, image_height: int,
    confidence_threshold: float, top_k: int,
) -> sv.Detections:
    """sigmoid -> flatten -> top-k over (query,class) -> shared-index gather ->
    drop class_idx==0 -> threshold -> cxcywh->xyxy, scale to original image ->
    per-query mask resize+threshold. Returns Detections in original image coords."""
```

Split into two pure functions (not one big `postprocess`, not inline in the class) specifically because this decode is subtle enough to need synthetic-array unit tests, which are far easier against small pure functions than against a class that also owns an `onnxruntime.InferenceSession`. Leading underscore = internal to `components/`; nothing outside imports it directly, always go through `Detector`.

### `detector.py`

```python
class Detector:
    def __init__(self, weights_path: Path | str, confidence_threshold: float = 0.5,
                 top_k: int = 300, providers: list[str] | None = None) -> None:
        """Loads the RF-DETR ONNX model via onnxruntime.InferenceSession."""

    def predict(self, image: np.ndarray) -> sv.Detections:
        """image: BGR np.ndarray (H,W,3) e.g. from cv2.imread or Frame.load_image().
        Returns sv.Detections in the image's own pixel coordinate space:
        .xyxy, .mask (bool, per-detection HxW), .confidence, .class_id (always 0),
        .data["class_name"] = "surgical_instrument"."""
```

Confirmed choice: `predict()` returns `sv.Detections` — same type Phase 1's `Frame.to_detections()` bridges ground truth into, so the exact same annotator code from notebook 01 reuses verbatim in notebook 02, and this sets up any future IoU/matching eval code (which operates on pairs of `Detections`) with no translation layer. `predict()` internally: `preprocess → session.run → decode_predictions`.

Constructor takes a **path** (not a preloaded session) so the notebook controls exactly when the model load happens.

### Notebook — `playground/02_first_predictions.ipynb`

1. Imports: `orc_model.data`, `orc_model.components.detector.Detector`, `supervision as sv`.
2. **Parameters cell**: same `CLIP_NAME`/`IMAGE_INDEX` as notebook 01 (same values, with a comment pointing back at it, so both notebooks visibly line up on one sample frame), plus `WEIGHTS_PATH = Path("../weights/rf-detr-seg-2xl-v1.onnx")`, `CONFIDENCE_THRESHOLD = 0.5`.
3. Load ground truth exactly as in notebook 01 (`ClipDataset` → `Clip` → `Frame` → `frame.to_detections()`) — needed for the side-by-side, not just the image.
4. `detector = Detector(WEIGHTS_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)`; `predicted = detector.predict(image)`; print `len(predicted)`, `predicted.confidence`.
5. **Side-by-side cell** — annotate ground truth and prediction onto two copies of the same image (masks+boxes; predictions fall back to confidence-score labels since there's no `gemini_description` for a prediction), `sv.plot_images_grid([gt_annotated, pred_annotated], grid_size=(1,2), titles=["ground truth", "prediction"])`. This is the actual deliverable — eyeball GT vs. prediction on the same frame.
6. Optional stretch cell: loop over 2-3 more images/clips for a quick qualitative sense of failure modes (plain `for` loop, no new infrastructure).

**Markdown callout in this notebook**: a systematically-offset or empty prediction mask/box is the signal to revisit the decode assumptions (§ RF-DETR ONNX contract above) — the synthetic unit tests validate the decode *logic*, not that this specific exported `.onnx` file's channel-0-is-background assumption actually holds; this notebook comparison is the real validation for that.

### Why no `inference/` or `visualization/` yet
`inference/` is for reusable batch/video runners — a single-image notebook loop doesn't need it; build it when there's an actual driving need (e.g. running `Detector` over a full clip's video for Tracker development). `visualization/` is deferred per the "promote once reusable" rule above. Both are natural next increments, not part of this plan.

---

## Dependencies

- Add `pycocotools` to `pyproject.toml` (verified: resolves cleanly to `2.0.11` via a prebuilt wheel against this project's Python 3.13 venv — no build-from-source risk).
- No `pydantic` — not used anywhere in this plan.
- `onnxruntime`, `supervision` already present from prior work.

---

## Testing

`tests/` (pytest, synthetic/deterministic — no real weights or images required):

- `tests/data/test_models.py` — `Clip.from_directory` against a small hand-written synthetic COCO JSON (2-3 images, real RLE segmentation — can generate via `pycocotools.mask.encode` on a hand-drawn small boolean array for a proper round-trip, one image with zero annotations matching the real IMG_2081 edge case, `attributes` present on some/absent on others). Assert `InstrumentAnnotation.to_mask()` decodes correctly (not empty!) and matches the original hand-drawn mask; assert `track_id`/`instance_id`/`description` come through correctly; assert the zero-annotation image produces a `Frame` with `annotations == []`, not a crash.
- `tests/data/test_dataset.py` — `ClipDataset.from_data_dir` against a synthetic `tmp_path` directory tree (one well-formed clip, one deliberately incomplete one to confirm skip-not-crash), `get_clip`/`__getitem__` lookups.
- `tests/components/test_rfdetr_postprocess.py` — **highest-value test in this plan.** Hand-built synthetic `dets`/`labels`/`masks` (~5 fake queries, most low-confidence/background) asserting: background-channel (`class_idx==0`) rows excluded even at high sigmoid score; box and mask in a returned row come from the *same* query index (give each synthetic query a distinguishable mask "signature" so a mismatched gather is detectable); cxcywh→xyxy + 768×768→original-size rescale numerically correct on a hand-computed example; `confidence_threshold`/`top_k` boundary behavior.
- `preprocess()` — shape `(1,3,768,768)`, dtype float32, correct ImageNet normalization on a known constant-color synthetic input.

Notebook-only (not in `tests/`): whether predictions look qualitatively right on real footage (no assertion substitutes for eyeballing), and full `Detector.predict()` against the real `.onnx` file (slow, weights gitignored/not guaranteed in CI, no pixel-exact ground truth to assert against — plausibility is exactly what the notebook is for).

---

## Sequencing (small, visually-checkpointed steps)

1. `models.py`'s `Clip.from_directory` (via `pycocotools`) → verify against one real clip: `len(clip)` matches known frame count, one frame's `InstrumentAnnotation.to_mask()` produces a real non-empty mask.
2. `dataset.py`'s `ClipDataset.from_data_dir` → sanity-check all 3 clips discovered with correct paths and frame counts (9/12/13).
3. Unit tests for `models.py`/`dataset.py` (including a test that would catch a wrong iscrowd-gated decode if that mistake were reintroduced).
4. **Checkpoint 1**: build `playground/01_explore_ground_truth.ipynb` end-to-end, get a real mask+box+label overlay on a real frame. Pick and note the shared sample image (clip + index) for Phase 2.
5. `_rfdetr_postprocess.preprocess()` + unit test.
6. `_rfdetr_postprocess.decode_predictions()` + synthetic-array unit tests — green *before* running against real model output (index-alignment bugs are much easier to debug against hand-built arrays).
7. `components/detector.py`'s `Detector` — smoke-test it loads the real `.onnx` and runs on one real image without crashing (not yet judging correctness).
8. **Checkpoint 2**: build `playground/02_first_predictions.ipynb` on the same sample image from checkpoint 1, visually compare GT vs. prediction side by side — the actual deliverable.
9. Only after checkpoint 2 looks visually sane: iterate on `confidence_threshold`/`top_k`, try other clips/images, decide if anything's worth promoting toward `visualization/`/`inference/` (likely not yet).

## Verification

- Steps 3, 5, 6: `uv run pytest` inside `model/`, green.
- Checkpoints 1 and 2: open the notebook in Jupyter (`uv run jupyter lab`), run top to bottom, visually confirm the overlay looks correct (masks align to actual instruments, labels readable) and, at checkpoint 2, that predicted boxes/masks are in the right ballpark vs. ground truth on the same frame.
