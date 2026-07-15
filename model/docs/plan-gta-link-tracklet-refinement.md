# Plan — GTA-Link tracklet refinement for Deep-OC-SORT

**Goal:** fix the re-identification failure in `playground/trackers/deep_ocsort.ipynb` —
an instrument occluded for a while comes back under a *new* track id — by adding
[GTA-Link](https://github.com/sjc042/gta-link)-style **offline tracklet post-processing**
(split + connect) after the tracking loop, adapted to this application (surgical
instruments, not sports players).

**Status:** plan only, nothing implemented. Each step below is scoped to be delegated
to a subagent independently, in order.

---

## 1. Background

### 1.1 What GTA-Link does

GTA-Link (MIT-licensed, verified 2026-07-15) is a **model-agnostic, offline** refinement
of finished tracking output. Two sequential stages over a `dict[track_id, Tracklet]`:

1. **Split** (`split_tracklets`): within each sufficiently long tracklet, run DBSCAN
   (cosine metric) on the per-detection appearance embeddings. Multiple clusters ⇒ the
   tracklet contains an id switch ⇒ fragment it into one tracklet per cluster (noise
   points reassigned to nearest cluster center). Params: `eps=0.7`, `min_samples=10`,
   `max_k=3`, `len_thres=100` frames.
2. **Connect** (`merge_tracklets`): hierarchical agglomerative merging. Pairwise
   distance = mean cosine distance across all embedding pairs of the two tracklets,
   forced to 1.0 (unmergeable) if the tracklets **overlap in time**. Repeatedly merge
   the closest pair below `merge_dist_thres=0.4`, subject to a **spatial gate**
   (exit position of the earlier fragment vs entry position of the later fragment must
   fall within `max_x_range/max_y_range`, derived from tracklet extents × `spatial_factor`).

Upstream files that matter: `Tracklet.py` (plain data class: `times`, `scores`,
`bboxes`, `features`, `extract()`), `refine_tracklets.py` (all algorithm functions).
`generate_tracklets.py` (OSNet person-ReID feature extraction) is **not** used — see 2.2.

### 1.2 Why this fits our failure mode

The observed problem is fragmentation: occlusion (hand, other instrument) or leaving
the frame kills a track past `max_age`, and the reappearance spawns a fresh id. The
**connect** stage is exactly the fix. The **split** stage guards the opposite failure
(one tracklet swallowing two identities after a crossover) — useful but secondary here.

### 1.3 Application specifics (vs. sports MOT this was tuned for)

From `docs/tracker-interface.md` and the MVP context:

- **Few objects** (one surgical set, roughly 5–20 instruments), single class
  `surgical_instrument`, and the demo set contains **no duplicate instrument types**
  (mvp issue #2) — appearance-based re-association is viable *by design*.
- **Temporal-overlap gate is highly informative here**: two tracklets visible at the
  same time are necessarily different instruments. Keep it strict.
- **Spatial gate is weakly informative here**: unlike a player running across a pitch,
  an instrument can be picked up and put back down *anywhere* (or re-enter the frame at
  a different edge). Default this gate **off** (or `spatial_factor` large), keep it as
  an option.
- **Our embeddings are not OSNet person-ReID.** Do **not** port gta-link's
  torchreid/OSNet pipeline. Instead, **swap the tracker's appearance embedder to
  DINOv2-S** (`dinov2_vits14` / `facebook/dinov2-small`, 384-dim, self-supervised,
  general-purpose visual features) in place of the current `TorchvisionEmbedder`
  (MobileNetV3-Small/ImageNet) in `playground/trackers/deep_ocsort_vendor/embedder.py`
  — see step 2. DINOv2 features are known to transfer well to fine-grained
  instance re-identification without task-specific fine-tuning, which is exactly
  what unlabeled surgical instruments need and what ImageNet-classification
  features are weakest at. Consequence: **every distance threshold (`eps`,
  `merge_dist_thres`) must still be calibrated for this embedding space** —
  upstream gta-link defaults (tuned for OSNet) are meaningless here regardless of
  backbone (see step 5).
- Notebook runs offline on cached detections (`PredictedClip`) resampled to
  `TARGET_FPS=30`; tracklet `times` should store **native video frame indices**
  (consistent with `frame_indices` in the notebook) so lengths/gaps in frames map to
  seconds via `TARGET_FPS`.

### 1.4 Offline vs. online — scope boundary

GTA-Link is inherently **offline** (it needs the whole sequence). The production
contract (`docs/tracker-interface.md`, "track linking") demands **online** re-linking
within ≤ 1.0 s and no retroactive id rewrites. This plan does *not* build the online
linker. It builds the offline version in the playground to:

1. prove that our appearance embeddings can re-associate instrument tracklets at all,
   and find the working distance threshold;
2. produce the calibration (distance distributions, thresholds) that a future online
   linker — essentially the "connect" test run incrementally against a gallery of dead
   tracklets whenever a new track spawns — will reuse.

Note this in the notebook so the offline/online distinction stays visible.

---

## 2. Design decisions (already made — do not re-litigate)

### 2.1 Where the code lives

- `playground/trackers/gta_link_vendor/` — a small vendored module, same pattern as
  `deep_ocsort_vendor/`. Contents:
  - `tracklet.py` — `Tracklet` class (ported, typed, cleaned).
  - `refine.py` — `split_tracklets`, distance matrix, spatial-constraint helpers,
    `merge_tracklets`, plus a top-level `refine_tracklets(tracklets, config) -> dict[int, Tracklet]`
    convenience entry point.
  - `__init__.py` — exports + module docstring stating provenance
    (github.com/sjc042/gta-link, MIT) and what was changed.
- Keep the module **tracker-agnostic**: input is `dict[int, Tracklet]`, no imports
  from `deep_ocsort_vendor`. The sibling notebooks (bytetrack/ocsort/sort) can reuse it
  later if given embeddings.
- This is playground code, mirroring the existing convention. Promotion into
  `src/orc_model/` happens only when the online linker is designed — out of scope.

### 2.2 Where embeddings come from

**Expose the per-detection embeddings Deep-OC-SORT already computes** (after swapping
in the DINOv2-S backbone, step 2) rather than re-extracting features in a second pass
(gta-link's `generate_tracklets.py` approach):

- `OCSort.update()` already has `dets_embs` for every retained detection
  (`deep_ocsort_vendor/ocsort.py`). It must additionally return the embedding row for
  each output track (aligned with `det_indices`). Note: return the **per-frame raw
  detection embedding**, *not* the track's EMA-smoothed `trk.emb` — split/connect need
  per-frame features.
- `DeepOCSortTracker.update()` (`tracker.py`) attaches them to the returned detections
  as `result.data["embedding"]` (float32 `(N, 384)` once DINOv2-S is wired in).
  `sv.Detections.data` carries custom per-detection arrays through indexing natively,
  and nothing downstream breaks if the key is ignored.
- Edge case: when `embedding_off=True`, `dets_embs` is a dummy `(N, 1)` of ones —
  either skip attaching the key then, or document that refinement requires
  `EMBEDDING_OFF = False`.

This guarantees the refinement operates on *exactly* the appearance space the tracker
associated with, and costs zero extra inference beyond the embedder swap itself.

### 2.3 Tracklets built from tracker output, not detector output

The notebook loop accumulates tracklets from what `tracker.update()` returns
(confirmed tracks only, post-`min_hits`). Pre-confirmation detections are simply
absent from tracklets — acceptable, do not work around it.

### 2.4 Defaults for this application

| Param | Upstream | Ours (starting point) | Why |
|---|---|---|---|
| `use_split` | on | **off by default**, keep the flag | weaker generic embedder ⇒ DBSCAN over-splitting risk; our primary failure is fragmentation, not impurity |
| `use_connect` | on | on | the actual fix |
| spatial gate | on, `factor=1.0` | **off** (or factor ≥ 3) | instruments relocate arbitrarily (see 1.3) |
| `merge_dist_thres` | 0.4 | **from calibration (step 5)** | different embedding space |
| `eps` | 0.7 | from calibration | same |
| `min_samples` | 10 | 10 (≈ ⅓ s at 30 fps) | fine |
| `len_thres` (split) | 100 frames | 100 (≈ 3.3 s) | fine, expose it |
| `max_k` | 3 | 3 | fine |

Temporal-overlap veto in `get_distance`: keep exactly as upstream (distance 1.0 on any
shared frame index).

---

## 3. Implementation steps

Execute in order; steps 1–2 are independent of each other and can run in parallel.
Step 3 depends on step 2 (needs the swapped embedder in place before wiring its
output through). Step 3 does not depend on step 1.

### Step 1 — Vendor the refinement module

**Create** `playground/trackers/gta_link_vendor/{__init__.py, tracklet.py, refine.py}`.

- Port `Tracklet` and, from `refine_tracklets.py`: `detect_id_switch`/`split_tracklets`,
  `get_distance` + distance-matrix construction, `find_consecutive_segments` /
  `get_spatial_constraints` / `check_spatial_constraints`, `merge_tracklets`.
- **Strip**: all CLI/argparse, dataset-directory walking, `.pkl` I/O, MOT-format
  `save_results`, torchreid imports, hard-coded `.cuda()`.
- **Device handling**: cosine-distance math in torch on the best available device via
  the same `select_device()` pattern as `deep_ocsort_vendor/embedder.py` (MPS on this
  machine), with numpy fallback; tracklet counts here are tiny, so simplicity beats speed.
- Spatial gate behind an explicit flag (`use_spatial_gate: bool = False`).
- Merging keeps the **lower (older) track id** of each merged pair so output ids remain
  stable relative to first appearance.
- Add a `refine_tracklets(...)` entry point returning a refined
  `dict[int, Tracklet]` **plus an id-mapping** `dict[old_id, new_id]` so the notebook
  can relabel already-rendered structures without re-tracking.
- Module/file docstrings: provenance + MIT attribution + summary of deviations.
- Style: match `deep_ocsort_vendor` (typed, short docstrings explaining deviations
  from upstream).

**Acceptance:** module imports cleanly; a synthetic unit-style check in a scratch
script (e.g., three hand-built tracklets with orthogonal vs. identical fake features,
with/without temporal overlap) shows: overlapping never merge, identical-feature
disjoint tracklets merge, orthogonal ones don't.

### Step 2 — Swap the appearance embedder to DINOv2-S

**Modify** `playground/trackers/deep_ocsort_vendor/embedder.py`: replace (or add
alongside, selectable) `TorchvisionEmbedder`'s MobileNetV3-Small/ImageNet backbone
with DINOv2-S, keeping the same `compute_embedding(frame_bgr, bboxes_xyxy, masks=None)
-> np.ndarray` interface so `OCSort`/`DeepOCSortTracker` don't need to know which
backbone is behind it.

- **Model**: `facebook/dinov2-small` (ViT-S/14, 384-dim), loaded via `transformers`
  (`Dinov2Model`/`AutoModel`) rather than `torch.hub` — pinned, cacheable install
  through the project's normal dependency management instead of a GitHub-code
  download at runtime. Add `transformers` to `pyproject.toml` dependencies.
- **Embedding**: CLS token output, L2-normalized (matches the current convention).
- **Input constraint**: ViT-S/14 requires crop height/width divisible by 14. The
  current `crop_size=(128, 256)` (width, height) is not. Switch to a
  patch-size-aligned crop — default **224×224** (14×16, DINOv2's standard
  pretraining resolution) unless a quick check shows a taller aspect ratio (closer
  to the current 1:2) helps for elongated instruments (e.g. 126×252 = 9×14 by
  18×14); pick one and document why.
- **Reuse as-is**: the `mask_crop` background-suppression logic
  (`_suppress_background`) and device selection (`select_device()`, MPS-first) —
  both are resolution/backbone-independent.
- **Device/normalization**: use whatever mean/std the `transformers` DINOv2
  processor specifies (may differ from plain ImageNet mean/std) rather than
  hard-coding the existing constants.
- Update the module's provenance docstring: DINOv2-S is a general-purpose
  self-supervised backbone, not a person-ReID model — appropriate for
  never-seen-at-pretraining objects like surgical instruments, unlike gta-link's
  OSNet.

**Acceptance:** `compute_embedding` returns L2-normalized `(N, 384)` float32 for a
batch of crops on the notebook's `IMG_2112` frames; a quick sanity check (cosine
similarity of two crops of the *same* instrument in adjacent frames vs. two crops of
*different* instruments in the same frame) shows same-instrument similarity
noticeably higher — this is the first, cheap signal (ahead of step 5's full
calibration) that the backbone swap is worth keeping.

### Step 3 — Expose per-detection embeddings from the tracker

**Modify** `playground/trackers/deep_ocsort_vendor/ocsort.py` and `tracker.py` per 2.2:

- `OCSort.update()` returns `(tracked, det_indices, det_embeddings)` — embedding rows
  aligned with output rows (for a track updated this frame, the embedding of the
  detection it matched; shape `(M, D)`).
- `DeepOCSortTracker.update()` sets `result.data["embedding"]`.
- Handle `embedding_off` per 2.2. Update the vendor-deviation notes in the module
  docstring (`ocsort.py` documents its deltas from upstream — extend that list).

**Acceptance:** `deep_ocsort.ipynb` runs unchanged except that returned detections now
carry `data["embedding"]` with L2-normalized rows; other notebooks unaffected.

### Step 4 — Accumulate tracklets in the notebook loop

**Modify** `playground/trackers/deep_ocsort.ipynb` tracking loop: alongside the
existing per-frame bookkeeping, build `tracklets: dict[int, Tracklet]` — for each
output row append `(frame_index, confidence, bbox_xyxy, embedding)`. Store
native-video `frame_index` (the loop variable), bbox as `xyxy` (keep our convention;
if ported spatial helpers assume `xywh`, convert inside the vendor module, not the
notebook).

**Acceptance:** after the loop, `sum(len(t.times) for t in tracklets.values())` equals
the total tracked-row count; ids match the presence-timeline plot.

### Step 5 — Calibration cell: distance distributions before thresholds

New notebook section, **before** refinement. Compute and plot (plotly, consistent with
the existing timeline cell):

1. Histogram of **intra-tracklet** pairwise cosine distances (within each long
   tracklet) — the "same instrument" distribution.
2. Histogram of **inter-tracklet** mean cosine distances for **temporally overlapping**
   pairs — guaranteed "different instruments" distribution (no GT needed).
3. The gta-link pairwise distance matrix as a heatmap, tracklets ordered by first frame.

Pick `merge_dist_thres` (and `eps`, if split is exercised) from the gap between the
two distributions; surface both `MASK_CROP=True/False` if time allows, since mask-crop
should tighten the intra distribution.

**Acceptance:** the cell renders the three plots and prints a suggested threshold
(e.g., midpoint between intra p95 and inter p5), with a short markdown note reading
the result. If the distributions **overlap heavily, stop and report** — that means the
generic embedder can't separate instruments and merging would be noise (see Risks).

### Step 6 — Refinement + before/after comparison in the notebook

New notebook section after calibration:

- Config cell mirroring the notebook's existing config style
  (`USE_SPLIT = False`, `USE_CONNECT = True`, `MERGE_DIST_THRES = <from step 5>`, …).
- Run `refine_tracklets(...)`; apply the returned id-mapping to
  `track_ids_per_frame` / `track_positions_per_frame`.
- **Before/after** presence-timeline plots (reuse the existing plot code; same y-axis
  ordering where possible) and a printed summary: unique ids before → after, number of
  merges, number of splits.
- Re-render the annotated video with refined ids to
  `output/deep_ocsort/{CLIP_NAME}/tracked_{TARGET_FPS}fps*_refined.mp4`. Simplest
  faithful approach: relabel and re-annotate from the stored raw detections + id map
  (annotator colors key off `tracker_id`, so relabeled ids must be set before
  annotation, not after).

**Acceptance:** notebook runs end-to-end on `IMG_2112`; refined video + both timelines
present; unique-id count after refinement is ≤ before and plausibly close to the true
instrument count in the clip.

### Step 7 — Evaluation & write-up (small)

- Eyeball the refined video: does the occluded instrument keep its id? Any wrong merge
  (two different instruments under one id) is **worse than fragmentation** for the
  downstream completeness report — call these out explicitly.
- Optional, only if cheap: the hand-annotated GT (`ClipDataset` / notebook
  `01_explore_ground_truth.ipynb`) carries human `track_id` on sparse frames — a simple
  count of GT identities vs refined identities on those frames. No full IDF1/HOTA
  harness; that's out of scope.
- Add a short markdown conclusion cell: does connect work with the generic embedder,
  chosen thresholds, and implications for the online linker (1.4).

---

## 4. Risks / open questions

- **DINOv2-S still might not separate visually similar instruments well enough**
  (main risk, downgraded from "likely" to "possible" by the backbone swap — self-
  supervised DINOv2 features are known to be strong for fine-grained instance
  discrimination, but scissors-vs-scissors-like surgical tools are a genuinely hard
  case). Step 2's quick same/different-instrument sanity check and step 5's full
  calibration detect this *before* any merging is trusted. If DINOv2-S still isn't
  enough: try `dinov2_vitb14` (bigger backbone) or registers-variant
  (`dinov2_vits14_reg`, more robust local features) before abandoning the
  appearance-based approach.
- **Compute cost**: DINOv2 ViT-S/14 is heavier than MobileNetV3-Small — expect a
  noticeably slower embedding step per frame on CPU/MPS. This is offline notebook
  work so wall-clock is a nuisance, not a blocker, but don't assume the old
  `batch_size=64` default is still fine; re-tune if the loop stalls.
- **New dependency / first-run download**: `transformers` + the `facebook/dinov2-small`
  weights (downloaded from the Hugging Face Hub on first use, then cached) is new
  infra for this repo — confirm network access is available where the notebook runs,
  and that pulling a HF model doesn't conflict with any offline/air-gapped constraint
  (check with the team if unsure; nothing in current project docs suggests one).
- **Wrong merges are costly**: prefer a conservative `merge_dist_thres` (favor leftover
  fragmentation over id contamination) — the downstream human-in-the-loop fixes misses
  more easily than silent identity mix-ups.
- **DBSCAN split** may still shred long tracklets even with better embeddings — that's
  why `USE_SPLIT=False` by default; only enable to test crossover cases.
- **Low/variable fps in production** (~10–15 fps live vs 30 here): fewer embedding
  samples per tracklet; thresholds calibrated at 30 fps should be re-checked once the
  real fps is pinned (open item in `tracker-interface.md`).
- **`min_samples`/`len_thres` are frame-count params** — if `TARGET_FPS` changes,
  they change meaning; express them in seconds × `TARGET_FPS` in the notebook config.

## 5. Explicitly out of scope

- Online track linking behind `InstrumentTracker` (production contract) — this plan
  only produces its calibration and feasibility evidence.
- Porting gta-link's OSNet/torchreid ReID stack or `.pkl`/MOT-format I/O.
- Refining the other tracker notebooks (module is built reusable, wiring them is not
  part of this).
- Promotion of the vendor module into `src/orc_model/`.
- Full MOT metrics (IDF1/HOTA) harness.
