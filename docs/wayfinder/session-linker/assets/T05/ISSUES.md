# Build issues — SessionLinker + `load_tracker()` (T05 brief, checked)

**Provenance:** checker pass 2026-07-15 ~11:45 CEST over the session handoff,
every claim verified against the working tree. This file is the build brief for
coders — the handoff itself lives in a temp dir and may vanish; this file is
self-contained. Effectively delivers
[T05](../../tickets/T05-implementation-spec.md)'s "scoped issues for the build" —
link it there when updating the ticket trail (W2).

**Goal:** UZ Leuven demo **2026-07-20**. `app/backend/backend/main.py:318-323`
already does `from orc_model.pipelines.tracking import load_tracker` — that
function does not exist. Everything it composes (detector, vendored Deep
OC-SORT, SRC matcher) exists in-repo but is unwired.

**Read first (in order):**

1. `model/docs/linker-design.md` — pinned behaviour (§6 is stale, see W1).
2. `model/docs/tracker-interface.md` — the seam contract, esp. § "`tracker_id`
   across absence" (line 103) and § "`confidence`".
3. `experiments/matcher-autoresearch/FINDINGS.md` §0–§2 — why SRC won, and the
   near-final §6 replacement prose.
4. `experiments/matcher-autoresearch/champion/PARAMS.md` — SRC hyperparameters
   + guarded numbers.
5. `docs/wayfinder/session-linker/MAP.md` — ticket conventions; append
   decisions there, don't duplicate.

---

## Verified facts — do NOT re-derive these

| Fact | Where verified |
|---|---|
| `load_tracker` missing; `tracking.py` has only the `InstrumentTracker` Protocol + `FakeInstrumentTracker` | `model/src/orc_model/pipelines/tracking.py` |
| Call site exists and also uses `tracker.model_version`; `CaptureLoop` calls `tracker.reset()` | `app/backend/backend/main.py:318-326`, `capture.py:196` |
| Detector weights on disk | `model/weights/checkpoint_best_regular.onnx` (150 MB) |
| Detector emits **masks** (RF-DETR instance-segmentation ONNX, `masks` output) | `model/src/orc_model/components/detector/detector.py:64` |
| Vendored tracker: `DeepOCSortTracker.update(detections, frame) -> sv.Detections`, carries `mask`/`class_id` through untouched, emits `trk.id + 1` (ids start at 1, monotonic) | `model/playground/trackers/deep_ocsort_vendor/tracker.py`, `ocsort.py:360` |
| SRC is still champion (round 3, promoted 2026-07-15); rounds 4–9 failed to unseat it | `experiments/matcher-autoresearch/LEADERBOARD.md` (mtime 09:59) |
| Champion module is standalone (no `frozen/`/`method/` imports); defaults match PARAMS.md exactly (`alpha=0.0003, size_alpha=0.5, tau=0.30, margin=0.02, mask_dilate_px=2, max_iter=2000`) | `experiments/matcher-autoresearch/champion/*.py` |
| `model/.venv` missing `sklearn` and `transformers` (import errors reproduced); torch/torchvision present but **dev-only** in pyproject | `model/pyproject.toml`, venv import checks |
| `facebook/dinov2-base` is in the local HF cache — the offline-forced embedder (`HF_HUB_OFFLINE=1`) will load on this machine | `~/.cache/huggingface/hub/` |
| Smoke-test data exists: `model/data/instruments/instrument{1..8}/images/` (15 frames each) | listed |
| `cv2` imports in `model/.venv` (transitive via supervision) — no explicit dep needed for the tracker move | venv import check |

**Latency risk from the handoff is now measured and downgraded.** The
"never-profiled `MultiTaskLasso` solve" (FINDINGS §0/§7) was benchmarked at demo
scale on this machine (D=770, `alpha=0.0003`, `max_iter=2000`, window=3):
**0.2–0.8 ms** across dictionary widths 3→120 (up to 8 candidates × 15 atoms) —
three orders of magnitude inside the ≤1.0 s link budget. The DINOv2 embed
(~0.2–0.5 s/event on MPS) dominates, as already budgeted in MAP.md. Synthetic
data caveat: still log per-birth-event latency in W6/W8 and watch for sklearn
`ConvergenceWarning` (max_iter hits) on real correlated embeddings — but this is
no longer a reason to hold the fallback ready-to-swap.

---

## Checker findings — corrections & gotchas (cross-cutting)

### C1 · BGR→RGB conversion is required, or the tuned thresholds are void — HIGH

The research pipeline embedded **RGB** crops
(`experiments/matcher-autoresearch/frozen/loader.py:81` —
`Image.open(...).convert("RGB")`). The live seam delivers **BGR** frames
(`tracking.py:52`, cv2 convention). The champion's `_apply_mask()` passes
whatever channel order it's given straight to `PIL.Image.fromarray` → DINOv2.
Feeding BGR silently shifts every embedding and invalidates `tau=0.30` /
`margin=0.02`, which were tuned on RGB. **`SessionLinker` must convert crops to
RGB before every `build_gallery`/`score` call.** Add an explicit smoke-test
check in W8 (e.g. embed one known crop both ways and confirm which matches the
research pipeline's behaviour).

### C2 · The autoresearch loop is STILL RUNNING — snapshot before porting — MEDIUM

`runs/r5-c2/` had an mtime equal to check time (11:43). `champion/` and
`method/` sit in that loop's mutable working dir and can change or be promoted
over at any moment. Before W5: re-check `LEADERBOARD.md` + `champion/` mtimes;
if a new champion was promoted, build against whatever `champion/` now contains
(the `build_gallery/score/accept` interface is stable). The port itself is the
snapshot — record the source files' mtime/sha in the port commit message.
`experiments/` is **untracked in git**, so there is no history to recover from.

### C3 · Round-0 fallback location is wrong in the handoff/LEADERBOARD — MEDIUM

LEADERBOARD's "Round 0 detail" points at `runs/baseline/` — **that directory no
longer exists** (the loop pruned `runs/`; only `r5-*` remain). The baseline
code survives at `experiments/matcher-autoresearch/method/` (docstring: "the
CHAMPION BASELINE"; DINOv2 + size-fusion + top-3-mean + voting). It imports
`frozen.interface` so a port needs a one-line import fix. Given C2, if you want
the fallback insurance, copy `method/` somewhere safe at the same time as W5.
(With the latency risk downgraded above, the fallback is unlikely to be needed —
but it costs one `cp -r`.)

### C4 · `DeepOCSortTracker` converts `max_age_seconds` using `frame_rate=30` default — HIGH

`max_age = round(max_age_seconds * frame_rate)` (`tracker.py:36`). Live fps is
inference-bound, historically **~10–15**, and `capture.py` has no fps handling.
Leaving `frame_rate=30` at real ~12 fps makes a "1.0 s" coast last ~2.5 s of
wall time — breaching the design's tracker/linker boundary (linker-design §9:
coast must sit *below* the app's 1.5 s off-debounce) and widening the
foreign-object mis-bind window. `load_tracker()` must accept/pin the real fps
(measure it in W9's live run; `docs/wayfinder/session-linker/assets/T04/RESULTS.md`
has offline numbers but not a live end-to-end fps pin — that's still the open
"fps alignment" item in tracker-interface.md). Same fps dependency applies to
`min_hits=3` maturation and the ~0.5 s enrolment/evidence windows (count frames,
don't hardcode counts derived from 30 fps).

### C5 · Dependency list in the handoff is incomplete — MEDIUM

W3 must add **`torchvision`** to main deps too (the relocated Deep OC-SORT's
`TorchvisionEmbedder` needs it; it's currently dev-only alongside torch). Full
main-dep additions: `torch`, `torchvision`, `transformers`, `scikit-learn`,
`scipy`, `pillow`. (`scipy` serves three consumers: champion's
`binary_dilation`, the vendored tracker's `scipy.spatial`, and W6's
`linear_sum_assignment`.) `cv2` already arrives via supervision — verified.

### C6 · §6.5 assignment layer — implementation gotchas — MEDIUM

FINDINGS §2 explicitly says simultaneous multi-returns were **never evaluated**
for SRC; the handoff's engineering default (per-track `score()` against the full
Missing dictionary → score matrix → mask per-row τ/margin failures → Hungarian
maximize) is sound and consistent with `accept()`'s semantics. Gotchas:

- `scipy.optimize.linear_sum_assignment` **raises** ("cost matrix is
  infeasible") if a row/column is entirely `-inf`. Prefilter fully-masked
  rows/columns (or use a large finite negative sentinel and reject those
  assignments afterwards) — don't pass raw `-inf` matrices.
- SCI depends on the candidate-set size K. Score every simultaneous new track
  against the **same** full Missing dictionary and solve the matrix **once**;
  don't remove a claimed candidate mid-batch and re-solve, or scores stop being
  comparable across rows.
- The margin gate is per-row (best vs second-best *within that track's
  scores*), replicating `accept()` — you can't reuse `accept()` directly since
  it discards the score vector.
- Flag it in the doc (W1) as an engineering call pending validation — normally
  a `/grilling` item with Bram; time pressure justifies shipping the default
  and revisiting only if it misbehaves live.

> **Update 2026-07-15 — grilled with Bram and PINNED** in `linker-design.md`
> §6.5: natural batching (no grace timer), per-row τ/margin gates, assignment,
> **one extra re-score round** for unresolved tracks against the reduced
> Missing set, then Unknown. Implement that section verbatim; the gotchas
> above still apply. Validation: W8 gets a video-003 teardown→re-lay replay.

### C7 · Stale "DINOv3" references in both docs — LOW

`linker-design.md` (glossary, §4, §9) and `tracker-interface.md:114` say
**DINOv3**; the pinned, benchmarked embedder is **DINOv2-B**
(`facebook/dinov2-base` — T02/T04 used it as the ungated stand-in and all
guarded numbers are DINOv2-B's). W1's rewrite should correct these mentions (or
pin DINOv2-B explicitly) — otherwise the doc contradicts the code it specifies.

### C8 · `confidence` / `class_names` / `model_version` plumbing — LOW

The Protocol requires all three. Nothing provides them yet: expose `confidence`
as a plain attribute forwarded per-call to `Detector.predict(...,
confidence_threshold=...)` (the demo app never changes it, but the seam demands
it exist); OC-SORT's internal `det_thresh` can stay fixed at the startup value.
`class_names` → `{0: "surgical_instrument"}` (single-class detector);
`model_version` → derive from the weights filename + short hash (the app stores
it with every frame).

### C9 · Handoff's task-tracker and memory-file pointers are wrong — INFO

The session task tracker is **empty** (tasks #1–#8 did not survive the session
boundary) — work from this file instead. The memory file is
`session-linker-wayfinder-map.md` (+ `t04-linker-benchmark.md`), not
`ss-wayfinder-session-linker.md`.

---

## Work items (build order)

### W1 — Rewrite `linker-design.md` §6 around SRC

§6 still describes the superseded nearest-view-max-cosine + Hungarian rule.
FINDINGS §2 has near-final replacement prose + the full parameter table —
adapt, don't re-derive. Include: the §6.5 assignment layer per C6 (flagged as
engineering default, not research-backed); the C7 DINOv3→DINOv2-B correction;
a review of §4's aggregation language (top-K/nearest-view-max is baseline
vocabulary — SRC uses every gallery view as a dictionary atom, no aggregation
step).

### W2 — Ticket trail

Point-don't-restate: note in `tickets/T02-benchmark-embeddings.md` that stage-2
findings are consumed into the build; `T03-linker-design.md` is closed but now
stale — add a post-closure update section (or reopen), don't silently edit; one
line under MAP.md "Decisions so far"; link this file from
`tickets/T05-implementation-spec.md`.

### W3 — Dependencies

Add to `model/pyproject.toml` **main** deps: `torch`, `torchvision`,
`transformers`, `scikit-learn`, `scipy`, `pillow` (move torch/torchvision out
of dev-only). Then `uv sync` in `model/`. (C5.)

### W4 — Relocate Deep OC-SORT

`model/playground/trackers/deep_ocsort_vendor/` →
`model/src/orc_model/pipelines/deep_ocsort/`. Relative imports inside the
package should survive the move; no logic changes. Leave the playground copy or
delete it — but don't let two copies drift.

### W5 — Port the SRC champion

`experiments/matcher-autoresearch/champion/{interface,embedder,size_features,champion}.py`
(301 lines) → `model/src/orc_model/pipelines/matching/`, near-verbatim. **Do C2
first** (re-check LEADERBOARD/champion mtimes, record source sha/mtime in the
commit). Optionally snapshot `method/` as the fallback (C3). Keep PARAMS.md's
provenance reachable (copy it alongside or link it).

### W6 — `SessionLinker`

New `model/src/orc_model/pipelines/session_linker.py` implementing
linker-design §§1–7 on the ported matcher:

- enrolment window / roster freeze (§1); Active/Missing bookkeeping + private
  raw→session map (§2);
- **session-only galleries** — §3's persistent-binding path must no-op
  gracefully (T07 photos don't exist yet); degrade, don't crash;
- event-driven embedding at track birth/death only, never per-frame (§4);
- the rewritten §6/§6.5 rule from W1, with C1 (RGB) and C6 (assignment
  gotchas) applied;
- interface mirrors the tracker wrapper:
  `linker.update(detections, frame) -> detections`; `reset()` clears session
  state;
- log per-birth-event latency (embed + solve, separately);
- **out of scope, mark as TODO comments (not silently dropped):** §8 Line-2/3
  rotation augmentation, §5 stability-gated refresh, §9 reactivation
  re-validation if it doesn't fall out naturally.

### W7 — `load_tracker()`

`load_tracker(weights_path, confidence=0.5, ...) -> InstrumentTracker` in
`model/src/orc_model/pipelines/tracking.py`, composing `Detector` →
`DeepOCSortTracker` → `SessionLinker` per linker-design "Where it composes".
Apply C4 (fps/`frame_rate`) and C8 (protocol attrs). This is the concrete
unblock for `app/backend/backend/main.py:318`.

### W8 — Offline smoke test + video-003 §6.5 replay

Script driving `load_tracker`/`SessionLinker` against
`model/data/instruments/*/images` (run the real detector on those frames —
weights are local): simulate a pickup (drop an instrument's frames
mid-sequence) and a return (reintroduce under a fresh raw id); assert the
original session id comes back. Include a multi-return swap + foreign-object
case, the C1 RGB check, and the latency log. Catches integration bugs before
burning hardware time.

**Plus (grilled decision):** replay video `09-07-26-003`'s teardown→re-lay
(31 real returns; labels + cache tooling in `assets/T04/`) through the real
`SessionLinker` — the only real-data validation of §6.5 before the live test.

### W9 — First live test

```
cd app/backend && uv run python -m backend.main --camera 0 \
  --weights ../../model/weights/checkpoint_best_regular.onnx
```

**Pre-flight (found during W7's real smoke):** Deep OC-SORT's torchvision
embedder downloads `mobilenet_v3_small-047dcff4.pth` from download.pytorch.org
on first use if `~/.cache/torch/hub/checkpoints/` doesn't have it. It is
cached on this machine now (2026-07-15), but verify before demo day — an
offline demo room + cold cache = load_tracker() hangs at startup. Same check
for the HF cache (`facebook/dinov2-base`).

Physically pick up and set down an instrument; its tile must keep its
colour/id in the frontend. Measure real end-to-end fps here and pin it (C4).
Expect to retune `tau`/`margin` after this — every guarded number is from a
tiny synthetic crop set (8×15 crops, no real return-window video; see
`assets/T04/RESULTS.md`, whose matcher numbers are the **old baseline's**, not
SRC's).
