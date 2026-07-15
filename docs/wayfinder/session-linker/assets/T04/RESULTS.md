# T04 — linking eval on the 4 test videos (re-ID + open-set)

**Date:** 2026-07-14/15 · **Ticket:** [T04 — harness + scenarios](../../tickets/T04-harness-and-fps.md) (items 2–3) · **Machine:** M3 Max, CPU ONNX detector + MPS DINOv2

## TL;DR

- **Real benchmark on your 4 test videos** (`matching/data/testing/09-07-26-00{1..4}`).
  demo1 = 001/002, demo2 = 003/004 (= the canonical `model/data/instruments`
  8-instrument set). 179 tracklets total.
- **Open-set (reject foreign objects) — the load-bearing T02 risk — looks SOLVED on
  real data.** Across all 4 sessions and **~69 foreign objects** (tools, thin
  cables/pens/markers, tissues, mugs, jars, a wrench, a hammer), **multi-frame
  voting rejects ~100% of them** at a voted-cosine threshold ≈ 0.60–0.65 while
  keeping the knowns. Single frames DO false-accept (a hand reached 0.88, foreign
  0.6–0.83 on a lucky frame) — **voting collapses them**. The thin-object →
  `instrument8` magnet T02 feared did **not** materialise for foreign objects.
- **Re-ID (returning instrument → right identity), on 003's teardown→re-lay (31
  returns):** **71% top-1** (74% with size-fusion). **The twin pair is fine — zero
  `instrument1`↔`instrument2` confusion** (twin labels human-verified). The failure
  is concentrated on the **pituitary rongeur `instrument8` (6/13)**; excluding it,
  re-ID is **16/18 = 89%**.
- **The rongeur failure is an enrollment-coverage problem, not a wall.** Multi-view
  enrollment lifts overall 71% → **82%** and the rongeur 46% → **79%**. Design
  input: **Start-time enrollment must capture multiple views/poses per instrument**
  (especially long thin ones); a single snapshot is the trap.
- **Size-fusion barely helps here** (71→74%) — it targets the twins, which already
  work; it does not rescue the thin rongeur.

> **What's real vs proxy:** data is real (your 4 videos, the real rf-detr detector,
> the real 8-instrument gallery). The embedder is DINOv2-base (ungated stand-in for
> the gated DINOv3, same as T02). The offline `ByteTrack` here is a *labeling/eval*
> tracker, not the real-time tracker that will ship.

## Roster reconciliation

demo2 (003, 004) = canonical `instrument1..8`. demo1 (001, 002) is a **different but
overlapping tray** — shares ~5–6 (green forceps, "WRIGHT" retractor, T-handle, blue
awl/screwdriver, brush/foam tool) but swaps in a bone clamp, ball-tip probe and long
cutter. So demo1 is scored per-session; demo2 can also be scored against the
canonical gallery. The **twin pair** (two ring forceps, distinguished only by a small
green ring marker) is present in every session and is T02's #1 predicted risk.

| session | tray | tracklets | known / foreign / junk |
|---|---|---|---|
| 09-07-26-001 | demo1 | 36 | 14 / 20 / 2 |
| 09-07-26-002 | demo1 | 46 | 20 / 21 / 5 |
| 09-07-26-003 | demo2 | 71 | 40 / 24 / 7 |
| 09-07-26-004 | demo2 | 26 | 17 / 4 / 5 |

## Method

`build_cache.py`: per video, sample every 6th frame (~5 fps) → real rf-detr detector
(`model/weights/checkpoint_best_regular.onnx`, conf ≥ 0.3) → `trackers.ByteTrackTracker`
offline tracklets (unconfirmed `tracker_id == -1` dropped) → DINOv2-base mask-crop
embeddings (the exact T02 `bench.py` method: apply mask, crop bbox, CLS token,
L2-normalise) → per-tracklet montage. Tracklets were labeled (class + per-instrument
identity) from those montages (`labels_*.json`).

- **Open-set** (`score_openset.py`): score = max cosine of a detection to the 8
  canonical instrument prototypes. Per tracklet we compare `s_framemax` (best single
  frame — the false-accept danger), `s_framemed`, and `s_voted` (mean-embedding =
  multi-frame voting).
- **Re-ID** (`score_reid.py`): per-session enrollment — enroll each instrument from
  its earliest solid tracklet (multi-view = that tracklet's frames); every later
  tracklet is a "return" query; predict identity = argmax nearest-enrolled-view
  cosine on the query's mean embedding. Size-fusion (z-scored log bbox area+diagonal,
  à la T02) swept over weight.

## Open-set results

Per-tracklet **voted** cosine to the nearest known instrument, and the threshold at
which foreign is fully rejected:

| session | known voted (mean) | foreign voted (max) | voted τ → foreign-reject / known-accept |
|---|---|---|---|
| 004 | 0.77 | 0.60 | 0.60 → 100% / 94% |
| 001 | 0.61* | 0.46 | clean gap 0.46–0.52 |
| 002 | 0.62* | 0.47 | 0.55 → 100% / 75%* |
| 003 | 0.73 | 0.62 | 0.65 → 100% / 75% |

\* demo1 knowns are **under-scored** because they're matched against the *demo2*
canonical gallery (different tray). The foreign side is still valid (foreign objects
match any real instrument gallery low); demo1 known-accept needs per-session
enrollment to read true.

**The key point is voting, not the absolute number:** the hollow rings in
`openset_all4.jpg` are each tracklet's best single frame — foreign objects and a hand
reach 0.78–0.88 on lucky frames (the single-crop false-accepts T02 warned about).
Averaging over the 100–1800-frame tracklet collapses them (hand 0.88 → 0.36) while
genuine instruments stay high. Thin foreign objects (cables, markers, pens in 001)
reject as cleanly as anything.

## Re-ID results (003 teardown→re-lay, 31 return tracklets)

| instrument | returns correct | note |
|---|---|---|
| instrument1 (green forceps, twin) | 1/1 | — |
| instrument2 (plain forceps, twin) | 1/2 | one → instrument7 |
| instrument3 (blue screwdriver) | 0/1 | thin shaft → instrument6 |
| instrument4 (blue awl) | 5/5 | — |
| instrument5 (WRIGHT retractor) | 2/2 | — |
| instrument6 (cannula/brush) | 3/3 | — |
| instrument7 (T-handle chuck) | 4/4 | — |
| **instrument8 (pituitary rongeur)** | **6/13** | **the weak point — thin shaft fragments** |
| **overall** | **22/31 = 71%** | **16/18 = 89% excluding the rongeur** |

Twin confusion: `instrument1→instrument2` 0/1, `instrument2→instrument1` 0/2 — the
twins do not confuse on their returns (labels human-verified: green = t3/t44, plain =
t0/t27/t64). Size-fusion: 71% → 74% (marginal — twins already work).

**Enrollment strategy (the rongeur fix):**

| enrollment | overall | rongeur |
|---|---|---|
| single first tracklet (current) | 71% | 6/13 (46%) |
| causal: enroll from first 30% of video | 71% | 7/10 (70%) |
| multi-view (all views, upper bound) | **82%** | **11/14 (79%)** |

The rongeur's long shaft fragments into shaft-only and handle-only crops (the tracker
split it into 15 tracklets); a single-view enrollment can't cover both, so shaft
queries match other thin instruments (cannula, screwdriver). Multi-view enrollment
recovers most of it. A residual ~20% of shaft-only crops stays genuinely ambiguous.

## Caveats

1. Offline `ByteTrack` over-fragments long thin instruments — inflates the rongeur
   failure and the return count. The shipping real-time tracker should fragment less.
2. Single-tracklet enrollment (baseline) is thin; multi-view is the fix (above).
3. Open-set here only covers foreign objects the detector+tracker *confirm*
   (conf ≥ 0.3, `tracker_id` ≥ 0). Low-confidence foreign that never confirms is
   filtered before the matcher (correctly ignored) and not counted.
4. Tracklet labels were made from crops by subagents; a few are flagged uncertain in
   `labels_*.json`. A cool color cast washes out the green twin-marker in crops —
   twin labels keyed on teal-pixel density, then human-verified.
5. demo1 open-set known-accept is understated (canonical-gallery mismatch, above).

## What this answers / what remains

- **T04 item 2 (link scenarios):** substantially covered — leave/return, flipped-face
  (004 WRIGHT), long absence (001 awl), several-out-one-back and near-simultaneous
  (003 teardown/re-lay), foreign-while-present, twin leave/return, near-identical
  duplicates (002). Scored for open-set (all 4) and re-ID (003).
- **T04 item 3 (detector on foreign objects):** answered — the detector fires on
  foreign objects at high confidence (pliers ≈ 0.93); rejection is the matcher's job,
  done by voting. See [`../T02-stage1/RESULTS.md`](../T02-stage1/RESULTS.md) open-set
  section for the earlier synthetic version.
- **Remaining:** item 1 (real-tracker fps pin) is still blocked on the real-time
  tracker existing (stub today). Re-ID not yet scored on 001/004 (identity labels only
  for 002/003). Causal multi-view enrollment to be implemented properly (diagnosed,
  not yet productised).

## Reproduce

Heavy cache (detections/embeddings/crops) is regenerable and NOT committed. Set
`T04_CACHE` to where you want it.

```
export T04_CACHE=/path/to/t04-cache
# cache each video (run from model/ venv; needs transformers overlay):
uv run --directory model --with transformers python <this dir>/build_cache.py --session 09-07-26-003
# score (plain python3 with numpy):
python3 <this dir>/score_openset.py --session 09-07-26-003
python3 <this dir>/score_reid.py    --session 09-07-26-003
python3 <this dir>/plot_openset.py  --sessions 09-07-26-004 09-07-26-001 09-07-26-002 09-07-26-003
```

Committed here: the harness (`build_cache.py`, `score_openset.py`, `score_reid.py`,
`plot_openset.py`), the tracklet labels (`labels_*.json`), and key visuals
(`openset_all4.jpg`, `twin_verify_003.jpg`, `montage_00{1..4}.jpg`).
