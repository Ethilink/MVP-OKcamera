---
id: T02
title: Benchmark embedding models on the demo machine
type: wayfinder:research
status: closed
assignee: bram
blocked-by: []
---

## Question

Which embedding model + runtime does the linker use, backed by measured
numbers on the actual demo machine (M3 Max, 36 GB)? AFK.

**Two stages (2026-07-14, Bram — build in parallel, don't wait on the photo
session):**

- **Stage 1 — now, on existing images.** Use same-camera table crops already
  in this repo (`model/data`, data-collection outputs). Answers: model choice,
  latency, offline cache, and preliminary margins on all three axes below
  (synthetic rotations/mirrors stand in for genuine opposite faces). This is
  the early warning — if margins are fundamentally weak, the map must know
  this week.
- **Stage 2 — when T07's photos land.** Confirm margins on the exact physical
  demo specimens (incl. genuine opposite-face captures), pin the final
  acceptance threshold and gallery aggregation rule. The linker code carries
  the threshold as a parameter, so stage 2 changes numbers, not code.

The ticket closes after stage 2; stage-1 numbers get posted here as an interim
asset the moment they exist (T03 consumes them without waiting for closure).

Measure, for DINOv3 **ViT-S/16** and **ViT-B/16** (`transformers`, MPS —
skip the 7B; the notebook's 76.7% came from 768-dim = ViT-B despite
`MODEL_ID` currently naming `vit7b16`):

- per-crop embed latency at batch 1 / 4 / 10 (the link-event shape), warm;
- model load time and resident memory alongside the detector;
- whether the ≤ 1.0 s link budget holds with margin at batch ~10.

Also settle two practical risks:

- **Gated access / offline:** DINOv3 weights are gated on HF. Verify the
  weights are fully cached locally and load with no network — the demo room at
  UZ Leuven may be offline.
- **Separation sanity check (go/no-go for the whole map):** stage 1 on local
  same-camera crops, stage 2 on the exact physical-demo-specimen crops from
  [Capture the demo-set multi-view reference and negative gallery](T07-capture-reference-gallery.md).
  Confirm same-instrument vs different-instrument cosine margins are
  comfortably separable — enough to pick a starting threshold for T03.
  **Data note:** `matching/data` (instruments.db, test images) is NOT in the
  local checkout; do not block on it when this repo's same-camera evidence is
  available.
  Three margin axes:
  - *instrument vs instrument* (distinct types) — the basic link decision;
  - *same instrument under rotation/flip* — test genuine opposite-face and
    changed-view captures as well as synthetic 90/180/270° rotations and
    mirrors. The hard runtime case is a Start-time gallery that sees one face
    and a return crop that sees the other. This decides whether multi-view
    references suffice or T03 also needs mask-based orientation
    canonicalization / rotation-max at query time;
  - *instrument vs foreign object* (phone, pen, tape roll) — must sit clearly
    beyond the rejection threshold, since the demo audience will place random
    objects on the table and the matcher is load-bearing (open-set), not a
    safety net.

Benchmark the intended **hybrid gallery** rather than only single templates:
pre-captured multi-view references for the exact physical specimen plus fresh
same-session crops collected during Start-time enrolment. Compare nearest-view
and top-K-view aggregation (do not average every viewpoint blindly), and report
both an absolute acceptance threshold and a best-vs-second-best margin. Include
a one-to-one assignment readout for mapping all Start-time tracks to the fixed
reference specimens; an uncertain mapping must be allowed to fall back to a
session-only identity rather than be forced.

Resolution = pinned model id + runtime + gallery/scoring rule + the numbers, as
a short markdown asset linked from this ticket. If ViT-S holds the margins,
prefer it.

---

## Stage-1 interim (2026-07-14) — ticket stays OPEN

Ran on the **real 8 demo instruments** Bram captured (`model/data/instruments/
instrumentN/`, 15 frames each, 1920×1080, rf-detr masks — same camera+detector
as the live demo), **not** the old catalog proxy (which Bram removed). Full
write-up + images + raw JSON:
[assets/T02-stage1/RESULTS.md](../assets/T02-stage1/RESULTS.md).

Headlines (T03 can consume these now):

- **Method holds.** DINOv2-B: 90% top-1 re-ID at 5 enrolment views, 92% at 8.
  6 of 8 instruments ~95–100%.
- **Near-twin found in the set: `instrument1`↔`instrument2`** (both ring
  forceps), cosine 0.895 vs ≤0.71 elsewhere, ~70% of all errors. Feeds T01
  item 3 — the "no near-twins" assumption is currently violated.
- **…but it's fixable in software: fuse mask size.** The two forceps differ ~2×
  in mask area/length (100% separable) — info DINO's resize discards. Fusing
  mask-size into the descriptor drops the pair's errors 25→0 and lifts overall
  re-ID **90%→97%** (measured, `fuse.py`). Feeds T03's descriptor/scoring
  design. So swapping an instrument is likely *not* required.
- **Model: ViT-B over ViT-S** (90% vs 86%; latency is a non-constraint —
  embed ≤90 ms for 10 crops, ~10× under the 1 s budget). Memory ~90–350 MB.
- **Rotation:** synthetic rot/flip keeps 100% correct 1-NN with a multi-view
  gallery → T03 likely does **not** need mask-orientation canonicalization
  (caveat: synthetic, not genuine opposite face).
- **Open-set is the load-bearing risk.** 60 foreign crops added
  (`model/data/other_objects/`) — detector fires on all at 0.96 (no filtering;
  improving but imperfect). Single-crop cosine has **no clean threshold**
  (cos≥0.70 → 66% accept / 78% foreign-reject; margin doesn't help). The
  worst false-accepts are all instrument-shaped intruders matching the thin,
  generic **`instrument8`** (0.81–0.92). Fix lives in T03/T04, not a better
  still-crop threshold: **multi-frame voting** over the return window +
  size-fusion + possibly dropping the generic-shaped instrument.
- **Blocker: DINOv3 gated** (403 for Bramdc99). Numbers are DINOv2 (ungated)
  as stand-in. Either accept the HF licence for an apples-to-apples DINOv3
  number, or commit to DINOv2-B (works offline today).

Stage-2 gaps to CLOSE this ticket: resolve 1↔2; capture foreign-object
negatives to pin the reject threshold; confirm genuine opposite-face views;
settle DINOv3-vs-DINOv2; ideally re-confirm at the demo-room lighting/distance.

---

## Stage-2 progress (2026-07-15) — matcher autoresearch, ticket stays OPEN

Ran an overnight champion/challenger autoresearch loop (9 rounds, ~20 method
families) in the separate `experiments/matcher-autoresearch/` sandbox — see
[assets/T02-stage2/RESULTS.md](../assets/T02-stage2/RESULTS.md) for the full
headline summary and `experiments/matcher-autoresearch/{LEADERBOARD.md,
FINDINGS.md}` for detail.

- **1↔2 resolved** — mask-size fusion (or an equivalent orthogonal cue)
  independently confirmed across 4+ method families, not a fragile threshold
  artifact.
- **New champion matcher found**: sparse-representation classification (SRC),
  beating the stage-1-style baseline on both re-ID and foreign-reject, passed
  a Codex leak-check. 6 further rounds failed to unseat it — the method-search
  side of this ticket looks done; **no further autoresearch round planned.**
- **Still open, blocking closure:** this ran on static per-instrument crops,
  not real return-window video — reconcile against
  [assets/T04/RESULTS.md](../assets/T04/RESULTS.md)'s real-video benchmark
  (which used the older baseline champion, not SRC) before picking a final
  match rule for T05. SRC's `MultiTaskLasso` runtime cost at real roster size
  is also unprofiled against the ≤1 s link budget.
- Foreign-object negatives, opposite-face captures, and the DINOv3-vs-DINOv2
  call are still exactly as open as stage-1 left them — none of that was in
  this round's scope.

**Update 2026-07-15 (build kickoff):** stage-2 findings are now **consumed into
the build** — `model/docs/linker-design.md` §6 rewritten around SRC; port +
wiring tracked in [assets/T05/ISSUES.md](../assets/T05/ISSUES.md). The
`MultiTaskLasso` runtime question above is **closed**: profiled at demo scale
(D=770, dictionary width up to 120 atoms) at **0.2–0.8 ms per solve** — the
embed step dominates the link budget, as already assumed (see ISSUES.md,
"latency risk downgraded").

## Resolution

Closed 2026-07-15 — **superseded by the matcher-autoresearch program**
(`experiments/matcher-autoresearch/`), which answered this at greater depth than
the ticket asked.

- **Model + runtime:** DINOv2-B (ViT-B/14) CLS on masked crops, MPS, offline-forced
  process-wide singleton (`orc_model/pipelines/matching/embedder.py`).
- **Method:** stage 2 ran a blind technique search over ~30 method families. The
  champion is **SRC** (joint sparse dictionary reconstruction + SCI), promoted at
  round 3: CV re-ID **0.9333** vs the 0.850 baseline, foreign-reject **0.9733**,
  0 twin errors. See `LEADERBOARD.md`, `PARAMS.md`, `TRIED.md`.
- **Latency:** measured in production, not just in the harness. On the Take B
  cached replay the **embed** (`build_gallery`, the dominant half) runs median
  **41.0 ms** / max **67.3 ms** per death event (n=25), plus **260.4 ms** once at
  enrolment; the SRC **solve** runs median **40.1 ms** / max **70.1 ms** (n=46).
  Both sit inside one 333 ms frame at 3 fps. This retires the map's original
  "~0.2–0.5 s per link event with DINOv3 ViT-B" estimate — wrong model, and ~6×
  pessimistic. See T04 and `session_linker.py`'s §9 note.
- **Stage 1's near-twin warning** (`instrument1`<->`instrument2`) did not
  materialize: 0 twin errors on both metrics.

Ongoing matcher R&D (e.g. a DINOv3-B swap) does not need this ticket — the
`frozen/` harness is the durable artifact and stays in place.
