---
id: T08
title: Gallery binding — make "Instrument N" mean the specimen, not the tracker counter
type: wayfinder:task
status: open
assignee:
blocked-by: []
---

## Question

Implement [`linker-design.md` §3 "Gallery binding (persistent references)"](../../../../model/docs/linker-design.md)
— the last unbuilt load-bearing piece of the linker. It is designed and grilled;
this ticket is the build, plus three corrections the design predates.

### Why now (the symptom Bram hit, 2026-07-15)

Live on the camera, the app showed **"Instrument 10"** on a tray of 8. Nothing is
broken — the number is simply `f"Instrument {tracker_id}"` where `tracker_id` is
Deep OC-SORT's raw counter (the linker sets `session_id = raw_id` at enrolment,
`session_linker.py`). Every transient detection before Start burns an id, so the
survivors land on arbitrary numbers. Three unrelated numbering systems exist
today and none are connected: dataset folder names (`instrument1..8`),
`CATALOG.md` ids (002, 007, 010...), and the raw counter on screen.

**Decision (Bram, 2026-07-15):** catalog names stay out of scope; `CATALOG.md`
can go. Generic **"Instrument N" is fine — but N must mean the specimen, and it
must rematch correctly on return.** That is exactly §3.

Building §3 fixes the symptom at its root: once a track is bound at enrolment,
the session id can *be* the specimen number — stable across every recording —
instead of whatever OC-SORT counted to. It also strengthens the return path for
free: a bound identity's gallery becomes `persistent ∪ Start` (15 known views +
Start crops) instead of today's ≤3 Start crops.

## What §3 specifies (unchanged, still correct)

- At enrolment close, embed each identity's Start crops, match against the ≤8
  persistent specimen galleries, solve a **one-to-one global assignment** — no
  two identities claim the same specimen.
- **Bind only when confident**: absolute threshold **and** best-vs-second margin.
- Bound identity's gallery = `persistent ∪ Start`.
- **Uncertain → session-only** (gallery = Start crops only), still a first-class
  identity, just fewer views. *Prefer session-only over a wrong bind.*
- **Bind once, at enrolment. Never re-bind mid-recording** (late bind poisons).
- **Degrades gracefully**: no persistent galleries load → everyone session-only,
  linker still works.

## Three corrections the design predates

1. **The match rule is SRC, not cosine.** §3 says "cosine-match against the ≤8
   persistent galleries"; §6 was rewritten 2026-07-15 and **explicitly superseded
   that rule** with SRC (joint sparse dictionary reconstruction + SCI). Build the
   assignment cost matrix from **SRC scores**, not raw cosine. Do not reintroduce
   crop-to-crop cosine — the champion computes no such similarity anywhere.

2. **Hungarian is not reachable through today's matcher seam.** §3 and old §6.5
   both assume a full row×column score matrix. The current interface
   (`interface.py`) is `score() -> dict[candidate, float]` then
   `accept(scores) -> one id | REJECT`, which **collapses each row to a single
   decision** — that is why the shipped batch path is greedy-per-identity, not
   `linear_sum_assignment` (see §6.5 as rewritten). Enrolment binding needs the
   matrix. Either:
   - call `score()` per identity and assemble the 8×8 matrix yourself, gate
     cells on τ/margin, then `scipy.optimize.linear_sum_assignment` over
     admissible cells (Hungarian *is* already a dependency — Deep OC-SORT uses
     it in `association.py`); **or**
   - do greedy one-to-one with the confidence gates, consistent with §6.5.

   Whichever is chosen, **state the choice and the reason in the design doc.**
   This is a real deviation point, not an implementation detail.

3. **Atom-count asymmetry — decide it deliberately, do not discover it.** SRC's
   dictionary is sensitive to how many atoms each candidate contributes: more
   atoms span a larger subspace, reconstruct *any* query (including foreign)
   with lower residual, and can attract more L1 mass — inflating both factors of
   `score = sim * sci`. Today Missing carries ≤6 atoms (Start ∪ last-seen) vs
   Active's ≤3, and **the CV sweep that tuned `tau=0.30`/`margin=0.02` never saw
   this** — `frozen/eval.py` gives every gallery an equal `n_gallery_groups=5`.
   §3 sharpens the question: a **bound** identity would carry ~18 atoms (15
   persistent + 3 Start) against an **unbound** one's 3.
   - All 8 bind → symmetric → fine.
   - Mixed bind → far worse imbalance than today, biasing every decision toward
     the identities that happen to be bound.

   **This must be measured, not assumed** (it was raised 2026-07-15 and never
   measured). Options if it bites: cap atoms per gallery, weight the dictionary
   by atom count, or renormalize per-candidate. Decide before shipping.

## Data

Already present — see [T07](T07-capture-reference-gallery.md):
`model/data/instruments/instrument{1..8}/images/` (15 views each, COCO masks)
and `model/data/other_objects/` (60 negatives). The code's TODO claiming *"T07's
photos don't exist yet"* is **stale — delete it** as part of this ticket.

⚠️ **Unverified assumption:** nobody has confirmed those photos are the same
physical specimens as the demo tray. §3 binds to a specimen, not a type. T07
carries that question; it does not block the build (graceful degradation), but it
decides whether binding helps.

## Acceptance

- Replay both takes via `model/scripts/replay_session.py --from-cache`: the
  roster binds to specimen numbers, and re-links after absence still resolve
  ≤ 1.0 s with the foreign windows (~29 s, 122–124 s, 165 s on Take B) still
  settling **Unknown**. No regression against T04's traces.
- A deliberate mis-bind attempt (a foreign object present at Start) stays
  session-only rather than claiming a specimen.
- Atom-asymmetry measurement recorded, with the chosen mitigation (or evidence
  that none is needed).
- 73+ model tests green; `session_linker.py`'s §3 TODO removed;
  `linker-design.md` §3 updated with the SRC + assignment-choice corrections.

## Out of scope

- Catalog names / `CATALOG.md` (Bram, 2026-07-15 — "Instrument N is fine").
- §5 stability-gated refresh, §8 rotation augmentation (§3's 15 real poses
  largely subsume it), §9 async execution (**closed by measurement, re-verified
  2026-07-15** — timing both halves of the synchronous path on the Take B cached
  replay: solve median 40.1 / max 70.1 ms (n=46), embed-at-death median 41.0 /
  max 67.3 ms (n=25), enrolment embed 260.4 ms once. Worst combined ~137 ms
  against a 333 ms frame at 3 fps).

⚠️ **§3 will change these numbers** — a bound gallery embeds **15 persistent
views + Start crops** instead of ≤3 crops. The 41 ms death-embed figure above is
for ~3 crops. Persistent galleries should be embedded **once at enrolment and
cached**, not re-embedded per event; if binding is implemented naively, the
enrolment freeze grows from 260 ms toward multiple seconds and §9 reopens.
**Re-measure `build_ms` after T08 lands** and record it in demo-validation.md.
