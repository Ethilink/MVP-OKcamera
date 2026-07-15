---
id: T07
title: Capture the demo-set multi-view reference and negative gallery
type: wayfinder:task
status: closed
assignee: bram
blocked-by: []
---

## Question

Create the small, auditable image set that the embedding benchmark and linker
design need for the exact physical demo specimens — data preparation, not the
runtime implementation.

**Bram captures this himself (2026-07-14)** — he has the 8 demo objects, so
this does not wait for the T01 conversation; T01 still *confirms* the set with
Constantijn (no near-twins), so hold off publishing final galleries if that
confirmation might swap an instrument. The checklist below is the handoff.

For each of the 5–8 physical instruments expected in the demo, assign a stable
specimen key (not a semantic catalog name) and capture roughly 10–20 diverse,
clean views with the intended camera/table/lighting where possible: both faces,
in-plane rotations, modest tilt, position/scale variation. Avoid hundreds of
near-identical consecutive frames. Keep the source crop/image and a manifest so
every reference is traceable to exactly one physical specimen.

Also capture a negative set of likely audience objects (phone, pen, tape roll,
packaging and similar) plus hard instrument-vs-instrument alternatives. Clean
multi-view specimen images are reference candidates; occluded, clipped,
flipped-return, and foreign-object examples primarily belong in evaluation so
the gallery does not learn bad views merely to improve its own test score.

This task does not choose masked vs bbox crops, an embedding model, thresholds,
or scoring. It supplies the raw evidence for [Benchmark embedding models on the
demo machine](T02-benchmark-embeddings.md), which makes those decisions.

Resolution = gallery folder + manifest location recorded here, physical demo
set confirmed, and enough coverage to run the hybrid-gallery/viewpoint-flip and
open-set rejection benchmark without inventing synthetic evidence.

## Status update — 2026-07-15: the data appears to already exist

Re-scoped to a **one-question confirmation for Bram**. This ticket was written as
"go capture the image set", and the code still carries a TODO saying *"T07's
photos don't exist yet"* (`session_linker.py`). That statement looks stale:

- `model/data/instruments/instrument{1..8}/images/` — **15 views each** (120 crops
  with COCO masks)
- `model/data/other_objects/` — **60** adversarially tool-like negatives
- Both are exactly what §3 needs, and both were what the matcher-autoresearch
  benchmark and its tuned tau/margin ran on.

**The only open question — and nobody has checked it:** are those photos of the
*same physical specimens* that will be on the UZ Leuven demo tray? §3 binds to a
specimen, not to an instrument type, so this is load-bearing:

- **Yes** -> [T08](T08-gallery-binding.md) is unblocked with the data in hand.
- **No** -> a fresh capture is needed (the original checklist below still applies),
  or T08 ships knowing every identity will fall back to session-only.

This does **not** block T08 from being built: §3 degrades gracefully by design —
no usable persistent gallery means everyone runs session-only and the linker
still works. The answer decides whether T08 *helps*, not whether it *runs*.

## Resolution

Closed 2026-07-16 (grilling, Bram) — **the one open question is answered: YES.**
`model/data/instruments/instrument{1..8}` are the **exact physical specimens**
that go on the UZ Leuven demo tray (Bram confirmed, he has them in hand). So
binding genuinely *helps* — it is not merely degrading-gracefully theatre. T08 is
vindicated with the data in hand.

**Correction the wayfinder was missing — there are two capture sessions, 30
views per specimen, not 15:**

- `model/data/instruments/` — **session 1**, 8 × 15 views, `date_created`
  2026-07-14 ~16:49. This is `DEFAULT_INSTRUMENTS_DIR` → **what the demo binds.**
- `model/data/instruments_session2/` — **session 2**, 8 × 15 views, 2026-07-15
  ~13:38 (same day as takes `001/002`). Freshly discovered; used by no code path,
  named in no doc until now.
- `model/data/other_objects/` — 60 negative crops across 5 frames (benchmark
  asset, not a runtime input; the "60 negatives" figure was crops, not images).

Both sessions are of the same 8 physical specimens under (Bram) different
conditions, both representative of the demo table.

**Which gallery the demo uses was grilled and measured — see the bake-off in
`model/docs/demo-validation.md` and MAP Decisions (2026-07-16). Verdict: keep
session 1. Do NOT merge to 30** — merging fattens instrument 3's knife-edge bind
(+0.0007 → +0.0958) but *loses a genuine return re-identification* (instrument 1
at Take A t=180.4s flips `linked:1` → `unknown`), and the lost return is the
demo's money-shot while the thin bind fails safe to session-only. Session 2 and
merged-30 are documented post-demo experiments; the merged builder + all six
bake-off traces live in the session scratchpad (volatile).
