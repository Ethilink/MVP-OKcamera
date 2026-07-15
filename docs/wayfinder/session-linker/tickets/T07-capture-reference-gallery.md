---
id: T07
title: Capture the demo-set multi-view reference and negative gallery
type: wayfinder:task
status: open
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
