---
label: wayfinder:map
title: Session linker — real-time track identity for the UZ Leuven demo
created: 2026-07-14
tracker: local-markdown
---

# Map — Session linker

**Tracker conventions (local-markdown):** tickets live in [`tickets/`](tickets/),
one file each, id = `T##`. Frontmatter carries `status` (open/closed),
`assignee` (a set assignee IS the claim), `blocked-by` (list of ids — markdown
has no native blocking, this is the fallback convention), and `type`
(`wayfinder:research|prototype|grilling|task`). The **frontier** = open,
unblocked, unassigned tickets. Resolve a ticket → append `## Resolution` to its
file, set `status: closed`, add one line under Decisions-so-far here.

## Destination

The way to a **demo-ready session linker** is clear: Start-time enrolment,
hybrid reference galleries, and real-time track linking behind
`InstrumentTracker` in `model/` are fully specified — build blessed by
Constantijn, embedding model pinned by benchmark on the demo machine, linker
design doc written, eval scenarios in the harness — ready to hand to
implementation, in time for the UZ Leuven demo **2026-07-20**.

## Notes

- **Deadline:** demo is 2026-07-20 — six days from charting. Bias every
  resolution toward the smallest thing that de-risks the demo.
- Canonical contracts: [`model/docs/tracker-interface.md`](../../../model/docs/tracker-interface.md)
  (§ "tracker_id across absence" — linking behind the seam, ≤ 1.0 s resolve,
  no leaks) and [`app/docs/DESIGN.md`](../../../app/docs/DESIGN.md) D8.
  Decisions here must land back in those docs when tickets close.
- Embeddings R&D: local checkout at `matching/` (repo `Ethilink/matching`,
  branch `feat/embeddings`); the method is `notebooks/embeddings.ipynb` —
  DINOv3 crops → CLS embedding → cosine 1-NN. 76.7% top-1 on the *catalog*
  problem (14 instruments incl. near-twins, single-crop); the linker's
  session-scoped problem is strictly easier (5–8 distinct instruments,
  same-camera galleries, multi-frame voting — though multi-removal makes
  multi-candidate matching the normal case, see choreography below).
- Standing constraints: demo set has **no duplicate/visually-similar types**
  (⚠️ **now in question** — T02 stage 1 found a near-twin pair
  `instrument1`↔`instrument2` in Bram's real captured set; decision pending in
  [Sync with Constantijn](tickets/T01-sync-constantijn.md) item 3);
  link must resolve **≤ 1.0 s** after return.
- **Identity vocabulary:** raw tracker ids are disposable implementation detail;
  the linker emits stable **session ids** for the physical objects enrolled at
  Start. Pre-captured references, where available, are keyed to those exact
  physical demo specimens, not to semantic catalog names or instrument types.
- **Demo choreography (updated 2026-07-14, Bram):** snapshot the table at
  Start, then instruments get moved, occluded, and **removed several at a
  time**, returned in any order — and the audience will place **random foreign
  objects** on the table. One-at-a-time is NOT guaranteed. ⇒ the embeddings
  matcher is **load-bearing** (open-set: match against all missing galleries,
  reject below threshold), not a safety net; the one-missing heuristic is at
  most a tie-break prior. `Session` already handles ≥ 2 simultaneous absences.
- **Build in parallel (2026-07-14, Bram):** everything gets spec'd and built
  without waiting on externals. Exactly two external inputs exist:
  (1) **Bram's photo session** of the 8 demo specimens (T07) — feeds the
  reference galleries and the final threshold numbers; these are *data and
  parameters*, never code; (2) **detector weights from Constantijn** (T01) —
  needed only for the fps pin, live-camera tuning, and
  detector-on-foreign-object evidence; the harness replays cached detections,
  so code + tests never wait for weights. When T05 closes, the build starts
  immediately (Notes override of wayfinder's plan-only default).
- HITL tickets: use `/grilling` + `/domain-modeling`. The `matching/` checkout
  has its own skill (`LLM-guidelines-general`) that applies when touching its
  files.
- Housekeeping: `matching/` is an untracked nested git repo inside this repo —
  gitignore it or move it to a sibling directory (fold into any convenient
  ticket).

## Decisions so far

Charted 2026-07-14 (grilling session, Bram):

- **Real-time, behind the seam** — D8 reaffirmed with commitment: identity
  resolution stays inside `InstrumentTracker` in `model/`, event-driven
  embeddings (compute only at track death/birth, not per frame); post-hoc-only
  rejected because it breaks the live Recording panel. Feasibility confirmed:
  ~0.2–0.5 s per link event with DINOv3 ViT-B on the M3 Max.
- **Linker code lives in `model/`** — vendored embed+cosine (~100 lines);
  no cross-repo dependency on `matching` before the demo. `matching` stays the
  R&D home for catalog identification.
- **Catalog identification is post-MVP** — demo labels stay "Instrument N";
  see Out of scope.
- **Builder: Claude agents under Bram** — pending Constantijn's blessing
  ([T01](tickets/T01-sync-constantijn.md)), since it crosses into his half.
- **Matcher scope = session re-identification, open-set** (2026-07-14
  follow-up): the matcher only decides "same physical object as one that
  left?" — never which catalog instrument it is. Multi-removal + foreign
  objects in the demo make it load-bearing; heuristic-only rejected.

<!-- one line per closed ticket below -->

- [Unknowns in the demo story — what the app shows for unmatched new tracks](tickets/T06-unknown-objects-ux.md)
  — freeze the known roster at Start; unmatched newcomers may be tracked and
  shown as Unknown but never enter Usage or Completeness.
- [Pin the session-linker design](tickets/T03-linker-design.md) — full behaviour
  pinned in [`model/docs/linker-design.md`](../../../model/docs/linker-design.md):
  ~0.5 s enrolment freeze; session id = promoted OC-SORT id in one int space
  (Unknown = not-in-roster, seam unchanged); hybrid `persistent ∪ Start` galleries
  auto-bound one-to-one (session-only fallback); masked crops, nearest-view max;
  stability-gated refresh ON+toggle; open-set match with τ+margin+multi-frame,
  Hungarian for simultaneous returns, one-missing = margin-skip-only; Pending
  spinner + gray Unknown + 8 roster colours app-side; rotation via gallery-side
  augmentation, PCA canonicalization md-only pending T02; Deep OC-SORT with
  `max_age ≈ 1 s` + reactivation re-validation; standalone `SessionLinker` in
  `load_tracker()`. Numbers are T02's; fps/`max_age` are T04's.
- **§6 match rule replaced by SRC; §6.5 batched assignment pinned** (2026-07-15,
  grilling) — T02 stage-2's autoresearch champion (SRC: joint sparse dictionary
  reconstruction + SCI) supersedes nearest-view-max cosine in
  `model/docs/linker-design.md` §6; §6.5 = natural batching → per-row τ/margin
  gates → assignment → one extra re-score round, no forced links ever. Build
  brief: [assets/T05/ISSUES.md](assets/T05/ISSUES.md); T03 has a post-closure
  note.

## Not yet specified

- **Link-failure recovery during the demo** — Unknown presentation is decided,
  but what the operator does after a wrong or persistently ambiguous link (if
  anything) still depends on the measured failure modes and rates.
- **Shared embedding package (post-demo)** — the linker in `model/` and the
  catalog matcher in `matching/` will converge on one embed primitive; extract
  it once both sides stabilize.

## Out of scope

- **Catalog identification in the demo** (real instrument names from the
  template DB) — post-MVP; 76.7% top-1 with twins at 20% recall is not
  demo-safe, and it adds catalog-building work in the final week. Stays R&D in
  `matching/`. A multi-view reference gallery for the exact fixed physical demo
  specimens remains in scope; it strengthens session identity while labels stay
  generic "Instrument N". (Ruled out while charting, 2026-07-14.)
- **Post-hoc report-correction pass** — a second, after-Stop matching system;
  rejected with the real-time commitment.
- **Cross-session identity / recording history** — already a non-goal (D7).
