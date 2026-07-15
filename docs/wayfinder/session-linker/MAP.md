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

> **Re-cut 2026-07-15 — the map now tracks what is left, not how we got here.**
> The original destination ("ready to hand to implementation") was reached: the
> linker is built, composed in `load_tracker()`, and replay-validated. T01/T02/
> T04/T05 closed together on 2026-07-15 — they described the road to v1 and had
> gone stale against a shipped build. One thing remains genuinely unbuilt, and
> it now carries the frontier: **[T08 — gallery binding](tickets/T08-gallery-binding.md)**.

A **demo-ready session linker** means "Instrument N" identifies the *specimen*
and survives absence. Start-time enrolment, open-set SRC matching, and real-time
linking behind `InstrumentTracker` are **built and replay-validated** on both
takes. The last load-bearing gap is §3 gallery binding (T08): until it lands, the
session id is Deep OC-SORT's raw counter, so the tray shows arbitrary numbers
("Instrument 10" on a tray of 8) and returns rematch against ≤3 Start crops
instead of 15 known views. UZ Leuven demo **2026-07-20**.

**The frontier is three independent tickets** (plus [T07](tickets/T07-capture-reference-gallery.md),
a one-question confirmation for Bram that gates whether T08 *helps*):

- **[T09 — first live camera check](tickets/T09-first-live-check.md)** — validate
  what is already built. Every validation to date is replay (cached detections)
  or fake mode; no *validated* live run exists — no live fps pin, no pre-flight
  cache check. ⚠️ T08 cites a live symptom ("Instrument 10" on a tray of 8), so
  an **informal** live glance did happen; T09 carries the exact wording and
  Bram should correct it if that run went further.
- **[T08 — gallery binding](tickets/T08-gallery-binding.md)** — build what is
  left in the model (§3): make "Instrument N" mean the specimen.
- **[T10 — app-side Unknown & Pending](tickets/T10-app-side-unknown.md)** — build
  what is left in the app (§7). T06 decided it; nobody built it. **The audience
  will trigger this on purpose** by putting a phone on the table.

None blocks another, but two ordering facts matter:

- **T09's fps number lands first if it differs materially from the pinned 3.0** —
  every `*_s` window and OC-SORT's `max_age` derive from it, so T08's binding
  thresholds would otherwise be tuned against a wrong clock.
- **T09 will surface T10 live** ("the phone said Instrument 12"). That is T10, not
  a linker bug — don't debug the matcher for it.

## Notes

- **Deadline:** demo is 2026-07-20 — six days from charting. Bias every
  resolution toward the smallest thing that de-risks the demo.
- Canonical contracts: [`model/docs/linker-design.md`](../../../model/docs/linker-design.md)
  is **canonical for linker behaviour** (§6/§6.5 especially);
  [`model/docs/tracker-interface.md`](../../../model/docs/tracker-interface.md)
  § "Identity semantics" is the seam contract (≤ 1.0 s resolve, no leaks); and
  [`app/docs/DESIGN.md`](../../../app/docs/DESIGN.md) D8/D8a is the app-side
  contract. Decisions here must land back in those docs when tickets close.
  ⚠️ [`model/docs/session-linker-SPEC.md`](../../../model/docs/session-linker-SPEC.md)
  is a **superseded build record** — three of its clauses were deliberately
  reversed; never treat it as a contract.
- **Pinned matcher (2026-07-15, T02):** DINOv2-B (`facebook/dinov2-base`) CLS on
  masked crops, MPS, offline-forced singleton, + **SRC** (joint sparse dictionary
  reconstruction + SCI). CV re-ID 0.9333, foreign-reject 0.9733, 0 twin errors.
  Not DINOv3 — earlier notes said DINOv3; every guarded number is DINOv2-B's.
  A DINOv3-B swap is live R&D and needs no ticket here.
- Embeddings R&D: local checkout at `matching/` (repo `Ethilink/matching`,
  branch `feat/embeddings`) — the **catalog** problem (14 instruments,
  single-crop, 76.7% top-1), which is out of scope for the demo. The linker's
  session re-ID matcher is **vendored** into
  `model/src/orc_model/pipelines/matching/`; the durable research harness is
  `experiments/matcher-autoresearch/frozen/`.
- Standing constraints: the near-twin pair `instrument1`↔`instrument2` flagged in
  T02 stage 1 **did not materialize** — 0 twin errors on both metrics at round 3,
  so no instrument swap is needed (T01, closed). Link must resolve **≤ 1.0 s**
  after return.
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
- **Build, don't just plan** (2026-07-14, Bram; still in force) — a Notes
  override of wayfinder's plan-only default. Both external inputs have landed:
  detector weights are on the demo machine
  (`model/weights/checkpoint_best_regular.onnx`, **`.onnx` not `.pt`**), and the
  specimen photos exist (`model/data/instruments/instrument{1..8}/images/`,
  15 views each + 60 negatives in `model/data/other_objects/`). The one
  unconfirmed thing about the photos is whether they are the **same physical
  specimens** as the demo tray — that is [T07](tickets/T07-capture-reference-gallery.md),
  and it decides whether binding *helps*, not whether it *runs*.
- **Replay is the fast loop:** `model/scripts/replay_session.py --from-cache`
  re-runs tracker+linker over cached detections in ~2 min instead of ~25 min of
  inference. Detection caches for both 2026-07-15 takes live in
  `/private/tmp/orc-take-{a,b}-final-uncached.dets.npz` — **untracked and
  volatile; re-generate rather than trust them.**
- HITL tickets: use `/grilling` + `/domain-modeling`. The `matching/` checkout
  has its own skill (`LLM-guidelines-general`) that applies when touching its
  files.

## Decisions so far

Charted 2026-07-14 (grilling session, Bram):

- **Real-time, behind the seam** — D8 reaffirmed with commitment: identity
  resolution stays inside `InstrumentTracker` in `model/`, event-driven
  embeddings (compute only at track death/birth, not per frame); post-hoc-only
  rejected because it breaks the live Recording panel. *(The original
  feasibility estimate here — "~0.2–0.5 s per link event with DINOv3 ViT-B" —
  named the wrong model and was never measured end-to-end. Superseded by the
  §9 entry below, which has the real numbers.)*
- **Linker code lives in `model/`** — vendored, no cross-repo dependency on
  `matching` before the demo. *(Charted as "embed+cosine, ~100 lines"; what
  shipped is the ported SRC champion — see the §6 entry below.)* `matching`
  stays the R&D home for catalog identification.
- **Catalog identification is post-MVP** — demo labels stay "Instrument N";
  see Out of scope. Reaffirmed 2026-07-15, and `CATALOG.md` may go — but N must
  mean the *specimen* ⇒ [T08](tickets/T08-gallery-binding.md).
- **Builder: Claude agents under Bram** — built. The blessing question was
  overtaken by events; the build shipped in `model/` as charted
  ([T01](tickets/T01-sync-constantijn.md), closed).
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
  note. ⚠️ **§6.5 superseded 2026-07-15 — see the next entry.**
- **§6.5 structurally corrected; comparison ≠ eligibility** (2026-07-15) — the
  grilled §6.5 scored against the *Missing* dictionary and shrank it between
  rounds. That collapsed the ordinary one-missing case to **K=1**, where SCI is
  `0/0` and `sim` floors at ~0.47 above `tau=0.30` — measured, all 7 other
  instruments force-linked into instrument1's slot. Now: score against the
  **full frozen roster** (Active identities are the distractors SCI needs),
  filter eligibility (**Missing only**) *after* the gate, greedy per-identity
  assignment (loser → Unknown, never promoted to a second choice), **no** second
  re-score round, plus a **coasting-handoff deferral** for an OC-SORT id change
  one frame before its old id dies. A `cos_tau=0.60` absolute gate covers the
  genuine K=1 roster. Two of these (greedy, deferral) are engineering calls that
  were **not** grilled — flagged for Bram.
- **§9 async execution closed by measurement** (2026-07-15, re-measured) — the
  synchronous matcher inside `update()` is a non-problem. Both halves of the
  synchronous path were timed on the Take B cached replay (the first pass timed
  only the solve — the cheap half, and never the reason anyone wanted async):
  **solve** median 40.1 / max 70.1 ms (n=46); **embed at death** median 41.0 /
  max 67.3 ms (n=25); **embed at enrolment** 260.4 ms once, at Start, by design.
  Worst case both land on one frame: ~137 ms of linker work atop the detector's
  ~330 ms — a spike, not a stall, and three orders inside the 1.0 s resolve
  contract. TODO deleted rather than built. **This also kills the old
  "~0.2–0.5 s per link event" estimate above — a guess, ~6× pessimistic.**
  Reproduce via `replay_session.py --from-cache`; batch count (46 on Take B) is
  stable, absolute latency drifts with machine load.
- **Enrolment over-counting → workspace boundary** (2026-07-15) — enrolment froze
  **11** identities on a tray of 8; the extras were tripod/table objects below
  the mat, not tracker noise. Fixed with a fixed-camera
  `workspace_max_center_y_ratio` (0.88) applied before OC-SORT.
- **Catalog naming dropped for good** (2026-07-15, Bram) — "Instrument N is
  fine", `CATALOG.md` may go. But **N must mean the specimen and rematch on
  return** ⇒ that is [T08](tickets/T08-gallery-binding.md), not catalog work.
- **[Unknowns in the demo story](tickets/T06-unknown-objects-ux.md) is decided but
  UNBUILT app-side** (found 2026-07-15) — the model settles foreign objects as
  Unknown correctly (6/6 on Take B), but the app has no roster and still renders
  `"Instrument {id}"` for anything it sees, so a phone counts toward completeness.
  `linker-design.md` §7's premise ("nothing new crosses the seam; the app already
  has the frozen roster `{1…N}`") is **false on both halves** — no roster exists,
  and ids are OC-SORT's raw counter, not `{1…N}`. ⇒ [T10](tickets/T10-app-side-unknown.md),
  which carries a seam-contract choice that needs Bram, not a coder.

## Not yet specified

- **Link-failure recovery during the demo** — *partly cleared.*
  `linker-design.md` §6.5 now pins recovery as **lift-and-replace** (a bad link
  is corrected by picking the instrument up and putting it down again), and the
  failure modes are no longer hypothetical: two real fail-closed rejects on Take B
  (302.0 s, a swapped ring-handled forceps scoring 0.0688; 368.0 s, a flipped
  instrument staying Unknown instead of identity 6). Both fail **safe**. What is
  still unspecified is whether the operator does anything else, and that now waits
  on [T09](tickets/T09-first-live-check.md)'s live failure *rates* rather than on
  design. **Do not lower `tau` to hide these** — see `demo-validation.md`.
- **Evaluation harness on labelled data** — the only labelled real returns
  (31, video `09-07-26-003`, `assets/T04/labels_09-07-26-003.json` + `score_reid.py`)
  went unused; §6.5 was validated by reading traces on unlabelled July-15 takes.
  Without a scored baseline, every §3 claim in T08 (15 views beat 3 crops; atom
  asymmetry bites or doesn't) is decided by argument rather than measurement.
  Sharp enough to ticket **the moment the demo stops setting the clock** — it is
  the natural first step of any post-demo linker work.
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
