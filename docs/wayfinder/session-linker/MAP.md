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
> gone stale against a shipped build.
>
> **Updated 2026-07-15 (evening): [T08](tickets/T08-gallery-binding.md) and
> [T10](tickets/T10-app-side-unknown.md) shipped and closed.** Nothing
> load-bearing is unbuilt any more. The frontier is now
> **[T09 — first live camera check](tickets/T09-first-live-check.md)** alone —
> everything the demo needs exists, and none of it has met a live camera.

A **demo-ready session linker** means "Instrument N" identifies the *specimen*
and survives absence. Start-time enrolment, open-set SRC matching, §3 gallery
binding, and real-time linking behind `InstrumentTracker` are **built and
replay-validated** on both takes; the app renders foreign objects gray and keeps
them out of Completeness (§7). "Instrument N" now means the specimen — the session
id is the bound specimen number, not Deep OC-SORT's raw counter, and returns
rematch against `persistent ∪ Start` instead of ≤3 Start crops. What is left is
**proving it live**. UZ Leuven demo **2026-07-20**.

**The frontier is now one ticket.** [T07](tickets/T07-capture-reference-gallery.md)
**closed 2026-07-16: the photos ARE the demo specimens (Bram), binding helps.**
That leaves [T09](tickets/T09-first-live-check.md) alone:

- **[T09 — first live camera check](tickets/T09-first-live-check.md)** — validate
  what is built. Every validation to date is replay (cached detections)
  or fake mode; no *validated* live run exists — no live fps pin, no pre-flight
  cache check. ⚠️ T08 cited a live symptom ("Instrument 10" on a tray of 8), so
  an **informal** live glance did happen; T09 carries the exact wording and
  Bram should correct it if that run went further.

Two facts carry into T09:

- **Its fps number lands first if it differs materially from the pinned 3.0** —
  every `*_s` window, OC-SORT's `max_age`, and the 2-frame enrolment window derive
  from it, so T08's binding thresholds (and the freeze-frame timing argument behind
  B-N5) would otherwise rest on a wrong clock.
- **T09 is the first live test of T10's Unknown path** — a phone on the tray should
  render gray "Unknown" and stay out of Completeness. If it still says "Instrument
  12", that is a T10 regression, not a linker bug — don't debug the matcher for it.

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
  specimen photos exist. **There are TWO capture sessions of the same 8 specimens,
  30 views total** (corrected 2026-07-16): `model/data/instruments/` (**session 1**,
  15 views, the `DEFAULT_INSTRUMENTS_DIR` the demo binds) and
  `model/data/instruments_session2/` (**session 2**, 15 views, used by no code path);
  plus 60 negative crops in `model/data/other_objects/`. **[T07](tickets/T07-capture-reference-gallery.md)
  is CLOSED (2026-07-16): the photos are the demo specimens, binding helps.** The
  gallery choice was grilled + measured — **keep session 1, do not merge to 30**
  (§ Decisions 2026-07-16 and `model/docs/demo-validation.md`).
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

Grilled 2026-07-16 (remote-control session, Bram):

- **[T07](tickets/T07-capture-reference-gallery.md) closed YES + a data
  correction.** The `instrument{1..8}` photos are the exact demo-tray specimens
  (Bram), so binding helps. Discovered in the same pass: there are **two capture
  sessions, 30 views total** — `instruments/` (session 1, what the demo binds) and
  the previously-undocumented `instruments_session2/` (session 2). The wayfinder's
  "15 views" was per-session.
- **Gallery for the demo = session 1; do NOT merge to 30 (measured).** A three-way
  cached bake-off on both takes (session 1 / session 2 / merged-30) settled the
  "mix them" instinct empirically: on Take B all three are decision-identical;
  on Take A merging **rescues instrument 3's knife-edge bind** (+0.0007 → +0.0958)
  **but loses a genuine return** (instrument 1 at t=180.4s flips `linked:1` →
  `unknown`). Both failures are fail-safe, but session 1's weakness (thin bind →
  session-only) is audience-invisible while merged's (a return going Unknown) is
  the demo's money-shot. Session 2 / merged are post-demo experiments. Full table
  in `model/docs/demo-validation.md`.
- **Offline cold-start verified — the demo-day killer is cleared (this machine).**
  With no internet at the venue (Bram), a cold model cache would hang
  `load_tracker()` at startup. Forced-offline construction succeeds in **16.6 s**
  (DINOv2 + mobilenet embedder + ONNX all cache-resident, 8 galleries). ⚠️ Verified
  on *this* machine — re-run with the network actually off on the demo box if it
  differs. This is [T09](tickets/T09-first-live-check.md)'s gating pre-flight.
- **`orc-demo --debug` built (T09 prep) + B-O1 fixed.** A behaviour-free,
  off-by-default event console (`backend/debug.py`) renders the pipeline narrative
  — gallery load, an ENROLMENT FREEZE block, each link/unknown/deferral, deaths —
  from structured `record.orc` payloads on the `orc_model` loggers. The freeze log
  now exposes **`raw_id → specimen`** (T08 flag B-O1: the old `{session_id:specimen}`
  was tautological), so a live mis-bind is diagnosable. `ORC_DEBUG=1` also enables
  it. Model 125 / backend 209 green.
- **Nothing blocks T09** — it can run once the offline pre-flight is confirmed on
  the demo machine; Bram runs it. The eight T08/T10 engineering flags are ratify/
  defer/demo-polish, none gating (dispositions being recorded on the tickets).

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
- **[§3 gallery binding shipped — "Instrument N" now means the specimen](tickets/T08-gallery-binding.md)**
  (2026-07-15) — persistent galleries embedded **once** at `load_tracker()` (2926 ms,
  startup only); greedy one-to-one bind at the freeze; bound gallery =
  `persistent ∪ Start`; uncertain → session-only. Both takes bind **8/8**, roster
  renumbered `{1..8}`, batch counts (24/46) and both fail-closed rejects unchanged.
  **Renumbering, not binding, is the visible win** — it fixes "Instrument 10" even
  with no photos loaded, while binding itself buys exactly **one** rescued
  re-identification on Take B. Trap 4 measured (mixed bind = 10× atom spread, **0**
  decision differences ⇒ no mitigation); freeze `build_ms` 260 → 496 ms, still 2× inside
  the 1.0 s contract ⇒ §9 stays closed. **Eight ungrilled engineering calls flagged in
  the resolution — read #1 first: an ambiguous B-N5 clause was resolved *strict* and
  the SPEC amended by the orchestrator.**
- **[§7 app-side Unknown & Pending shipped — the roster crosses the seam](tickets/T10-app-side-unknown.md)**
  (2026-07-15, grilling) — **route (b)**: `tracker.roster` is on the protocol and
  sampled in the **same tick** as `present_ids`, which kills route (a)'s coordination
  risk and, with it, §7's "nothing new crosses the seam". Unknowns are **video-only**
  (gray mask + resolving spinner, then gray "Unknown"; no panel row, out of
  Usage/Completeness); each roster instrument's `/status` entry carries a **`colour`**
  so the panel swatch and the overlay mask are the same hex and a return regains its
  hue. Backend **and** frontend halves both shipped. **Seven ungrilled calls flagged in
  the resolution**, including a named-and-accepted renderer race, a pre-existing
  monotonic-clock race in `orc-demo` today, and a BRANDING.md conflict over the 8-hue
  palette.

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
- **§8 rotation/mirror augmentation — a real gap, no longer a hypothesis.** T08's
  brief assumed "§3's 15 real poses largely subsume §8" and asked for evidence either
  way. The evidence is **negative** (2026-07-15, measured): with all 15 persistent
  views bound, Take B's **368.0 s flipped instrument still settles Unknown** instead
  of identity 6. §3 does not cover §8; the gap is unaddressed. It fails **safe**, so
  it does not block the demo — ticket it if flip-returns ever have to link, or if
  [T09](tickets/T09-first-live-check.md)'s live failure *rates* say it matters.
  (**Do not lower `tau` to make it pass** — same rule as the two rejects above.)
- **Evaluation harness on labelled data** — the only labelled real returns
  (31, video `09-07-26-003`, `assets/T04/labels_09-07-26-003.json` + `score_reid.py`)
  went unused; §6.5 was validated by reading traces on unlabelled July-15 takes.
  Without a scored baseline, T08's §3 claims were settled by **trace-reading on two
  unlabelled takes**, not by scoring: "15 views beat 3 crops" is worth exactly **one**
  rescued re-identification there, and the atom-asymmetry mitigation rests on **zero
  decision differences** across 46 batches of one forced-partial configuration. Both
  are honest measurements of a very narrow sample.
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
