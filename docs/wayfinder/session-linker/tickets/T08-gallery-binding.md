---
id: T08
title: Gallery binding — make "Instrument N" mean the specimen, not the tracker counter
type: wayfinder:task
status: closed
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

## Resolution

Closed 2026-07-15 — **§3 shipped**. Built blind-TDD from
[assets/T08-T10/T08-SPEC.md](../assets/T08-T10/T08-SPEC.md) (distilled from §3,
this ticket, the [BUILD-BRIEF](../assets/T08-T10/BUILD-BRIEF.md), and a grilling
with Bram the same day). The acceptance replays below were run by the
orchestrator, not the coder. T09 (live camera) is Bram's and stays **open**.

### What shipped

- **`model/src/orc_model/pipelines/persistent_gallery.py`** (new) —
  `load_persistent_galleries()` scans `instrument{N}/`, decodes COCO RLE (loader
  logic **vendored** from `experiments/matcher-autoresearch/frozen/loader.py`, not
  imported), and makes **one `build_gallery()` call per specimen**. Never raises:
  a missing dir or an unparseable folder is a logged warning and `{}`.
- **Binding at the enrolment freeze** (`session_linker.py`) — SRC scores from the
  existing matcher against the **complete** persistent dict (Trap 2 applies to
  binding too: compare against everything), a per-identity `bind_tau`/`bind_margin`
  gate, **greedy one-to-one**, contested loser → session-only. Bound gallery =
  `persistent ∪ Start (∪ last-seen)`. Bind once, at the freeze; no later event
  re-binds.
- **Renumbering (B-N)** — a bound identity's `session_id` **is** its specimen
  number; session-only identities take `base + k`; every emission for a track that
  is not a roster identity (pending, deferred, settled-Unknown) is
  `raw_id + unknown_id_offset` (default 1000), so the roster and unknown id spaces
  are disjoint forever. This replaces `session_id = raw_id` **in all cases** — so
  "Instrument 10 on a tray of 8" is fixed even when no photos load at all.
- **Seam:** `roster` (frozenset) on the `InstrumentTracker` protocol, the real
  tracker, and `FakeInstrumentTracker`. Galleries load eagerly in `load_tracker()`
  and survive `reset()`. The stale §3 TODO ("T07's photos don't exist yet") is gone.
- `model/`: **124 tests pass** (73 before this build).

### Acceptance — every check green

Five cached replays (`--from-cache`): `take-a-t08` (Take A, `15-07-26-001`),
`take-b-t08` (Take B, `15-07-26-002`, all 8 galleries), `take-b-partial4` (4 of 8,
forced partial bind), `take-b-nogal` (binding disabled), `take-b-clean`
(quiet-machine latency). Traces were written to a session scratchpad — **volatile;
regenerate rather than hunt for them.**

- **Roster freezes at 8 on both takes, renumbered `{1..8}`.** Raw OC-SORT ids were
  non-contiguous (e.g. `{3,5,7,9,10,11,12,14}`) — B-N3's renumbering is what makes
  the tray read 1..8.
- **All 8 identities BOUND on both takes**; `session_only = []`.
- **0 full-roster comparison violations** — `comparison_galleries = 8` on every
  batch log.
- **Batch decision counts exactly 24 (A) / 46 (B)** — unchanged from the pre-T08
  baseline.
- **The six Take B foreign tracks all still settle Unknown**: t = 29.2, 122.2,
  122.4, 123.4, 123.8, 165.0 s.
- **Both known fail-closed rejects still fail closed**: 302.0 s →
  `{71: 'linked:1', 72: 'unknown'}`; 368.0 s → `{88: 'unknown'}`.
  `tau`/`margin`/`cos_tau` were **not** touched.
- ⚠️ **One acceptance line was never exercised on video:** "a deliberate mis-bind
  attempt (a foreign object present at Start) stays session-only". Both takes bind
  8/8 with `session_only = []`, so no real instance exists. The gate is covered by
  unit tests only (`test_b_b3_*` for the tau/margin gate, `test_b_b4_greedy_*` for
  loser → session-only).

### Trap 4 · atom-count asymmetry — MEASURED; no mitigation, with evidence

Atom spread per comparison gallery (B-O2 `atom_counts`, read off the batch logs):

| configuration | atoms per gallery | spread |
|---|---|---|
| no galleries (today's baseline) | 2..5 | 2.5× |
| all 8 bind (both real takes) | 17..20 | **1.18×** — better than today |
| mixed bind (forced 4 of 8) | 2..20 | **10.0×** — 4× worse than today |

**The brief's prediction is confirmed: mixed bind is the bad case.** And it changes
nothing measurable. `partial4` vs `full8`: **0 decision differences across all 46
batches**. The identity numbers differ only by B-N2 renumbering, and the permutation
is a **consistent bijection at every timestamp** (full8→partial4: 1→1, 2→2, 3→3,
4→4, 5→6, 6→8, 7→5, 8→7) — every link lands on the same *physical* instrument. The
10× dictionary-size bias does not flip a single decision: the champion's absolute
`cos_tau = 0.60` gate plus `bind_tau` dominate SRC's relative SCI bias.

**Mitigation: none.** Not capped, not weighted, not renormalized — there is nothing
to fix on this evidence, and each option would perturb a champion tuned by CV sweep.

**Honest limits.** One take, one partial configuration (specimens 1–4 — a
convenience split, not an adversarial one). **The mechanism is real**; only its
effect is nil here. **Re-check if `tau`/`cos_tau`/`bind_tau` ever move, or if a real
take ever produces a mixed bind** (neither take has yet).

### Trap 5 · embed once — held; `build_ms` re-measured, §9 stays closed

Persistent galleries are embedded **exactly once**, in `load_persistent_galleries()`
at `load_tracker()` time: **2926 ms for 8 specimens × 15 views, at startup**, never
per freeze. `reset()` keeps them.

Clean quiet-machine run (`take-b-clean`) vs the demo-validation baseline:

| cost | counter | n | median | max | was |
|---|---|---:|---:|---:|---:|
| SRC solve (score + assign) | `total_ms` | 46 | 43.1 ms | 79.3 ms | 40.1 / 70.1 |
| embed at track death | `build_ms` | 26 | 55.0 ms | 82.6 ms | 41.0 / 67.3 |
| embed at enrolment freeze | `build_ms` | 1 | 495.9 ms | — | 260.4 |

- The **freeze roughly doubles** (260 → 496 ms) and stays **2.0× inside the 1.0 s
  resolve contract** ⇒ **§9 stays closed, still by measurement.** By gallery count:
  240.1 ms (none) → 484.9 (4) → 495.9/508.9 (8).
- **The SRC solve barely moved** (43 vs 40 ms) even though the dictionary grew 3–8×.
  Persistent views cost far less at solve time than their atom counts suggest.
- **The death-embed rise (41 → 55 ms) is machine drift, not T08.** `_kill` is
  untouched and still embeds ≤3 session crops. Per the brief: **counts reproduce,
  milliseconds drift** — this is drift, not a finding.
- `n = 26` deaths vs 25 baseline = the extra lifecycle from the one re-identification
  below.

### What T08 actually bought — one rescued re-identification

Be honest: on these takes binding converts **exactly one Unknown into a correct
re-identification** — Take B, t = 261.0 s, raw track 60: `'unknown'` with no
galleries → `'linked:2'` with them. That single event is the entire 24-vs-23 link /
16-vs-17 unknown delta against `take-b-nogal`. **It is the only behavioural
difference binding makes on either take.** The renumbering (B-N3) — which needs no
photos — is the larger visible win.

### §8 is NOT subsumed — negative finding

The brief hypothesised that "§3's 15 real poses largely subsume §8
(rotation/mirror augmentation)" and asked for evidence either way. **The evidence is
negative:** with all 15 persistent views bound, the **368.0 s flipped instrument
still settles Unknown** rather than identity 6. §8 remains a real, unaddressed gap.
It fails safe, so it does not block the demo. Recorded in MAP.md "Not yet specified".

### ⚠️ Engineering calls Bram did NOT grill — FLAG

The last build silently reversed three grilled decisions and the map found out days
later. So, loudly:

1. **B-N5 freeze-frame flip — a SPEC AMBIGUITY the orchestrator RESOLVED, and it
   AMENDED [T08-SPEC.md](../assets/T08-T10/T08-SPEC.md) and migrated three tests.**
   The most important flag here. The original wording let the freeze frame read as
   **either** side of the boundary; the test-writer and the coder both read it
   leniently (flip on the *next* frame), Codex read it strictly. Ruled **strict**
   (the freeze frame itself emits session ids) because:
   (a) **the spec's own timing only works strict** — at the pinned 3 fps the freeze
   frame is t ≈ 0.67 s and the next is t ≈ 1.0 s, so only the strict reading is
   "~0.7 s in, inside the app's 1 s entry debounce"; the lenient one lands the flip
   *at* the debounce boundary and destroys the safety argument; and
   (b) **the lenient reading is a live cross-seam bug, not cosmetic lag** — the
   freeze publishes the session-id roster before the frame returns, so emitting raw
   ids on that frame lets them **collide** with session ids: a tray of 8 with raw ids
   `{3,5,7,9,10,11,12,14}` intersected against roster `{1..8}` yields `{3,5,7}` —
   presence booked against three instruments that are not on the table. T10's route
   (b) exists to remove exactly that class of bug.
   Three tests were migrated to the clarified contract
   (`test_b_n5_enrolled_ids_flip_from_raw_to_session_at_the_freeze`,
   `test_b1_2_passthrough_before_and_during_enrolment`, `test_b8_reset_drops_all_state`).
   **If Bram reads one flag, this is it: an ambiguous clause was resolved and the
   spec amended without him.**
2. **Greedy one-to-one binding, NOT Hungarian** — §3 says Hungarian; this ticket
   offered both. Reason: a global optimum would push a losing identity onto its
   **second-choice** specimen, and a wrong bind poisons every later decision while
   session-only merely has fewer views. So a contested loser falls back to
   session-only and never settles for second. (Consistent with §6.5's shipped
   philosophy — no forced links, ever.)
3. **The match rule is SRC, not cosine** — §3's "cosine-match" text predates §6's
   rewrite and is **stale**; the champion computes no crop-to-crop cosine anywhere.
   Built on SRC scores.
4. **`bind_tau = 0.30` / `bind_margin = 0.02`** start at the champion's values,
   unswept for binding. To be tuned at T07/T09.
5. **`bind_margin` is doing almost no work.** Measured: nearly every second-score is
   **0.0**, because the champion's `cos_tau = 0.60` K=1 gate zeroes hopeless
   candidates *before* the margin is consulted. The absolute `tau` gate carries the
   decision essentially alone. **Anyone tuning `margin` expecting leverage should
   know this first.**
6. **Instrument 3's Take A bind is razor-thin** — `bind_scores 3: (0.3007, 0.0928)`
   against `bind_tau = 0.30`: cleared by **0.0007 (0.23%)**. It binds today; any
   drift in lighting or pose sends it to session-only. Not a bug — it degrades
   safely, which is the design — but a real tuning signal **to know before the live
   camera test (T09)**. Take B's instrument 3 scored 0.4683, comfortably clear.
   `tau` was **not** lowered to widen the margin.
7. **B-O1's bound mapping is tautological.** The freeze log prints `bound` as
   `{session_id: specimen}`, but B-N1 makes `session_id == specimen` for every bound
   identity — so it always reads `{1:1, 2:2, …}` and reveals nothing about which
   **raw** tracker id claimed which photo. The code matches the spec exactly; **the
   spec asked for the wrong thing.** Suggest also logging `raw_id → specimen`: one
   line, real diagnostic value at T07/T09.
8. **B-L4 "unreadable images" was read per-image, not folder-level** — a folder is
   skipped only if *zero* crops survive. More robust, and matches "never raise": one
   corrupt jpg loads 14 views instead of dropping the specimen. Note the asymmetry:
   one unresolvable `image_id` still skips the whole folder via the outer `except`.
