# Session linker — design

**Status:** pinned 2026-07-14 (grilling with Bram, wayfinder ticket T03);
**§6 rewritten + §6.5 added 2026-07-15** — the matcher-autoresearch champion
(SRC) replaced §6's original match rule; §6.5's batched assignment grilled with
Bram the same day. Numeric
thresholds are **parameters filled by [T02](../../docs/wayfinder/session-linker/tickets/T02-benchmark-embeddings.md)**;
`max_age` and fps-sensitive gaps are tuned by
[T04](../../docs/wayfinder/session-linker/tickets/T04-harness-and-fps.md). This
doc pins *behaviour*; the build is [T05](../../docs/wayfinder/session-linker/tickets/T05-implementation-spec.md).

Companion to the seam contract in [`tracker-interface.md`](./tracker-interface.md)
(§ "Identity semantics") and the demo app in
[`app/docs/DESIGN.md`](../../app/docs/DESIGN.md) (D8/D8a).

---

## What the linker is for

An instrument leaves the table and comes back. A plain tracker (we use
**Deep OC-SORT**, vendored — see below) spawns a *fresh* id on return. The seam
promises the opposite: an instrument returns under its **original** id, within
≤ 1.0 s, with no id-rewrites leaking to the consumer
(`tracker-interface.md` § "Identity semantics"). The **session linker**
is the layer that keeps that promise. It is **session re-identification, open-set**:
it only ever decides *"is this the same physical object as one that left?"* —
never *which catalog instrument it is*. Labels stay generic "Instrument N";
catalog identification is out of scope (map decision).

The demo choreography makes this load-bearing, not a safety net: instruments are
moved, occluded, and **removed several at a time**, returned in any order, and the
audience places **random foreign objects** on the table. So the matcher must be a
real open-set, multi-candidate matcher, not a one-missing heuristic.

---

## Glossary

- **Raw (tracker) id** — the id Deep OC-SORT assigns. Monotonic from a counter
  that `reset()` zeroes; emitted as `id+1`, so ids start at 1. Disposable
  implementation detail; **never crosses the seam.**
- **Session id** — the stable id a physical object keeps for the whole recording,
  even across absence. This is the *only* identity a consumer sees, carried in
  the `tracker_id` field of `sv.Detections`.
- **Known roster** — the set of session ids enrolled at Start, frozen at the
  enrolment freeze; a track born later never joins it. **Not necessarily
  `{1…N}`** — §3's binding assigns the numbers, and a partial bind leaves gaps.
  Published across the seam as `InstrumentTracker.roster` (§7).
- **Enrolment window** — the ~0.5 s after `reset()` during which the roster is
  formed and each identity's Start gallery is seeded.
- **Gallery** — the set of per-view DINOv2-B embeddings that represent a
  session identity (§6's SRC uses each view as a dictionary atom). Composed of **persistent references** (pre-captured, T07) ∪ **Start
  crops** (fresh, same-camera) ∪ optional **session-refresh** views.
- **State** of a session identity: **Active** (a live raw track is mapped to it),
  or **Missing** (its raw track died; gallery retained, awaiting a return).
- **Pending** — a young raw track not yet in the roster, still being decided
  (returning instrument vs foreign object). **Unknown** — a settled track that
  matched no missing identity. Both are *derived app-side*, not model states (see
  § App-side encoding).

---

## Where it composes

```
load_tracker(weights, confidence, linker_config)  →  InstrumentTracker
    update(frame):
        dets = detector.predict(frame)          # raw detections (+ mask)
        dets = deep_ocsort.update(dets, frame)   # raw tracker_ids
        dets = linker.update(dets, frame)        # ← remaps raw → session ids
        return dets                              # session ids only
```

- The linker is a **standalone `SessionLinker` unit**, not logic baked into the
  tracker. Its interface mirrors the OC-SORT wrapper:
  `linker.update(detections: sv.Detections, frame: np.ndarray) -> sv.Detections`
  — detections-with-raw-ids in, detections-with-session-ids out. This is the
  testability decision: **T04's harness drives it directly with replayed frames +
  cached detections**, asserting id-remaps offline with no live camera.
- **`reset()` clears session state** (roster, session galleries, the private
  `raw→session` map, the Missing set, pending decisions) **but keeps the loaded
  persistent references** (embedded once from disk, reused next enrolment).
- **Seam invariant:** output carries session ids in `tracker_id`; the
  `raw→session` map stays internal; no alias map, no retroactive rewrites,
  original id **forever**.

### Tracker: Deep OC-SORT (vendored)

Constantijn vendored **Deep OC-SORT** (`model/playground/trackers/deep_ocsort_vendor/`,
commit `94b8bd8`) — an `OCSort` with an internal appearance embedder
(MobileNetV3-Small, ImageNet stand-in) fused with a Kalman motion model + CMC.
Two embedders exist **deliberately**: the tracker's weak one for cheap
per-frame association, and the linker's strong **DINOv2-B** (the T02-pinned
embedder every guarded number was measured with), event-driven at track
death/birth only. The tracker owns *short-gap* continuity; the linker owns
*long-absence* re-identification. They are complementary — the tracker does not
make the linker redundant.

---

## 1 · Enrolment window & roster freeze

`reset()` fires at **Start**, when Setup has already guaranteed 2 s of scene
stability + the operator's confirmation that every instrument is present
(`app/docs/DESIGN.md` D8/setup). So the linker needs no stability detector of its
own — but the first frame or two after `reset()` are risky (OC-SORT `min_hits=3`
maturation; single-frame detector flicker).

- **Enrolment window ≈ 0.5 s** (≈5–7 frames at demo fps) opens at the first
  post-`reset()` frame. It collects crops of every **mature** (`tracker_id` real)
  track.
- The window closes when the detected set has been **stable** (same ids/count)
  across it. The **roster = every identity present in the majority of the window
  frames** (tolerates one flicker).
- The roster is then **frozen**: no track born afterward joins it (matches
  [T06](../../docs/wayfinder/session-linker/tickets/T06-unknown-objects-ux.md)).
- Every identity's Start gallery is seeded from **all** window frames →
  multi-frame from birth.

## 2 · Identity model

- At the enrolment window's close, each enrolled track is **assigned a canonical
  session id** — its bound specimen's number, or the next reserved session-only
  number. Raw ids are *renumbered*, never promoted; see §3 "Session numbering".
  (Superseded 2026-07-15: this bullet used to say the raw id was promoted as-is
  and the roster was therefore `{1…N}`. Neither survives T08.)
- The linker keeps a **private `raw id → session id` map**; the seam only ever
  emits the session id.
- When an Active identity's raw track dies, its session id → **Missing** (gallery
  retained). When a returning raw track links to it, the linker maps
  `new raw id → that session id` and re-emits the session id. Original id forever.
- A post-enrolment newcomer that matches nothing is emitted at
  **`raw_id + unknown_id_offset`** (default 1000) — permanently disjoint from the
  roster's range, so **Unknown stays derived app-side from roster membership**
  and needs no status field. (Superseded 2026-07-15: the old text leaned on
  "OC-SORT ids are monotonic, so a newcomer's id is always `> N`". Renumbering
  breaks that — raw ids and session ids now share a small integer range and would
  *collide* — which is what the offset exists to prevent. §7, and the seam is
  widened after all: by `roster`.)

## 3 · Gallery binding (persistent references)

> **Rewritten 2026-07-15 — T08 is built and replay-validated.** This section had
> drifted from the shipped design in three ways:
>
> 1. **The match rule is SRC, not cosine** — ⚠️ **engineering call, ungrilled.**
>    The old text said "cosine-match"; that predates §6's rewrite and is stale.
>    The champion computes **no crop-to-crop cosine anywhere** — binding scores
>    through the same `matcher.score()` sparse solve as every other decision.
> 2. **The assignment is greedy one-to-one, not Hungarian** — ⚠️ **engineering
>    call, ungrilled.** Deliberate deviation from the text that stood here;
>    reason stated below.
> 3. **The numbering was never described here.** It is now — the "reserved
>    specimen numbers" scheme, which *was* grilled with Bram (2026-07-15) and is
>    what makes "Instrument 3" mean *specimen 3*.
>
> Build contract:
> [`T08-SPEC.md`](../../docs/wayfinder/session-linker/assets/T08-T10/T08-SPEC.md)
> §B-B/§B-L/§B-N. Measurements: [`demo-validation.md`](./demo-validation.md).

Each session identity may combine (a) **persistent multi-view references** for
its exact physical specimen (T07's pre-captured photo sets — 8 specimens × 15
views + COCO masks under `model/data/instruments/instrument{N}/`) and (b) **fresh
Start crops**. Binding is automatic, and happens exactly once:

- **At the enrolment freeze**, once every identity's Start gallery is built,
  each identity's Start crops (the same best-≤3-by-quality crops that seed
  `start_views`) are scored against the loaded persistent specimen galleries with
  `matcher.score()` — **SRC, not cosine** (flag 1). Identities are processed in
  ascending raw-id order.
- **Comparison is not eligibility here either.** Every identity scores against
  the **complete** set of loaded specimens, always. Never score against a
  shrunken dictionary and never re-score in rounds: dropping a claimed gallery
  changes K and therefore changes every remaining SCI, invalidating the
  calibration. §6.5 point 2's rule applies to binding verbatim.
- **Bind only when confident.** An identity *proposes* its best specimen iff
  `best ≥ bind_tau` **and** `best − second ≥ bind_margin`. With a single loaded
  gallery the margin is vacuous and skipped — the matcher's internal K=1
  `cos_tau` gate is what covers that case (§6, Trap 1). Do not touch `cos_tau`.
- **Greedy one-to-one, not Hungarian** (flag 2). Proposals sort by descending
  score, tie-broken by ascending raw id; the first proposer of a specimen is
  awarded it, and any later proposer of a claimed specimen becomes
  **session-only**. The reason to deviate from a global optimum: Hungarian would
  push a losing identity onto its **second-choice** specimen. A wrong bind
  poisons every later decision that identity is ever part of, while session-only
  merely means fewer views. So a contested loser falls back to session-only and
  **never settles for second**. This is §6.5 point 3's shipped philosophy, for
  the same reason. (It is also the pragmatic route: Hungarian is not reachable
  through the `score()`/`accept()` seam without assembling the matrix by hand —
  build-brief Trap 3.)
- **A bound identity's matching gallery = `persistent ∪ Start`** (∪ its last-seen
  view once that exists) — ~18 dictionary atoms instead of ≤3.
  `_comparison_galleries()` concatenates the bound specimen's persistent views
  with the identity's session views. Unbound identities keep session views only.
- **Fallback:** an identity whose mapping is uncertain stays **session-only** —
  still a first-class identity with a small stable number, just fewer views.
- **Bind once, at the freeze. Never re-bind mid-recording.** No later event —
  link, death, return, batch decision — may change any identity↔specimen
  binding. A late bind poisons.
- **Degrades gracefully, and must never raise.** No photos, a missing or empty
  directory, an unparseable folder → no persistent galleries → everyone runs
  session-only (plus the renumbering below) and the linker still works, weaker
  only on opposite-face returns. Persistent views are embedded **once**, at
  `load_tracker()` time, and survive `reset()` (§4, and Trap 5 in the build
  brief).

Prefer session-only over a wrong bind: no duplicate types in the demo set means a
correct match is high-margin, so a *low* margin genuinely means "don't bind."

> **Flag — loader robustness reads per-image, not per-folder** (engineering call,
> ungrilled). B-L4 says "unreadable images are skipped with a logged warning".
> The build applies that **per image**: one corrupt `.jpg` costs that view, and
> the specimen still loads its other 14. A folder is skipped only if *zero* crops
> survive. More robust, and consistent with "never raise" — but note the one
> asymmetry: an image_id that cannot be resolved against the COCO index trips the
> folder's outer `except` and skips the whole specimen.

### Session numbering — reserved specimen numbers

Grilled with Bram 2026-07-15. Binding is what assigns session ids, so the
numbering lives here rather than in §2.

- **Bound identity → `session_id = its specimen number`.** That is the point of
  the whole build: "Instrument 3" on screen means specimen 3's photographed
  physical object, stable across recordings.
- **Session-only identity → numbered from `max(loaded specimen numbers) + 1`
  upward**, in ascending raw-id order. With no galleries loaded the base is `0`,
  so numbering starts at **1** and the roster is `{1…N}` contiguous.
- **A specimen number is never worn by an identity that did not confidently
  bind.** Reserving the range is what keeps the label honest. The cost is that a
  partial bind leaves a **gapped** roster — e.g. `{1, 2, 5, 9, 10}`. Contiguity
  is not a property anyone may assume, the app included (§7).
- This replaces the old `session_id = raw_id` promotion **in every case**,
  galleries or not — deliberate, grilled: renumbering is what stops "Instrument
  10 on a tray of 8". Measured on both takes: the raw OC-SORT ids at the freeze
  were non-contiguous (Take B: `{3,5,7,9,10,11,12,14}`) and renumbered to
  `{1…8}`.
- Every id emitted for a track that is *not* mapped to a roster identity is
  offset out of the roster's range — see §7 and
  [`tracker-interface.md`](./tracker-interface.md).

> **Flag — `bind_margin` is doing almost no work** (measured at T08, ungrilled
> consequence). Nearly every second-best bind score comes back **0.0**: the
> champion's K=1 `cos_tau = 0.60` gate zeroes hopeless candidates *before* the
> margin is ever consulted, so the absolute `bind_tau` carries the bind decision
> essentially alone. `bind_tau = 0.30` / `bind_margin = 0.02` simply start at the
> champion's own τ/δ and are **to be tuned at T07/T09** (flag: nobody chose those
> values *for binding*). Anyone reaching for the margin expecting leverage should
> know it has almost none — and see `demo-validation.md` for the one bind that
> currently clears `bind_tau` by 0.23%.

## 4 · Gallery representation

- **Masked crops.** Suppress background to a neutral fill (mask *before* resize so
  bilinear feathers the edge — the vendored embedder's `mask_crop`), with a small
  mask dilation so imperfect masks don't clip the instrument. On a shared table,
  an unmasked bbox bakes common-mode background (and sometimes a neighbour) into
  the embedding and shrinks separation. (Padded-bbox is the fallback only if T02
  shows mask noise > background gain — unlikely for thin metal on a table.)
  Crops are **RGB**: the research pipeline embedded RGB and every threshold is
  calibrated to it — live BGR frames must be converted before any embed.
- **Per-view storage; views are dictionary atoms (no aggregation step).** Keep
  every gallery view as its own vector (persistent + Start pooled, origin
  discarded) — §6's SRC uses each view as a dictionary atom, so the old
  nearest-view-max / top-K aggregation step no longer exists (superseded
  baseline vocabulary). The per-view principle survives unchanged: **never
  mean-pool across faces** (the centroid matches none of them); a return in
  the Start pose is reconstructed by Start atoms, a face only T07 saw by
  persistent atoms — automatically, via the sparse solve.
- **Start crops deduped to top ~3 by quality** (mask-area × confidence) so
  near-identical enrolment frames don't over-weight the Start pose.
- **Quality filter:** drop crops below a min mask-area, low confidence, or
  truncated at the frame edge.
- **Event-driven embedding.** Per live track keep a small rolling buffer of recent
  good crops (pixels); embed **nothing per-frame**. Run DINOv2-B only at a **death**
  (finalize last-seen appearance) or **birth** (embed newcomer to match).
  Persistent + Start embeddings are computed once and held resident.

## 5 · Gallery safety — stability-gated refresh (ON, toggleable)

Galleries = a **frozen core** (`persistent ∪ Start`, persistent never evicted)
plus a bounded ring of **session-refresh** views. Refresh is **ON by default,
with a config flag to turn it OFF**.

The tracker's continuity, gated on stillness, is what makes refresh *safe*: the
poisoning vector is a wrong *link*; a crop added from a **continuously-tracked,
still, unobstructed** instrument is one whose identity is not in question. Add a
crop to identity `k`'s session ring only when **all** hold:

- track currently mapped to `k`, high detection confidence, **mature**;
- **still** (low box velocity) and **unobstructed** (full mask, spatially isolated
  — no other track's box nearby) for a **dwell ≥ ~1–2 s**;
- **not** within a cooldown after any link/recovery event;
- crop is **novel** vs the current gallery (a genuinely new pose/face, not a
  duplicate — this is what makes refresh *worth it*: it fires only when the
  instrument adopts a new stable pose, catching new faces live);
- into a **bounded ring** of session-only views (persistent refs never evicted).

**Never** learn from Pending, Unknown, ambiguous, or just-recovered tracks.

> Toggle OFF ⇒ galleries frozen at enrolment (branch-3 poisoning impossible by
> construction). Ship-decision to *keep* it on is validated by T04: if
> `persistent ∪ Start` already handles opposite-face returns, refresh is redundant;
> if not, it's the cheap live insurance.

## 6 · Match rule — open-set, multi-candidate (SRC)

> **Rewritten 2026-07-15.** The matcher-autoresearch program
> (`experiments/matcher-autoresearch/`, FINDINGS §0–§2) **replaced** this
> section's original rule (nearest-view max cosine + Hungarian) with
> **sparse-representation-based classification (SRC)** — a genuinely different
> method family, promoted champion after beating the baseline on both guarded
> axes (CV re-ID 0.9333 vs 0.850, foreign-reject 0.9733 vs 0.9467, 0 twin
> errors) and clearing a leak-check. The ported module lives at
> `model/src/orc_model/pipelines/matching/` behind the same
> `build_gallery/score/accept` interface; §6.5 is an engineering addition the
> research never evaluated (grilled with Bram 2026-07-15).

On a **birth** (raw id `> N`, unseen):

1. **Evidence:** collect the new track's crops over its **first ~0.5 s**
   (multi-frame); embed with DINOv2-B (masked crops, **RGB** — §4).
2. **Candidate set = all identities in *Missing* state**, plus the implicit
   **reject** option. Multi-removal makes a multi-candidate set the *normal* case.
3. **Score = joint sparse reconstruction, not similarity.** Every Missing
   candidate's gallery views are concatenated into **one dictionary** (each
   view a unit-norm atom). The query window's fused embeddings are jointly
   reconstructed as a sparse linear combination of that dictionary — a single
   L1-penalized `MultiTaskLasso` solve **shared across the whole window**
   ("joint sparse representation", Wright et al. 2009 extended frame→window).
   Per-candidate score = reconstruction quality of *that candidate's atoms
   alone* (inverse residual) × the **Sparsity Concentration Index** (SCI: how
   much of the solved code's L1 mass sits on that one candidate vs. spread
   across all). No crop-to-crop similarity is computed anywhere. Ablations:
   SCI is the open-set mechanism (without it, reject collapses 0.9733→0.6133);
   the *joint* window code is load-bearing (independent per-frame solves:
   reject 0.9200); window=1 drops reject to 0.7500.
4. **Accept iff** best ≥ **τ_accept** **and** best beats second-best by
   **margin δ** → link. Else → stays Unknown.
5. **Simultaneous returns → batched, gated assignment** — see §6.5.
6. **One-missing prior = margin-skip only, never threshold relaxation.** With
   one Missing identity SCI is vacuous (=1) and there's no second-best, so the
   margin test is skipped — but **τ_accept still holds**. This is the
   foreign-object safety valve: a phone placed while one instrument is missing
   must **not** be force-linked into that slot. No forced linking, ever.

**Parameters** (winners of the 630-point CV sweep — `champion/PARAMS.md`):
`alpha=0.0003` (MultiTaskLasso L1), `size_alpha=0.5`, `τ_accept=0.30`,
`margin δ=0.02`, `mask_dilate_px=2`, `max_iter=2000`, `window=3` frames.
All guarded numbers come from the 8×15 synthetic crop set — **expect a
τ/margin retune after the first live test.**

### 6.5 · Simultaneous returns — batched, gated assignment

> **Engineering call, not research-backed** (grilled with Bram 2026-07-15;
> FINDINGS §2 flags multi-simultaneous returns as a gap no challenger ever
> evaluated). Validated before the demo on the video-003 teardown→re-lay
> replay + the live test; revisit if either misbehaves.

> **Revised 2026-07-15 (structural fix).** Points 2–4 below were rewritten to
> match the implementation. The original text scored against the *Missing*
> dictionary and shrank it between rounds; that collapsed the ordinary
> one-missing case to K=1, where SCI is `0/0` and cannot reject anything (see
> §6's `cos_tau` note). Comparison and eligibility are now separate concepts.

1. **Natural batching, no added waiting.** At each decision tick, all new
   tracks whose evidence windows have completed form one batch. Returns born
   within ~0.5 s of each other share a batch automatically; genuinely
   staggered returns resolve one-by-one. (The embed may overlap the window's
   tail; nothing waits on a grace timer.)
2. **Comparison is not eligibility.** Each batch track gets `score()`
   independently against the **full frozen roster** — every enrolled identity
   with a usable gallery, Active *and* Missing alike (`_comparison_galleries`).
   K therefore stays at the roster size (8 in the demo), which is what SCI was
   calibrated at; Active identities serve as the distractors SCI needs. Only
   *after* `accept()`'s τ/margin gate is the winner checked for **eligibility**:
   an identity may receive a link only while Missing (`_eligible_missing_ids`).
   A winner that is an Active identity is **never** force-linked — the row
   settles Unknown (or defers, per point 4).
3. **Assignment is greedy per identity, not Hungarian.** Each admissible row
   proposes its single accepted identity; where two rows claim the same
   identity the higher scorer wins (`best_by_session`) and the loser settles
   Unknown. It is deliberately *not* `linear_sum_assignment`: a losing row is
   never pushed to its second choice, because a duplicate track reading Unknown
   is far safer than a forced link to the wrong instrument. **No forced links,
   ever** — τ always gates.
4. **No second round; coasting handoff defers instead.** There is no re-score
   against a shrunken dictionary — removing a claimed gallery would change K and
   therefore change every remaining SCI, invalidating the calibration. Instead,
   if the gated winner is an Active identity whose raw track is currently absent
   but still inside its death grace (`deferable_active_ids`), the decision is
   **deferred** until that identity either dies (becomes Missing → revalidate
   and link) or resumes (→ Unknown). This covers an OC-SORT id change one frame
   before its old id dies. Anything else unresolved settles **Unknown**;
   recovery is lift-and-replace.
5. **Edges** (all follow from the above): more new tracks than Missing →
   extras settle Unknown; unclaimed identities stay Missing; a track dying
   mid-window gets no decision; settled decisions are never revisited (seam
   invariant, § "Where it composes").

## 7 · Rejection & Pending — app-side encoding

> ⚠️ **Rewritten 2026-07-15 — this section's premise was false, and T08/T10
> reversed it.** It used to open *"Nothing new crosses the seam. The app already
> has the frozen roster `{1…N}`."* **Both halves were false**: the app had no
> roster at all, and the ids were OC-SORT's raw counter, so a tray of 8 read e.g.
> `{3,5,7,9,10,11,12,14}`. Everything below is the shipped design, grilled with
> Bram the same day (route (b); wayfinder T10) and built. The old `> N` threshold
> tests are gone — they are set-membership tests now.

**The roster crosses the seam.** `InstrumentTracker` exposes a read-only
`roster` property — a `frozenset[int]` of session ids, empty until the enrolment
freeze and immediately after `reset()`. The app samples it in the **same tick**
as the detections it belongs to, and every "is this ours?" test is **set
membership against that property**. Contiguity is never assumed and `> N` is
never computed: a partial bind gaps the roster (§3).

This is route (b), grilled with Bram 2026-07-15, and it is a **deliberate
widening** of the `tracker-interface.md` pin and `app/docs/DESIGN.md` D8. Route
(a) — the app deriving its own roster from the ids present at its Start snapshot
— was rejected: the app's snapshot and the linker's freeze are ~0.7 s apart, and
any disagreement between those two sets silently corrupts every Unknown decision
for the whole recording. One read-only property removes that entire class of
coordination bug. The app also gets a 1 s entry debounce (`Session`), which the
spinner below rides.

- **Two disjoint id ranges, not a threshold.** Roster session ids (specimen
  numbers, then session-only numbers above them) are emitted **unchanged**. Every
  id emitted for a track that is *not* mapped to a roster identity — pending in
  its evidence window, deferred, or settled Unknown — is `raw_id +
  unknown_id_offset` (default 1000). The ranges never overlap, so "not in roster"
  is exact, cheap, and stable for the whole recording. Details and the freeze-frame
  rule: [`tracker-interface.md`](./tracker-interface.md).
- **During the decision window** the linker emits the track's **provisional
  offset id** and marks it **`data["resolving"] = True`** (pending or deferred).
  The app draws a **small "resolving" spinner** on that box, **gray mask**,
  gated on that flag — not on a clock.
- **On resolution the box's id flips:**
  - **Linked** → box now carries a roster session id → spinner clears, it snaps
    to that instrument's **fixed colour**. The provisional id lived < 1 s → no
    report trace (seam-designed behaviour; the app renders per-box, so the
    transition is visually continuous though the integer changed).
  - **Unknown** → still not-in-roster at debounce → spinner clears to a **gray
    Unknown** mask.
- **Unknowns are video-only** (grilled 2026-07-15). A not-in-roster track exists
  on the **feed** and nowhere else: gray mask, ~1 s resolving spinner, then the
  settled label **"Unknown"** — never `"Instrument N"`, at any age, in any state.
  It gets **no panel row**: it never enters `/status`'s recording instruments,
  never reaches Usage or Completeness (T06), never lands in the report. The video
  says "the system sees something and it is not one of yours"; the panel stays
  the roster's. This is enforced model-side-in-spirit but app-side-in-fact — the
  backend filters `present_ids & roster` before anything becomes a tracked
  instrument.
- **Colour crosses the seam too.** Each `/status` recording instrument carries
  its own **fixed mask colour as a hex string**, derived from the frozen roster,
  so the panel swatch and the overlay mask are literally the *same value* and can
  never drift apart. Roster ids get **8 distinct colours**; not-in-roster →
  **gray** (`#9ca3af`). Because the linker re-emits the *original* session id, a
  returned instrument **regains its original colour** (visual identity
  continuity); a foreign object stays gray.
- The spinner tracks the linker's **actual decision**, via the per-detection
  `data["resolving"]` flag (widened 2026-07-16). **Superseded:** T10 originally
  rode the app's debounce clock instead, because the model then carried no
  "still-deciding" flag — but a track **deferred** behind a coasting active id
  (the normal handoff path, §7's revalidation) outlasts that ~1 s clock, so it
  flickered to a settled gray "Unknown" and *then* snapped to its instrument
  once it linked. The flag removes the disagreement: the spinner clears exactly
  when the linker settles the track (linked → its colour; rejected → gray
  Unknown), never before. It retires the renderer's `pending_s` timer and
  first-seen map (`T10-BACKEND-SPEC.md` B-V3/B-V5).

> **Shipped in T10** (`app/docs/DESIGN.md` D8a, `T10-BACKEND-SPEC.md`): resolving
> spinner on not-in-roster tracks; gray Unknown mask, video-only; 8-colour
> roster mapping by session id, reported per instrument. Route (b) widened the
> model seam by one read-only property (`roster`); the wait-state fix
> (2026-07-16) added a second widening, the per-detection `data["resolving"]`
> flag, so the spinner follows the linker's decision rather than a clock. Those
> two are the whole cost, and the reason the app needs no coordination logic of
> its own.

## 8 · Rotation & flip robustness

> **Status 2026-07-15: NOT built, and NOT subsumed.** Line 2 is marked "(built,
> on)" below — that is stale; no augmentation ships. T08 tested the hypothesis
> that §3's 15 real persistent poses would make §8 unnecessary, and the evidence
> came back **negative**: with all 15 persistent views bound, Take B's 368.0 s
> flipped instrument **still** settles Unknown rather than identity 6. §8 remains
> a real, unaddressed gap. It **fails safe** (a false reject, never a wrong
> link), so it does not block the demo. Evidence:
> [`demo-validation.md`](./demo-validation.md).

Layered; only Line 3 is gated on T02's rotation-margin numbers.

- **Line 1 (always):** gallery diversity (T07 real faces/rotations — more
  dictionary atoms) + the joint multi-frame window (§6).
- **Line 2 (built, on): gallery-side synthetic augmentation.** For each masked
  view, store synthetically **rotated + mirrored** copies (a coarse in-plane angle
  set + horizontal flip), embed each, keep as extra gallery views. A rotated return
  matches one via nearest-view max. **Augmentation is gallery-side so the per-return
  query stays a single embed** (friendly to the ≤ 1 s budget). Cost stays sane:
  **persistent** augmented embeddings are **precomputed offline and loaded**; only
  the top-3 **Start** crops × the synthetic set embed at enrolment (~a second).
- **Line 3 (md-only, T02-gated — can backfire): mask principal-axis
  canonicalization.** PCA / `minAreaRect` on the mask → rotate the crop upright
  before embedding, killing *arbitrary* in-plane rotation. But it leans on mask
  quality; thin instruments can have noisy masks → a wrong axis rotates the crop
  *wrong* and hurts. **Default OFF, documented here as an escalation path, not
  built, until T02 shows Lines 1+2 miss.** Algorithm when needed: threshold the
  full-frame mask to the crop region → `cv2.minAreaRect` → rotate crop + mask so
  the principal axis is vertical → embed; resolve residual 180°/mirror by the
  Line-2 gallery variants (max-sim already covers it).
- **Never train or fine-tune on the demo instruments.** The model stays frozen;
  robustness comes only from gallery diversity + augmentation + voting.

**Hard limit:** synthetic rotation covers *in-plane* angle only — it **cannot
synthesize the opposite physical face**. Opposite-face coverage comes solely from
T07's real both-face capture or the § 5 refresh module catching the face live.

## 9 · Tracker interplay

- **`max_age` ≈ 1.0 s** (Deep OC-SORT coasting window; the wrapper's
  `max_age_seconds`). This is the tracker/linker boundary: ≤ it = tracker rides
  brief occlusions invisibly; > it = death → the linker owns re-id. Sits below the
  app's 1.5 s off-debounce so a brief hand-over occlusion never reaches the linker,
  while the coast is never long enough to mis-bind a foreign object at the old
  location. **T04 tunes** against real fps + measured occlusion lengths.
- **Re-validate meaningful-gap reactivations.** Within `max_age`, the tracker can
  reactivate a coasted track under the same raw id via its *weak* embedder — a
  different object could silently inherit the id, bypassing the linker's
  embedder. So: if a track
  reactivates after a gap **longer than a small threshold** (a couple frames =
  trust the tracker; longer = suspect), treat it as a **birth-like event** — embed
  and validate against **that session id's own gallery**; on failure, **break the
  link** and re-run the open-set match. Event-driven, cheap. This is what stops a
  raw-id switch from passing through silently.
- **No `-1` strip needed.** OC-SORT's `min_hits=3` gates immature tracks out of
  the output entirely — the linker just keys enrolment/birth off the first
  **mature** emission (~3-frame delay, folded into the budget).
- **Non-blocking / amortized budget.** DINOv2-B (~0.2–0.5 s/event) must **not** stall
  `update()` beyond its per-frame budget (~0.07–0.1 s at 10–15 fps), and
  multi-removal means several simultaneous births. So link resolution is
  **multi-frame**: cache crops over the ~0.5 s window cheaply, embed off the
  critical path, apply the id-remap the frame the decision lands, within ≤ 1 s.
  **Whether that's a background worker or batched deferral is T05's call, gated on
  T04's measured headroom** — this doc pins the *constraint*, not the threading.

---

## Parameters (defaults filled by T02, gaps tuned by T04)

| Parameter | Meaning | Source |
|---|---|---|
| `enrolment_window_s` ≈ 0.5 | roster-freeze settle | this doc; T04 sanity |
| `τ_accept` = 0.30 | absolute SRC-score (recon × SCI) acceptance | **autoresearch** (630-pt CV sweep); retune live |
| `margin_δ` = 0.02 | best-vs-second-best gap | **autoresearch**; retune live |
| `alpha` = 0.0003 | `MultiTaskLasso` L1 penalty | **autoresearch** |
| `size_alpha` = 0.5 | mask-size fusion weight | **autoresearch** (resolves the twin pair) |
| `match_frames` = 3 (~0.5 s) | multi-frame evidence window | **autoresearch** |
| `bind_tau` = 0.30 | absolute §3 bind score | starts at the champion's `τ_accept`; **nobody tuned it for binding** — T07/T09 |
| `bind_margin` = 0.02 | §3 best-vs-second bind gap | starts at the champion's `margin_δ`; **measured to do almost no work** (§3 flag) — T07/T09 |
| `unknown_id_offset` = 1000 | keeps emitted non-roster ids disjoint from the roster | §7; demo-scale rosters never approach it |
| crop = masked RGB (+ 2 px dilation) | crop form | **autoresearch** confirms |
| rotation augmentation angle set + mirror | Line 2 | **T02** |
| `refresh_enabled` = true | § 5 toggle | this doc; keep-on validated by T04 |
| refresh gates (dwell, velocity, isolation, novelty, cooldown) | § 5 | T04 tunes |
| `max_age_s` ≈ 1.0 | tracker coast / boundary | **T04** |
| `revalidate_gap` | meaningful-gap threshold | **T04** |

## Open dependencies & hand-offs

- **T02** posts stage-1 numbers as an interim asset; this design consumes them
  without waiting for T02's closure. Threshold *values* are parameters.
- **T04 harness must replay cached *frames* (or pre-extracted crops) alongside the
  cached detection boxes** — box-only replay cannot exercise linking (the linker
  crops + embeds from frames). Flag for the harness build.
- **T05** builds it: standalone `SessionLinker`, the refresh module (on), Line-2
  augmentation; Line-3 only if triggered. Also lands the § 7 app-side deltas in
  `app/docs/DESIGN.md`, and the `matching/` gitignore housekeeping.
- **T07** photo session feeds persistent galleries + final threshold numbers
  (data/parameters, never code).

## Out of scope (map decisions)

Catalog identification (real instrument names); post-hoc report-correction pass;
cross-session identity / recording history. All confirmed non-goals for the demo.
