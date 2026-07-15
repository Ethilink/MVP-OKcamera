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
- **Known roster** — the frozen set `{1…N}` of session ids enrolled at Start.
  Fixed after the enrolment window; a track born later never joins it.
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

- At the enrolment window's close, each enrolled track's current raw id is
  **promoted to its canonical session id**, so the roster is a fixed set `{1…N}`.
- The linker keeps a **private `raw id → session id` map**; the seam only ever
  emits the session id.
- When an Active identity's raw track dies, its session id → **Missing** (gallery
  retained). When a returning raw track links to it, the linker maps
  `new raw id → that session id` and re-emits the session id. Original id forever.
- A post-enrolment newcomer that matches nothing **passes through its own raw id**.
  Because OC-SORT ids are monotonic, a newcomer's id is always `> N` — strictly
  disjoint from the roster — so **Unknown is derived app-side from roster
  membership**. No status field, no widened seam.

## 3 · Gallery binding (persistent references)

Each session identity may combine (a) **persistent multi-view references** for
its exact physical specimen (T07, pre-captured) and (b) **fresh Start crops**.
Binding is automatic:

- At enrolment close, embed each identity's Start crops → N query vectors.
  Cosine-match against the ≤8 persistent specimen galleries and solve a
  **one-to-one global assignment** (Hungarian) — no two identities claim the same
  specimen.
- **Bind only when confident** (above absolute threshold **and** best-vs-second
  margin). A bound identity's gallery = `persistent ∪ Start`.
- **Fallback:** an identity whose mapping is uncertain stays **session-only**
  (gallery = Start crops only) — still a first-class identity, just fewer views.
- **Bind once, at enrolment. Never re-bind mid-recording** (a late bind risks
  poisoning).
- **Degrades gracefully:** if T07's photos don't land, no persistent galleries
  load → everyone runs session-only and the linker still works (weaker only on
  opposite-face returns).

Prefer session-only over a wrong bind: no duplicate types in the demo set means a
correct match is high-margin, so a *low* margin genuinely means "don't bind."

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

> ⚠️ **Premise resolved by grilling (Bram, 2026-07-15); build in flight (T10).**
> The paragraph below originally claimed the app already has the frozen roster
> `{1…N}` — both halves were false (no roster app-side; ids are OC-SORT's raw
> counter). **Decided route: (b) the roster crosses the seam** — `InstrumentTracker`
> gains a read-only `roster` property (frozenset of session ids, empty until the
> enrolment freeze), a deliberate widening of the `tracker-interface.md`/D8 pin;
> the app reads it instead of deriving its own Start snapshot (route (a) was
> rejected for its ~0.7 s snapshot-vs-freeze coordination risk). Every "`> N`"
> membership test below is therefore a **set-membership test against that
> property**, and contiguity is not assumed (T08 numbering: bound → specimen
> number 1–8, session-only → 9+, foreign ids offset out of the roster range).
> Also grilled the same day: unknowns are **video-only** (gray mask on the feed,
> no panel row), and each roster instrument's `/status` entry carries its mask
> colour so panel and overlay can never drift.

Nothing new crosses the seam. The app already has the frozen roster `{1…N}` and a
1 s entry debounce (`Session`).

- **During the decision window** the linker emits the track's **provisional raw
  id** (`> N`). The app sees a not-in-roster id present for **< the debounce** →
  draws a **small "resolving" spinner** on that box, **gray mask**.
- **On resolution the box's id flips:**
  - **Linked** → box now carries a roster session id (∈ `{1…N}`) → spinner clears,
    it snaps to that instrument's **fixed colour**. Provisional id lived < 1 s → no
    report trace (seam-designed behaviour; app renders per-box, so the transition
    is visually continuous though the integer changed).
  - **Unknown** → still not-in-roster at debounce → spinner clears to a **gray
    Unknown** mask; out of Usage/Completeness (T06).
- **Mask colours:** the 8 roster ids each get one of **8 distinct colours**;
  not-in-roster → **gray**. Because the linker re-emits the *original* session id,
  a returned instrument **regains its original colour** (visual identity
  continuity); a foreign object stays gray.
- The spinner rides the app's debounce clock, not the linker's exact decision
  instant (the model deliberately carries no "decided-Unknown" flag) — a clear
  foreign object may spin the full ~1 s then settle gray. Harmless.

> **App-side deltas to land in `app/docs/DESIGN.md` + the T05 spec:** resolving
> spinner on young not-in-roster tracks; gray Unknown mask; 8-colour roster
> mapping by session id. Model seam unchanged.

## 8 · Rotation & flip robustness

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
| `bind_threshold`, `bind_margin` | persistent binding confidence | tune at **T07** |
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
