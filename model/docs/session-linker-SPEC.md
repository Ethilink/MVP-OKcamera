# SPEC — `SessionLinker`

**Target:** `model/src/orc_model/pipelines/session_linker.py`
**Test file:** `model/tests/pipelines/test_session_linker.py`
**Design source:** `model/docs/linker-design.md` §§1–7 (rationale lives there; THIS
file is the build contract). Consumer contract:
`model/docs/tracker-interface.md` § "`tracker_id` across absence".

## Goal

`SessionLinker` sits after Deep OC-SORT in `load_tracker()`'s composition and
turns *raw* (disposable) tracker ids into *session* ids that survive absence:
an instrument that leaves the table and returns is re-identified against the
identities currently Missing and re-emits its **original** id. It enrols a
frozen roster at Start, keeps per-identity appearance galleries (session-only
in v1), embeds **only** at track birth/death events (never per frame), decides
returns with the SRC matcher behind an open-set gate (no forced links, ever),
and resolves simultaneous returns with a batched, gated assignment plus one
re-score round. It does NOT: identify catalog instruments, rewrite already-
emitted ids, load persistent reference gaĺleries (v1), or run its own detector
or tracker.

## Public interface

```python
REJECT  # re-exported from orc_model.pipelines.matching.interface

class SessionLinker:
    def __init__(
        self,
        matcher,                          # ChampionMethod-compatible; see Dependencies
        *,
        fps: float,                       # REQUIRED — real capture fps, converts all *_s params to frames
        enrolment_window_s: float = 0.5,
        evidence_window_s: float = 0.5,
        evidence_frames: int = 3,
        absent_death_s: float = 1.0,
        min_mask_area_px: int = 200,
        logger: logging.Logger | None = None,   # default: logging.getLogger("orc_model.session_linker")
    ) -> None: ...

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections: ...
    def reset(self) -> None: ...

    @property
    def roster(self) -> frozenset[int]: ...   # empty before freeze; frozen session ids after
```

- `fps` is deliberately **required** (T05 brief C4: no hidden 30-fps default
  anywhere in the linker chain).
- A freshly constructed instance behaves exactly as if `reset()` was just
  called.

## Behavior

### B1 — Emission contract (every `update()` call)

1. Output is a **new/derived** `sv.Detections`; the input object and the input
   `frame` are never mutated. All fields (`xyxy`, `confidence`, `class_id`,
   `mask`, `data`) are preserved row-aligned; **only `tracker_id` values may
   differ** from the input.
2. Mapping per row: raw id present in the private raw→session map → emit its
   session id; otherwise emit the raw id unchanged (pass-through). There is no
   sentinel/-1 case (upstream guarantees mature ids only).
3. Empty input (zero rows or `sv.Detections.empty()`) → return an empty
   `sv.Detections`. **Empty frames still advance all internal clocks**
   (enrolment window, evidence windows, absence counters).
4. Once a raw id has been emitted under a given output id, that mapping never
   changes retroactively; a new link affects the current call's output and
   later calls only. A linked raw id keeps emitting its session id for the
   rest of the recording.

### B2 — Enrolment & roster freeze (design §1–§2)

1. The enrolment window opens at the **first `update()` call that contains at
   least one detection** (post-construction or post-`reset()`); it spans
   `round(enrolment_window_s * fps)` consecutive `update()` calls (minimum 1),
   counting empty frames.
2. During the window, output is pure pass-through, and per-track quality crops
   (B6) are buffered.
3. At window close the roster **freezes**: roster = every raw id present in
   **more than half** of the window's frames. Each enrolled raw id is promoted
   to its session id **as-is** (no renumbering — the live track keeps emitting
   the same integer). `roster` property returns exactly this frozen set from
   then on; before freeze it returns an empty frozenset.
4. Each roster identity gets a Start gallery:
   `matcher.build_gallery(crops, masks, {"identity": session_id})` from its up
   to **3 best** buffered crops ranked by (mask-area × confidence). Fewer or
   zero quality crops → best effort with whatever it has; an identity with no
   crops at all gets no gallery and simply can never be re-linked (degenerate,
   must not crash).
5. A track born **after** the freeze never joins the roster.
6. *(Clarified 2026-07-15.)* Enrolment crops feed **only** the Start gallery;
   each identity's B6.4 rolling buffer starts **empty** at freeze. A track
   that dies with no post-freeze quality crops therefore skips its death
   embed (B3.2) — its Start views already are its last-seen appearance.

### B3 — Death (Active → Missing)

1. Post-freeze, a roster-mapped raw id (original or re-linked) that has been
   absent from the input for **more than** `round(absent_death_s * fps)`
   consecutive frames dies: its identity moves to **Missing** and its buffered
   last-seen crops (the B6.4 rolling buffer — the most recent
   `evidence_frames` quality-passing crops; ordering is immaterial to the
   matcher, whose gallery is an unordered atom set) are embedded via
   `matcher.build_gallery` and become the identity's **last-seen views**.
   *(Amended 2026-07-15 — consensus round 1: the earlier "quality-ranked"
   wording contradicted B6.4's bounded most-recent buffer.)*
2. The scoring gallery for a Missing identity = Start views + (only) the
   **most recent** death's last-seen views (a new death replaces the previous
   last-seen views, never appends unboundedly). Empty buffer at death → skip
   the embed; gallery stays as-is.
3. A raw id reappearing **before** the death threshold is just the tracker
   coasting/reactivating — no event fires (§9 re-validation is an explicit
   TODO, see Out of scope).
4. Death of a non-roster track (pending or settled-Unknown) just drops its
   internal state — no embed, no event.

### B4 — Birth & evidence window (design §6.1)

1. Post-freeze, an unseen raw id (not mapped, not settled) becomes **Pending**:
   it keeps emitting its own raw id while quality crops are buffered.
2. Its evidence window closes when **either** `evidence_frames` quality crops
   are collected **or** `round(evidence_window_s * fps)` frames have elapsed
   since its first appearance, whichever comes first.
3. A pending track that dies (absence > death threshold) before its window
   closes gets **no decision**; its state is dropped. (If it was a real
   instrument, the next placement makes a fresh attempt.)
4. If the window closes with an **empty** crop buffer, or with **no Missing
   identities**, the track settles Unknown **without any matcher call** (no
   wasted embeds when nothing is missing).

### B5 — Batched decision (design §6/§6.5, grilled 2026-07-15)

1. **Natural batching:** all pending tracks whose evidence windows close on
   the same `update()` call form one batch. No waiting, no grace timer.
2. **Round 1:** every batch row scores against the **same, full** current
   Missing gallery dict: `matcher.score(crops_i, masks_i, {}, missing)` →
   `matcher.accept(scores_i)`. A row whose `accept` returns `REJECT`
   contributes no admissible cell; a row whose `accept` returns identity `j`
   contributes the single admissible cell `(i, j, scores_i[j])`.
3. **Assignment:** conflicts (two rows accepting the same identity) are
   resolved by maximizing total score over admissible cells —
   `scipy.optimize.linear_sum_assignment` on the gated matrix. Implementation
   constraints (T05 brief C6): drop rows/columns with no admissible cell
   before solving; use a large negative **finite** sentinel for inadmissible
   cells, never `-inf`; discard any assignment landing on an inadmissible
   cell. Winners link: raw→session mapping takes effect on THIS call's output;
   the identity leaves Missing (Active again).
4. **Round 2 (exactly one):** rows left unresolved (rejected or assignment
   losers) re-run step 2–3 against the **reduced** Missing dict (claimed
   identities removed) — SCI redistributes over the smaller dictionary, so a
   round-1 reject may legitimately link in round 2. Skipped if no rows or no
   Missing identities remain.
5. After round 2, unresolved rows settle **Unknown permanently**: recorded,
   never re-decided, raw id passes through forever.
6. **No forced links, ever**: a link happens only through `matcher.accept`
   returning that identity (τ gate included) plus assignment victory.

### B6 — Crops, masks, RGB, quality (design §4; T05 brief C1)

1. Crop extraction per detection row: clamp the box to frame bounds, round to
   int; slice the frame and the row's full-frame mask identically. Zero-area
   after clamping → no crop.
2. **Crops handed to the matcher are RGB.** Input frames are BGR (seam
   contract); convert at crop time. This is calibration-critical — the
   research thresholds were tuned on RGB.
3. Quality admission to any buffer: mask area ≥ `min_mask_area_px` AND the
   (rounded, pre-clamp) box does not touch/exceed the frame edge. Confidence
   is used only for ranking (mask-area × confidence), not admission.
4. Per live track keep only the most recent `evidence_frames`
   quality-passing crops (bounded deque of pixel crops + aligned masks).
   **No embedding ever happens during buffering** — matcher calls occur only
   inside B2.4, B3.1, B5.

### B7 — Observability

Per matcher-touching event, log via the instance logger at INFO:
- enrolment freeze: roster size, per-identity gallery view counts, build ms;
- death: identity, buffered crops used, build ms;
- batch decision: rows, candidate count, per-round score ms, assignment ms,
  total ms, and each row's outcome (linked-to-id / unknown).
Formats/levels are not contractual (tests must not assert on them).

### B8 — `reset()`

Drops ALL state: roster, galleries, raw→session map, Missing set, pending
buffers, settled-Unknown records, frame counters. The next non-empty
`update()` opens a fresh enrolment window. (v1 is session-only: there are no
persistent references to preserve — see Out of scope.)

## Inputs / Outputs

- `frame`: BGR `np.ndarray` (H, W, 3) uint8, read-only.
- `detections` in/out: `sv.Detections` with `xyxy` float32 (N,4) in frame
  pixel space (may extend past bounds), `confidence` float32 (N,), `class_id`
  int (N,), `tracker_id` int (N,) — unique per row, monotonic across a
  session, starting at 1 — and `mask` bool (N, H, W).
- Gallery dict keys handed to `matcher.score` are **int session ids**;
  `matcher.accept` returns such a key or `REJECT`.

## Dependencies

- `orc_model.pipelines.matching` — `ChampionMethod` duck-type:
  `build_gallery(crops, masks, meta) -> Gallery` (has `.views` ndarray),
  `score(query_crops, query_masks, meta, galleries: dict) -> dict[key, float]`,
  `accept(scores) -> key | REJECT`; plus `interface.REJECT` and
  `ChampionGallery` (the linker may construct one from concatenated `.views`
  to merge Start + last-seen views — documented, accepted coupling).
- `scipy.optimize.linear_sum_assignment`; `numpy`; `supervision`.
- Must NOT import or touch: the detector, `deep_ocsort`, anything in `app/`.
- Tests must be able to run without loading HF models: the matcher is
  **injected** — tests pass a fake with the same duck-type.

## Out of scope (mark each with a `TODO(linker-design §X)` comment in the module)

- §3 persistent reference galleries & Hungarian binding (v1 = session-only;
  T07's photos don't exist).
- §5 stability-gated session refresh.
- §8 rotation/mirror gallery augmentation (Lines 2/3).
- §9 meaningful-gap reactivation re-validation.
- §9 non-blocking/async decision execution — v1 decides synchronously inside
  `update()`; threading is deferred until measured headroom demands it.
- Catalog identification; `confidence` attribute plumbing (lives in
  `load_tracker`); `-1` id stripping (upstream guarantees none).

## Tunable parameters (test-writer: do NOT hard-assert their values or formulas)

`enrolment_window_s`, `evidence_window_s`, `evidence_frames`,
`absent_death_s`, `min_mask_area_px`, the quality-ranking formula, log
message formats. Matcher thresholds (τ/margin/α) belong to the matcher and
are invisible here.
