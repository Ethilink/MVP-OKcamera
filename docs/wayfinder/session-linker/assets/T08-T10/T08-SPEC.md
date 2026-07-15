# SPEC — T08 gallery binding (`model/`)

**Single source of truth for all blind-TDD roles.** Distilled 2026-07-15 from
`model/docs/linker-design.md` §3 (canonical), the T08 ticket, the T08-T10
BUILD-BRIEF, and a grilling with Bram (same day). Where this file and any other
doc disagree, this file wins for this build; disagreements get flagged to the
orchestrator, not silently resolved.

## Goal

At the enrolment freeze, bind each enrolled identity one-to-one to one of the
pre-captured **persistent specimen galleries** (photo sets of the 8 physical
demo instruments), using **SRC scores** from the existing matcher. A bound
identity's **session id becomes its specimen number** (stable across
recordings) and its matching gallery becomes `persistent ∪ Start` (≈18 views
instead of ≤3). An identity that does not bind **confidently** stays
**session-only** — a first-class identity with a small stable number. Binding
happens **once, at the freeze — never re-bind mid-recording**. With no
persistent galleries at all, the linker degrades gracefully to today's
session-only behaviour (plus renumbering, below) and **must never crash**.

What this build does NOT do: no gallery refresh (§5), no rotation augmentation
(§8), no threading (§9 closed by measurement), no catalog names, no app/`app/`
changes, no matcher-interface (`build_gallery/score/accept`) signature changes,
no tau/margin retuning.

## Public interface

### New module `model/src/orc_model/pipelines/persistent_gallery.py`

```python
DEFAULT_INSTRUMENTS_DIR: Path
    # <repo>/model/data/instruments, resolved relative to this module's file
    # (parents up from src/orc_model/pipelines/), NOT the process cwd.

def load_persistent_galleries(
    matcher,
    instruments_dir: str | Path | None = None,   # None -> DEFAULT_INSTRUMENTS_DIR
) -> dict[int, "ChampionGallery"]:
    """specimen_number -> embedded gallery. {} when nothing usable is found."""
```

### `SessionLinker` (modify `model/src/orc_model/pipelines/session_linker.py`)

```python
DEFAULT_BIND_TAU = 0.30      # starts at the champion's tau; tune at T07/T09
DEFAULT_BIND_MARGIN = 0.02   # starts at the champion's margin
DEFAULT_UNKNOWN_ID_OFFSET = 1000

class SessionLinker:
    def __init__(
        self, matcher, *, fps: float,
        persistent_galleries: dict[int, ChampionGallery] | None = None,
        bind_tau: float = DEFAULT_BIND_TAU,
        bind_margin: float = DEFAULT_BIND_MARGIN,
        unknown_id_offset: int = DEFAULT_UNKNOWN_ID_OFFSET,
        # ... all existing kwargs unchanged ...
    ) -> None: ...
```

`roster` property keeps its exact signature (`frozenset[int]`, empty before
freeze) — its **contents** change per "Session-id assignment" below.

### `tracking.py` (modify)

```python
class InstrumentTracker(Protocol):
    # ... existing members unchanged ...
    @property
    def roster(self) -> frozenset[int]:
        """Frozen Start roster of session ids. Empty before the enrolment
        freeze and immediately after reset(). (Seam widening grilled with
        Bram 2026-07-15 — route (b), wayfinder T10.)"""

class FakeInstrumentTracker:
    # gains: roster property -> frozenset(range(self.n_instruments))
    # (its emitted tracker_ids are 0..n-1; the fake "enrols" instantly)

def load_tracker(
    weights_path, confidence=0.5, *,
    instruments_dir: str | Path | None = persistent_gallery.DEFAULT_INSTRUMENTS_DIR,
    # ... existing kwargs unchanged ...
) -> InstrumentTracker: ...
    # None -> binding disabled (no galleries loaded).
    # Missing/empty dir -> log INFO and proceed with no galleries. Never raise.
```

`_RealInstrumentTracker.roster` → `self._session_linker.roster`.

## Behaviour

### B-L: persistent gallery loading (`persistent_gallery.py`)

- **B-L1** Scan `instruments_dir` for subfolders named `instrument{N}` (N =
  positive int, parsed from the name; `instrument3` → specimen 3). Each folder
  has `images/*.jpg` + `annotations/annotations.json` (COCO).
- **B-L2** COCO handling mirrors `experiments/matcher-autoresearch/frozen/loader.py`
  (`_parse_field`/`_decode_rle`/`_load_coco_crops` — reuse the logic, vendored;
  do NOT import from `experiments/`): `bbox`/`segmentation` fields may be real
  JSON values **or python-repr strings**; segmentation is COCO RLE
  (`{"size", "counts"}`, decode via `pycocotools.mask.decode` — already a
  `model/` dependency). Crop = bbox-cropped RGB image + aligned bbox-cropped
  binary mask. Skip annotations with bbox `w < 4 or h < 4`.
- **B-L3** One `matcher.build_gallery(crops, masks, {"identity": N})` call per
  specimen — **embedding happens here, exactly once**; the linker never re-embeds
  persistent views (BUILD-BRIEF Trap 5).
- **B-L4** Robustness: a missing dir, a folder with no parseable annotations,
  unreadable images, or a folder name that isn't `instrument{N}` is **skipped
  with a logged warning** — never an exception. Zero usable specimens → `{}`.
- **B-L5** Returns `dict[int, ChampionGallery]` keyed by specimen number.

### B-B: binding at the enrolment freeze (`session_linker.py`)

- **B-B1** Binding runs inside the roster freeze, after Start galleries are
  built, **only if** persistent galleries are non-empty. An enrolled identity
  with no usable Start crops (`start_views is None`) never binds.
- **B-B2** For each enrolled identity **in ascending raw-id order**, call
  `matcher.score(start_crops, start_masks, {}, persistent_galleries)` where
  `start_crops/start_masks` are the same best-≤3-by-quality enrolment crops
  used for `start_views`, and `persistent_galleries` is the **complete** loaded
  dict (comparison set = everything; BUILD-BRIEF Trap 2 applies to binding
  too — never score against a shrunken dictionary, no re-score rounds).
- **B-B3** Per-identity confidence gate: let `best`/`second` be the two highest
  scores across specimens. The identity **proposes** its best specimen iff
  `best >= bind_tau` **and** (`len(galleries) == 1` or
  `best - second >= bind_margin`). (With one gallery the margin is vacuous;
  the matcher's internal K=1 `cos_tau` gate already zeroes hopeless scores —
  do not touch `cos_tau`.)
- **B-B4** **Greedy one-to-one, no forced links** (deliberate deviation from
  §3's "Hungarian", consistent with §6.5's shipped philosophy — the reason to
  state in the design doc: a losing identity must fall back to session-only,
  **never** be pushed to its second-choice specimen, because a wrong bind
  poisons every later decision while session-only merely has fewer views):
  sort proposals by descending score, tie-break ascending raw id; award a
  specimen to the first proposer; later proposers of a claimed specimen become
  **session-only**.
- **B-B5** A bound identity's matching gallery becomes
  `persistent ∪ Start (∪ last-seen when it later exists)` — i.e.
  `_comparison_galleries()` concatenates the bound specimen's persistent views
  with the identity's session views. Unbound identities keep session views
  only.
- **B-B6** Bind exactly once, at freeze. No later event (link, death, return,
  batch decision) may change any identity↔specimen binding.

### B-N: session-id assignment & the emitted id space

Grilled with Bram 2026-07-15 — "reserved specimen numbers":

- **B-N1** Bound identity → `session_id = specimen_number`.
- **B-N2** Session-only identity → `session_id = base + k` (k = 1, 2, … in
  ascending raw-id order over the session-only identities), where
  `base = max(loaded specimen numbers)` when any persistent galleries were
  loaded, else `0`. So: photos loaded + partial bind → e.g. `{1,2,5,9,10}`;
  no photos at all → `{1..N}` contiguous. **A specimen number is never worn
  by an identity that didn't confidently bind.**
- **B-N3** This replaces today's `session_id = raw_id` promotion **in all
  cases** (deliberate, grilled: renumbering fixes "Instrument 10 on a tray of
  8" even when no galleries load). `roster` returns the frozenset of assigned
  session ids.
- **B-N4** Every emitted id for a track that is NOT mapped to a roster
  identity — pending tracks in their evidence window, deferred tracks, and
  settled-Unknown tracks — is `raw_id + unknown_id_offset`. Internal state
  stays keyed by raw ids; only the *emitted* `tracker_id` is offset. Roster
  session ids are emitted unchanged. (This keeps the roster and the unknown id
  ranges disjoint forever; the seam contract's "Unknown = not in roster" test
  is unaffected.)
- **B-N5** Every frame *strictly before* the freeze passes its raw ids through,
  as today. **The freeze frame ITSELF emits session ids** — the flip happens on
  the very call in which the roster freezes, not on the following one (~0.7 s
  in, inside the app's 1 s entry debounce — same provisional-id mechanism the
  seam already documents for link flips). On that frame, tracks that are not
  roster identities emit in the offset id space per B-N4, exactly like every
  other post-freeze frame.

  > **Clarified 2026-07-15 by the orchestrator — was ambiguous. FLAG to Bram.**
  > The original wording ("During the enrolment window (pre-freeze) detections
  > pass through with their raw ids… At the freeze the emitted ids flip") let
  > the freeze frame read as either side of the boundary. The T08 test-writer
  > and the T08 coder independently read it the **lenient** way (flip on the
  > NEXT frame); Codex read it **strict**. Resolved **strict**, for two reasons:
  >
  > 1. **The spec's own timing only works strict.** fps is pinned at 3.0, so the
  >    enrolment window is 2 frames: the freeze frame is t≈0.67 s, the next is
  >    t≈1.0 s. Only flipping on the freeze frame is "~0.7 s in, inside the 1 s
  >    entry debounce" — the lenient reading lands the flip *at* the debounce
  >    boundary, destroying the safety argument the parenthetical exists to make.
  > 2. **The lenient reading is a live cross-seam bug, not a cosmetic lag.** The
  >    roster freeze publishes `self._roster` (session ids) before the frame
  >    returns, and the app samples `tracker.roster` and the emitted ids in the
  >    SAME tick, then intersects them (T10 B-C1/B-S1). Emitting raw ids on that
  >    frame lets raw ids **collide** with session ids: a tray of 8 with raw ids
  >    `{3,5,7,9,10,11,12,14}` intersected against roster `{1..8}` yields
  >    `{3,5,7}` — presence booked against three instruments that are not on the
  >    table. Route (b) was chosen precisely to remove that class of
  >    coordination bug; the lenient reading reintroduces it for one frame.
  >
  > This invalidates any test asserting the freeze frame passes raw ids through
  > — see the migration list below.
- **B-N6** If any assigned session id would reach `unknown_id_offset`, log a
  warning (do not raise; demo-scale rosters never get near it).

### B-R: reset & composition

- **B-R1** `reset()` clears all session state but **keeps** the constructor's
  persistent galleries (already embedded) for the next enrolment.
- **B-R2** `load_tracker()` loads galleries eagerly (after matcher
  construction, before returning) via `load_persistent_galleries` and passes
  them to `SessionLinker`. Missing dir / `None` → linker gets `None`, INFO log.
  `reset()` must not re-load or re-embed.
- **B-R3** `_RealInstrumentTracker.roster` and `FakeInstrumentTracker.roster`
  per the Public interface. The protocol stays `runtime_checkable`-conformant
  for both.

### B-O: observability (feeds the Trap-4/Trap-5 measurements)

- **B-O1** Freeze INFO log gains: bound mapping (session id → specimen),
  session-only ids, per-identity best/second bind scores, and the freeze's
  total `build_ms` (existing counter, now including the binding solves).
- **B-O2** Batch-decision INFO log gains per-candidate **atom counts**
  (`views.shape[0]` per comparison gallery) so replay traces expose the
  bound-vs-unbound gallery-size asymmetry.

## Existing tests the new contract invalidates (test-writer owns migrating them)

`model/tests/pipelines/test_session_linker.py` (22 tests) encodes the
superseded conventions `session_id == raw_id` and `unknown emits raw id`.
Migrate every affected assertion to the new contract (constructed WITHOUT
persistent galleries unless the test is about binding: session-only numbering
from 1 in ascending raw-id order; unknown emissions offset — construct linkers
with an explicit `unknown_id_offset` rather than asserting the default
constant). Programmable fake matchers that key `score()`/`accept()` responses
by session id must key by the RENUMBERED ids. `test_tracking.py` may need the
`roster` property on fakes/protocol checks. `test_champion.py` and everything
under `tests/components|data` are untouched. The full suite (73 before this
build) must be green at the end, with new binding/loader/numbering tests on
top.

**B-N5 clarification (added 2026-07-15, second migration wave).** B-N5 now says
explicitly that **the freeze frame itself emits session ids**. Three tests
encode the superseded lenient reading and must be migrated to the clarified
contract:

- `test_b_n5_enrolled_ids_flip_from_raw_to_session_at_the_freeze`
- `test_b1_2_passthrough_before_and_during_enrolment` — its "during enrolment"
  passthrough claim must now stop *strictly before* the freeze frame. Keep the
  pre-freeze passthrough assertion; it is still correct and still load-bearing.
- `test_b8_reset_drops_all_state` — fails only incidentally, via a scenario that
  runs through a freeze. Preserve what it actually tests (reset clears session
  state); do not weaken it.

The contract to assert: on the freeze frame, an enrolled track emits its
**session id**, a non-roster track emits `raw_id + unknown_id_offset` (B-N4),
and `linker.roster` sampled in that same call is consistent with the ids just
emitted. Frames strictly before the freeze still emit raw ids.

## Dependencies / reuse

- `orc_model.pipelines.matching` — `ChampionGallery` (opaque `.views`
  `(n, D)` ndarray + `.identity`), `ChampionMethod.build_gallery/score/accept`.
  **Do not modify the matcher.** The linker treats gallery objects opaquely
  (concatenate `.views`, pass dicts into `score()`), exactly as today.
- `pycocotools.mask.decode`, `PIL.Image`, `numpy` — already dependencies.
- Logic to mirror (vendored, cite in a comment):
  `experiments/matcher-autoresearch/frozen/loader.py:_load_coco_crops`.
- Files the coder may touch: `persistent_gallery.py` (new),
  `session_linker.py`, `tracking.py`. Nothing else. The stale §3 TODO in
  `session_linker.py`'s module docstring must be replaced by a short
  description of the shipped binding (it falsely implies T08 is unbuilt).

## Tunable parameters (never hard-assert values in tests)

`bind_tau`, `bind_margin`, `unknown_id_offset`, `DEFAULT_INSTRUMENTS_DIR`,
crop-quality internals. Tests pass explicit values through the constructor and
assert *behaviour* (gating, ordering, disjointness), not the defaults.

## Acceptance beyond unit tests (orchestrator runs these, not the coder)

Cached replays of both takes (roster 8, zero comparison violations, six Take B
foreign tracks Unknown, batch counts 24/46, the 302 s / 368 s fail-closed
rejects untouched), atom-asymmetry measurement, `build_ms` re-measurement.
**Never lower `tau`/`margin`/`cos_tau` to make a replay pass.**
