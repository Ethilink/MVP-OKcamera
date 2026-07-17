# SPEC-M — T11 Phase 1: model layer (catalog seam + catalog-only freeze)

> **⚠️ SUPERSEDED IN PART (2026-07-17):** the public `prepare` endpoint and Rescan
> button were later dropped. Recording Start preserves the approved roster and
> does not reset; a changed confidence is the only public setup action that starts
> a fresh reset/enrolment pass. This model layer is otherwise unchanged. Canonical:
> `app/docs/api-contract.md`.

This is the authoritative spec for the **model layer only** of ticket T11
(`docs/wayfinder/session-linker/tickets/T11-safe-setup-enrolment.md`, sections
M1–M4 and the "Model tests" list in §8). Two agents work from THIS document:

- **coder** — edits production source only (`model/src/**`, `model/docs/**`,
  `app/mvp.toml`). MUST NOT open or edit any test file. Implements against this
  spec; never sees the tests.
- **test-writer** — edits test files only (`model/tests/**`). Writes the failing
  tests from this spec; may read production source for current signatures but
  writes assertions from THIS spec, not from the coder's new implementation.

Repo root: `/Users/bramdecraeke/Documents/Projects/Github/P_ss/MVP-OKcamera`.
Everything below is `model/`-relative unless it says `app/`.

## Why (one paragraph)

Today, at the enrolment freeze, **every** majority-present raw track becomes a
roster identity — bound to a specimen if it can, otherwise renumbered into a
"session-only" identity. That is the unsafe behaviour T11 kills: a foreign
object on the table during setup silently becomes an instrument. In
**catalog-only mode**, only a track that confidently binds to a loaded persistent
specimen may join the roster; everything else stays in the offset (Unknown)
range and is never renumbered. Legacy (non-catalog) behaviour is preserved
unchanged behind a default-false flag.

---

## M1 — `catalog` crosses the tracker seam

### M1a. Protocol (`src/orc_model/pipelines/tracking.py`, in `InstrumentTracker`, right after the `roster` property ~L103-108)

Add:

```python
@property
def catalog(self) -> frozenset[int]:
    """Persistent specimen IDs available for catalog binding.

    Constant for the tracker lifetime and preserved across reset(). Unlike
    `roster` (empty until the enrolment freeze), `catalog` is known from
    construction. In catalog-only mode `roster` is always a subset of
    `catalog`.
    """
    ...
```

### M1b. `FakeInstrumentTracker` (same file, ~L111-191) — normalise to 1-based

The fake is currently **0-based** (emits tracker_ids `0..n-1`,
`roster == frozenset(range(n))`). T11 M1 requires it normalised to the same
**1-based** convention as the app fake.

- `roster` property → `frozenset(range(1, self.n_instruments + 1))`.
- Add `catalog` property → identical to roster: `frozenset(range(1, self.n_instruments + 1))`
  (the fake's full identity range; constant, independent of `_frame`).
- In `update()`, the emitted tracker id becomes 1-based: the per-instrument loop
  keeps its 0-based index `i` for geometry/phase math, but appends `i + 1` as the
  `tracker_id`. Net effect: emitted ids are `1..n`, matching `roster`/`catalog`.
  Nothing else about the fake changes.

### M1c. `SessionLinker` (`src/orc_model/pipelines/session_linker.py`)

- Constructor: add keyword-only param `catalog_only_enrolment: bool = False`
  (place it near `bind_tau`/`bind_margin`). Store as
  `self._catalog_only_enrolment`.
- Add property (constant across `reset()`, since `reset()` never touches
  `self._persistent_galleries`):

```python
@property
def catalog(self) -> frozenset[int]:
    """Loaded persistent specimen IDs — the fixed catalog. Constant for the
    linker's lifetime; unaffected by reset() or the enrolment freeze."""
    return frozenset(self._persistent_galleries)
```

### M1d. `_RealInstrumentTracker` (`tracking.py`, right after its `roster` property ~L486-488)

```python
@property
def catalog(self) -> frozenset[int]:
    return self._session_linker.catalog
```

---

## M2 — `LinkerConfig` gains catalog-only knobs + validation

### `src/orc_model/pipelines/config.py`, `LinkerConfig` (~L93-131)

Add two fields (defaults preserve legacy behaviour):

```python
catalog_only_enrolment: bool = False
expected_catalog_size: int | None = None
```

Add to `__post_init__` (config-time validation is the positivity check only —
the "requires galleries" / "count must match" checks need the loaded galleries
and live in `load_tracker`, see M2b):

```python
if self.expected_catalog_size is not None and self.expected_catalog_size < 1:
    raise ValueError("expected_catalog_size must be positive when provided")
```

### M2b. `load_tracker` fail-closed wiring (`tracking.py`, ~L308-337)

After `galleries` is resolved (the existing `if config.instruments_dir is None:
... else: galleries = load_persistent_galleries(...) or None` block) and BEFORE
constructing `SessionLinker`, add the fail-closed gate:

```python
if linker_config.catalog_only_enrolment:
    loaded = 0 if galleries is None else len(galleries)
    if galleries is None:
        raise ValueError(
            f"catalog_only_enrolment is set but no persistent galleries loaded "
            f"from {config.instruments_dir!r}; refusing session-only fallback"
        )
    expected = linker_config.expected_catalog_size
    if expected is not None and loaded != expected:
        raise ValueError(
            f"catalog_only_enrolment expected {expected} galleries under "
            f"{config.instruments_dir!r} but loaded {loaded}"
        )
```

Then thread the flag into the `SessionLinker(...)` construction:

```python
catalog_only_enrolment=linker_config.catalog_only_enrolment,
```

`load_persistent_galleries` returns a `dict[int, ChampionGallery]` keyed by
specimen number, or `{}` (→ `None` after `or None`) when nothing usable loads.

---

## M3 — catalog-only freeze behaviour

### `_freeze_roster` (`session_linker.py`, ~L375-452)

Preserve the complete-dictionary scoring, `bind_tau`, `bind_margin`,
one-to-one greedy contest handling, and specimen-number identity in
`_bind_specimens` / `_assign_session_ids` — do **not** touch those methods.

The only change is which raw ids become identities. Today the freeze builds
`_identities` / `_raw_to_session` / `_roster` from **all** `roster_ids`. In
catalog-only mode, build them from **bound raw ids only**, and do not call
`_assign_session_ids` (a bound identity's session id is exactly its specimen
number).

Replace the two lines:

```python
bound, bind_scores = self._bind_specimens(roster_ids, start_crops)
session_ids = self._assign_session_ids(roster_ids, bound)
```

with:

```python
bound, bind_scores = self._bind_specimens(roster_ids, start_crops)
if self._catalog_only_enrolment:
    # Catalog-only: ONLY confidently bound raw tracks join the roster. An
    # unbound setup track (foreign object, below-threshold, or contested
    # loser) is never renumbered into a session-only identity. It emits its
    # offset id on the freeze frame and proceeds through the normal
    # Pending -> Unknown path on later frames. A catalog specimen missed at
    # the freeze has no identity and can only be recognised by a fresh
    # prepare/enrolment pass (Rescan).
    enrolled_raw_ids = [raw_id for raw_id in roster_ids if raw_id in bound]
    session_ids = {raw_id: bound[raw_id] for raw_id in enrolled_raw_ids}
else:
    enrolled_raw_ids = roster_ids
    session_ids = self._assign_session_ids(roster_ids, bound)
```

Then change the identity/raw_to_session construction to iterate
`enrolled_raw_ids` (was `roster_ids`):

```python
self._identities = {
    session_ids[raw_id]: _Identity(
        session_id=session_ids[raw_id],
        active_raw_id=raw_id,
        crop_buffer=deque(maxlen=self._evidence_frames),
        start_views=start_views.get(raw_id),
        bound_specimen=bound.get(raw_id),
    )
    for raw_id in enrolled_raw_ids
}
self._raw_to_session = {raw_id: session_ids[raw_id] for raw_id in enrolled_raw_ids}
self._roster = frozenset(session_ids.values())
self._enrolled = True
```

**Freeze log must not KeyError.** The existing structured freeze log indexes
`session_ids[r]` for `r in roster_ids`; in catalog-only mode unbound raw ids are
absent from `session_ids`. Make the log robust and still complete (it is the M4
diagnostic — it must show every majority-present raw id, its bind score, and
whether it made the roster):

- use `session_ids.get(r)` everywhere the log currently uses `session_ids[r]`;
- `session_only` list: report unbound majority-present tracks. In catalog-only
  mode they have no session id, so report them by raw id:
  `sorted(r for r in roster_ids if r not in bound)`. (In legacy mode this is
  unchanged in spirit; either raw-id or session-id reporting is acceptable so
  long as legacy tests still pass — prefer keeping legacy output identical by
  branching, or verify no test asserts the exact `session_only=` field.)
- `roster_size`: keep as the count of enrolled identities → `len(enrolled_raw_ids)`.

Everything downstream already handles unbound tracks correctly and must NOT be
modified:

- **Freeze frame emission** (`_update_during_enrolment` ~L358-364) already emits
  `self._raw_to_session.get(raw_id, raw_id + offset)`; unbound raw ids are absent
  from `_raw_to_session`, so they emit their offset id automatically.
- **Freeze-frame resolving** (~L370-372) is `[sid not in self._roster]`; unbound
  offset ids are not in roster → `resolving=True`; bound specimen ids are in
  roster → `resolving=False`.
- **Post-freeze** (`_process_row`) sees an unbound raw id not in `_raw_to_session`
  → creates a `_PendingTrack`, emits offset, and the normal batch decision later
  settles it Unknown (a genuinely foreign object is rejected by the matcher; a
  missed catalog specimen has no Missing identity to link into, so it also stays
  Unknown until Rescan).

### Invariants this must satisfy (the coder should self-check these)

1. Catalog-only roster ⊆ catalog, always. No session-only ids, no id above
   `max(catalog)`.
2. `reset()` leaves `catalog` unchanged and `roster` empty.
3. Legacy mode (`catalog_only_enrolment=False`) is byte-for-byte the current
   behaviour: `enrolled_raw_ids is roster_ids` path, `_assign_session_ids` runs.
4. Same-tick consistency holds on the freeze frame: emitted ids, `roster`,
   `resolving`, and `catalog` are mutually consistent for that one `update()`.

---

## M4 — docs

Update `model/docs/tracker-interface.md`:

- Add `catalog` to the `Interface` code block (after `roster`).
- Add a short subsection under "Identity semantics" (mirroring "The roster
  crosses the seam"): `catalog` is a constant `frozenset[int]` of loaded
  persistent specimen ids, known from construction, preserved across `reset()`;
  in catalog-only mode `roster ⊆ catalog` and no id outside `catalog` ever
  enters the roster. Note the fake's `catalog == roster == frozenset(range(1,
  n+1))`.
- Note this is a third deliberate widening of the contract (after `roster` and
  `data["resolving"]`), dated 2026-07-16 (T11).

Optionally add a one-line pointer in `model/docs/linker-design.md` if there is a
natural spot, but do not rewrite it.

## `app/mvp.toml` — enable catalog-only for the demo

In the `[tracker.linker]` table add:

```toml
catalog_only_enrolment = true
expected_catalog_size = 8
```

(There are exactly 8 specimen galleries under `model/data/instruments/`, so this
matches. `mvp_settings` loading rejects unknown keys, so these must be real
`LinkerConfig` fields — which M2 makes them.)

---

## Model tests (test-writer) — T11 §8 list, tests 1–9

Put new tests in `model/tests/pipelines/test_session_linker.py` (freeze/roster
behaviour) and `model/tests/pipelines/test_config.py` / `test_tracking.py`
(config + seam) as appropriate. Mirror existing conventions exactly:

- Reuse the `FakeMatcher` scripting (`matcher.program(marker, [({specimen:
  score}, None)])`), `persistent_gallery(n)`, `rgb_marker_for_bgr`,
  `build_call(rows)`, `UNKNOWN_OFFSET` (= 5000 in this file), and the
  `bind_linker(...)` helper. Add a thin `catalog_linker(...)` helper =
  `bind_linker(..., catalog_only_enrolment=True)`.
- Never hard-assert tunable defaults; derive frame counts from `round(seconds *
  fps)` as the file header instructs.
- Name new tests clearly, e.g. `test_catalog_only_*` and `test_catalog_*`.

Required coverage:

1. **catalog constant + survives reset.** `SessionLinker(..., persistent_galleries={1:…, 2:…})`
   → `.catalog == {1, 2}` before enrolment, after freeze, and after `reset()`.
   Also `FakeInstrumentTracker(n_instruments=3).catalog == {1, 2, 3}` and equal
   before/after `reset()`. (Optionally `_RealInstrumentTracker.catalog` via the
   `test_config`/`test_tracking` monkeypatched path — a linker stub exposing
   `.catalog` suffices.)
2. **catalog-only freeze admits a confidently bound known specimen.**
   `catalog_linker` with `{1: persistent_gallery(1)}`, matcher programmed so the
   crop scores ≥ `bind_tau` (and beats second by `bind_margin`) → after the
   freeze frame `roster == {1}`, an identity exists, emitted id 1 ∈ roster.
3. **below-threshold setup object emits offset + absent from roster.** A quality
   crop scored `< bind_tau` (e.g. programmed `{1: 0.1}` with `bind_tau=0.5`) →
   not bound → roster excludes it; on the freeze frame it emits `raw_id +
   UNKNOWN_OFFSET` with `resolving=True`.
4. **contested binding loser is absent from roster and never takes second
   choice.** Two raw tracks both best-match specimen 1; the lower score loses the
   contest → loser is unbound (not renumbered, not given specimen 2) → absent
   from roster, emits offset.
5. **eight known + rejected foreign → roster exactly == catalog, one non-roster
   emitted detection.** Program 8 specimen crops to bind to specimens 1..8 and a
   9th (foreign) crop to score `< bind_tau`. On the freeze frame: `roster ==
   catalog == frozenset(range(1, 9))`; emitted ids = `{1..8} ∪ {foreign_raw +
   UNKNOWN_OFFSET}`; `resolving` True only for the foreign row.
6. **catalog-only mode never creates a session-only roster id.** Any unbound
   majority-present track (from tests 3/4/5) never appears in `roster`, and
   `roster ⊆ linker.catalog` in every catalog-only scenario. Assert no id ≥ the
   first session-only number that legacy mode WOULD have assigned.
7. **legacy non-catalog mode retains session-only behaviour.** With
   `catalog_only_enrolment=False` (default), the existing `test_b_b*` binding
   tests stay green, and a targeted test shows an unbound majority-present track
   still gets a session-only id and joins the roster (i.e. the pre-T11
   behaviour). Do not delete or weaken existing binding tests.
8. **missing / wrong-sized persistent catalog fails fast in safe mode.** Drive
   `load_tracker` (mirror `test_config.py`'s monkeypatch style: stub
   `_build_detector`, `ChampionMethod`, `SessionLinker`, and
   `load_persistent_galleries`). With a `TrackerConfig` whose
   `linker.catalog_only_enrolment=True` and `expected_catalog_size=8`:
   - galleries load empty/None → `load_tracker` raises `ValueError` naming the
     directory;
   - galleries load a wrong count (e.g. 7) → `load_tracker` raises `ValueError`
     naming expected vs loaded.
   Also a `LinkerConfig(expected_catalog_size=0)` → `ValueError` at config time.
9. **freeze-frame IDs, roster, resolving, catalog same-tick consistent.**
   Strengthen test 5 (or a dedicated test): read `emitted`, `linker.roster`,
   `output.data["resolving"]`, and `linker.catalog` from the SAME `update()`
   return and assert they are mutually consistent (every in-roster row
   `resolving=False` and ∈ catalog; every offset row `resolving=True` and ∉
   roster).

Additionally, the test-writer OWNS updating any existing test broken by the
1-based `FakeInstrumentTracker` change (M1b) — e.g. assertions in
`test_tracking.py` that expect roster `{0,1,2}` or emitted ids starting at 0.
Update them to the 1-based convention per this spec.

---

## Out of scope for Phase 1 (do NOT do here)

- `ScenarioTracker.catalog` (app fake) and any `app/backend/**` consumption of
  `catalog` — that is Phase 2. (Widening the runtime_checkable protocol here
  does not break Phase 1 model tests, which only exercise the model trackers.)
- Any capture/session/render/API/frontend change.
- Any matcher threshold change (M4 calibration is Phase 5, real camera).

## Definition of done for Phase 1

- `catalog` on the protocol, both model trackers, and the linker; constant
  across reset.
- `LinkerConfig` fields + validation; `load_tracker` fail-closed; `app/mvp.toml`
  set.
- catalog-only freeze enrols only bound ids, no session-only roster ids, legacy
  mode unchanged.
- `tracker-interface.md` documents catalog.
- `python -m pytest` green in `model/` (new tests + all existing).
