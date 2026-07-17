# SPEC-B — T11 Phase 2: backend state (snapshot fields, readiness, prepare/start/confidence)

> **⚠️ SUPERSEDED IN PART (2026-07-17):** `POST /recording/prepare` and
> `PrepareResponse` were removed (no Rescan; New recording never called them).
> `Session.prepare()` remains an internal confidence-change cleanup. Start remains
> the no-reset, fail-closed transition described below: it preserves the approved
> roster. Ignore only the public prepare-endpoint sections/tests. Canonical:
> `app/docs/api-contract.md`.

Authoritative spec for the **backend layer** of ticket T11 (sections B1–B5 of
`docs/wayfinder/session-linker/tickets/T11-safe-setup-enrolment.md`, plus the
backend tests it enables). Phase 1 (model catalog seam + catalog-only freeze) is
DONE and merged; `tracker.catalog` / `tracker.roster` / `data["resolving"]` all
cross the seam already, and `ScenarioTracker`/`FakeInstrumentTracker` implement
`catalog`.

Two agents work from THIS document, blind to each other:
- **coder** — edits production source only: `app/backend/backend/*.py`. MUST NOT
  open/edit any test file.
- **test-writer** — edits tests only: `app/backend/tests/*.py`. Writes the
  backend tests from THIS spec; may read production source for signatures but
  writes assertions from the spec.

Repo root: `/Users/bramdecraeke/Documents/Projects/Github/P_ss/MVP-OKcamera`.
Backend source: `app/backend/backend/`. Tests: `app/backend/tests/`.
Interpreter/venv: `app/backend/.venv/bin/python`. Run tests from `app/backend/`:
`.venv/bin/python -m pytest tests/ -q`.

## What Phase 2 is (and is NOT)

IN: same-tick catalog+resolving in the capture snapshot and the on_frame
callback; pure `SetupReadiness` + `Session.prepare`; the `POST /recording/prepare`
endpoint; `POST /recording/start` stops resetting the tracker and becomes a
server-enforced fail-closed gate; the runtime confidence API + capture-thread
`set_confidence_and_reset`; mutation serialisation; `/status` carrying the
readiness fields + `detector_control`.

OUT (Phase 3, do NOT do here): the setup `detections[]` per-item
`state`/`label`/`colour` restructure (B6), `catalog_colour` (R2), renderer
unification (R1), thumbnail label/state (R3), and all api-contract.md / OpenAPI /
`schema.d.ts` / fixtures / MSW / frontend work. In Phase 2, leave
`setup.detections` in its CURRENT shape (`{tracker_id, label, thumbnail}`,
labels still `Instrument N` from `thumbnails.build_detections`) — only ADD the
readiness scalar fields and `detector_control`. Leave the recording overlay
colour path (`roster_colour`) untouched.

Also OUT (charter, don't do): `backend/debug.py` freeze-panel guard (carried from
Phase 1 review) — Phase 3.

---

## B1 — capture snapshot carries catalog + resolving (`capture.py`)

### DetectionBox → frozen dataclass

Replace the tuple alias (`capture.py:40`):

```python
@dataclass(frozen=True)
class DetectionBox:
    tracker_id: int
    xyxy: tuple[float, float, float, float]
    resolving: bool
```

Add `from dataclasses import dataclass` to imports. All fields are builtin
Python types (int/float/bool) — no numpy retained (mirrors the existing
`present_ids` builtin-int contract, AC12).

### Latest gains `catalog`

Add to the `Latest` NamedTuple (after `roster`):

```python
catalog: frozenset[int]        # tracker.catalog sampled the SAME tick as
                               # present_ids/roster (builtin ints). Constant for
                               # the tracker's life; carried so /status can report
                               # the expected set without touching the tracker.
```

### OnFrame callback widens to same-tick 5-tuple

Replace (`capture.py:61-62`):

```python
OnFrame = Callable[[float, frozenset[int], frozenset[int], frozenset[int], frozenset[int]], None]
"""(t, present_ids, roster, catalog, resolving_ids) — all sampled the same tick."""
```

### `_capture_tick` samples catalog + resolving from the same update result

In `_capture_tick` (`capture.py:228-268`), after `dets = self._tracker.update(frame)`:

- sample `roster` (exists) and `catalog = frozenset(int(x) for x in self._tracker.catalog)` the same tick;
- derive per-row resolving from `dets.data`: a row is resolving iff its
  `data["resolving"]` entry is truthy. A MISSING `resolving` key defaults every
  row to `False` (the existing safe default — see tracker-interface.md). Build:
  - `resolving_ids = frozenset` of the present `tracker_id`s whose row resolving flag is True;
  - the per-detection `DetectionBox(tracker_id, xyxy, resolving)` list (row-aligned).
- put `catalog` into `Latest`;
- pass the 5-tuple to `self._notify(t, present_ids, roster, catalog, resolving_ids)`.

Update `_notify` signature + the `self._on_frame(...)` call to the 5-tuple.
Update `_detection_boxes(dets)` to build `DetectionBox` dataclasses carrying the
row-aligned resolving flag (read `dets.data.get("resolving")`; None/absent → all
False). Keep it numpy-free (plain int/float/bool).

Helper for the resolving flags (row-aligned, safe default):
```python
def _resolving_flags(dets):
    flags = dets.data.get("resolving") if dets.data is not None else None
    n = len(dets.tracker_id) if dets.tracker_id is not None else 0
    if flags is None:
        return [False] * n
    return [bool(f) for f in flags]
```

### `set_confidence_and_reset` — one serialised capture command (B5 plumbing)

Reuse the existing reset handshake. Add a `self._pending_confidence: float | None = None`
field (init in `__init__`). Add:

```python
def set_confidence_and_reset(self, confidence: float, timeout_s: float = 2.0) -> None:
    """Atomically set the detector confidence and reset the tracker, applied
    together on the CAPTURE thread between frames (never mutate tracker.confidence
    from the HTTP thread mid-update). Serialised with reset_tracker() via the
    same _reset_lock, so two tracker mutations can never overlap."""
    thread = self._thread
    if thread is None:
        # Not running yet: just set it so a later start() uses the new value.
        self._tracker.confidence = confidence
        return
    if not thread.is_alive():
        raise TrackerResetError("capture thread is not running")
    with self._reset_lock:
        self._pending_confidence = confidence
        self._reset_error = None
        self._reset_done.clear()
        self._reset_request.set()
        if not self._reset_done.wait(timeout_s):
            raise TimeoutError("confidence change was not applied (capture thread stalled)")
        if self._reset_error is not None:
            raise TrackerResetError("confidence change failed") from self._reset_error
```

Update `_apply_pending_reset` to apply the pending confidence (if any) BEFORE the
reset, and always clear it:

```python
def _apply_pending_reset(self) -> None:
    if not self._reset_request.is_set():
        return
    try:
        if self._pending_confidence is not None:
            self._tracker.confidence = self._pending_confidence
        self._tracker.reset()
    except BaseException as exc:
        self._reset_error = exc
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        logger.exception("tracker reset/confidence apply failed; capture loop will continue")
    finally:
        self._pending_confidence = None
        self._reset_request.clear()
        self._reset_done.set()
```

`reset_tracker()` is unchanged (it leaves `_pending_confidence` None → pure reset).

---

## B2 — Session owns setup readiness as pure state (`session.py`)

`Session` stays clock-free and tracker-free. It STORES the latest same-tick setup
observation and computes an immutable `SetupReadiness`. It does not read the
tracker or a clock.

### `SetupReadiness` dataclass (module level)

```python
@dataclass(frozen=True)
class SetupReadiness:
    detected_count: int
    expected_count: int
    recognised_count: int
    resolving_count: int
    unknown_count: int
    stable_for_s: float
    ready: bool
    blocking_reason: str | None   # "recognising" | "missing_instruments" |
                                  # "unknown_objects" | "hold_steady" | None
```

### readiness threshold

Add a Session constructor param `setup_stable_s: float = 2.0` (stored as
`self._setup_stable_s`). This is D3 condition 5 ("unchanged for at least 2.0 s").

### store the observation

`observe` widens to accept the same-tick catalog + resolving:

```python
def observe(self, t, present_ids, roster=None, catalog=None, resolving_ids=None) -> None:
```

Keep the existing behaviour (advance, `_idset` stability tracking, RECORDING
roster-filtered debounce) EXACTLY. Additionally store the latest setup
observation so `setup_readiness` can be pure:

```python
self._setup_present = present_ids
self._setup_roster = roster or frozenset()
self._setup_catalog = catalog or frozenset()
self._setup_resolving = resolving_ids or frozenset()
```

Initialise these to `None`/empty in `__init__` (`self._setup_present = None`,
the rest `frozenset()`). The recording debounce path still uses `roster` exactly
as today (`recorded_ids = present_ids if roster is None else present_ids & roster`).

### `setup_readiness(t)` — pure computation

Valid only in SETUP/FINISHED (raise `InvalidPhase` otherwise). Definition:

```python
def setup_readiness(self, t: float) -> SetupReadiness:
    if self._phase not in (Phase.SETUP, Phase.FINISHED):
        raise InvalidPhase(f"setup_readiness invalid in {self._phase}")
    if self._setup_present is None:
        # Nothing observed yet -> not ready, still coming up.
        return SetupReadiness(0, 0, 0, 0, 0, 0.0, False, "recognising")

    present = self._setup_present
    roster = self._setup_roster
    catalog = self._setup_catalog
    resolving_ids = self._setup_resolving

    recognised = present & roster
    non_roster = present - roster
    resolving = resolving_ids & present
    unknown = non_roster - resolving

    detected_count = len(present)
    expected_count = len(catalog)
    recognised_count = len(recognised)
    resolving_count = len(resolving)
    unknown_count = len(unknown)
    stable_for_s = max(0.0, self._effective_t(t) - self._idset_since_t)

    all_recognised = expected_count > 0 and recognised == catalog
    ready = (
        all_recognised
        and unknown_count == 0
        and resolving_count == 0
        and stable_for_s >= self._setup_stable_s
    )
    if ready:
        blocking_reason = None
    elif resolving_count > 0:
        blocking_reason = "recognising"
    elif unknown_count > 0:
        blocking_reason = "unknown_objects"
    elif not all_recognised:
        blocking_reason = "missing_instruments"
    else:
        blocking_reason = "hold_steady"
    return SetupReadiness(detected_count, expected_count, recognised_count,
                          resolving_count, unknown_count, stable_for_s, ready,
                          blocking_reason)
```

Priority order (D3 / B2): recognising > unknown_objects > missing_instruments >
hold_steady. Capture health is NOT in this pure result — the API combines it.

`setup_readiness(t)` replaces the legacy `setup_status(t)` projection; the
legacy method was later removed once its remaining tests migrated.

### `Session.prepare(t)`

```python
def prepare(self, t: float) -> None:
    """Begin a fresh setup pass. Legal only from SETUP or FINISHED. Clears the
    setup id-set/stability and recording tracks so the next observation starts at
    zero seconds, but PRESERVES the finished report and the current phase (the
    previous report is discarded only on a successful start())."""
    if self._phase not in (Phase.SETUP, Phase.FINISHED):
        raise InvalidPhase(f"cannot prepare from {self._phase}")
    self._advance(t, strict=False)
    self._idset = None
    self._idset_since_t = t
    self._setup_present = None
    self._setup_roster = frozenset()
    self._setup_catalog = frozenset()
    self._setup_resolving = frozenset()
    self._tracks = {}
    # Deliberately does NOT touch self._phase or self._report.
```

---

## Endpoints and locks (`main.py`)

### Locks

Add a coarse `mutation_lock = threading.Lock()` alongside the existing `lock`.
- `lock` (session lock): held briefly for session reads/writes and by `_on_frame`.
  NEVER held while a blocking capture command runs (would deadlock the capture
  thread, which needs `lock` for `_on_frame`).
- `mutation_lock`: held for the WHOLE of prepare / start / confidence so those
  three cannot interleave (two tracker mutations can never overlap — test 7).
  `_on_frame` does NOT take `mutation_lock`, so a handler can hold it across a
  blocking capture command while the capture thread keeps publishing frames.

`_on_frame` widens to the 5-tuple:
```python
def _on_frame(t, present_ids, roster, catalog, resolving_ids) -> None:
    with lock:
        session.observe(t, present_ids, roster, catalog, resolving_ids)
```

### create_app signature

Add `default_confidence: float = 0.5` (keyword, defaulted so existing
`create_app(capture, session, model_version)` test calls still work). Store the
runtime confidence view:
```python
detector_state = {"confidence": default_confidence, "default": default_confidence}
```
`main()` passes `default_confidence=settings.tracker.detector.confidence`.

Confidence bounds are module constants: `CONF_MIN = 0.30`, `CONF_MAX = 0.90`,
`CONF_STEP = 0.05`.

### Pydantic models

Add:
```python
class DetectorControlModel(BaseModel):
    confidence: float
    default_confidence: float
    minimum: float
    maximum: float
    step: float

class PrepareResponse(BaseModel):
    prepared_at: str

class ConfidencePatch(BaseModel):
    confidence: float = Field(ge=CONF_MIN, le=CONF_MAX)  # out-of-range -> 422
```

Extend `SetupStatus` with the readiness fields (KEEP `detections` as the current
`list[DetectionModel]` shape — Phase 3 restructures it):
```python
class SetupStatus(BaseModel):
    detected_count: int
    expected_count: int
    recognised_count: int
    resolving_count: int
    unknown_count: int
    stable_for_s: float
    ready: bool
    blocking_reason: str | None
    detections: list[DetectionModel]
```

Extend `StatusResponse` with `detector_control: DetectorControlModel`.

Helper to build the control object:
```python
def _detector_control() -> DetectorControlModel:
    return DetectorControlModel(
        confidence=detector_state["confidence"],
        default_confidence=detector_state["default"],
        minimum=CONF_MIN, maximum=CONF_MAX, step=CONF_STEP,
    )
```

### `/status` (`get_status`)

- Always include `detector_control=_detector_control()` in the response.
- Replace the legacy setup-status path: when phase in SETUP/FINISHED, read
  `readiness = session.setup_readiness(t)` under `lock` (cheap, pure). Build the
  setup object from the readiness fields + the existing `_setup_detections()`
  crops (unchanged). Combine health into `ready`:
  ```python
  setup = SetupStatus(
      detected_count=readiness.detected_count,
      expected_count=readiness.expected_count,
      recognised_count=readiness.recognised_count,
      resolving_count=readiness.resolving_count,
      unknown_count=readiness.unknown_count,
      stable_for_s=readiness.stable_for_s,
      ready=readiness.ready and capture_health == "ok",   # D3 cond 1 combined here
      blocking_reason=readiness.blocking_reason,
      detections=_setup_detections(),
  )
  ```
  (`blocking_reason` stays the pure value; the frontend shows the camera-stalled
  message when `capture_health != "ok"`, which it reads separately.)
- Recording path unchanged (still `roster_colour`, Phase 3 changes it).

### `POST /recording/prepare` (B3)

```python
@app.post("/recording/prepare", response_model=PrepareResponse)
def post_recording_prepare() -> PrepareResponse:
    with mutation_lock:
        with lock:
            if session.phase is Phase.RECORDING:
                raise HTTPException(status_code=409, detail="cannot prepare during recording")
        # Reset the tracker OUTSIDE the session lock (blocks on the capture thread).
        try:
            capture.reset_tracker()
        except TimeoutError:
            raise HTTPException(status_code=503, detail="capture stalled")
        except TrackerResetError:
            raise HTTPException(status_code=503, detail="tracker reset failed")
        with lock:
            if session.phase is Phase.RECORDING:
                raise HTTPException(status_code=409, detail="cannot prepare during recording")
            session.prepare(clock())
            renderer.set_recording(False)   # leave overlay in non-recording/setup state
            prepared_at = now().isoformat()
    return PrepareResponse(prepared_at=prepared_at)
```
Preserves the finished report (session.prepare does not touch it). Allowed from
setup and finished; 409 only during recording.

### `POST /recording/start` — no reset, fail-closed gate (B4)

Remove the `capture.reset_tracker()` call entirely. Replace with a server-enforced
gate on readiness + health, all under the locks (no blocking capture call, so it
can stay under `lock`):

```python
@app.post("/recording/start", response_model=StartResponse)
def post_recording_start() -> StartResponse:
    with mutation_lock:
        with lock:
            t = clock()
            if session.phase not in (Phase.SETUP, Phase.FINISHED):
                raise HTTPException(status_code=409, detail=f"cannot start from {session.phase}")
            capture_health = "ok" if capture.health == "ok" else "stalled"
            readiness = session.setup_readiness(t)
            detail = _start_block_detail(readiness, capture_health)
            if detail is not None:
                raise HTTPException(status_code=409, detail=detail)
            session.start(clock())
            renderer.set_recording(True)
            timestamps["started_at"] = now().isoformat()
            timestamps["stopped_at"] = None
            return StartResponse(started_at=timestamps["started_at"])
```

with, at module or closure scope:
```python
def _start_block_detail(readiness, capture_health) -> str | None:
    if capture_health != "ok":
        return "capture stalled"
    reason = readiness.blocking_reason
    if reason is None:
        return None
    if reason == "recognising":
        return "recognition still in progress"
    if reason == "unknown_objects":
        return "remove unknown objects before starting"
    if reason == "missing_instruments":
        return f"all {readiness.expected_count} instruments must be recognised before starting"
    if reason == "hold_steady":
        return "hold the tray steady before starting"
    return "setup is not ready"
```

The gate is a 409 for every not-ready state, including unhealthy capture. Start
MUST NOT call `capture.reset_tracker()` or touch tracker confidence/roster/catalog.

### `PATCH /settings/detection-confidence` (B5)

```python
@app.patch("/settings/detection-confidence", response_model=DetectorControlModel)
def patch_detection_confidence(body: ConfidencePatch) -> DetectorControlModel:
    value = body.confidence   # Field(ge=CONF_MIN, le=CONF_MAX) already 422s out-of-range
    with mutation_lock:
        with lock:
            if session.phase is Phase.RECORDING:
                raise HTTPException(status_code=409, detail="cannot change confidence during recording")
            current = detector_state["confidence"]
        if value == current:
            return _detector_control()   # no-op: no reset, immediate return
        try:
            capture.set_confidence_and_reset(value)
        except TimeoutError:
            raise HTTPException(status_code=503, detail="capture stalled")
        except TrackerResetError:
            raise HTTPException(status_code=503, detail="confidence change failed")
        detector_state["confidence"] = value   # source of truth updated only on success
        with lock:
            session.prepare(clock())   # a changed confidence restarts enrolment; readiness clears
    return _detector_control()
```

Rules satisfied: 422 out-of-range (Pydantic Field); 409 while recording; no-op
value returns without resetting; a changed value is applied as one serialised
capture command then readiness cleared; response sent only after the reset is
acknowledged (set_confidence_and_reset blocks); on failure the last confirmed
`detector_state["confidence"]` is preserved (503, not updated); TOML never
written.

---

## `thumbnails.py` — minimal adaptation only

`build_detections` currently unpacks `DetectionBox` as a `(tracker_id, xyxy)`
tuple. Now `DetectionBox` is a frozen dataclass — update the access to
`.tracker_id` / `.xyxy`. Keep the CURRENT label behaviour (`Instrument
{tracker_id}`) for Phase 2; the label/state/colour-from-catalog change is Phase 3
(R3). Do not use the new `resolving` field yet.

---

## Backend tests (test-writer) — T11 §8, tests 1–7, 10–12

Tests 8 and 9 (setup detection labels/colours) are Phase 3 — do NOT write them.
Put tests in the existing files (`test_capture.py`, `test_session.py`,
`test_api.py`, `test_fakes.py`) mirroring their conventions (the file has
`FakeCapture`, `_SpySession`, `_RecordingSession`, `_FixedRosterTracker`,
`_RosterTracker`, tracker doubles, etc.).

1. **capture snapshot stores catalog + row-aligned resolving** (`test_capture.py`).
   A tracker double whose `update` returns detections with `data["resolving"]`
   and a `catalog`: after a tick, `snapshot().catalog` equals the sampled catalog
   (builtin ints, JSON-native) and each `DetectionBox.resolving` is row-aligned;
   a tracker returning no `resolving` key defaults all to False; the `on_frame`
   callback receives the 5-tuple `(t, present_ids, roster, catalog,
   resolving_ids)` same-tick.
2. **readiness false while resolving / with missing knowns / with any unknown /
   before 2 s** (`test_session.py`, pure Session). Feed observations via
   `observe(t, present, roster, catalog, resolving_ids)` and assert
   `setup_readiness(t)`: resolving present → not ready, reason "recognising";
   catalog member absent → not ready, "missing_instruments"; a non-roster settled
   id → not ready, "unknown_objects"; full catalog present+settled but < 2 s
   stable → not ready, "hold_steady". Derive the 2 s from the constructor param
   (pass `setup_stable_s`), don't hard-code timing.
3. **readiness true only for exact present catalog after stability**
   (`test_session.py`). present == roster == catalog, no resolving, no unknown,
   stable ≥ threshold → ready True, blocking_reason None. Adding one extra
   (unknown) id, or removing one catalog id, flips ready False.
4. **server rejects Start for every blocking state even when called directly**
   (`test_api.py`). Drive `create_app` so readiness returns each blocking reason
   (a session double or scripted observations) and assert `POST /recording/start`
   → 409 with the mapped detail, for recognising / unknown / missing / hold_steady
   / unhealthy-capture. Even when the frontend would disable the button, the
   server refuses.
5. **successful Start does not reset the tracker** (`test_api.py`). With a
   `FakeCapture` that records `reset_tracker`/`set_confidence_and_reset` calls and
   a ready session, `POST /recording/start` succeeds and records ZERO tracker
   resets.
6. **prepare resets tracker + readiness and preserves a finished report**
   (`test_api.py`). From FINISHED with a stored report: `POST /recording/prepare`
   → 200 `{prepared_at}`, calls `capture.reset_tracker()` once, clears readiness
   (a following `/status` shows recognising/zeroed setup), and `GET /report` STILL
   returns the previous report (preserved). `POST /recording/prepare` during
   RECORDING → 409.
7. **prepare / Start / confidence mutations cannot overlap** (`test_api.py`).
   Assert the three mutation handlers serialise on the mutation lock — e.g. a
   `FakeCapture` whose `reset_tracker`/`set_confidence_and_reset` blocks on a
   gate lets you show a second mutation call cannot proceed until the first
   releases (no two tracker mutations interleave). A focused test of the lock
   discipline is acceptable (you need not spin real threads if the design exposes
   the ordering) — but prefer a real 2-thread interleaving test if practical.
10. **confidence PATCH** (`test_api.py`): out-of-range (`0.2`, `1.0`) → 422;
    forbidden while RECORDING → 409; a changed in-range value → calls
    `set_confidence_and_reset(value)` once, returns the updated `detector_control`,
    and clears readiness; a no-op value equal to current → returns 200 WITHOUT any
    `set_confidence_and_reset`/reset call; the TOML file is never written (assert
    no write to the config path / that `mvp.toml` is untouched).
11. **failed confidence reconfiguration preserves the last confirmed value**
    (`test_api.py`). `set_confidence_and_reset` raising `TrackerResetError` (or
    TimeoutError) → PATCH returns 503 and a subsequent `/status`
    `detector_control.confidence` still shows the last confirmed value (unchanged).
12. **fake mode implements catalog + can exercise an extra setup object**
    (`test_fakes.py` / `test_api.py`). `ScenarioTracker` exposes `catalog ==
    frozenset(range(1, n+1))`; with a foreign object present its readiness/roster
    treats it as non-roster (unknown/resolving), never in the catalog roster.

Also update the existing `_on_frame` / observe / capture callers in the test
doubles to the new 5-tuple signature where they assert on it (e.g. existing
`test_capture.py` BC1 on_frame test, `test_session.py` observe calls). Any
existing test that calls `session.observe(t, present, roster)` positionally still
works (catalog/resolving default to None) — only widen the ones that must assert
the new behaviour. `create_app(...)` existing calls keep working (default
confidence defaulted).

## Definition of done

- Capture snapshot + on_frame carry same-tick catalog + row-aligned resolving.
- Pure `SetupReadiness` + `Session.prepare`; `/status` exposes readiness +
  `detector_control`.
- `POST /recording/prepare` works and preserves a finished report; Start no
  longer resets and fail-closes on readiness+health; confidence PATCH with the
  serialised capture command and all its rules.
- Mutation serialisation via the coarse lock.
- `app/backend` test suite green (`.venv/bin/python -m pytest tests/ -q`), and the
  `model/` suite still green.
