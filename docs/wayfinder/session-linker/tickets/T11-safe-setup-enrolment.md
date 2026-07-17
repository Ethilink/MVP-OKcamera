# T11 — Safe setup enrolment, stable identities, and detection-confidence control

> **⚠️ SUPERSEDED IN PART (2026-07-17):** the public `prepare` endpoint and Rescan
> button were dropped, and **New recording** was never wired to prepare. Recording
> Start preserves the approved roster and performs no tracker reset. The internal
> `Session.prepare()` method stays for a changed confidence, which is the only
> public setup action that resets/restarts enrolment. Every other T11
> decision here still holds (exact-catalog Start gate, unified identity/colour,
> `detector_control` confidence PATCH). **Canonical current contract:
> `app/docs/api-contract.md`** — treat the `prepare`-endpoint / Rescan passages below
> as historical.

**Status:** implementation specification  
**Scope:** `model/`, `app/backend/`, `app/frontend/`, API/schema documentation  
**Primary outcome:** pressing **Track** must never rename/recolour an already
recognised instrument, and a foreign object present during setup must never be
silently enrolled as one of the demo instruments.

This document is the implementation handoff. Where it conflicts with the old
“human judgment / any stable count >= 1” Start-gate wording in
`app/docs/api-contract.md`, this document wins and the old contract must be
updated in the same change.

## 1. Problem statement

The pre-T11 application had two related unsafe behaviours.

1. **Setup and recording expose different identities and colour systems.**
   Setup drew tracker IDs with Supervision's track palette. Recording Start then
   reset the tracker. The linker's new enrolment mapped
   the new raw IDs onto canonical specimen IDs, and recording uses the fixed
   roster palette. A physical object can therefore change both label and colour
   when Track is pressed.
2. **Every stable detection at the enrolment freeze becomes a roster identity.**
   Gallery binding is gated, but failing to bind only sends the detection to a
   “session-only instrument” ID. It does not exclude the detection from the
   roster. A foreign object can therefore appear as an instrument, and a false
   positive bind can claim one of specimen IDs 1–8.

The existing post-enrolment Pending -> Unknown path remains useful. This ticket
closes the special unsafe gap at initial enrolment and makes setup show the same
canonical identity that recording will use.

## 2. Product decisions (not open implementation questions)

### D1 — The KU Leuven demo runs in catalog-only mode

The demo has exactly eight known physical specimens backed by persistent
galleries. In catalog-only mode:

- only a detection that confidently binds one-to-one to a loaded specimen may
  join the roster;
- an unbound or contested setup detection is Pending/Unknown, never a
  “session-only instrument”;
- the expected catalog is the set of loaded persistent-gallery IDs, expected to
  be exactly eight for this deployment;
- the generic session-only linker behaviour may remain available behind a
  disabled-by-default-for-the-app compatibility setting, but the MVP config
  must enable catalog-only mode.

Fail closed. If the eight galleries are missing, duplicated, or do not load,
the real demo application must fail at startup with a clear configuration
error. It must not silently fall back to session-only enrollment.

### D2 — Historical prepare/reset proposal (public endpoint later removed)

Entering a fresh setup pass resets the tracker and starts enrolment. Once setup
has recognised the tray, pressing **Track** only starts recording and must not
reset, renumber, rebind, or recolour anything.

- First application launch starts with a fresh tracker and therefore already
  constitutes a prepare pass.
- The Report screen's **New recording** action calls the new prepare endpoint
  before showing the run-2 setup UI.
- Setup also offers **Rescan**. Rescan performs the same prepare operation.
- Preparing from `finished` preserves the previous report so **Back to report**
  still works. The previous report is discarded only after a successful Start,
  matching current product behaviour.

### D3 — Start is a server-enforced exact-catalog gate

The frontend does not decide safety. It renders the backend's `setup.ready`
field. `POST /recording/start` independently checks the same backend state.

Start is allowed only when all conditions hold:

1. capture health is `ok`;
2. the linker's frozen roster equals the expected catalog (all eight unique
   specimen IDs confidently recognised);
3. every expected catalog identity is currently present;
4. no non-roster detection is present, whether resolving or settled Unknown;
5. the complete emitted ID set has been unchanged for at least 2.0 seconds;
6. phase is `setup` or the run-2 setup view represented by `finished`.

An unknown extra object blocks Start for the demo. This is deliberately stricter
than merely excluding it from the report: it gives the operator a chance to
clear the table before the recording begins.

### D4 — Raw tracker identity is never presented as instrument identity

Before a detection binds to the catalog it is neutral gray:

- resolving: gray mask/box plus spinner, no `Instrument N` label;
- rejected/settled: gray mask/box plus `Unknown`;
- recognised: canonical `Instrument N` label and its fixed catalog colour.

The same rules apply in setup, recording, and the run-2 setup view. There is no
special setup palette and no raw tracker ID in user-facing text.

### D5 — Colours derive from the fixed catalog, not the current roster

Palette assignment must be a pure function of `(catalog, specimen_id)`, using
the index of `specimen_id` in `sorted(catalog)`. It must not use the current
recognised roster as the palette basis: a partial roster later gaining a lower
ID would otherwise shift existing colours.

For the expected catalog `{1, ..., 8}`, each ID has one colour before, during,
and after recording. Unknown always uses the existing neutral gray.

### D6 — Detection confidence is an advanced setup control only

Add an advanced control labelled **Detection confidence threshold**, not a
generic “identity threshold” and not a primary-screen control.

- Current startup default remains the value from `app/mvp.toml` (`0.50` today).
- Allowed runtime range: `0.30` through `0.90`, step `0.05`.
- Helper text: “Higher hides weak detections; lower finds more objects but may
  include clutter.”
- A **Reset to default** action restores the startup-configured value.
- Changes are runtime-only; do not rewrite `mvp.toml` and do not persist to the
  browser.
- The control is available only in a non-recording setup view.
- Applying a new value atomically changes the detector confidence and starts a
  fresh prepare/enrolment pass. Readiness immediately clears.
- The identity/binding threshold is not operator-facing. It remains calibrated
  configuration.

The confidence control is an operational fallback, not an identity-safety
mechanism. A high-confidence foreign object must still be stopped by catalog
binding and the exact-catalog Start gate.

## 3. Target state flow

```text
app launch / New recording / Rescan / confidence change
                         |
                         v
                  reset + prepare
                         |
                         v
          gray resolving setup detections
                         |
                         v
            catalog-only binding freeze
               /                    \
              v                      v
    canonical ID + colour       gray Unknown
              \                      /
               +---- readiness -----+
                         |
             8/8 known, 0 extra,
               stable >= 2 seconds
                         |
                         v
                    Track enabled
                         |
                         v
             start recording WITHOUT reset
```

## 4. Model contract and linker changes

### M1 — Widen `InstrumentTracker` with a fixed catalog

Add:

```python
@property
def catalog(self) -> frozenset[int]:
    """Persistent specimen IDs available for catalog binding.

    Constant for the tracker lifetime and preserved across reset().
    """
```

Semantics:

- real tracker: keys of the successfully loaded persistent galleries;
- `ScenarioTracker`: `frozenset(range(1, n_instruments + 1))`;
- `FakeInstrumentTracker`: use its existing fake identity range, normalised to
  the same 1-based convention as the app fake;
- `catalog` is constant across `reset()`;
- `roster` remains empty until an enrolment freeze and then constant until the
  next reset;
- `roster` is the recognised subset of `catalog` in catalog-only mode.

Update `model/docs/tracker-interface.md` and every protocol implementation/test
double. Consumers must not infer the catalog from numeric ranges.

### M2 — Add catalog-only linker configuration

Extend `LinkerConfig` with:

```python
catalog_only_enrolment: bool = False
expected_catalog_size: int | None = None
```

Set these in `app/mvp.toml`:

```toml
[tracker.linker]
catalog_only_enrolment = true
expected_catalog_size = 8
```

Validation:

- `expected_catalog_size`, when provided, is positive;
- catalog-only mode requires persistent galleries;
- when `expected_catalog_size` is provided, the number of loaded galleries must
  equal it;
- in the real app's catalog-only mode, gallery-load failure is fatal and names
  the directory/count in the error.

### M3 — Catalog-only freeze behaviour

Keep the existing complete-dictionary scoring, bind threshold, margin,
one-to-one contest handling, and specimen-number identity.

When `catalog_only_enrolment` is true:

- create `_Identity` and `_raw_to_session` entries only for confidently bound
  raw tracks;
- set `roster` to the bound specimen IDs only;
- never call `_assign_session_ids` for unbound setup tracks;
- on the freeze frame, every unbound track emits its offset ID and
  `resolving=True`; on following frames it proceeds through the normal Pending
  decision and may settle Unknown;
- no ID above the highest specimen ID may enter the roster in this mode;
- contested losers are Unknown, never reassigned to their second choice;
- roster remains frozen after the freeze. In the final product there is no
  Rescan; a changed confidence (or application restart) starts fresh enrolment.

When catalog-only mode is false, preserve current session-only behaviour and its
tests.

### M4 — Binding calibration is separate work inside this ticket

Do not “fix” false binding by guessing a higher numeric threshold. The current
bind threshold is already separate from return matching. Validate it against:

- all eight demo instruments in expected setup poses;
- phone, pen, keys, hand, cable, packaging, scissors/pliers not in the set, and
  any foreign objects observed during recent testing;
- mixed scenes containing all eight instruments plus at least one foreign
  object;
- scenes with one known instrument absent and one foreign object present.

Record per-object best score, runner-up score, accepted/rejected outcome, and
roster result. A matcher false accept is not structurally detectable after the
matcher declares it known, so this dataset check is a release gate. Prefer
fail-closed threshold/margin or multi-view-consensus changes only when the data
supports them.

## 5. Capture and backend state changes

### B1 — Capture snapshots carry catalog and resolving state

Replace the loose `DetectionBox` tuple alias with an immutable named structure:

```python
@dataclass(frozen=True)
class DetectionBox:
    tracker_id: int
    xyxy: tuple[float, float, float, float]
    resolving: bool
```

Extend `Latest` with:

```python
catalog: frozenset[int]
```

Each capture tick samples `tracker.catalog`, `tracker.roster`, detection IDs,
and `data["resolving"]` from the same update result. Missing `resolving` remains
the existing safe default `False`. All stored scalar values are builtin Python
types.

Extend the capture callback so the pure session state receives same-tick:

```python
(t, present_ids, roster, catalog, resolving_ids)
```

### B2 — Session owns setup readiness as pure state

The `Session` remains clock-free and tracker-free. Store the latest setup
observation and expose an immutable result such as:

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
    blocking_reason: str | None
```

Definitions:

```text
recognised = present_ids intersect roster
non_roster = present_ids minus roster
resolving = resolving_ids intersect present_ids
unknown = non_roster minus resolving
```

`ready` implements D3 exactly. `blocking_reason` is one of:

- `recognising`
- `missing_instruments`
- `unknown_objects`
- `hold_steady`
- `null` when ready

Priority when multiple conditions apply:

1. `recognising`
2. `unknown_objects`
3. `missing_instruments`
4. `hold_steady`

Capture health is kept outside this pure result and is combined by the API.

Add `Session.prepare(t)` (or equivalently named method) that:

- is legal only from `setup` or `finished`;
- clears setup ID-set/readiness stability so the next observation begins at
  zero seconds;
- clears recording tracks when appropriate;
- preserves an existing finished report and finished phase;
- does not own or call the tracker.

### B3 — Historical prepare endpoint (removed from the final API)

```http
POST /recording/prepare
```

Response:

```json
{"prepared_at": "2026-07-16T12:34:56+02:00"}
```

Rules:

- allowed from `setup` and `finished`;
- `409` during `recording`;
- synchronously resets the tracker through `CaptureLoop`;
- resets setup readiness through `Session.prepare`;
- preserves the finished report;
- leaves the overlay in non-recording/setup state;
- timeout/reset failures use the existing `503` behaviour;
- concurrent prepare/start/settings mutations are serialised so two tracker
  resets cannot overlap.

### B4 — Start no longer resets and must fail closed

`POST /recording/start` must:

1. read phase, latest readiness, and capture health under the appropriate
   locks;
2. return `409` with a useful detail when setup is not ready;
3. call `session.start(...)` and enter recording only after the gate passes;
4. **not call `capture.reset_tracker()`**;
5. not alter tracker confidence, roster, catalog, IDs, or colours.

Suggested error details:

- `recognition still in progress`
- `remove unknown objects before starting`
- `all 8 instruments must be recognised before starting`
- `hold the tray steady before starting`
- existing capture-stalled detail for unhealthy capture.

The server check is mandatory even if the frontend button is disabled.

### B5 — Runtime confidence API

Add to every `GET /status` response:

```jsonc
"detector_control": {
  "confidence": 0.50,
  "default_confidence": 0.50,
  "minimum": 0.30,
  "maximum": 0.90,
  "step": 0.05
}
```

Add:

```http
PATCH /settings/detection-confidence
Content-Type: application/json

{"confidence": 0.60}
```

Return the updated `detector_control` object.

Rules:

- reject values outside `[0.30, 0.90]` with `422`;
- allowed only from `setup` and `finished`; `409` while recording;
- no-op value may return immediately without resetting;
- a changed value is applied with a tracker reset as one serialised capture
  command, then setup readiness is cleared;
- response is sent only after the new value/reset has been acknowledged;
- failure leaves the last confirmed confidence as source of truth and returns
  `503`;
- never write the TOML file.

Prefer a capture-thread command such as `set_confidence_and_reset(value)` over
mutating `tracker.confidence` from the HTTP thread while an inference update is
running.

### B6 — Setup API shape

Replace the old locally-derived Start gate with:

```jsonc
"setup": {
  "detected_count": 9,
  "expected_count": 8,
  "recognised_count": 8,
  "resolving_count": 0,
  "unknown_count": 1,
  "stable_for_s": 2.4,
  "ready": false,
  "blocking_reason": "unknown_objects",
  "detections": [
    {
      "tracker_id": 1,
      "state": "recognised",
      "label": "Instrument 1",
      "colour": "#4285f4",
      "thumbnail": "data:image/jpeg;base64,..."
    },
    {
      "tracker_id": 1042,
      "state": "unknown",
      "label": "Unknown",
      "colour": "#9ca3af",
      "thumbnail": "data:image/jpeg;base64,..."
    }
  ]
}
```

Detection `state` is `"recognising" | "recognised" | "unknown"`.

- `recognised` iff ID is in the same snapshot's roster;
- otherwise `recognising` iff row-aligned `resolving=True`;
- otherwise `unknown`;
- never expose a raw ID inside the label;
- `tracker_id` remains the emitted seam ID for React keys/API correlation; it is
  not rendered to the operator for non-recognised states;
- recognised colour comes from the fixed catalog mapping (D5);
- recognising/unknown colour is gray.

Update `app/docs/api-contract.md`, OpenAPI output, generated
`app/frontend/src/api/schema.d.ts`, fixtures, MSW handlers, and both backend and
frontend API tests.

## 6. Renderer and thumbnail changes

### R1 — One identity rendering policy in all phases

Remove the setup/finished branch that delegates to the old raw-track `render()`
appearance. `OverlayRenderer` applies the same roster-aware rules in all phases.
The recording flag may remain for other behaviour, but it must not select a
different identity label or palette.

Renderer inputs include both `catalog` and same-tick `roster`.

- recognised ID: `catalog_colour(catalog, id)`, `Instrument {id}`;
- non-roster resolving: gray + spinner, no label;
- non-roster settled: gray + `Unknown`.

The setup frame immediately before Track and the first recording frame after
Track must be visually identical for unchanged detections, apart from the
recording UI chrome.

### R2 — Stable catalogue colour function

Replace/retire `roster_colour(roster, id)` as the source of known colours with:

```python
catalog_colour(catalog: frozenset[int], specimen_id: int) -> str
```

Return gray for IDs outside the catalog. Tests assert:

- eight distinct known colours;
- stability across partial rosters;
- stability across setup -> recording;
- return after absence regains the same colour;
- no tests depend on the literal hex values.

### R3 — Setup thumbnails use the same identity metadata

`backend/thumbnails.py` must stop unconditionally constructing
`Instrument {tracker_id}`. Build label/state/colour from the snapshot's catalog,
roster, and resolving flag so the thumbnail list and video cannot disagree.

## 7. Frontend behaviour

> **Correction (2026-07-17).** An earlier draft of this section attached a dense
> eight-tile list with an overflow-scroll region to the SETUP screen and added a
> Rescan button + a prepare-on-New-recording flow. That was wrong on two counts:
> the setup screen keeps its radial hub-and-spoke **`DetectionConstellation`**
> (F2 below — restored verbatim from the prior commit at Bram's request), and the
> "complete list visible / overflow scrolls" requirement belongs to the
> **tracking** screen (F5). Rescan and the prepare-on-New-recording call are
> removed (F3). The backend-driven gate (F1), per-item identity in the backend
> data (D4/D5), and the confidence control (F4) are unchanged.

### F1 — Backend-driven Start gate

`LiveScreen` uses `setup.ready` directly. Delete the local
`detectedCount >= 1 && stableForS >= 2` eligibility calculation.

Map blockers to operator copy:

- `recognising`: “Recognising instruments…”
- `missing_instruments`: “Recognised X of 8 instruments”
- `unknown_objects`: “Remove N unknown object(s)”
- `hold_steady`: “Hold the tray steady…”
- capture unhealthy: existing camera-stalled message wins.

Show a compact summary such as `8/8 recognised` and, when applicable,
`1 unknown`.

### F2 — Setup keeps the radial DetectionConstellation

The setup screen keeps the **`DetectionConstellation`** from T06 — a dynamic
hub-and-spoke ring: the currently-detected instruments arrange around a floating
count pill and glide to re-spread as detections join or drop out. It is the
operator's one calm brand moment on the idle screen, restored **verbatim** from
the prior commit (Bram's call — the grid rework that replaced it is reverted).

- each tile shows the detection's crop, falling back to a representative icon
  when a crop is unavailable;
- the count pill reads the true detected count and a Ready / Stabilizing /
  Stalled state line;
- the ring renders at most the eight `recognised` catalog detections;
- resolving/unknown detections remain visible on video and affect the true count,
  blocker copy, and gate, but are not added to the constellation.

The safety gate does not rely on the ring — it lives in the backend
`setup.ready` verdict and the Track caption (`Remove N unknown …`).

The dense per-instrument list with an overflow-scroll region is a **tracking**
screen concern (F5), not a setup concern.

### F3 — Run-2 setup and the confidence reset hold

- First-run setup uses the already-fresh startup tracker.
- **New recording** opens the run-2 setup layout directly (a local flag; it does
  NOT call `POST /recording/prepare`). The gated Start owns the real transition.
  Run 2 reuses the approved tracker state. There is no public prepare endpoint
  and no Rescan button.
- **Back to report** remains available in the run-2 setup layout.
- A confidence change resets enrolment on the backend, clearing readiness. The
  frontend holds "Recognising" and keeps Track disabled across the one-poll
  stale-`ready` gap (D2), disarming on the mutation-fresh `/status`.

### F4 — Advanced confidence control

Add a collapsed **Advanced** section under the setup controls:

- label: `Detection confidence threshold`;
- `<input type="range">` or accessible shadcn equivalent;
- min/max/step/current/default come from `status.detector_control`;
- display the numeric value with two decimals;
- helper text from D6;
- `Reset to default` is disabled at the default value;
- UI changes may update the local displayed value immediately, but PATCH only
  after a 250 ms debounce or pointer/key commit;
- disable Track while a PATCH is pending;
- successful PATCH returns to recognising because the backend reset enrollment;
- failed PATCH restores the last server-confirmed value and shows an inline
  error;
- hide the control during recording and on the report view.

Do not add a slider for bind threshold or winner margin.

### F5 — Tracking list fits the full tray

On the recording (tracking) screen the per-instrument list shows the full
eight-instrument tray at once **without scrolling** on the demo screen: rows
share the panel height with a readable floor, so every instrument's ON/OFF TABLE
state is visible at a glance. An unusually short viewport degrades to scrolling
gracefully. This is where "the complete list is visible / overflow scrolls"
belongs — NOT the setup screen (F2).

## 8. Required TDD coverage

Write failing tests before production changes. Keep existing tests green unless
this specification explicitly changes their contract.

### Model tests

1. catalog is constant and survives reset;
2. catalog-only freeze admits a confidently bound known specimen;
3. below-threshold setup object emits offset ID and is absent from roster;
4. contested binding loser is absent from roster and never takes second choice;
5. eight known plus rejected foreign yields roster exactly equal to catalog and
   one non-roster emitted detection;
6. catalog-only mode never creates a session-only roster ID;
7. legacy non-catalog mode retains session-only behaviour;
8. missing/wrong-sized persistent catalog fails fast in configured safe mode;
9. freeze-frame IDs, roster, resolving flags, and catalog are same-tick
   consistent.

### Backend tests

1. capture snapshot stores catalog and row-aligned resolving state;
2. readiness false during resolving, with missing knowns, with any unknown, or
   before 2 s stability;
3. readiness true only for exact present catalog after stability;
4. server rejects Start for every blocking state even when called directly;
5. successful Start does not call tracker reset and preserves the approved
   tracker roster/catalog state;
6. failed confidence reset rolls the detector confidence back;
7. Start/confidence mutations cannot overlap;
8. setup labels and colours follow state and never expose raw IDs as labels;
9. known colour is identical before and after Start;
10. confidence PATCH validates range, is forbidden while recording, resets on
    change, does not reset on no-op, and leaves TOML untouched;
11. failed confidence reconfiguration preserves the last confirmed API value;
12. fake mode implements catalog and can exercise an extra setup object.

### Frontend tests

1. Track follows `setup.ready`, not a locally reconstructed condition;
2. each blocker renders the specified reason/count;
3. the setup constellation renders up to eight recognised catalog instruments
   (crop, or icon fallback), excludes resolving/unknown detections, and the count
   pill still shows the true detected count;
4. Track stays blocked with the unknown-count reason when the backend reports an
   unknown object (the gate is the backend `setup.ready` verdict, not the ring);
5. a setup recognised slot keeps its label/colour across the transition to the
   recording row (pressing Track never recolours it);
6. New recording opens run-2 setup directly (flag only; no prepare call);
7. the tracking list shows the full eight-instrument tray without scrolling;
8. confidence slider reads server metadata, debounces PATCH, resets to default,
   rolls back on failure, and is absent during recording/report;
9. Track stays disabled while a confidence update is pending;
10. generated API schema and fixtures include every new required field.

## 9. Acceptance scenarios

All must pass on the real camera, not only mocked unit tests.

### A — Normal tray

1. Present exactly the eight demo instruments during setup.
2. Setup resolves to `8/8 recognised`, `0 unknown`, then becomes ready after 2 s.
3. Record each visible label and colour.
4. Press Track.
5. Every physical instrument retains exactly the same label and colour; no gray
   flash and no renumbering occurs.

### B — Eight instruments plus foreign object

1. Add a phone/pen/keys/foreign tool during setup.
2. The foreign object is resolving then gray Unknown.
3. The roster contains only catalog IDs.
4. The eight known instruments ring the hub; the foreign object is not added to
   the constellation. The roster contains only catalog IDs, setup reports at
   least one unknown, and
   Track remains disabled with the `Remove N unknown …` caption.
5. Remove the foreign object; after the complete set is stable for 2 s, Track
   enables without a rescan when the eight known bindings were already valid.

### C — Missing known replaced by foreign object

1. Remove one demo instrument and add one foreign object so the raw detection
   count remains eight.
2. Setup must not become ready merely because the count is eight.
3. It reports missing recognition and/or an unknown object.

### D — Confidence adjustment

1. In setup, change `0.50` to another allowed value.
2. Existing readiness clears immediately; tracker/enrolment restarts.
3. No stale known label/colour is reused from the prior scan.
4. Reset to default returns to the startup value.
5. During recording the control is absent and direct PATCH returns `409`.

### E — Second recording

1. Finish a recording and open the report.
2. Click New recording: setup is shown without resetting the tracker.
3. Back to report still returns the previous report.
4. Complete setup and press Track: the old report is then discarded and the new
   run starts while preserving the approved roster.

## 10. Implementation sequence and ownership seams

Implement in this order so intermediate contracts are reviewable:

1. **Model:** catalog property, catalog-only freeze, validation, model tests,
   tracker-interface docs.
2. **Backend state:** same-tick snapshot fields, pure readiness, prepare/start
   semantics, mutation serialisation, backend tests.
3. **Rendering/API:** catalog colours, setup detection states, updated API
   contract/OpenAPI schema/fakes.
4. **Frontend:** backend-driven gate, setup states, Rescan/New recording flow,
   advanced confidence control, frontend tests.
5. **Real-data validation:** run the acceptance scenes and record bind scores and
   outcomes before changing any matcher thresholds.

Likely files (not an exclusive list):

- `model/src/orc_model/pipelines/config.py`
- `model/src/orc_model/pipelines/tracking.py`
- `model/src/orc_model/pipelines/session_linker.py`
- `model/tests/pipelines/test_config.py`
- `model/tests/pipelines/test_tracking.py`
- `model/tests/pipelines/test_session_linker.py`
- `model/docs/tracker-interface.md`
- `app/mvp.toml`
- `app/backend/backend/capture.py`
- `app/backend/backend/session.py`
- `app/backend/backend/main.py`
- `app/backend/backend/render.py`
- `app/backend/backend/thumbnails.py`
- `app/backend/backend/fakes.py`
- `app/backend/tests/`
- `app/docs/api-contract.md`
- `app/frontend/src/api/`
- `app/frontend/src/screens/LiveScreen.tsx`
- `app/frontend/src/components/DetectionConstellation.tsx`
- corresponding frontend tests/fixtures/handlers

## 11. Explicit non-goals

- training an eight-class detector;
- exposing bind threshold, margin, tracker association, or debounce tuning in
  the normal UI;
- persisting runtime confidence changes;
- allowing recording to start with unknown setup objects;
- retroactively correcting report identities;
- changing post-start Unknown recovery except where the catalog-only roster
  contract requires test updates.

## 12. Definition of done

- all model, backend, and frontend test suites pass;
- generated frontend API types match FastAPI OpenAPI;
- old contract text saying Start accepts any stable count >= 1 is removed;
- pressing Track contains no tracker reset call;
- catalog-only mode has no session-only roster IDs;
- server and frontend both enforce exact-catalog readiness;
- setup and recording use one canonical label/colour mapping;
- confidence control is setup-only, runtime-only, and reset/re-enrolment-safe;
- real-camera acceptance scenarios A–E are recorded as passing evidence.
