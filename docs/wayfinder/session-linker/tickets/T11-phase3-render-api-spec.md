# SPEC-C — T11 Phase 3: rendering + API shape

> **⚠️ SUPERSEDED IN PART (2026-07-17):** the `POST /recording/prepare` endpoint this
> spec documents was later removed (no Rescan; New recording never wired it).
> Recording Start remains a no-reset transition that preserves the approved
> roster. The rendering work (unified overlay,
> `catalog_colour`, setup thumbnails, per-item `state`/`label`/`colour`) is unchanged.
> Canonical contract: `app/docs/api-contract.md`.

Authoritative spec for the **rendering/API layer** of ticket T11 (sections **R1,
R2, R3** of §6 and **B6** of §5 in
`docs/wayfinder/session-linker/tickets/T11-safe-setup-enrolment.md`, plus backend
tests 8–9 and the test migration R1/R2 force). Phases 1 (model catalog seam +
catalog-only freeze) and 2 (backend state: same-tick snapshot fields, pure
`SetupReadiness`, prepare/start/confidence, mutation serialisation) are DONE and
signed off. `tracker.catalog`/`tracker.roster`/`data["resolving"]` cross the seam;
`Latest` already carries `catalog`; the capture snapshot and `on_frame` already
carry same-tick catalog + row-aligned resolving.

Repo root: `/Users/bramdecraeke/Documents/Projects/Github/P_ss/MVP-OKcamera`.
Backend source: `app/backend/backend/`. Tests: `app/backend/tests/`.
Interpreter/venv: `app/backend/.venv/bin/python`. Run from `app/backend/`:
`.venv/bin/python -m pytest tests/ -q`. Model suite must stay green too
(`cd model && .venv/bin/python -m pytest tests/ -q`) — Phase 3 does not touch
`model/`.

## What Phase 3 is (and is NOT)

**IN (this layer):**
- **R1** — one identity-rendering policy in ALL phases: `OverlayRenderer` stops
  branching setup/finished onto the old raw-track appearance. Renderer inputs
  widen to carry same-tick `catalog`.
- **R2** — `catalog_colour(catalog, specimen_id)` replaces `roster_colour(roster,
  id)` as the source of known colours (D5 — colours derive from the fixed
  catalog, stable across partial rosters). Recording `/status` colours switch to
  it too.
- **R3** — setup thumbnails build `state`/`label`/`colour` from the snapshot's
  catalog/roster/resolving, so the tile list and the video can never disagree.
- **B6** — `GET /status` `setup.detections[]` gains per-item `state`, `colour`
  (label logic changes); `DetectionModel` widens.
- **Backend tests 8–9**, plus migrating every test whose contract R1/R2 change.
- **`app/docs/api-contract.md`** — the human contract doc: rewrite the `/status`
  setup block to the readiness + per-item shape, delete the old "any stable count
  ≥ 1" Start-gate wording, document `detector_control` + `/recording/prepare` +
  `PATCH /settings/detection-confidence` (all shipped in Phase 2 but never
  documented), and the catalog-colour rule.

**OUT (→ Phase 4, do NOT do here):** regenerating
`app/frontend/src/api/schema.d.ts` (`npm run gen:api`), and all
`app/frontend/**` work — `LiveScreen`/`DetectionConstellation` behaviour (F1–F4),
`src/test/fixtures.ts`, `src/dev/devHandlers.ts` (MSW), and frontend tests.
**Rationale:** `schema.d.ts` is generated from the backend OpenAPI and the
frontend currently compiles GREEN against the *stale* pre-Phase-2 schema via its
fixtures/MSW. Regenerating it forces its consumers (components + fixtures + MSW +
frontend tests) to change in lockstep, which IS Phase 4 (ticket §10 step 4). Doing
it here would leave the frontend red mid-layer. Phase 4 begins by regenerating
`schema.d.ts` against the now-final backend contract. The **8-hex palette is
duplicated in `fixtures.ts` + `devHandlers.ts`** and its *indexing basis* changes
(roster→catalog) — Phase 4 keeps those mirrors in sync.

**Also OUT (charter):** `backend/debug.py` freeze-panel guard (carried from Phase
1 review) — do it when `debug.py` is next touched, not required by this layer.

---

## Design decisions pinned for this layer

- **DC-a — one identity policy, one function.** Add ONE pure classifier in
  `render.py` and call it from BOTH the overlay renderer and the thumbnail
  builder, so the video mask and the setup tile for the same detection can never
  drift (R1/R3's shared intent):
  ```python
  def classify_detection(
      tracker_id: int, roster: frozenset[int], catalog: frozenset[int], resolving: bool
  ) -> tuple[str, str, str]:
      """(state, label, colour) under the single identity policy (D4/D5).
      recognised iff in roster; else recognising iff resolving; else unknown.
      Never exposes a raw id as a label."""
      if tracker_id in roster:
          return "recognised", f"Instrument {tracker_id}", catalog_colour(catalog, tracker_id)
      if resolving:
          return "recognising", "", UNKNOWN_COLOUR
      return "unknown", _UNKNOWN_LABEL, UNKNOWN_COLOUR
  ```
- **DC-b — recognising label is `""`.** The recognising tile shows a spinner, not
  a name (F2); the video draws a spinner, not a label. The API `label` for a
  recognising detection is the empty string — never a raw id (B6: "never expose a
  raw ID inside the label"). `recognised → "Instrument {id}"`, `unknown →
  "Unknown"` (unchanged from the overlay's existing `_UNKNOWN_LABEL`).
- **DC-c — recording colour switches to catalog.** `main.py`'s
  `InstrumentStatusModel.colour` becomes `catalog_colour(snapshot.catalog, id)`
  (was `roster_colour(snapshot.roster, id)`). Same snapshot → the panel swatch and
  the mask are still the same hex by construction. For the demo catalog `{1..8}`
  with a full roster this is byte-identical to before; it only differs (correctly,
  per D5) for a partial roster. `roster_colour` is **retired** (removed).
- **DC-d — renderer signature widens.** `OverlayRenderer.__call__(frame, dets,
  roster, catalog, t)`; `capture.py:_capture_tick` passes the same-tick `catalog`
  it already samples. `OnFrame` is unaffected (Phase 2 already widened it).
- **DC-e — `set_recording`/`_recording` retained as a no-op lifecycle hook.** R1
  permits the recording flag to remain; after unification it no longer selects a
  label or palette. Keeping it (rather than deleting) avoids rippling into
  `main.py`'s start/prepare/stop handlers and the `test_api.py` `FakeRenderer` /
  B-V6 wiring. Documented in the class docstring that it no longer gates identity.
- **DC-f — `render()` free function deleted.** Its only caller was the
  now-removed not-recording branch. Nothing else imports it.
- **DC-g — palette renamed `ROSTER_PALETTE` → `CATALOG_PALETTE`.** Same 8 hexes;
  the name now reflects the catalog indexing basis (D5). `UNKNOWN_COLOUR`
  unchanged. Hexes stay a TUNABLE — tests assert distinct/stable/well-formed,
  never enumerate.

---

## R1 + R2 — `render.py`

- Rename `ROSTER_PALETTE` → `CATALOG_PALETTE` (values unchanged).
- Add `catalog_colour(catalog, specimen_id)`: gray (`UNKNOWN_COLOUR`) if
  `specimen_id not in catalog`, else `CATALOG_PALETTE[sorted(catalog).index(
  specimen_id) % len(CATALOG_PALETTE)]`. Pure; stable because `catalog` is
  constant for the tracker's life.
- **Remove** `roster_colour`.
- Add `classify_detection(...)` (DC-a).
- **Delete** the `render(frame, dets)` free function (DC-f).
- `OverlayRenderer.__call__(frame, dets, roster, catalog, t)`:
  - empty/no-id early-return unchanged;
  - **delete** the `if not self._recording: return render(frame, dets)` branch —
    setup and recording now render identically;
  - for each row, call `classify_detection(id, roster, catalog, resolving_row)`;
    colour = returned colour; draw the returned `label` when `state !=
    "recognising"`; when `state == "recognising"` draw the spinner and no label.
    (Same mask/box/label/spinner annotator machinery as today's recording branch.)
- `set_recording`/`_recording` kept (DC-e); update docstrings so they no longer
  claim to switch the overlay appearance.

## R3 — `thumbnails.py`

- `Detection` dataclass gains `state: str` and `colour: str` (after `tracker_id`,
  before `thumbnail`): `tracker_id, state, label, colour, thumbnail`.
- `build_detections(frame_bgr, boxes, roster, catalog, *, size=..., pad=...)`:
  widen the signature to take same-tick `roster` and `catalog`. For each box call
  `render.classify_detection(box.tracker_id, roster, catalog, box.resolving)` →
  `(state, label, colour)`; keep the sort-by-`tracker_id` and the isolated-crop
  (`thumbnail=None` on failure) behaviour. Import `classify_detection` from
  `backend.render` (no cycle: render imports nothing from backend).

## B6 — `main.py`

- `DetectionModel` gains `state` + `colour`:
  ```python
  class DetectionModel(BaseModel):
      tracker_id: int
      state: Literal["recognising", "recognised", "unknown"]
      label: str
      colour: str
      thumbnail: str | None
  ```
- `_snapshot_detections()` passes the snapshot's `roster`/`catalog` to
  `build_detections(snapshot.frame_bgr, snapshot.detections, snapshot.roster,
  snapshot.catalog)`. It is shared by the setup preview and the recording live
  crops; the recording crops dict only reads `tracker_id`→`thumbnail`, so the
  extra fields are harmless there.
- `_setup_detections()` unchanged (still `model_validate(..., from_attributes=
  True)` — new fields map by name).
- Recording path: `colour=catalog_colour(catalog, status.tracker_id)` where
  `catalog = snapshot.catalog if snapshot is not None else frozenset()`. Import
  `catalog_colour` (drop the `roster_colour` import). The transient before the
  freeze (empty roster) still resolves every colour to gray — unchanged wording in
  the contract.

---

## Tests (orchestrator writes inline — no blind test-writer this layer)

### New — backend tests 8 & 9 (T11 §8)
- **Test 8 — setup labels/colours follow state, never a raw id** (`test_api.py`
  and/or `test_thumbnails.py`): a recognised id → `state "recognised"`, label
  `Instrument {id}`, colour `catalog_colour(catalog, id)`; a resolving non-roster
  id → `state "recognising"`, label `""` (assert the raw id string is NOT in the
  label), gray colour; a settled non-roster id → `state "unknown"`, label
  `"Unknown"`, gray. Assert distinctness/gray via the palette, never a literal hex.
- **Test 9 — known colour identical before and after Start** (`test_api.py` or
  `test_render.py`): the colour a recognised id gets in a setup detection equals
  the colour the recording overlay/`/status` gives it (both `catalog_colour(
  catalog, id)`), for a partial-and-full roster.

### Migrations forced by R1/R2 (their contracts change — §8 allows it)
- **`test_render.py`:**
  - `_draw` / `_recording_renderer` helpers thread the new `catalog` arg;
    `renderer(frame, dets, roster, catalog, t)`.
  - **`TestBV1NotRecordingIsTodaysOverlay`** — DELETE/REWRITE: the
    "setup labels every id Instrument N / off-roster id is not gray / ignores
    resolving" assertions INVERT under R1 (setup now uses the roster-aware policy:
    off-roster is gray resolving/Unknown). Keep the AC10 in-place/shape/dtype
    checks, re-aimed at the unified renderer.
  - **`TestBV2*`** — `roster_colour`→`catalog_colour`, `ROSTER_PALETTE`→
    `CATALOG_PALETTE`; "follows the roster index" tests become "follows the
    CATALOG index"; the pure-palette suite tests `catalog_colour`.
  - **`TestBV3*`/`TestBV4*`/resolving-flag suite** — behaviour preserved (gray /
    spinner / Unknown) but now phase-independent; thread `catalog` through
    `_draw`. `_recording_renderer()` still fine (flag is a no-op).
- **`test_capture.py`:** the render-fn spy at ~L1127 (`spy_render(frame, dets,
  roster, t)`) widens to `(frame, dets, roster, catalog, t)`; any other test
  `render_fn` double likewise.
- **`test_api.py`:** import `catalog_colour`/`CATALOG_PALETTE` (drop
  `roster_colour`/`ROSTER_PALETTE`); B-A1 recording-colour assertions →
  `catalog_colour(catalog, id)` and set the fixture tracker's `catalog`; the
  "unknown colour not in palette" check → `CATALOG_PALETTE`.
- **`test_thumbnails.py`:** `build_detections` callers pass `roster`/`catalog`;
  assert the new `state`/`label`/`colour` per the policy. The existing `_box()`
  helper stays.

Keep every other existing test green.

## Definition of done
- One identity policy in all phases; `catalog_colour` is the sole known-colour
  source; `roster_colour` and the `render()` free function are gone.
- `GET /status` `setup.detections[]` carries `state` + `colour`; recognising/
  unknown never expose a raw id as a label; recognised colour == recording colour.
- `api-contract.md` matches the shipped Phase 2 + Phase 3 contract (readiness
  fields, per-item detections, `detector_control`, prepare/confidence endpoints);
  the old "stable count ≥ 1" Start gate wording is removed.
- `app/backend` suite green; `model/` suite still green.
- Frontend UNTOUCHED and still green against its stale schema (Phase 4 regenerates
  it).
