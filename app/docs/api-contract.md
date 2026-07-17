# API contract — `app/backend` ↔ `app/frontend`

The single thing the **frontend** needs from the backend: an HTTP API to see the
live camera, start/stop a recording, watch per-instrument state while recording,
and fetch the Usage/Completeness report after Stop. This doc is that contract.

It sits one level above [`model/docs/tracker-interface.md`](../../model/docs/tracker-interface.md):
the backend consumes `InstrumentTracker` (real or `FakeInstrumentTracker`), the
frontend consumes this API. Neither side ever reaches through its seam.

> **Decisions baked in (2026-07-07):** Bram owns backend + frontend; Constantijn
> owns the model behind `InstrumentTracker`. Recording screen shows a live
> per-instrument list. Report is **in-memory only** — one finished session,
> cleared only on a successful next Start.
>
> **Superseded (2026-07-16 / T11):** the old "human judgment, any stable count
> ≥ 1" Start gate is replaced by a **server-enforced exact-catalog gate** (D3);
> setup and recording now share **one identity/colour system** (D4/D5). The
> **Start preserves the exact approved tracker roster** and performs no reset;
> resetting after the readiness check would empty that roster and re-enrol during
> recording. There is no separate public prepare step or Rescan button. A changed
> detection confidence is the one setup action that resets enrolment. This
> document reflects the final T11 contract;
> where a sentence below still described the old behaviour it has been updated in
> place.

---

## Transport & shape

- **Polling, not WebSocket.** The frontend polls `GET /status` at ~2 Hz. At demo
  scale this is indistinguishable from push and removes a whole failure class.
- **MJPEG for video.** `GET /stream` is a `multipart/x-mixed-replace` stream the
  frontend renders with a plain `<img src="/stream">` — same proven mechanism as
  the data-collection dashboard.
- **Backend computes, frontend renders.** Usage windows, completeness,
  debouncing — all backend. The frontend never derives analytics from raw data.
- **Preview cutouts are backend-derived.** The model returns frame-aligned
  boxes/masks, not encoded UI assets. The backend derives small transparent PNG
  cutouts from its camera frame and returns them inline with `/status`; the
  frontend does not crop the MJPEG stream or call the model directly.
- **Times:** absolute moments are ISO-8601 strings; everything inside a session
  is **seconds relative to recording start** (floats) — what a timeline needs.
- **Errors:** wrong-phase actions → `409` with `{"detail": "..."}`. Everything
  else standard FastAPI validation errors.

## Phases

One global state machine, owned by the backend:

```
      setup ──POST /recording/start──▶ recording ──POST /recording/stop──▶ finished
        ▲    (server gate: exact catalog;  ▲                                │
        │     preserves approved roster)  └────POST /recording/start───────┘
        │                                      (preserves roster; discards the
        │                                       previous report)
 (first launch =
  fresh tracker)
```

Only `start` (→ `recording`) and `stop` (→ `finished`) change the phase.
**New recording** is a frontend-only step: it routes back to the setup layout
while the backend still holds the finished report (so **Back to report** works),
and the gated **Start** preserves that approved setup state.

- `setup` — camera live, tracker running for the overlay, nothing recorded. The
  first launch boots with a fresh tracker.
- `recording` — Start atomically re-checks readiness, preserves the approved
  tracker roster/catalog state, and begins accumulating per-instrument state.
- `finished` — report available AND the camera/tracker keep live-observing the
  table (overlay still drawn, readiness still updating) so the Start gate works
  here just like in `setup`. **New recording** routes to the setup layout while
  preserving the previous report (so **Back to report** still works); that report
  is discarded only after the next successful Start. See §/status: `finished`
  carries the same `setup` block as `setup`.

---

## Endpoints

### `GET /status` — poll ~2 Hz

```jsonc
{
  "phase": "setup",                  // "setup" | "recording" | "finished"
  "capture_health": "ok",            // "ok" | "stalled"  (frozen camera is visible)
  "model_version": "scenario-0.1",

  // present when phase == "setup" OR "finished": the exact-catalog Start gate
  // state (in "finished" the table is still being observed for the next run)
  "setup": {
    "detected_count": 9,             // ids detected this snapshot (everything on the table)
    "expected_count": 8,             // size of the loaded catalog (the demo's 8 specimens)
    "recognised_count": 8,           // detected ids bound to a catalog specimen (in the roster)
    "resolving_count": 0,            // detected ids still being recognised
    "unknown_count": 1,              // detected ids settled as not-a-specimen
    "stable_for_s": 2.4,             // how long the complete emitted id-set has been unchanged
    "ready": false,                  // the server's Start verdict (D3); folds in capture health
    "blocking_reason": "unknown_objects",  // pure-readiness blocker; null when the readiness
                                     // gate passes (may be null while ready is false — see below)
    "detections": [                  // always present; may be []; sorted by tracker_id
      {
        "tracker_id": 1,
        "state": "recognised",       // "recognising" | "recognised" | "unknown"
        "label": "Instrument 1",     // "Instrument N" recognised | "Unknown" | "" recognising
        "colour": "#4285f4",         // catalog colour recognised, gray otherwise
        "thumbnail": "data:image/png;base64,…"  // string | null
      },
      {
        "tracker_id": 1042,
        "state": "unknown",
        "label": "Unknown",          // never the raw tracker id
        "colour": "#9ca3af",         // gray
        "thumbnail": "data:image/png;base64,…"
      }
    ]
  },

  // phase == "recording": drives the lean live instrument panel.
  // Usage/completeness analytics are available only after Stop via /report.
  "recording": {
    "started_at": "2026-07-20T10:31:04+02:00",
    "elapsed_s": 74.3,
    "instruments": [
      {
        "tracker_id": 3,
        "label": "Instrument 3",     // single class today; label comes from backend
        "colour": "#4285f4",         // required; the hex this instrument's mask
                                     // is drawn with on /stream (see below)
        "on_table": false,
        "thumbnail": "data:image/png;base64,…"  // string | null — a live cutout this
                                     // frame, null when off-table/not detected (the
                                     // app keeps showing the last-seen cutout)
      }
    ]
  },

  // ALWAYS present (T11/B5): runtime detection-confidence control metadata,
  // drives the Advanced setup control. See PATCH /settings/detection-confidence.
  "detector_control": {
    "confidence": 0.50,              // current runtime value
    "default_confidence": 0.50,      // the startup value from mvp.toml
    "minimum": 0.30,
    "maximum": 0.90,
    "step": 0.05
  }
}
```

`setup` block is non-null when `phase == "setup" | "finished"`; `recording`
block is non-null only when `phase == "recording"` (so the two are never both
non-null, but in `finished` the `setup` block IS present — the report is fetched
separately via `GET /report`). `detector_control` is **always** present.

The recording payload intentionally exposes only elapsed time, instrument
identity/presentation, and current on-table state. Off-table windows and final
Completeness are report-only so the demo's analytics payoff appears after Stop.

#### `setup` — the exact-catalog Start gate (T11/D3)

The five counts partition the detected id-set: `recognised` (bound to a catalog
specimen, i.e. in the frozen roster) + `resolving` (still being recognised) +
`unknown` (settled as not-a-specimen) = everything detected; `detected_count` is
the whole set and `expected_count` is the loaded catalog size (8 for the demo).

**`ready` is the server's verdict — the frontend renders it, never recomputes
it.** `POST /recording/start` re-checks the same backend state, so the button
being disabled is never the enforcement. `ready` is true only when: capture health
is `ok`; the frozen roster equals the expected catalog (all 8 recognised); every
catalog identity is currently present; no unknown or resolving object is on the
table; and the complete id-set has been unchanged for ≥ 2.0 s.

`blocking_reason` is one of `"recognising" | "missing_instruments" |
"unknown_objects" | "hold_steady" | null`, at this priority when several apply:
**recognising → unknown_objects → missing_instruments → hold_steady**. It is
`null` when the **pure readiness** conditions pass — but that is not the same as
`ready`: capture health is folded into `ready` only, NOT into `blocking_reason`.
So a stalled camera with an otherwise-perfect tray gives `ready: false` **and**
`blocking_reason: null`; the frontend shows the separate `capture_health` banner
in that case (it takes precedence over the readiness copy).

#### `setup.detections` — per-item identity (T11/B6)

Each detection carries its identity `state`/`label`/`colour` from **one policy
shared with the video overlay**, so a tile and the mask for the same object can
never disagree:

- `recognised` (id in this snapshot's roster) → `label` `"Instrument N"`, `colour`
  the catalog colour;
- `recognising` (not in the roster, still resolving) → `label` `""` (the frontend
  shows a spinner), `colour` gray;
- `unknown` (not in the roster, settled) → `label` `"Unknown"`, `colour` gray.

`tracker_id` is the seam id for React keys / API correlation; it is **never**
rendered to the operator for a non-recognised state (no raw id ever appears in a
label — D4). `detections` is sorted by `tracker_id` and comes from one capture
snapshot; `thumbnail` is normally a small aspect-preserving transparent PNG
cutout data URI. If a producer has no usable mask, the backend defensively falls
back to the previous JPEG crop; `null` means that neither preview could be
validated/encoded (one bad item must not fail the response).
The API returns **all** current detections in order. The setup constellation is a
catalog view: it renders only `recognised` detections, capped at the catalog's
eight instruments around the centre. `recognising` and `unknown` objects remain
visible on the video and still affect the count, blocking reason, and Start gate,
but are deliberately not added to the constellation. `detected_count` (session
state) and `detections.length` (latest snapshot) may differ by a frame; tolerate
that and fall back to a representative icon for a recognised detection whose
thumbnail is missing/null.

#### Colours derive from the fixed catalog (T11/D5)

**`recording.instruments[].colour`** and **`setup.detections[].colour`** are the
SAME value the overlay draws that object's mask with on `/stream`. Render the
swatch straight from this field; never re-derive a colour from `tracker_id`.

- **Pure function of `(catalog, specimen_id)`** — palette slot at the id's index
  in `sorted(catalog)`. Because the basis is the *fixed catalog*, not the current
  roster, a partial roster gaining a lower id later **cannot** shift an
  already-recognised instrument's colour. Distinct within the catalog, and stable
  before/during/after recording — a returned instrument regains its exact colour.
- **The hexes are a TUNABLE.** Assert *distinctness* and *stability*; never assert
  specific hex values. The palette will be re-tuned against branding + real camera.
- **Unknown/foreign → gray `"#9ca3af"`.** In `recording`, unknown/foreign ids are
  absent from `instruments` entirely (the roster filter in `Session`, D8a) — no
  entry, no Usage, no Completeness. In `setup` they appear as gray
  `recognising`/`unknown` detections in the payload and on the video (the operator
  must clear them to pass the gate); they do not become constellation tiles.
- **One transient:** before the first published frame there is no snapshot yet and
  a recording colour resolves to the gray placeholder. Harmless and
  self-correcting; don't special-case it.

### `GET /stream` — MJPEG

Live frames with the tracker overlay drawn (boxes + masks, live count burned in
or not — overlay content is mostly backend's call and NOT part of this contract).
Since T11/R1 the overlay uses **one identity policy in every phase** (setup,
recording, finished): a recognised instrument's mask is drawn in exactly the
`colour` `/status` reports for it and labelled `"Instrument {id}"`; a not-in-roster
object is gray, with a resolving spinner before it settles to `"Unknown"`. The
setup frame immediately before Track and the first recording frame after Track are
therefore visually identical for unchanged detections (apart from the recording UI
chrome) — pressing Track never recolours or relabels an already-recognised object.

### `POST /recording/start` → `200 {"started_at": ...}`

Allowed from `setup` or `finished`. It is a **server-enforced fail-closed gate**
(D3): it re-checks the same state behind `setup.ready` and returns
`409 {"detail": ...}` for every not-ready condition even if the frontend button
was enabled — mapped details: `"recognition still in progress"` (recognising),
`"remove unknown objects before starting"` (unknown_objects), `"all 8 instruments
must be recognised before starting"` (missing_instruments), `"hold the tray steady
before starting"` (hold_steady), `"capture stalled"` (unhealthy capture). `409` if
already `recording`. On success it changes only the session phase: the exact
roster/catalog state that passed the gate is preserved into recording, the
previous report is discarded, and recording begins. Start does not reset the
tracker and therefore has no tracker-reset `503` path.

### `POST /recording/stop` → `200` (the same body as `GET /report`)

Allowed from `recording` only, else `409`. Freezes the session, computes the
report, phase → `finished`. The response body is **byte-identical in shape** to
`GET /report` (same serializer; a later `GET /report` returns the same payload
until the next Start) — frontend parses one `Report` type for both.

### `GET /report`

`409` unless `phase == "finished"`.

```jsonc
{
  "started_at": "2026-07-20T10:31:04+02:00",
  "stopped_at": "2026-07-20T10:36:40+02:00",
  "duration_s": 336.0,
  "model_version": "scenario-0.1",
  "instruments": [
    {
      "tracker_id": 3,
      "label": "Instrument 3",
      "completeness": "present",     // "present" | "missing"
      "usage": [                      // off-table windows, chronological
        { "off_s": 61.0, "on_s": 84.5 },
        { "off_s": 190.2, "on_s": null }   // null = never came back → "missing"
      ]
    }
  ]
}
```

Invariants: every instrument ever tracked in the session appears exactly once;
`completeness == "missing"` ⟺ the last usage window has `on_s: null`; windows
don't overlap and are sorted; an instrument never picked up has `usage: []`.

### `PATCH /settings/detection-confidence` → `200` (a `detector_control` object)  (T11/B5/D6)

Body `{"confidence": 0.60}`. Sets the **runtime** detection-confidence threshold —
an advanced setup control, an operational fallback, NOT an identity-safety
mechanism. Returns the updated `detector_control` object (same shape as in
`/status`).

- `422` for a value outside `[0.30, 0.90]`.
- Allowed only from `setup` and `finished`; `409` during `recording`.
- A **no-op** value equal to the current one returns immediately without resetting.
- A **changed** value is applied together with a tracker reset as one serialised
  capture command, then setup readiness clears (enrolment restarts) — so the
  frontend returns to `recognising`. The response is sent only after the new value
  and reset are acknowledged.
- If reset fails after the new confidence was assigned, the tracker rolls back to
  the previous confidence; the last confirmed API value stays the source of truth
  and the call returns `503`.
- Runtime-only: `mvp.toml` is **never** rewritten, and the value is not persisted
  in the browser. **Reset to default** re-sends `default_confidence`.

```jsonc
{ "confidence": 0.60, "default_confidence": 0.50, "minimum": 0.30, "maximum": 0.90, "step": 0.05 }
```

---

## Semantics the backend owns (not exposed, but pinned here)

- **Off-table debounce.** An instrument is "off table" only after its
  `tracker_id` has been absent for `> OFF_DEBOUNCE` (~1.5 s), and "back" after
  present for `> ON_DEBOUNCE` (~1.0 s). Otherwise detector flicker becomes fake
  pickups. Config values, not API.
- **Identity = `tracker_id`.** The report trusts the tracker's ids completely.
- **The roster filters recording; setup reports everything, classified.** The
  model freezes a roster of enrolled catalog identities during enrolment and
  exposes it across the model seam; the backend admits only those ids into
  recording state, which keeps foreign objects out of `instruments`, Usage and
  Completeness. The `setup` block is **not** filtered — it reports *every* detected
  id, but classified against the roster/catalog (`recognised` / `recognising` /
  `unknown`), because the T11 Start gate is a server check on the whole table
  (exact catalog present, zero unknowns, stable ≥ 2 s), not an unstructured count.
  Config and internals, not API; see DESIGN D8a and T11/D3.
- **Detection previews are presentation data, not model output.** The backend
  owns the original frame passed to `InstrumentTracker.update()` and derives
  `/status` cutouts from that frame plus the returned `xyxy`, `mask`, and
  `tracker_id`. Cutout geometry, encoding and HTTP transport do not extend the
  model contract; see
  [`tracker-interface.md` §Consumer-generated crops and previews](../../model/docs/tracker-interface.md#consumer-generated-crops-and-previews).
- **Completeness = debounced state at Stop.** `"missing"` ⟺ the instrument is
  in a *confirmed* off-table window at Stop (absence already past
  `OFF_DEBOUNCE`); otherwise `"present"`. A sub-debounce detector blink at the
  Stop instant does NOT mark an on-table instrument missing and creates no
  phantom window. (Keeps the `missing ⟺ last usage window on_s == null`
  invariant above.) The value is `"missing"`, not `"lost"` — glossary-canonical
  (2026-07-08): the camera observes the *symptom* (not on the table); whether
  the instrument is misplaced or genuinely lost is unknowable from here.
- **Re-identification (PRD acceptance criterion 3) — mechanism agreed
  2026-07-08:** an instrument returning after absence gets a fresh **Deep
  OC-SORT** raw id (the tracker was ByteTrack when this was written; the shipped
  composition is RF-DETR → Deep OC-SORT → `SessionLinker`) which is then
  **linked** back to its original session identity by the SRC matcher, entirely
  behind `InstrumentTracker`; once linked, `update()` re-emits the original id
  (≤ 1 s after return). The backend suppresses provisional pre-link ids with a 1 s
  **entry debounce** (T02), so linking never leaks into this API. Residual
  risk: a link that takes > 1 s produces a phantom never-returned instrument
  on the report. See `model/docs/tracker-interface.md` §tracker_id across
  absence.

---

## How the frontend is built before the model exists

Fake at the **deepest stable seam**, not at HTTP. The backend takes a
`--fake` flag and runs the very same code against **`ScenarioTracker`** (which
implements `InstrumentTracker`, lives in `app/backend/` per D10, and scripts
real pickups/returns/losses) plus `FakeCaptureSource` (synthetic frames) when no
camera is attached. The frontend therefore always talks to the one real API
implementation — there is no mock server to drift out of sync. (Note:
`FakeInstrumentTracker` in `model/` only *drifts* — it can't demo Usage, so it
is NOT the app's fake; see D10.)

The fake's pickup timeline is anchored by injected `begin_recording` /
`end_recording` lifecycle callbacks. Those callbacks change only the synthetic
scene clock; they do not call `tracker.reset()`. This keeps `--fake` aligned with
the production rule that Start preserves the approved roster, and returns the
synthetic tray to setup choreography after Stop so a second run can become ready.

To make the fake useful for THIS app, extend it with a **scripted scenario**
(the frozen `DEFAULT_SCENARIO` in T01: instrument 1 leaves at t=20 s and returns
at t=35 s; instrument 3 leaves at t=50 s and never returns) so the live panel,
timeline, and "missing" badge can be developed against realistic data.
`FakeInstrumentTracker` today only drifts and flickers — it never simulates a
pickup.

Type safety across the seam: FastAPI serves `openapi.json` for free — generate
the frontend's TypeScript types from it (`openapi-typescript`) instead of
hand-writing interfaces, so contract drift becomes a compile error.
