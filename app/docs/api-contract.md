# API contract — `app/backend` ↔ `app/frontend`

The single thing the **frontend** needs from the backend: an HTTP API to see the
live camera, start/stop a recording, watch per-instrument state while recording,
and fetch the Usage/Completeness report after Stop. This doc is that contract.

It sits one level above [`model/docs/tracker-interface.md`](../../model/docs/tracker-interface.md):
the backend consumes `InstrumentTracker` (real or `FakeInstrumentTracker`), the
frontend consumes this API. Neither side ever reaches through its seam.

> **Decisions baked in (2026-07-07):** Bram owns backend + frontend; Constantijn
> owns the model behind `InstrumentTracker`. Recording screen shows a live
> per-instrument list. Start gate is **human judgment** (operator eyeballs the
> overlay; Start enables once the detected count is stable, at any value ≥ 1).
> Report is **in-memory only** — one finished session, cleared on the next Start.

---

## Transport & shape

- **Polling, not WebSocket.** The frontend polls `GET /status` at ~2 Hz. At demo
  scale this is indistinguishable from push and removes a whole failure class.
- **MJPEG for video.** `GET /stream` is a `multipart/x-mixed-replace` stream the
  frontend renders with a plain `<img src="/stream">` — same proven mechanism as
  the data-collection dashboard.
- **Backend computes, frontend renders.** Usage windows, completeness,
  debouncing — all backend. The frontend never derives analytics from raw data.
- **Setup thumbnails are backend-derived.** The model returns frame-aligned
  detections, not encoded crops. The backend derives small previews from its
  camera frame and returns them inline with `/status`; the frontend does not
  crop the MJPEG stream or call the model directly.
- **Times:** absolute moments are ISO-8601 strings; everything inside a session
  is **seconds relative to recording start** (floats) — what a timeline needs.
- **Errors:** wrong-phase actions → `409` with `{"detail": "..."}`. Everything
  else standard FastAPI validation errors.

## Phases

One global state machine, owned by the backend:

```
setup ──POST /recording/start──▶ recording ──POST /recording/stop──▶ finished
                                    ▲                                    │
                                    └──────POST /recording/start─────────┘
                                         (discards the previous report)
```

- `setup` — camera live, tracker running for the overlay, nothing recorded.
- `recording` — tracker was `reset()` at Start; per-instrument state accumulates.
- `finished` — report available AND the camera/tracker keep live-observing the
  table (overlay still drawn, detected-count still updating) so the Start gate
  works here just like in `setup`. Next Start discards the report and re-enters
  `recording`. See §/status: `finished` carries the same `setup` block as `setup`.

---

## Endpoints

### `GET /status` — poll ~2 Hz

```jsonc
{
  "phase": "setup",                  // "setup" | "recording" | "finished"
  "capture_health": "ok",            // "ok" | "stalled"  (frozen camera is visible)
  "model_version": "scenario-0.1",

  // present when phase == "setup" OR "finished": what the Start gate needs
  // (in "finished" the table is still being observed for the next run)
  "setup": {
    "detected_count": 5,             // size of the current detected id-set
    "stable_for_s": 3.2,             // how long the detected ID-SET has been
                                     // unchanged (an id swap at constant count
                                     // resets it — T02 owns this definition)
    "detections": [                  // always present; may be []
      {
        "tracker_id": 1,
        "label": "Instrument 1",
        "thumbnail": "data:image/jpeg;base64,…" // string | null
      }
    ]
  },

  // phase == "recording": drives the live instrument panel
  "recording": {
    "started_at": "2026-07-20T10:31:04+02:00",
    "elapsed_s": 74.3,
    "on_table_count": 4,
    "instruments": [
      {
        "tracker_id": 3,
        "label": "Instrument 3",     // single class today; label comes from backend
        "on_table": false,
        "off_since_s": 61.0,         // null when on_table
        "pickup_count": 2
      }
    ]
  }
}
```

`setup` block is non-null when `phase == "setup" | "finished"`; `recording`
block is non-null only when `phase == "recording"`. (So `setup` and `recording`
are never both non-null, but in `finished` the `setup` block IS present — the
report is fetched separately via `GET /report`.)

`setup.detections` is sorted by `tracker_id` and contains detections from one
backend capture snapshot. `thumbnail` is a small aspect-preserving JPEG preview
returned as a data URI; it is `null` when that individual crop cannot be
validated or encoded. One bad crop must not fail the entire `/status` response.
The API returns all current detections; presentation limits such as the setup
constellation's seven-tile cap belong to the frontend.

`detected_count`/`stable_for_s` come from the session state while `detections`
comes from the latest capture snapshot. They may differ by one frame, and
`detections` may temporarily be empty when no usable snapshot exists. The
frontend must tolerate `detected_count != detections.length` and fall back to a
representative icon for any missing or null thumbnail.

Frontend enables **Start** when `phase == "setup" | "finished"`,
`capture_health == "ok"`, `detected_count ≥ 1`, and `stable_for_s ≥ 2` — the
operator confirms visually on the overlay that everything is detected.

### `GET /stream` — MJPEG

Live frames with the tracker overlay drawn (boxes + masks, coloured by
`tracker_id`, live count burned in or not — overlay content is backend's call
and NOT part of this contract).

### `POST /recording/start` → `200 {"started_at": ...}`

Allowed from `setup` or `finished` (discards the old report). Calls
`tracker.reset()`. `409` if already `recording`.

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

---

## Semantics the backend owns (not exposed, but pinned here)

- **Off-table debounce.** An instrument is "off table" only after its
  `tracker_id` has been absent for `> OFF_DEBOUNCE` (~1.5 s), and "back" after
  present for `> ON_DEBOUNCE` (~1.0 s). Otherwise detector flicker becomes fake
  pickups. Config values, not API.
- **Identity = `tracker_id`.** The report trusts the tracker's ids completely.
- **Detection previews are presentation data, not model output.** The backend
  owns the original frame passed to `InstrumentTracker.update()` and derives
  `/status` thumbnails from that frame plus the returned `xyxy` and
  `tracker_id`. Crop geometry, encoding and HTTP transport do not extend the
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

To make the fake useful for THIS app, extend it with a **scripted scenario**
(the frozen `DEFAULT_SCENARIO` in T01: instrument 1 leaves at t=20 s and returns
at t=35 s; instrument 3 leaves at t=50 s and never returns) so the live panel,
timeline, and "missing" badge can be developed against realistic data.
`FakeInstrumentTracker` today only drifts and flickers — it never simulates a
pickup.

Type safety across the seam: FastAPI serves `openapi.json` for free — generate
the frontend's TypeScript types from it (`openapi-typescript`) instead of
hand-writing interfaces, so contract drift becomes a compile error.
