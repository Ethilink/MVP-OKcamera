# DESIGN — ORC demo app (`app/`)

**This is the MVP demo for UZ Leuven (2026-07-20), not the data-collection
dashboard.** It runs a live surgical-set recording and reports per-instrument
**Usage** and **Completeness**. Nothing in `data-collection/` is a specification
for this app.

Decision authority: **Bram**. The detector/tracker lives behind the
`InstrumentTracker` seam. The current HTTP contract is
[`api-contract.md`](api-contract.md); the model seam is
[`model/docs/tracker-interface.md`](../../model/docs/tracker-interface.md).

## Product flow

The camera frames the demo catalog of eight surgical instruments.

1. **Setup** — the live overlay classifies every detected object as recognised,
   recognising, or unknown. Track is enabled only when the server confirms that
   all eight catalog identities are present, no unknown/resolving object exists,
   the complete detected id-set has held still for at least 2 seconds, and the
   camera is healthy. The frontend displays this verdict; it does not reproduce
   the gate logic.
2. **Recording** — the approved roster is preserved across Start. The screen
   shows the live feed, elapsed time, and the recognised instruments' current
   on-table/off-table states. Usage history and Completeness remain hidden until
   Stop. Instruments may leave and return in any order; the demo choreography
   normally removes one at a time, but `Session` tracks absences independently.
3. **Report** — after Stop, each recognised instrument has a Usage timeline and a
   Completeness badge. The report says `missing`, not `lost`: the camera can
   observe absence but cannot infer why an item is absent.

Acceptance is that the report matches what physically happened during the run.

## Architecture

```text
camera ──▶ CaptureLoop (one capture/inference thread)             app/backend
              │
              ├─ atomic Latest(frame, detections, roster, catalog)
              ├─ on_frame(t, present, roster, catalog, resolving)
              ▼
           Session (phase, readiness, debounce, usage, report)
              │
              ▼
           FastAPI ── HTTP/MJPEG ──▶ React + TypeScript          app/frontend
                                      polls /status at ~2 Hz
```

`CaptureLoop` is the only owner of camera reads, `tracker.update()`, tracker
resets, and runtime confidence mutations. It samples detections, roster, catalog,
and resolving state in the same tick so the session, overlay, thumbnails, and API
all judge one coherent frame.

## Settled decisions

| # | Decision | Why |
|---|---|---|
| D1 | FastAPI backend under `app/backend/` | Matches the project stack and exposes OpenAPI. |
| D2 | Vite + React + TypeScript + Tailwind + shadcn/ui under `app/frontend/` | Single-laptop SPA; no SSR requirement. |
| D3 | MJPEG live video via `GET /stream` | Simple, proven, and no WebRTC state machine. |
| D4 | Poll `GET /status` at about 2 Hz | Demo-scale live state does not need push transport. |
| D5 | 1920×1080 capture path | No media is persisted by this app. |
| D6 | One capture/inference thread with an atomic `Latest` snapshot, camera buffer size 1, and stale-health detection | Prevents tracker races and stale-frame buildup. |
| D7 | One in-memory report; the next successful Start discards it | Recording history is outside the MVP. “New recording” only changes frontend layout, so Back to report remains possible before Start. |
| D8 | Identity is the session id emitted by the tracker; consumers treat roster/catalog as opaque sets | Re-identification and specimen binding stay model-side. Never infer identity from numeric ranges. |
| D8a | The fixed catalog supplies stable identity colours. Recognised masks, setup tiles, and recording rows use the same backend-provided colour; unknown/resolving objects are gray | One identity policy prevents overlay/UI disagreement. |
| D9 | `Session` owns 1.5 s off and 1.0 s on debounce defaults | Detector flicker must not become a pickup. |
| D10 | `ScenarioTracker` + `FakeCaptureSource` provide the app demo fake | The frontend still exercises the real HTTP implementation. |
| D11 | `Session` is pure and clock-free; callers feed monotonic timestamps and same-tick sets | Readiness and report behavior remain deterministic and unit-testable. |
| D12 | Frontend behavior uses component tests with Vitest, Testing Library, and MSW | Covers the real UI/API seam without browser-only tests for every state. |
| D13 | OpenAPI generates the frontend schema; small presentation types wrap it locally | Contract drift becomes a type-check failure. |
| D14 | `finished` continues observing the table and exposes the same setup-readiness block | A second run can be approved without restarting the camera or backend. |
| D15 | Setup/live thumbnails are backend-derived from the owned frame plus public detection boxes and masks | The model seam does not emit UI assets or transport bytes; the backend turns masks into transparent PNG cutouts and retains a JPEG fallback for unusable masks. |
| D16 | Start is a pure, fail-closed phase transition and **does not reset the tracker** | Resetting after approval empties the approved roster and causes unsafe re-enrolment during recording. |
| D17 | A changed runtime detection confidence is applied with a tracker reset as one serialized capture-thread command; failure rolls the confidence back | Confidence and tracker state must remain atomic. A no-op confidence change does not reset. |
| D18 | The setup constellation renders at most the eight **recognised** catalog items. Unknown/resolving detections remain on video and affect count/readiness, but are not constellation tiles | The constellation is the catalog view, while the video and gate cover everything on the table. |

## Identity and lifecycle invariants

- `catalog` is the fixed set loaded at tracker construction and survives resets.
- `roster` is the currently enrolled subset. Setup readiness requires the
  recognised present set to equal the catalog, with zero unknown/resolving ids.
- Start holds the mutation lock across the readiness re-check and phase change.
  A confidence reset therefore cannot land between approval and recording.
- Start preserves tracker state. It clears the previous report only after the
  gate passes and `Session.start()` succeeds.
- A changed confidence may be submitted only outside recording. The capture
  thread applies `confidence = new_value` and `reset()` between frames. If reset
  fails, the old confidence is restored and the API remains on the old confirmed
  value.
- Tracker commands are serialized inside `CaptureLoop`, not only by the HTTP
  layer, so direct concurrent callers cannot overwrite its one pending slot.
- Only ids in the same-tick roster enter recording instruments, Usage, or
  Completeness. Unknown objects never become recording rows.
- Setup `/status` still reports every detection and every gate count. The
  frontend constellation filters that payload for recognised identities only;
  this display filter must never be reused as gate logic.
- The setup count and latest detection list may be one frame apart. Readiness is
  always the server verdict, never a frontend count comparison.

## Frontend state recovery

- Start/Stop transitions are poll-confirmed rather than optimistic.
- A mutation refresh that fails waits for the next successful ordinary poll
  before releasing the pending UI state. A temporary `/status` outage therefore
  cannot leave Track disabled forever, and a stale response cannot falsely
  confirm the mutation.
- The MJPEG `<img>` has a visible fallback and retries automatically after a
  stream error.
- The report remains reachable from the run-2 setup layout until a successful
  Start discards it backend-side.

## Threading rules

- Exactly one thread touches the camera and calls tracker update/reset/confidence
  mutation: `CaptureLoop`.
- `Session.observe()` runs from the capture callback. HTTP handlers read/write
  session state under a short lock.
- The coarse mutation lock serializes Start and confidence changes. Never hold
  the short session lock while waiting for a capture-thread command.
- Each tracker command has its own completion/error state. The submission lock is
  held for the full queue-and-wait lifecycle.
- Plain `def` FastAPI endpoints perform blocking work in the thread pool; the
  async stream loop is not blocked by tracker commands.

## Repository layout

```text
app/backend/    FastAPI, capture/session/render/fakes, backend tests
app/frontend/   Vite React app, API hooks, screens/components, colocated tests
app/docs/       current design, API contract, runbook, historical task briefs
model/          tracker implementation and model contract
```

Run modes:

- `uv run orc-demo --fake` — scenario tracker + synthetic frames.
- `uv run orc-demo --camera 1 --weights <path>` — real camera and tracker.
