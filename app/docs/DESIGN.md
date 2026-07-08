# DESIGN — ORC demo app (`app/`)

**This is the MVP demo for UZ Leuven (2026-07-20), NOT the data-collection
dashboard.** Different product: the dashboard captures training frames; this app
runs a live demo recording and reports per-instrument **Usage** and
**Completeness**. Nothing in `data-collection/` is a spec for this app.

Coder agents cannot reach the PRD (it lives in an Obsidian vault), so the facts
they need are restated here. Decision authority: **Bram**. Model half
(detector + tracker behind `InstrumentTracker`): **Constantijn**.

## What the demo does (from the PRD)

A camera on a tripod frames a sterile table with 5–6 surgical instruments.
One operator screen:

1. **Setup** — live feed with detection overlay. The operator eyeballs that
   every instrument is detected, then presses **Start**. Two distinct things
   gate Start: (a) the *target count* is **human judgment** — there is no
   configured "expected N"; the operator decides the overlay looks right; and
   (b) the *stability window* is a **fixed constant (2 s)** — Start only enables
   once the detected **id-set** has held unchanged (count ≥ 1) for ≥ 2 s; an
   instrument swap that keeps the count the same still resets the window. (a) is
   unconfigured; (b) is not negotiable.
2. **Recording** — live feed + a live per-instrument panel (on table /
   off table, off-since, pickup count). Any instrument may be picked up,
   carried out of frame, and returned — any order, repeatedly, **one off the
   table at a time** (a *demo choreography* constraint, not an engine limit:
   `Session` tracks each `tracker_id` independently and handles ≥ 2 simultaneous
   absences correctly — see T02 semantics — the live panel just reads cleaner
   with one at a time).
3. **Report** (after **Stop**) — per instrument: a **Usage** timeline (its
   off-table windows) and a **Completeness** badge (*present* on the table at
   Stop, or *missing* — glossary-canonical: the camera observes absence; it
   cannot claim *lost* vs *misplaced*).

Acceptance = the report matches what physically happened in the run.

## Architecture

```
camera ──▶ CaptureLoop (1 thread, owns tracker.update)          app/backend
              │ on_frame(t, present_ids)      │ snapshot() → overlay JPEG
              ▼                               ▼
           Session (state machine,         FastAPI  ──HTTP──▶  app/frontend
           debounce, usage windows,        (contract:           (Vite+React+TS
           completeness, report)           api-contract.md)     +shadcn, polls
                                                                /status @2Hz)
```

The HTTP seam is **frozen** in [`api-contract.md`](api-contract.md). The model
seam is **frozen** in [`model/docs/tracker-interface.md`](../../model/docs/tracker-interface.md)
(`InstrumentTracker`: `update(bgr_frame) -> sv.Detections` with stable
`tracker_id`, `confidence` attr, `reset()`, `class_names`, `model_version`).

## Decisions (settled — do not relitigate)

| # | decision | rationale |
|---|----------|-----------|
| D1 | Backend: **FastAPI**, Python, `uv` project under `app/backend/` | matches team stack |
| D2 | Frontend: **Vite + React + TS + Tailwind + shadcn/ui** under `app/frontend/` | SPA on a laptop; no SSR need |
| D3 | Live video: **MJPEG** `multipart/x-mixed-replace`, rendered via `<img>` | trivial, proven, no WebRTC |
| D4 | Live data: frontend **polls `GET /status` at ~2 Hz** | at demo scale, push adds only failure modes |
| D5 | **Single resolution 1920×1080** end-to-end | nothing is saved to disk here; no 4K machinery |
| D6 | One background **capture-infer thread**, frames in order, atomic `Latest` snapshot, `CAP_PROP_BUFFERSIZE=1`, stale-detection | the one lesson imported from data-collection |
| D7 | Report **in-memory only**; next Start discards it | demo shows the report right after Stop |
| D8 | Identity = `tracker_id`, verbatim. Labels are `"Instrument {tracker_id}"` | single class; re-id is model-side. **Resolved 2026-07-08:** ids survive absence via **track linking** behind the seam (tracker-interface §tracker_id across absence) — linked tracks re-emit the original id within ≤ 1 s of return; `Session` adds a 1 s **entry debounce** (T02) so provisional pre-link ids leave no trace. Linker owner: Constantijn (TBC) |
| D9 | **Off/on debounce** in `Session` (defaults 1.5 s off, 1.0 s on) so detector flicker ≠ pickups | correctness of the report |
| D10 | Fake = **`ScenarioTracker`** (scripted pickups/returns/losses) + `FakeCaptureSource`, living in `app/backend/` — `model/` is not touched | fake at the deepest seam; frontend always talks to the one real API. `FakeInstrumentTracker` in `model/` only drifts — it can't demo Usage |
| D11 | `Session` is **pure** (fed `(t, present_ids)`; no clock, no threads inside) | the whole report correctness becomes unit-testable |
| D12 | Frontend testing: **full component tests** (Vitest + React Testing Library + MSW) plus pure-logic units | Bram's call, 2026-07-07 |
| D13 | Process: **blind-TDD** (`/blind-tdd`) for backend tasks; frontend tasks single-agent TDD + **Claude-in-Chrome** visual verification against the fake backend | Bram's call, 2026-07-07 |
| D14 | TS API types hand-written from `api-contract.md` at scaffold; swapped for `openapi-typescript` generation at integration | breaks the chicken-and-egg with the not-yet-built backend |
| D15 | **`finished` keeps live-observing** the table (tracker running, overlay drawn, detected-count updating) and carries the same `setup` block as `setup`, so the Start gate applies identically for a second run | the camera never stopped; lets the operator re-confirm the table before re-recording without a separate "back to setup" transition (Bram, 2026-07-07) |

## Threading rules

- Exactly one thread touches the camera and the tracker: the `CaptureLoop`.
- `Session.observe()` is called from that thread (via `on_frame`); HTTP handlers
  read `Session`/`Latest` under a lock. Keep `Session` methods O(instruments).
- Endpoints that do blocking work stay plain `def` (FastAPI threadpool) — never
  block the async loop that serves `/stream`.

## Repository layout the tasks build

```
app/backend/    pyproject.toml (uv), backend/{capture,session,fakes,render,main}.py, tests/
app/frontend/   Vite app: src/{api,screens,components}, tests colocated, MSW in src/test/
app/docs/       DESIGN.md (this), api-contract.md (frozen), tasks/ (the board), RUNBOOK.md (T08)
```

Run modes: `uv run orc-demo --fake` (ScenarioTracker + synthetic frames — what
frontend dev and Chrome verification use) · `uv run orc-demo --camera 1
--weights <path>` (real tracker from Constantijn's `load_tracker` factory when
it lands).
