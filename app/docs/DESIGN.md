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
| D7 | Report **in-memory only**; next Start discards it. **Recording history across sessions is an explicit non-goal for the MVP** (2026-07-08) — it would need persistence + new `/reports` contract endpoints + a history screen, re-coupling the frontend to backend storage decisions for a single controlled demo run. Post-MVP feature if wanted. Losing the *current* report on an accidental "New recording" is handled in-UI instead (T06 "Back to report"). | demo shows the report right after Stop |
| D8 | Identity = the **session id** the tracker emits, verbatim. Labels are `"Instrument {id}"` | single class; re-id is model-side. **Resolved 2026-07-08:** ids survive absence via **track linking** behind the seam (tracker-interface § "Identity semantics") — linked tracks re-emit the original id within ≤ 1 s of return; `Session` adds a 1 s **entry debounce** so provisional pre-link ids leave no trace. **Updated 2026-07-15:** the linker is **built and owned in `model/`** (`SessionLinker`, composed by `load_tracker()`) — not Constantijn's. Two deltas the app must respect: (a) the roster is **frozen ~0.7 s after Start**, and the app **reads it across the seam** (`tracker.roster`) rather than inferring it — a track that matches no enrolled identity is **Unknown**, is never labelled `"Instrument N"`, and stays out of Usage and Completeness (see D8a); (b) session ids are **renumbered at the freeze** (wayfinder T08 shipped 2026-07-15), so the roster reads `{1…N}` and a tray of 8 labels "Instrument 1"…"Instrument 8" — the earlier "a tray of 8 can show Instrument 10" is no longer true. Measured on both 2026-07-15 takes, *including* a replay with only 4 of 8 galleries bound. **But the app still treats the roster as an opaque set** — membership tests, never `id > N`. Contiguity is a model-side property; depending on it would re-couple the app to the linker's internals. |
| D8a | **Unknown objects** render gray **on the video only**, outside Usage and Completeness; a not-yet-settled object shows a ~1 s **resolving** spinner. Roster instruments get 8 distinct, stable colours, reported per-instrument in `/status` | 2026-07-15, closing wayfinder T06. Freezing the roster at Start is what keeps an audience member's phone from becoming "Instrument 7" and counting as present. **Grilled 2026-07-15 (Bram); SHIPPED same day** (wayfinder T10, spec in `docs/wayfinder/session-linker/assets/T08-T10/T10-BACKEND-SPEC.md`). **Route (b) — the roster crosses the seam:** `InstrumentTracker` exposes a read-only `roster` property; `CaptureLoop` samples it in the *same tick* as `present_ids` and carries it into `Latest`, `on_frame`, the renderer, `Session.observe(t, present_ids, roster)` and `/status`. **Route (a) — the app deriving the roster from the ids present at Start — was rejected:** the app's Start snapshot and the linker's enrolment freeze are *different instants* (~0.7 s apart), and if the two sets ever disagree then **every** Unknown decision disagrees. **Unknowns are video-only:** gray mask + ~1 s resolving spinner, then a settled gray `"Unknown"` label — never `"Instrument N"`, at any age, in any state; **no panel row**, never in `/status.recording.instruments`, Usage, or Completeness. **Colour:** each roster instrument gets one of 8 distinct hexes from `roster_colour(roster, id)` — derived from the id and the frozen roster, so it is pure and a returned instrument **regains its colour with no renderer memory**; the same hex is drawn on the mask and reported in `/status`, so swatch and video cannot drift. **Only the RECORDING half filters by roster** — see "D8a in practice" below. See `model/docs/linker-design.md` §7 |
| D9 | **Off/on debounce** in `Session` (defaults 1.5 s off, 1.0 s on) so detector flicker ≠ pickups | correctness of the report |
| D10 | Fake = **`ScenarioTracker`** (scripted pickups/returns/losses) + `FakeCaptureSource`, living in `app/backend/` — `model/` is not touched | fake at the deepest seam; frontend always talks to the one real API. `FakeInstrumentTracker` in `model/` only drifts — it can't demo Usage |
| D11 | `Session` is **pure** (fed `(t, present_ids)`; no clock, no threads inside) | the whole report correctness becomes unit-testable |
| D12 | Frontend testing: **full component tests** (Vitest + React Testing Library + MSW) plus pure-logic units | Bram's call, 2026-07-07 |
| D13 | Process: **blind-TDD** (`/blind-tdd`) for backend tasks; frontend tasks single-agent TDD + **Claude-in-Chrome** visual verification against the fake backend | Bram's call, 2026-07-07 |
| D14 | TS API types hand-written from `api-contract.md` at scaffold; swapped for `openapi-typescript` generation at integration | breaks the chicken-and-egg with the not-yet-built backend |
| D15 | **`finished` keeps live-observing** the table (tracker running, overlay drawn, detected-count updating) and carries the same `setup` block as `setup`, so the Start gate applies identically for a second run | the camera never stopped; lets the operator re-confirm the table before re-recording without a separate "back to setup" transition (Bram, 2026-07-07) |
| D16 | **`VideoFeed` degrades gracefully when `/stream` is unreachable.** It stays a bare `<img src={api.streamUrl}>`, but an `onError` handler swaps in a plain styled **"no stream (dev mode)"** panel (no bundled image asset). MSW cannot intercept an `<img>` MJPEG load, so in pre-backend dev/`npm run dev` (T05 AC6) and MSW-only component tests the feed always shows this panel; the real MJPEG is only exercised against the `--fake` or real backend (T06 AC7, T08). Same fallback covers a genuinely dead camera in production. | Bram, 2026-07-08. Unblocks building/verifying every live screen with no backend, without pretending there's a video source that isn't there |
| D17 | **Setup detection thumbnails are derived in `app/backend` from the owned camera frame plus public `InstrumentTracker` output, then returned inline in `/status`. The model never emits UI crops or transport-specific bytes.** | preserves the frozen model seam; Constantijn supplies same-frame, row-aligned `xyxy`/`mask`/`tracker_id`, while crop geometry, encoding and presentation remain consumer concerns (2026-07-13) |

### D8a in practice — what the roster gates, and what it must NOT

**The single most load-bearing detail of the T10 build: only the RECORDING half
filters by roster.** `Session.observe(t, present_ids, roster)` passes
`present_ids & roster` to `_observe_recording`, so a not-in-roster id never
becomes a track, never confirms, never reaches `recording_status()`, the report,
`on_table_count`, Usage or Completeness. But the **Start/setup gate deliberately
still sees the full `present_ids`** — `detected_count` and `stable_for_s` are
computed from everything detected. That is not an oversight to tidy up later:
the Start gate is *the operator's judgment on everything on the table*, made
**before any roster exists** (there is nothing to filter against yet, and
filtering the gate by a stale roster from the previous run would let a real
instrument fail to appear in the count the operator is eyeballing). Inverting
this — filtering setup, or not filtering recording — silently breaks the demo's
Start flow. Same reasoning keeps the overlay unfiltered outside `recording`
(B-V1): in `setup` and `finished`, everything is drawn with its per-track colour
and an `"Instrument {id}"` label, because the roster there is stale by design.

The **~0.7 s window between Start and the linker's enrolment freeze**, where
`roster` is still empty and therefore *nothing* passes the recording filter, is
harmless: it is shorter than `Session`'s 1 s entry debounce, so no track can
confirm inside it and the report is unchanged. The same window is why every
`/status` `colour` is briefly gray (see `api-contract.md`). Both are transient
and self-correcting; neither needed code.

`roster=None` (the default) means unfiltered, so every pre-T10 caller and test
keeps today's semantics.

### ⚠️ Not grilled — engineering calls flagged for Bram (T10, 2026-07-15)

These were decided *by the build*, not by Bram. The last build silently reversed
grilled decisions and it took days to notice, so they are listed here to be
vetoed rather than discovered. Nothing below is load-bearing enough to block the
demo; all of it is cheap to reverse.

1. **Renderer wiring (B-V6).** The T10 spec never said how `create_app` reaches
   the `OverlayRenderer` to call `set_recording`. Ruled: **`create_app` owns the
   renderer** and wires it through a new `CaptureLoop.set_render_fn`, mirroring
   the existing `set_on_frame` precedent. Consequence worth knowing: `create_app`
   now **replaces any `render_fn` passed to `CaptureLoop`'s constructor**.
2. **`roster_colour` returns gray for a non-roster id** rather than raising. The
   spec defines it only for roster ids; the recording filter (B-S1) stops
   unknowns from ever reaching it, and returning gray collapses the
   no-snapshot placeholder (B-A1) into the same path. A raise would have been
   the stricter contract.
3. **A residual race in `OverlayRenderer._age`, named and accepted.** Two races
   were fixed (`set_recording` now clears the first-seen map *before* publishing
   the flag; `_evict` mutates in place and never rebinds). A third survives:
   `_age` is a read-modify-write, so if `set_recording(True)`'s clear lands
   between the get and the store, an in-flight frame from the *previous*
   recording writes its `first_seen` into the fresh map — that object then shows
   a settled `"Unknown"` immediately instead of the resolving spinner, until
   eviction ~5 s later. It needs a GIL switch inside a 2-op window **and** a
   frame in flight across a stop→start within one ~100–300 ms render tick.
   Judged cosmetic and improbable. Closing it costs ~4 lines (an epoch counter)
   or a lock on the capture hot path — **Bram can call for the epoch counter.**
4. **A pre-existing race, flagged but NOT fixed** (found independently by two
   agents; it is not T10's). `session.start(clock())` samples the clock under the
   lock while the capture thread's in-flight frame still carries an earlier `t`,
   so that frame's `observe` arrives backwards → `ValueError: t is not
   monotonic`, caught and logged by `_notify`'s AC8 handler, frame skipped. This
   happens in `orc-demo` today, T10 or no T10.
5. **BRANDING.md now disagrees with the product.** `BRANDING.md` says the camera
   detection overlay "should use the functional status colours, not the logo
   spectrum", and that colour "must carry meaning". **An 8-hue roster palette is
   exactly what that line pushes back on.** It was built as grilled, and the hue
   *does* carry meaning — it maps mask ↔ panel row identity, it is not
   decoration — but the panel now shows up to 8 hues on a surface branded "teal +
   semantic only". Either BRANDING carves out an identity-mapping exception, or
   the doc and the product stay in conflict. **Not changed here — Bram's call.**
6. **Frontend swatch a11y convention (new).** The panel swatch is
   `aria-hidden` + `data-testid`, not `role="img"`: a hue is meaningless to a
   screen reader (the row already carries "Instrument 3"), and `role="img"` would
   have broken `LiveScreen.test.tsx`'s `queryByRole("img")` assertion. The repo
   had no convention for this before; this is the first one.

Also known, not a call: **nothing tests that the spinner *stops* at settle**
(B-V4 only implies it via "solid gray"). The shipped code does stop it, but no
gate holds it there.

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
