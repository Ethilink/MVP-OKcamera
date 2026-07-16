# RUNBOOK — ORC demo (UZ Leuven)

Demo-day operating sheet for the OR-camera instrument-tracking demo. Two
processes: the **backend** (`orc-demo`, serves the HTTP API + MJPEG stream) and
the **frontend** (Vite SPA, the operator screen). The frontend polls
`GET /status` at ~2 Hz and renders the live overlay from `GET /stream`.

> **STATUS (2026-07-09) — FAKE MODE VERIFIED END-TO-END (API level).** The
> backend HTTP layer + `orc-demo` CLI (T04) are implemented; `uv run orc-demo
> --fake` boots FastAPI on `:8000` and serves the full contract. Generated types
> (`npm run gen:api`) match the hand-written ones exactly, all 42 frontend tests
> + `npm run build` pass against them, and the scripted scenario round-trips into
> the report (instrument 1 present with one closed usage window; instrument 3
> `missing` with one open window). **Still open for Bram:** (1) the visual Chrome
> pass — driving the actual SPA in a browser (this runbook + the API smoke cover
> everything below the pixels); (2) real-camera + real-weights mode (needs
> Constantijn's weights). **Timing caveat:** in `--fake` the scripted t=20/35/50 s
> land ~1.4× later in wall-clock (≈28/48/68 s) because per-frame render+JPEG
> encode drops the capture loop below its nominal 10 fps — events fire in the
> right order, just later than the nominal seconds. See the rehearsal note.

---

## Preconditions (do these before demo day)

- [x] **T04 merged** — `orc-demo --fake` boots FastAPI + uvicorn on `:8000` and
      serves `/status`, `/stream`, `/recording/start`, `/recording/stop`,
      `/report`, and `/openapi.json`. (Verified 2026-07-09.)
- [x] `npm run gen:api` runs **against a live backend**; `src/api/schema.d.ts` is
      generated and `src/api/types.ts` re-exports it (see "Generated types").
      **Commit `schema.d.ts`** so a clean checkout builds without a live backend.
- [x] `npm test` (42 passing) + `npm run build` (frontend) green against the
      generated types. Backend `uv run pytest` — run before demo day to confirm.
- [x] End-to-end fake smoke passes at the API level (instrument 1 present with
      one closed usage window, instrument 3 `missing` with one open window;
      `POST /recording/stop` body byte-identical to `GET /report`).
- [ ] **Visual Chrome pass** — drive the SPA in a browser end-to-end (open for
      Bram / orchestrator; everything below the pixels is verified).
- [ ] Weights + camera decision from Constantijn confirmed (real vs fake demo);
      **real-mode remains unverified — open for Bram** (no weights/camera here).

---

## Hardware checklist

- [ ] **Laptop** — charged + charger; can reach `localhost:8000` / `:5173`.
- [ ] **Camera** — one of:
  - **Reincubate Camo** (phone as webcam): Camo app running on the phone,
    Camo Studio running on the laptop, phone on the tripod. **Do not assume an
    index** — there is no "typical" one. Measured on the demo laptop: Camo was
    `0`, then `1` twenty minutes later, with FaceTime/Iriun/Continuity shuffling
    around it. Always resolve it fresh, right before the demo:
    ```
    ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -A6 "video devices"
    ```
    Then confirm with the real-mode dry run (you should see the tray, not a face).
  - **USB webcam** — plugged in before launching the backend; note its index.
- [ ] **Tripod** — stable, framing the full instrument table top-down or at a
      steep angle so every instrument is unobstructed.
- [ ] **Lighting** — even, no hard shadows or blown-out highlights across the
      tray; instruments visually separated (detector likes contrast).
- [ ] **Instrument set** — the tray used for the demo, instruments spaced so the
      detected count is stable (the Start gate needs a steady id-set for ≥ 2 s).
- [ ] **Backup**: know the `--fake` command cold — if the camera misbehaves, the
      scripted fake demo is the fallback and needs no hardware.

---

## Commands

Two terminals. Start the **backend first** (the frontend and `gen:api` both
expect `:8000` up).

The detector, Deep OC-SORT, matcher, linker/Unknown re-detection, report
debounce, and capture tuning are all documented in [`../mvp.toml`](../mvp.toml).
That file is loaded by default. Use `--config /path/to/profile.toml` to run a
different tuning profile without editing the default. Camera index and weights
remain CLI inputs because they identify runtime hardware/artifacts rather than
algorithm tuning.

### Fake mode (no camera — scripted scenario, what dev/rehearsal uses)

Terminal 1 — backend:
```
uv run --directory app/backend orc-demo --fake
```
Serves on `http://localhost:8000`. Runs `ScenarioTracker` + `FakeCaptureSource`
(synthetic 1080p frames). The scripted `DEFAULT_SCENARIO`: **instrument 1 leaves
at t=20 s and returns at t=35 s; instrument 3 leaves at t=50 s and never
returns.** Times are seconds from the moment you press **Start**.

Terminal 2 — frontend (MSW OFF — talk to the real backend):
```
cd app/frontend && npm run dev
```
Open the printed URL (default `http://localhost:5173`).

### Real mode (camera + Constantijn's weights)

Terminal 1 — backend:
```
uv run --directory app/backend orc-demo --camera 1 \
  --weights /Users/<you>/…/MVP-OKcamera/model/weights/checkpoint_best_regular.onnx
```
- `--weights PATH` — the RF-DETR **ONNX** file (`model/weights/checkpoint_best_regular.onnx`).
  **Must be ABSOLUTE.** `uv run --directory app/backend` makes `app/backend` the
  working directory, so a repo-relative path resolves under it and dies with a
  confusing `FileNotFoundError: 'model/weights/.coreml_cache'` (the CoreML cache
  is created next to the weights, so it fails there first).
- `--camera N` — **verify it every session; it is not stable.** Camo was index 0
  and then index 1 on the same machine ~20 min apart, and any Iriun/Continuity
  device shuffles it further. Do not trust a remembered number:
  ```
  ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -A6 "video devices"
  ```
  Use the index printed next to `Camo Camera`. Backend boot takes ~11 s (ONNX +
  DINOv2 load) with no "ready" line before it — wait for uvicorn's port-8000 log.
  Real mode runs at ~3 fps by design (`tracker.expected_processing_fps` in
  `app/mvp.toml`, matching the measured 2.87–3.13; the linker's frame-based
  windows are calibrated to it).

Terminal 2 — frontend: same `npm run dev` as above.

### Generated types (`gen:api`) — backend up FIRST

`gen:api` fetches `openapi.json` over HTTP from the running backend, so the
**backend must already be running** (`--fake` is fine) before you run it:
```
cd app/frontend && npm install         # one-time; see peer-dep note below
# terminal 1: uv run --directory app/backend orc-demo --fake   (leave running)
cd app/frontend && npm run gen:api      # writes src/api/schema.d.ts
```
`src/api/types.ts` re-exports the generated `schema.d.ts` under the
contract-facing names, so a shape drift between the backend and the frontend is
now a **TypeScript compile error**, not a silent runtime bug. After regenerating,
`npm test` + `npm run build` must still pass; a diff in behaviour is a contract
bug (fix the backend/T04, not `api-contract.md`).

> **Peer-dep note.** `openapi-typescript@7` still declares `typescript@^5` as a
> peer while this app is on `typescript@6`. `package.json` carries an `overrides`
> entry (`openapi-typescript → typescript: $typescript`) so a plain
> `npm install` resolves cleanly — do **not** fall back to `--legacy-peer-deps`,
> which drops `@testing-library/dom` (an optional RTL peer) and breaks the test
> suite. If a future `openapi-typescript` adds native TS 6 support, the override
> can be removed.

---

## Weights handoff (from Constantijn)

- The real tracker is loaded behind the `InstrumentTracker` seam via
  Constantijn's `load_tracker` factory (D8/D10). The backend imports it lazily,
  only in the `--camera`/`--weights` branch (never at fake-mode startup).
- **Ask Constantijn for:** the weights file path + the exact model/class the
  weights were trained for, and confirmation that `load_tracker(weights_path)`
  returns something implementing `InstrumentTracker`.
- Copy the weights somewhere stable on the demo laptop and pass that absolute
  path to `--weights`. Do a real-mode dry run **before** the audience arrives.
- `model_version` in `/status` and `/report` reflects the loaded model — sanity
  check it is not `scenario-0.1` (that means you booted fake by mistake).

---

## Pre-demo rehearsal (10 steps, ~5 min)

Run this end-to-end once before the demo, in the mode you'll present in.

1. Backend up: `uv run --directory app/backend orc-demo --fake` (or real).
2. Frontend up: `npm run dev`; open the URL. Screen shows **Setup**.
3. Confirm the **live video** panel shows frames (fake: synthetic frame with the
   overlay; real: the table). If it shows "no stream (dev mode)", the backend
   isn't reachable — fix before continuing.
4. Watch the **detected count** climb and settle; **Start** enables once the
   id-set is stable ≥ 2 s and count ≥ 1 (human-judgment gate — eyeball the
   overlay: every instrument boxed).
5. Press **Start** → screen switches to **Recording**; timer runs from 0.
6. (Fake) Around **t≈28 s** (scripted t=20, dilated ~1.4× — see timing caveat)
   watch **instrument 1** flip to **OFF TABLE**; around **t≈48 s** it returns
   to **ON TABLE**. The usage window is intentionally not revealed yet.
7. (Fake) Around **t≈68 s** watch **instrument 3** go OFF and stay off.
8. Press **Stop** (wait until **after t≈70 s** in fake, so instrument 3's loss is
   confirmed past the off-table debounce) → screen switches to **Report**.
9. Verify the report: **instrument 1 = present**, one closed window
   (~20 s → ~35 s); **instrument 3 = missing**, one open window (off ~50 s, no
   return). Every tracked instrument listed exactly once.
10. Press **New recording** → back to Setup (report discarded, gate re-arms).
    You're rehearsed. Leave both processes running for the real thing, or Ctrl-C
    and restart clean.

---

## Failure playbook

**Stalled / frozen camera (live feed stops updating).**
- `/status` reports `capture_health: "stalled"`; the UI shows a **stalled
  banner**. The stream image freezes.
- Fix: check the camera cable / Camo connection. If it doesn't recover in a few
  seconds, restart the backend:
  ```
  Ctrl-C  (terminal 1)
  uv run --directory app/backend orc-demo --camera 1 \
    --weights /abs/path/to/model/weights/checkpoint_best_regular.onnx
  ```
  The frontend reconnects on its next poll; re-**Start** the recording.
  If Camo was re-connected, **re-check the index first** (see "Commands") — it
  can move between runs.
- If the camera can't be recovered, fall back to **fake mode** (no hardware).

**Backend crash mid-demo.**
- Frontend requests start failing (stream falls back to "no stream", polls
  error). Restart the backend with the **same command** you launched with:
  ```
  uv run --directory app/backend orc-demo --fake        # or the --camera/--weights line
  ```
- The report is **in-memory only** — a crash loses the current session's report.
  After restart, re-run: wait for the gate, **Start**, reproduce, **Stop**.

**Start button won't enable.**
- Gate needs `capture_health == "ok"`, `detected_count ≥ 1`, and the id-set
  stable for ≥ 2 s. Steady the scene / lighting so detections stop flickering;
  the `stable_for_s` counter resets on any id-set change.

**`npm run gen:api` fails / hangs.**
- The backend isn't up on `:8000`. Start `orc-demo --fake` first, then re-run.

**Wrong camera index.**
- Feed shows the built-in webcam (your face) instead of the tray → wrong
  `--camera N`. Try the next index (0 ↔ 1 most commonly); confirm Camo Studio is
  running if using Camo.

---

## Clean shutdown

- Ctrl-C each terminal. Nothing is persisted to disk (report is in-memory,
  frames are not saved — D5/D7), so there's no cleanup beyond stopping the two
  processes.
