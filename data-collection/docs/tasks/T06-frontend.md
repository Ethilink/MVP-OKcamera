# T06 — Frontend (static UI)

status: done (claude)
depends-on: T01 (and the frozen endpoint contract in T05-api.md — not T05's code)
blocks: T07
spec: [DASHBOARD.md](../DASHBOARD.md) §Scope (Watch/Snapshot/Settings), §Stack (Frontend)

## Goal

Plain HTML + vanilla JS served by FastAPI: live stream, confidence slider,
SPACE-to-flag with visible feedback, settings form, status poll. No build step,
no framework.

## In scope

`static/index.html`, `static/app.js`, `static/style.css`. Develop against T05's
endpoint contract table; a tiny mock server (or T05 once done) is fine for
manual testing.

## Out of scope

Any editing/correction UI (separate annotation dashboard), recording controls
(there are none — the tool is stateless), backend changes.

## UI contract

- `<img src="/stream">` fills the main area (it's just an MJPEG `<img>` — no WebRTC).
- Slider `0.00–1.00` (step 0.01) → `POST /confidence`, **debounced ~150 ms**;
  current value shown as text.
- **SPACE** anywhere (except when a text input has focus) and a big "FLAG"
  button → `POST /flag`. Hold-to-repeat must NOT machine-gun: ignore key
  auto-repeat (`event.repeat`).
- Settings panel: `camera_index`, `output_path`, `dataset_name` → `POST /settings`.
- Poll `GET /status` every 1 s → live instrument **count**, `n_flagged`,
  active dataset name, and a **capture-health indicator** that goes visibly red
  on `"stale"`/`"dead"` (the operator must never flag a frozen stream).

## Acceptance criteria

Manual checklist (screen-record one pass when done); automated tests optional.

- [x] **AC1** SPACE flags; a flash/toast + incrementing `n_flagged` confirm each
  save within ~200 ms. Key auto-repeat does not fire extra flags.
- [x] **AC2** SPACE while typing in a settings field does NOT flag.
- [x] **AC3** A `409` from `/flag` (no target) or `/settings` (collision) is shown
  as a visible error message with the server's `detail` — never swallowed.
- [x] **AC4** Slider drags produce at most ~7 requests/s (debounce works) and the
  overlay visibly responds.
- [x] **AC5** Health indicator turns red within ~3 s of the stream dying and the
  FLAG button disables while unhealthy.
- [x] **AC6** "Validate" button shows `/validate` errors/warnings in a readable list.
- [x] **AC7** Zero external assets — works offline (no CDN fonts/JS).

## Log

- 2026-07-07 (claude) Shipped `static/index.html`, `static/app.js`,
  `static/style.css` against the T05 endpoint contract. All 7 ACs verified with
  a jsdom harness (18/18 checks: auto-repeat + typing guards, debounce collapses
  a 20-input burst to 1 request, 409 detail surfaced, health→dead disables FLAG
  and inerts SPACE, validate list).
- 2026-07-07 (claude) Chrome-driven pass done against the scratchpad mock server
  (Claude-in-Chrome). Confirmed live: SPACE→flag (n_flagged 0→1, toast, green
  flash), /flag 409 banner "no valid dataset target configured", health→dead
  red indicator + FLAG disabled, 20-input slider burst → 1 /confidence request,
  validate errors/warnings rendered colour-coded, and typing "or set two" in the
  dataset field kept both spaces as text without flagging. All 7 ACs green.
- 2026-07-07 (claude) **Two items for T05 / Bram:**
  (1) The page loads `/static/app.js` + `/static/style.css`, so T05 must
  `app.mount("/static", StaticFiles(directory="static"))` — the contract table
  only lists `GET /` for the index. (2) Touched `tests/test_scaffold.py`
  (T01-owned): its AC3 asserted the placeholder string "dashboard here", which
  this task removes. Re-pointed the assertion at the shipped `<title>` — AC3's
  intent (app boots, serves index) preserved. Flagging per BOARD rule 5.
- 2026-07-07 (claude) **Static mount gap closed (item 1 above was never wired).**
  `create_app` served `GET /` and the API routes but had no `/static` mount, so
  `style.css` + `app.js` 404'd in the running app — the page rendered as raw
  unstyled HTML with a dead script (no polling, health frozen at "connecting…",
  FLAG inert, settings form falling back to a native GET nav). Added the mount +
  a `test_static_assets_are_served` regression test in `tests/test_api.py`
  (asserts 200 + content-type for both, and that the index still references
  them). Suite now 64 green. Verified live in Chrome on the built-in webcam:
  styled UI, green "live" chip, apply-settings (no reload), FLAG×N writes stills,
  validate 0 errors.
- 2026-07-07 (claude) **Design elevation pass.** Rebuilt `style.css` and the
  `index.html` structure into a "capture console" look (kept every DOM id the JS
  contract depends on; still zero external assets — AC7): gradient brand mark +
  inline aperture/gear SVGs, LIVE + instrument-count HUD pills floating on the
  video, a "no signal" veil that shows when the feed goes stale/dead, a
  full-frame capture-flash on every flag, a custom filled confidence slider, and
  refined cards/buttons/toast. `app.js` gained small guarded hooks for these
  (slider `--fill`, HUD count mirror, live-badge + no-signal sync, capture
  flash) — all null-guarded so the tested behaviour is untouched.
