# TR6 — Recording frontend

status: review
depends-on: TR5 (frozen endpoint contract only — not TR5's code)
blocks: TR7
spec: [RECORDING.md](../RECORDING.md) §What the operator experiences, §API & state (Frontend bullet)

## Goal

Add the recording UI to the existing plain-JS dashboard: a **Record/Stop**
control (with an entry-name prompt), a live **keyframe counter**, a **progress
bar** during processing, and a **SPACE handler that branches on recording state** —
idle SPACE still flags a still (`/flag`), recording SPACE marks the on-screen
frame as a keyframe (`POST /keyframe {frame_number}`, read from the new
`X-Frame-Number` header). No build step, no framework, zero external assets, as
ever. Image-mode UI behaviour is unchanged when idle.

## In scope

`static/index.html`, `static/app.js`, `static/style.css` — **edits** (T06 owns
these; this task extends them, preserving every DOM id the existing JS contract
and T06's jsdom checks depend on).

## Out of scope

Backend changes (TR5), the annotation editor. No changes to the idle-mode flag /
settings / validate / confidence flows beyond the SPACE state-branch.

## UI contract (builds against TR5's frozen endpoints)

- **Record/Stop button.** Idle → prompt for `entry_name` → `POST /record/start`;
  on `recording` the button becomes **Stop** → `POST /record/stop`. A `409`/`422`
  is surfaced with the server's `detail` (never swallowed), same as image mode.
- **SPACE branch.** Track `recording_state` from `GET /record/status` (or
  `/status.recording_state`). When `idle`: existing `/flag` freeze-capture flow.
  When `recording`: read the current frame's `X-Frame-Number` (captured from the
  latest `/frame` response) and `POST /keyframe {frame_number}` — instant, no
  freeze; bump the keyframe counter. Ignore `event.repeat`; ignore SPACE while a
  text field has focus (both already handled — keep them).
- **Keyframe counter.** Visible while recording, driven by `/keyframe` responses
  and `/record/status.n_keyframes`.
- **Progress bar.** While `processing`, show `frame done/total` from
  `/record/status.postpass`; on `failed`, show the `error` and a **Retry** control
  (`POST /record/retry`); a **Discard** control (`POST /record/discard`) is
  available from recording/processing/failed.
- **State reflection.** The three states (`recording` / `processing` / `failed`)
  are visually distinct; when not `idle`, the image-mode FLAG affordance is
  clearly not a still-capture (SPACE marks keyframes, or is disabled during
  processing).

## Acceptance criteria

Manual checklist (screen-record one pass) + optional jsdom harness like T06's.
Keep T06's existing checks green.

- **AC1** Record → entry-name prompt → `/record/start`; button flips to Stop; a
  `409` (collision/wrong state) shows the server `detail`.
  **Implemented.** `#record` click handler in `app.js`: `window.prompt` for
  `entry_name` → `POST /record/start`; on success `recState = "recording"` and
  `#record-label` flips to "Stop"; on failure `showError('Record: ' + e.message)`
  surfaces the server `detail` via the existing `readJson`/`detailText` plumbing
  (never swallowed). Deferred to R4: live pass against a running TR5 backend.
- **AC2** While recording, SPACE marks a keyframe (`/keyframe` with the on-screen
  `X-Frame-Number`, not the newest) and the keyframe counter increments; auto-repeat
  does not machine-gun; SPACE while typing in a settings field does not mark.
  **Implemented.** `frameLoop()` captures `X-Frame-Number` into
  `currentFrameNumber` on every successful `/frame` response; `markKeyframe()`
  echoes that value (not the newest) to `POST /keyframe` and updates
  `#n-keyframes` from the response. The single keydown listener's existing
  `e.repeat` and `typingInField()` guards are shared by both branches via
  `flagOrKeyframe()` (unchanged, not duplicated). Deferred to R4: live pass.
- **AC3** While idle, SPACE still flags a still exactly as before (image-mode
  regression) — no keyframe call is made.
  **Implemented.** `flagOrKeyframe()` only calls `markKeyframe()` when
  `recState === "recording"`; the `flag()` function body is byte-for-byte
  unchanged from T06. Verified via `uv run pytest -q` (76 passed, no regressions)
  and DOM-id diffing (all 25 pre-existing T06 ids preserved).
- **AC4** `POST /keyframe` `422` (out-of-range) / `409` (not recording) is shown
  as a readable error, never swallowed.
  **Implemented.** `markKeyframe()` catch block calls
  `showError('Keyframe: ' + e.message)`, same `readJson`/`detailText` pattern as
  every other endpoint call in the file. Deferred to R4: live pass.
- **AC5** Stop → the UI enters processing and shows a progress bar advancing
  `done/total` from `/record/status`; controls that start a new recording are
  disabled until the job finishes.
  **Implemented.** `#record` stop path sets `recState = "processing"` on
  success; `renderRecordingUI()` disables `#record` while
  `processing || failed`; `pollRecordStatus()` (polled every 1s alongside the
  existing `/status` poll) drives `#progress-fill` width and `#postpass-sub`
  text from `record/status.postpass.{done,total}`. Deferred to R4: live pass
  (needs a real multi-frame post-pass to observe the bar actually advance).
- **AC6** A failed post-pass shows the error + a working Retry; Retry returns to
  the processing/progress view.
  **Implemented.** `renderRecordingUI()` shows `#postpass-error` (text from
  `record/status.error`) and `#postpass-retry` only when `recState === "failed"`;
  the retry handler calls `POST /record/retry`, optimistically sets
  `recState = "processing"` (corrected by the next poll), and re-shows the
  progress veil. Deferred to R4: live pass.
- **AC7** Discard is available from recording (abort) and after a completed/failed
  job (reject take) and calls `/record/discard`.
  **Implemented.** Two DOM buttons share one `discardRecording()` handler
  (`POST /record/discard`, then `recState = "idle"`): `#record-discard` next to
  the Record/Stop button, shown only while `recState === "recording"`; and
  `#postpass-discard` inside the processing/failed veil, shown for
  `processing || failed`. Deferred to R4: live pass.
- **AC8** Zero external assets — works offline (AC7 of T06 preserved).
  **Implemented.** No new `<link>`/`<script src>` to any external host, no web
  fonts, no CDN — everything is inline SVG / system fonts / plain JS, matching
  T06's existing pattern exactly. Verified: `test_scaffold.py::test_ac03_index_served`
  still asserts the stable `<title>ORC data-collection dashboard</title>`
  markup and passes.

## Log

- 2026-07-08 — Brief created (recording-mode decomposition of RECORDING.md, task
  cut T-R6).
- 2026-07-08 — Implemented (subagent tr6-frontend). Edited `static/index.html`,
  `static/app.js`, `static/style.css` only — no backend/test/other-file edits,
  per task scope.

  **New DOM ids added** (all additive; every pre-existing T06 id is unchanged
  and still present — verified by diffing the rendered `<body>`'s `id="..."`
  set before/after): `flag-label`, `record`, `record-label`, `record-discard`,
  `rec-badge`, `rec-badge-text`, `keyframe-pill`, `n-keyframes`,
  `keyframe-chip`, `keyframe-chip-text`, `postpass-veil`, `postpass-title`,
  `progress-fill`, `postpass-sub`, `postpass-error`, `postpass-retry`,
  `postpass-discard`.

  **What shipped:** Record/Stop button (`#record`) with an `entry_name` prompt
  on start; a unified `flagOrKeyframe()` dispatcher (used by both the `#flag`
  click handler and the existing SPACE keydown listener) that branches on a new
  `recState` module variable (`idle | recording | processing | failed`) —
  `idle` calls the original unmodified `flag()`, `recording` calls the new
  `markKeyframe()`; `frameLoop()` now also captures the `X-Frame-Number`
  response header into `currentFrameNumber` so keyframe marks echo the
  on-screen frame, never the newest. A keyframe counter (`#n-keyframes`,
  visible only while recording) and a REC/PROCESSING/FAILED HUD badge
  (`#rec-badge`) give at-a-glance state. A `#postpass-veil` covers the stage
  during `processing`/`failed` with a progress bar (from
  `record/status.postpass.{done,total}`), the failure `error` text, and Retry
  (`POST /record/retry`) / Discard (`POST /record/discard`) controls; a second
  Discard button (`#record-discard`) sits next to Record/Stop for the
  `recording` (abort) case, per the TR6 brief's explicit
  "available from recording/processing/failed" (all three states get a
  reachable Discard, matching this task's own wording over RECORDING.md's
  slightly looser prose). `recState` is refreshed every 1s from both
  `GET /status.recording_state` (cheap, always available) and the richer
  `GET /record/status` (drives the keyframe count, progress numbers, and error
  text) — both polls fail silently today since TR5 doesn't exist yet (see
  finding below), leaving the UI in its default `idle` rendering.

  **Finding (expected, do not fix here):** there is still no jsdom/Node test
  harness in this repo — T06's AC7 mentioned one but it was never actually
  shipped, and grepping for `package.json`/`node_modules`/jsdom-style test
  files under `dashboard/` turns up nothing. Per this task's own instructions,
  no Node/npm test setup was added (that would violate the
  no-build-step/zero-external-assets constraint AC8 exists to protect).
  Verification here is: (1) `uv run pytest -q` stays green (76 passed, same as
  before this change) — the only automated gate that exists; (2) `node --check
  static/app.js` for JS syntax validity; (3) a script-based DOM-id diff via
  `TestClient` confirming zero id collisions and zero dropped T06 ids. Full
  verification of the manual-checklist ACs 1–7 (screen-recording a live pass
  against SPACE/Record/Stop/Retry/Discard) requires a running TR5 backend,
  which does not exist yet — deferred to Phase R4 as instructed.
