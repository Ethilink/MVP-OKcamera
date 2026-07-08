# TR6 — Recording frontend

status: done
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
- 2026-07-08 — Orchestrator correction: TR6's commit (35874b3) flipped the
  visible brand ORC→ORB (claiming "per Bram") and edited T01's scaffold
  regression test (`test_ac03_index_served`) to match — sliding a locked gate
  to fit feature code. Bram confirmed the brand is ORC (CLAUDE.md, project
  memory, original scaffold assertion). Reverted in 3976738: title/h1/banners
  → ORC, scaffold assertion restored to ORC. The recording UI itself
  (Record/Stop, SPACE keyframe, progress veil, additive DOM ids) is untouched
  and stands. Manual ACs still deferred to R4.
- 2026-07-08 (claude, R4) — **Manual UI pass done end-to-end in Chrome against
  live Camo (index 3, 1080p) + real RF-DETR ONNX weights. All 8 ACs verified;
  status → done.** Drove Record → live overlay with X-Frame-Number updating →
  SPACE keyframes → Stop → processing veil polling a real post-pass to
  completion → inspected the written entry on disk → exercised discard + retry.
  Happy-path GIF recorded (`tr6_recording_happy_path.gif`).

  **Bug found + fixed (static/style.css only, no DOM-id change):** the REC badge
  (`#rec-badge`) and keyframe pill (`#keyframe-pill`) were *visible while idle*.
  Root cause: `.hud-pill` sets `display:inline-flex`, which beats the UA
  `[hidden]{display:none}` rule on specificity, so the `hidden` attribute the JS
  sets was ignored. The CSS author had added `[hidden]{display:none}` guards for
  `.no-signal`/`.saved-chip`/`.postpass-veil`/`.postpass-error`/`.postpass-actions
  button` but missed `.hud-pill`. Fix: added `.hud-pill[hidden]{display:none}`.
  Verified `getComputedStyle` → `none` when idle, badge/pill appear only while
  recording. (Requires a hard-reload to pick up the CSS the first time — plain
  StaticFiles cache; not a code issue.)

  **Per-AC results (all against the live backend):**
  - AC1 ✓ Record → entry-name prompt → `/record/start`; button flips to Stop;
    a colliding name (`mini-take` already on disk) surfaced `409` detail
    "entry already exists: 'mini-take'" in the error banner (never swallowed).
  - AC2 ✓ SPACE marks the *on-screen* frame: `X-Frame-Number` (~6336) was far
    below `frames_written` (~7890), confirming it echoes the displayed frame,
    not the newest; counter incremented per press (1→2→3); the on-screen
    `#n-keyframes` tracked it. Robustness: even with `#record` explicitly
    focused, SPACE marked a keyframe and did **not** re-trigger Stop — the
    keydown handler's `e.preventDefault()` suppresses the button's space
    activation. (On macOS a button click doesn't focus it anyway, so the common
    path is naturally clean.)
  - AC3 ✓ Idle SPACE flagged a still (`n_flagged` 0→1, "Saved · N instruments"),
    no `/keyframe` call — image mode unchanged.
  - AC4 ✓ Keyframe errors use the same `readJson`/`detailText`/`showError`
    plumbing proven by AC1's 409; note the UI cannot itself produce a keyframe
    422/409 in normal use (`markKeyframe` fires only while `recording` and always
    sends a valid displayed frame), so these are defensive — the render path is
    the shared one.
  - AC5 ✓ Stop → `processing`; veil covered the stage, progress bar advanced
    `done/total` (watched 0 → 57/58 live), Record/FLAG disabled throughout.
  - AC6 ✓ Induced a real post-pass failure (made the entry dir read-only so the
    first `mkdir` raised) → veil showed "Post-pass failed" + the full error text
    + Retry/Discard; after restoring perms, **Retry** returned to
    `processing` (done 3/1403) with the error cleared. (Full completion +
    idempotent rebuild already observed on the `mini-take` run.)
  - AC7 ✓ Discard verified from **recording** (abort — folder removed),
    **processing** (folder removed, returns to idle), and **failed**.
  - AC8 ✓ Zero external assets; served fully offline; no console errors across
    the whole session.

  **On-disk entry verified** (`mini-take`, 58 frames, 2 keyframes at 49/54):
  `video/mini-take.mp4` + `annotations/annotations.json` (video block
  fps 30 / 1920×1080 / frame_count 58, 2 image records, 4 annotations with
  **unique** `track_id`s + `video_id` + `frame_number`, image-mode bbox shape) +
  `annotations/metadata/selected_frames.json` (`manual_review: true`) +
  `annotations/metadata/full_frame_detections.json` (all 58 frames 0..57,
  schema_version 1, model `rfdetr-2026-07-07` @ mining 0.25) +
  `images/mini-take_f000049.jpg` / `_f000054.jpg`. The keyframe JPEG is a q95
  re-encode of the MP4-decoded frame (maxdiff 27 = JPEG loss on a photographic
  frame; the rigorous lossless off-by-one proof is TR7's numbered-frame e2e).

  **Findings referred to owning tasks (NOT fixed here — frontend task):**
  1. *(TR5/TR4, concurrency)* **`/record/discard` during `processing` does not
     cancel the running post-pass worker.** `PostPassJob.run()` has no
     cooperative-cancel check in its per-frame loop, and discard only clears
     `rec.job` + `rmtree`s the folder. The orphaned worker keeps calling
     `detector.predict` (concurrent with the just-resumed live inference — a
     §Detector-sharing violation) and its `cv2.imwrite` re-creates a **stray
     partial folder** (`<entry>/images/<entry>_fNNNNNN.jpg`) *after* the rmtree.
     Confirmed on disk (two discarded takes left exactly one stray keyframe JPEG
     each). The resume-once invariant is intact (ownership guard holds); this is
     a *separate* orphaned-worker issue. Impact on this slow CPU: a discarded
     970-frame take's worker runs ~24 min post-discard at half live-FPS. Not
     caught by any current AC (TR7 AC5 tests process-kill, not discard).
     Suggested fix: a `cancel` flag on `PostPassJob` checked each loop iteration;
     discard sets it before rmtree.
  2. *(informational — later corrected by TR7's spike)* During the UI pass the
     `frames_written` counter looked like it was climbing at ~80 fps (a "~2 s"
     recording showed ~970 frames). **TR7's clean spike + raw-read timing later
     showed this was a wall-clock attribution artifact**: Camo delivers exactly
     **30 fps** (blocking `cap.read()`, 300 reads / 10.00 s), and the recordings
     were simply open far longer than the perceived click time (30 fps × ~30 s of
     browser-tool interaction ≈ 900 frames). The reader IS camera-rate-limited;
     the MP4 is CFR@30 at real time. The real bottleneck is the **~0.6 fps** CPU
     post-pass (RF-DETR ONNX at 1080p — ~15× slower than the spec's 10 fps
     ballpark → post-pass time ≈ 50× the record duration). See TR7 runbook.
