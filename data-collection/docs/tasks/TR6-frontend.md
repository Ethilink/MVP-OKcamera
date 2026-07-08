# TR6 — Recording frontend

status: todo
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
- **AC2** While recording, SPACE marks a keyframe (`/keyframe` with the on-screen
  `X-Frame-Number`, not the newest) and the keyframe counter increments; auto-repeat
  does not machine-gun; SPACE while typing in a settings field does not mark.
- **AC3** While idle, SPACE still flags a still exactly as before (image-mode
  regression) — no keyframe call is made.
- **AC4** `POST /keyframe` `422` (out-of-range) / `409` (not recording) is shown
  as a readable error, never swallowed.
- **AC5** Stop → the UI enters processing and shows a progress bar advancing
  `done/total` from `/record/status`; controls that start a new recording are
  disabled until the job finishes.
- **AC6** A failed post-pass shows the error + a working Retry; Retry returns to
  the processing/progress view.
- **AC7** Discard is available from recording (abort) and after a completed/failed
  job (reject take) and calls `/record/discard`.
- **AC8** Zero external assets — works offline (AC7 of T06 preserved).

## Log

- 2026-07-08 — Brief created (recording-mode decomposition of RECORDING.md, task
  cut T-R6).
