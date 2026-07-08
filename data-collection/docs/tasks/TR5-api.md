# TR5 — Recording API + state machine

status: done
depends-on: TR1, TR2, TR3, TR4
blocks: TR6, TR7
spec: [RECORDING.md](../RECORDING.md) §API & state, §Thresholds, §Detector sharing, §Encoder

## Goal

Wire the recording pieces into the FastAPI app: the `idle → recording →
processing → idle` state machine (`processing → failed → processing` on retry),
the `/record/*` + `/keyframe` endpoints, the `X-Frame-Number` header on `/frame`,
and the `recording_state` addition to `/status` — all **additive**, leaving every
image-mode endpoint's observable behaviour unchanged. `/record/stop` runs the
post-pass on a worker thread (blocking work off the event loop), pausing live
inference for its duration so the post-pass owns the detector.

## In scope

- `backend/app.py` — **edit**: new endpoints, the recording controller/state,
  `X-Frame-Number` on `/frame`, `recording_state` in `/status`, storing
  `output_path` on `/settings` success. (The `/flag` unpack was already made
  6-field-safe by TR1's one-line carve-out — do not re-touch it.)
- `backend/main.py` — **edit**: `--capture-fps` (default 30) and
  `--mining-threshold` (default 0.25) CLI args, threaded into `app.state`.
- `tests/test_recording_api.py` — **new** test file (do not edit T05's
  `tests/test_api.py`; it is the image-mode regression gate — keep it green).

## Out of scope

Encoder internals (TR2), the post-pass loop (TR4), `VideoEntryWriter` (TR3), the
capture split (TR1) — consume their frozen interfaces. Frontend (TR6).

## Dependencies consumed (frozen, from sibling tasks)

- TR1: `capture.start_recording(encoder)`, `capture.stop_recording() -> int`,
  `capture.frames_written`, `capture.is_recording`, `capture.pause_inference()`,
  `capture.resume_inference()`, `Latest.frame_number`.
- TR2: `open_encoder(path, fps, frame_size, ...) -> VideoEncoder`,
  `probe_video(path) -> VideoProbe`.
- TR3: `VideoEntryWriter` (via TR4).
- TR4: `PostPassJob(entry_dir, entry_name, detector, *, keyframes, frame_count,
  mining_threshold, operator_threshold, model_version, ...)`, `.run()`,
  `.status -> PostPassStatus`.

## Public interface (frozen — TR6 builds against this)

`create_app(detector, writer_factory, capture, validate_fn=...)` signature is
**unchanged** (recording deps are constructed inside from `capture` + `detector`
+ new `app.state` config; `open_encoder`/`PostPassJob`/`probe_video` are imported,
injectable via `app.state` for tests). New `app.state`: `output_path` (str|None,
set on `/settings`), `capture_fps` (float, default 30), `mining_threshold` (float,
default 0.25), a `recording` controller, `recording_lock`.

| method · path | request | response |
|---|---|---|
| `POST /record/start` | `{"entry_name": str}` | `200 {"ok": true, "entry_name", "operator_threshold"}` · `409` collision or wrong state or no `output_path` · `422` invalid `entry_name` |
| `POST /keyframe` | `{"frame_number": int}` | `200 {"n_keyframes": int}` · `409` not recording · `422` `frame_number` < 0 or ≥ `frames_written` |
| `POST /record/stop` | — | `200 {"ok": true, "frames_written": int}` · `409` not recording |
| `POST /record/discard` | — | `200 {"ok": true}` · `409` wrong state |
| `POST /record/retry` | — | `200 {"ok": true}` · `409` not in `failed` |
| `GET /record/status` | — | `200 {"state","entry_name","frames_written","n_keyframes","postpass":{"done","total"},"error"}` |
| `GET /frame?after=N` | — | unchanged **+ `X-Frame-Number` header** when recording (the on-screen frame's `frame_number`); header omitted when idle |
| `POST /flag` | (unchanged) | unchanged **+ `409` while `recording`** ("stop recording to snapshot stills") |
| `GET /status` | — | unchanged fields **+ `"recording_state"`** |

State machine (guard every transition under `recording_lock`):

```
idle       --/record/start-->  recording
recording  --/record/stop--->  processing   (kick post-pass on a worker thread)
recording  --/record/discard-> idle         (abort: stop+release encoder, rm folder)
processing --job success----->  idle
processing --job failure----->  failed
failed     --/record/retry--->  processing   (fresh PostPassJob from the MP4)
failed     --/record/discard-> idle         (reject the take: rm folder)
processing --/record/discard-> idle         (reject a completed take: rm folder)
/record/start while recording|processing -> 409
```

## Acceptance criteria

TestClient + T01 fakes. Real `CaptureLoop`+`FakeCapture` where a live loop
matters; a deterministic capture stub where a thread would make assertions racy.
No real encoder/ffmpeg (inject a `FakeEncoder` via `app.state`), no real
post-pass video (inject a fake `PostPassJob` / fake video reader).

- **AC1 (plain def for blocking I/O).** `/keyframe`, `/record/start`,
  `/record/stop`, `/record/discard`, `/record/retry`, `/flag` are plain `def`
  (`not inspect.iscoroutinefunction`) — folder mint, encoder open, and the thread
  kick stay off the event loop.
- **AC2 (start).** `/record/start {entry_name}` on `idle` with `output_path`
  configured: mints `<output_path>/<entry_name>/video/`, opens the encoder
  (`open_encoder`, fps=`capture_fps`, frame_size from the current snapshot frame
  dims), freezes the operator threshold = `detector.confidence_threshold` now,
  calls `capture.start_recording(encoder)`, state → `recording`; response echoes
  the frozen `operator_threshold`.
- **AC3 (start guards).** `/record/start` → `409` on an existing
  `<output_path>/<entry_name>`, `409` when already `recording`/`processing`,
  `409` when no `output_path` set, `409` when the capture has produced **no frame
  yet** (frame_size for the encoder comes from the current snapshot — no snapshot,
  no start; never an unhandled `AttributeError`); `422` on an invalid `entry_name`
  (same rule as `dataset_name`: single path component, no leading `.`).
- **AC4 (keyframe).** `/keyframe {frame_number}` while `recording` appends to a
  deduped set (a repeat of the same frame_number does not grow `n_keyframes`) and
  returns the new count; `409` when not recording; `422` when `frame_number` < 0
  or ≥ `capture.frames_written` (beyond frames written).
- **AC5 (X-Frame-Number).** While recording, `/frame` returns an `X-Frame-Number`
  header equal to the on-screen snapshot's `frame_number`; a test with a numbered
  fake camera asserts the value the client would echo to `/keyframe` is the
  displayed frame's index (spec AC3). Header absent when idle.
- **AC6 (stop kicks post-pass, off the loop).** `/record/stop` on `recording`
  returns immediately with `frames_written`, transitions to `processing`, calls
  `capture.pause_inference()`, and runs a `PostPassJob` on a worker thread; when
  the job finishes the state moves to `idle` (success) or `failed`, and
  `capture.resume_inference()` is called in both cases. `409` if not recording.
- **AC7 (record/status).** `GET /record/status` reflects the live machine:
  `state`, `entry_name`, `frames_written` (from capture during recording),
  `n_keyframes`, `postpass.{done,total}` (from the job during processing),
  `error` (set only in `failed`).
- **AC8 (flag blocked while recording).** `POST /flag` during `recording` → `409`
  with a "stop recording to snapshot stills" detail; image-mode `/flag` is
  otherwise unchanged and still passes T05's `tests/test_api.py`.
- **AC9 (idle-mode regression — the hard gate).** T05's entire
  `tests/test_api.py` passes unchanged: `/flag`, `/discard`, `/settings`,
  `/confidence`, `/validate`, `/status`, `/stream`, `/frame` behave exactly as
  before when no recording is active. `/status` gains `recording_state`
  (== `"idle"` when not recording) without dropping any existing field.
- **AC10 (discard).** `/record/discard` from `recording` stops+releases the
  encoder and deletes the entry folder (state → idle); from `failed` or
  `processing`-completed deletes the folder (reject the take); `409` from `idle`.
- **AC11 (retry).** `/record/retry` is valid only from `failed`: it constructs a
  fresh `PostPassJob` from the on-disk MP4 + keyframe list and re-runs it on a
  worker thread (state → `processing`); `409` from any other state.
- **AC12 (one recording at a time).** Concurrent `/record/start` requests yield
  exactly one `recording` and a `409` for the loser (transitions serialized on
  `recording_lock`).

## Log

- 2026-07-08 — Brief created (recording-mode decomposition of RECORDING.md, task
  cut T-R5).
- 2026-07-08 — **Done** via `/blind-tdd`. `backend/app.py` (recording state
  machine + /record/* + /keyframe endpoints; recording touches on /frame, /flag,
  /status), `backend/main.py` (--capture-fps / --mining-threshold CLI args →
  app.state), `tests/test_recording_api.py` (29 tests, all 12 ACs). Blind coder
  ran on **Opus** (concurrency-heavy). Phase-1.5 test review caught a missing
  transition (start-while-`processing` → 409) + strengthened the stop_recording
  return-contract; added start-while-`failed`/retry-while-`processing` guards.
  **Opus code review found one ❌ concurrency blocker**: discard-during-processing
  left `inference_paused` inconsistent (orphaned post-pass worker resumed the
  detector mid-way through a *later* take → §Detector-sharing violation). Fixed by
  tying resume to job ownership — invariant: `resume_inference()` fires **exactly
  once** per processing episode (worker iff `rec.job is job` at completion, else
  discard when it takes ownership away; an orphaned worker never resumes).

  > **Follow-up bug surfaced during TR6 R4 manual pass (2026-07-08, claude) —
  > NOT yet fixed.** The resume-once invariant above holds, but the orphaned
  > worker is never *cancelled*: `PostPassJob.run()` (TR4) has no cooperative-
  > cancel check in its `for frame_number in range(...)` loop, and
  > `/record/discard` during `processing` only clears `rec.job`, resumes
  > inference, and `rmtree`s the folder. The still-running worker therefore keeps
  > calling `detector.predict` **concurrently with the just-resumed live
  > inference** (a §Detector-sharing violation) for the full remaining duration,
  > and its `writer.add_frame` → `cv2.imwrite` **re-creates a stray partial
  > folder** (`<entry>/images/<entry>_fNNNNNN.jpg`) *after* the rmtree deleted it.
  > Reproduced on disk: two discard-during-processing takes each left exactly one
  > stray keyframe JPEG. On the CPU-bound detector (~0.6 fps) a discarded
  > 970-frame take's worker runs ~24 min post-discard at halved live FPS. Not
  > caught by any AC (TR7 AC5 exercises process-kill, not discard). Suggested
  > fix: add a `cancel`/`cancelled` flag to `PostPassJob`, check it at the top of
  > each loop iteration (bail before predict/imwrite), and have `/record/discard`
  > set it before the rmtree. Owner decision needed (TR5 discard + TR4 job).

  review independently corroborated the lock area (noted GET-side `rec.state`
  reads outside the lock — harmless, read-only) but **stalled before a verdict**;
  consensus rested on Opus + the green test gate. Phase-4 fixes: (1) /record/start
  mints `<output>/<entry>/video/` itself (open_encoder is injectable, a fake won't
  mkdir); (2) AC5 racy-wait + a JPEG-round-trip decode flake fixed (escape hatch),
  and the ring-eviction race in the AC5 header re-lookup fixed by sizing the test's
  capture ring past its frame count. **AC9 carve-out**: T05's `test_ac7` key-set
  assertion gained `recording_state` (the one field AC9 mandates on /status) —
  mirrors TR1's sanctioned /flag carve-out; image-mode behavior otherwise
  unchanged. Full suite: 149 passed, stable across repeated + full-suite runs.
