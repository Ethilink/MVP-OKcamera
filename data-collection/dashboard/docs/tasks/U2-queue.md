# U2 — Idle-draining post-pass queue + pause/resume (blind-TDD SPEC)

Frozen SPEC cut from `docs/tasks/REDESIGN.md` §U2 (Endpoint contract table + 7
ACs), `CONTEXT.md` (terms), and `docs/adr/0001-idle-draining-postpass-queue.md`
(the post-pass model). Builds on U1 (merged): storage split
`images/<base>/` + `videos/<base>_NNN/`; `/record/start {entry_base}` returns the
resolved `{entry_name}`.

Blind-TDD rules (`docs/tasks/BOARD.md`): the **coder** rewrites
`backend/postpass.py` (pause/resume additions) + the `backend/app.py` recording
state machine **without ever seeing `tests/`**. The **test-writer** owns all of
`tests/` and never sees the coder's implementation. Both consume only this SPEC.

---

## 1. What changes, in one paragraph

Today the recording machine is a **foreground blocking** pipeline:
`idle → recording → processing → idle` (`processing → failed → processing` on
retry). One post-pass at a time; `/record/start` 409s while `processing`/`failed`.
U2 **replaces** that with an **orthogonal background queue**: the foreground
state collapses to `idle ↔ recording` only, and post-pass work becomes a **FIFO
queue drained by a single background worker that runs iff `state == idle` and the
queue is non-empty**. `/record/stop` finalizes the MP4, **enqueues** the job, and
returns to `idle` immediately (never blocks). Starting a new recording **pauses**
the in-flight drain job between frames, hands the single shared detector back to
the live overlay, and the job **resumes from its paused frame index** when the
tool returns to idle. `processing` and `failed` cease to be foreground states —
drain progress and errors are reported under a new `drain` block.

---

## 2. Frozen `GET /record/status` shape (U3 builds against THIS — do not deviate)

```jsonc
{
  "state": "idle" | "recording",          // FOREGROUND mode only
  "drain": {
    "current": { "entry_name": str, "done": int, "total": int } | null,
    "queued": [ "<entry_name>", ... ],      // entries waiting behind current, FIFO order
    "eta_seconds": number                    // remaining frames across all jobs ÷ detect-fps
  },
  "error": string | null                    // set iff the current (head) drain job has FAILED
}
```

- `state` is the **foreground** mode. Draining is a background concern reported
  under `drain`; a status of `idle` with a non-null `drain.current` is the normal
  "recording done, post-pass grinding" situation.
- `drain.current` is the job the worker is actively draining **or** the failed
  head that has halted the queue (see §5). `null` when the queue is empty and
  nothing is mid-drain.
- `drain.queued` lists the `entry_name`s waiting behind `current`, in FIFO order.
- `error` is non-null **only** when `drain.current` is a failed head (halt-on-
  failure, §5). It mirrors the failed job's error string.

> The pre-U2 fields `entry_name`, `frames_written`, `n_keyframes`, `postpass`
> at the top level of `/record/status` are **removed**. `state` no longer takes
> the values `processing`/`failed`.

Everything else in the frozen Endpoint contract table (REDESIGN §"Endpoint
contract") is authoritative for the other endpoints; the deltas U2 must deliver:

| endpoint | U2 behavior |
|---|---|
| `POST /record/stop` | finalize MP4, **enqueue** the post-pass job, return `{ok, frames_written}` immediately with `state == idle`. Never blocks; never a `processing` state. |
| `POST /record/start` | while a job is mid-drain, **pause** it (within one frame) and reclaim the detector before recording begins. 409 only if already `recording` (not for a draining queue). Otherwise unchanged from U1 (422 bad base, 400 unwritable, `{entry_name}` resolved). |
| `POST /record/discard` | recording → abort (as today). Otherwise → cancel + drop the **current** drain job (the one in `drain.current`, failed or not), delete its Entry folder, and advance the queue. 409 from a fully idle machine (nothing recording, nothing draining). |
| `POST /record/retry {entry_name}` | valid only when `drain.current` is a failed head whose `entry_name` matches. Clears the error and re-drains that Entry from frame 0 (idempotent rebuild from the MP4). 409 otherwise. |
| `POST /keyframe`, `POST /flag` | unchanged from TR5/U1 (`/flag` still 409s while `state == recording`). |
| `GET /status` | unchanged except `recording_state` continues to report `state` (`idle`/`recording` only now). |

---

## 3. Load-bearing invariants (test these)

**INV-1 (foreground states).** `state ∈ {idle, recording}` at all times. The
strings `processing` and `failed` never appear as `state`.

**INV-2 (drain only when idle).** The worker predicts/writes frames **only** while
`state == idle`. While `state == recording`, no drain frame is processed (the
current job, if any, is paused at its frame index).

**INV-3 (single detector owner).** At no instant do the live capture loop and the
drain worker both call `detector.predict`. Enforce with a real lock/flag around
detector access, not two detector instances. Mechanism: the capture loop already
skips inference and publishes the raw frame when `capture.inference_paused` is
True (`backend/capture.py`); the drain worker sets `capture.pause_inference()`
while it owns the detector and `capture.resume_inference()` when it yields or
finishes. `/record/start` must not begin recording until the worker has yielded
the detector (bounded wait), so a recording tick's predict can never race a drain
predict.

**INV-4 (pause within one frame, resume from index).** `/record/start` while a
job is mid-drain causes the worker to stop before its next per-frame
read/predict/write (cooperative flag checked between frames), release the
detector, and **park** the job at its current frame index. When the tool next
returns to idle with that job still current, the worker resumes it **from the
paused index** (not frame 0) and it completes with output identical to an
uninterrupted run.

**INV-5 (no data loss on failure — HARD).** A failed, paused, interrupted, or
killed drain job **never deletes its Entry folder**. The MP4 is finalized at
`/record/stop` before enqueue, so the raw recording survives independent of the
post-pass. A failed job leaves a valid **partial** Entry (video +
`selected_frames.json`, no `annotations.json`) that Retry rebuilds from the MP4.
The **only** path that deletes an Entry folder is an explicit operator
`/record/discard`.

**INV-6 (FIFO, one at a time).** The worker drains queued jobs strictly in
enqueue order, one at a time, using the shared detector.

---

## 4. Drain worker model (implementation guidance — coder owns the shape)

A **single** long-lived daemon drain-worker thread, created inside `create_app`
(must work under a bare `TestClient(app)` with no lifespan/`with` block — do NOT
gate the worker on FastAPI startup events). It waits on a `threading.Condition`
and does nothing until signalled. The recording machine's fields live on
`app.state.recording` under `app.state.recording_lock`; suggested shape:

- `queue: list` — FIFO of enqueued `PostPassJob`s not yet started. Each job is
  built at `/record/stop` time via `app.state.post_pass_factory` (preserve this
  seam — tests inject fakes through it), so it carries its own `frame_count`
  (== `total`).
- `current` — the job being drained, or the failed head, or `None`.
- `error: str | None` — the failed head's error (mirrors `current.status.error`).
- The worker loop: wait until `state == idle` and (`current` is a resumable
  paused job **or** `queue` non-empty and no failed head blocking). Pop/resume →
  set as `current` → `capture.pause_inference()` → run the job's frames with
  per-frame checks of a pause flag and a cancel flag. On normal completion:
  `capture.resume_inference()` (if the queue is now empty/idle), clear `current`,
  advance. On failure: leave the job as `current`, set `error`, resume inference,
  and **stop advancing** (halt, §5). On pause request: release detector
  (`resume_inference`), park, signal `/record/start` it may proceed.

`/record/start` sets the pause flag + notifies the condition, then waits
(bounded) on an Event the worker sets once it has yielded the detector, before
opening the encoder / calling `capture.start_recording`.

`/record/discard` of a live/failed `current` cancels the job (cooperative
`job.cancel()`, as today) and waits (bounded, lock released) until the worker has
stopped touching that Entry's folder, **then** `rmtree`s it — the existing
"no stray folder after rmtree" guarantee (see `tests/test_discard_cancel.py`)
must survive the single-worker refactor.

### PostPassJob additions

`backend/postpass.py` `PostPassJob` gains **pause/resume**, symmetric to its
existing `cancel()`:

- `pause()` — request the run to yield before its next read/predict/write; when
  honored, `run()` **releases the detector** (does not call `predict`) and parks
  (blocks on a resume Event) at the current frame index, leaving `status.done`
  at the last fully-written frame.
- `resume()` — un-park; the run continues its loop from the parked index.
- `status.state` gains `"paused"` while parked. Existing states
  (`pending/running/done/failed/cancelled`) and the `cancel()` semantics (bail
  before read/predict and again before the disk write; a cancel landing
  mid-predict drops the result before the write) are **unchanged** — the TR4
  contract in `tests/test_postpass.py` still holds, including AC7 (`run()` touches
  the detector solely through `.predict`; pause/resume must not access any other
  detector attribute from inside `run()` — the yield/reclaim of the *live*
  detector is the caller's job via `capture.pause_inference/resume_inference`).
- Resume-from-index must produce output **byte-identical** to an uninterrupted
  run (the writer is deterministic; resuming continues `add_frame` at the parked
  frame_number). Retry-after-failure still rebuilds from frame 0 (fresh job).

---

## 5. Halt-on-failure semantics (decided 2026-07-09 with Bram)

When a drain job **fails** mid-run, the queue **halts on it** (head-of-line):

- The failed job **stays** as `drain.current` (with `done < total`), top-level
  `error` is set to its message, and the worker **stops advancing** — queued jobs
  wait untouched (their MP4s are safe on disk, INV-5).
- The operator resolves it via the chip: **Retry** (`/record/retry {entry_name}`
  matching `drain.current.entry_name`) re-drains that Entry from frame 0; on
  success the queue advances. **Discard** (`/record/discard`) drops the failed
  head, deletes its folder (explicit operator choice — the one sanctioned delete),
  and advances the queue.
- Because the worker halts on the first failure, **at most one** entry is ever in
  the failed state at a time — which is exactly what the single top-level `error`
  + single `drain.current` frozen shape can faithfully report. (This resolves the
  retry-by-`entry_name` vs single-`error` tension in the frozen contract in favor
  of the status shape.)

Rationale: post-pass failures should be rare (idempotent rebuild from MP4), so
stalling the batch on a failure the operator sees immediately on the chip is
acceptable and keeps the queue model simple. **No dataset is ever lost** (INV-5).

---

## 6. `eta_seconds` (AC5)

`eta_seconds = remaining_frames / detect_fps`, where
`remaining_frames = (current.total − current.done) + Σ queued[i].total` and
`detect_fps = app.state.detect_fps` (a plain settable float; default `0.6` per
the ADR). The worker MAY update `app.state.detect_fps` as a running EMA of
observed per-frame detect durations, but `/record/status` MUST compute eta from
the current `app.state.detect_fps` value so a test can inject a known fps and
assert the formula deterministically. `eta_seconds == 0` when nothing is queued
or draining (`drain.current is None and not drain.queued`). Guard div-by-zero
(`detect_fps` floored to a small epsilon).

---

## 7. Acceptance criteria (restated with observable assertions)

**AC1 — stop enqueues, returns to idle immediately.** After `/record/start` →
`/record/stop`, the response is `{ok, frames_written}` and `state` is `idle`
right away (never `processing`); a subsequent `/record/start` is **accepted**
(200) even while the just-enqueued job is still draining (contrast the old 409).

**AC2 — idle FIFO drain.** With the tool idle and ≥1 job queued, the worker drains
them one at a time in enqueue order using the shared detector; `drain.current`
walks the queue front-to-back, `drain.queued` shrinks accordingly, each Entry
becomes a completed video project (`annotations.json` present).

**AC3 — pause within one frame, reclaim detector, resume from index.** With a job
mid-drain at frame `k`, `/record/start` pauses it before frame `k+1`
(`drain.current.done` stops advancing), the live overlay regains the detector
(recording proceeds; INV-3 holds), and after `/record/stop` returns to idle the
same job resumes **from `k`** and completes; the produced artifacts are identical
to an uninterrupted run.

**AC4 — partial + retry idempotent.** A paused/interrupted/killed job leaves a
valid partial Entry (video + `selected_frames.json`, no `annotations.json`);
`/record/retry {entry_name}` completes it and the result is byte-identical to an
uninterrupted run (rebuild from MP4). INV-5: the folder is never auto-deleted.

**AC5 — status shape.** `GET /record/status` returns exactly the §2 shape:
`state`, `drain.{current,queued,eta_seconds}`, `error`. `eta_seconds` follows §6.

**AC6 — single detector owner.** The drain worker and the live overlay never touch
the detector concurrently (INV-3) — provable with a detector double that flags
re-entrant/concurrent `predict` calls, or by asserting `capture.inference_paused`
is True exactly while the worker is predicting and False while recording. Enforced
by a lock/flag, not two detector instances.

**AC7 — regressions green + new concurrency tests.** All prior recording/image
tests stay green after migration (§8); new tests cover pause/resume and FIFO
queue ordering with a fake detector (no hardware), using bounded `_wait_for`
polls — never sleep-and-hope.

---

## 8. Legacy tests that encode the SUPERSEDED blocking contract (must migrate)

The test-writer migrates these to the new `{state, drain, error}` contract; the
coder never sees them. Grepped for `processing` / `record/status` / the
`state`-machine assertions:

- **`tests/test_recording_api.py`** — the heaviest. Rewrites needed:
  - `_state(client)` reads `["state"]` — still valid, but assertions of
    `"processing"`/`"failed"` become assertions on `drain.current` / `error`.
  - `test_ac3_start_409_when_processing`, `..._when_failed` — **invert**: a start
    while draining is now **200** (AC1), pausing the drain job (AC3). Replace with
    tests that start-while-draining pauses rather than rejects.
  - `test_ac6_*`, `test_ac7_*` — migrate the `processing`/`postpass:{done,total}`
    assertions to `drain.current:{entry_name,done,total}` + `error`.
  - `test_ac10_discard_from_processing_*`, `test_ac11_retry_*` — retarget to the
    drain-current/failed-head semantics (§5); retry now takes `{entry_name}`.
  - `_FakeJob` needs enrichment: per-frame progress + `pause()/resume()` +
    `status.state == "paused"`, so pause/resume/queue-ordering are observable.
    Keep the frozen constructor shape and the `post_pass_factory` seam.
  - `RecordingStubCapture` already exposes `pause_inference/resume_inference`
    (call counters) — reuse for INV-3 assertions.
- **`tests/test_discard_cancel.py`** — the cancel + no-stray-folder regression
  must survive the single-worker refactor; migrate its `state == "idle"` /
  `pause_calls`/`resume_calls` assertions to the new drain lifecycle (the resume
  count is now tied to the worker yielding, not a per-job worker completing).
- **`tests/test_recording_e2e.py`** (long-term U4-owned, but U2 breaks it) —
  `_wait_for(... ["state"] != "processing")` and `state == "idle"/"failed"`
  assertions migrate to `drain`/`error`; the kill→partial→retry flow (AC5 there)
  maps onto U2 AC4. Keep it green.
- **`tests/test_u1_storage.py`** — `test_ac2_*` waits `state == "idle"` after stop
  (now immediate) and expects `videos/X_001` to persist; adapt the `_FakeJob`
  release/drain handshake to the queue worker. Storage ACs themselves are U1 and
  must stay green.
- **`tests/test_postpass.py`** (TR4) — must stay green as-is (pause/resume are
  additive; `cancel`/`run`/AC7 unchanged). New unit tests for `pause()/resume()`
  and resume-from-index are additive.

Green baseline before U2: **164** `uv run pytest`. Target: green after migration
(+ new AC tests).

---

## 9. Owned / out of scope

**Owned by U2:** `backend/postpass.py` (pause/resume additions), the
`backend/app.py` recording state machine + `/record/*` + `/record/status`, and
all of `tests/` needed to migrate + cover the ACs.

**Out of scope:** frontend (`static/*` — U3), docs prose (RECORDING.md /
DASHBOARD.md — U4), the CoreML spike (S1). Discarding an arbitrary **queued**
(non-current) Entry by name is **not** in U2 — `/record/discard` targets the
current/head drain job only (the chip's Discard control). The `writer_factory`,
`open_encoder`, `probe_video`, `post_pass_factory`, and capture
`pause_inference/resume_inference` seams are **preserved** so every existing fake
keeps working.
