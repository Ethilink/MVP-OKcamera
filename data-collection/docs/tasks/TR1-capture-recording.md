# TR1 — Reader/encoder split + `Latest.frame_number`

status: todo
depends-on: —
blocks: TR4 (test fakes), TR5, TR7
spec: [RECORDING.md](../RECORDING.md) §Runtime — two threads while recording, §The frame-number contract, §Detector sharing

## Goal

Extend the idle-only `CaptureLoop` into a loop that can **record**: while
recording, a reader/encoder path owns the camera and writes **every** frame to an
injected encoder at full camera rate, stamping each frame with the 0-based index
**as it writes it into the encoder** (writer index == MP4 index — the load-bearing
invariant), and hands the newest `(frame, frame_number)` to the inference thread,
which publishes a `Latest` carrying that `frame_number`. When not recording the
loop keeps today's exact observable behaviour. Also add the `pause_inference` /
`resume_inference` hooks the post-pass uses to take sole ownership of the detector.

## In scope

- `backend/capture.py` — the reader/encoder split, `Latest.frame_number`,
  `start_recording` / `stop_recording` / `frames_written` / `is_recording`,
  `pause_inference` / `resume_inference` / `inference_paused`.
- `tests/test_capture_recording.py` — **new** file (do not touch T03's
  `tests/test_capture.py`; it is the idle-mode regression gate — keep it green).
- `tests/recording_fakes.py` — **new** shared test module: `FakeEncoder`
  (records written frames; `is_open`; `release()`) and `make_numbered_frames(n,
  w, h)` (deterministic frames whose pixels encode their index, so off-by-one is
  detectable). TR4/TR5/TR7 import these — this task owns the file.
- `backend/app.py` — **ONE line only** (coordinated carve-out from TR5, which
  has not started — TR5 depends on this task): `/flag`'s 5-name tuple unpack of
  `Latest` becomes attribute access (`snap.frame`, `snap.dets`, …). Without it,
  the 6-field `Latest` makes the unpack raise `ValueError` → image-mode `/flag`
  500s and T05's `tests/test_api.py` goes red the moment this task lands. That
  suite passing unchanged is part of this task's AC1 gate. Touch nothing else
  in `app.py`.

## Out of scope

- The real encoder wrapper (`open_encoder`, `probe_video`) — TR2. This task
  consumes an **injected** encoder object (duck-typed `.write`/`.release`/`.is_open`);
  tests inject `FakeEncoder`.
- The post-pass, the API/state machine, the frontend (TR4/TR5/TR6).
- Where the encoder file lives / how it is opened — TR5 opens it and passes it in.

## Public interface (frozen)

```python
# backend/capture.py

class Latest(NamedTuple):          # was a 5-tuple; gains frame_number (default keeps
    frame: np.ndarray              #   peripheral 5-arg constructions valid)
    dets: "sv.Detections"
    threshold: float
    overlay_jpeg: bytes
    count: int
    frame_number: int | None = None   # NEW — the MP4 write index this frame was
                                       # detected on while recording; None when idle

class CaptureLoop:
    # __init__ signature UNCHANGED from T03 (same params, same defaults).

    def start_recording(self, encoder) -> None:
        """Enter recording mode. The reader path writes EVERY frame to `encoder`
        (`encoder.write(frame)`) in capture order, assigning frame_number = the
        0-based index as written (frame 0 is the first written frame), and hands
        the newest (frame, frame_number) to inference. `encoder` must already be
        open (`.is_open` True). RuntimeError if already recording."""

    def stop_recording(self) -> int:
        """Leave recording mode. Stop writing, then `encoder.release()` BEFORE
        returning (so the post-pass can reopen the file), and return the total
        number of frames written — the authoritative frame_count. The release
        happens only once the reader has stopped writing (no in-flight write).
        RuntimeError if not recording."""

    def pause_inference(self) -> None:
        """Stop calling detector.predict in the loop so the post-pass owns the
        detector solely (spec §Detector sharing). The stream keeps serving the
        last overlay / raw frames; health is unaffected. Idempotent."""

    def resume_inference(self) -> None:
        """Undo pause_inference. Idempotent."""

    @property
    def is_recording(self) -> bool: ...

    @property
    def frames_written(self) -> int:
        """Frames written to the encoder so far this recording; 0 when idle."""

    @property
    def inference_paused(self) -> bool: ...
```

Encoder duck-type this task depends on (frozen in TR2, mimicked by `FakeEncoder`):
`encoder.write(frame: np.ndarray) -> None`, `encoder.release() -> None`,
`encoder.is_open: bool`.

## Acceptance criteria

Fakes only — **no real camera, no real encoder**. Use `FakeCapture` (T01),
`FakeDetector` (T01), and `FakeEncoder` / `make_numbered_frames` (this task).

- **AC1 (idle regression — the hard gate).** With no recording, observable
  behaviour is unchanged: `snapshot()`, `snapshot_with_generation()`,
  `snapshot_at()`, `health`, `generation`, `set_camera()`, the ring buffer, and
  the start-up warmup all behave exactly as T03's `tests/test_capture.py` asserts
  (that suite must stay green untouched), **and** T05's `tests/test_api.py` stays
  green (the one-line `/flag` unpack carve-out above is what keeps it green).
  `Latest.frame_number` is `None` when idle.
- **AC2 (frame-number contract).** Feed `make_numbered_frames`; `start_recording`;
  the encoder receives frames in order with **no gaps or dupes**, and the i-th
  frame handed to `encoder.write` carries frame_number == i (0-based). Assert the
  encoder's recorded frames are exactly frames 0..N-1 in order.
- **AC3.** `Latest.frame_number` published while recording equals the frame_number
  of the frame the inference thread actually detected on (the value stamped by the
  reader when it wrote that frame), **not** a newest-at-publish counter — prove by
  making inference skip frames (slow `FakeDetector`) and asserting the published
  frame_number matches the frame's own index, not the reader's latest.
- **AC4.** Inference still skips freely: the reader writes every frame (AC2) while
  inference detects a subset; `frames_written` advances at reader rate and only
  displayed frames become selectable — assert `frames_written` > count of distinct
  published frame_numbers under a slow detector.
- **AC5.** `stop_recording()` returns the exact `frames_written`, calls
  `encoder.release()` exactly once, and does so only after the last `write`
  (assert order via `FakeEncoder` event log). After stop, `is_recording` is False,
  `frames_written` reads 0, and the loop is back to idle behaviour (AC1 holds again).
- **AC6.** `start_recording` while already recording → `RuntimeError`;
  `stop_recording` while not recording → `RuntimeError`.
- **AC7 (detector pause).** After `pause_inference()`, the loop makes **no**
  `detector.predict` calls (assert `FakeDetector.seen_frames` stops growing) while
  the reader stays alive and `health` stays `"ok"`; `resume_inference()` restores
  predict calls. Works in idle mode (post-pass runs between takes).
- **AC8 (encoder failure is contained).** If `encoder.write` raises, the loop does
  not crash the process / wedge the camera health silently — surface it so
  `stop_recording`/status can report it (state it in the Log which mechanism you
  chose: propagate on stop, or a recording-error flag). Idle mode never touches
  the encoder, so a broken encoder can't affect idle behaviour.
- **AC9.** The reader owns the camera during recording: exactly one thread calls
  `cap.read()` at a time (no double-read races) — assert the reader and inference
  do not both read from `FakeCapture` concurrently (e.g. read count is consistent
  with single-reader semantics).

## Log

- 2026-07-08 — Brief created (recording-mode decomposition of RECORDING.md, task
  cut T-R1).
