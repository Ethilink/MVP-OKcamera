# TR4 — Post-pass job runner

status: done
depends-on: TR1 (test fakes: `tests/recording_fakes.py`), TR2, TR3
blocks: TR5
spec: [RECORDING.md](../RECORDING.md) §Post-pass (background job, after stop), §Post-pass specifics

## Goal

The offline job that, after `stop`, reopens the recorded MP4 and runs the
detector over **every** frame at the **mining threshold**, streaming each frame's
decoded pixels + detections into a `VideoEntryWriter` (TR3) which produces the
four artifacts. It reports progress, and its failure/retry story is load-bearing:
a job that dies leaves a **valid partial project**, and a **retry** rebuilds
everything from the MP4 + keyframe list — idempotent by construction (never
appends).

## In scope

`backend/postpass.py` + `tests/test_postpass.py`. The frame-iteration loop,
progress tracking, the `frame_count` reconciliation (writer counter vs probe),
the failure → partial + error-surfacing, and idempotent retry.

## Out of scope

- Threading it onto a worker / the state machine / detector pause-resume calls —
  TR5 owns those. `run()` is a plain blocking method TR5 calls on a thread and
  brackets with `capture.pause_inference()` / `resume_inference()`.
- Serialization + JPEG extraction (TR3's `VideoEntryWriter`), the encoder/probe
  (TR2). This task **consumes** `probe_video` and `VideoEntryWriter` via injected
  factories so it is testable with fakes.

## Public interface (frozen)

```python
# backend/postpass.py
import cv2
from backend.encoder import probe_video
from backend.video_writer import VideoEntryWriter

@dataclass
class PostPassStatus:
    state: str            # "pending" | "running" | "done" | "failed"
    done: int             # frames processed so far
    total: int            # authoritative frame_count (== frames_written)
    error: str | None

class PostPassJob:
    def __init__(
        self,
        entry_dir: Path,               # <output_path>/<entry_name>
        entry_name: str,
        detector,                      # .predict(frame, confidence_threshold=...)
        *,
        keyframes,                     # 0-based frame numbers marked while recording
        frame_count: int,             # AUTHORITATIVE — the recording's frames_written
        mining_threshold: float,       # sidecar / every-frame predict threshold (default 0.25)
        operator_threshold: float,     # frozen at /record/start; keyframe filter
        model_version: str,
        probe_fn=probe_video,          # inject a fake in tests
        cap_factory=cv2.VideoCapture,  # inject a numbered fake reader in tests
        video_writer_factory=VideoEntryWriter,
    ): ...

    def run(self) -> None:
        """Blocking. Open entry_dir/video/<entry_name>.mp4 via cap_factory; probe
        it via probe_fn; build the `video` dict — fps/width/height from the probe,
        frame_count = the AUTHORITATIVE constructor arg (assert probe.frame_count
        ≈ frame_count, prefer the counter on mismatch, import doc §7). Construct
        the VideoEntryWriter, call write_selected_frames() FIRST (partial safety),
        then for each frame 0..frame_count-1: predict at mining_threshold,
        add_frame(n, frame, dets), bump `done`. Finally finalize(). On any
        exception: state -> "failed", record the message in `error`, leave the
        partial project intact (selected_frames.json present, no annotations.json),
        and DO NOT re-raise past run() (the worker thread must not die silently —
        status carries the error). On success: state -> "done"."""

    @property
    def status(self) -> PostPassStatus: ...
```

## Acceptance criteria

No real ffmpeg/video/detector. Inject `cap_factory` (a fake reader yielding
`make_numbered_frames`, TR1's shared fake), `probe_fn` (returns a `VideoProbe`),
`FakeDetector` (T01), and either the real `VideoEntryWriter` (TR3) or a recording
double.

- **AC1 (every frame processed).** `run()` calls `detector.predict` exactly
  `frame_count` times with `confidence_threshold == mining_threshold`, and
  `add_frame` is called once per frame_number `0..frame_count-1` in order. With
  the real `VideoEntryWriter`, the resulting `full_frame_detections.json` has all
  `frame_count` frames (spec AC4).
- **AC2 (keyframe outputs).** Keyframe frame_numbers get a JPEG + COCO records
  filtered at `operator_threshold`; non-keyframes contribute only to the sidecar.
  (Delegated to TR3, but assert end-to-end with the real writer for one keyframe.)
- **AC3 (progress).** `status.done` advances from 0 toward `status.total ==
  frame_count` during the run; `status.state` goes `pending → running → done`.
- **AC4 (frame_count reconciliation).** When `probe_fn` reports a frame_count that
  differs from the constructor's authoritative `frame_count` (re-encode gotcha),
  the `video` block written uses the **authoritative** value, and the run still
  processes exactly `frame_count` frames. State it in the Log which way the
  ≈-assert tolerates (e.g. warn, prefer counter).
- **AC5 (failure → partial).** If `detector.predict` (or `add_frame`) raises
  mid-run, `run()` does not propagate; `status.state == "failed"`,
  `status.error` carries the message, `selected_frames.json` exists, and
  `annotations.json` does **not** — a valid partial project.
- **AC6 (idempotent retry — spec AC7).** After a failed run, constructing a fresh
  `PostPassJob` with the same args and calling `run()` to completion yields a
  result **identical** to an uninterrupted run: byte-identical `annotations.json`
  and `full_frame_detections.json`, same JPEGs — nothing appended or duplicated
  from the aborted attempt.
- **AC7 (detector-only interaction).** `run()` touches the detector solely through
  `predict(...)` (no pause/resume calls inside — TR5 brackets those); assert no
  attribute of the detector other than `predict` is invoked.

## Log

- 2026-07-08 — Brief created (recording-mode decomposition of RECORDING.md, task
  cut T-R4).
- 2026-07-08 — **Done** via `/blind-tdd`. `backend/postpass.py` +
  `tests/test_postpass.py` (8 tests, all 7 ACs). Blind boundary held: test-writer
  (Sonnet) and coder (Sonnet) never saw each other's output; Codex + Opus reviewed
  code vs spec (consensus, zero ❌) before tests ran. Phase-1.5 test review caught a
  gate hole (AC2 confidences couldn't distinguish an operator↔mining threshold
  swap) — fixed by seeding a detection in the (mining, operator) gap; also added
  gates for the MP4 path handed to `cap_factory`/`probe_fn` and the sidecar `model`
  block (`conf_threshold`=mining, `version`=model_version). One blind fix in Phase 4
  (pass `str(mp4_path)` to `probe_fn` to match `cap_factory`). **frame_count
  reconciliation**: probe value is read but the `video` block and the frame loop
  both use the AUTHORITATIVE constructor `frame_count` unconditionally — "always
  prefer the counter", no hard assert, so a re-encode-style probe mismatch never
  crashes the job. Full suite: 120 passed (112 prior + 8 new).
