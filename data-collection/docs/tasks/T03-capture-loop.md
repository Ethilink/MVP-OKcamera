# T03 — Capture-infer loop

status: done
depends-on: T01
blocks: T05
spec: [DASHBOARD.md](../DASHBOARD.md) §Runtime, §Threading rules, §Failure-mode guardrails

## Goal

The single background thread that owns the camera: read one native-1080p frame,
predict, render overlay, and publish an atomic `latest` snapshot + heartbeat —
never crashing, never tight-spinning, never serving stale frames.

## In scope

`backend/capture.py` + `tests/test_capture.py`. Uses T01's `FakeCapture` /
`FakeDetector` — **no real camera in tests** (injected `cap_factory`).

## Out of scope

Dataset writing, HTTP, what `render_fn` draws (T04 — inject a stub).

## Public interface (frozen)

```python
# backend/capture.py
class Latest(NamedTuple):          # spec's 5-tuple, same order
    frame: np.ndarray              # native 1080p BGR — the frame /flag saves
    dets: "sv.Detections"          # frame coords (unscaled)
    threshold: float               # value at predict time
    overlay_jpeg: bytes            # encoded 1080p overlay for /stream
    count: int                     # len(dets), UI-only

class CaptureLoop:
    def __init__(self, detector, camera_index: int, render_fn,
                 cap_factory=cv2.VideoCapture,
                 target_size=(1920, 1080), stale_after_s=2.0,
                 start_read_attempts=15, reopen_after_s=3.0): ...
    def start(self) -> None       # warms up to start_read_attempts reads; RuntimeError (naming the camera index) only if no valid frame arrives at all — resolution-agnostic
    def stop(self) -> None
    def set_camera(self, index: int) -> None    # loop reopens via cap_factory
    def snapshot(self) -> Latest | None          # atomic read under the lock
    @property
    def health(self) -> str        # "ok" | "stale" | "dead"
    @property
    def generation(self) -> int    # increments per published Latest (for /stream pacing)
```

## Acceptance criteria

- **AC1** On start, sets `CAP_PROP_FRAME_WIDTH=1920`, `CAP_PROP_FRAME_HEIGHT=1080`,
  `CAP_PROP_BUFFERSIZE=1` on the capture (assert via `FakeCapture` records).
- **AC2** `start()` warms up to `start_read_attempts` reads and raises
  `RuntimeError` naming the camera index **only if no valid frame arrives at all**
  — the pipeline is resolution-agnostic, no exact-resolution assert.
- **AC3** The detector receives the frame as-is (native 1080p, no downscale);
  `snapshot().frame` is that same single-resolution array.
- **AC4** `threshold` in the snapshot is the detector's value **at predict time**:
  mutate `FakeDetector.confidence_threshold` between programmed frames and
  assert each `Latest` carries the value its dets were produced with.
- **AC5** `render_fn` gets a **copy** of the frame: mutating what render
  receives does not alter any array reachable from `snapshot()`.
- **AC6** Each tick **rebinds** a new `Latest` (successive snapshots are
  different objects; `generation` strictly increases).
- **AC7** Failed reads: `read() → (False, None)` for >`stale_after_s` flips
  `health` to `"stale"`, the thread stays alive, and it does not busy-spin
  (bounded read-call count per second). On sustained failure (>`reopen_after_s`)
  the loop reopens the camera via `cap_factory` (self-heal / reconnect); recovery
  flips `health` back to `"ok"`.
- **AC8** `stop()` joins the thread within 2 s; `health` becomes `"dead"` if the
  thread exits abnormally.
- **AC9** `set_camera(n)` reopens through `cap_factory` with the new index
  (old capture released).
- **AC10** `overlay_jpeg` decodes (cv2.imdecode) to the frame's 1080p dimensions.

## Log

- 2026-07-07 — Built `backend/capture.py` via blind-TDD (SPEC at
  `backend/SPEC-capture.md`). Test-writer + coder ran blind in parallel; each
  caught its own bug via self-checks (test-writer: two timing-flaky tests → poll
  `generation`/`snapshot()` not tick-index; coder: don't seed `_pending_camera_index`
  with the startup index). Dual review vs SPEC found threading issues — both
  reviewers independently flagged the **`_pending_camera_index` lost-update race**
  (read+clear split across two lock acquisitions) as a ❌. Codex raised 3 more.
  Fixed all four: (1) read+clear pending index in ONE critical section; (2)
  `start()` uses `getattr(frame,"shape",None)` so a malformed successful read →
  RuntimeError not AttributeError; (3) `_dead` written/read under the lock; (4)
  `stop()` releases the cap only once the thread has stopped. Opus re-review:
  zero remaining ❌ (conforms). Codex re-confirmation pending (hit its own session
  limit mid-review, resets 19:10) — but every Codex-raised blocker was concretely
  fixed and independently verified by Opus + the test gate. `uv run pytest` =
  12/12 for T03 (AC1–AC10, no flakiness over repeated runs), 44/44 whole suite.
  Done.
