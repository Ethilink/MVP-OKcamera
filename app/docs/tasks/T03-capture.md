# T03 — Capture-infer loop + overlay

status: todo
depends-on: T01
blocks: T04
spec: [DESIGN.md](../DESIGN.md) D5, D6, §Threading rules

## Goal

The single background thread that owns camera + tracker: read a 1080p frame,
`tracker.update()`, draw the overlay, publish an atomic `Latest` snapshot,
invoke `on_frame` — never crashing, never tight-spinning, never serving stale
frames. This deliberately adapts the data-collection capture pattern
(BUFFERSIZE=1, staleness, atomic rebind) **minus** all 4K/two-resolution
machinery — single 1920×1080 pipeline (D5).

## In scope

`backend/capture.py`, `backend/render.py` + `tests/test_capture.py`,
`tests/test_render.py`. Tests use T01's `FakeCaptureSource` + `ScenarioTracker`
(injected `cap_factory` — no real camera). Process: **blind-TDD**.

## Out of scope

`Session` (T04 wires `on_frame` → `session.observe`), HTTP, JPEG streaming
pacing.

## Public interface (frozen)

```python
# backend/render.py
def render(frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
    """Draw the overlay IN PLACE on `frame` and return it: boxes + masks
    coloured by tracker_id, label 'Instrument {tracker_id}', on-table count
    burned into a corner. Copy ownership: the CAPTURE LOOP passes a frame it
    owns (a copy of the camera buffer if that buffer may be reused), so render
    is free to mutate. render does NOT copy — single owner, no double-copy."""

# backend/capture.py
class Latest(NamedTuple):
    overlay_jpeg: bytes            # encoded 1080p overlay for /stream
    present_ids: frozenset[int]    # tracker_ids — BUILTIN int (cast from the
                                   # tracker's np.int64) so /status JSON-encodes;
                                   # T04 must not have to sanitize (see AC12)
    count: int                     # len(present_ids)
    t: float                       # monotonic seconds at capture

OnFrame = Callable[[float, frozenset[int]], None]   # (t, present_ids)

class CaptureLoop:
    def __init__(self, tracker, camera_index: int,
                 on_frame: OnFrame | None = None,
                 cap_factory: Callable[[int], "VideoCaptureLike"] = cv2.VideoCapture,
                 # called as cap_factory(camera_index) -> obj with
                 # isOpened/read/set/get/release. Tests inject
                 # `lambda _idx: FakeCaptureSource()` (the index is ignored by
                 # the fake) — do NOT pass FakeCaptureSource as the factory
                 # directly, or camera_index would be read as its `size` arg.
                 frame_size: tuple[int, int] = (1920, 1080),
                 stale_after_s: float = 2.0,
                 render_fn=render) -> None: ...
    def reset_tracker(self, timeout_s: float = 2.0) -> None: ...
    # Schedule tracker.reset() ON THE CAPTURE THREAD (the only thread allowed to
    # touch the tracker, per DESIGN §Threading) and block until applied, so the
    # NEXT published frame comes from a reset tracker. T04's /recording/start
    # calls this before session.start(). No-op safe if called before start().
    # Raises TimeoutError if not applied within timeout_s (capture thread hung
    # in a stalled read) — T04 maps that to 503, so a stalled camera can't
    # deadlock a Start handler.
    def start(self) -> None        # RuntimeError if capture won't open
    def stop(self) -> None
    def snapshot(self) -> Latest | None   # atomic read under the lock
    @property
    def health(self) -> str
    # One rule: let age = now - (last publication time, or start() time if
    # nothing has published yet).
    # "ok"   — thread alive AND age <= stale_after_s
    # "stale"— thread alive AND age >  stale_after_s (camera not returning
    #          frames OR tracker failing every tick — same rule covers both;
    #          also covers "never got a first frame" once the grace elapses,
    #          so a normal boot reads "ok" for its first stale_after_s)
    # "dead" — thread has exited (after stop(), or an abnormal exit)
    @property
    def generation(self) -> int    # increments per PUBLISHED Latest
```

## Acceptance criteria

- **AC1** On start, sets `CAP_PROP_FRAME_WIDTH/HEIGHT` to `frame_size` and
  `CAP_PROP_BUFFERSIZE=1` (assert via `FakeCaptureSource` records).
- **AC2** The tracker receives frames in capture order from ONE thread; each
  published `Latest.present_ids` equals the tracker_ids of that frame's dets.
- **AC3** `render_fn` receives a copy — mutating its input does not alter any
  array the tracker or later frames see.
- **AC4** Each **published** tick rebinds a new `Latest`; `generation` strictly
  increases per published frame; `snapshot()` is `None` only before the first
  published frame. A tick that produces no publication (see AC7) does NOT bump
  `generation`.
- **AC5** `overlay_jpeg` decodes (`cv2.imdecode`) to `frame_size` dimensions.
- **AC6** No publication for > `stale_after_s` (failed reads OR tracker
  exceptions every tick) flips `health` to `"stale"` without killing the thread
  or busy-spinning — the loop caps its retry rate at **≤ 30 attempts/s**;
  recovery (a fresh publication) flips back to `"ok"`.
- **AC7** A tracker exception on one frame is caught + logged and that tick is
  **skipped** (no new `Latest`, no `generation` bump, no `on_frame` call); the
  loop continues and — **given at least one prior publication** — an isolated
  blip leaves `health == "ok"` (a sustained run of exceptions surfaces as
  `"stale"` once age exceeds `stale_after_s`, per AC6). An abnormal thread exit
  → `"dead"`.
- **AC8** `on_frame(t, present_ids)` fires exactly once per published frame
  with strictly increasing `t`; an `on_frame` exception doesn't kill the loop.
- **AC9** `stop()` joins the thread ≤ 2 s, releases the capture, and leaves
  `health == "dead"`.
- **AC10** `render` mutates the frame where dets exist (compare against a
  pre-render copy — pixels changed), returns the same array (same size/dtype),
  and two dets with different `tracker_id` get different colours.
- **AC11** `reset_tracker()` invokes `tracker.reset()` exactly once, on the
  capture thread (never the caller's thread), and returns only after it has been
  applied so the next published frame reflects the reset; calling it before
  `start()` is a safe no-op (must not raise); a capture thread stuck in `read()`
  past `timeout_s` → `TimeoutError` (assert with a blocking fake read).
- **AC12** Every element of `Latest.present_ids` is a builtin `int`
  (`type(x) is int`, not `np.int64`), so a raw `json.dumps` of a status dict
  built from it round-trips — T04 does no numpy sanitizing (ties to T04 AC3).

## Log

- (append dated one-liners here)
