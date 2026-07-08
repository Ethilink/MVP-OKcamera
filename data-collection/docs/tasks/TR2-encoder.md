# TR2 — H.264 encoder wrapper + video probe

status: todo
depends-on: —
blocks: TR4, TR5
spec: [RECORDING.md](../RECORDING.md) §Encoder (settled, with fallback), §Post-pass specifics (`video` block values), §FPS — 30 vs 60

## Goal

A small module that opens a genuine **H.264 MP4** writer — first choice
`cv2.VideoWriter` with the `avc1` fourcc (VideoToolbox hardware H.264 on macOS),
falling back to piping raw BGR frames to an `ffmpeg -c:v h264_videotoolbox`
subprocess when `avc1` is unavailable in the installed OpenCV build — and a
`probe_video` helper that reads back the finished file's real `(fps, frame_count,
width, height)`. Both are the seam the recording loop (TR1) and the post-pass
(TR4) build on; neither must require a real camera or real encoder in unit tests.

## In scope

`backend/encoder.py` + `tests/test_encoder.py`. All the encoder-selection logic
(`avc1` probe → `isOpened()` check → ffmpeg fallback → fail loudly if neither
opens), a `write`/`release` wrapper that is safe to call from the reader thread,
and `probe_video`.

## Out of scope

The capture loop that drives the encoder (TR1 — consumes the frozen interface via
an injected object), the post-pass (TR4 — calls `probe_video`), the API (TR5 —
calls `open_encoder`). No frame writing to the loop, no COCO, no HTTP.

## Public interface (frozen)

```python
# backend/encoder.py
from pathlib import Path
import cv2

@dataclass(frozen=True)
class VideoProbe:
    fps: float
    frame_count: int
    width: int
    height: int

class VideoEncoder:
    """An opened H.264 MP4 encoder. Construct via `open_encoder`. Write frames in
    capture order, then `release()` before the file is reopened for the post-pass."""
    def write(self, frame: np.ndarray) -> None: ...
    def release(self) -> None: ...              # idempotent; flushes + closes the file
    @property
    def is_open(self) -> bool: ...
    @property
    def backend(self) -> str: ...               # "avc1" | "ffmpeg"
    @property
    def path(self) -> Path: ...
    @property
    def frames_written(self) -> int: ...

def open_encoder(
    path,                                        # str | Path — output .mp4
    fps: float,
    frame_size,                                  # (W, H)
    *,
    writer_factory=cv2.VideoWriter,              # inject a fake in tests
    ffmpeg_factory=None,                         # None -> real subprocess; inject a fake in tests
) -> VideoEncoder:
    """Open an H.264 MP4 writer at `path`, creating parent dirs. First tries
    `writer_factory(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)`
    and keeps it only if `isOpened()` is True. Otherwise falls back to an ffmpeg
    pipe (`-f rawvideo -pix_fmt bgr24 -s WxH -r fps -i - -c:v h264_videotoolbox`).
    RuntimeError (naming `path`) if neither backend opens."""

def probe_video(path, *, cap_factory=cv2.VideoCapture) -> VideoProbe:
    """Open the finished file and read CAP_PROP_FPS / FRAME_COUNT / FRAME_WIDTH /
    FRAME_HEIGHT. Do NOT trust these blindly downstream (import doc §7): TR4
    asserts frame_count ≈ its own recording counter and prefers the counter on a
    re-encode-style mismatch. RuntimeError if the file will not open."""
```

## Acceptance criteria

**No real VideoToolbox / ffmpeg in unit tests** — inject `writer_factory` /
`ffmpeg_factory` / `cap_factory`. A `FakeWriter` exposing `isOpened()` /
`write()` / `release()` and a `FakeFfmpeg` recording stdin bytes are fine to
define locally in `tests/test_encoder.py`.

- **AC1 (avc1 happy path).** With a `writer_factory` whose `FakeWriter.isOpened()`
  is True, `open_encoder` returns a `VideoEncoder` with `backend == "avc1"`; the
  ffmpeg factory is **never** called; the requested fourcc is `avc1`, fps and
  `(W, H)` are passed through unchanged.
- **AC2 (fallback).** With a `writer_factory` whose `FakeWriter.isOpened()` is
  False, `open_encoder` releases that dead writer and returns a `VideoEncoder`
  with `backend == "ffmpeg"`, having launched the ffmpeg factory with a command
  that contains `h264_videotoolbox`, the correct `WxH`, and the fps.
- **AC3 (both fail).** `isOpened()` False **and** the ffmpeg factory raising /
  reporting failure → `open_encoder` raises `RuntimeError` naming `path`; no
  half-open handle is leaked (the dead writer was released).
- **AC4 (write routing).** `write(frame)` forwards to the underlying writer
  (`FakeWriter.write` records the array) or the ffmpeg stdin (`FakeFfmpeg`
  records the raw BGR bytes, length == `W*H*3`); `frames_written` increments once
  per `write`.
- **AC5 (release).** `release()` closes the active backend (VideoWriter.release,
  or ffmpeg stdin close + process wait), is **idempotent** (second call is a
  no-op, no raise), and flips `is_open` to False. For the ffmpeg backend, release
  waits for the subprocess so the moov atom is finalized before the file is read.
- **AC6 (probe).** `probe_video` with a `cap_factory` returning programmed
  CAP_PROP_* values yields a `VideoProbe` with those exact `fps / frame_count /
  width / height` (frame_count and dims as `int`, fps as `float`); a
  won't-open capture → `RuntimeError`.
- **AC7 (parent dirs).** `open_encoder` creates missing parent directories of
  `path` (the `<entry>/video/` dir may not exist yet).

## Log

- 2026-07-08 — Brief created (recording-mode decomposition of RECORDING.md, task
  cut T-R2).
