"""H.264 MP4 encoder wrapper + video probe (TR2).

`open_encoder` tries `cv2.VideoWriter` with the `avc1` fourcc (VideoToolbox
hardware H.264 on macOS) first, falling back to piping raw BGR frames to an
`ffmpeg -c:v h264_videotoolbox` subprocess when `avc1` is unavailable in the
installed OpenCV build. `probe_video` reads back a finished file's real
`(fps, frame_count, width, height)` — the post-pass must not trust these
blindly (see RECORDING.md §Post-pass specifics).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoProbe:
    fps: float
    frame_count: int
    width: int
    height: int


class VideoEncoder:
    """An opened H.264 MP4 encoder. Construct via ``open_encoder``. Write frames
    in capture order, then ``release()`` before the file is reopened for the
    post-pass."""

    def __init__(self, path: Path, *, backend: str, writer=None, process=None):
        self._path = path
        self._backend = backend
        self._writer = writer
        self._process = process
        self._frames_written = 0
        self._is_open = True

    def write(self, frame: np.ndarray) -> None:
        if self._backend == "avc1":
            self._writer.write(frame)
        else:
            self._process.stdin.write(frame.tobytes())
        self._frames_written += 1

    def release(self) -> None:
        if not self._is_open:
            return
        if self._backend == "avc1":
            self._writer.release()
        else:
            self._process.stdin.close()
            self._process.wait()
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def path(self) -> Path:
        return self._path

    @property
    def frames_written(self) -> int:
        return self._frames_written


def _open_ffmpeg(path: Path, fps: float, width: int, height: int, ffmpeg_factory):
    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "h264_videotoolbox",
        str(path),
    ]
    try:
        if ffmpeg_factory is None:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            process = ffmpeg_factory(cmd)
    except Exception:
        return None
    # A launch failure that doesn't raise still needs to surface as failure —
    # no writable stdin means there is no pipe to feed frames into.
    if getattr(process, "stdin", None) is None:
        return None
    return process


def open_encoder(
    path,
    fps: float,
    frame_size,
    *,
    writer_factory=cv2.VideoWriter,
    ffmpeg_factory=None,
) -> VideoEncoder:
    """Open an H.264 MP4 writer at `path`, creating parent dirs. First tries
    `writer_factory(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)`
    and keeps it only if `isOpened()` is True. Otherwise falls back to an ffmpeg
    pipe (`-f rawvideo -pix_fmt bgr24 -s WxH -r fps -i - -c:v h264_videotoolbox`).
    RuntimeError (naming `path`) if neither backend opens."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frame_size

    writer = writer_factory(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, frame_size)
    if writer.isOpened():
        return VideoEncoder(path, backend="avc1", writer=writer)
    writer.release()

    process = _open_ffmpeg(path, fps, width, height, ffmpeg_factory)
    if process is not None:
        return VideoEncoder(path, backend="ffmpeg", process=process)

    raise RuntimeError(f"could not open video encoder for {path}: no avc1 or ffmpeg backend available")


def probe_video(path, *, cap_factory=cv2.VideoCapture) -> VideoProbe:
    """Open the finished file and read CAP_PROP_FPS / FRAME_COUNT / FRAME_WIDTH /
    FRAME_HEIGHT. Do NOT trust these blindly downstream (import doc §7): TR4
    asserts frame_count ≈ its own recording counter and prefers the counter on a
    re-encode-style mismatch. RuntimeError if the file will not open."""
    cap = cap_factory(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video for probing: {path}")

    probe = VideoProbe(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return probe
