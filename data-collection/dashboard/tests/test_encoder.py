"""TR2 AC1-AC7: behavior contract for ``backend.encoder``.

Written from ``docs/tasks/TR2-encoder.md`` (the frozen SPEC) and
``docs/RECORDING.md`` §Encoder / §FPS. Blind-TDD: this file is written
against the SPEC's behavior contract, not against any particular
implementation — the coder replaces ``backend/encoder.py``'s Phase-0 stub
without ever seeing this file.

No real VideoToolbox / ffmpeg / VideoCapture — every backend collaborator is
injected via ``writer_factory`` / ``ffmpeg_factory`` / ``cap_factory``, using
the local fakes below:

- ``FakeWriter`` / ``FakeWriterFactory`` — stand in for ``cv2.VideoWriter`` /
  the injected ``writer_factory``. ``FakeWriter.isOpened()`` is programmable
  so a test can force the avc1-happy-path or the fallback/failure paths.
  ``FakeWriterFactory`` records every call's args so AC1 can assert the
  fourcc/fps/frame_size were passed through unchanged.
- ``FakeFfmpegProcess`` / ``FakeFfmpegFactory`` — stand in for a
  ``subprocess.Popen``-shaped ffmpeg handle / the injected ``ffmpeg_factory``.
  ``FakeFfmpegFactory(fail=True)`` raises instead of returning a process,
  simulating a launch failure (AC3). ``FakeFfmpegProcess.stdin`` is a
  ``_FakeStdin`` recording raw bytes written to it and whether it was closed.
- ``FakeCapture`` — stands in for ``cv2.VideoCapture`` / the injected
  ``cap_factory`` (AC6). Mirrors the real API's quirk of ``get()`` always
  returning ``float``, so a test catches an implementation that forgets to
  cast frame_count/width/height back to ``int``.

Frame content for AC4's write-routing tests is built by ``_make_frame`` — a
deterministic, non-constant BGR array (not all-zeros) so a bug that drops,
truncates, or zeroes bytes on the way to the writer/ffmpeg stdin is caught by
comparing against the exact bytes of the array that was passed in (computed
independently of the encoder, straight from the input).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.encoder import VideoProbe, open_encoder, probe_video

# --- Fakes: cv2.VideoWriter side -------------------------------------------


class FakeWriter:
    """Fake stand-in for a ``cv2.VideoWriter`` instance, as returned by an
    injected ``writer_factory``. ``isOpened()`` is programmable so a test can
    force either the avc1-happy-path (True) or the fallback/failure paths
    (False). Records every written frame (as a copy, so later in-place
    mutation of the caller's buffer can't retroactively corrupt the record)
    and counts ``release()`` calls so idempotency can be verified exactly.
    """

    def __init__(self, *, opened: bool) -> None:
        self._opened = opened
        self.frames: list[np.ndarray] = []
        self.release_calls = 0

    def isOpened(self) -> bool:
        return self._opened

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.release_calls += 1
        self._opened = False

    @property
    def released(self) -> bool:
        return self.release_calls > 0


class FakeWriterFactory:
    """Callable stand-in for the injected ``writer_factory`` (real default:
    ``cv2.VideoWriter``). Records every call's args (path, fourcc, fps,
    frame_size) and always returns the programmed ``FakeWriter``.
    """

    def __init__(self, writer: FakeWriter) -> None:
        self.writer = writer
        self.calls: list[tuple] = []

    def __call__(self, path, fourcc, fps, frame_size):
        self.calls.append((path, fourcc, fps, frame_size))
        return self.writer


# --- Fakes: ffmpeg side ------------------------------------------------------


class _FakeStdin:
    """Fake stand-in for an ffmpeg subprocess's ``stdin`` pipe. Records every
    chunk of raw bytes written to it, in order, and whether it was closed.
    """

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        if self.closed:
            raise ValueError("write to closed FakeFfmpeg stdin")
        self.chunks.append(bytes(data))

    def close(self) -> None:
        self.closed = True


class FakeFfmpegProcess:
    """Fake stand-in for the ``subprocess.Popen``-shaped handle an
    ``ffmpeg_factory`` returns on a successful launch. ``wait_calls`` counts
    calls so idempotent ``release()`` can be verified exactly (not just "did
    not raise").
    """

    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        return 0


class FakeFfmpegFactory:
    """Callable stand-in for the injected ``ffmpeg_factory`` (real default:
    ``None`` -> real subprocess). Records every command it was invoked with.
    When ``fail=True``, raises instead of returning a process — simulating
    ffmpeg failing to launch (AC3), the same way ``subprocess.Popen`` would
    raise if the binary were missing.
    """

    def __init__(self, *, process: FakeFfmpegProcess | None = None, fail: bool = False) -> None:
        self.process = process if process is not None else FakeFfmpegProcess()
        self.fail = fail
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]):
        self.calls.append(cmd)
        if self.fail:
            raise OSError("FakeFfmpegFactory: simulated ffmpeg launch failure")
        return self.process


def _fps_tokens(fps: float) -> set[str]:
    """Acceptable string forms of ``fps`` inside an ffmpeg argv token — either
    the plain float repr, or (when ``fps`` is a whole number) the bare int
    form, since the SPEC does not freeze which formatting the implementation
    uses for ``-r``.
    """
    tokens = {str(fps)}
    if fps == int(fps):
        tokens.add(str(int(fps)))
    return tokens


# --- Fakes: cv2.VideoCapture side (probe_video / AC6) -----------------------


class FakeCapture:
    """Fake stand-in for a ``cv2.VideoCapture`` instance, as returned by an
    injected ``cap_factory``. ``opened`` programs ``isOpened()``. Mirrors the
    real API's quirk that ``get()`` always returns ``float`` (even for
    integral properties like frame count/width/height), so a test that feeds
    float-valued props catches an implementation that forgets to cast back
    to ``int``.
    """

    def __init__(
        self,
        *,
        opened: bool,
        fps: float = 0.0,
        frame_count: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
    ) -> None:
        self._opened = opened
        self._props = {
            cv2.CAP_PROP_FPS: float(fps),
            cv2.CAP_PROP_FRAME_COUNT: float(frame_count),
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
        }
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop_id: int) -> float:
        return self._props[prop_id]

    def release(self) -> None:
        self.released = True


# --- Numbered/distinctive frame helper (AC4) --------------------------------


def _make_frame(width: int, height: int, offset: int = 0) -> np.ndarray:
    """A deterministic, non-constant BGR ``(height, width, 3)`` uint8 frame
    (not all-zeros/all-same-value), so a bug that drops, truncates, or zeroes
    bytes on the way to the writer/ffmpeg stdin is caught by comparing
    against the exact bytes of this array — computed independently of the
    encoder, straight from what was passed in.
    """
    count = height * width * 3
    return ((np.arange(count, dtype=np.uint32) + offset) % 256).astype(np.uint8).reshape(height, width, 3)


# --- AC1: avc1 happy path ----------------------------------------------------


def test_ac01_avc1_happy_path(tmp_path):
    path = tmp_path / "clip.mp4"
    width, height, fps = 64, 32, 30.0
    fake_writer = FakeWriter(opened=True)
    writer_factory = FakeWriterFactory(fake_writer)
    ffmpeg_factory = FakeFfmpegFactory()

    encoder = open_encoder(
        path, fps, (width, height), writer_factory=writer_factory, ffmpeg_factory=ffmpeg_factory
    )

    assert encoder.backend == "avc1"
    assert encoder.is_open is True
    assert ffmpeg_factory.calls == [], "ffmpeg factory must never be called on the avc1 happy path"

    assert len(writer_factory.calls) == 1, "writer_factory must be called exactly once"
    _path_arg, fourcc_arg, fps_arg, size_arg = writer_factory.calls[0]
    assert fourcc_arg == cv2.VideoWriter_fourcc(*"avc1")
    assert fps_arg == fps
    assert size_arg == (width, height)


# --- AC2: ffmpeg fallback ----------------------------------------------------


def test_ac02_ffmpeg_fallback(tmp_path):
    path = tmp_path / "clip.mp4"
    width, height, fps = 64, 32, 30.0
    dead_writer = FakeWriter(opened=False)
    writer_factory = FakeWriterFactory(dead_writer)
    ffmpeg_process = FakeFfmpegProcess()
    ffmpeg_factory = FakeFfmpegFactory(process=ffmpeg_process)

    encoder = open_encoder(
        path, fps, (width, height), writer_factory=writer_factory, ffmpeg_factory=ffmpeg_factory
    )

    assert encoder.backend == "ffmpeg"
    assert encoder.is_open is True
    assert dead_writer.released is True, "the dead avc1 writer must be released, not leaked"

    assert len(ffmpeg_factory.calls) == 1, "ffmpeg_factory must be launched exactly once"
    cmd = ffmpeg_factory.calls[0]
    assert isinstance(cmd, list)
    assert all(isinstance(token, str) for token in cmd)
    assert "h264_videotoolbox" in cmd
    assert f"{width}x{height}" in cmd
    assert _fps_tokens(fps) & set(cmd), f"no token in {cmd!r} matches fps={fps!r}"


# --- AC3: both backends fail --------------------------------------------------


def test_ac03_both_backends_fail_raises_runtime_error_naming_path(tmp_path):
    path = tmp_path / "clip.mp4"
    dead_writer = FakeWriter(opened=False)
    writer_factory = FakeWriterFactory(dead_writer)
    ffmpeg_factory = FakeFfmpegFactory(fail=True)

    with pytest.raises(RuntimeError) as exc_info:
        open_encoder(path, 30.0, (64, 32), writer_factory=writer_factory, ffmpeg_factory=ffmpeg_factory)

    assert str(path) in str(exc_info.value), "RuntimeError message must name the path"
    assert dead_writer.released is True, "the dead avc1 writer must not be leaked half-open"


# --- AC4: write routing -------------------------------------------------------


def test_ac04_write_routes_to_avc1_writer_and_counts_frames(tmp_path):
    path = tmp_path / "clip.mp4"
    width, height, fps = 64, 32, 30.0
    fake_writer = FakeWriter(opened=True)
    writer_factory = FakeWriterFactory(fake_writer)
    encoder = open_encoder(
        path, fps, (width, height), writer_factory=writer_factory, ffmpeg_factory=FakeFfmpegFactory()
    )

    frame_a = _make_frame(width, height, offset=0)
    frame_b = _make_frame(width, height, offset=7)

    encoder.write(frame_a)
    assert encoder.frames_written == 1
    assert len(fake_writer.frames) == 1
    np.testing.assert_array_equal(fake_writer.frames[0], frame_a)

    encoder.write(frame_b)
    assert encoder.frames_written == 2
    assert len(fake_writer.frames) == 2
    np.testing.assert_array_equal(fake_writer.frames[1], frame_b)


def test_ac04_write_routes_to_ffmpeg_stdin_and_counts_frames(tmp_path):
    path = tmp_path / "clip.mp4"
    width, height, fps = 64, 32, 30.0
    dead_writer = FakeWriter(opened=False)
    writer_factory = FakeWriterFactory(dead_writer)
    ffmpeg_process = FakeFfmpegProcess()
    ffmpeg_factory = FakeFfmpegFactory(process=ffmpeg_process)
    encoder = open_encoder(
        path, fps, (width, height), writer_factory=writer_factory, ffmpeg_factory=ffmpeg_factory
    )

    frame = _make_frame(width, height)
    encoder.write(frame)

    assert encoder.frames_written == 1
    assert len(ffmpeg_process.stdin.chunks) == 1
    written_bytes = ffmpeg_process.stdin.chunks[0]
    assert len(written_bytes) == width * height * 3
    assert written_bytes == frame.tobytes()

    encoder.write(frame)
    assert encoder.frames_written == 2
    assert len(ffmpeg_process.stdin.chunks) == 2


# --- AC5: release -------------------------------------------------------------


def test_ac05_release_avc1_is_idempotent_and_flips_is_open(tmp_path):
    path = tmp_path / "clip.mp4"
    fake_writer = FakeWriter(opened=True)
    writer_factory = FakeWriterFactory(fake_writer)
    encoder = open_encoder(
        path, 30.0, (64, 32), writer_factory=writer_factory, ffmpeg_factory=FakeFfmpegFactory()
    )

    assert encoder.is_open is True

    encoder.release()
    assert fake_writer.release_calls == 1
    assert encoder.is_open is False

    encoder.release()  # idempotent: no raise, and no-op on the underlying writer
    assert fake_writer.release_calls == 1
    assert encoder.is_open is False


def test_ac05_release_ffmpeg_closes_stdin_and_waits_and_is_idempotent(tmp_path):
    path = tmp_path / "clip.mp4"
    dead_writer = FakeWriter(opened=False)
    writer_factory = FakeWriterFactory(dead_writer)
    ffmpeg_process = FakeFfmpegProcess()
    ffmpeg_factory = FakeFfmpegFactory(process=ffmpeg_process)
    encoder = open_encoder(
        path, 30.0, (64, 32), writer_factory=writer_factory, ffmpeg_factory=ffmpeg_factory
    )

    assert encoder.is_open is True

    encoder.release()
    assert ffmpeg_process.stdin.closed is True
    assert ffmpeg_process.wait_calls == 1
    assert encoder.is_open is False

    encoder.release()  # idempotent: no raise, and no-op on the underlying process
    assert ffmpeg_process.wait_calls == 1
    assert encoder.is_open is False


# --- AC6: probe_video ---------------------------------------------------------


def test_ac06_probe_video_returns_values_from_cap_factory():
    fake_cap = FakeCapture(opened=True, fps=29.97, frame_count=451, width=1920, height=1080)

    probe = probe_video("some/finished/clip.mp4", cap_factory=lambda path: fake_cap)

    assert probe == VideoProbe(fps=29.97, frame_count=451, width=1920, height=1080)
    assert isinstance(probe.fps, float)
    assert isinstance(probe.frame_count, int)
    assert isinstance(probe.width, int)
    assert isinstance(probe.height, int)


def test_ac06_probe_video_raises_when_capture_will_not_open():
    fake_cap = FakeCapture(opened=False)

    with pytest.raises(RuntimeError):
        probe_video("some/bad/clip.mp4", cap_factory=lambda path: fake_cap)


# --- AC7: parent dirs ----------------------------------------------------------


def test_ac07_open_encoder_creates_missing_parent_dirs(tmp_path):
    nested = tmp_path / "entry" / "video" / "out.mp4"
    assert not nested.parent.exists()

    fake_writer = FakeWriter(opened=True)
    writer_factory = FakeWriterFactory(fake_writer)

    encoder = open_encoder(
        nested, 30.0, (64, 48), writer_factory=writer_factory, ffmpeg_factory=FakeFfmpegFactory()
    )

    assert nested.parent.exists()
    assert nested.parent.is_dir()
    assert Path(encoder.path) == nested
