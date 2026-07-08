"""Background capture-infer loop: owns the camera, reads a frame at its native
1080p, runs the detector, renders an overlay, and publishes an atomic ``Latest``
snapshot + heartbeat every tick.

Camera-free, dataset-free, HTTP-free — ``render_fn`` decides what the overlay
looks like, ``detector`` is injected. Never crashes on a read failure, never
tight-spins, never serves a stale frame as fresh, and reopens a wedged/unplugged
camera on its own so a dropped stream recovers without a restart.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import NamedTuple

import cv2
import numpy as np

# One resolution end to end. Camo streams 1080p, RF-DETR resizes any input to
# 768x768 internally, and the saved still IS the frame we detect on — so there
# is no capture/detection split and no scale factor: `predict()`, the overlay,
# the MJPEG stream, and the saved image all share the camera's native frame.


class Latest(NamedTuple):
    frame: np.ndarray
    dets: "sv.Detections"
    threshold: float
    overlay_jpeg: bytes
    count: int


class CaptureLoop:
    def __init__(
        self,
        detector,
        camera_index: int,
        render_fn,
        cap_factory=cv2.VideoCapture,
        target_size=(1920, 1080),
        stale_after_s: float = 2.0,
        start_read_attempts: int = 15,
        reopen_after_s: float = 3.0,
        ring_size: int = 16,
    ):
        self._detector = detector
        self._render_fn = render_fn
        self._cap_factory = cap_factory
        self._target_size = target_size
        self._stale_after_s = stale_after_s
        self._start_read_attempts = start_read_attempts
        self._reopen_after_s = reopen_after_s
        # Ring of recent (generation, Latest) so /flag can save the EXACT frame
        # the operator froze on, not just the newest. Each Latest holds a full
        # 1080p BGR frame (~6 MB), so this is a deliberately shallow window — a
        # freeze-capture flags within ~1 s of display, and the detector runs well
        # under 30 fps, so ~16 frames is seconds of history at a bounded ~100 MB.
        self._ring: deque = deque(maxlen=ring_size)

        self._cap = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._camera_index = camera_index
        self._pending_camera_index: int | None = None
        self._latest: Latest | None = None
        self._heartbeat: float | None = None
        self._last_read_ok: float | None = None
        self._generation = 0
        self._dead = False

    def _apply_capture_settings(self) -> None:
        width, height = self._target_size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> None:
        self._cap = self._cap_factory(self._camera_index)
        self._apply_capture_settings()

        # We accept whatever resolution the camera negotiates (Camo hands us
        # 1080p) — the pipeline is resolution-agnostic. The only hard failure is
        # a camera that yields no valid frame at all: catch it here, at startup,
        # rather than silently running a dead loop.
        #
        # Virtual cameras (Camo, Continuity) routinely fail the FIRST read or two
        # right after open while the stream spins up, then stream fine — so we
        # retry a handful of reads before declaring the camera dead. A single
        # one-shot read here spuriously crashed startup on a healthy Camo feed.
        ok, frame, actual = False, None, None
        for _ in range(self._start_read_attempts):
            ok, frame = self._cap.read()
            actual = getattr(frame, "shape", None) if ok else None
            if ok and actual is not None and len(actual) == 3 and actual[2] == 3:
                break
            time.sleep(0.1)
        else:
            # Release the handle we opened before bailing — the loop thread never
            # started, and main() calls start() before entering its try/finally,
            # so nothing else will release this dead-camera capture otherwise.
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            raise RuntimeError(
                f"camera at index {self._camera_index} returned no valid frame "
                f"after {self._start_read_attempts} attempts: ok={ok}, shape={actual}"
            )

        self._last_read_ok = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _reopen_current(self) -> None:
        """Release and reopen the camera at the active index.

        The ONLY way to recover a handle AVFoundation has invalidated (real
        unplug/replug, a stream that hiccupped, or a rapid release→reopen). A
        plain ``read()``-retry loop on the dead handle never comes back — the
        device has to be reopened. Best-effort: a release/open that itself throws
        must not kill the loop, so we swallow and let the next tick retry.
        """
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = self._cap_factory(self._camera_index)
        self._apply_capture_settings()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._tick()
        except BaseException:
            with self._lock:
                self._dead = True

    def _tick(self) -> None:
        with self._lock:
            pending_index = self._pending_camera_index
            self._pending_camera_index = None
        if pending_index is not None:
            self._camera_index = pending_index
            self._reopen_current()
            # Fresh handle — start its failure window now so the reopen-on-stale
            # guard below doesn't immediately reopen a camera still warming up.
            self._last_read_ok = time.monotonic()

        ok, frame = self._cap.read()
        if not ok:
            # A handle that keeps failing past the reopen window is wedged (real
            # unplug, or AVFoundation invalidated it) — reopen the device to
            # recover. Retrying read() on a dead handle never comes back. Paced
            # by _reopen_after_s so this is a slow backoff, not a reopen storm;
            # _heartbeat is untouched, so /status stays "stale" until frames
            # actually resume.
            now = time.monotonic()
            if self._last_read_ok is None or now - self._last_read_ok > self._reopen_after_s:
                self._reopen_current()
                self._last_read_ok = now
            time.sleep(0.05)
            return

        self._last_read_ok = time.monotonic()
        # Snapshot the threshold and pin it into predict() so the value recorded
        # as provenance is exactly the one that filtered `dets` — a /confidence
        # change racing between the read and predict()'s own read can't split them.
        threshold = self._detector.confidence_threshold
        dets = self._detector.predict(frame, confidence_threshold=threshold)
        overlay = self._render_fn(frame.copy(), dets)
        ok_encode, buf = cv2.imencode(".jpg", overlay)
        overlay_jpeg = buf.tobytes() if ok_encode else b""

        latest = Latest(frame, dets, threshold, overlay_jpeg, len(dets))
        with self._lock:
            self._latest = latest
            self._heartbeat = time.monotonic()
            self._generation += 1
            self._ring.append((self._generation, latest))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Only release once the loop thread has actually stopped — releasing a
        # capture the loop is still reading (join timed out) risks a native
        # use-after-release.
        if self._cap is not None and (self._thread is None or not self._thread.is_alive()):
            self._cap.release()

    def set_camera(self, index: int) -> None:
        with self._lock:
            self._pending_camera_index = index

    def snapshot(self) -> Latest | None:
        with self._lock:
            return self._latest

    def snapshot_with_generation(self) -> tuple[int, Latest | None]:
        """Newest ``Latest`` and its generation, read together under one lock.

        ``/frame`` returns both so the browser knows the generation of the frame
        it is painting — the id it later hands to ``/flag`` to capture that exact
        frame. Reading them atomically (vs two separate locked properties) also
        means the pair can never be mismatched.
        """
        with self._lock:
            return self._generation, self._latest

    def snapshot_at(self, generation: int) -> Latest | None:
        """The ``Latest`` for ``generation``, or ``None`` if it aged out of the ring.

        Lets ``/flag`` save the frame the operator actually froze on. Returns
        ``None`` once the frame has been evicted (the window is only a few
        seconds), so the caller can reject the flag rather than silently save a
        different frame.
        """
        with self._lock:
            for gen, latest in reversed(self._ring):
                if gen == generation:
                    return latest
        return None

    @property
    def health(self) -> str:
        with self._lock:
            dead = self._dead
            heartbeat = self._heartbeat
        if dead:
            return "dead"
        if heartbeat is None or time.monotonic() - heartbeat > self._stale_after_s:
            return "stale"
        return "ok"

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def camera_index(self) -> int:
        """The index the loop is currently capturing from.

        Reflects the *active* device: ``set_camera(n)`` only takes effect once the
        loop thread reopens on the next tick, so this tracks what is really being
        streamed, not a pending request. Surfaced via ``/status`` so the UI can
        show the live camera instead of a misleading placeholder.
        """
        with self._lock:
            return self._camera_index
