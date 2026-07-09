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
    frame_number: int | None = None  # MP4 write index this frame was detected on
                                      # while recording; None when idle. Default keeps
                                      # peripheral 5-arg constructions valid.


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
        # Serialises cap.read() across the idle inference loop and the recording
        # reader thread so exactly one thread ever reads the camera at a time
        # (and the idle loop never steals a frame the encoder should have got).
        self._read_lock = threading.Lock()
        # Serialises the ACTUAL detector.predict call between the live capture
        # loop and the idle-drain post-pass worker so the two never predict on the
        # shared detector concurrently (INV-3, U2). The `_inference_paused` flag
        # decides WHO should predict; this lock guarantees no overlap even in the
        # flag-check→predict window (a claim can't slip in mid-predict). It is a
        # LEAF lock: held ONLY around predict, never while holding self._lock /
        # recording_lock / drain_cond, and released between frames on both sides
        # (the post-pass job never holds it across a pause/park). The drain worker
        # shares THIS object via app.state.detector_lock (see backend/app.py).
        self._detector_lock = threading.Lock()

        self._camera_index = camera_index
        self._pending_camera_index: int | None = None
        self._latest: Latest | None = None
        self._heartbeat: float | None = None
        self._last_read_ok: float | None = None
        self._generation = 0
        self._dead = False

        # Recording state. Idle until start_recording: the reader thread only
        # exists while recording; the persistent loop thread flips between the
        # idle tick (reads the camera itself) and the recording tick (consumes
        # the reader's hand-off slot).
        self._recording = False
        self._encoder = None
        self._reader_thread = None
        self._frames_written = 0
        self._slot: tuple[np.ndarray, int] | None = None  # newest (frame, frame_number)
        self._frame_ready = threading.Event()
        self._inference_paused = False
        self._record_error: BaseException | None = None

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
                if self._recording:
                    self._tick_recording()
                else:
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

        # While recording the reader thread owns cap.read(); the guard (checked
        # under the same lock the reader reads under) hands the camera over
        # cleanly — no double-read, no frame stolen from the encoder stream.
        with self._read_lock:
            if self._recording:
                return
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
        # Paused (the post-pass owns the detector between takes): keep the camera
        # alive and health fresh, but don't run predict or publish — the stream
        # keeps serving the last overlay. The check→predict is made atomic under
        # the detector lock (INV-3): hold the lock, re-read the paused flag, and
        # either skip (drain owns the detector) or predict WHILE holding it, so the
        # drain worker's per-frame predict can never overlap this one. The flag is
        # read WITHOUT self._lock here to keep the detector lock a leaf (never
        # acquire self._lock while holding it); a plain bool read is atomic under
        # the GIL, and pause_inference()/resume_inference() publish it under
        # self._lock, so a value that has already been set is observed.
        with self._detector_lock:
            if self._inference_paused:
                self._heartbeat = time.monotonic()
                return
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
        self._recording = False
        self._frame_ready.set()  # wake a parked recording tick so it can exit
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
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

    # --- Recording / inference-pause interface (TR1) --------------------------

    def _reader_run(self) -> None:
        """Own the camera while recording: read every frame, write it to the
        encoder in capture order stamping frame_number = its 0-based write index
        (writer index == MP4 index), and hand the newest (frame, frame_number)
        to the inference tick via the slot.
        """
        while self._recording and not self._stop_event.is_set():
            with self._read_lock:
                if not self._recording:
                    break
                ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_number = self._frames_written  # only this thread advances it
            try:
                self._encoder.write(frame)
            except BaseException as exc:  # AC8: contain, surface, stop writing
                with self._lock:
                    self._record_error = exc
                self._frame_ready.set()
                return

            now = time.monotonic()
            self._last_read_ok = now
            with self._lock:
                self._frames_written = frame_number + 1
                self._slot = (frame, frame_number)
                self._heartbeat = now  # keep health "ok" independent of inference
            self._frame_ready.set()

    def _tick_recording(self) -> None:
        # Inference runs on whatever the reader last handed over, skipping the
        # in-between frames freely when the detector is slower than the reader.
        if not self._frame_ready.wait(timeout=0.5):
            return
        self._frame_ready.clear()
        with self._lock:
            slot = self._slot
        if slot is None:
            return
        frame, frame_number = slot
        threshold = self._detector.confidence_threshold
        # Same atomic check→predict under the detector lock as the idle tick
        # (INV-3). During recording the drain is parked, so this lock is normally
        # uncontended — but holding it keeps the "live loop and drain never predict
        # concurrently" invariant true unconditionally, in both directions.
        with self._detector_lock:
            if self._inference_paused:
                return
            dets = self._detector.predict(frame, confidence_threshold=threshold)
        overlay = self._render_fn(frame.copy(), dets)
        ok_encode, buf = cv2.imencode(".jpg", overlay)
        overlay_jpeg = buf.tobytes() if ok_encode else b""

        # Publish the frame_number stamped when THIS frame was written — never a
        # newest-at-publish counter — so a slow inference reports the frame it
        # actually detected on.
        latest = Latest(frame, dets, threshold, overlay_jpeg, len(dets), frame_number)
        with self._lock:
            # Recording may have ended while predict() ran; publishing a
            # recording-stamped Latest now would leave a non-None frame_number
            # visible after stop_recording() returned (idle must read None).
            # Drop this just-detected frame — the next idle tick publishes
            # frame_number=None. Re-checked inside the publish lock so the flip
            # can't slip between the check and the publish.
            if not self._recording:
                return
            self._latest = latest
            self._heartbeat = time.monotonic()
            self._generation += 1
            self._ring.append((self._generation, latest))

    def start_recording(self, encoder) -> None:
        with self._lock:
            if self._recording:
                raise RuntimeError("already recording")
            self._encoder = encoder
            self._frames_written = 0
            self._slot = None
            self._record_error = None
            self._frame_ready.clear()
            self._recording = True
        self._reader_thread = threading.Thread(target=self._reader_run, daemon=True)
        self._reader_thread.start()

    def stop_recording(self) -> int:
        with self._lock:
            if not self._recording:
                raise RuntimeError("not recording")
            self._recording = False
        # Wake a parked inference tick, then let the reader finish any in-flight
        # write and exit BEFORE releasing the encoder (no write races release).
        # Join WITHOUT a timeout cap: `encoder.release()` must provably run after
        # the last `encoder.write`, so we block until the reader has actually
        # stopped writing. The reader loop exits promptly once `_recording` is
        # False (one more read+write at most), so this cannot hang in normal use.
        self._frame_ready.set()
        if self._reader_thread is not None:
            self._reader_thread.join()
        if self._encoder is not None:
            self._encoder.release()
        with self._lock:
            count = self._frames_written
            self._frames_written = 0
            self._slot = None
            self._encoder = None
        self._reader_thread = None
        return count

    def pause_inference(self) -> None:
        with self._lock:
            self._inference_paused = True

    def resume_inference(self) -> None:
        with self._lock:
            self._inference_paused = False

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def frames_written(self) -> int:
        """Frames written to the encoder so far this recording; 0 when idle."""
        with self._lock:
            return self._frames_written

    @property
    def inference_paused(self) -> bool:
        with self._lock:
            return self._inference_paused

    @property
    def detector_lock(self) -> threading.Lock:
        """The leaf lock serialising ``detector.predict`` between the live loop
        and the idle-drain post-pass worker (INV-3, U2). The app injects THIS
        object into each PostPassJob so both sides guard the shared detector with
        the same lock; see ``backend/app.py``.
        """
        return self._detector_lock

    @property
    def recording_error(self) -> BaseException | None:
        """The exception raised by ``encoder.write`` if one occurred this
        recording, else ``None`` (AC8 — encoder failure is surfaced, not silent).
        """
        with self._lock:
            return self._record_error

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
