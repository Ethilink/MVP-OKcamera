"""TR4 — Post-pass job runner.

After a recording stops, reopens the MP4 and runs the detector over every
frame at the mining threshold, streaming decoded frames + detections into a
``VideoEntryWriter`` (TR3). A run that dies mid-way leaves a valid partial
project (``selected_frames.json`` written, no ``annotations.json``); a retry
with a fresh job rebuilds everything from scratch (see RECORDING.md
§Post-pass specifics).
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2

from backend.encoder import probe_video
from backend.video_writer import VideoEntryWriter


@dataclass
class PostPassStatus:
    state: str            # "pending"|"running"|"paused"|"done"|"failed"|"cancelled"
    done: int             # frames processed so far
    total: int            # authoritative frame_count (== frames_written)
    error: str | None


class PostPassJob:
    def __init__(
        self,
        entry_dir: Path,
        entry_name: str,
        detector,
        *,
        keyframes,
        frame_count: int,
        mining_threshold: float,
        operator_threshold: float,
        model_version: str,
        probe_fn=probe_video,
        cap_factory=cv2.VideoCapture,
        video_writer_factory=VideoEntryWriter,
    ):
        self._entry_dir = Path(entry_dir)
        self._entry_name = entry_name
        self._detector = detector
        self._keyframes = keyframes
        self._frame_count = frame_count
        self._mining_threshold = mining_threshold
        self._operator_threshold = operator_threshold
        self._model_version = model_version
        self._probe_fn = probe_fn
        self._cap_factory = cap_factory
        self._video_writer_factory = video_writer_factory

        self._state = "pending"
        self._done = 0
        self._error: str | None = None
        self._cancelled = threading.Event()
        # Leaf lock serialising each per-frame detector.predict against the live
        # capture loop (INV-3, U2). Injected by the app as an ATTRIBUTE after
        # construction (capture.detector_lock) — NOT a constructor arg — so the
        # frozen post_pass_factory signature is unchanged and injected fakes keep
        # working. Defaults to a no-op so a job built directly (TR4 unit tests, or
        # any caller that never shares the detector) runs unchanged. Held ONLY
        # around the predict call — never across the pause/park below — so a parked
        # job holds no lock and can't deadlock the live loop.
        self.predict_lock = nullcontext()
        # Pause/resume (U2). `_pause_requested` is honored between frames; when
        # honored, run() parks by blocking on `_resume_gate` WITHOUT calling
        # predict, so the caller can reclaim the single shared detector for the
        # live overlay (via capture.pause_inference/resume_inference — never from
        # in here; AC7 keeps run() touching the detector solely through .predict).
        self._pause_requested = threading.Event()
        self._resume_gate = threading.Event()

    def pause(self) -> None:
        """Request the run to yield before its next read/predict/write (U2).

        Honored cooperatively between frames: ``run()`` transitions to
        ``"paused"``, freezes ``status.done`` at the last fully-written frame, and
        blocks on the resume gate WITHOUT calling ``predict`` — so the live
        detector is free for the overlay while parked. Idempotent and thread-safe.
        """
        self._pause_requested.set()

    def resume(self) -> None:
        """Un-park a paused run so it continues its loop from the parked index
        (U2). Byte-identical to an uninterrupted run: ``run()`` never closed the
        capture, so it reads the very next frame it was about to read. Idempotent.
        """
        self._pause_requested.clear()
        self._resume_gate.set()

    def cancel(self) -> None:
        """Request cooperative cancellation (TR5 /record/discard during
        ``processing``). ``run()`` checks the flag before every read/predict
        and again before every disk write, so after the caller joins the
        worker thread a cancelled job is guaranteed to perform no further
        ``detector.predict`` calls and to write no further files — that is
        what lets discard safely ``rmtree`` the entry folder without a stray
        keyframe JPEG reappearing behind it. Idempotent and thread-safe.
        """
        self._cancelled.set()

    def run(self) -> None:
        self._done = 0
        self._error = None
        if self._cancelled.is_set():
            # Cancelled before the worker was even scheduled: touch nothing
            # (no cap, no writer, no selected_frames.json).
            self._state = "cancelled"
            return
        self._state = "running"
        cap = None
        try:
            mp4_path = self._entry_dir / "video" / f"{self._entry_name}.mp4"
            cap = self._cap_factory(str(mp4_path))
            probe = self._probe_fn(str(mp4_path))

            # Re-encoded MP4s can misreport CAP_PROP_FRAME_COUNT (import doc
            # §7). The recording's own counter is authoritative — prefer it
            # over the probe and keep going rather than crash on the gap.
            video = {
                "fps": probe.fps,
                "width": probe.width,
                "height": probe.height,
                "frame_count": self._frame_count,
            }
            writer = self._video_writer_factory(
                self._entry_dir,
                self._entry_name,
                self._model_version,
                video=video,
                keyframes=self._keyframes,
                operator_threshold=self._operator_threshold,
                mining_threshold=self._mining_threshold,
            )
            writer.write_selected_frames()

            for frame_number in range(self._frame_count):
                # Cooperative cancel: bail BEFORE the next read/predict/write
                # so a discarded take's worker stops calling the shared
                # detector (§Detector sharing) and creates no further files.
                if self._cancelled.is_set():
                    self._state = "cancelled"
                    return
                # Cooperative pause (U2, INV-4): park HERE — before this frame's
                # read/predict/write, so the frame is NOT consumed — and yield the
                # detector (no predict while parked). `self._done` stays at the
                # last fully-written frame; on resume we fall straight through and
                # read frame_number, i.e. the very next frame we were about to read
                # (byte-identical output). `cancel` still wins: it breaks the park
                # and returns before any further read/predict/write.
                if self._pause_requested.is_set():
                    self._state = "paused"
                    self._resume_gate.clear()
                    while self._pause_requested.is_set() and not self._cancelled.is_set():
                        # Bounded wait, not a naked block: a cancel that arrives
                        # while parked is observed within one tick even though it
                        # does not set the gate (cancel() is unchanged, TR4 AC7).
                        self._resume_gate.wait(timeout=0.05)
                    if self._cancelled.is_set():
                        self._state = "cancelled"
                        return
                    self._state = "running"
                    # fall through: process this same frame_number now.
                _, frame = cap.read()
                # Serialise THIS predict against the live capture loop (INV-3):
                # the app injects the shared capture.detector_lock as predict_lock.
                # Held per-frame only — released before the next loop iteration's
                # cancel/pause checks — so the live loop waits at most one frame and
                # a parked job (which parks above, outside this block) holds nothing.
                # Leaf discipline: no other lock is acquired inside here.
                with self.predict_lock:
                    dets = self._detector.predict(
                        frame, confidence_threshold=self._mining_threshold
                    )
                # Re-check after the (slow, CPU-bound) predict: a cancel that
                # landed mid-predict must never reach the disk write — the
                # write is what re-creates a stray folder after discard's
                # rmtree if the join ever times out.
                if self._cancelled.is_set():
                    self._state = "cancelled"
                    return
                writer.add_frame(frame_number, frame, dets)
                self._done = frame_number + 1

            if self._cancelled.is_set():
                self._state = "cancelled"
                return
            writer.finalize()
            self._state = "done"
        except Exception as exc:
            self._state = "failed"
            self._error = str(exc)
        finally:
            if cap is not None:
                cap.release()

    @property
    def status(self) -> PostPassStatus:
        return PostPassStatus(state=self._state, done=self._done, total=self._frame_count, error=self._error)
