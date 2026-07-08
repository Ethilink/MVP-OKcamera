"""TR4 — Post-pass job runner.

After a recording stops, reopens the MP4 and runs the detector over every
frame at the mining threshold, streaming decoded frames + detections into a
``VideoEntryWriter`` (TR3). A run that dies mid-way leaves a valid partial
project (``selected_frames.json`` written, no ``annotations.json``); a retry
with a fresh job rebuilds everything from scratch (see RECORDING.md
§Post-pass specifics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    def run(self) -> None:
        self._state = "running"
        self._done = 0
        self._error = None
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
                _, frame = cap.read()
                dets = self._detector.predict(frame, confidence_threshold=self._mining_threshold)
                writer.add_frame(frame_number, frame, dets)
                self._done = frame_number + 1

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
