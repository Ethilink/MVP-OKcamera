"""Regression: /record/discard during ``processing`` must CANCEL the running
post-pass worker (TR5-api.md "Follow-up bug surfaced during TR6 R4").

Before the fix, discard only cleared ``rec.job``, resumed inference, and
rmtree'd the entry folder — the orphaned worker kept calling
``detector.predict`` concurrently with the just-resumed live inference
(§Detector-sharing violation) and its keyframe ``cv2.imwrite`` re-created a
stray partial folder AFTER the rmtree. The fix: ``PostPassJob.cancel()``
(cooperative, checked before every read/predict and again before every disk
write) + discard joining the worker thread (outside ``recording_lock``,
bounded) BEFORE deleting the folder.

Deterministic, no sleeps-as-synchronization: the real ``PostPassJob`` is fed
an instant fake cap and a huge ``frame_count`` (millions of microsecond
iterations — the worker cannot possibly finish before the test discards it),
so the discard always lands mid-loop; every wait is a bounded ``_wait_for``
poll. The writer fake creates a real file per ``add_frame`` — a faithful
superset of the real writer's keyframe JPEGs — so the stray-folder assertion
exercises the actual write-after-rmtree race.
"""

from __future__ import annotations

import threading

import numpy as np
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.encoder import VideoProbe
from backend.postpass import PostPassJob
from tests.fakes import make_fake_frame
from tests.recording_fakes import FakeEncoder
from tests.test_recording_api import (
    RecordingStubCapture,
    _wait_for,
    _writer_factory,
)

W, H = 16, 16


class _CountingDetector:
    """Minimal detector double: thread-safe call counter, no per-call
    allocation (the worker may run thousands of iterations before the test
    cancels it)."""

    def __init__(self) -> None:
        self.confidence_threshold = 0.5
        self.calls = 0
        self._lock = threading.Lock()

    def predict(self, frame, confidence_threshold=None):
        with self._lock:
            self.calls += 1
        return None  # the writer fake below ignores dets


class _InstantCap:
    """cv2.VideoCapture stand-in: every read() instantly returns the same tiny
    frame — the loop pace is pure CPU, so cancellation latency is the only
    thing the test measures."""

    def __init__(self) -> None:
        self._frame = np.zeros((H, W, 3), dtype=np.uint8)
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        return True, self._frame

    def release(self) -> None:
        self.released = True


class _FileWritingWriter:
    """``VideoEntryWriter``-shaped fake whose ``add_frame`` writes a REAL file
    under ``<entry>/images/`` on every frame — a superset of the real writer
    (which writes only keyframe JPEGs), so any post-rmtree write recreates a
    stray folder exactly like the reproduced bug."""

    def __init__(
        self,
        entry_dir,
        entry_name,
        model_version,
        *,
        video,
        keyframes,
        operator_threshold,
        mining_threshold,
        **kwargs,
    ):
        self.entry_dir = entry_dir
        self.entry_name = entry_name
        self.add_frame_calls = 0
        self.finalized = False

    def write_selected_frames(self) -> None:
        pass

    def add_frame(self, frame_number, frame, dets) -> None:
        images = self.entry_dir / "images"
        images.mkdir(parents=True, exist_ok=True)
        (images / f"{self.entry_name}_f{frame_number:06d}.jpg").write_bytes(b"stub")
        self.add_frame_calls += 1

    def finalize(self) -> None:
        self.finalized = True


def _build_app_with_real_postpass(tmp_path, capture, detector):
    """App wired like tests/test_recording_api.py but with the REAL
    ``PostPassJob`` (fed the fakes above through its injectable seams), so the
    cancel/join behavior under test is the production code path."""
    app = create_app(detector, _writer_factory(), capture)
    app.state.output_path = str(tmp_path)
    app.state.capture_fps = 30.0
    app.state.mining_threshold = 0.25
    app.state.model_version = "rfdetr-test"
    app.state.open_encoder = lambda path, fps, frame_size, **kw: FakeEncoder()

    jobs: list[PostPassJob] = []
    writers: list[_FileWritingWriter] = []

    def post_pass_factory(entry_dir, entry_name, det, **kwargs):
        def writer_factory(*args, **kw):
            writer = _FileWritingWriter(*args, **kw)
            writers.append(writer)
            return writer

        job = PostPassJob(
            entry_dir,
            entry_name,
            det,
            probe_fn=lambda path, **kw: VideoProbe(
                fps=30.0, frame_count=0, width=W, height=H
            ),
            cap_factory=lambda path: _InstantCap(),
            video_writer_factory=writer_factory,
            **kwargs,
        )
        jobs.append(job)
        return job

    app.state.post_pass_factory = post_pass_factory
    return app, jobs, writers


# ---------------------------------------------------------------------------
# API-level regression: discard mid-post-pass stops the worker before the
# rmtree; no stray folder; resume_inference exactly once; retry path clean.
# ---------------------------------------------------------------------------


def test_discard_during_processing_cancels_worker_and_leaves_no_stray(tmp_path):
    # frame_count is minutes of work at the fake's pace — the discard below
    # (fired within milliseconds of the first confirmed write) always lands
    # mid-loop; without the fix the worker would run on long after discard.
    frame_count = 5_000_000
    capture = RecordingStubCapture(
        frame=make_fake_frame(64, 48), frames_written=frame_count
    )
    detector = _CountingDetector()
    app, jobs, writers = _build_app_with_real_postpass(tmp_path, capture, detector)
    client = TestClient(app)

    client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
    client.post("/record/stop").raise_for_status()

    # The post-pass is live mid-loop: at least one frame fully processed
    # (predicted AND written to disk).
    assert _wait_for(lambda: writers and writers[0].add_frame_calls >= 1)
    worker = app.state.recording.worker
    assert worker is not None and worker.is_alive()
    entry_dir = tmp_path / "take1"
    assert entry_dir.exists()

    resp = client.post("/record/discard")
    assert resp.status_code == 200

    # (a) The worker has actually STOPPED by the time discard returns —
    # discard cancelled the job and joined the thread before the rmtree.
    assert not worker.is_alive()
    assert jobs[0].status.state == "cancelled"
    assert jobs[0].status.done < frame_count  # bailed early, not run to completion
    n_predicts = detector.calls
    n_writes = writers[0].add_frame_calls
    assert not writers[0].finalized

    # (b) NO stray folder after discard returns — and it STAYS gone (bounded
    # poll for a folder that must never reappear; pre-fix the orphaned
    # worker's next write recreated it here).
    assert not entry_dir.exists()
    assert _wait_for(lambda: entry_dir.exists(), timeout=0.5) is False
    # The dead worker performed no further detector calls and wrote no
    # further files during that window.
    assert detector.calls == n_predicts
    assert writers[0].add_frame_calls == n_writes

    # (c) resume_inference fired EXACTLY once for the processing episode
    # (discard-side; the orphaned worker's ownership guard must not fire it).
    assert capture.pause_calls == 1
    assert capture.resume_calls == 1

    assert client.get("/record/status").json()["state"] == "idle"

    # The cancelled prior job doesn't interfere with a fresh take (AC-shaped
    # sanity: the machine is fully reusable after a cancel-discard).
    assert client.post("/record/start", json={"entry_name": "take2"}).status_code == 200
    assert capture.resume_calls == 1  # still exactly once


# ---------------------------------------------------------------------------
# Unit-level: PostPassJob.cancel() semantics.
# ---------------------------------------------------------------------------


def _make_job(tmp_path, detector, cap_factory, writers, frame_count):
    def writer_factory(*args, **kw):
        writer = _FileWritingWriter(*args, **kw)
        writers.append(writer)
        return writer

    return PostPassJob(
        tmp_path / "entry1",
        "entry1",
        detector,
        keyframes=[],
        frame_count=frame_count,
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=lambda path, **kw: VideoProbe(
            fps=30.0, frame_count=0, width=W, height=H
        ),
        cap_factory=cap_factory,
        video_writer_factory=writer_factory,
    )


def test_cancel_mid_run_stops_before_the_next_read_predict_and_write(tmp_path):
    """A cancel landing mid-iteration (here: during the 3rd read) lets the
    in-flight predict finish but drops its result before the disk write, and
    the loop performs no 4th read — synchronous, fully deterministic."""
    holder: dict = {}

    class _CancellingCap(_InstantCap):
        def read(self):
            ok, frame = super().read()
            if self.reads == 3:
                holder["job"].cancel()
            return ok, frame

    cap = _CancellingCap()
    detector = _CountingDetector()
    writers: list[_FileWritingWriter] = []
    job = _make_job(tmp_path, detector, lambda path: cap, writers, frame_count=100)
    holder["job"] = job

    job.run()

    assert job.status.state == "cancelled"
    assert cap.reads == 3  # top-of-loop check stopped the 4th iteration's read
    assert detector.calls == 3  # the in-flight predict for frame 2 completed...
    assert writers[0].add_frame_calls == 2  # ...but never reached the disk write
    assert job.status.done == 2
    assert not writers[0].finalized
    assert cap.released  # the finally still releases the cap


def test_cancel_before_run_touches_nothing(tmp_path):
    cap_calls: list = []

    def cap_factory(path):
        cap_calls.append(path)
        return _InstantCap()

    detector = _CountingDetector()
    writers: list[_FileWritingWriter] = []
    job = _make_job(tmp_path, detector, cap_factory, writers, frame_count=10)

    job.cancel()
    job.run()

    assert job.status.state == "cancelled"
    assert cap_calls == []  # no video opened
    assert writers == []  # no writer constructed, no files minted
    assert detector.calls == 0
    assert not (tmp_path / "entry1").exists()
