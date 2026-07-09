"""U2 (docs/tasks/U2-queue.md) behavior contract for the recording-mode API +
idle-draining post-pass queue, migrated from TR5's blocking-`processing`
contract (`docs/tasks/TR5-api.md`).

Blind-TDD: this file exercises the HTTP behaviour contract via
``fastapi.testclient.TestClient`` against the frozen ``create_app`` seam; the
coder replaces ``backend/app.py``'s pre-U2 blocking recording state machine
without ever seeing this file.

The frozen `/record/status` shape (U2-queue.md §2) this file tests against:

    {"state": "idle" | "recording",
     "drain": {"current": {"entry_name", "done", "total"} | None,
               "queued": [entry_name, ...],
               "eta_seconds": number},
     "error": string | None}

``state`` is the FOREGROUND mode only -- `processing`/`failed` NEVER appear as
`state` anymore; draining progress and halt-on-failure live under `drain`/
`error` instead (§5 halt-on-failure: the queue stops on the first failure,
which stays as `drain.current` with `error` set, until Retry or Discard acts
on it).

Everything is camera-free / ffmpeg-free / onnx-free:

- ``app.state.open_encoder`` is overridden with a factory returning TR1's
  ``FakeEncoder`` (``tests/recording_fakes.py``) so no real ffmpeg runs, and so
  the call args (path / fps / frame_size) are inspectable.
- ``app.state.post_pass_factory`` is overridden with ``_FakeJob`` -- enriched
  for U2 with per-frame progress and a ``pause()``/``resume()`` seam
  (``status.state`` gains ``"paused"``) alongside the pre-U2 block-until-
  ``release`` model, so both the endpoint-guard regressions AND the new
  queue/pause/resume ACs are observable deterministically via bounded
  ``_wait_for`` polls -- never sleep-and-hope.
- ``app.state.probe_video`` is overridden with a fake returning a ``VideoProbe``.
- ``RecordingStubCapture`` is a deterministic capture stand-in exposing exactly
  the recording surface the endpoints touch (``snapshot``, ``start_recording``,
  ``stop_recording``, ``frames_written``, ``pause_inference`` /
  ``resume_inference`` / ``inference_paused``). AC5 (X-Frame-Number) uses a
  REAL ``CaptureLoop`` + numbered ``FakeCapture`` because a live loop is what
  makes that header true.

The REAL, byte-identical proof of "resume continues from the parked frame
index, not 0" (INV-4) lives in ``tests/test_postpass.py``'s pause/resume unit
tests (the real ``PostPassJob``). The REAL, single-shared-detector proof that
the live overlay and the drain worker never call ``.predict`` concurrently
(INV-3) lives in ``tests/test_recording_e2e.py``'s AC3/AC6 test (a real
``CaptureLoop`` + real ``PostPassJob`` sharing one detector). This file's job
is the HTTP/status CONTRACT: response codes, the exact `/record/status` shape,
`drain.current`/`drain.queued` bookkeeping, `eta_seconds`, and entry_name
matching for retry -- using the controllable fake job.

Image-mode regression (AC9) is covered by the untouched ``tests/test_api.py``
staying green; here AC9 is a single additive check (``/status`` gains
``recording_state == "idle"`` when idle; ``/flag`` still works when idle).
"""

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.capture import CaptureLoop, Latest
from backend.dataset_writer import DatasetWriter
from backend.encoder import VideoProbe
from tests.fakes import FakeCapture, FakeDetector, make_fake_dets, make_fake_frame
from tests.recording_fakes import (
    FakeEncoder,
    decode_frame_index,
    make_numbered_frames,
)


# --- helpers ----------------------------------------------------------------


def _wait_for(predicate, timeout=3.0, interval=0.02) -> bool:
    """Bounded poll: returns True as soon as ``predicate()`` is truthy, else
    False after ``timeout`` (mirrors ``tests/test_api.py``'s helper). Never a
    bare/unbounded wait — a stuck machine fails the assertion fast."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _writer_factory(model_version="rfdetr-test"):
    def factory(output_path, dataset_name):
        return DatasetWriter(output_path, dataset_name, model_version)

    return factory


class _FakeJob:
    """Controllable stand-in for TR4/U2's ``PostPassJob``.

    Matches the frozen constructor shape (``entry_dir, entry_name, detector, *,
    keyframes, frame_count, mining_threshold, operator_threshold,
    model_version, ...``) and exposes ``run()`` + a ``status`` property whose
    value mirrors ``PostPassStatus`` (``.state/.done/.total/.error``), plus
    U2's ``pause()``/``resume()``/``cancel()`` seam.

    Two run modes, selected by ``frame_pace`` (set by ``_make_job_factory``
    right after construction — never through the frozen constructor's own
    kwargs, so every existing app-driven call site is untouched):

    - ``frame_pace is None`` (default) — the pre-U2 "block until released"
      model: ``run()`` sits parked (polling, so it can still be interrupted by
      a pause/cancel request) until a test calls ``release.set()``, at which
      point it jumps straight to ``final_state`` (``"done"`` or ``"failed"``).
      ``done``/``state`` may be driven directly by a test in the meantime
      (``jobs[0]._done = 20``) exactly as before U2. On a FAILED release,
      ``done`` is left wherever the test put it (not forced to ``total`` —
      the frozen contract requires ``done < total`` for a failed head, §5).
    - ``frame_pace`` set to a float (seconds/frame) — the U2 "autonomous
      per-frame" model: ``run()`` advances ``done`` 0→``total`` on its own at
      that pace, checking pause/cancel between every frame — this is what
      lets a test observe genuine in-flight progress and a pause landing
      "within one frame" (AC2 FIFO, AC3 pause/resume).

    ``pause()``/``resume()``/``cancel()`` work identically in both modes:
    ``pause()`` requests a park at the CURRENT frame (``status.state ==
    "paused"``, ``done`` frozen); ``resume()`` un-parks it, continuing from
    that same ``done`` (never reset to 0, INV-4); ``cancel()`` stops it
    (``status.state == "cancelled"``) from either a running or parked state.
    """

    def __init__(
        self,
        entry_dir,
        entry_name,
        detector,
        *,
        keyframes,
        frame_count,
        mining_threshold,
        operator_threshold,
        model_version,
        **kwargs,
    ):
        self.entry_dir = Path(entry_dir)
        self.entry_name = entry_name
        self.detector = detector
        self.keyframes = keyframes
        self.frame_count = frame_count
        self.mining_threshold = mining_threshold
        self.operator_threshold = operator_threshold
        self.model_version = model_version
        self.kwargs = kwargs

        self.started = threading.Event()
        self.release = threading.Event()
        self.final_state = "done"
        self.error_message = "boom: post-pass exploded"

        self._state = "pending"
        self._done = 0
        self._total = frame_count
        self._error = None

        # --- U2 additions ---
        self._cancelled = threading.Event()
        self._pause_requested = threading.Event()
        self._resume_requested = threading.Event()
        self.paused_event = threading.Event()  # set exactly while parked
        self.pause_calls = 0
        self.resume_calls = 0
        # Set post-construction by _make_job_factory; None => legacy mode.
        self.frame_pace: float | None = None

    def run(self) -> None:
        self._state = "running"
        self.started.set()
        while self._done < self._total:
            if self._cancelled.is_set():
                self._state = "cancelled"
                return
            if self._pause_requested.is_set():
                self._enter_paused()
                if self._cancelled.is_set():
                    self._state = "cancelled"
                    return
                continue
            if self.release.is_set():
                if self.final_state != "failed":
                    self._done = self._total  # a success always finishes 100%
                break  # a failure stops wherever `done` already was (< total)
            if self.frame_pace is None:
                # Legacy block-until-release: poll so pause/cancel still land.
                self._pause_requested.wait(timeout=0.02)
                continue
            # Autonomous per-frame mode: advance one simulated frame at a
            # deliberate pace so a concurrent pause() reliably lands within
            # one frame (bounded by frame_pace; every assertion built on this
            # still polls via ``_wait_for``, never a raw sleep-and-hope).
            time.sleep(self.frame_pace)
            self._done += 1
        if self._cancelled.is_set():
            self._state = "cancelled"
            return
        if self.final_state == "failed":
            self._error = self.error_message
        self._state = self.final_state

    def _enter_paused(self) -> None:
        self._state = "paused"
        self.paused_event.set()
        self._resume_requested.wait()
        self._resume_requested.clear()
        self.paused_event.clear()
        self._pause_requested.clear()
        if not self._cancelled.is_set():
            self._state = "running"

    def pause(self) -> None:
        self.pause_calls += 1
        self._pause_requested.set()

    def resume(self) -> None:
        self.resume_calls += 1
        self._resume_requested.set()

    def cancel(self) -> None:
        self._cancelled.set()
        self._resume_requested.set()  # unstick a paused park

    @property
    def status(self):
        return SimpleNamespace(
            state=self._state, done=self._done, total=self._total, error=self._error
        )


def _make_job_factory(frame_pace: float | None = None):
    jobs: list[_FakeJob] = []

    def factory(entry_dir, entry_name, detector, **kwargs):
        job = _FakeJob(entry_dir, entry_name, detector, **kwargs)
        job.frame_pace = frame_pace
        jobs.append(job)
        return job

    return factory, jobs


class RecordingStubCapture:
    """Deterministic ``CaptureLoop`` stand-in exposing exactly the surface the
    recording endpoints touch — no background thread, so state assertions are
    race-free. ``stop_recording`` releases the encoder (mirroring the real
    loop) so AC10 can assert the release. ``inference_paused`` tracks net
    pause/resume calls (True while more pauses than resumes have landed) —
    this stub has no live inference thread to race against; the REAL,
    concurrency-proof INV-3 check lives in ``tests/test_recording_e2e.py``."""

    def __init__(self, frame=None, frames_written=42, generation=0):
        self._frame = frame
        self._frames_written = frames_written
        self._generation = generation
        self._recording = False
        self.encoder = None
        self.start_recording_calls: list = []
        self.stop_recording_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0

    # --- snapshot surface ---
    def snapshot(self):
        if self._frame is None:
            return None
        return Latest(
            frame=self._frame,
            dets=make_fake_dets([]),
            threshold=0.5,
            overlay_jpeg=b"jpeg",
            count=0,
            frame_number=None,
        )

    def snapshot_with_generation(self):
        return self._generation, self.snapshot()

    def snapshot_at(self, generation):
        return self.snapshot() if generation == self._generation else None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def health(self) -> str:
        return "ok"

    @property
    def camera_index(self) -> int:
        return 0

    def set_camera(self, index: int) -> None:
        pass

    # --- recording surface ---
    def start_recording(self, encoder) -> None:
        if self._recording:
            raise RuntimeError("already recording")
        self.encoder = encoder
        self._recording = True
        self.start_recording_calls.append(encoder)

    def stop_recording(self) -> int:
        # Mirror the real CaptureLoop.stop_recording: capture the count to
        # return, then reset frames_written to 0 (idle). An impl that reads
        # capture.frames_written AFTER stop (instead of using this return value)
        # would get 0 against the real loop — so this reset is what forces AC6 to
        # depend on the returned int, not the post-stop property.
        self._recording = False
        self.stop_recording_calls += 1
        if self.encoder is not None:
            self.encoder.release()
        count = self._frames_written
        self._frames_written = 0
        return count

    def pause_inference(self) -> None:
        self.pause_calls += 1

    def resume_inference(self) -> None:
        self.resume_calls += 1

    @property
    def inference_paused(self) -> bool:
        return self.pause_calls > self.resume_calls

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def frames_written(self) -> int:
        return self._frames_written


def _build_app(
    capture,
    *,
    output_path,
    detector=None,
    capture_fps=30.0,
    mining_threshold=0.25,
    model_version="rfdetr-test",
    encoder=None,
    frame_pace=None,
):
    """Assemble the app around ``capture`` and override every recording
    injection point with a fake. Returns (app, detector, encoder, encoder_calls,
    jobs) so a test can inspect what the endpoints did through the seam.
    ``frame_pace`` (None by default) is forwarded to every job the factory
    builds — see ``_FakeJob`` for the two run modes it selects between."""
    detector = detector or FakeDetector(confidence_threshold=0.5)
    app = create_app(detector, _writer_factory(model_version), capture)
    app.state.output_path = str(output_path) if output_path is not None else None
    app.state.capture_fps = capture_fps
    app.state.mining_threshold = mining_threshold
    app.state.model_version = model_version

    enc = encoder or FakeEncoder()
    encoder_calls: list = []

    def open_encoder(path, fps, frame_size, **kwargs):
        encoder_calls.append(
            {"path": Path(path), "fps": fps, "frame_size": tuple(frame_size)}
        )
        return enc

    app.state.open_encoder = open_encoder
    app.state.probe_video = lambda path, **kw: VideoProbe(
        fps=capture_fps, frame_count=0, width=0, height=0
    )
    factory, jobs = _make_job_factory(frame_pace=frame_pace)
    app.state.post_pass_factory = factory
    return app, detector, enc, encoder_calls, jobs


def _release_jobs(jobs) -> None:
    """Unblock every fake job's ``run()`` so no worker thread is left parked —
    covers both the legacy block-until-release park and a pause() park."""
    for job in jobs:
        job.release.set()
        job.resume()


def _status(client) -> dict:
    return client.get("/record/status").json()


def _drain(client) -> dict:
    return _status(client)["drain"]


def _state(client) -> str:
    return _status(client)["state"]


# ---------------------------------------------------------------------------
# AC1 — the blocking endpoints are plain ``def`` (folder mint, encoder open,
# thread kick stay off the event loop). May legitimately PASS against the stub
# (stub endpoints are already plain def) — that is expected and fine for AC1.
# ---------------------------------------------------------------------------


def test_ac1_recording_endpoints_are_plain_def():
    app = create_app(FakeDetector(), _writer_factory(), RecordingStubCapture())
    endpoints = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}

    for path in (
        "/keyframe",
        "/record/start",
        "/record/stop",
        "/record/discard",
        "/record/retry",
        "/flag",
    ):
        assert path in endpoints, f"missing route {path}"
        assert not inspect.iscoroutinefunction(
            endpoints[path]
        ), f"{path} must be plain def (blocking I/O off the event loop)"


# ---------------------------------------------------------------------------
# AC2 — /record/start on idle: mint the entry, open the encoder at capture_fps
# with frame_size from the snapshot dims, freeze operator threshold, call
# capture.start_recording, transition to recording, echo the frozen threshold.
# ---------------------------------------------------------------------------


def test_ac2_start_records_and_opens_encoder_at_capture_fps_and_snapshot_size(tmp_path):
    frame = make_fake_frame(1920, 1080)  # (H, W, 3) -> frame_size (1920, 1080)
    capture = RecordingStubCapture(frame=frame)
    detector = FakeDetector(confidence_threshold=0.42)
    app, _detector, enc, encoder_calls, _jobs = _build_app(
        capture, output_path=tmp_path, detector=detector, capture_fps=24.0
    )
    client = TestClient(app)

    resp = client.post("/record/start", json={"entry_base": "take1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["entry_name"] == "take1_001"  # resolved base_NNN (U1)
    assert body["operator_threshold"] == 0.42  # frozen = detector threshold NOW

    # Encoder opened once, at capture_fps, with frame_size from the snapshot.
    assert len(encoder_calls) == 1
    call = encoder_calls[0]
    assert call["fps"] == 24.0
    assert call["frame_size"] == (1920, 1080)
    expected_mp4 = tmp_path / "videos" / "take1_001" / "video" / "take1_001.mp4"
    assert call["path"] == expected_mp4

    # The entry's video dir was minted, and capture.start_recording got the enc.
    assert (tmp_path / "videos" / "take1_001" / "video").is_dir()
    assert capture.start_recording_calls == [enc]

    # State advanced to recording.
    assert _state(client) == "recording"
    assert client.get("/status").json()["recording_state"] == "recording"


# ---------------------------------------------------------------------------
# AC3 — start guards: already recording -> 409; a draining/failed-head queue
# no longer blocks a new recording (U2 inverts the pre-U2 processing/failed
# 409s — see AC1/AC3 below); 422 on invalid entry_base. U1: the folder-
# collision guard is gone — a repeat base auto-suffixes instead.
# ---------------------------------------------------------------------------


def test_ac3_start_same_base_auto_suffixes_no_folder_collision(tmp_path):
    # A pre-existing videos/taken_001/ must NOT 409 a start with base "taken" —
    # U1 replaces the old folder-collision guard with counter auto-suffix.
    (tmp_path / "videos" / "taken_001").mkdir(parents=True)
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    resp = client.post("/record/start", json={"entry_base": "taken"})
    assert resp.status_code == 200
    assert resp.json()["entry_name"] == "taken_002"


def test_ac3_start_409_when_already_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_base": "a"}).status_code == 200
    assert client.post("/record/start", json={"entry_base": "b"}).status_code == 409


def test_ac3_start_200_while_draining_pauses_the_job(tmp_path):
    # U2/AC1 inverts the pre-U2 ``test_ac3_start_409_when_processing``: a new
    # recording is ACCEPTED even while the prior take's post-pass is mid-drain
    # — doing so PAUSES that job rather than racing it for the shared
    # detector. The deep pause/reclaim/resume assertions live in
    # test_u2_ac3_* below; this is the "start guard" regression: 200, not 409.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        assert _wait_for(lambda: _drain(client)["current"] is not None)

        resp = client.post("/record/start", json={"entry_base": "take2"})
        assert resp.status_code == 200  # <- was 409 pre-U2
        assert _state(client) == "recording"
    finally:
        _release_jobs(jobs)


def test_ac3_start_200_while_failed_head_blocks_queue(tmp_path):
    # U2/AC1 inverts ``test_ac3_start_409_when_failed``: a failed head halts
    # the QUEUE (§5), but never blocks a new recording — only
    # ``state == "recording"`` does that (endpoint contract table).
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        jobs[0].final_state = "failed"
        jobs[0].release.set()
        assert _wait_for(lambda: _status(client)["error"] is not None)

        resp = client.post("/record/start", json={"entry_base": "take2"})
        assert resp.status_code == 200  # <- was 409 pre-U2
        assert _state(client) == "recording"
        # The failed head is untouched by the new recording.
        assert _status(client)["error"] is not None
    finally:
        _release_jobs(jobs)


def test_ac3_start_409_when_no_output_path(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=None)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_base": "a"}).status_code == 409


def test_ac3_start_409_when_no_frame_captured_yet(tmp_path):
    # No snapshot yet -> frame_size for the encoder is unknowable -> 409, never
    # an unhandled AttributeError.
    capture = RecordingStubCapture(frame=None)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_base": "a"}).status_code == 409


def test_ac3_start_422_on_invalid_entry_base(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_base": "bad/name"}).status_code == 422
    assert client.post("/record/start", json={"entry_base": ".hidden"}).status_code == 422
    assert client.post("/record/start", json={"entry_base": ""}).status_code == 422


# ---------------------------------------------------------------------------
# AC4 — /keyframe: dedups a repeat frame_number, returns the count; 409 when
# not recording; 422 for frame_number < 0 or >= frames_written.
# ---------------------------------------------------------------------------


def test_ac4_keyframe_dedups_and_counts_while_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=100)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_base": "a"}).raise_for_status()

    assert client.post("/keyframe", json={"frame_number": 5}).json()["n_keyframes"] == 1
    # A repeat of the same frame does NOT grow the set.
    assert client.post("/keyframe", json={"frame_number": 5}).json()["n_keyframes"] == 1
    assert client.post("/keyframe", json={"frame_number": 6}).json()["n_keyframes"] == 2


def test_ac4_keyframe_409_when_not_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=100)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/keyframe", json={"frame_number": 3}).status_code == 409


def test_ac4_keyframe_422_out_of_range(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=100)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_base": "a"}).raise_for_status()

    assert client.post("/keyframe", json={"frame_number": -1}).status_code == 422
    # >= frames_written (beyond frames written) is rejected; the last valid is 99.
    assert client.post("/keyframe", json={"frame_number": 100}).status_code == 422
    assert client.post("/keyframe", json={"frame_number": 99}).status_code == 200


# ---------------------------------------------------------------------------
# AC5 — X-Frame-Number: while recording, /frame carries the on-screen
# snapshot's frame_number; absent when idle. Real CaptureLoop + numbered camera.
# ---------------------------------------------------------------------------


def test_ac5_frame_carries_frame_number_while_recording(tmp_path):
    frames = make_numbered_frames(4000, 64, 48)
    capture = CaptureLoop(
        FakeDetector(),
        camera_index=0,
        render_fn=lambda frame, dets: frame,
        cap_factory=lambda idx: FakeCapture(frames=frames, on_empty="repeat"),
        stale_after_s=5.0,
        # Large ring so the generation returned by /frame can't be evicted before
        # snapshot_at(gen) re-reads it: the real loop runs fast over these tiny
        # frames, and under full-suite CPU contention >16 generations (the default
        # ring) can roll over between the HTTP response and the next line — a test
        # -infra race, not an impl bug. Sizing the ring past the test's frame count
        # makes the atomic header<->Latest comparison deterministic.
        ring_size=5000,
    )
    capture.start()
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        assert _wait_for(lambda: capture.generation > 0)

        # Idle: no X-Frame-Number header.
        r_idle = client.get("/frame")
        assert r_idle.status_code == 200
        assert "x-frame-number" not in r_idle.headers

        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()

        # Wait until a real recording tick has actually published a
        # recording-stamped Latest — i.e. an on-screen snapshot whose
        # frame_number is non-None. Until predict+publish runs, snapshot() still
        # holds the last idle frame (frame_number None) and /frame correctly
        # omits the header, so polling the capture state directly (not the HTTP
        # header, which lags a tick) is the deterministic signal for AC5.
        assert _wait_for(
            lambda: capture.snapshot() is not None
            and capture.snapshot().frame_number is not None
        )

        # Recording: the header equals the exact generation's Latest.frame_number
        # — the on-screen snapshot's frame_number, i.e. the value the client
        # echoes to /keyframe. Tie header -> generation -> Latest atomically via
        # the ring so the header can never be compared against a different frame.
        r = client.get("/frame")
        assert "x-frame-number" in r.headers
        gen = int(r.headers["x-frame-generation"])
        latest = capture.snapshot_at(gen)
        assert latest is not None
        assert latest.frame_number is not None
        assert int(r.headers["x-frame-number"]) == latest.frame_number

        # The numbered fake camera confirms the on-screen frame is a real,
        # per-frame-indexed capture. Decode the RAW snapshot pixels (never the
        # lossy served JPEG, which chroma-subsampling would corrupt): its
        # absolute content index must be >= its recording-local frame_number,
        # since frame_number = absolute_index - (idle frames consumed before this
        # recording started) and that base offset is non-negative.
        assert decode_frame_index(latest.frame) >= latest.frame_number
    finally:
        capture.stop()


# ---------------------------------------------------------------------------
# AC6 — /record/stop returns immediately (never blocks), enqueues the job,
# `state` goes straight to "idle", pause_inference fires, the job drains on
# the background worker. On completion the job clears from drain.current and
# inference resumes; on FAILURE it halts as drain.current with `error` set
# (§5) — never a foreground "failed" state.
# ---------------------------------------------------------------------------


def test_ac6_stop_enqueues_and_drains_to_completion(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(1920, 1080), frames_written=50)
    detector = FakeDetector(confidence_threshold=0.42)
    app, _detector, _enc, _calls, jobs = _build_app(
        capture, output_path=tmp_path, detector=detector, mining_threshold=0.25
    )
    client = TestClient(app)
    try:
        start = client.post("/record/start", json={"entry_base": "take1"})
        start.raise_for_status()
        resolved = start.json()["entry_name"]
        client.post("/keyframe", json={"frame_number": 5}).raise_for_status()
        client.post("/keyframe", json={"frame_number": 6}).raise_for_status()
        # Freeze must not follow the live slider after start.
        detector.confidence_threshold = 0.9

        resp = client.post("/record/stop")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "frames_written": 50}

        # Immediately idle (AC1) — never a blocking "processing" state.
        assert _state(client) == "idle"

        # The job was enqueued and picked up; inference paused; worker launched.
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        assert _wait_for(lambda: capture.pause_calls == 1)

        job = jobs[0]
        # The job was built from the frozen record-start state (resolved name).
        assert job.entry_dir == tmp_path / "videos" / resolved
        assert job.entry_name == resolved
        assert job.frame_count == 50
        assert job.operator_threshold == 0.42
        assert job.mining_threshold == 0.25
        assert job.model_version == "rfdetr-test"
        assert set(job.keyframes) == {5, 6}

        # Visible as drain.current while draining.
        assert _wait_for(
            lambda: _drain(client)["current"]
            == {"entry_name": resolved, "done": job.status.done, "total": 50}
        )

        # Let the job finish -> drain.current clears, inference resumed.
        job.release.set()
        assert _wait_for(lambda: _drain(client)["current"] is None)
        assert _wait_for(lambda: capture.resume_calls == 1)
        assert _status(client)["error"] is None
        assert _state(client) == "idle"
    finally:
        _release_jobs(jobs)


def test_ac6_stop_job_failure_leaves_it_as_drain_current_with_error(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=12)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        start = client.post("/record/start", json={"entry_base": "take1"})
        start.raise_for_status()
        resolved = start.json()["entry_name"]
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        jobs[0].final_state = "failed"
        jobs[0].error_message = "boom: post-pass exploded"
        jobs[0].release.set()

        # Halt-on-failure (§5): stays as drain.current with done < total,
        # top-level error set, `state` remains "idle" (NEVER "failed").
        assert _wait_for(lambda: _status(client)["error"] is not None)
        body = _status(client)
        assert body["state"] == "idle"
        assert body["drain"]["current"]["entry_name"] == resolved
        assert body["drain"]["current"]["done"] < body["drain"]["current"]["total"]
        assert "boom" in body["error"]
        # resume_inference must fire on the failure path too (detector released).
        assert _wait_for(lambda: capture.resume_calls == 1)
    finally:
        _release_jobs(jobs)


def test_ac6_stop_409_when_not_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/stop").status_code == 409


# ---------------------------------------------------------------------------
# AC7 (frozen §2/§5 — "AC5" in the new U2 numbering) — /record/status returns
# EXACTLY {state, drain:{current, queued, eta_seconds}, error}; the pre-U2
# entry_name/frames_written/n_keyframes/postpass top-level fields are gone.
# drain.current reports the live/failed job's {entry_name, done, total};
# error surfaces only for a failed head.
# ---------------------------------------------------------------------------


def test_ac7_record_status_exact_shape(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=42)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    idle_body = _status(client)
    assert set(idle_body.keys()) == {"state", "drain", "error"}
    assert set(idle_body["drain"].keys()) == {"current", "queued", "eta_seconds"}
    assert idle_body["state"] == "idle"
    assert idle_body["drain"] == {"current": None, "queued": [], "eta_seconds": 0}
    assert idle_body["error"] is None

    client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
    recording_body = _status(client)
    assert set(recording_body.keys()) == {"state", "drain", "error"}
    assert recording_body["state"] == "recording"


def test_ac7_record_status_drain_current_reports_live_progress(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=42)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        start = client.post("/record/start", json={"entry_base": "take1"})
        start.raise_for_status()
        resolved = start.json()["entry_name"]
        client.post("/keyframe", json={"frame_number": 3}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        # Drive the job's reported progress while it is parked mid-drain.
        jobs[0]._done = 20
        assert _wait_for(
            lambda: _drain(client)["current"]
            == {"entry_name": resolved, "done": 20, "total": 42}
        )
        assert _status(client)["error"] is None
    finally:
        _release_jobs(jobs)


def test_ac7_record_status_surfaces_error_only_for_failed_head(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        # While draining but not yet failed, error stays null.
        assert _status(client)["error"] is None

        jobs[0].final_state = "failed"
        jobs[0].error_message = "boom: post-pass exploded"
        jobs[0].release.set()

        assert _wait_for(lambda: _status(client)["error"] is not None)
        body = _status(client)
        assert "boom" in body["error"]
        assert body["drain"]["current"]["entry_name"] == jobs[0].entry_name
    finally:
        _release_jobs(jobs)


# ---------------------------------------------------------------------------
# AC8 — /flag during recording -> 409 with a "stop recording to snapshot
# stills" detail.
# ---------------------------------------------------------------------------


def test_ac8_flag_blocked_while_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()

    resp = client.post("/flag")
    assert resp.status_code == 409
    assert "stop recording to snapshot stills" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# AC9 — additive idle behaviour (the image-mode regression itself lives in the
# untouched tests/test_api.py): /status carries recording_state == "idle" when
# idle, and /flag behaves normally when idle. (This may pass against the stub —
# expected; it is the regression gate.)
# ---------------------------------------------------------------------------


def test_ac9_status_idle_and_flag_works_when_idle(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(1920, 1080))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.get("/status").json()["recording_state"] == "idle"

    client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "ds"}
    ).raise_for_status()
    resp = client.post("/flag")
    assert resp.status_code == 200
    assert resp.json()["n_flagged"] == 1
    assert client.get("/status").json()["recording_state"] == "idle"


# ---------------------------------------------------------------------------
# AC10 — /record/discard: from recording (abort) stops+releases the encoder
# and deletes the folder (-> idle); from a draining/failed drain.current
# deletes ITS folder and advances the queue; 409 from a fully idle machine
# (nothing recording, nothing draining).
# ---------------------------------------------------------------------------


def test_ac10_discard_from_recording_aborts_and_deletes(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, _detector, enc, _calls, _jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    start = client.post("/record/start", json={"entry_base": "take1"})
    start.raise_for_status()
    resolved = start.json()["entry_name"]
    assert (tmp_path / "videos" / resolved).exists()

    resp = client.post("/record/discard")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert capture.stop_recording_calls == 1
    assert enc.is_open is False  # encoder released
    assert not (tmp_path / "videos" / resolved).exists()  # folder removed
    assert _state(client) == "idle"


def test_ac10_discard_drops_the_failed_head_and_advances(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        start = client.post("/record/start", json={"entry_base": "take1"})
        start.raise_for_status()
        resolved = start.json()["entry_name"]
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        jobs[0].final_state = "failed"
        jobs[0].release.set()
        assert _wait_for(lambda: _status(client)["error"] is not None)

        resp = client.post("/record/discard")
        assert resp.status_code == 200
        # INV-5's ONLY sanctioned delete.
        assert not (tmp_path / "videos" / resolved).exists()
        assert _wait_for(lambda: _drain(client)["current"] is None)
        assert _status(client)["error"] is None
        assert _state(client) == "idle"
    finally:
        _release_jobs(jobs)


def test_ac10_discard_drops_the_draining_current_job(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        start = client.post("/record/start", json={"entry_base": "take1"})
        start.raise_for_status()
        resolved = start.json()["entry_name"]
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        assert _wait_for(lambda: _drain(client)["current"] is not None)

        resp = client.post("/record/discard")
        assert resp.status_code == 200
        assert not (tmp_path / "videos" / resolved).exists()
        assert _wait_for(lambda: _drain(client)["current"] is None)
        assert _state(client) == "idle"
    finally:
        _release_jobs(jobs)


def test_ac10_discard_from_idle_returns_409(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/discard").status_code == 409


# ---------------------------------------------------------------------------
# AC11 — /record/retry {entry_name}: valid only when drain.current is a
# failed head whose entry_name matches; clears the error and re-drains from
# frame 0; 409 from any other state (idle/recording/draining-not-failed) or a
# mismatched entry_name.
# ---------------------------------------------------------------------------


def test_ac11_retry_from_failed_head_reruns_postpass(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=9)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        start = client.post("/record/start", json={"entry_base": "take1"})
        start.raise_for_status()
        resolved = start.json()["entry_name"]
        client.post("/keyframe", json={"frame_number": 2}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        jobs[0].final_state = "failed"
        jobs[0].release.set()
        assert _wait_for(lambda: _status(client)["error"] is not None)

        resp = client.post("/record/retry", json={"entry_name": resolved})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # Clears the error; a fresh job (second instance) re-drains from frame 0.
        assert _wait_for(lambda: _status(client)["error"] is None)
        assert _wait_for(lambda: len(jobs) == 2 and jobs[1].started.is_set())
        retry_job = jobs[1]
        assert retry_job.entry_dir == tmp_path / "videos" / resolved
        assert retry_job.entry_name == resolved
        assert set(retry_job.keyframes) == {2}
        assert retry_job.frame_count == 9

        retry_job.release.set()
        assert _wait_for(lambda: _drain(client)["current"] is None)
    finally:
        _release_jobs(jobs)


def test_ac11_retry_entry_name_mismatch_returns_409(tmp_path):
    # Retry is scoped to the CURRENT failed head by name (§5) — a mismatched
    # entry_name (stale UI, wrong queued entry, typo, ...) 409s and leaves the
    # failed head untouched.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        jobs[0].final_state = "failed"
        jobs[0].release.set()
        assert _wait_for(lambda: _status(client)["error"] is not None)

        resp = client.post("/record/retry", json={"entry_name": "not-the-failed-one"})
        assert resp.status_code == 409
        assert _status(client)["error"] is not None  # untouched
    finally:
        _release_jobs(jobs)


def test_ac11_retry_from_idle_returns_409(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/retry", json={"entry_name": "whatever"}).status_code == 409


def test_ac11_retry_while_draining_not_failed_returns_409(tmp_path):
    # Retry is valid only against a FAILED head — never a live/in-flight drain.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        start = client.post("/record/start", json={"entry_base": "take1"})
        start.raise_for_status()
        resolved = start.json()["entry_name"]
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: _drain(client)["current"] is not None)

        assert client.post("/record/retry", json={"entry_name": resolved}).status_code == 409
        assert _status(client)["error"] is None
    finally:
        _release_jobs(jobs)


def test_ac11_retry_from_recording_returns_409(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()

    assert client.post("/record/retry", json={"entry_name": "take1_001"}).status_code == 409


# ---------------------------------------------------------------------------
# AC12 — one recording at a time: two concurrent /record/start requests yield
# exactly one 200 (recording) and one 409 (serialized on recording_lock).
# ---------------------------------------------------------------------------


def test_ac12_concurrent_start_yields_exactly_one_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, _detector, _enc, _calls, _jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    barrier = threading.Barrier(2)
    results: list = []

    def worker(name):
        barrier.wait()  # fire both starts as simultaneously as possible
        results.append(client.post("/record/start", json={"entry_base": name}))

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    codes = sorted(r.status_code for r in results)
    assert codes == [200, 409]  # exactly one winner, one loser
    # Only the winner ever started the recording.
    assert len(capture.start_recording_calls) == 1
    assert _state(client) == "recording"


# ---------------------------------------------------------------------------
# U2 acceptance criteria (docs/tasks/U2-queue.md §7) — the queue/pause/resume/
# eta behavior itself, layered on top of the endpoint-guard migrations above.
# AC7 there ("regressions green + new concurrency tests") is the whole suite
# staying green, not a single discrete test.
# ---------------------------------------------------------------------------


def test_u2_ac1_stop_response_shape_and_start_accepted_while_draining(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=17)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
        resp = client.post("/record/stop")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "frames_written": 17}
        assert _state(client) == "idle"  # immediate, never "processing"

        resp2 = client.post("/record/start", json={"entry_base": "take2"})
        assert resp2.status_code == 200  # accepted even while take1 drains
    finally:
        _release_jobs(jobs)


def test_u2_ac2_fifo_drain_two_queued_jobs_one_at_a_time(tmp_path):
    # Two jobs queued back-to-back; the worker drains them strictly one at a
    # time in enqueue order (INV-6): drain.current walks job1 -> job2 -> None,
    # drain.queued shrinks front-to-back, and job2 never starts while job1 is
    # still live. Uses the fake's autonomous per-frame pacing (frame_pace) so
    # both jobs complete on their own — no manual release needed.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=50)
    app, _detector, _enc, _calls, jobs = _build_app(
        capture, output_path=tmp_path, frame_pace=0.01
    )
    client = TestClient(app)
    try:
        start1 = client.post("/record/start", json={"entry_base": "take1"})
        start1.raise_for_status()
        resolved1 = start1.json()["entry_name"]
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        # Start+stop take2 WHILE take1 still drains (AC1) -- job2 is enqueued
        # behind job1, not started (FIFO, one at a time).
        start2 = client.post("/record/start", json={"entry_base": "take2"})
        assert start2.status_code == 200
        resolved2 = start2.json()["entry_name"]
        client.post("/record/stop").raise_for_status()
        assert len(jobs) == 2
        job1, job2 = jobs

        # job2 must not begin running while job1 is still the live current.
        assert _drain(client)["current"]["entry_name"] == resolved1
        assert _drain(client)["queued"] == [resolved2]
        assert job2.started.is_set() is False

        # Both eventually complete, strictly in order.
        assert _wait_for(lambda: job1.status.state == "done", timeout=5.0)
        assert _wait_for(lambda: job2.started.is_set(), timeout=5.0)
        assert _wait_for(
            lambda: _drain(client)["current"] is None and _drain(client)["queued"] == [],
            timeout=5.0,
        )
        assert job2.status.state == "done"
    finally:
        _release_jobs(jobs)


def test_u2_ac3_pause_within_one_frame_reclaims_detector_and_resumes_same_job(tmp_path):
    """AC3 (crux). Contract-level proof using the enriched ``_FakeJob``'s
    autonomous per-frame pacing: (a) the SAME drain job pauses shortly after
    ``/record/start`` (``done`` stops advancing, ``status.state ==
    "paused"``); (b) the detector was reclaimed for the live overlay BEFORE
    recording began (``capture.resume_calls`` — INV-3's REAL, concurrent-
    predict-proof lives in ``tests/test_recording_e2e.py``'s AC3/AC6 test,
    since this fake capture has no live inference thread to race against);
    (c) after ``/record/stop`` returns to idle, the IDENTICAL job (never
    replaced) resumes from its parked frame count, not 0 (INV-4 — the strict,
    byte-identical-output proof lives in ``tests/test_postpass.py``'s
    pause/resume unit tests, which exercise the REAL ``PostPassJob``).
    """
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=40)
    app, _detector, _enc, _calls, jobs = _build_app(
        capture, output_path=tmp_path, frame_pace=0.01
    )
    client = TestClient(app)
    try:
        start1 = client.post("/record/start", json={"entry_base": "take1"})
        start1.raise_for_status()
        resolved1 = start1.json()["entry_name"]
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        job1 = jobs[0]

        # Let it genuinely make progress before we interrupt it.
        assert _wait_for(lambda: job1.status.done >= 3, timeout=3.0)

        resp = client.post("/record/start", json={"entry_base": "take2"})
        assert resp.status_code == 200
        assert _state(client) == "recording"

        # (a) paused promptly: state flips to "paused", and done then stays
        # put across a short bounded re-check window (never advances further
        # while parked).
        assert _wait_for(lambda: job1.status.state == "paused", timeout=3.0)
        done_at_pause = job1.status.done
        assert _wait_for(lambda: job1.status.done != done_at_pause, timeout=0.3) is False

        # Still the SAME job reported as drain.current — never replaced.
        current = _drain(client)["current"]
        assert current is not None
        assert current["entry_name"] == resolved1
        assert current["done"] == done_at_pause

        # (b) detector reclaimed before recording began.
        assert capture.resume_calls >= 1

        # (c) stop take2 -> idle; the SAME job resumes from done_at_pause (not
        # 0) and runs to completion.
        client.post("/record/stop").raise_for_status()
        assert len(jobs) == 2
        assert jobs[0] is job1  # never re-constructed

        assert _wait_for(lambda: job1.status.state in ("running", "done"), timeout=3.0)
        # Never reset to 0 -- resumed from (at least) the parked index.
        assert job1.status.done >= done_at_pause

        assert _wait_for(lambda: job1.status.state == "done", timeout=5.0)
        assert job1.status.done == job1.status.total
    finally:
        _release_jobs(jobs)


def test_u2_ac4_failed_job_folder_persists_until_discard_or_retry(tmp_path):
    # INV-5 (hard): a failed job never deletes its Entry folder on its own —
    # only an explicit /record/discard does. The folder (with its finalized
    # MP4) sits there, untouched, for as long as the operator leaves it failed.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        start = client.post("/record/start", json={"entry_base": "take1"})
        start.raise_for_status()
        resolved = start.json()["entry_name"]
        entry_dir = tmp_path / "videos" / resolved
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        assert entry_dir.exists()  # MP4 finalized at /record/stop, pre-enqueue

        jobs[0].final_state = "failed"
        jobs[0].release.set()
        assert _wait_for(lambda: _status(client)["error"] is not None)

        # Still there, and stays there across a bounded wait -- nothing
        # auto-deletes a failed job's folder.
        assert entry_dir.exists()
        assert _wait_for(lambda: not entry_dir.exists(), timeout=0.5) is False

        # Retry rebuilds it in place rather than the folder ever vanishing.
        resp = client.post("/record/retry", json={"entry_name": resolved})
        assert resp.status_code == 200
        assert entry_dir.exists()
    finally:
        _release_jobs(jobs)


def test_u2_ac5_eta_seconds_formula(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=10)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        # job1: current, mid-drain (never auto-advances -- legacy block mode).
        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        # job2: enqueued behind job1 while job1 is still draining (AC1).
        capture._frames_written = 5
        client.post("/record/start", json={"entry_base": "take2"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert len(jobs) == 2
        job1, job2 = jobs

        job1._done = 3  # 3 of 10 done -> 7 remaining
        app.state.detect_fps = 2.0

        body = _status(client)
        assert body["drain"]["current"] == {
            "entry_name": job1.entry_name, "done": 3, "total": 10,
        }
        assert body["drain"]["queued"] == [job2.entry_name]
        # remaining_frames = (10 - 3) + 5 = 12; eta = 12 / 2.0 = 6.0
        assert body["drain"]["eta_seconds"] == pytest.approx(6.0)
    finally:
        _release_jobs(jobs)


def test_u2_ac5_eta_seconds_zero_when_nothing_queued_or_draining(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    app.state.detect_fps = 0.6

    assert _drain(client)["eta_seconds"] == 0


def test_u2_ac6_pause_reclaims_detector_before_recording_begins(tmp_path):
    # Contract-level proof of INV-3 using RecordingStubCapture's counters/flag
    # (this stub has no live inference thread to race against — the decisive
    # "never concurrent" proof over a REAL shared detector lives in
    # tests/test_recording_e2e.py). By the time /record/start returns 200
    # (recording has begun), the detector must already have been reclaimed
    # from the draining job — capture.inference_paused is False.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        assert _wait_for(lambda: capture.inference_paused is True)  # worker owns it

        resp = client.post("/record/start", json={"entry_base": "take2"})
        assert resp.status_code == 200
        # Recording began -> the detector must already be back with the live
        # overlay (INV-3: no instant where both could call .predict).
        assert capture.inference_paused is False
    finally:
        _release_jobs(jobs)
