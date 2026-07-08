"""TR5 AC1-AC12: behavior contract for the recording-mode API + state machine.

Written from ``docs/tasks/TR5-api.md`` (the frozen SPEC — endpoint table, state
diagram, 12 ACs) and ``docs/RECORDING.md`` §API & state / §Thresholds /
§Detector sharing / §Encoder. Blind-TDD: this file exercises the HTTP behaviour
contract via ``fastapi.testclient.TestClient`` against the frozen ``create_app``
seam; the coder replaces ``backend/app.py``'s Phase-0 recording stubs without
ever seeing this file.

Everything is camera-free / ffmpeg-free / onnx-free:

- ``app.state.open_encoder`` is overridden with a factory returning TR1's
  ``FakeEncoder`` (``tests/recording_fakes.py``) so no real ffmpeg runs, and so
  the call args (path / fps / frame_size) are inspectable.
- ``app.state.post_pass_factory`` is overridden with ``_FakeJob`` — a
  controllable job whose ``run()`` blocks on an event the test releases, so the
  ``processing -> idle`` / ``processing -> failed`` transitions (kicked on a
  worker thread) are observed deterministically via a bounded ``_wait_for``
  poll of ``/record/status`` — never sleep-and-hope.
- ``app.state.probe_video`` is overridden with a fake returning a ``VideoProbe``.
- ``RecordingStubCapture`` is a deterministic capture stand-in exposing exactly
  the recording surface the endpoints touch (``snapshot``, ``start_recording``,
  ``stop_recording``, ``frames_written``, ``pause_inference`` /
  ``resume_inference``). AC5 (X-Frame-Number) uses a REAL ``CaptureLoop`` +
  numbered ``FakeCapture`` because a live loop is what makes that header true.

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
    """Controllable stand-in for TR4's ``PostPassJob``.

    Matches the frozen constructor shape (``entry_dir, entry_name, detector, *,
    keyframes, frame_count, mining_threshold, operator_threshold,
    model_version, ...``) and exposes ``run()`` + a ``status`` property whose
    value mirrors ``PostPassStatus`` (``.state/.done/.total/.error``).

    ``run()`` blocks on ``release`` so a test can hold the machine in
    ``processing`` and assert the transition, then release with
    ``final_state`` = ``"done"`` (-> recording idle) or ``"failed"`` (->
    recording failed). ``started`` is set the moment ``run()`` begins so a test
    can confirm the worker thread actually launched.
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

    def run(self) -> None:
        self._state = "running"
        self.started.set()
        self.release.wait()
        if self.final_state == "failed":
            self._error = self.error_message
        self._done = self._total
        self._state = self.final_state

    @property
    def status(self):
        return SimpleNamespace(
            state=self._state, done=self._done, total=self._total, error=self._error
        )


def _make_job_factory():
    jobs: list[_FakeJob] = []

    def factory(entry_dir, entry_name, detector, **kwargs):
        job = _FakeJob(entry_dir, entry_name, detector, **kwargs)
        jobs.append(job)
        return job

    return factory, jobs


class RecordingStubCapture:
    """Deterministic ``CaptureLoop`` stand-in exposing exactly the surface the
    recording endpoints touch — no background thread, so state assertions are
    race-free. ``stop_recording`` releases the encoder (mirroring the real
    loop) so AC10 can assert the release."""

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
):
    """Assemble the app around ``capture`` and override every recording
    injection point with a fake. Returns (app, detector, encoder, encoder_calls,
    jobs) so a test can inspect what the endpoints did through the seam."""
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
    factory, jobs = _make_job_factory()
    app.state.post_pass_factory = factory
    return app, detector, enc, encoder_calls, jobs


def _release_jobs(jobs) -> None:
    """Unblock every fake job's ``run()`` so no worker thread is left parked."""
    for job in jobs:
        job.release.set()


def _state(client) -> str:
    return client.get("/record/status").json()["state"]


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

    resp = client.post("/record/start", json={"entry_name": "take1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["entry_name"] == "take1"
    assert body["operator_threshold"] == 0.42  # frozen = detector threshold NOW

    # Encoder opened once, at capture_fps, with frame_size from the snapshot.
    assert len(encoder_calls) == 1
    call = encoder_calls[0]
    assert call["fps"] == 24.0
    assert call["frame_size"] == (1920, 1080)
    expected_mp4 = tmp_path / "take1" / "video" / "take1.mp4"
    assert call["path"] == expected_mp4

    # The entry's video dir was minted, and capture.start_recording got the enc.
    assert (tmp_path / "take1" / "video").is_dir()
    assert capture.start_recording_calls == [enc]

    # State advanced to recording.
    assert _state(client) == "recording"
    assert client.get("/status").json()["recording_state"] == "recording"


# ---------------------------------------------------------------------------
# AC3 — start guards: 409 on collision / already recording / no output_path /
# no frame captured; 422 on invalid entry_name (single component, no leading .).
# ---------------------------------------------------------------------------


def test_ac3_start_409_on_existing_entry_folder(tmp_path):
    (tmp_path / "taken").mkdir()
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_name": "taken"}).status_code == 409


def test_ac3_start_409_when_already_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_name": "a"}).status_code == 200
    assert client.post("/record/start", json={"entry_name": "b"}).status_code == 409


def test_ac3_start_409_when_processing(tmp_path):
    # An in-flight post-pass must block a new recording — otherwise a second
    # take would run on top of the job that owns the detector. Guards against an
    # impl that gates start on `state == "recording"` instead of `!= "idle"`.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        # Job parked (blocked) so the machine stays in processing, not racing to idle.
        assert _wait_for(lambda: _state(client) == "processing")

        assert client.post("/record/start", json={"entry_name": "take2"}).status_code == 409
        assert _state(client) == "processing"  # still processing, no new recording
    finally:
        _release_jobs(jobs)


def test_ac3_start_409_when_failed(tmp_path):
    # A failed take still owns its folder pending retry/discard — starting a new
    # recording from `failed` must be rejected (only retry/discard leave failed).
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        jobs[0].final_state = "failed"
        jobs[0].release.set()
        assert _wait_for(lambda: _state(client) == "failed")

        assert client.post("/record/start", json={"entry_name": "take2"}).status_code == 409
        assert _state(client) == "failed"
    finally:
        _release_jobs(jobs)


def test_ac3_start_409_when_no_output_path(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=None)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_name": "a"}).status_code == 409


def test_ac3_start_409_when_no_frame_captured_yet(tmp_path):
    # No snapshot yet -> frame_size for the encoder is unknowable -> 409, never
    # an unhandled AttributeError.
    capture = RecordingStubCapture(frame=None)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_name": "a"}).status_code == 409


def test_ac3_start_422_on_invalid_entry_name(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_name": "bad/name"}).status_code == 422
    assert client.post("/record/start", json={"entry_name": ".hidden"}).status_code == 422
    assert client.post("/record/start", json={"entry_name": ""}).status_code == 422


# ---------------------------------------------------------------------------
# AC4 — /keyframe: dedups a repeat frame_number, returns the count; 409 when
# not recording; 422 for frame_number < 0 or >= frames_written.
# ---------------------------------------------------------------------------


def test_ac4_keyframe_dedups_and_counts_while_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=100)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_name": "a"}).raise_for_status()

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
    client.post("/record/start", json={"entry_name": "a"}).raise_for_status()

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

        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()

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
# AC6 — /record/stop returns immediately with frames_written, -> processing,
# calls pause_inference, runs the job on a worker thread; on finish -> idle
# (success) or failed, resume_inference called in BOTH cases. 409 if not
# recording.
# ---------------------------------------------------------------------------


def test_ac6_stop_kicks_postpass_and_completes_to_idle(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(1920, 1080), frames_written=50)
    detector = FakeDetector(confidence_threshold=0.42)
    app, _detector, _enc, _calls, jobs = _build_app(
        capture, output_path=tmp_path, detector=detector, mining_threshold=0.25
    )
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/keyframe", json={"frame_number": 5}).raise_for_status()
        client.post("/keyframe", json={"frame_number": 6}).raise_for_status()
        # Freeze must not follow the live slider after start.
        detector.confidence_threshold = 0.9

        resp = client.post("/record/stop")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "frames_written": 50}

        # Immediately in processing, inference paused, worker launched.
        assert _wait_for(lambda: _state(client) == "processing")
        assert capture.pause_calls == 1
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        job = jobs[0]
        # The job was built from the frozen record-start state.
        assert job.entry_dir == tmp_path / "take1"
        assert job.entry_name == "take1"
        assert job.frame_count == 50
        assert job.operator_threshold == 0.42
        assert job.mining_threshold == 0.25
        assert job.model_version == "rfdetr-test"
        assert set(job.keyframes) == {5, 6}

        # Let the job finish -> idle, inference resumed.
        job.release.set()
        assert _wait_for(lambda: _state(client) == "idle")
        assert _wait_for(lambda: capture.resume_calls == 1)
    finally:
        _release_jobs(jobs)


def test_ac6_stop_job_failure_moves_to_failed_and_resumes(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=12)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        jobs[0].final_state = "failed"
        jobs[0].release.set()

        assert _wait_for(lambda: _state(client) == "failed")
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
# AC7 — /record/status reflects the live machine: state, entry_name,
# frames_written (from capture during recording), n_keyframes, postpass
# {done,total} (from the job during processing), error only in failed.
# ---------------------------------------------------------------------------


def test_ac7_record_status_reflects_recording_then_processing(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=42)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/keyframe", json={"frame_number": 3}).raise_for_status()

        recording = client.get("/record/status").json()
        assert recording["state"] == "recording"
        assert recording["entry_name"] == "take1"
        assert recording["frames_written"] == 42  # live, from capture
        assert recording["n_keyframes"] == 1
        assert recording["error"] is None

        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        # Drive the job's reported progress while it is parked in processing.
        jobs[0]._done = 20
        assert _wait_for(
            lambda: client.get("/record/status").json()["postpass"]
            == {"done": 20, "total": 42}
        )
        processing = client.get("/record/status").json()
        assert processing["state"] == "processing"
        assert processing["error"] is None  # error only surfaces in failed
    finally:
        _release_jobs(jobs)


def test_ac7_record_status_surfaces_error_only_in_failed(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())

        jobs[0].final_state = "failed"
        jobs[0].error_message = "boom: post-pass exploded"
        jobs[0].release.set()

        assert _wait_for(lambda: _state(client) == "failed")
        body = client.get("/record/status").json()
        assert body["error"] is not None
        assert "boom" in body["error"]
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
    client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()

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
# AC10 — /record/discard: from recording (abort) stops+releases the encoder and
# deletes the folder (-> idle); from failed / processing deletes the folder;
# 409 from idle.
# ---------------------------------------------------------------------------


def test_ac10_discard_from_recording_aborts_and_deletes(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, _detector, enc, _calls, _jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
    assert (tmp_path / "take1").exists()

    resp = client.post("/record/discard")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert capture.stop_recording_calls == 1
    assert enc.is_open is False  # encoder released
    assert not (tmp_path / "take1").exists()  # folder removed
    assert _state(client) == "idle"


def test_ac10_discard_from_failed_deletes_folder(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        jobs[0].final_state = "failed"
        jobs[0].release.set()
        assert _wait_for(lambda: _state(client) == "failed")

        resp = client.post("/record/discard")
        assert resp.status_code == 200
        assert not (tmp_path / "take1").exists()
        assert _state(client) == "idle"
    finally:
        _release_jobs(jobs)


def test_ac10_discard_from_processing_rejects_take(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: _state(client) == "processing")

        resp = client.post("/record/discard")
        assert resp.status_code == 200
        assert not (tmp_path / "take1").exists()
        assert _state(client) == "idle"
    finally:
        _release_jobs(jobs)


def test_ac10_discard_from_idle_returns_409(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/discard").status_code == 409


# ---------------------------------------------------------------------------
# AC11 — /record/retry: valid only from failed; builds a fresh job from the
# on-disk MP4 + keyframe list, re-runs on a worker thread (-> processing);
# 409 from any other state.
# ---------------------------------------------------------------------------


def test_ac11_retry_from_failed_reruns_postpass(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=9)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/keyframe", json={"frame_number": 2}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        jobs[0].final_state = "failed"
        jobs[0].release.set()
        assert _wait_for(lambda: _state(client) == "failed")

        resp = client.post("/record/retry")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        assert _wait_for(lambda: _state(client) == "processing")
        # A fresh job (second instance) was constructed from the same entry.
        assert _wait_for(lambda: len(jobs) == 2 and jobs[1].started.is_set())
        retry_job = jobs[1]
        assert retry_job.entry_dir == tmp_path / "take1"
        assert retry_job.entry_name == "take1"
        assert set(retry_job.keyframes) == {2}
        assert retry_job.frame_count == 9

        retry_job.release.set()
        assert _wait_for(lambda: _state(client) == "idle")
    finally:
        _release_jobs(jobs)


def test_ac11_retry_from_idle_returns_409(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/retry").status_code == 409


def test_ac11_retry_from_processing_returns_409(tmp_path):
    # Retry is valid only from failed — never on an in-flight post-pass.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=8)
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: _state(client) == "processing")

        assert client.post("/record/retry").status_code == 409
        assert _state(client) == "processing"
    finally:
        _release_jobs(jobs)


def test_ac11_retry_from_recording_returns_409(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_name": "take1"}).raise_for_status()

    assert client.post("/record/retry").status_code == 409


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
        results.append(client.post("/record/start", json={"entry_name": name}))

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
