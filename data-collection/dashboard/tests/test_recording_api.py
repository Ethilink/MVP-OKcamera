"""Recording-mode HTTP contract (keyframe-only, synchronous stop — ADR-0002).

The 2026-07-09 simplification dropped the all-frames post-pass and the
idle-draining drain queue: ``/record/stop`` now writes the reviewed
video-project artifacts synchronously (from keyframes captured live at
SPACE-press time) and returns to idle. So there is no ``/record/retry``, no
``drain`` block, no pause/resume, no ``eta_seconds``. Both recording endpoints
carry an ``error`` field surfacing a mid-take encoder failure (AC8): ``null``
on a clean take, the ``recording_error`` message on a truncated one.

The endpoint contract exercised here via ``fastapi.testclient.TestClient``
against the frozen ``create_app`` seam:

    POST /record/start  {entry_base}  -> {ok, entry_name}              (mint, encode, record)
    POST /keyframe       {generation} -> {generation, n_keyframes}     (mark on-screen frame)
    POST /record/stop                 -> {ok, frames_written, error}   (write artifacts sync)
    POST /record/discard              -> {ok}                          (recording -> abort+delete)
    GET  /record/status               -> {state, error}

Everything is camera-free / ffmpeg-free / onnx-free:

- ``app.state.open_encoder`` is overridden with ``_StubEncoder`` (writes a 1-byte
  stub MP4 on release so ``video/`` is non-empty for the discovery rule) — the
  call args (path / fps / frame_size) are inspectable.
- ``app.state.probe_video`` returns a ``VideoProbe`` with the stub capture's
  frame dims, so the real ``VideoEntryWriter`` (default, not faked) writes real
  ``annotations.json`` / ``selected_frames.json`` / JPEGs we can assert on disk.
- ``RecordingStubCapture`` is a deterministic capture stand-in exposing exactly
  the recording surface the endpoints touch (``snapshot`` / ``snapshot_at`` /
  ``start_recording`` / ``stop_recording`` / ``frames_written``), with no
  background thread so state assertions are race-free. The real-CaptureLoop
  keyframe test uses a live loop + numbered fake camera, because a live ring is
  what makes the generation->frame_number resolution real.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.capture import CaptureLoop, Latest
from backend.dataset_writer import DatasetWriter
from backend.encoder import VideoProbe
from tests.fakes import FakeCapture, FakeDetector, make_fake_dets, make_fake_frame
from tests.recording_fakes import make_numbered_frames


# --- helpers ----------------------------------------------------------------


def _wait_for(predicate, timeout=3.0, interval=0.02) -> bool:
    """Bounded poll: True as soon as ``predicate()`` is truthy, else False after
    ``timeout``. Never a bare/unbounded wait — a stuck machine fails fast."""
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


class _StubEncoder:
    """Encoder duck-type (``write`` / ``release`` / ``is_open``). Records written
    frames and, on release, drops a 1-byte stub file at the MP4 path so the
    finished entry's ``video/`` is non-empty (video-project discovery rule)."""

    def __init__(self, path):
        self.path = Path(path)
        self.written: list = []
        self.is_open = True

    def write(self, frame) -> None:
        self.written.append(frame)

    def release(self) -> None:
        if not self.is_open:
            return
        self.is_open = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"\x00")


class RecordingStubCapture:
    """Deterministic ``CaptureLoop`` stand-in exposing exactly the surface the
    recording endpoints touch — no background thread, so state assertions are
    race-free. ``snapshot_at(generation)`` returns a recording-stamped ``Latest``
    whose ``frame_number == generation`` (so a repeat press on the same
    generation dedups, and distinct presses count distinctly), unless the
    generation has been marked aged-out. ``stop_recording`` releases the encoder
    (mirroring the real loop) and returns the frame count."""

    def __init__(self, frame=None, frames_written=42, generation=0, dets=None):
        self._frame = frame
        self._frames_written = frames_written
        self._generation = generation
        self._dets = (
            dets
            if dets is not None
            else make_fake_dets([[10, 10, 40, 30]], confidences=[0.9], size=(64, 48))
        )
        self._recording = False
        self.encoder = None
        self.start_recording_calls: list = []
        self.stop_recording_calls = 0
        self.aged_out: set[int] = set()  # generations snapshot_at reports as evicted
        # Mirror CaptureLoop.recording_error: a mid-take encoder.write failure,
        # reset at start_recording, left intact by stop_recording. Tests set it
        # directly to drive the truncated-recording path.
        self.recording_error: BaseException | None = None

    # --- snapshot surface ---
    def snapshot(self):
        if self._frame is None:
            return None
        return Latest(self._frame, self._dets, 0.5, b"jpeg", len(self._dets), None)

    def snapshot_with_generation(self):
        return self._generation, self.snapshot()

    def snapshot_at(self, generation):
        if self._frame is None or generation in self.aged_out:
            return None
        return Latest(self._frame, self._dets, 0.5, b"jpeg", len(self._dets), generation)

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
        self.recording_error = None  # fresh take clears any prior error (real loop parity)
        self.start_recording_calls.append(encoder)

    def stop_recording(self) -> int:
        # Mirror the real loop: capture the count, then reset frames_written to 0.
        self._recording = False
        self.stop_recording_calls += 1
        if self.encoder is not None:
            self.encoder.release()
        count = self._frames_written
        self._frames_written = 0
        return count

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
):
    """Assemble the app around ``capture`` with a fake encoder + fake probe (real
    ``VideoEntryWriter``). Returns (app, detector, encoder_holder, encoder_calls);
    ``encoder_holder['enc']`` is the last encoder opened."""
    detector = detector or FakeDetector(confidence_threshold=0.5)
    app = create_app(detector, _writer_factory(), capture)
    app.state.output_path = str(output_path) if output_path is not None else None
    app.state.capture_fps = capture_fps

    holder: dict = {}
    encoder_calls: list = []

    def open_encoder(path, fps, frame_size, **kwargs):
        encoder_calls.append({"path": Path(path), "fps": fps, "frame_size": tuple(frame_size)})
        enc = _StubEncoder(path)
        holder["enc"] = enc
        return enc

    app.state.open_encoder = open_encoder

    def probe(path, **kw):
        snap = capture.snapshot()
        if snap is not None and snap.frame is not None:
            h, w = snap.frame.shape[:2]
        else:
            w, h = 0, 0
        return VideoProbe(fps=capture_fps, frame_count=0, width=w, height=h)

    app.state.probe_video = probe
    return app, detector, holder, encoder_calls


def _status(client) -> dict:
    return client.get("/record/status").json()


def _state(client) -> str:
    return _status(client)["state"]


# ---------------------------------------------------------------------------
# The blocking recording endpoints are plain ``def`` (folder mint, encoder open,
# reader-thread join, synchronous keyframe write all stay off the event loop).
# ---------------------------------------------------------------------------


def test_recording_endpoints_are_plain_def():
    app = create_app(FakeDetector(), _writer_factory(), RecordingStubCapture())
    endpoints = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}

    for path in ("/keyframe", "/record/start", "/record/stop", "/record/discard", "/flag"):
        assert path in endpoints, f"missing route {path}"
        assert not inspect.iscoroutinefunction(
            endpoints[path]
        ), f"{path} must be plain def (blocking I/O off the event loop)"

    # /record/retry is gone (ADR-0002 — no failed-job retry path).
    assert "/record/retry" not in endpoints


# ---------------------------------------------------------------------------
# /record/start: mint the entry, open the encoder at capture_fps with frame_size
# from the snapshot dims, call start_recording, transition to recording.
# ---------------------------------------------------------------------------


def test_start_records_and_opens_encoder_at_capture_fps_and_snapshot_size(tmp_path):
    frame = make_fake_frame(1920, 1080)  # (H, W, 3) -> frame_size (1920, 1080)
    capture = RecordingStubCapture(frame=frame)
    app, _detector, holder, encoder_calls = _build_app(
        capture, output_path=tmp_path, capture_fps=24.0
    )
    client = TestClient(app)

    resp = client.post("/record/start", json={"entry_base": "take1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["entry_name"] == "take1_001"  # resolved base_NNN (U1)

    assert len(encoder_calls) == 1
    call = encoder_calls[0]
    assert call["fps"] == 24.0
    assert call["frame_size"] == (1920, 1080)
    expected_mp4 = tmp_path / "videos" / "take1_001" / "video" / "take1_001.mp4"
    assert call["path"] == expected_mp4

    assert (tmp_path / "videos" / "take1_001" / "video").is_dir()
    assert capture.start_recording_calls == [holder["enc"]]

    assert _state(client) == "recording"
    assert client.get("/status").json()["recording_state"] == "recording"


def test_start_same_base_auto_suffixes_no_folder_collision(tmp_path):
    (tmp_path / "videos" / "taken_001").mkdir(parents=True)
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    resp = client.post("/record/start", json={"entry_base": "taken"})
    assert resp.status_code == 200
    assert resp.json()["entry_name"] == "taken_002"


def test_start_409_when_already_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_base": "a"}).status_code == 200
    assert client.post("/record/start", json={"entry_base": "b"}).status_code == 409


def test_start_409_when_no_output_path(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=None)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_base": "a"}).status_code == 409


def test_start_409_when_no_frame_captured_yet(tmp_path):
    # No snapshot yet -> frame_size unknowable -> 409, never an AttributeError.
    capture = RecordingStubCapture(frame=None)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_base": "a"}).status_code == 409


def test_start_422_on_invalid_entry_base(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/start", json={"entry_base": "bad/name"}).status_code == 422
    assert client.post("/record/start", json={"entry_base": ".hidden"}).status_code == 422
    assert client.post("/record/start", json={"entry_base": ""}).status_code == 422


# ---------------------------------------------------------------------------
# /keyframe {generation}: dedups a repeat generation, echoes generation + count;
# 409 when not recording; 409 (non-fatal) when the frame aged out of the ring.
# ---------------------------------------------------------------------------


def test_keyframe_dedups_and_counts_while_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=100)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_base": "a"}).raise_for_status()

    r1 = client.post("/keyframe", json={"generation": 5})
    assert r1.json() == {"generation": 5, "n_keyframes": 1}
    # A repeat of the same generation (same frame_number) does NOT grow the set.
    assert client.post("/keyframe", json={"generation": 5}).json()["n_keyframes"] == 1
    assert client.post("/keyframe", json={"generation": 6}).json()["n_keyframes"] == 2


def test_keyframe_409_when_not_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/keyframe", json={"generation": 3}).status_code == 409


def test_keyframe_409_when_frame_aged_out(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    capture.aged_out.add(7)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_base": "a"}).raise_for_status()

    resp = client.post("/keyframe", json={"generation": 7})
    assert resp.status_code == 409
    assert "aged out" in resp.json()["detail"]


def test_keyframe_resolves_generation_over_a_real_capture_loop(tmp_path):
    # The meaningful identity: /frame's generation, echoed to /keyframe, resolves
    # via the ring to a real recording-stamped frame. Real CaptureLoop + numbered
    # camera (a live ring is what makes this true).
    frames = make_numbered_frames(4000, 64, 48)
    capture = CaptureLoop(
        FakeDetector(),
        camera_index=0,
        render_fn=lambda frame, dets: frame,
        cap_factory=lambda idx: FakeCapture(frames=frames, on_empty="repeat"),
        stale_after_s=5.0,
        ring_size=5000,
    )
    capture.start()
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        assert _wait_for(lambda: capture.generation > 0)
        client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
        # Wait for a recording-stamped snapshot (frame_number not None).
        assert _wait_for(
            lambda: capture.snapshot() is not None
            and capture.snapshot().frame_number is not None
        )
        r = client.get("/frame")
        gen = int(r.headers["x-frame-generation"])
        resp = client.post("/keyframe", json={"generation": gen})
        assert resp.status_code == 200
        assert resp.json()["generation"] == gen
        assert resp.json()["n_keyframes"] == 1
    finally:
        capture.stop()


# ---------------------------------------------------------------------------
# /record/stop: returns {ok, frames_written, error} immediately (state -> idle),
# and synchronously writes the reviewed video project (annotations.json,
# selected_frames.json, one JPEG per keyframe, non-empty video/). `error` is
# non-null (and ok=False) when a mid-take encoder failure truncated the take.
# ---------------------------------------------------------------------------


def test_stop_writes_video_project_synchronously(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=50)
    app, _detector, _holder, _calls = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    start = client.post("/record/start", json={"entry_base": "take1"})
    start.raise_for_status()
    resolved = start.json()["entry_name"]
    client.post("/keyframe", json={"generation": 5}).raise_for_status()
    client.post("/keyframe", json={"generation": 6}).raise_for_status()

    resp = client.post("/record/stop")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "frames_written": 50, "error": None}
    assert _state(client) == "idle"

    entry = tmp_path / "videos" / resolved
    # Discovery rule: annotations.json + non-empty video/.
    assert (entry / "annotations" / "annotations.json").is_file()
    video_files = list((entry / "video").iterdir())
    assert video_files and all(p.stat().st_size > 0 for p in video_files)
    # selected_frames.json lists exactly the marked frames; one JPEG + image
    # record each.
    selected = (entry / "annotations" / "metadata" / "selected_frames.json")
    assert selected.is_file()
    sel = json.loads(selected.read_text())
    assert sel["selected_frames"] == [5, 6]
    anns = json.loads((entry / "annotations" / "annotations.json").read_text())
    assert sorted(im["frame_number"] for im in anns["images"]) == [5, 6]
    jpgs = sorted(p.name for p in (entry / "images").glob("*.jpg"))
    assert jpgs == [f"{resolved}_f{fn:06d}.jpg" for fn in (5, 6)]


def test_stop_with_no_keyframes_writes_empty_project(tmp_path):
    # A record -> stop with no SPACE press is a real operator action: the entry
    # is still a valid (empty) video project satisfying the discovery rule.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=30)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    start = client.post("/record/start", json={"entry_base": "take1"})
    start.raise_for_status()
    resolved = start.json()["entry_name"]

    resp = client.post("/record/stop")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "frames_written": 30, "error": None}

    entry = tmp_path / "videos" / resolved
    anns = json.loads((entry / "annotations" / "annotations.json").read_text())
    assert anns["images"] == []
    assert anns["annotations"] == []
    sel = json.loads(
        (entry / "annotations" / "metadata" / "selected_frames.json").read_text()
    )
    assert sel["selected_frames"] == []


def test_stop_partial_write_leaves_valid_partial_entry(tmp_path):
    # INV-5: /record/stop writes selected_frames.json FIRST, so a failure during
    # the keyframe/annotation write leaves exactly the contract's valid partial
    # (video + selected_frames, no annotations -> "needs detection"). Guards the
    # WRITE ORDER at the app level (the writer's own partial test proves only the
    # mechanism; reordering record_stop would silently violate this).
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=20)
    app, _detector, _holder, _calls = _build_app(capture, output_path=tmp_path)

    real_factory = app.state.video_writer_factory

    def exploding_factory(*args, **kwargs):
        writer = real_factory(*args, **kwargs)

        def boom(*a, **k):
            raise RuntimeError("simulated keyframe write failure")

        writer.add_keyframe = boom
        return writer

    app.state.video_writer_factory = exploding_factory

    # raise_server_exceptions=False so the 500 comes back as a response, not a
    # re-raised exception, while the partial entry stays on disk.
    client = TestClient(app, raise_server_exceptions=False)
    start = client.post("/record/start", json={"entry_base": "take1"})
    start.raise_for_status()
    resolved = start.json()["entry_name"]
    client.post("/keyframe", json={"generation": 5}).raise_for_status()

    resp = client.post("/record/stop")
    assert resp.status_code == 500

    entry = tmp_path / "videos" / resolved
    assert (entry / "annotations" / "metadata" / "selected_frames.json").is_file()
    assert not (entry / "annotations" / "annotations.json").exists()
    assert list((entry / "video").iterdir())  # MP4 finalized before the write
    # State was reset before the (failed) write, so the tool is idle + reusable.
    assert _state(client) == "idle"


def test_stop_surfaces_encoder_error_and_still_writes_partial(tmp_path):
    # AC8: an encoder.write failure mid-take freezes the frame count and sets
    # capture.recording_error. /record/stop must STILL finalize the MP4 + the
    # keyframes marked before the failure (INV-5 — no captured data lost), but
    # report the truncation (ok=False + the message) instead of a clean success,
    # so the operator learns the saved clip is incomplete and re-records.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48), frames_written=12)
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    start = client.post("/record/start", json={"entry_base": "take1"})
    start.raise_for_status()
    resolved = start.json()["entry_name"]
    client.post("/keyframe", json={"generation": 3}).raise_for_status()
    capture.recording_error = RuntimeError("encoder write failed: disk full")

    resp = client.post("/record/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["frames_written"] == 12
    assert "disk full" in body["error"]
    assert _state(client) == "idle"

    # Data captured before the failure is preserved, not discarded.
    entry = tmp_path / "videos" / resolved
    assert (entry / "annotations" / "annotations.json").is_file()
    assert list((entry / "video").iterdir())
    sel = json.loads((entry / "annotations" / "metadata" / "selected_frames.json").read_text())
    assert sel["selected_frames"] == [3]


def test_stop_409_when_not_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/stop").status_code == 409


# ---------------------------------------------------------------------------
# /record/status: {state, error}; idle | recording.
# ---------------------------------------------------------------------------


def test_record_status_exact_shape(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    idle_body = _status(client)
    assert set(idle_body.keys()) == {"state", "error"}
    assert idle_body["state"] == "idle"
    assert idle_body["error"] is None

    client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
    assert _status(client)["state"] == "recording"


def test_status_surfaces_encoder_error_only_while_recording(tmp_path):
    # A mid-take encoder failure is visible on /record/status while recording, so
    # the UI can prompt Discard/restart without waiting for Stop. Idle never
    # surfaces it — a finished (or not-yet-started) take's error can't linger.
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    # Idle: even a lingering capture error stays hidden.
    capture.recording_error = RuntimeError("stale from a past take")
    assert _status(client) == {"state": "idle", "error": None}

    # Recording: start clears the stale error (real-loop parity), a fresh
    # encoder failure IS surfaced live.
    client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()
    assert _status(client)["error"] is None
    capture.recording_error = RuntimeError("encoder write failed")
    body = _status(client)
    assert body["state"] == "recording"
    assert body["error"] == "encoder write failed"


# ---------------------------------------------------------------------------
# /flag during recording -> 409; idle flag/status regression.
# ---------------------------------------------------------------------------


def test_flag_blocked_while_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    client.post("/record/start", json={"entry_base": "take1"}).raise_for_status()

    resp = client.post("/flag")
    assert resp.status_code == 409
    assert "stop recording to snapshot stills" in resp.json()["detail"]


def test_status_idle_and_flag_works_when_idle(tmp_path):
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
# /record/discard: from recording -> abort (stop+release encoder, delete folder),
# back to idle; 409 from a fully idle machine.
# ---------------------------------------------------------------------------


def test_discard_from_recording_aborts_and_deletes(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, _detector, holder, _calls = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    start = client.post("/record/start", json={"entry_base": "take1"})
    start.raise_for_status()
    resolved = start.json()["entry_name"]
    assert (tmp_path / "videos" / resolved).exists()

    resp = client.post("/record/discard")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert capture.stop_recording_calls == 1
    assert holder["enc"].is_open is False  # encoder released
    assert not (tmp_path / "videos" / resolved).exists()  # folder removed
    assert _state(client) == "idle"


def test_discard_from_idle_returns_409(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    assert client.post("/record/discard").status_code == 409


# ---------------------------------------------------------------------------
# One recording at a time: two concurrent /record/start requests yield exactly
# one 200 (recording) and one 409 (serialized on recording_lock).
# ---------------------------------------------------------------------------


def test_concurrent_start_yields_exactly_one_recording(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
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
    assert len(capture.start_recording_calls) == 1
    assert _state(client) == "recording"
