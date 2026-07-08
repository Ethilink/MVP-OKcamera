"""T05 FastAPI layer — AC1..AC9 (spec §Runtime, §Threading rules, §Guardrails).

Everything here is camera-free and onnx-free: the real ``CaptureLoop`` is driven
by T01's ``FakeCapture`` / ``FakeDetector`` where a live loop matters (health),
and a deterministic ``StubCapture`` stands in where a background thread would make
the assertion racy (stream pacing, snapshot provenance).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app, mjpeg_stream
from backend.capture import CaptureLoop, Latest
from backend.dataset_writer import DatasetWriter
from backend.render import render
from tests.fakes import FakeCapture, FakeDetector, make_fake_dets, make_fake_frame


# --- test doubles + helpers -------------------------------------------------


class StubCapture:
    """Deterministic ``CaptureLoop`` stand-in exposing exactly the surface the
    endpoints touch: ``snapshot()``, ``generation``, ``health``, ``set_camera()``.

    No background thread — the test controls ``generation`` / ``latest`` directly,
    so stream-pacing and provenance assertions are race-free.
    """

    def __init__(self, latest=None, health="ok", generation=0):
        self._latest = latest
        self._health = health
        self._generation = generation
        self.set_camera_calls: list[int] = []

    def snapshot(self):
        return self._latest

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def health(self) -> str:
        return self._health

    def set_camera(self, index: int) -> None:
        self.set_camera_calls.append(index)

    def publish(self, latest) -> None:
        self._latest = latest
        self._generation += 1


def _latest(jpeg=b"jpeg", frame=None, dets=None, threshold=0.5, count=0) -> Latest:
    return Latest(
        frame=frame,
        dets=dets,
        threshold=threshold,
        overlay_jpeg=jpeg,
        count=count,
    )


def _writer_factory(model_version="rfdetr-test"):
    def factory(output_path, dataset_name):
        return DatasetWriter(output_path, dataset_name, model_version)

    return factory


def _dets():
    return make_fake_dets([[100, 100, 300, 300], [400, 400, 600, 600]], size=(1920, 1080))


def _snap_latest(threshold=0.3, count=2):
    return _latest(
        frame=make_fake_frame(1920, 1080),
        dets=_dets(),
        threshold=threshold,
        count=count,
    )


def _wait_for(predicate, timeout=3.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# --- AC1: /flag and /validate are plain def ---------------------------------


def test_ac1_flag_and_validate_are_plain_def():
    app = create_app(FakeDetector(), _writer_factory(), StubCapture())
    endpoints = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}

    assert not inspect.iscoroutinefunction(endpoints["/flag"])
    assert not inspect.iscoroutinefunction(endpoints["/validate"])
    # /stream, by contrast, IS async (it owns the event-loop paced generator).
    assert inspect.iscoroutinefunction(endpoints["/stream"])


# --- AC2: concurrent flags serialize on dataset_lock ------------------------


def test_ac2_concurrent_flags_yield_sequential_ids(tmp_path):
    stub = StubCapture(latest=_snap_latest())
    app = create_app(FakeDetector(), _writer_factory(), stub)
    client = TestClient(app)
    client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "ds"}
    ).raise_for_status()

    n = 8
    responses: list = []
    barrier = threading.Barrier(n)

    def worker():
        barrier.wait()  # release all flags at once to maximize contention
        responses.append(client.post("/flag"))

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.status_code == 200 for r in responses)
    image_ids = sorted(r.json()["image_id"] for r in responses)
    assert image_ids == list(range(1, n + 1))  # N sequential, no duplicates

    data = json.loads((tmp_path / "ds" / "annotations" / "annotations.json").read_text())
    assert len(data["images"]) == n
    ann_ids = [a["id"] for a in data["annotations"]]
    assert ann_ids == sorted(ann_ids)
    assert len(ann_ids) == len(set(ann_ids))  # no torn/duplicated ids
    assert data["annotations"]  # dets were non-empty, so annotations were written


# --- AC3: /flag with no target -> 409 (never a silent no-op) -----------------


def test_ac3_flag_without_target_returns_409():
    stub = StubCapture(latest=_snap_latest())
    app = create_app(FakeDetector(), _writer_factory(), stub)
    client = TestClient(app)

    resp = client.post("/flag")
    assert resp.status_code == 409
    assert resp.json()["detail"]  # human-readable, not empty


# --- AC4: /settings collision / invalid / success ---------------------------


def test_ac4_settings_collision_invalid_and_swap(tmp_path):
    stub = StubCapture(latest=_snap_latest())
    app = create_app(FakeDetector(), _writer_factory(), stub)
    client = TestClient(app)

    # success + camera swap
    resp = client.post(
        "/settings",
        json={"output_path": str(tmp_path), "dataset_name": "ds1", "camera_index": 3},
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert stub.set_camera_calls == [3]

    # flag once so counters are non-zero, then prove the swap resets them
    client.post("/flag").raise_for_status()
    assert client.get("/status").json()["n_flagged"] == 1

    client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "ds2"}
    ).raise_for_status()
    assert client.get("/status").json()["n_flagged"] == 0  # fresh writer, reset

    # collision: an existing folder is rejected with 409
    (tmp_path / "taken").mkdir()
    resp = client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "taken"}
    )
    assert resp.status_code == 409

    # invalid dataset name -> 422
    resp = client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "bad/name"}
    )
    assert resp.status_code == 422


# --- AC5: /confidence live + Pydantic range -------------------------------


def test_ac5_confidence_live_and_range():
    detector = FakeDetector(confidence_threshold=0.5)
    app = create_app(detector, _writer_factory(), StubCapture())
    client = TestClient(app)

    resp = client.post("/confidence", json={"value": 0.73})
    assert resp.status_code == 200
    assert resp.json() == {"confidence": 0.73}
    assert detector.confidence_threshold == 0.73  # set live on the detector

    assert client.post("/confidence", json={"value": 1.5}).status_code == 422
    assert client.post("/confidence", json={"value": -0.1}).status_code == 422


# --- AC6: /stream sends each overlay exactly once ---------------------------


def test_ac6_stream_sends_each_overlay_once():
    stub = StubCapture()
    stub.publish(_latest(jpeg=b"JPEG-1"))  # generation -> 1

    async def scenario():
        parts: list[bytes] = []
        agen = mjpeg_stream(stub)

        async def pump():
            async for part in agen:
                parts.append(part)

        task = asyncio.create_task(pump())
        try:
            # First (and only) generation so far -> exactly one part, promptly.
            await asyncio.sleep(0.15)
            assert len(parts) == 1
            assert b"JPEG-1" in parts[0]
            assert b"Content-Type: image/jpeg" in parts[0]

            # Paused capture (generation unchanged) -> generator must NOT repeat.
            await asyncio.sleep(0.2)
            assert len(parts) == 1

            # A freshly published Latest is streamed promptly, exactly once.
            stub.publish(_latest(jpeg=b"JPEG-2"))  # generation -> 2
            await asyncio.sleep(0.15)
            assert len(parts) == 2
            assert b"JPEG-2" in parts[1]
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())


# --- AC7: /status fields + capture_health -----------------------------------


def test_ac7_status_fields_and_health_ok(tmp_path):
    detector = FakeDetector(
        predictions=[make_fake_dets([[100, 100, 300, 300]])], confidence_threshold=0.6
    )
    frame = make_fake_frame(1920, 1080)
    capture = CaptureLoop(
        detector,
        camera_index=0,
        render_fn=render,
        cap_factory=lambda idx: FakeCapture(frames=[frame], on_empty="repeat"),
        stale_after_s=5.0,
    )
    capture.start()
    app = create_app(detector, _writer_factory(), capture)
    client = TestClient(app)
    try:
        assert _wait_for(lambda: capture.health == "ok")
        body = client.get("/status").json()
        assert set(body) == {
            "count",
            "confidence",
            "dataset_name",
            "output_path",
            "n_flagged",
            "capture_health",
        }
        assert body["confidence"] == 0.6
        assert body["dataset_name"] is None
        assert body["output_path"] is None
        assert body["n_flagged"] == 0
        assert isinstance(body["count"], int)
        assert body["capture_health"] == "ok"
    finally:
        capture.stop()


def test_ac7_status_reports_stale(tmp_path):
    detector = FakeDetector(predictions=[make_fake_dets([[10, 10, 50, 50]])])
    frame = make_fake_frame(1920, 1080)
    # One good frame satisfies start()'s no-valid-frame guard; then reads fail,
    # so the loop never publishes a heartbeat -> health goes stale.
    capture = CaptureLoop(
        detector,
        camera_index=0,
        render_fn=render,
        cap_factory=lambda idx: FakeCapture(frames=[frame], on_empty="fail"),
        stale_after_s=0.05,
    )
    capture.start()
    app = create_app(detector, _writer_factory(), capture)
    client = TestClient(app)
    try:
        assert _wait_for(lambda: client.get("/status").json()["capture_health"] == "stale")
    finally:
        capture.stop()


# --- AC8: /validate calls the in-process validator --------------------------


def test_ac8_validate_in_process(tmp_path):
    stub = StubCapture(latest=_snap_latest())
    app = create_app(FakeDetector(), _writer_factory(), stub)
    client = TestClient(app)

    # No dataset configured -> 409 (never a silent empty result).
    assert client.post("/validate").status_code == 409

    client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "ds"}
    ).raise_for_status()
    client.post("/flag").raise_for_status()

    resp = client.post("/validate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["errors"] == []  # the writer emits an import-ready dataset
    assert isinstance(body["warnings"], list)  # advisory summary lives here


def test_ac8_validate_is_in_process_no_subprocess(tmp_path, monkeypatch):
    """The endpoint calls validate_fn directly — assert it's the injected callable."""
    calls: list = []

    def fake_validate(dataset_dir):
        calls.append(dataset_dir)
        return (["synthetic error"], ["synthetic warning"])

    stub = StubCapture(latest=_snap_latest())
    app = create_app(FakeDetector(), _writer_factory(), stub, validate_fn=fake_validate)
    client = TestClient(app)
    client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "ds"}
    ).raise_for_status()

    body = client.post("/validate").json()
    assert body == {"errors": ["synthetic error"], "warnings": ["synthetic warning"]}
    assert len(calls) == 1  # called once, in-process, with the active dataset dir
    assert calls[0] == (tmp_path / "ds")


# --- AC9: provenance uses the snapshot threshold, not the live slider --------


def test_ac9_flag_records_snapshot_threshold(tmp_path):
    detector = FakeDetector(confidence_threshold=0.3)
    # The snapshot froze threshold=0.3 at (fake) predict time.
    stub = StubCapture(latest=_snap_latest(threshold=0.3))
    app = create_app(detector, _writer_factory("rfdetr-test"), stub)
    client = TestClient(app)
    client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "ds"}
    ).raise_for_status()

    # Move the slider AFTER the snapshot was taken, BEFORE the flag.
    client.post("/confidence", json={"value": 0.8}).raise_for_status()
    assert detector.confidence_threshold == 0.8

    client.post("/flag").raise_for_status()

    data = json.loads((tmp_path / "ds" / "annotations" / "annotations.json").read_text())
    assert data["annotations"], "expected annotations from non-empty dets"
    for ann in data["annotations"]:
        # The frozen snapshot value survives — not the live 0.8 slider.
        assert ann["confidence_threshold"] == 0.3
        assert ann["model_version"] == "rfdetr-test"


# --- static assets: index.html's CSS + JS must actually be served -----------


def test_static_assets_are_served():
    """Regression: index.html pulls in /static/style.css + /static/app.js. Without
    the StaticFiles mount both 404 and the page renders as raw unstyled HTML with a
    dead script (no polling, settings form falls back to a native GET). Guard it."""
    app = create_app(FakeDetector(), _writer_factory(), StubCapture())
    client = TestClient(app)

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert "addEventListener" in js.text  # real script, not an error page

    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert "css" in css.headers["content-type"]

    # And the index still references them (mount + markup stay in sync).
    index = client.get("/").text
    assert "/static/style.css" in index
    assert "/static/app.js" in index
