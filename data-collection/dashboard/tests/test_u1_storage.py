"""Storage layout split and entry auto-suffix behavior.

Exercises the HTTP contract through ``fastapi.testclient.TestClient`` and the
public ``create_app`` seam.

Terms (CONTEXT.md): **Output path** (base drop-zone), **Dataset**
(``images/<base>/``, image mode), **Entry** (``videos/<base>_NNN/``, video
mode), **Base name** (the single Settings/``entry_base`` field).

Everything here is camera-free / ffmpeg-free / onnx-free — it reuses the
existing fakes: ``StubCapture`` + ``_writer_factory`` (real ``DatasetWriter``)
from ``tests.test_api`` for image mode, and ``RecordingStubCapture`` +
``_build_app`` (fake encoder / fake post-pass job factory) from
``tests.test_recording_api`` for recording mode.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import create_app
from tests.fakes import FakeDetector, make_fake_frame
from tests.test_api import StubCapture, _snap_latest, _writer_factory
from tests.test_recording_api import RecordingStubCapture, _build_app


# ---------------------------------------------------------------------------
# AC1 — image Dataset nests under output_path/images/<dataset_name>/.
# ---------------------------------------------------------------------------


def test_ac1_settings_nests_dataset_under_images(tmp_path):
    stub = StubCapture(latest=_snap_latest())
    app = create_app(FakeDetector(), _writer_factory(), stub)
    client = TestClient(app)

    resp = client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "X"}
    )
    assert resp.status_code == 200

    flag_resp = client.post("/flag")
    assert flag_resp.status_code == 200

    assert (tmp_path / "images" / "X" / "images" / "frame_00001.jpg").is_file()
    assert (
        tmp_path / "images" / "X" / "annotations" / "annotations.json"
    ).is_file()
    # And NOT at the old (pre-U1) un-nested location.
    assert not (tmp_path / "X" / "images" / "frame_00001.jpg").exists()


# ---------------------------------------------------------------------------
# AC2 — /record/start {entry_base} mints output_path/videos/<base>_NNN/ and
# echoes the resolved name; a repeat base auto-suffixes.
# ---------------------------------------------------------------------------


def test_ac2_record_start_mints_videos_counter_and_returns_resolved_name(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    resp = client.post("/record/start", json={"entry_base": "X"})
    assert resp.status_code == 200
    assert resp.json()["entry_name"] == "X_001"
    assert (tmp_path / "videos" / "X_001").is_dir()

    # Finalize the first take WITHOUT discarding, so videos/X_001/ persists on
    # disk and the counter must step past it. /record/stop now writes the video
    # project synchronously and returns to idle (ADR-0002) — the entry is fully
    # on disk the instant it returns, so no drain to wait for.
    client.post("/record/stop").raise_for_status()

    resp2 = client.post("/record/start", json={"entry_base": "X"})
    assert resp2.status_code == 200
    assert resp2.json()["entry_name"] == "X_002"
    assert (tmp_path / "videos" / "X_002").is_dir()


# ---------------------------------------------------------------------------
# AC3 — counter is max(existing)+1, survives gaps; existing Entries are never
# reused or overwritten.
# ---------------------------------------------------------------------------


def test_ac3_counter_is_max_plus_one_survives_gaps(tmp_path):
    (tmp_path / "videos" / "X_001").mkdir(parents=True)
    (tmp_path / "videos" / "X_003").mkdir(parents=True)
    (tmp_path / "videos" / "X_001" / "marker.txt").write_text("keep-001")
    (tmp_path / "videos" / "X_003" / "marker.txt").write_text("keep-003")

    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    resp = client.post("/record/start", json={"entry_base": "X"})
    assert resp.status_code == 200
    assert resp.json()["entry_name"] == "X_004"  # max(1, 3) + 1, not first-free
    assert (tmp_path / "videos" / "X_004").is_dir()

    # Pre-existing entries are untouched.
    assert (tmp_path / "videos" / "X_001" / "marker.txt").read_text() == "keep-001"
    assert (tmp_path / "videos" / "X_003" / "marker.txt").read_text() == "keep-003"


# ---------------------------------------------------------------------------
# AC4 — base-name validation unchanged, applied to entry_base: bad base -> 422,
# no folder minted; a valid base on the same app still succeeds.
# ---------------------------------------------------------------------------


def test_ac4_invalid_entry_base_422_no_folder(tmp_path):
    capture = RecordingStubCapture(frame=make_fake_frame(64, 48))
    app, *_ = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)

    for bad in ("bad/name", ".hidden", ""):
        resp = client.post("/record/start", json={"entry_base": bad})
        assert resp.status_code == 422
    assert not (tmp_path / "videos").exists()

    # The rejection is specific to the bad names, not a broken endpoint: a
    # valid base on the same app still succeeds and mints a folder.
    good = client.post("/record/start", json={"entry_base": "good"})
    assert good.status_code == 200
    assert (tmp_path / "videos" / "good_001").is_dir()


# ---------------------------------------------------------------------------
# AC5 — reusing an existing images/<name>/ Dataset now APPENDS to it (operator
# request 2026-07-10), while a bare images/ parent (holding a different Dataset)
# still does not collide. A non-dataset folder still errors — see test_api.
# ---------------------------------------------------------------------------


def test_ac5_image_reuse_appends_to_existing_dataset(tmp_path):
    stub = StubCapture(latest=_snap_latest())
    app = create_app(FakeDetector(), _writer_factory(), stub)
    client = TestClient(app)

    # Dataset Y is configured and flagged once -> creates
    # output_path/images/Y/... and, incidentally, the images/ parent.
    client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "Y"}
    )
    client.post("/flag")
    assert (tmp_path / "images" / "Y").is_dir()

    # A bare images/ parent holding only Y must NOT collide with a fresh
    # Dataset X.
    resp = client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "X"}
    )
    assert resp.status_code == 200

    # Flag once under X so images/X/ genuinely exists on disk...
    flag_resp = client.post("/flag")
    assert flag_resp.status_code == 200
    assert (tmp_path / "images" / "X").is_dir()

    # ...then re-configuring the SAME Dataset X now RESUMES it (append), 200 with
    # appended=True and the existing frame count — not a 409.
    resp2 = client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "X"}
    )
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["appended"] is True
    assert body["existing_images"] == 1  # the frame flagged under X above

    # A further flag continues the sequence (frame_00002) and keeps frame_00001.
    flag2 = client.post("/flag")
    assert flag2.status_code == 200
    assert (tmp_path / "images" / "X" / "images" / "frame_00001.jpg").is_file()
    assert (tmp_path / "images" / "X" / "images" / "frame_00002.jpg").is_file()


# ---------------------------------------------------------------------------
# AC6 — lightweight guard: /status still reports recording_state, and image
# flag still works, under the new nesting. The real AC6 gate is the whole
# `uv run pytest` staying green (the migrated legacy files).
# ---------------------------------------------------------------------------


def test_ac6_prior_suites_unaffected(tmp_path):
    stub = StubCapture(latest=_snap_latest())
    app = create_app(FakeDetector(), _writer_factory(), stub)
    client = TestClient(app)

    assert client.get("/status").json()["recording_state"] == "idle"

    client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "ds"}
    )
    resp = client.post("/flag")
    assert resp.status_code == 200
    assert resp.json()["n_flagged"] == 1
    assert (tmp_path / "images" / "ds" / "images" / "frame_00001.jpg").is_file()

    # /status reports the operator's BASE Output path, not the images/ nesting
    # the writer now lives under (U1 must not change what /status shows).
    assert client.get("/status").json()["output_path"] == str(tmp_path)
