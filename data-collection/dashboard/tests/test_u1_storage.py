"""U1 AC1..AC6 — storage layout split + entry auto-suffix (blind-TDD SPEC).

Written from ``docs/tasks/U1-storage.md`` (the frozen SPEC). Blind-TDD: this
file exercises the HTTP behaviour contract via ``fastapi.testclient.TestClient``
against the frozen ``create_app`` seam; the coder replaces ``backend/app.py``'s
pre-U1 storage layout without ever seeing this file.

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
from tests.test_recording_api import RecordingStubCapture, _build_app, _drain, _wait_for


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
    app, _detector, _enc, _calls, jobs = _build_app(capture, output_path=tmp_path)
    client = TestClient(app)
    try:
        resp = client.post("/record/start", json={"entry_base": "X"})
        assert resp.status_code == 200
        assert resp.json()["entry_name"] == "X_001"
        assert (tmp_path / "videos" / "X_001").is_dir()

        # Finalize the first take (stop -> enqueue -> drain) WITHOUT discarding,
        # so videos/X_001/ persists on disk and the counter must step past it.
        # (A discard would delete the folder, and the disk-scan counter would
        # then correctly reuse X_001 — so discarding here would falsely reject a
        # spec-correct implementation.) U2: `state` goes "idle" the instant
        # /record/stop returns (AC1) — that alone no longer means the post-pass
        # finished, so wait for drain.current to clear instead before minting
        # the second entry.
        client.post("/record/stop").raise_for_status()
        assert _wait_for(lambda: len(jobs) == 1 and jobs[0].started.is_set())
        jobs[0].release.set()
        assert _wait_for(lambda: _drain(client)["current"] is None)

        resp2 = client.post("/record/start", json={"entry_base": "X"})
        assert resp2.status_code == 200
        assert resp2.json()["entry_name"] == "X_002"
        assert (tmp_path / "videos" / "X_002").is_dir()
    finally:
        for job in jobs:
            job.release.set()


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
# AC5 — image-mode collision-reject (409) fires only for a genuine re-use of
# an existing images/<name>/ Dataset, never merely because images/ (holding a
# different Dataset) already exists.
# ---------------------------------------------------------------------------


def test_ac5_image_collision_only_on_genuine_reuse(tmp_path):
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

    # ...then re-configuring the SAME Dataset X is a genuine reuse -> 409.
    resp2 = client.post(
        "/settings", json={"output_path": str(tmp_path), "dataset_name": "X"}
    )
    assert resp2.status_code == 409


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
