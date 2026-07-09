"""Unified-capture frontend — asserted against the SERVED markup/JS.

These are *structure* tests, not live-behavior tests. They assert the settled
DESIGN is present in what the app serves — the mode toggle, the single
relabeling FLAG button, the Record toggle — and, post-ADR-0002, that the drain
queue is GONE: no queue chip / retry / processing veil, and ``pollRecordStatus``
reads only ``{state}``. ``/keyframe`` sends ``{generation}``.

Renders "/" and reads the static assets through ``TestClient(create_app(...))``
with the existing fakes.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.dataset_writer import DatasetWriter
from tests.fakes import FakeDetector


class _StubCapture:
    """Minimal capture surface — enough for create_app + serving "/"."""

    def __init__(self, health="ok", generation=0, camera_index=0):
        self._health = health
        self._generation = generation
        self.camera_index = camera_index

    def snapshot(self):
        return None

    def snapshot_with_generation(self):
        return self._generation, None

    def snapshot_at(self, generation):
        return None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def health(self) -> str:
        return self._health

    def set_camera(self, index: int) -> None:  # pragma: no cover - unused here
        self.camera_index = index


def _writer_factory(model_version="rfdetr-test"):
    def factory(output_path, dataset_name):
        return DatasetWriter(output_path, dataset_name, model_version)

    return factory


@pytest.fixture
def client():
    app = create_app(FakeDetector(), _writer_factory(), _StubCapture())
    return TestClient(app)


@pytest.fixture
def index_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


@pytest.fixture
def app_js(client):
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    return resp.text


@pytest.fixture
def style_css(client):
    resp = client.get("/static/style.css")
    assert resp.status_code == 200
    return resp.text


# --- AC1: segmented mode toggle at the top of the stage ----------------------


def test_ac1_mode_toggle_present(index_html):
    assert 'id="mode-toggle"' in index_html
    assert 'id="mode-image"' in index_html
    assert 'id="mode-video"' in index_html
    # Segmented labels are the 📷 Image / 🎬 Video pair from the spec.
    assert "📷" in index_html and "Image" in index_html
    assert "🎬" in index_html and "Video" in index_html
    # data-mode drives the JS switch.
    assert 'data-mode="image"' in index_html
    assert 'data-mode="video"' in index_html


def test_ac1_toggle_switches_mode_and_space_behavior(app_js):
    # setMode flips the client-side mode; flagOrKeyframe dispatches on it.
    assert "function setMode" in app_js
    assert re.search(r'setMode\(\s*["\']image["\']\s*\)', app_js)
    assert re.search(r'setMode\(\s*["\']video["\']\s*\)', app_js)


# --- AC2: SPACE marks always; single relabeling FLAG button ------------------


def test_ac2_flag_button_relabels_in_place(index_html, app_js):
    # One button, whose label is relabeled FLAG <-> KEYFRAME (kept in position).
    assert 'id="flag"' in index_html
    assert 'id="flag-label"' in index_html
    assert "KEYFRAME" in app_js and "FLAG" in app_js


def test_ac2_space_dispatch_keyed_on_mode_with_idle_hint(app_js):
    # Dispatch is on MODE now (not the old recState branch): image -> flag,
    # video+recording -> keyframe, video+idle -> hint.
    assert "function flagOrKeyframe" in app_js
    assert 'mode === "video"' in app_js
    assert "markKeyframe()" in app_js
    # video+idle is a no-op with a hint, not a rejection.
    assert "Press Record first" in app_js


# --- AC3: Record button toggles; Discard while recording ---------------------


def test_ac3_record_toggle_video_only(index_html, app_js):
    assert 'id="record-controls"' in index_html
    assert 'id="record"' in index_html
    assert 'id="record-label"' in index_html
    # Record controls exist in video mode only (hidden by default = image mode).
    assert re.search(r'id="record-controls"[^>]*\shidden', index_html)
    assert "els.recordControls.hidden = !video" in app_js
    # Label toggles Record <-> Stop.
    assert re.search(r'recording\s*\?\s*"Stop"\s*:\s*"Record"', app_js)


def test_ac3_discard_appears_while_recording(index_html, app_js):
    assert 'id="record-discard"' in index_html
    assert "els.recordDiscard.hidden = !recording" in app_js


# --- AC4: settings name field relabels per mode ------------------------------


def test_ac4_name_field_relabels_and_sends_per_mode(index_html, app_js):
    assert 'id="name-field-label"' in index_html
    assert 'id="dataset_name"' in index_html
    # Label text flips per mode.
    assert "Recording session name" in app_js
    assert "Dataset name" in app_js
    # Value is sent as dataset_name (image /settings) and entry_base (video start).
    assert "dataset_name:" in app_js
    assert "entry_base:" in app_js


# --- AC5: mode toggle greyed while recording ---------------------------------


def test_ac5_toggle_greyed_while_recording(app_js):
    # lockMode = recording; both toggle buttons get disabled. (ADR-0002 dropped
    # the drain queue, so there is no "draining" lock condition anymore.)
    assert re.search(r'lockMode\s*=\s*recording', app_js)
    assert "els.modeImage.disabled = lockMode" in app_js
    assert "els.modeVideo.disabled = lockMode" in app_js


# --- AC6: no processing veil, no drain-queue chip (ADR-0002) ------------------


def test_ac6_no_queue_chip_or_veil(index_html, app_js, style_css):
    # The drain-queue chip and the old full-screen veil are both gone from the
    # served markup / JS — /record/stop writes synchronously, nothing to report.
    assert 'id="queue-chip"' not in index_html
    assert 'id="queue-retry"' not in index_html
    assert "postpass-veil" not in index_html
    assert "postpassVeil" not in app_js
    assert "renderQueueChip" not in app_js
    assert ".postpass-veil" not in style_css


def test_ac6_record_status_parsed_as_state_only(app_js):
    # pollRecordStatus now reads just {state: "idle" | "recording"}.
    assert "/record/status" in app_js
    assert "s.state" in app_js
    # The superseded drain block + retry endpoint are gone.
    assert "s.drain" not in app_js
    assert "/record/retry" not in app_js


# --- AC7: toggling modes when idle does NOT re-fire /settings ----------------


def test_ac7_setmode_does_not_call_settings(app_js):
    # setMode is a pure client-side flip: extract its body and prove no fetch.
    m = re.search(r"function setMode\([^)]*\)\s*\{(.*?)\n\}", app_js, re.S)
    assert m, "setMode not found"
    body = m.group(1)
    assert "fetch(" not in body
    assert "/settings" not in body


# --- foreground states collapsed to idle | recording (U2 §INV-1) -------------


def test_no_processing_or_failed_foreground_state(app_js):
    # The old blocking states are gone; the Record flow returns to idle on stop
    # (enqueue), and there is no window.prompt dialog anymore.
    assert 'recState = "processing"' not in app_js
    assert 'recState = "failed"' not in app_js
    assert "window.prompt" not in app_js
