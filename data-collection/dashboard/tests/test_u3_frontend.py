"""U3 frontend — the unified capture UI, asserted against the SERVED markup/JS.

These are *structure* tests, not live-behavior tests: the backend that produces
the new ``/record/status`` ``drain`` shape lands with U2 (concurrently). So we
assert that the frozen DESIGN is present in what the app serves — the mode
toggle, the single relabeling FLAG button, the top-bar queue chip, the removal
of the old full-screen ``postpass-veil`` — and that ``app.js`` contains the
SPACE-dispatch and chip-render logic keyed off the frozen ``drain`` shape
(``{state, drain:{current,queued,eta_seconds}, error}``, U2 §2).

Owned by U3 (REDESIGN §"File-ownership constraints"). Renders "/" and reads the
static assets through ``TestClient(create_app(...))`` with the existing fakes.
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


# --- AC5: mode toggle greyed while recording AND while draining ---------------


def test_ac5_toggle_greyed_while_recording_or_draining(app_js):
    # lockMode = recording OR draining; both toggle buttons get disabled.
    assert re.search(r'lockMode\s*=\s*recording\s*\|\|\s*isDraining\(\)', app_js)
    assert "els.modeImage.disabled = lockMode" in app_js
    assert "els.modeVideo.disabled = lockMode" in app_js
    # Draining is derived from the frozen drain.current field.
    assert "function isDraining" in app_js
    assert "drain.current != null" in app_js


# --- AC6: status chip in the top bar; veil removed ---------------------------


def test_ac6_queue_chip_in_topbar(index_html):
    assert 'id="queue-chip"' in index_html
    assert 'id="queue-chip-text"' in index_html
    assert 'id="queue-retry"' in index_html
    assert 'id="queue-discard"' in index_html


def test_ac6_postpass_veil_removed(index_html, app_js, style_css):
    # The old full-screen veil is gone from markup, JS, and CSS.
    assert "postpass-veil" not in index_html
    assert 'id="postpass-veil"' not in index_html
    assert "postpassVeil" not in app_js
    assert ".postpass-veil" not in style_css


def test_ac6_chip_render_keyed_off_frozen_drain_shape(app_js):
    assert "function renderQueueChip" in app_js
    # Reads the frozen U2 §2 fields off `drain`.
    assert "drain.current" in app_js
    assert "drain.queued" in app_js
    assert "drain.eta_seconds" in app_js
    # `done/total · N queued · ~ETA` composition.
    assert re.search(r'cur\.done.*cur\.total', app_js)
    assert "queued" in app_js
    assert "function formatEta" in app_js


def test_ac6_retry_on_error_targets_entry_name(app_js):
    # Retry surfaces only on a failed head (drainError) and targets it by name.
    assert "drainError" in app_js
    assert "els.queueRetry.hidden = !failed" in app_js
    assert "entry_name:" in app_js
    assert "drain.current.entry_name" in app_js
    # Retry hits /record/retry.
    assert "/record/retry" in app_js


def test_ac6_status_parsed_from_frozen_shape(app_js):
    # pollRecordStatus reads {state, drain:{current,queued,eta_seconds}, error}.
    assert "/record/status" in app_js
    assert "s.drain.current" in app_js
    assert "s.drain.queued" in app_js
    assert "s.drain.eta_seconds" in app_js
    assert "s.error" in app_js
    # The superseded pre-U2 top-level fields are no longer read.
    assert "s.postpass" not in app_js
    assert "s.n_keyframes" not in app_js


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
