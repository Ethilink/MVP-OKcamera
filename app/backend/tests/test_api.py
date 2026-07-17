"""AC1-AC8 for T04: the FastAPI HTTP seam wiring `CaptureLoop` + `Session`
behind the frozen `app/docs/api-contract.md`.

No real camera and no real capture thread anywhere in this suite: `capture`
is always `FakeCapture` below (a tiny double exposing only the surface
`create_app` actually touches -- `set_confidence_and_reset`/`health`/`generation`/
`snapshot`), and `session` is always a real `backend.session.Session` (T02,
already tested on its own in `test_session.py`) driven directly via its
public `observe`/`start`/`stop` methods to simulate what the capture
thread's `on_frame` wiring would otherwise do -- exactly the seam DESIGN.md
draws between the capture thread and the session lock.

`clock`/`now` are always injected (never `time.monotonic`/real wall clock,
per AC7) via the small `_Clock` double and plain fixed-value lambdas, so
every timestamp/duration in this suite is deterministic.

MIGRATED for T11: `Latest` carries same-tick roster/catalog/resolving state,
`_on_frame` receives all of it, and recording rows carry a `colour`. The one
exception to the "`capture` is always `FakeCapture`" rule
above is `TestBV6`, which needs a REAL `CaptureLoop` + `OverlayRenderer`: B-V6
is a wiring claim about the start/stop handlers reaching the renderer, and the
spec leaves the route between them free — so that test asserts the only thing
that is actually contracted, the published overlay's own pixels. It is also the
one test that lets `clock` default to `time.monotonic`: the session and the real
capture thread must read the SAME clock or `start()` would rewind time. It still
never waits on that clock — it waits on `capture.generation`.

`/stream` (AC5) is the one endpoint that must, in the CORRECT
implementation, stream forever (paced by `capture.generation`) -- but in
this dependency stack `httpx`'s ASGI transport (used by both
`fastapi.testclient.TestClient` and a raw `httpx.AsyncClient`) fully drains
an ASGI call before returning ANY response data, which would hang the test
process against a genuinely never-ending generator. `_drive_stream_two_
parts_then_stall` below sidesteps this by driving the app's raw ASGI
`(scope, receive, send)` callable directly -- the actual wire-level
interface any ASGI server (uvicorn included) talks to -- reading chunks as
they're produced and cancelling the task when the test is done. Verified
against a runnable never-ending `StreamingResponse` before use here.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
import supervision as sv
from fastapi.testclient import TestClient

from backend.capture import CaptureLoop, DetectionBox, Latest, TrackerResetError
from backend.fakes import FakeCaptureSource
from backend.main import create_app
from backend.render import CATALOG_PALETTE, UNKNOWN_COLOUR, catalog_colour
from backend.session import Phase, Session
from tests.overlay_probe import GRAY_CHROMA_MAX, chroma_ratio, hex_to_bgr


class _Clock:
    """Settable stand-in for `create_app`'s injected `clock` -- lets a test
    pin exactly what `t` each session-touching handler observes, with no
    real waiting anywhere."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def set(self, value: float) -> None:
        self._value = value

    def __call__(self) -> float:
        return self._value


class FakeCapture:
    """Test double for the `capture` collaborator: the surface `create_app`
    needs from a `CaptureLoop` (`set_confidence_and_reset`,
    `health`, `generation`, `snapshot`, plus the `set_on_frame`/`set_render_fn`
    handoffs) -- never a real camera or background thread.

    The renderer handoff is deliberately PERMISSIVE: `set_render_fn` accepts and
    stores the renderer, and nothing here asserts on it. Since T11/R1 the overlay
    is stateless (no recording flag), so there is nothing for a handler to signal;
    `TestBV6` (real `CaptureLoop` + real `OverlayRenderer`) is what proves the
    overlay wiring works, through the published overlay's pixels.

    `calls` is an optionally shared list so tests can assert lifecycle order.
    """

    def __init__(self, calls: list[str] | None = None) -> None:
        self.health = "ok"
        self.generation = 0
        self.calls: list[str] = calls if calls is not None else []
        self._frames: dict[int, Latest] = {}
        # T11: the confidence PATCH command surface. `set_confidence_values`
        # records each applied value; `_confidence_error` forces a 503; `_gate`
        # can hold the command while a test exercises mutation serialization.
        self.set_confidence_values: list[float] = []
        self._confidence_error: BaseException | None = None
        self._gate: threading.Event | None = None
        self.on_frame = None
        self.render_fn = None

    def set_on_frame(self, cb) -> None:
        self.on_frame = cb

    def set_render_fn(self, render_fn) -> None:
        self.render_fn = render_fn

    def set_frame(self, generation: int, jpeg: bytes) -> None:
        # Mirror the real `CaptureLoop`: each publish rebinds a NEW `Latest`
        # (stored once here), and `snapshot()` returns that SAME object until
        # the next publish — the identity `/stream` paces on (T03 AC4). Snapshot
        # fields the setup branch reads (frame_bgr/detections) get complete, if
        # empty, crop material — never optional-in-production placeholders.
        self.generation = generation
        self._frames[generation] = Latest(
            overlay_jpeg=jpeg,
            present_ids=frozenset(),
            count=0,
            t=0.0,
            frame_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            detections=(),
            roster=frozenset(),
            catalog=frozenset(),
        )

    def set_snapshot(
        self,
        frame_bgr: np.ndarray,
        detections: tuple,
        roster: frozenset[int] | None = None,
        catalog: frozenset[int] | None = None,
    ) -> None:
        """Publish a snapshot carrying real crop material (a frame + boxes) so
        the setup branch of /status has something to crop. Boxes may be given as
        raw `(tracker_id, xyxy)` tuples (the convenient test form) or as real
        `DetectionBox`es — either way they are stored as `DetectionBox`
        (frozen dataclass, T11/B1) with `resolving=False`.

        `roster`/`catalog` default to the detected ids — the ordinary live case,
        where the tray in front of the camera IS the roster/catalog. Tests that
        care about the roster/colour or catalog seam pass them explicitly.
        """
        boxes = tuple(
            box
            if isinstance(box, DetectionBox)
            else DetectionBox(tracker_id=int(box[0]), xyxy=tuple(box[1]), resolving=False)
            for box in detections
        )
        detected = frozenset(box.tracker_id for box in boxes)
        self.generation += 1
        self._frames[self.generation] = Latest(
            overlay_jpeg=b"",
            present_ids=detected,
            count=len(boxes),
            t=0.0,
            frame_bgr=frame_bgr,
            detections=boxes,
            roster=detected if roster is None else roster,
            catalog=detected if catalog is None else catalog,
        )

    def snapshot(self) -> Latest | None:
        return self._frames.get(self.generation)

    def fail_confidence_with(self, exc: BaseException) -> None:
        self._confidence_error = exc

    def gate_mutations(self) -> threading.Event:
        """Make the tracker-mutating capture command block until the returned
        Event is set, so a test can hold one handler inside its capture command
        (still holding the mutation lock) and prove a second cannot proceed."""
        self._gate = threading.Event()
        return self._gate

    def set_confidence_and_reset(self, confidence: float, timeout_s: float = 2.0) -> None:
        self.calls.append("set_confidence_and_reset")
        self.set_confidence_values.append(confidence)
        if self._gate is not None:
            self._gate.wait(timeout_s)
        if self._confidence_error is not None:
            raise self._confidence_error


class _SpySession(Session):
    """A real `Session` whose `start()` also appends to a call-order list
    shared with a `FakeCapture` -- lets AC2 assert Start advances only the
    session and does not mutate the already-approved tracker state."""

    def __init__(self, calls: list[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._calls = calls

    def start(self, t: float) -> None:
        self._calls.append("session_start")
        super().start(t)


def _prime_ready(session: Session, ids: frozenset[int] = frozenset({1})) -> None:
    """Drive `session` past the T11 fail-closed Start gate at t=0.0: the whole
    catalog present, recognised, no unknown, no resolving. Build the session with
    `setup_stable_s=0.0` so it is instantly stable; a Start at clock 0.0 then
    reads it ready, and any post-start observe at t>0 stays monotonic. This is the
    "the operator cleared the tray and it settled" precondition Start now demands
    (before T11, Start reset the tracker and always fired — now it fail-closes)."""
    session.observe(0.0, ids, ids, ids, frozenset())


def _ready_session(ids: frozenset[int] = frozenset({1})) -> Session:
    """A real `Session` already primed to pass the Start gate (see `_prime_ready`)."""
    session = Session(setup_stable_s=0.0)
    _prime_ready(session, ids)
    return session


class TestAC1FreshAppInSetupPhase:
    """AC1: a fresh app: /status -> phase="setup", setup non-null with
    detected_count/stable_for_s, recording null; report/stop -> 409."""

    def test_ac01_status_shows_setup_phase_with_setup_block_and_null_recording(
        self,
    ) -> None:
        app = create_app(FakeCapture(), Session(), "test-model")
        client = TestClient(app)

        body = client.get("/status").json()

        assert body.get("phase") == "setup"
        setup = body.get("setup")
        assert setup is not None
        assert "detected_count" in setup
        assert "stable_for_s" in setup
        assert body.get("recording") is None

    def test_ac01_report_and_stop_are_409_before_any_recording(self) -> None:
        app = create_app(FakeCapture(), Session(), "test-model")
        client = TestClient(app)

        assert client.get("/report").status_code == 409
        assert client.post("/recording/stop").status_code == 409


class TestAC2RecordingStart:
    """AC2: start -> 200 with an aware ISO started_at. Start preserves the
    tracker state that passed the fail-closed readiness gate (see TestStartGate).
    A second start while already recording -> 409; /status then flips to
    "recording" with elapsed_s and lean instrument rows matching the contract's
    field names exactly."""

    def test_ac02_start_preserves_the_approved_tracker_then_returns_started_at(
        self,
    ) -> None:
        calls: list[str] = []
        capture = FakeCapture(calls)
        session = _SpySession(calls, setup_stable_s=0.0)
        _prime_ready(session)
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        resp = client.post("/recording/start")

        assert resp.status_code == 200
        assert calls == ["session_start"]
        started_at = resp.json().get("started_at")
        assert isinstance(started_at, str) and started_at != ""
        assert datetime.fromisoformat(started_at).utcoffset() is not None

    def test_ac02_start_notifies_an_injected_scenario_without_resetting_capture(
        self,
    ) -> None:
        calls: list[str] = []
        capture = FakeCapture(calls)
        session = _SpySession(calls, setup_stable_s=0.0)
        _prime_ready(session)
        app = create_app(
            capture,
            session,
            "test-model",
            clock=_Clock(0.0),
            on_recording_start=lambda: calls.append("scenario_start"),
        )

        response = TestClient(app).post("/recording/start")

        assert response.status_code == 200
        assert calls == ["scenario_start", "session_start"]

    def test_ac02_second_start_while_already_recording_is_409(self) -> None:
        app = create_app(FakeCapture(), _ready_session(), "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        first = client.post("/recording/start")
        assert first.status_code == 200

        second = client.post("/recording/start")

        assert second.status_code == 409

    def test_ac02_status_reflects_recording_phase_with_contract_field_names(
        self,
    ) -> None:
        session = _ready_session()
        app = create_app(FakeCapture(), session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        client.post("/recording/start")
        session.observe(1.0, frozenset({3}))
        session.observe(2.5, frozenset({3}))  # entry confirmed (1.5s > 1.0s debounce)

        body = client.get("/status").json()

        assert body.get("phase") == "recording"
        recording = body.get("recording") or {}
        assert set(recording) == {"started_at", "elapsed_s", "instruments"}
        assert isinstance(recording.get("elapsed_s"), (int, float))
        instruments = recording.get("instruments") or []
        assert len(instruments) == 1
        instrument = instruments[0]
        assert instrument.get("tracker_id") == 3
        assert instrument.get("label") == "Instrument 3"
        assert instrument.get("on_table") is True
        assert set(instrument) == {
            "tracker_id",
            "label",
            "on_table",
            "thumbnail",
            "colour",
        }

    def test_recording_instrument_carries_live_crop_thumbnail(self) -> None:
        """The recording branch attaches a live crop per visible instrument,
        matched by tracker_id from the current snapshot — the same crop path as
        the setup preview, so an off-table instrument (absent from the frame)
        gets no crop and the app falls back to its last-seen one."""
        capture = FakeCapture()
        capture.set_snapshot(
            np.full((120, 160, 3), 90, dtype=np.uint8),  # non-black → encodes
            ((3, (10.0, 10.0, 50.0, 60.0)),),
        )
        session = _ready_session()
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        client.post("/recording/start")
        session.observe(1.0, frozenset({3}))
        session.observe(2.5, frozenset({3}))  # id 3 confirmed on the table

        instruments = client.get("/status").json()["recording"]["instruments"]
        assert len(instruments) == 1
        assert instruments[0]["tracker_id"] == 3
        thumbnail = instruments[0]["thumbnail"]
        assert thumbnail.startswith("data:image/jpeg;base64,")
        base64.b64decode(thumbnail.split(",", 1)[1], validate=True)


class TestAC3StopAndReportShareIdenticalReportShape:
    """AC3: driving the fakes through a scripted pickup, POST
    /recording/stop returns the contract report shape (usage windows with
    off_s/on_s, completeness, model_version, duration_s) and a subsequent
    GET /report returns the identical body; every value is JSON-native (a
    raw json.dumps of the body round-trips -- no numpy types)."""

    def test_ac03_stop_report_shape_and_get_report_return_identical_body(
        self,
    ) -> None:
        session = _ready_session()
        clock = _Clock(0.0)
        app = create_app(FakeCapture(), session, "scenario-0.1", clock=clock)
        client = TestClient(app)

        client.post("/recording/start")
        session.observe(1.0, frozenset({1, 3}))
        session.observe(2.5, frozenset({1, 3}))  # both confirmed on table
        session.observe(20.0, frozenset({1, 3}))  # id1 last seen before leaving
        session.observe(21.6, frozenset({3}))  # id1 confirmed off (1.6s > 1.5s)

        clock.set(30.0)
        stop_resp = client.post("/recording/stop")
        assert stop_resp.status_code == 200
        stop_body = stop_resp.json()

        assert stop_body.get("model_version") == "scenario-0.1"
        assert stop_body.get("duration_s") == pytest.approx(30.0, abs=0.05)

        instruments = {i.get("tracker_id"): i for i in stop_body.get("instruments") or []}
        assert set(instruments) == {1, 3}

        missing = instruments[1]
        assert missing.get("completeness") == "missing"
        usage = missing.get("usage") or []
        assert len(usage) == 1
        assert usage[0].get("off_s") == pytest.approx(20.0, abs=0.05)
        assert usage[0].get("on_s") is None

        present = instruments[3]
        assert present.get("completeness") == "present"
        assert present.get("usage") == []

        report_resp = client.get("/report")
        assert report_resp.status_code == 200
        assert report_resp.json() == stop_body

        assert json.loads(json.dumps(stop_body)) == stop_body

    def test_ac03_stop_notifies_an_injected_scenario_lifecycle(self) -> None:
        events: list[str] = []
        app = create_app(
            FakeCapture(),
            _ready_session(),
            "test-model",
            clock=_Clock(10.0),
            on_recording_start=lambda: events.append("start"),
            on_recording_stop=lambda: events.append("stop"),
        )
        client = TestClient(app)

        assert client.post("/recording/start").status_code == 200
        assert client.post("/recording/stop").status_code == 200

        assert events == ["start", "stop"]


class TestAC4RestartAfterFinishedDiscardsReport:
    """AC4: start-after-finished -> 200, and /report -> 409 until the next
    stop (D7: old report discarded). While phase == "finished", /status
    carries a non-null setup block (D15) and null recording."""

    def test_ac04_status_in_finished_phase_has_setup_block_and_null_recording(
        self,
    ) -> None:
        session = _ready_session()
        clock = _Clock(0.0)
        app = create_app(FakeCapture(), session, "test-model", clock=clock)
        client = TestClient(app)

        client.post("/recording/start")
        session.observe(1.0, frozenset({1}))
        session.observe(2.5, frozenset({1}))
        clock.set(5.0)
        assert client.post("/recording/stop").status_code == 200

        body = client.get("/status").json()

        assert body.get("phase") == "finished"
        assert body.get("setup") is not None
        assert body.get("recording") is None

    def test_ac04_start_after_finished_succeeds_and_report_is_409_until_next_stop(
        self,
    ) -> None:
        session = _ready_session()
        clock = _Clock(0.0)
        app = create_app(FakeCapture(), session, "test-model", clock=clock)
        client = TestClient(app)

        client.post("/recording/start")
        session.observe(1.0, frozenset({1}))
        session.observe(2.5, frozenset({1}))
        clock.set(5.0)
        client.post("/recording/stop")
        assert client.get("/report").status_code == 200  # old report available

        # From FINISHED, the operator re-settles a clean tray (its catalog is
        # observed again) so the gate re-opens, then starts a fresh recording.
        session.observe(5.5, frozenset({1}), frozenset({1}), frozenset({1}), frozenset())
        clock.set(6.0)
        restart = client.post("/recording/start")
        assert restart.status_code == 200

        assert client.get("/report").status_code == 409  # discarded until next stop

        session.observe(7.0, frozenset({1}))
        clock.set(8.0)
        assert client.post("/recording/stop").status_code == 200

        assert client.get("/report").status_code == 200


class TestSetupDetections:
    """`/status` setup block carries per-detection crops (data-URI thumbnails),
    sorted by tracker_id, derived from the capture snapshot — independent of the
    session's detected_count. A missing snapshot yields an empty list, and the
    whole body stays JSON-native (no numpy leaks through the crop path)."""

    @staticmethod
    def _frame() -> np.ndarray:
        # A non-black frame so crops actually encode to JPEG.
        return np.full((120, 160, 3), 90, dtype=np.uint8)

    def test_setup_detections_sorted_with_labels_and_data_uri_thumbnails(self) -> None:
        capture = FakeCapture()
        capture.set_snapshot(
            self._frame(),
            ((2, (60.0, 20.0, 110.0, 80.0)), (1, (10.0, 10.0, 50.0, 60.0))),
        )
        app = create_app(capture, Session(), "test-model")
        client = TestClient(app)

        setup = client.get("/status").json().get("setup") or {}
        detections = setup.get("detections")

        assert [d["tracker_id"] for d in detections] == [1, 2]  # sorted
        assert [d["label"] for d in detections] == ["Instrument 1", "Instrument 2"]
        for detection in detections:
            assert detection["thumbnail"].startswith("data:image/jpeg;base64,")
            # the payload after the comma is valid base64
            base64.b64decode(detection["thumbnail"].split(",", 1)[1], validate=True)

    def test_detected_count_is_independent_of_detections_length(self) -> None:
        session = Session()
        session.observe(0.0, frozenset({1, 2, 3, 4, 5}))  # count comes from here
        capture = FakeCapture()
        capture.set_snapshot(self._frame(), ((1, (10.0, 10.0, 50.0, 60.0)),))  # one tile
        app = create_app(capture, session, "test-model")
        client = TestClient(app)

        setup = client.get("/status").json()["setup"]

        assert setup["detected_count"] == 5
        assert len(setup["detections"]) == 1  # snapshot may lag the id-set by a frame

    def test_no_snapshot_yields_empty_detections(self) -> None:
        app = create_app(FakeCapture(), Session(), "test-model")  # never published
        client = TestClient(app)

        setup = client.get("/status").json()["setup"]

        assert setup["detections"] == []

    def test_finished_phase_also_carries_detections(self) -> None:
        session = _ready_session()
        clock = _Clock(0.0)
        capture = FakeCapture()
        app = create_app(capture, session, "test-model", clock=clock)
        client = TestClient(app)

        client.post("/recording/start")
        session.observe(1.0, frozenset({1}))
        session.observe(2.5, frozenset({1}))
        clock.set(5.0)
        client.post("/recording/stop")
        capture.set_snapshot(self._frame(), ((1, (10.0, 10.0, 50.0, 60.0)),))

        body = client.get("/status").json()

        assert body["phase"] == "finished"
        assert body["setup"]["detections"][0]["tracker_id"] == 1

    def test_status_body_is_json_native_through_the_crop_path(self) -> None:
        capture = FakeCapture()
        capture.set_snapshot(self._frame(), ((1, (10.0, 10.0, 50.0, 60.0)),))
        app = create_app(capture, Session(), "test-model")
        client = TestClient(app)

        body = client.get("/status").json()

        assert json.loads(json.dumps(body)) == body


class TestSetupDetectionIdentity:
    """T11/B6 + backend tests 8-9: setup `detections[]` carry per-item
    state/label/colour from the single identity policy (D4/D5), never expose a raw
    id as a label, and a recognised id's colour is identical in setup and
    recording (pressing Track must not recolour a known instrument)."""

    _CATALOG = frozenset(range(1, 9))  # the eight loaded specimens

    def _frame(self) -> np.ndarray:
        return np.full((120, 160, 3), 90, dtype=np.uint8)

    def test_setup_detections_carry_state_label_colour_and_hide_raw_ids(self) -> None:
        capture = FakeCapture()
        capture.set_snapshot(
            self._frame(),
            (
                DetectionBox(3, (10.0, 10.0, 50.0, 60.0), resolving=False),       # recognised
                DetectionBox(1042, (60.0, 10.0, 100.0, 60.0), resolving=True),    # recognising
                DetectionBox(2001, (110.0, 10.0, 150.0, 60.0), resolving=False),  # unknown
            ),
            roster=frozenset({3}),
            catalog=self._CATALOG,
        )
        app = create_app(capture, Session(), "test-model")
        client = TestClient(app)

        detections = {
            d["tracker_id"]: d for d in client.get("/status").json()["setup"]["detections"]
        }

        assert detections[3]["state"] == "recognised"
        assert detections[3]["label"] == "Instrument 3"
        assert detections[3]["colour"] == catalog_colour(self._CATALOG, 3)
        assert detections[3]["colour"] != UNKNOWN_COLOUR

        assert detections[1042]["state"] == "recognising"
        assert detections[1042]["label"] == ""            # a spinner, no name yet
        assert detections[1042]["colour"] == UNKNOWN_COLOUR

        assert detections[2001]["state"] == "unknown"
        assert detections[2001]["label"] == "Unknown"
        assert detections[2001]["colour"] == UNKNOWN_COLOUR

        # D4: a raw tracker id is never leaked into a user-facing label.
        for detection in detections.values():
            assert "1042" not in detection["label"]
            assert "2001" not in detection["label"]

    def test_recognised_colour_is_identical_in_setup_and_recording(self) -> None:
        # backend test 9: the swatch a known instrument wears must not change when
        # Track is pressed — both come from catalog_colour(catalog, id).
        roster = frozenset({3})
        capture = FakeCapture()
        capture.set_snapshot(
            self._frame(),
            (DetectionBox(3, (10.0, 10.0, 50.0, 60.0), resolving=False),),
            roster=roster,
            catalog=self._CATALOG,
        )
        session = _ready_session(frozenset({3}))
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        setup_colour = client.get("/status").json()["setup"]["detections"][0]["colour"]

        client.post("/recording/start")
        _confirm_recording(session, frozenset({3}), roster)
        recording_colour = client.get("/status").json()["recording"]["instruments"][0]["colour"]

        assert setup_colour == recording_colour == catalog_colour(self._CATALOG, 3)

    def test_recognised_colour_is_identical_across_start_for_a_full_roster(self) -> None:
        # The full-roster half of backend test 9 (SPEC-C asks partial AND full): a
        # regression that rotated colours only once the whole tray is recognised
        # (roster == catalog) would slip past the partial-roster case above.
        full = frozenset({1, 2, 3})
        capture = FakeCapture()
        capture.set_snapshot(
            self._frame(),
            (DetectionBox(2, (10.0, 10.0, 50.0, 60.0), resolving=False),),
            roster=full,
            catalog=full,
        )
        session = _ready_session(full)
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        setup_colour = {
            d["tracker_id"]: d["colour"]
            for d in client.get("/status").json()["setup"]["detections"]
        }[2]

        client.post("/recording/start")
        _confirm_recording(session, full, full)
        recording_colour = {
            i["tracker_id"]: i["colour"]
            for i in client.get("/status").json()["recording"]["instruments"]
        }[2]

        assert setup_colour == recording_colour == catalog_colour(full, 2)


def _content_type(start_message: dict | None) -> str:
    """Extract the `content-type` header value from a captured
    `http.response.start` ASGI message; never raises, even on `None`."""
    if not start_message:
        return ""
    headers = dict(start_message.get("headers", []))
    return headers.get(b"content-type", b"").decode()


async def _drive_stream_two_parts_then_stall(
    app,
    advance,
    chunk_timeout: float = 2.0,
    stall_timeout: float = 0.3,
):
    """Issue a raw ASGI GET /stream against `app`, collect the response
    start message plus the first body chunk, call `advance()` (synchronous;
    runs with no other coroutine interleaved, since everything here shares
    one event loop), then collect a second chunk and check whether a THIRD
    one shows up within `stall_timeout` once nothing further changes.

    Every "await a message" step degrades to `None` on timeout rather than
    raising, so a caller always gets a value to assert on (never an
    unhandled `asyncio.TimeoutError`) regardless of how wrong/short the
    implementation under test is.
    """
    messages: asyncio.Queue = asyncio.Queue()

    async def receive():
        await asyncio.Event().wait()  # a client that never disconnects

    async def send(message: dict) -> None:
        await messages.put(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/stream",
        "raw_path": b"/stream",
        "headers": [],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 123),
        "scheme": "http",
        "root_path": "",
    }

    async def _next(timeout: float) -> dict | None:
        try:
            return await asyncio.wait_for(messages.get(), timeout)
        except asyncio.TimeoutError:
            return None

    task = asyncio.create_task(app(scope, receive, send))
    try:
        start = await _next(chunk_timeout)
        first_msg = await _next(chunk_timeout)
        advance()
        second_msg = await _next(chunk_timeout)
        third_msg = await _next(stall_timeout)
        return (
            start,
            first_msg.get("body") if first_msg else None,
            second_msg.get("body") if second_msg else None,
            third_msg,
        )
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestAC5MjpegStreamPacedByGeneration:
    """AC5: /stream responds multipart/x-mixed-replace and yields >= 2
    distinct JPEG parts as capture.generation advances; once generation
    stalls, no further (duplicate) part shows up -- no busy loop."""

    def test_ac05_stream_content_type_and_generation_paced_distinct_parts(
        self,
    ) -> None:
        capture = FakeCapture()
        capture.set_frame(1, b"JPEG-FRAME-ONE")
        app = create_app(capture, Session(), "test-model")

        def advance() -> None:
            capture.set_frame(2, b"JPEG-FRAME-TWO")

        start, first, second, third = asyncio.run(
            _drive_stream_two_parts_then_stall(app, advance)
        )

        assert _content_type(start).startswith("multipart/x-mixed-replace")
        assert first is not None and b"JPEG-FRAME-ONE" in first
        assert second is not None and b"JPEG-FRAME-TWO" in second
        assert first != second
        assert third is None  # stalled: no third part shows up


async def _drive_stream_collect_bodies(app, capture, frames, drain_timeout: float = 0.2):
    """Issue a raw ASGI GET /stream, then walk through `frames`: publish each
    one and drain every body chunk that appears before `drain_timeout` of
    silence. The drain window spans many `_STREAM_POLL_S` cycles, so a broken
    dedup (emitting on every poll) would surface as duplicate bodies; correct
    snapshot-identity pacing yields exactly one body per distinct frame."""
    messages: asyncio.Queue = asyncio.Queue()

    async def receive():
        await asyncio.Event().wait()  # a client that never disconnects

    async def send(message: dict) -> None:
        await messages.put(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/stream",
        "raw_path": b"/stream",
        "headers": [],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 123),
        "scheme": "http",
        "root_path": "",
    }

    async def _drain() -> list[bytes]:
        bodies: list[bytes] = []
        while True:
            try:
                msg = await asyncio.wait_for(messages.get(), drain_timeout)
            except asyncio.TimeoutError:
                return bodies
            if msg.get("type") == "http.response.body":
                bodies.append(msg.get("body"))

    task = asyncio.create_task(app(scope, receive, send))
    try:
        collected: list[bytes] = []
        collected += await _drain()  # first frame (already published) + idle polls
        for jpeg in frames[1:]:
            capture.set_frame(capture.generation + 1, jpeg)
            collected += await _drain()
        return collected
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestStreamHasNoDuplicateParts:
    """T04: `/stream` sends a part only when generation advances. Publishing N
    distinct frames — each held across many poll cycles — yields <= N parts."""

    def test_n_distinct_frames_yield_at_most_n_parts(self) -> None:
        frames = [b"JPEG-A", b"JPEG-B", b"JPEG-C"]
        capture = FakeCapture()
        capture.set_frame(1, frames[0])
        app = create_app(capture, Session(), "test-model")

        bodies = asyncio.run(_drive_stream_collect_bodies(app, capture, frames))

        assert 0 < len(bodies) <= len(frames)  # no duplicate parts
        for i, jpeg in enumerate(frames[: len(bodies)]):
            assert jpeg in bodies[i]


class TestAC6CaptureHealthMapping:
    """AC6: /status shows capture_health:"stalled" whenever capture.health
    isn't exactly "ok" (covers both a stale and a dead capture loop)."""

    @pytest.mark.parametrize("raw_health", ["stale", "dead"])
    def test_ac06_non_ok_capture_health_is_reported_as_stalled(
        self, raw_health: str
    ) -> None:
        capture = FakeCapture()
        capture.health = raw_health
        app = create_app(capture, Session(), "test-model")
        client = TestClient(app)

        body = client.get("/status").json()

        assert body.get("capture_health") == "stalled"


_STARTED_AT_DT = datetime(2026, 7, 20, 10, 31, 4, tzinfo=timezone(timedelta(hours=2)))
_STOPPED_AT_DT = datetime(2026, 7, 20, 10, 36, 40, tzinfo=timezone(timedelta(hours=2)))


class TestAC7InjectableClockAndNow:
    """AC7: create_app never touches a real camera or wall clock -- both
    `clock` and `now` are injectable; pinning `now` produces a
    deterministic, timezone-aware started_at/stopped_at in the report
    (using the contract doc's own worked example values)."""

    def test_ac07_pinned_now_produces_deterministic_report_timestamps(
        self,
    ) -> None:
        timestamps = iter([_STARTED_AT_DT, _STOPPED_AT_DT])
        clock = _Clock(0.0)
        app = create_app(
            FakeCapture(),
            _ready_session(),
            "test-model",
            clock=clock,
            now=lambda: next(timestamps),
        )
        client = TestClient(app)

        start_resp = client.post("/recording/start")
        assert start_resp.status_code == 200
        assert start_resp.json().get("started_at") == "2026-07-20T10:31:04+02:00"

        clock.set(336.0)
        stop_resp = client.post("/recording/stop")
        stop_body = stop_resp.json()

        assert stop_body.get("started_at") == "2026-07-20T10:31:04+02:00"
        assert stop_body.get("stopped_at") == "2026-07-20T10:36:40+02:00"
        assert stop_body.get("duration_s") == pytest.approx(336.0, abs=0.05)


class TestAC7MainFakeSmoke:
    """AC7: `main(["--fake"])` constructs the fake stack (ScenarioTracker +
    FakeCaptureSource) into a working app -- smoke-tested by patching
    `uvicorn.run` (the one genuine external-process boundary `main()` owns)
    to capture the app it was handed instead of actually serving it, then
    hitting /status through a TestClient built on that captured app."""

    def test_ac07_main_fake_builds_an_app_that_answers_status_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import backend.main as main_module

        captured: dict = {}

        def fake_run(app, **kwargs) -> None:
            captured["app"] = app

        monkeypatch.setattr(main_module.uvicorn, "run", fake_run)

        main_module.main(["--fake"])

        assert captured.get("app") is not None
        client = TestClient(captured["app"])
        resp = client.get("/status")
        assert resp.status_code == 200

    def test_ac07_main_can_select_a_different_toml_profile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import backend.main as main_module
        from backend.mvp_settings import DEFAULT_MVP_CONFIG_PATH

        custom_config = tmp_path / "rehearsal.toml"
        custom_config.write_text(
            DEFAULT_MVP_CONFIG_PATH.read_text().replace(
                "fake_fps = 10.0", "fake_fps = 7.0"
            )
        )
        captured: dict = {}
        real_scenario_tracker = main_module.ScenarioTracker

        def scenario_tracker(*, fps):
            captured["fake_fps"] = fps
            return real_scenario_tracker(fps=fps)

        monkeypatch.setattr(main_module, "ScenarioTracker", scenario_tracker)
        monkeypatch.setattr(main_module.uvicorn, "run", lambda app, **kwargs: None)

        main_module.main(["--fake", "--config", str(custom_config)])

        assert captured["fake_fps"] == 7.0


def _resolve_schema(openapi: dict, schema: dict) -> dict:
    """Resolve a top-level `$ref` (FastAPI emits these for declared Pydantic
    response models) into its `components.schemas` definition; a schema
    with no `$ref` is returned unchanged. Never raises -- missing keys
    resolve to `{}`."""
    ref = schema.get("$ref")
    if not ref:
        return schema
    node = openapi
    for part in ref.lstrip("#/").split("/"):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node


class TestAC8OpenApiDocsAndResponseSchema:
    """AC8: /docs is served, and the /status response has a declared
    (Pydantic) schema exposing the contract's field names in openapi.json,
    so T08 can generate the frontend's TS types from it."""

    def test_ac08_docs_served_and_status_schema_matches_contract_fields(
        self,
    ) -> None:
        app = create_app(FakeCapture(), Session(), "test-model")
        client = TestClient(app)

        assert client.get("/docs").status_code == 200

        openapi = client.get("/openapi.json").json()
        status_get = openapi.get("paths", {}).get("/status", {}).get("get", {})
        content = status_get.get("responses", {}).get("200", {}).get("content", {})
        schema = content.get("application/json", {}).get("schema", {})
        schema = _resolve_schema(openapi, schema)

        properties = set(schema.get("properties", {}))
        assert {"phase", "capture_health", "model_version", "setup", "recording"} <= properties


# --- T10: the roster crosses the seam ----------------------------------------

def _status_frame() -> np.ndarray:
    """A non-black frame, so the crop path actually encodes JPEGs."""
    return np.full((120, 160, 3), 90, dtype=np.uint8)


def _confirm_recording(session: Session, ids: frozenset[int], roster: frozenset[int]) -> None:
    """Two observes past the 1.0s entry debounce, so `ids & roster` is
    confirmed on the table before the test reads /status."""
    session.observe(1.0, ids, roster)
    session.observe(3.0, ids, roster)


class _RecordingSession(Session):
    """A real `Session` that also records the exact arguments the wiring handed
    `observe` — B-S4 is a claim about what main.py forwards, and the roster's
    effect is already Session's own tested business (test_session.py)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.observed: list[tuple[float, frozenset[int], frozenset[int] | None]] = []

    def observe(self, t, present_ids, roster=None, catalog=None, resolving_ids=None) -> None:
        self.observed.append((t, present_ids, roster))
        super().observe(t, present_ids, roster, catalog, resolving_ids)


class TestBS4OnFrameForwardsTheRoster:
    """B-S4: `main._on_frame` forwards the roster from the capture callback
    into `session.observe`, under the existing lock."""

    def test_b_s4_on_frame_forwards_the_roster_it_was_given_into_session_observe(
        self,
    ) -> None:
        capture = FakeCapture()
        session = _RecordingSession()
        create_app(capture, session, "test-model", clock=_Clock(0.0))

        capture.on_frame(1.0, frozenset({1, 9}), frozenset({1}), frozenset({1}), frozenset())

        assert session.observed == [(1.0, frozenset({1, 9}), frozenset({1}))]

    def test_b_s4_every_published_frame_forwards_its_own_roster(self) -> None:
        # The roster is empty for ~0.7s after Start (the linker's enrolment
        # freeze hasn't fired) and populated afterwards: the wiring must relay
        # each frame's own value, not the first one it ever saw.
        capture = FakeCapture()
        session = _RecordingSession()
        create_app(capture, session, "test-model", clock=_Clock(0.0))

        capture.on_frame(1.0, frozenset({1, 2}), frozenset(), frozenset(), frozenset())
        capture.on_frame(2.0, frozenset({1, 2}), frozenset({1, 2}), frozenset({1, 2}), frozenset())

        assert [roster for _, _, roster in session.observed] == [
            frozenset(),
            frozenset({1, 2}),
        ]


class TestBA1StatusCarriesTheCatalogColour:
    """B-A1 (T11/R2): every `/status` recording entry carries `colour` =
    `catalog_colour(catalog, tracker_id)` with the CATALOG taken from the current
    capture snapshot — so the panel swatch and the overlay mask can never drift,
    and a partial roster can never shift a known colour (D5). With no snapshot yet,
    a gray placeholder stands in."""

    _CATALOG = frozenset(range(1, 9))  # the eight loaded specimens

    def test_b_a1_recording_entries_carry_the_colour_catalog_colour_reports(
        self,
    ) -> None:
        roster = frozenset({1, 2, 3})
        capture = FakeCapture()
        capture.set_snapshot(
            _status_frame(),
            ((1, (10.0, 10.0, 50.0, 60.0)), (2, (60.0, 10.0, 100.0, 60.0))),
            roster=roster,
            catalog=self._CATALOG,
        )
        session = _ready_session()
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        client.post("/recording/start")
        _confirm_recording(session, frozenset({1, 2}), roster)

        instruments = client.get("/status").json()["recording"]["instruments"]
        assert {i["tracker_id"]: i["colour"] for i in instruments} == {
            1: catalog_colour(self._CATALOG, 1),
            2: catalog_colour(self._CATALOG, 2),
        }

    def test_b_a1_a_full_tray_gets_eight_distinct_swatches(self) -> None:
        # OC-SORT hands out a raw counter, so a tray of 8 is any 8 ids — and the
        # catalog is exactly those eight loaded specimens.
        catalog = frozenset({3, 5, 7, 9, 10, 11, 12, 14})
        capture = FakeCapture()
        capture.set_snapshot(_status_frame(), (), roster=catalog, catalog=catalog)
        session = _ready_session()
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        client.post("/recording/start")
        _confirm_recording(session, catalog, catalog)

        instruments = client.get("/status").json()["recording"]["instruments"]
        assert len(instruments) == 8  # sanity: the whole tray confirmed
        assert len({i["colour"] for i in instruments}) == 8

    def test_b_a1_a_partial_roster_does_not_shift_a_known_colour(self) -> None:
        # D5: id 5's colour keys off the full catalog, so it is identical whether
        # only 5 is recognised or the whole tray is. The roster is NOT the basis.
        roster = frozenset({5})
        capture = FakeCapture()
        capture.set_snapshot(_status_frame(), (), roster=roster, catalog=self._CATALOG)
        session = _ready_session()
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        client.post("/recording/start")
        _confirm_recording(session, frozenset({5}), roster)

        instruments = client.get("/status").json()["recording"]["instruments"]
        assert {i["tracker_id"]: i["colour"] for i in instruments} == {
            5: catalog_colour(self._CATALOG, 5)  # palette slot 4, not slot 0
        }

    def test_b_a1_an_instruments_colour_survives_an_absence(self) -> None:
        # The linker re-emits original session ids, so a returned instrument
        # must come back wearing the same swatch it left with.
        roster = frozenset({1, 2, 3})
        capture = FakeCapture()
        capture.set_snapshot(_status_frame(), (), roster=roster, catalog=self._CATALOG)
        session = _ready_session()
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        client.post("/recording/start")
        _confirm_recording(session, frozenset({1, 2}), roster)
        before = {i["tracker_id"]: i["colour"] for i in client.get("/status").json()["recording"]["instruments"]}

        session.observe(20.0, frozenset({1}), roster)  # id 2 is picked up
        session.observe(22.0, frozenset({1}), roster)
        session.observe(40.0, frozenset({1, 2}), roster)  # ... and comes back
        session.observe(42.0, frozenset({1, 2}), roster)

        after = {i["tracker_id"]: i["colour"] for i in client.get("/status").json()["recording"]["instruments"]}
        assert after == before
        assert len(set(before.values())) == 2  # sanity: not one colour for both

    def test_b_a1_colour_falls_back_to_a_gray_placeholder_before_the_first_frame(
        self,
    ) -> None:
        capture = FakeCapture()  # nothing published yet
        session = _ready_session()
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        client.post("/recording/start")
        _confirm_recording(session, frozenset({3}), frozenset({3}))

        assert capture.snapshot() is None  # sanity: the fallback is what's exercised
        response = client.get("/status")
        assert response.status_code == 200  # a transient, HARMLESS placeholder
        colour = response.json()["recording"]["instruments"][0]["colour"]
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", colour), colour
        # A placeholder must not impersonate a real swatch, and must read gray.
        assert colour not in CATALOG_PALETTE
        assert chroma_ratio(hex_to_bgr(colour)) <= GRAY_CHROMA_MAX


class TestBA2UnknownsAreAbsentFromTheApi:
    """B-A2: no other endpoint or shape changes. Unknown ids are absent from
    `/status` entirely — B-S1 guarantees it, so main.py filters nothing."""

    def test_b_a2_a_not_in_roster_id_never_appears_in_status(self) -> None:
        capture = FakeCapture()
        session = _ready_session()
        app = create_app(capture, session, "test-model", clock=_Clock(0.0))
        client = TestClient(app)

        client.post("/recording/start")
        # Driven through the real capture callback — the path the capture
        # thread takes on every published frame (the 5-tuple T11 widened it to).
        capture.on_frame(1.0, frozenset({1, 9}), frozenset({1}), frozenset({1}), frozenset())
        capture.on_frame(3.0, frozenset({1, 9}), frozenset({1}), frozenset({1}), frozenset())

        recording = client.get("/status").json()["recording"]
        assert [i["tracker_id"] for i in recording["instruments"]] == [1]

    def test_b_a2_the_status_instrument_row_gains_only_the_colour_field(self) -> None:
        app = create_app(FakeCapture(), Session(), "test-model")
        client = TestClient(app)

        openapi = client.get("/openapi.json").json()
        schema = openapi.get("components", {}).get("schemas", {}).get("InstrumentStatusModel", {})

        assert set(schema.get("properties", {})) == {
            "tracker_id",
            "label",
            "on_table",
            "thumbnail",
            "colour",
        }


class TestBA3TheReportIsUnchanged:
    """B-A3: the report keeps its exact shape; its instruments are roster-only
    purely as a consequence of B-S1."""

    def test_b_a3_report_instruments_are_roster_only(self) -> None:
        capture = FakeCapture()
        session = _ready_session()
        clock = _Clock(0.0)
        app = create_app(capture, session, "test-model", clock=clock)
        client = TestClient(app)

        client.post("/recording/start")
        capture.on_frame(1.0, frozenset({1, 9}), frozenset({1}), frozenset({1}), frozenset())
        capture.on_frame(3.0, frozenset({1, 9}), frozenset({1}), frozenset({1}), frozenset())
        clock.set(10.0)

        report = client.post("/recording/stop").json()

        assert [i["tracker_id"] for i in report["instruments"]] == [1]

    def test_b_a3_the_report_shape_is_untouched_by_t10(self) -> None:
        session = _ready_session()
        clock = _Clock(0.0)
        app = create_app(FakeCapture(), session, "test-model", clock=clock)
        client = TestClient(app)

        client.post("/recording/start")
        _confirm_recording(session, frozenset({1}), frozenset({1}))
        clock.set(10.0)

        body = client.post("/recording/stop").json()

        assert set(body) == {
            "started_at",
            "stopped_at",
            "duration_s",
            "model_version",
            "instruments",
        }
        assert body["instruments"]  # sanity: there is a row to inspect
        for instrument in body["instruments"]:
            # No `colour` here: the swatch is a live-panel concern, and the
            # report's shape is frozen by the contract.
            assert set(instrument) == {"tracker_id", "label", "completeness", "usage"}


_OVERLAY_SIZE = (320, 240)  # (width, height)
_ROSTER_BOX = (20, 60, 110, 150)
_FOREIGN_BOX = (170, 60, 260, 150)


class _FixedRosterTracker:
    """A tracker with a fixed roster/catalog and fixed box geometry: instrument 1
    (in the roster and catalog) and object 9 (in neither). Fixed geometry is what
    lets a test decode the published overlay and look at exactly the pixels it
    means.

    `emit_foreign` (T11): when True the tray also carries object 9. A test clears
    it (False) so the fail-closed Start gate opens — you cannot begin recording
    with an unknown object on the tray — then flips it back so the foreign object
    reappears mid-recording, exactly as it does in the real scenario."""

    def __init__(self) -> None:
        self.confidence = 0.5
        self.emit_foreign = True

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    @property
    def roster(self) -> frozenset[int]:
        return frozenset({1})

    @property
    def catalog(self) -> frozenset[int]:
        return frozenset({1})

    def reset(self) -> None:
        pass

    def update(self, frame: np.ndarray) -> sv.Detections:
        height, width = frame.shape[:2]
        ids = [1]
        boxes = [_ROSTER_BOX]
        if self.emit_foreign:
            ids.append(9)
            boxes.append(_FOREIGN_BOX)
        mask = np.zeros((len(ids), height, width), dtype=bool)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask[i, y1:y2, x1:x2] = True
        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            mask=mask,
            confidence=np.full(len(ids), 0.9, dtype=np.float32),
            class_id=np.zeros(len(ids), dtype=int),
            tracker_id=np.array(ids, dtype=int),
        )


def _advance_frames(capture: CaptureLoop, n: int = 3, timeout: float = 2.0) -> bool:
    """Wait until `n` more frames have been published. Paced by the capture's
    OWN generation counter, never by a wall-clock guess at frame timing."""
    target = capture.generation + n
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if capture.generation >= target:
            return True
        time.sleep(0.005)
    return capture.generation >= target


def _overlay_chroma(capture: CaptureLoop, box: tuple[int, int, int, int]) -> float:
    """Decode the published overlay and measure how far the pixels inside `box`
    are from gray — the one thing B-V6 can observe without knowing HOW the
    handlers reach the renderer (the spec leaves that route free)."""
    latest = capture.snapshot()
    assert latest is not None, "nothing published yet"
    frame = cv2.imdecode(np.frombuffer(latest.overlay_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = box
    region = frame[y1 + 10 : y2 - 10, x1 + 10 : x2 - 10]
    return chroma_ratio(region.reshape(-1, 3).astype(np.float64).mean(axis=0))


class TestBV6TheOverlayIsRosterAwareEndToEnd:
    """B-V6, revised for T11/R1: end-to-end against the real `CaptureLoop` + its
    default `OverlayRenderer`, through the published overlay. Since R1 there is ONE
    identity policy in every phase, so the contracted observable is now that the
    policy is CONSISTENT: a not-in-roster object is gray in setup, recording, AND
    finished (never a coloured 9th instrument), while a roster instrument keeps its
    colour throughout. The start/stop handlers still drive the recording lifecycle
    and the fail-closed gate — this proves that real wiring — but pressing Track no
    longer recolours anything (the whole point of the ticket)."""

    def _app(self):
        tracker = _FixedRosterTracker()
        cap = FakeCaptureSource(size=_OVERLAY_SIZE, fps=None)
        capture = CaptureLoop(
            tracker, 0, cap_factory=lambda _idx: cap, frame_size=_OVERLAY_SIZE
        )
        # The real capture thread stamps `t` with time.monotonic, so the session
        # must read the same clock or start() would rewind time. setup_stable_s=0
        # keeps the readiness gate off the wall clock, so a cleared tray is ready
        # to start as soon as it is observed (no real 2 s wait in the test).
        session = Session(setup_stable_s=0.0)
        return tracker, capture, TestClient(create_app(capture, session, "test-model"))

    @staticmethod
    def _clear_tray_and_start(tracker, capture, client) -> None:
        """Remove the foreign object so the fail-closed Start gate opens, start
        recording, then let the foreign object reappear mid-recording."""
        tracker.emit_foreign = False
        assert _advance_frames(capture, 3)  # the session observes the clean tray
        assert client.post("/recording/start").status_code == 200
        tracker.emit_foreign = True

    def test_b_v6_foreign_object_is_gray_consistently_from_setup_through_finished(
        self,
    ) -> None:
        tracker, capture, client = self._app()

        capture.start()
        try:
            assert _advance_frames(capture, 2)
            setup_chroma = _overlay_chroma(capture, _FOREIGN_BOX)

            self._clear_tray_and_start(tracker, capture, client)  # foreign reappears mid-recording
            assert _advance_frames(capture, 3)
            recording_chroma = _overlay_chroma(capture, _FOREIGN_BOX)

            assert client.post("/recording/stop").status_code == 200
            assert _advance_frames(capture, 3)
            finished_chroma = _overlay_chroma(capture, _FOREIGN_BOX)
        finally:
            capture.stop()

        # R1: the foreign object (not in the roster) is gray in EVERY phase — the
        # setup overlay no longer presents it as a coloured 9th instrument, so
        # pressing Track causes no gray-flash / recolour for unchanged detections.
        assert setup_chroma <= GRAY_CHROMA_MAX
        assert recording_chroma <= GRAY_CHROMA_MAX
        assert finished_chroma <= GRAY_CHROMA_MAX

    def test_b_v6_a_roster_instrument_keeps_a_colour_through_the_whole_run(
        self,
    ) -> None:
        # The mirror of the test above, on the id that IS in the roster: Start
        # must not gray out the tray, only what is foreign to it.
        tracker, capture, client = self._app()

        capture.start()
        try:
            assert _advance_frames(capture, 2)
            self._clear_tray_and_start(tracker, capture, client)
            assert _advance_frames(capture, 3)
            recording_chroma = _overlay_chroma(capture, _ROSTER_BOX)
        finally:
            capture.stop()

        assert recording_chroma > GRAY_CHROMA_MAX

    def test_b_v6_a_wrong_phase_409_leaves_the_roster_aware_overlay_intact(self) -> None:
        tracker, capture, client = self._app()

        capture.start()
        try:
            assert _advance_frames(capture, 2)
            assert client.post("/recording/stop").status_code == 409  # not recording
            assert _advance_frames(capture, 3)
            setup_foreign = _overlay_chroma(capture, _FOREIGN_BOX)
            setup_roster = _overlay_chroma(capture, _ROSTER_BOX)

            self._clear_tray_and_start(tracker, capture, client)
            assert _advance_frames(capture, 3)
            assert client.post("/recording/start").status_code == 409  # already recording
            assert _advance_frames(capture, 3)
            recording_foreign = _overlay_chroma(capture, _FOREIGN_BOX)
            recording_roster = _overlay_chroma(capture, _ROSTER_BOX)
        finally:
            capture.stop()

        # A refused transition leaves the (phase-independent) roster-aware overlay
        # intact: the foreign object stays gray and the roster instrument stays
        # coloured, in both setup and recording.
        assert setup_foreign <= GRAY_CHROMA_MAX
        assert setup_roster > GRAY_CHROMA_MAX
        assert recording_foreign <= GRAY_CHROMA_MAX
        assert recording_roster > GRAY_CHROMA_MAX


# --- T11 Phase 2: the fail-closed Start gate, prepare, and the confidence PATCH -


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.005) -> bool:
    """Poll `predicate` until true or `timeout` elapses — used only to
    synchronise the two-thread mutation-lock test on an observable state change
    (a call recorded), never on business timing."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class TestStartGate:
    """T11/B4, backend test 4: `POST /recording/start` is a server-enforced,
    fail-closed gate. It returns 409 with the mapped detail for EVERY not-ready
    state — recognising, unknown objects, missing instruments, hold-steady, and an
    unhealthy capture — even when called directly (the frontend disabling the
    button is not the enforcement). A successful Start preserves the exact
    tracker roster/catalog state that passed the gate (backend test 5)."""

    def _app(self, session, capture=None):
        capture = capture if capture is not None else FakeCapture()
        client = TestClient(create_app(capture, session, "test-model", clock=_Clock(0.0)))
        return capture, client

    def test_recognising_blocks_start_with_409(self) -> None:
        session = Session(setup_stable_s=0.0)
        catalog = frozenset({1, 2, 3})
        session.observe(0.0, catalog, catalog, catalog, frozenset({2}))  # id 2 resolving
        _, client = self._app(session)

        resp = client.post("/recording/start")

        assert resp.status_code == 409
        assert resp.json()["detail"] == "recognition still in progress"

    def test_unknown_object_blocks_start_with_409(self) -> None:
        session = Session(setup_stable_s=0.0)
        catalog = frozenset({1, 2, 3})
        session.observe(0.0, catalog | {9}, catalog, catalog, frozenset())  # id 9 settled unknown
        _, client = self._app(session)

        resp = client.post("/recording/start")

        assert resp.status_code == 409
        assert resp.json()["detail"] == "remove unknown objects before starting"

    def test_missing_instrument_blocks_start_with_409(self) -> None:
        session = Session(setup_stable_s=0.0)
        catalog = frozenset({1, 2, 3})
        session.observe(0.0, frozenset({1, 2}), frozenset({1, 2}), catalog, frozenset())
        _, client = self._app(session)

        resp = client.post("/recording/start")

        assert resp.status_code == 409
        assert resp.json()["detail"] == "all 3 instruments must be recognised before starting"

    def test_hold_steady_blocks_start_with_409(self) -> None:
        session = Session(setup_stable_s=5.0)  # never settles at clock 0.0
        catalog = frozenset({1, 2, 3})
        session.observe(0.0, catalog, catalog, catalog, frozenset())
        _, client = self._app(session)

        resp = client.post("/recording/start")

        assert resp.status_code == 409
        assert resp.json()["detail"] == "hold the tray steady before starting"

    def test_unhealthy_capture_blocks_start_even_when_readiness_is_ready(self) -> None:
        # Capture health outranks every readiness reason: a ready tray still
        # cannot start while the camera is stalled.
        capture = FakeCapture()
        capture.health = "stale"
        _, client = self._app(_ready_session(), capture=capture)

        resp = client.post("/recording/start")

        assert resp.status_code == 409
        assert resp.json()["detail"] == "capture stalled"

    def test_a_successful_start_does_not_mutate_the_tracker(self) -> None:
        # Backend test 5: the approved roster is the recording roster. Start is a
        # session phase transition and touches neither reset nor confidence.
        capture = FakeCapture()
        _, client = self._app(_ready_session(), capture=capture)

        assert client.post("/recording/start").status_code == 200

        assert capture.calls == []
        assert capture.set_confidence_values == []

    def test_status_carries_detector_control_with_bounds(self) -> None:
        capture = FakeCapture()
        client = TestClient(
            create_app(capture, Session(), "test-model", default_confidence=0.55, clock=_Clock(0.0))
        )
        control = client.get("/status").json()["detector_control"]
        assert control["confidence"] == pytest.approx(0.55)
        assert control["default_confidence"] == pytest.approx(0.55)
        assert control["minimum"] == pytest.approx(0.30)
        assert control["maximum"] == pytest.approx(0.90)
        assert control["step"] == pytest.approx(0.05)


class TestMutationSerialisation:
    """T11, backend test 7: start / confidence are serialised on the coarse
    mutation lock, so two tracker mutations can never overlap. With a FakeCapture
    whose commands block on a gate, a second confidence change cannot reach its own
    capture command until the first releases the lock."""

    def test_a_second_mutation_waits_for_the_first_to_release_the_lock(self) -> None:
        capture = FakeCapture()
        gate = capture.gate_mutations()  # capture commands block until gate is set
        app = create_app(
            capture, Session(), "test-model", default_confidence=0.50, clock=_Clock(0.0)
        )
        client_a = TestClient(app)
        client_b = TestClient(app)

        first = threading.Thread(
            target=lambda: client_a.patch(
                "/settings/detection-confidence", json={"confidence": 0.60}
            )
        )
        first.start()
        try:
            # A is now inside set_confidence_and_reset, holding the mutation lock.
            assert _wait_for(
                lambda: capture.calls.count("set_confidence_and_reset") == 1
            )

            second = threading.Thread(
                target=lambda: client_b.patch(
                    "/settings/detection-confidence", json={"confidence": 0.70}
                )
            )
            second.start()
            try:
                # Give B time to try. It must block on the mutation lock — it must
                # NOT reach its own capture command while A still holds it.
                time.sleep(0.2)
                assert capture.calls.count("set_confidence_and_reset") == 1

                gate.set()  # release A; it finishes and frees the lock
                second.join(timeout=2.0)
                assert not second.is_alive()  # B completed once the lock was free
            finally:
                second.join(timeout=2.0)
        finally:
            gate.set()
            first.join(timeout=2.0)

        # B ran its command only after A released the lock — two total, never overlapping.
        assert capture.calls.count("set_confidence_and_reset") == 2


class TestConfidencePatch:
    """T11/B5, backend tests 10 & 11: `PATCH /settings/detection-confidence`. Out
    of range → 422; forbidden while recording → 409; a changed value is applied
    as one serialised capture command then readiness clears; a no-op value returns
    without resetting; a failed reconfiguration preserves the last confirmed
    value; the TOML on disk is never written."""

    def _client(self, capture, session, default_confidence=0.5, clock=None):
        clock = clock if clock is not None else _Clock(0.0)
        return TestClient(
            create_app(
                capture, session, "test-model", default_confidence=default_confidence, clock=clock
            )
        )

    @pytest.mark.parametrize("value", [0.2, 1.0])
    def test_out_of_range_confidence_is_422(self, value: float) -> None:
        capture = FakeCapture()
        client = self._client(capture, Session())

        resp = client.patch("/settings/detection-confidence", json={"confidence": value})

        assert resp.status_code == 422
        assert capture.calls == []  # never reached the capture command

    def test_confidence_change_while_recording_is_409(self) -> None:
        capture = FakeCapture()
        session = _ready_session()
        client = self._client(capture, session)
        client.post("/recording/start")  # → RECORDING

        resp = client.patch("/settings/detection-confidence", json={"confidence": 0.7})

        assert resp.status_code == 409
        assert "set_confidence_and_reset" not in capture.calls

    def test_changed_value_applies_one_command_updates_control_and_clears_readiness(
        self,
    ) -> None:
        capture = FakeCapture()
        session = Session(setup_stable_s=0.0)
        _prime_ready(session)
        clock = _Clock(0.0)
        client = self._client(capture, session, clock=clock)
        assert client.get("/status").json()["setup"]["ready"] is True  # ready before

        clock.set(1.0)  # keep prepare(clock()) monotonic after the t=0.0 observe
        resp = client.patch("/settings/detection-confidence", json={"confidence": 0.7})

        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence"] == pytest.approx(0.7)
        assert body["default_confidence"] == pytest.approx(0.5)
        assert capture.set_confidence_values == [0.7]
        assert capture.calls.count("set_confidence_and_reset") == 1
        # A changed confidence restarts enrolment: readiness clears.
        setup = client.get("/status").json()["setup"]
        assert setup["ready"] is False
        assert setup["blocking_reason"] == "recognising"

    def test_a_noop_value_returns_without_resetting(self) -> None:
        capture = FakeCapture()
        client = self._client(capture, Session(), default_confidence=0.5)

        resp = client.patch("/settings/detection-confidence", json={"confidence": 0.5})

        assert resp.status_code == 200
        assert resp.json()["confidence"] == pytest.approx(0.5)
        assert capture.calls == []  # no reset, no confidence command
        assert capture.set_confidence_values == []

    def test_a_failed_reconfiguration_preserves_the_last_confirmed_value(self) -> None:
        capture = FakeCapture()
        capture.fail_confidence_with(TrackerResetError("reset failed"))
        client = self._client(capture, Session(setup_stable_s=0.0), default_confidence=0.5)

        resp = client.patch("/settings/detection-confidence", json={"confidence": 0.7})

        assert resp.status_code == 503
        assert resp.json()["detail"] == "confidence change failed"
        control = client.get("/status").json()["detector_control"]
        assert control["confidence"] == pytest.approx(0.5)  # unchanged

    def test_a_confidence_timeout_is_503_and_preserves_the_value(self) -> None:
        capture = FakeCapture()
        capture.fail_confidence_with(TimeoutError("stalled"))
        client = self._client(capture, Session(setup_stable_s=0.0), default_confidence=0.5)

        resp = client.patch("/settings/detection-confidence", json={"confidence": 0.7})

        assert resp.status_code == 503
        assert resp.json()["detail"] == "capture stalled"
        control = client.get("/status").json()["detector_control"]
        assert control["confidence"] == pytest.approx(0.5)

    def test_the_toml_on_disk_is_never_written(self) -> None:
        from backend.mvp_settings import DEFAULT_MVP_CONFIG_PATH

        before = DEFAULT_MVP_CONFIG_PATH.read_bytes()
        capture = FakeCapture()
        client = self._client(capture, Session(setup_stable_s=0.0), default_confidence=0.5)

        client.patch("/settings/detection-confidence", json={"confidence": 0.7})

        assert DEFAULT_MVP_CONFIG_PATH.read_bytes() == before
