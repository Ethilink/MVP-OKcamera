"""T04: FastAPI app assembly, `--fake`/`--camera` CLI, and the `orc-demo`
console-script entrypoint (see pyproject.toml `[project.scripts]`).

Wires `CaptureLoop` + `Session` behind the frozen `api-contract.md` HTTP
surface. See `app/docs/tasks/T04-api.md` for the frozen signatures + ACs.
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from datetime import datetime
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.capture import CaptureLoop
from backend.fakes import FakeCaptureSource, ScenarioTracker
from backend.session import InvalidPhase, Phase, Session
from backend.thumbnails import build_detections

_STREAM_BOUNDARY = "frame"
_STREAM_POLL_S = 0.02
_VITE_DEV_ORIGIN = "http://localhost:5173"


class DetectionModel(BaseModel):
    tracker_id: int
    label: str
    thumbnail: str | None


class SetupStatus(BaseModel):
    detected_count: int
    stable_for_s: float
    detections: list[DetectionModel]


class InstrumentStatusModel(BaseModel):
    tracker_id: int
    label: str
    on_table: bool
    off_since_s: float | None
    pickup_count: int
    # A live crop of the instrument from the current frame, when it is visible
    # this poll (letterboxed data-URI, same crop path as the setup thumbnails).
    # `None` whenever the instrument is off the table / not detected this frame —
    # the app keeps showing its last-seen crop, so a missing instrument's tile
    # never blanks out.
    thumbnail: str | None


class RecordingStatus(BaseModel):
    started_at: str
    elapsed_s: float
    on_table_count: int
    instruments: list[InstrumentStatusModel]


class StatusResponse(BaseModel):
    phase: Literal["setup", "recording", "finished"]
    capture_health: Literal["ok", "stalled"]
    model_version: str
    setup: SetupStatus | None
    recording: RecordingStatus | None


class StartResponse(BaseModel):
    started_at: str


class UsageWindowModel(BaseModel):
    off_s: float
    on_s: float | None


class InstrumentReportModel(BaseModel):
    tracker_id: int
    label: str
    completeness: Literal["present", "missing"]
    usage: list[UsageWindowModel]


class ReportResponse(BaseModel):
    started_at: str
    stopped_at: str
    duration_s: float
    model_version: str
    instruments: list[InstrumentReportModel]


async def _mjpeg_stream(capture: CaptureLoop):
    """Yield one multipart JPEG part per new `Latest`, paced by SNAPSHOT
    IDENTITY — T03 AC4 guarantees each publish rebinds a NEW `Latest` object,
    so `latest is not last_sent` advances exactly once per published frame:
    no duplicate parts, no busy loop (AC5)."""
    last_sent = None
    while True:
        latest = capture.snapshot()
        if latest is not None and latest is not last_sent:
            last_sent = latest
            jpeg = latest.overlay_jpeg
            yield (
                b"--" + _STREAM_BOUNDARY.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
        await asyncio.sleep(_STREAM_POLL_S)


def create_app(
    capture: CaptureLoop,
    session: Session,
    model_version: str,
    clock=time.monotonic,
    now=lambda: datetime.now().astimezone(),
) -> FastAPI:
    """Wire `capture` + `session` behind the frozen `api-contract.md` HTTP
    surface. Owns the session lock: `capture.on_frame` is registered here to
    call `session.observe` under it, and every handler that reads/mutates the
    session takes the same lock (DESIGN §Threading)."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[_VITE_DEV_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    lock = threading.Lock()
    timestamps: dict[str, str | None] = {"started_at": None, "stopped_at": None}

    def _on_frame(t: float, present_ids: frozenset[int]) -> None:
        with lock:
            session.observe(t, present_ids)

    capture.set_on_frame(_on_frame)

    def _report_response(report) -> ReportResponse:
        return ReportResponse(
            started_at=timestamps["started_at"],
            stopped_at=timestamps["stopped_at"],
            duration_s=report.duration_s,
            model_version=model_version,
            instruments=[
                InstrumentReportModel(
                    tracker_id=instrument.tracker_id,
                    label=instrument.label,
                    completeness=instrument.completeness,
                    usage=[UsageWindowModel(off_s=w.off_s, on_s=w.on_s) for w in instrument.usage],
                )
                for instrument in report.instruments
            ],
        )

    def _snapshot_detections() -> list:
        """Crop the current snapshot's boxes into per-detection thumbnails. Runs
        OUTSIDE the session lock — the capture thread takes that lock on every
        frame, so no image work may block it. `snapshot()` is immutable/thread-
        safe, and `[]` is the honest answer before the first frame is published.
        Shared by the setup preview and the live recording crops."""
        snapshot = capture.snapshot()
        if snapshot is None:
            return []
        return build_detections(snapshot.frame_bgr, snapshot.detections)

    def _setup_detections() -> list[DetectionModel]:
        # Detection (a plain dataclass) → DetectionModel by matching field names,
        # so adding a field can't leave this mapping silently behind.
        return [
            DetectionModel.model_validate(detection, from_attributes=True)
            for detection in _snapshot_detections()
        ]

    @app.get("/status", response_model=StatusResponse)
    def get_status() -> StatusResponse:
        # Read all session/capture state under the lock, but do NOTHING heavy
        # here — the capture thread holds this same lock every frame. Thumbnail
        # cropping happens after the lock is released (see below).
        recording_raw: tuple[float, list] | None = None
        with lock:
            t = clock()
            phase = session.phase
            capture_health = "ok" if capture.health == "ok" else "stalled"

            setup_meta: tuple[int, float] | None = None
            if phase in (Phase.SETUP, Phase.FINISHED):
                setup_meta = session.setup_status(t)

            # Just read the session state under the lock; the image work (crops)
            # happens below, after the lock is released.
            if phase is Phase.RECORDING:
                recording_raw = session.recording_status(t)

        setup = None
        if setup_meta is not None:
            detected_count, stable_for_s = setup_meta
            setup = SetupStatus(
                detected_count=detected_count,
                stable_for_s=stable_for_s,
                detections=_setup_detections(),
            )

        recording = None
        if recording_raw is not None:
            elapsed_s, statuses = recording_raw
            # Live crops for the instruments visible THIS frame, matched by
            # tracker_id. Off-table instruments simply aren't in the frame, so
            # they get `thumbnail=None` and the app falls back to their last-seen
            # crop. Cropped outside the lock (see `_snapshot_detections`).
            crops = {det.tracker_id: det.thumbnail for det in _snapshot_detections()}
            recording = RecordingStatus(
                started_at=timestamps["started_at"],
                elapsed_s=elapsed_s,
                on_table_count=sum(1 for status in statuses if status.on_table),
                instruments=[
                    InstrumentStatusModel(
                        tracker_id=status.tracker_id,
                        label=status.label,
                        on_table=status.on_table,
                        off_since_s=status.off_since_s,
                        pickup_count=status.pickup_count,
                        thumbnail=crops.get(status.tracker_id),
                    )
                    for status in statuses
                ],
            )

        return StatusResponse(
            phase=phase,
            capture_health=capture_health,
            model_version=model_version,
            setup=setup,
            recording=recording,
        )

    @app.get("/stream")
    async def get_stream() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_stream(capture),
            media_type=f"multipart/x-mixed-replace; boundary={_STREAM_BOUNDARY}",
        )

    @app.post("/recording/start", response_model=StartResponse)
    def post_recording_start() -> StartResponse:
        # reset_tracker() blocks waiting for the capture thread, which itself
        # takes `lock` (via `_on_frame`) to publish each frame — holding `lock`
        # across the wait would deadlock the two threads against each other.
        # So: check eligibility, reset OUTSIDE the lock, then re-take it for
        # the (fast) session mutation, sampling clock() right before it.
        with lock:
            if session.phase not in (Phase.SETUP, Phase.FINISHED):
                raise HTTPException(status_code=409, detail=f"cannot start from {session.phase}")
        try:
            capture.reset_tracker()
        except TimeoutError:
            raise HTTPException(status_code=503, detail="capture stalled")
        with lock:
            if session.phase not in (Phase.SETUP, Phase.FINISHED):
                raise HTTPException(status_code=409, detail=f"cannot start from {session.phase}")
            session.start(clock())
            timestamps["started_at"] = now().isoformat()
            timestamps["stopped_at"] = None
            return StartResponse(started_at=timestamps["started_at"])

    @app.post("/recording/stop", response_model=ReportResponse)
    def post_recording_stop() -> ReportResponse:
        with lock:
            try:
                report = session.stop(clock())
            except InvalidPhase as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            timestamps["stopped_at"] = now().isoformat()
            return _report_response(report)

    @app.get("/report", response_model=ReportResponse)
    def get_report() -> ReportResponse:
        with lock:
            try:
                report = session.report()
            except InvalidPhase as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            return _report_response(report)

    return app


def main(argv: list[str] | None = None) -> None:
    """CLI: `orc-demo --fake` | `orc-demo --camera N --weights PATH`."""
    parser = argparse.ArgumentParser(prog="orc-demo")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--weights", type=str)
    args = parser.parse_args(argv)

    session = Session()

    if args.fake:
        tracker = ScenarioTracker(fps=10)
        # Share ONE ScenarioState so the drawn frames and the detections stay in
        # lockstep (and re-align on Start, which resets the shared clock).
        capture = CaptureLoop(
            tracker,
            0,
            cap_factory=lambda _idx: FakeCaptureSource(fps=10.0, scenario=tracker.state),
        )
    else:
        if args.camera is None or args.weights is None:
            parser.error("--camera and --weights are required without --fake")
        try:
            from orc_model.pipelines.tracking import load_tracker
        except ImportError as exc:
            raise SystemExit(
                f"real camera mode requires orc_model.pipelines.tracking.load_tracker: {exc}"
            )
        tracker = load_tracker(args.weights)
        capture = CaptureLoop(tracker, args.camera)

    app = create_app(capture, session, tracker.model_version)
    capture.start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        # Always release the camera on exit (Ctrl-C, error, or normal return);
        # without this the capture thread + device leak past process teardown.
        capture.stop()


if __name__ == "__main__":
    main()
