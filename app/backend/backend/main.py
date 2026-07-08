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

_STREAM_BOUNDARY = "frame"
_STREAM_POLL_S = 0.02
_VITE_DEV_ORIGIN = "http://localhost:5173"


class SetupStatus(BaseModel):
    detected_count: int
    stable_for_s: float


class InstrumentStatusModel(BaseModel):
    tracker_id: int
    label: str
    on_table: bool
    off_since_s: float | None
    pickup_count: int


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
    """Yield one multipart JPEG part per new `Latest`, paced by
    `capture.generation` — no duplicate parts, no busy loop (AC5)."""
    last_generation = -1
    while True:
        generation = capture.generation
        if generation != last_generation:
            latest = capture.snapshot()
            if latest is not None:
                last_generation = generation
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

    capture._on_frame = _on_frame

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

    @app.get("/status", response_model=StatusResponse)
    def get_status() -> StatusResponse:
        with lock:
            t = clock()
            phase = session.phase
            capture_health = "ok" if capture.health == "ok" else "stalled"

            setup = None
            if phase in (Phase.SETUP, Phase.FINISHED):
                detected_count, stable_for_s = session.setup_status(t)
                setup = SetupStatus(detected_count=detected_count, stable_for_s=stable_for_s)

            recording = None
            if phase is Phase.RECORDING:
                elapsed_s, statuses = session.recording_status(t)
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
        capture = CaptureLoop(tracker, 0, cap_factory=lambda _idx: FakeCaptureSource(fps=10.0))
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
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
