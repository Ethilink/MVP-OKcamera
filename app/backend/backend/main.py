"""FastAPI app assembly, ``--fake``/``--camera`` CLI, and ``orc-demo`` entrypoint.

Wires ``CaptureLoop`` and ``Session`` behind ``app/docs/api-contract.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time
from datetime import datetime
from typing import Callable, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.capture import CaptureLoop, TrackerResetError
from backend.fakes import FakeCaptureSource, ScenarioTracker
from backend.mvp_settings import DEFAULT_MVP_CONFIG_PATH, load_mvp_settings
from backend.render import OverlayRenderer, catalog_colour
from backend.session import InvalidPhase, Phase, Session
from backend.thumbnails import build_detections

_STREAM_BOUNDARY = "frame"
_STREAM_POLL_S = 0.02
_VITE_DEV_ORIGIN = "http://localhost:5173"

# Runtime detection-confidence bounds (T11/B5). The PATCH rejects values outside
# [CONF_MIN, CONF_MAX] with 422 (Pydantic Field); CONF_STEP is advisory metadata
# the frontend slider snaps to. These gate the RUNTIME value only — the TOML on
# disk is never rewritten.
CONF_MIN = 0.30
CONF_MAX = 0.90
CONF_STEP = 0.05


class DetectionModel(BaseModel):
    tracker_id: int
    # The detection's identity state (T11/B6). `recognised` iff its id is in the
    # same snapshot's roster; else `recognising` iff still resolving; else
    # `unknown`. The frontend renders the tile from this, not from the raw id.
    state: Literal["recognising", "recognised", "unknown"]
    # `Instrument N` when recognised, `Unknown` when settled non-roster, `""`
    # while recognising (spinner, no name) — never a raw id (D4).
    label: str
    # Fixed catalog colour when recognised, gray otherwise — the same hex the
    # overlay draws this detection's mask with (R1/R3).
    colour: str
    thumbnail: str | None
    # Experimental (feat/matching-tests) — a testing aid for tuning the matcher
    # live, NOT part of the frozen api-contract. All `None` in fake mode (no
    # real detector/matcher runs there). See thumbnails.Detection.
    detector_confidence: float | None = None
    matcher_score: float | None = None
    matcher_tau: float | None = None
    matcher_closest_id: int | None = None
    matcher_accepted: bool | None = None


class SetupStatus(BaseModel):
    # Readiness scalars from Session.setup_readiness (T11/B2); `ready` also folds
    # in capture health at the API. `detections` carries the per-item identity
    # shape (state/label/colour, T11/B6) — see DetectionModel.
    detected_count: int
    expected_count: int
    recognised_count: int
    resolving_count: int
    unknown_count: int
    stable_for_s: float
    ready: bool
    blocking_reason: str | None
    detections: list[DetectionModel]


class DetectorControlModel(BaseModel):
    confidence: float
    default_confidence: float
    minimum: float
    maximum: float
    step: float


class ConfidencePatch(BaseModel):
    confidence: float = Field(ge=CONF_MIN, le=CONF_MAX)  # out-of-range -> 422


class InstrumentStatusModel(BaseModel):
    tracker_id: int
    label: str
    on_table: bool
    # A live transparent cutout from the current frame, when it is visible this
    # poll (data-URI, same preview path as the setup thumbnails).
    # `None` whenever the instrument is off the table / not detected this frame —
    # the app keeps showing its last-seen crop, so a missing instrument's tile
    # never blanks out.
    thumbnail: str | None
    # The instrument's fixed mask colour (hex) — same value the overlay draws,
    # so the panel swatch and the video can never drift (T10/D8a).
    colour: str
    # Experimental (feat/matching-tests) — testing aid, not part of the frozen
    # api-contract. `detector_confidence` is null when off-table this frame
    # (no live detection to report); the matcher fields persist across frames —
    # the last score/tau/closest-candidate/accepted from whichever event
    # produced it (bind at Start, or a later re-id decision), since most
    # instruments never leave the table and would otherwise show nothing.
    detector_confidence: float | None = None
    matcher_score: float | None = None
    matcher_tau: float | None = None
    matcher_closest_id: int | None = None
    matcher_accepted: bool | None = None


class RecordingStatus(BaseModel):
    started_at: str
    elapsed_s: float
    instruments: list[InstrumentStatusModel]


class StatusResponse(BaseModel):
    phase: Literal["setup", "recording", "finished"]
    capture_health: Literal["ok", "stalled"]
    model_version: str
    setup: SetupStatus | None
    recording: RecordingStatus | None
    detector_control: DetectorControlModel


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


def _start_block_detail(readiness, capture_health) -> str | None:
    """Map a readiness+health verdict to the 409 detail that blocks Start, or
    None when everything is ready. Pure, so it is shared by the Start gate and
    trivially testable. Capture health outranks every readiness reason (D3)."""
    if capture_health != "ok":
        return "capture stalled"
    reason = readiness.blocking_reason
    if reason is None:
        return None
    if reason == "recognising":
        return "recognition still in progress"
    if reason == "unknown_objects":
        return "remove unknown objects before starting"
    if reason == "missing_instruments":
        return (
            f"all {readiness.expected_count} instruments must be recognised "
            "before starting"
        )
    if reason == "hold_steady":
        return "hold the tray steady before starting"
    return "setup is not ready"


def create_app(
    capture: CaptureLoop,
    session: Session,
    model_version: str,
    clock=time.monotonic,
    now=lambda: datetime.now().astimezone(),
    default_confidence: float = 0.5,
    on_recording_start: Callable[[], None] | None = None,
    on_recording_stop: Callable[[], None] | None = None,
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
    # Coarse mutation lock (T11): held for the WHOLE of start / confidence so a
    # confidence reset cannot land between Start's readiness check and its phase
    # transition. `_on_frame` does NOT take it, so confidence may hold it across
    # the blocking capture command while the capture thread keeps publishing.
    # The fine session `lock` is only ever held in short critical sections, and
    # NEVER while a blocking capture command runs (that would deadlock the
    # capture thread, which needs `lock` in `_on_frame`).
    mutation_lock = threading.Lock()
    timestamps: dict[str, str | None] = {"started_at": None, "stopped_at": None}
    # Runtime confidence view (B5). `confidence` is the last CONFIRMED value
    # (updated only after set_confidence_and_reset succeeds); `default` is the
    # config baseline shown to the operator. The TOML on disk is never written.
    detector_state = {"confidence": default_confidence, "default": default_confidence}

    def _on_frame(
        t: float,
        present_ids: frozenset[int],
        roster: frozenset[int],
        catalog: frozenset[int],
        resolving_ids: frozenset[int],
    ) -> None:
        with lock:
            session.observe(t, present_ids, roster, catalog, resolving_ids)

    capture.set_on_frame(_on_frame)

    # The overlay renderer is CaptureLoop's render_fn. Since T11/R1 it is stateless
    # and roster-aware in every phase, so create_app just wires it in once — the
    # start/stop handlers no longer signal it (identity is phase-independent).
    renderer = OverlayRenderer()
    capture.set_render_fn(renderer)

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
        """Build per-detection cutouts from the current snapshot. Runs
        OUTSIDE the session lock — the capture thread takes that lock on every
        frame, so no image work may block it. `snapshot()` is immutable/thread-
        safe, and `[]` is the honest answer before the first frame is published.
        Shared by the setup preview and the live recording crops."""
        snapshot = capture.snapshot()
        if snapshot is None:
            return []
        return build_detections(
            snapshot.frame_bgr,
            snapshot.detections,
            snapshot.roster,
            snapshot.catalog,
            match_debug=snapshot.match_debug,
        )

    def _setup_detections() -> list[DetectionModel]:
        # Detection (a plain dataclass) → DetectionModel by matching field names,
        # so adding a field can't leave this mapping silently behind.
        return [
            DetectionModel.model_validate(detection, from_attributes=True)
            for detection in _snapshot_detections()
        ]

    def _detector_control() -> DetectorControlModel:
        return DetectorControlModel(
            confidence=detector_state["confidence"],
            default_confidence=detector_state["default"],
            minimum=CONF_MIN,
            maximum=CONF_MAX,
            step=CONF_STEP,
        )

    @app.get("/status", response_model=StatusResponse)
    def get_status() -> StatusResponse:
        # Read all session/capture state under the lock, but do NOTHING heavy
        # here — the capture thread holds this same lock every frame. Thumbnail
        # cropping happens after the lock is released (see below).
        recording_raw: tuple[float, list] | None = None
        readiness = None
        with lock:
            t = clock()
            phase = session.phase
            capture_health = "ok" if capture.health == "ok" else "stalled"

            # setup_readiness is cheap + pure (no tracker, no clock read of its
            # own), so it is safe to compute under the capture-shared lock.
            if phase in (Phase.SETUP, Phase.FINISHED):
                readiness = session.setup_readiness(t)

            # Just read the session state under the lock; the image work (crops)
            # happens below, after the lock is released.
            if phase is Phase.RECORDING:
                recording_raw = session.recording_status(t)

        setup = None
        if readiness is not None:
            setup = SetupStatus(
                detected_count=readiness.detected_count,
                expected_count=readiness.expected_count,
                recognised_count=readiness.recognised_count,
                resolving_count=readiness.resolving_count,
                unknown_count=readiness.unknown_count,
                stable_for_s=readiness.stable_for_s,
                # D3 cond 1: fold capture health into `ready` here. The pure
                # `blocking_reason` stays as-is; the frontend shows the
                # camera-stalled message from `capture_health`, which it reads
                # separately.
                ready=readiness.ready and capture_health == "ok",
                blocking_reason=readiness.blocking_reason,
                detections=_setup_detections(),
            )

        recording = None
        if recording_raw is not None:
            elapsed_s, statuses = recording_raw
            # Live crops for the instruments visible THIS frame, matched by
            # tracker_id. Off-table instruments simply aren't in the frame, so
            # they get `thumbnail=None` and the app falls back to their last-seen
            # crop. Cropped outside the lock (see `_snapshot_detections`).
            detections_by_id = {det.tracker_id: det for det in _snapshot_detections()}
            # The fixed catalog the overlay colours with right now, so a panel
            # swatch and its mask are the same hex by construction (R2/D5 — the
            # colour is a pure function of the CATALOG index, not the current
            # roster, so a partial roster can't shift it). Before the first frame
            # there is no snapshot yet and every colour resolves to the gray —
            # transient and harmless (B-A1).
            snapshot = capture.snapshot()
            catalog = snapshot.catalog if snapshot is not None else frozenset()
            # Experimental (feat/matching-tests): the matcher's last score per
            # id, sampled the same tick as the snapshot above. Looked up
            # DIRECTLY here (not via `detections_by_id`) because it must survive
            # for an off-table instrument that isn't in this frame's detections —
            # most instruments never leave the table, so this is the only way
            # their matcher score ever shows up at all.
            match_debug = snapshot.match_debug if snapshot is not None else {}
            recording = RecordingStatus(
                started_at=timestamps["started_at"],
                elapsed_s=elapsed_s,
                instruments=[
                    InstrumentStatusModel(
                        tracker_id=status.tracker_id,
                        label=status.label,
                        on_table=status.on_table,
                        thumbnail=(detections_by_id[status.tracker_id].thumbnail
                                   if status.tracker_id in detections_by_id else None),
                        colour=catalog_colour(catalog, status.tracker_id),
                        detector_confidence=(
                            detections_by_id[status.tracker_id].detector_confidence
                            if status.tracker_id in detections_by_id else None
                        ),
                        matcher_score=(
                            match_debug[status.tracker_id].score
                            if status.tracker_id in match_debug else None
                        ),
                        matcher_tau=(
                            match_debug[status.tracker_id].tau
                            if status.tracker_id in match_debug else None
                        ),
                        matcher_closest_id=(
                            match_debug[status.tracker_id].closest_id
                            if status.tracker_id in match_debug else None
                        ),
                        matcher_accepted=(
                            match_debug[status.tracker_id].accepted
                            if status.tracker_id in match_debug else None
                        ),
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
            detector_control=_detector_control(),
        )

    @app.get("/stream")
    async def get_stream() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_stream(capture),
            media_type=f"multipart/x-mixed-replace; boundary={_STREAM_BOUNDARY}",
        )

    @app.post("/recording/start", response_model=StartResponse)
    def post_recording_start() -> StartResponse:
        # Start is a pure, fail-closed phase transition: the exact roster/catalog
        # state approved here IS the state recording continues with. Resetting the
        # tracker after this check would empty the roster and silently perform a
        # second enrolment during recording. mutation_lock prevents a confidence
        # reset from landing between the check and session.start().
        with mutation_lock:
            with lock:
                t = clock()
                if session.phase not in (Phase.SETUP, Phase.FINISHED):
                    raise HTTPException(
                        status_code=409,
                        detail=f"cannot start from {session.phase}",
                    )
                capture_health = "ok" if capture.health == "ok" else "stalled"
                readiness = session.setup_readiness(t)
                detail = _start_block_detail(readiness, capture_health)
                if detail is not None:
                    raise HTTPException(status_code=409, detail=detail)
                # Optional lifecycle signal for injected simulations. Production
                # leaves it unset; `--fake` only anchors its pickup timeline.
                # This does not reset or otherwise mutate tracker identity state.
                if on_recording_start is not None:
                    on_recording_start()
                session.start(t)
                timestamps["started_at"] = now().isoformat()
                timestamps["stopped_at"] = None
                return StartResponse(started_at=timestamps["started_at"])

    @app.post("/recording/stop", response_model=ReportResponse)
    def post_recording_stop() -> ReportResponse:
        with lock:
            try:
                report = session.stop(clock())
            except InvalidPhase as exc:
                # A wrong-phase stop changes nothing, the overlay included.
                raise HTTPException(status_code=409, detail=str(exc))
            if on_recording_stop is not None:
                on_recording_stop()
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

    @app.patch("/settings/detection-confidence", response_model=DetectorControlModel)
    def patch_detection_confidence(body: ConfidencePatch) -> DetectorControlModel:
        # ConfidencePatch's Field(ge=CONF_MIN, le=CONF_MAX) already 422s an
        # out-of-range value before we get here. Serialise the whole op on
        # mutation_lock so it can't interleave with Start (T11/B5).
        value = body.confidence
        with mutation_lock:
            with lock:
                if session.phase is Phase.RECORDING:
                    raise HTTPException(
                        status_code=409,
                        detail="cannot change confidence during recording",
                    )
                current = detector_state["confidence"]
            if value == current:
                return _detector_control()   # no-op: no reset, immediate return
            # Apply the change as ONE serialised capture command (set confidence
            # + reset together, on the capture thread). Blocks outside `lock` so
            # the capture thread can keep publishing; on failure the last
            # confirmed value is preserved (503, detector_state untouched).
            try:
                capture.set_confidence_and_reset(value)
            except TimeoutError:
                raise HTTPException(status_code=503, detail="capture stalled")
            except TrackerResetError:
                raise HTTPException(status_code=503, detail="confidence change failed")
            # Source of truth advances only after the capture command succeeds.
            detector_state["confidence"] = value
            with lock:
                # A changed confidence restarts enrolment and clears readiness.
                session.prepare(clock())
        return _detector_control()

    return app


def main(argv: list[str] | None = None) -> None:
    """CLI: `orc-demo --fake` | `orc-demo --camera N --weights PATH`."""
    parser = argparse.ArgumentParser(prog="orc-demo")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--weights", type=str)
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_MVP_CONFIG_PATH),
        help="MVP TOML configuration (default: app/mvp.toml)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print a readable, event-level pipeline narrative to the console "
        "(enrolment freeze, each link/unknown/deferral, track deaths). "
        "Off by default; also enabled by ORC_DEBUG=1. Behaviour-free.",
    )
    args = parser.parse_args(argv)

    # Configure the debug console BEFORE load_tracker() so its startup logs
    # (galleries loaded, binding-disabled) are captured too. ORC_DEBUG must be an
    # explicit true value -- ORC_DEBUG=0 / "" / "false" leave it OFF (a bare
    # truthiness check would enable debug for any non-empty value).
    from backend.debug import configure_debug_logging, env_flag_enabled

    if args.debug or env_flag_enabled(os.environ.get("ORC_DEBUG")):
        configure_debug_logging()

    try:
        settings = load_mvp_settings(args.config)
    except ValueError as exc:
        parser.error(str(exc))
    session = Session(
        off_debounce_s=settings.session.off_debounce_s,
        on_debounce_s=settings.session.on_debounce_s,
    )

    if args.fake:
        tracker = ScenarioTracker(fps=settings.capture.fake_fps)
        # Share ONE ScenarioState so the drawn frames and the detections stay in
        # lockstep. Its scripted recording lifecycle is separate from reset(),
        # because recording Start preserves the approved tracker roster.
        capture = CaptureLoop(
            tracker,
            0,
            cap_factory=lambda _idx: FakeCaptureSource(
                fps=settings.capture.fake_fps,
                scenario=tracker.state,
            ),
            frame_size=settings.capture.frame_size,
            stale_after_s=settings.capture.stale_after_s,
        )
        on_recording_start = tracker.begin_recording
        on_recording_stop = tracker.end_recording
    else:
        if args.camera is None or args.weights is None:
            parser.error("--camera and --weights are required without --fake")
        try:
            from orc_model.pipelines.tracking import load_tracker
        except ImportError as exc:
            raise SystemExit(
                f"real camera mode requires orc_model.pipelines.tracking.load_tracker: {exc}"
            )
        tracker = load_tracker(args.weights, config=settings.tracker)
        capture = CaptureLoop(
            tracker,
            args.camera,
            frame_size=settings.capture.frame_size,
            stale_after_s=settings.capture.stale_after_s,
        )
        on_recording_start = None
        on_recording_stop = None

    app = create_app(
        capture,
        session,
        tracker.model_version,
        default_confidence=settings.tracker.detector.confidence,
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
    )
    capture.start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        # Always release the camera on exit (Ctrl-C, error, or normal return);
        # without this the capture thread + device leak past process teardown.
        capture.stop()


if __name__ == "__main__":
    main()
