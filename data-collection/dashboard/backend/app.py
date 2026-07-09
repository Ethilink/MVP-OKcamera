"""FastAPI layer (T05) — wires the capture loop, dataset writer, and renderer
into the six endpoints of the spec's Runtime table.

The threading rules that bite (spec §Threading rules) are load-bearing here:

- ``/flag`` and ``/validate`` are plain ``def`` (not ``async``). Their blocking
  work (``cv2.imwrite`` + RLE encode; the in-process validator's disk walk) runs
  on Starlette's threadpool, off the event loop, so a SPACE press never freezes
  ``/stream``.
- ``/flag`` grabs the ``Latest`` tuple **reference** atomically via
  ``capture.snapshot()`` (the loop rebinds it under its own lock, never mutates in
  place), then does its blocking work on that stable snapshot.
- ``/flag`` and the dataset-mutating half of ``/settings`` serialize on a single
  ``dataset_lock`` so rapid concurrent flags get sequential ``image_id``/``ann_id``
  and never a torn ``annotations.json``.
- Provenance uses the threshold captured **into the snapshot at predict time**
  (``snap.threshold``), never the live slider value at flag time.
- ``/stream`` paces on ``capture.generation`` so each overlay is sent exactly once.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# How often the MJPEG generator re-checks the generation counter. Small enough
# that a new overlay is streamed promptly; an ``await`` sleep so the wait yields
# the event loop instead of busy-spinning it.
_STREAM_POLL_S = 0.02

_STREAM_BOUNDARY = "frame"


class ConfidenceIn(BaseModel):
    """`/confidence` body — Pydantic rejects out-of-range with 422 (spec §AC5)."""

    value: float = Field(ge=0.0, le=1.0)


class SettingsIn(BaseModel):
    """`/settings` body. ``camera_index`` optional; the rest set the capture target."""

    camera_index: int | None = None
    output_path: str
    dataset_name: str


class FlagIn(BaseModel):
    """`/flag` body. ``generation`` pins the exact displayed frame to capture; when
    omitted (older clients / no-body POST) ``/flag`` falls back to the newest frame."""

    generation: int | None = None


class RecordStartIn(BaseModel):
    """`/record/start` body (TR5). ``entry_base`` is auto-suffixed with a
    zero-padded counter to mint the resolved ``entry_name`` (U1)."""

    entry_base: str


class KeyframeIn(BaseModel):
    """`/keyframe` body (TR5)."""

    frame_number: int


class RecordingController:
    """Recording-mode state (TR5).

    A plain data holder for the ``idle → recording → processing → idle`` machine
    (``processing → failed → processing`` on retry). Every field is read/written
    only under ``app.state.recording_lock`` — the endpoints own the transitions
    and the worker's completion handler; this class carries no logic so there is
    exactly one place (the lock-guarded closures in ``create_app``) that mutates
    the machine.
    """

    def __init__(self) -> None:
        self.state = "idle"
        self.entry_name: str | None = None
        self.entry_dir: Path | None = None
        self.operator_threshold: float | None = None
        self.keyframes: set[int] = set()
        self.frame_count = 0
        self.error: str | None = None
        self.job = None
        # The threading.Thread running the current job's post-pass. Kept so
        # /record/discard can cancel the job and JOIN the worker before it
        # deletes the entry folder (otherwise an in-flight keyframe write
        # could re-create a stray folder after the rmtree).
        self.worker: threading.Thread | None = None


async def mjpeg_stream(capture):
    """Yield one multipart JPEG part per *new* ``Latest`` (spec §AC6).

    Paces on ``capture.generation``: when the counter has not advanced we sleep
    and re-check rather than re-emit the last frame, so a paused capture produces
    no duplicate parts and a freshly published ``Latest`` is streamed promptly.
    """
    last_gen = -1
    while True:
        gen = capture.generation
        if gen != last_gen:
            snap = capture.snapshot()
            if snap is not None:
                last_gen = gen
                jpeg = snap.overlay_jpeg
                yield (
                    b"--" + _STREAM_BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
        await asyncio.sleep(_STREAM_POLL_S)


def _default_validate(dataset_dir: Path):
    """Resolve and call the in-process import validator (spec §AC8, no subprocess)."""
    from scripts.validate_import import validate

    return validate(Path(dataset_dir))


def create_app(detector, writer_factory, capture, validate_fn=_default_validate) -> FastAPI:
    """Build the dashboard app around its injected dependencies.

    ``detector`` exposes a live ``confidence_threshold`` and ``predict``;
    ``writer_factory(output_path, dataset_name) -> DatasetWriter`` mints a fresh
    writer per dataset (raising ``ValueError`` on a bad name, ``FileExistsError``
    on a collision); ``capture`` is the ``CaptureLoop``. ``validate_fn`` is
    injectable so tests can stub the validator.
    """
    app = FastAPI(title="ORC data-collection dashboard")
    app.state.detector = detector
    app.state.writer_factory = writer_factory
    app.state.capture = capture
    app.state.validate_fn = validate_fn

    # The active dataset writer (None until /settings configures one) and the
    # lock that serializes every mutation of it (flags + settings swaps).
    app.state.writer = None
    app.state.dataset_lock = threading.Lock()

    # --- Recording mode (TR5) -------------------------------------------------
    # output_path is captured on the /settings success below (recording writes
    # entries under it, same root as image-mode datasets). capture_fps /
    # mining_threshold / model_version are threaded in from main.py's CLI args;
    # defaults here keep tests that don't set them working. The encoder opener,
    # post-pass job, and probe are imported but injectable via app.state so tests
    # run with fakes (no real ffmpeg / video / detector).
    from backend.encoder import open_encoder, probe_video
    from backend.postpass import PostPassJob

    app.state.output_path = None
    app.state.capture_fps = 30.0
    app.state.mining_threshold = 0.25
    app.state.model_version = getattr(app.state, "model_version", None)
    app.state.open_encoder = open_encoder
    app.state.probe_video = probe_video
    app.state.post_pass_factory = PostPassJob
    app.state.recording = RecordingController()
    app.state.recording_lock = threading.Lock()

    # index.html pulls in /static/style.css + /static/app.js — without this mount
    # the page renders as raw unstyled HTML with a dead script (no styling, no
    # polling, the settings form falls back to a native GET navigation). Flagged
    # in the T06 log; wired here.
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(
            mjpeg_stream(app.state.capture),
            media_type=f"multipart/x-mixed-replace; boundary={_STREAM_BOUNDARY}",
        )

    @app.get("/frame")
    def frame(after: int = -1) -> Response:
        # Client-driven display (vs the MJPEG /stream): the browser polls this,
        # reads X-Frame-Generation, and paints that frame — so it knows the exact
        # frame id to pass back to /flag when the operator freezes. `after` lets
        # the client skip re-downloading an unchanged frame (204), so polling at
        # display rate costs one tiny request, not a full JPEG, per unchanged tick.
        capture = app.state.capture
        gen, snap = capture.snapshot_with_generation() if capture is not None else (0, None)
        if snap is None:
            raise HTTPException(status_code=503, detail="no frame captured yet")
        if gen == after:
            return Response(status_code=204)
        headers = {"X-Frame-Generation": str(gen), "Cache-Control": "no-store"}
        # While recording, expose the on-screen frame's MP4 write index so the
        # client can echo it to /keyframe (TR5 AC5). Omitted when idle.
        if app.state.recording.state == "recording" and snap.frame_number is not None:
            headers["X-Frame-Number"] = str(snap.frame_number)
        return Response(
            content=snap.overlay_jpeg,
            media_type="image/jpeg",
            headers=headers,
        )

    @app.post("/confidence")
    def confidence(body: ConfidenceIn) -> dict:
        # Live — the capture loop snapshots this value at predict time, so the
        # slider can move freely without corrupting already-captured provenance.
        app.state.detector.confidence_threshold = body.value
        return {"confidence": body.value}

    @app.post("/settings")
    def settings(body: SettingsIn) -> dict:
        try:
            with app.state.dataset_lock:
                # Swapping in a fresh writer resets the in-memory counters
                # (image_id / ann_id / n_flagged). Held under dataset_lock so an
                # in-flight flag can't land on a half-swapped writer.
                app.state.writer = app.state.writer_factory(
                    str(Path(body.output_path) / "images"), body.dataset_name
                )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if body.camera_index is not None:
            app.state.capture.set_camera(body.camera_index)

        # Recording entries are minted under the same output root (TR5).
        app.state.output_path = body.output_path

        return {"ok": True}

    @app.post("/flag")
    def flag(body: FlagIn | None = None) -> dict:
        # Plain def: cv2.imwrite + RLE encode are blocking and must stay off the
        # event loop. Resolve the snapshot ref first (its own lock), then
        # serialize the dataset mutation on dataset_lock.
        # Image-mode stills and recording are mutually exclusive (TR5 AC8).
        if app.state.recording.state == "recording":
            raise HTTPException(
                status_code=409, detail="stop recording to snapshot stills"
            )
        capture = app.state.capture
        requested_generation = body.generation if body is not None else None
        if capture is None:
            snap = None
        elif requested_generation is not None:
            # Capture the EXACT frame the operator froze on, not the newest.
            snap = capture.snapshot_at(requested_generation)
        else:
            snap = capture.snapshot()

        with app.state.dataset_lock:
            writer = app.state.writer
            if writer is None:
                raise HTTPException(
                    status_code=409,
                    detail="No dataset configured — set an output path and dataset "
                    "name in Settings before flagging.",
                )
            if snap is None:
                if requested_generation is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="The frame you flagged is no longer available — it "
                        "aged out of the buffer. Try again.",
                    )
                raise HTTPException(
                    status_code=409,
                    detail="No frame captured yet — the camera stream has not "
                    "produced a frame to flag.",
                )
            # Attribute access (not a positional unpack): Latest gained a 6th
            # field (frame_number, TR1) and a fixed-width unpack would raise.
            frame, dets, threshold = snap.frame, snap.dets, snap.threshold
            result = writer.flag(frame, dets, threshold)
            n_flagged = writer.n_flagged

        return {
            "image_id": result.image_id,
            "n_annotations": result.n_annotations,
            "n_flagged": n_flagged,
        }

    @app.post("/discard")
    def discard() -> dict:
        # Undo the most recent flag (the capture-preview "Discard"). Plain def:
        # unlink + annotations rewrite are blocking disk I/O. Same dataset_lock as
        # /flag so an undo can never interleave with a concurrent flag's id math.
        with app.state.dataset_lock:
            writer = app.state.writer
            if writer is None:
                raise HTTPException(
                    status_code=409, detail="No dataset configured — nothing to discard."
                )
            try:
                removed_id = writer.discard_last()
            except IndexError as exc:
                raise HTTPException(
                    status_code=409, detail="Nothing to discard yet."
                ) from exc
            return {"discarded_image_id": removed_id, "n_flagged": writer.n_flagged}

    @app.post("/validate")
    def validate() -> dict:
        # Plain def: the validator walks the dataset folder (disk I/O). Advisory,
        # in-process — never a subprocess or stdout scrape.
        writer = app.state.writer
        if writer is None:
            raise HTTPException(
                status_code=409,
                detail="No dataset configured — nothing to validate.",
            )
        errors, warnings = app.state.validate_fn(writer.dataset_dir)
        return {"errors": list(errors), "warnings": list(warnings)}

    @app.get("/status")
    def status() -> dict:
        capture = app.state.capture
        snap = capture.snapshot() if capture is not None else None
        writer = app.state.writer
        return {
            "count": snap.count if snap is not None else 0,
            "confidence": app.state.detector.confidence_threshold,
            "dataset_name": writer.dataset_name if writer is not None else None,
            # Report the operator's base Output path, not the writer's dataset dir
            # — U1 nests the writer under images/, so reporting writer.output_path
            # here would leak the nesting and change the value the frontend shows.
            "output_path": app.state.output_path,
            "n_flagged": writer.n_flagged if writer is not None else 0,
            "capture_health": capture.health if capture is not None else "dead",
            "camera_index": capture.camera_index if capture is not None else None,
            "recording_state": app.state.recording.state,
        }

    # --- Recording endpoints (TR5) --------------------------------------------
    # Plain def (AC1): folder mint, encoder open, the reader-thread join, and the
    # post-pass thread kick are all blocking and must stay off the event loop.
    # Every state transition is guarded by recording_lock (lock ordering is
    # always recording_lock -> capture's own lock, never the reverse, so the
    # capture calls made under recording_lock cannot deadlock).

    def _run_postpass(job) -> None:
        # The post-pass owns the detector for its duration (inference was paused
        # by the caller). run() is blocking, so it runs here on a worker thread,
        # off both the event loop and the lock. On completion we transition AND
        # resume inference ONLY if this is still the current job in `processing`.
        # A /record/discard that raced us cleared rec.job (and resumed inference
        # itself), so an orphaned/superseded worker must NOT touch state and must
        # NOT resume — otherwise it would un-pause the detector mid-way through a
        # later take's post-pass (§Detector sharing). Discard also CANCELS the
        # job (cooperative — run() bails at its next per-frame check) and joins
        # this thread before deleting the folder; a cancelled job is always
        # orphaned (cancel + rec.job-clear happen atomically under the lock), so
        # it can never pass the ownership guard below. Net: resume happens
        # exactly once per processing episode — here on normal completion, or in
        # discard.
        rec = app.state.recording
        job.run()
        with app.state.recording_lock:
            if rec.job is job and rec.state == "processing":
                st = job.status
                if st.state == "failed":
                    rec.state = "failed"
                    rec.error = st.error
                else:
                    rec.state = "idle"
                    rec.error = None
                rec.worker = None
                app.state.capture.resume_inference()

    @app.post("/record/start")
    def record_start(body: RecordStartIn) -> dict:
        rec = app.state.recording
        capture = app.state.capture
        entry_base = body.entry_base
        with app.state.recording_lock:
            # Start is valid ONLY from idle; the lock serializes concurrent
            # starts so exactly one wins the transition (AC12).
            if rec.state != "idle":
                raise HTTPException(
                    status_code=409, detail=f"cannot start recording from {rec.state!r}"
                )
            # Same rule as dataset_name: single path component, no leading dot.
            bad_base = (
                not entry_base
                or "/" in entry_base
                or "\\" in entry_base
                or entry_base.startswith(".")
            )
            if bad_base:
                raise HTTPException(
                    status_code=422, detail=f"invalid entry_base: {entry_base!r}"
                )
            if app.state.output_path is None:
                raise HTTPException(
                    status_code=409,
                    detail="No output path configured — set one in Settings first.",
                )
            # Auto-suffix (U1): scan videos/ for existing "<base>_NNN" siblings
            # and mint the next counter (max+1, never first-free — a gap like
            # _001/_003 still yields _004), so this never collides with an
            # existing Entry.
            videos_dir = Path(app.state.output_path) / "videos"
            pattern = re.compile(rf"^{re.escape(entry_base)}_(\d+)$")
            counters = []
            try:
                if videos_dir.is_dir():
                    for child in videos_dir.iterdir():
                        match = pattern.match(child.name)
                        if child.is_dir() and match:
                            counters.append(int(match.group(1)))
            except OSError:
                # An unreadable videos/ dir is an unusable Output path — same
                # friendly 400 as an unwritable one below, never a bare 500.
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot read output folder {videos_dir} — check that the Output path is a readable, writable location you own",
                )
            n = max(counters) + 1 if counters else 1
            name = f"{entry_base}_{n:03d}"
            entry_dir = videos_dir / name
            # The encoder frame_size comes from the current on-screen frame — no
            # snapshot yet means no dims, so no start (never an AttributeError).
            snap = capture.snapshot() if capture is not None else None
            if snap is None or snap.frame is None:
                raise HTTPException(
                    status_code=409, detail="No frame captured yet — cannot start recording."
                )
            frame = snap.frame
            frame_size = (frame.shape[1], frame.shape[0])
            # Freeze the operator threshold NOW: keyframe annotations use the
            # value the operator saw at start, not a later slider move (§Thresholds).
            operator_threshold = app.state.detector.confidence_threshold
            # Mint the video-project layout ourselves (AC2): open_encoder is
            # injectable and a fake won't mkdir. The counter scan above proved
            # entry_dir does not pre-exist, so a plain mkdir(parents=True) is safe.
            # TR4's PostPassJob.run reopens exactly <entry>/video/<entry>.mp4.
            video_dir = entry_dir / "video"
            mp4_path = video_dir / f"{name}.mp4"
            try:
                video_dir.mkdir(parents=True)
                encoder = app.state.open_encoder(mp4_path, app.state.capture_fps, frame_size)
            except OSError:
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot create output folder {entry_dir} — check that the Output path is a writable location you own",
                )
            capture.start_recording(encoder)
            rec.state = "recording"
            rec.entry_name = name
            rec.entry_dir = entry_dir
            rec.operator_threshold = operator_threshold
            rec.keyframes = set()
            rec.frame_count = 0
            rec.error = None
            rec.job = None
            rec.worker = None
        return {"ok": True, "entry_name": name, "operator_threshold": operator_threshold}

    @app.post("/keyframe")
    def keyframe(body: KeyframeIn) -> dict:
        rec = app.state.recording
        with app.state.recording_lock:
            if rec.state != "recording":
                raise HTTPException(status_code=409, detail="not recording")
            fn = body.frame_number
            if fn < 0 or fn >= app.state.capture.frames_written:
                raise HTTPException(
                    status_code=422, detail=f"frame_number out of range: {fn}"
                )
            rec.keyframes.add(fn)  # deduped: a repeat press does not grow the set
            return {"n_keyframes": len(rec.keyframes)}

    @app.post("/record/stop")
    def record_stop() -> dict:
        rec = app.state.recording
        capture = app.state.capture
        with app.state.recording_lock:
            if rec.state != "recording":
                raise HTTPException(status_code=409, detail="not recording")
            # stop_recording() stops+releases the encoder and RETURNS the final
            # frame count (and resets its own counter) — use the return value.
            frames_written = capture.stop_recording()
            rec.frame_count = frames_written
            capture.pause_inference()  # the post-pass owns the detector now
            job = app.state.post_pass_factory(
                rec.entry_dir,
                rec.entry_name,
                app.state.detector,
                keyframes=sorted(rec.keyframes),
                frame_count=frames_written,
                mining_threshold=app.state.mining_threshold,
                operator_threshold=rec.operator_threshold,
                model_version=app.state.model_version,
            )
            rec.job = job
            rec.state = "processing"
            rec.error = None
            worker = threading.Thread(target=_run_postpass, args=(job,), daemon=True)
            rec.worker = worker
            worker.start()
        # Return immediately — the job finishes on the worker thread.
        return {"ok": True, "frames_written": frames_written}

    @app.post("/record/discard")
    def record_discard() -> dict:
        rec = app.state.recording
        capture = app.state.capture

        def _finish_discard(prior_state: str) -> None:
            # Caller MUST hold recording_lock. Resets the machine to idle,
            # resumes inference iff the discarded episode was a processing one,
            # and deletes the entry folder in the SAME critical section so a
            # /record/start racing in behind us can't mint the same folder only
            # to have this delete it.
            entry_dir = rec.entry_dir
            rec.state = "idle"
            rec.entry_name = None
            rec.entry_dir = None
            rec.operator_threshold = None
            rec.keyframes = set()
            rec.frame_count = 0
            rec.error = None
            # Discarding a still-running post-pass: inference is paused and the
            # worker won't resume it (ownership guard fails — rec.job was
            # cleared before we got here) — resume it here, exactly once. From
            # "failed" the worker already resumed on completion; from
            # "recording" inference was never paused.
            if prior_state == "processing":
                capture.resume_inference()
            if entry_dir is not None:
                shutil.rmtree(entry_dir, ignore_errors=True)

        with app.state.recording_lock:
            state = rec.state
            if state == "idle":
                raise HTTPException(status_code=409, detail="nothing to discard")
            if state == "processing" and rec.job is None:
                # A concurrent discard already took ownership (cleared rec.job)
                # and is joining the worker outside the lock below; it will
                # finish the reset+rmtree itself. Letting a second discard
                # proceed would resume inference twice.
                raise HTTPException(status_code=409, detail="discard already in progress")
            # From recording this is an abort: stop+release the encoder. Inference
            # was NOT paused while recording, so nothing to resume here.
            if state == "recording":
                capture.stop_recording()
            # Clear job identity FIRST so an orphaned worker (state was
            # "processing") fails its ownership guard and neither transitions
            # state nor resumes — resume then happens exactly once, in
            # _finish_discard.
            job = rec.job
            worker = rec.worker
            rec.job = None
            rec.worker = None
            cancel = None
            if state == "processing" and job is not None:
                # Cooperatively cancel the still-running post-pass so it stops
                # calling the shared detector and writing keyframe JPEGs.
                # getattr: injected job fakes without cancel() keep the legacy
                # single-critical-section path (nothing to join — such a job
                # can't be stopped anyway).
                cancel = getattr(job, "cancel", None)
                if cancel is not None:
                    cancel()
            if cancel is None or worker is None:
                # No cancellable live worker: finish everything under this one
                # lock hold (recording/failed episodes, or a job double without
                # cancel support).
                _finish_discard(state)
                return {"ok": True}
        # A cancelled live worker: JOIN it before deleting the folder, so an
        # in-flight cv2.imwrite can't re-create a stray partial folder after
        # the rmtree. Lock ordering: the join happens with recording_lock
        # RELEASED. PostPassJob.run() itself takes no lock, but the worker's
        # wrapper (_run_postpass) acquires recording_lock immediately after
        # run() returns — joining while holding the lock would deadlock the
        # worker against us until the timeout, every time. With the lock
        # released the worker bails at its next cancel check, fails the
        # ownership guard (rec.job was cleared above, atomically with the
        # cancel), and exits. Meanwhile the machine still reads
        # state == "processing" with rec.job None: /record/start, /record/stop,
        # /record/retry and /keyframe all 409, and a second /record/discard
        # 409s on the in-progress guard above — nobody can mutate the machine
        # or mint the same entry folder before _finish_discard runs below.
        # The timeout is a bounded safety net (a healthy worker joins within
        # ~one detector predict); if it ever expires we still proceed — run()'s
        # post-predict cancel re-check keeps the stray-write window closed.
        worker.join(timeout=10.0)
        with app.state.recording_lock:
            _finish_discard(state)
        return {"ok": True}

    @app.post("/record/retry")
    def record_retry() -> dict:
        rec = app.state.recording
        capture = app.state.capture
        with app.state.recording_lock:
            if rec.state != "failed":
                raise HTTPException(
                    status_code=409, detail="can only retry a failed post-pass"
                )
            # Fresh job from the on-disk MP4 + stored keyframes (idempotent
            # rebuild — TR4 rebuilds every artifact from scratch).
            job = app.state.post_pass_factory(
                rec.entry_dir,
                rec.entry_name,
                app.state.detector,
                keyframes=sorted(rec.keyframes),
                frame_count=rec.frame_count,
                mining_threshold=app.state.mining_threshold,
                operator_threshold=rec.operator_threshold,
                model_version=app.state.model_version,
            )
            rec.job = job
            rec.state = "processing"
            rec.error = None
            capture.pause_inference()
            worker = threading.Thread(target=_run_postpass, args=(job,), daemon=True)
            rec.worker = worker
            worker.start()
        return {"ok": True}

    @app.get("/record/status")
    def record_status() -> dict:
        rec = app.state.recording
        capture = app.state.capture
        with app.state.recording_lock:
            state = rec.state
            entry_name = rec.entry_name
            n_keyframes = len(rec.keyframes)
            error = rec.error if state == "failed" else None
            job = rec.job
            frame_count = rec.frame_count
        # Live count from capture while recording; the last count (or 0) otherwise.
        frames_written = capture.frames_written if state == "recording" else frame_count
        if state in ("processing", "failed") and job is not None:
            st = job.status
            postpass = {"done": st.done, "total": st.total}
        else:
            postpass = {"done": 0, "total": 0}
        return {
            "state": state,
            "entry_name": entry_name,
            "frames_written": frames_written,
            "n_keyframes": n_keyframes,
            "postpass": postpass,
            "error": error,
        }

    return app
