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
import time
from dataclasses import dataclass
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


class RecordRetryIn(BaseModel):
    """`/record/retry` body (U2 §5). ``entry_name`` must match the failed head
    (``drain.current.entry_name``) — retry is head-of-line only, never by an
    arbitrary queued name."""

    entry_name: str


@dataclass
class DrainItem:
    """One enqueued post-pass unit: the ``PostPassJob`` plus the metadata the
    queue owns *independently* of the job's internals.

    Keeping ``entry_name``/``entry_dir``/``keyframes``/``frame_count``/
    ``operator_threshold`` here (rather than reaching into the job) lets
    ``/record/status`` report ``entry_name`` for a not-yet-started queued item,
    ``/record/retry`` rebuild a *fresh* job from frame 0 (idempotent rebuild from
    the MP4), and ``/record/discard`` ``rmtree`` the folder — none of which the
    frozen ``PostPassJob`` surface exposes. ``frame_count`` mirrors the job's own
    ``total`` so a queued item reports its total with no running job.
    """

    job: object
    entry_name: str
    entry_dir: Path
    keyframes: list
    frame_count: int
    operator_threshold: float | None


class RecordingController:
    """Foreground recording state + the orthogonal post-pass drain queue (U2).

    The foreground collapses to ``idle ↔ recording`` only (INV-1); ``processing``
    and ``failed`` are gone as foreground states. Post-pass becomes a FIFO queue
    (INV-6) drained by a *single* long-lived worker thread, but only while
    ``state == 'idle'`` (INV-2). Every field is read/written only under
    ``app.state.recording_lock`` — the same lock ``app.state.drain_cond`` waits
    on — so one monitor guards both the foreground transitions and the queue.
    """

    def __init__(self) -> None:
        self.state = "idle"  # "idle" | "recording"  (INV-1)

        # Recording-active scratch — meaningful only while state == "recording",
        # then handed to the enqueued PostPassJob at /record/stop and cleared.
        self.entry_name: str | None = None
        self.entry_dir: Path | None = None
        self.operator_threshold: float | None = None
        self.keyframes: set[int] = set()
        self.frame_count = 0

        # The drain queue. `current` is the job the worker is actively draining,
        # the job parked mid-drain (paused for a recording), or the failed head
        # that has halted the queue (§5); `queue` holds the items waiting behind
        # it in FIFO order. `error` is set iff `current` is a failed head.
        self.queue: list[DrainItem] = []
        self.current: DrainItem | None = None
        self.error: str | None = None

        # Set by /record/start for the window in which it commits to recording
        # but has released the lock to park the in-flight drain — so the worker
        # will not grab a fresh queued job (and a second /record/start cannot
        # also win) while state is still momentarily "idle".
        self.recording_pending = False


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
    # recording_lock guards every RecordingController field; drain_cond wraps that
    # SAME lock so the drain worker can wait/notify on it. `with recording_lock`
    # (endpoints) and `with drain_cond` (the worker's wait) are therefore mutually
    # exclusive — one monitor, no second lock to order against it.
    app.state.recording_lock = threading.Lock()
    app.state.drain_cond = threading.Condition(app.state.recording_lock)
    app.state.drain_shutdown = False
    # The shared detector lock (INV-3): the SAME object the live capture loop
    # guards each predict with, injected into every PostPassJob so the drain
    # worker serialises its per-frame predict against the live loop. Owned by the
    # real CaptureLoop (created at its construction, before this app, so its loop
    # thread has it from the first tick); a stub capture without one gets a
    # private fallback (a stub has no competing live predict, so INV-3 is moot
    # there). It is a LEAF lock — never acquired while holding recording_lock /
    # drain_cond / capture._lock, and only ever held around predict itself.
    app.state.detector_lock = getattr(capture, "detector_lock", None) or threading.Lock()
    # Post-pass throughput used to compute drain.eta_seconds (§6). A plain settable
    # float so a test can inject a known value and assert the formula; defaults to
    # the ADR's 0.6 frames/s. eta always reads THIS value (no hidden EMA).
    app.state.detect_fps = 0.6

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

    # --- Recording endpoints + drain worker (U2) ------------------------------
    # Plain def (AC1): folder mint, encoder open, the reader-thread join, and the
    # drain hand-off are blocking and must stay off the event loop.
    #
    # Lock ordering (deadlock-freedom argument):
    #   recording_lock (== drain_cond's lock)  →  capture's internal lock
    # is the ONLY order any thread ever takes. capture never calls back into the
    # recording machine, so there is no cycle. The drain worker holds
    # recording_lock only to pick a job, publish `current`, toggle the detector
    # flag, and handle completion — NEVER across job.run() (predict is slow and
    # must not hold the monitor). /record/start and /record/discard both do their
    # *bounded waits* (for the worker to park / reach a terminal state) with the
    # lock RELEASED, precisely because the worker needs recording_lock the instant
    # run() returns; waiting under the lock would deadlock them against it.
    #
    # Single-detector-owner (INV-3): the drain worker sets capture.pause_inference()
    # before it predicts and capture.resume_inference() when the queue drains, when
    # it halts on a failure, or when it yields to a recording. /record/start never
    # opens the encoder until the in-flight drain has PARKED (bounded wait on
    # job.status == "paused"/terminal), so a recording predict can never race a
    # drain predict.

    _ETA_FPS_EPS = 1e-6  # eta div-by-zero guard (§6)

    def _wait_for(predicate, timeout=10.0, interval=0.005) -> bool:
        # Bounded poll (never sleep-and-hope): runs on the endpoint's threadpool
        # thread (plain def), so a short time.sleep is off the event loop. Returns
        # False on timeout — a safety net; in practice the worker parks / reaches a
        # terminal state within ~one detector predict.
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                return False
            time.sleep(interval)
        return True

    def _new_postpass_job(*, entry_dir, entry_name, keyframes, frame_count, operator_threshold):
        # Build a post-pass job via the preserved factory seam, then inject the
        # shared detector lock as an ATTRIBUTE (INV-3) — the factory CALL signature
        # is unchanged, so injected fakes keep working; the real PostPassJob reads
        # `predict_lock` to guard each predict, a fake simply ignores it. setattr
        # is guarded so a slotted fake can't break the build.
        job = app.state.post_pass_factory(
            entry_dir,
            entry_name,
            app.state.detector,
            keyframes=keyframes,
            frame_count=frame_count,
            mining_threshold=app.state.mining_threshold,
            operator_threshold=operator_threshold,
            model_version=app.state.model_version,
        )
        try:
            job.predict_lock = app.state.detector_lock
        except (AttributeError, TypeError):
            pass
        return job

    def _drain_runnable_locked(rec) -> bool:
        # Caller holds recording_lock. True when the worker may (re)start draining:
        # the tool is idle, no /record/start is mid-commit, no failed head is
        # blocking the queue (§5), and there is a job to run — either a `current`
        # whose job is still "pending" (a retry's fresh job, or a just-assigned
        # head) or a queued item. A "running"/"paused" current means the worker is
        # already inside its run() (parked jobs are un-parked by the idle-return
        # endpoints, not re-picked here), so it is NOT runnable.
        if rec.state != "idle" or rec.recording_pending or rec.error is not None:
            return False
        if rec.current is not None:
            return rec.current.job.status.state == "pending"
        return bool(rec.queue)

    def _take_runnable_locked(rec):
        # Caller holds recording_lock and _drain_runnable_locked(rec) is True.
        if rec.current is not None and rec.current.job.status.state == "pending":
            return rec.current
        item = rec.queue.pop(0)
        rec.current = item
        return item

    def _resume_parked_current_locked() -> None:
        # Caller holds recording_lock. On return to idle, reclaim the detector for
        # the drain and un-park the head job so the worker's BLOCKED run() (parked
        # inside PostPassJob.run on /record/start) continues from the parked index.
        rec = app.state.recording
        cur = rec.current
        if cur is not None and cur.job.status.state == "paused":
            app.state.capture.pause_inference()
            cur.job.resume()

    def _drain_worker() -> None:
        # The single long-lived daemon worker (INV-6: FIFO, one job at a time).
        rec = app.state.recording
        capture = app.state.capture
        cond = app.state.drain_cond
        while True:
            with cond:
                while not app.state.drain_shutdown and not _drain_runnable_locked(rec):
                    cond.wait()
                if app.state.drain_shutdown:
                    return
                item = _take_runnable_locked(rec)
                # Claim the shared detector before predicting (INV-3). The live
                # capture loop skips predict + publishes the raw frame while
                # inference_paused is True.
                capture.pause_inference()
            try:
                # run() blocks HERE across any pause: /record/start parks it
                # internally and the idle-return path un-parks it; it returns only
                # on a terminal state (done / failed / cancelled). No lock held —
                # this is where the slow per-frame predict happens (INV-2: only
                # while state == idle, enforced by the parking handshake).
                item.job.run()
            except Exception:  # a job double misbehaving must not kill the worker
                pass
            with cond:
                st = item.job.status
                if rec.current is item:
                    if st.state == "failed":
                        # Halt-on-failure (§5): the failed job STAYS as the head,
                        # top-level error is set, the worker stops advancing.
                        # INV-5: nothing is deleted.
                        rec.error = st.error
                    else:
                        # "done" (normal) or "cancelled" without a discard having
                        # taken ownership — drop the head and advance.
                        rec.current = None
                        rec.error = None
                # else: /record/discard took ownership of this job (it cleared
                # rec.current before cancelling) and owns the rmtree — we must not
                # touch state for it.
                # Release the detector to the live overlay whenever no further
                # drain work is runnable (queue empty, or halted on a failed head,
                # or a recording is now in progress).
                if not _drain_runnable_locked(rec):
                    capture.resume_inference()
                cond.notify_all()

    drain_worker_thread = threading.Thread(
        target=_drain_worker, name="postpass-drain", daemon=True
    )
    # Started at build time (NOT gated on a FastAPI lifespan/startup event) so the
    # queue drains under a bare TestClient(app) with no `with` block (§4).
    drain_worker_thread.start()
    app.state.drain_worker = drain_worker_thread

    @app.post("/record/start")
    def record_start(body: RecordStartIn) -> dict:
        rec = app.state.recording
        capture = app.state.capture
        entry_base = body.entry_base

        # --- Phase 1 (under lock): validate, mint the folder, open the encoder,
        # and decide whether an in-flight drain must be parked. None of this
        # touches the drain yet, so a 4xx here leaves a draining queue undisturbed.
        with app.state.recording_lock:
            # 409 ONLY if already recording — or if a concurrent /record/start is
            # mid-commit (recording_pending serializes them so exactly one wins,
            # AC12). A DRAINING queue does NOT block start (AC1); we pause it.
            if rec.state == "recording" or rec.recording_pending:
                raise HTTPException(status_code=409, detail="already recording")
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
                # mkdir may have created the dir before open_encoder failed —
                # remove the orphan so a failed start leaves nothing behind.
                shutil.rmtree(entry_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot create output folder {entry_dir} — check that the Output path is a writable location you own",
                )
            except BaseException:
                # Any other failure opening the encoder: same cleanup, then let it
                # propagate (a real bug surfaces as 500, but no orphan folder).
                shutil.rmtree(entry_dir, ignore_errors=True)
                raise
            # Commit to recording. recording_pending closes the lock-released
            # window below: the worker won't grab a fresh queued job and a second
            # start can't win while state is still momentarily "idle".
            rec.recording_pending = True
            cur = rec.current
            # The worker owns the detector iff inference is currently paused for
            # an in-flight drain. Only then must we park that drain and reclaim
            # the detector for the live overlay (INV-3); if inference is not
            # paused the overlay already owns it, and a start must NOT spuriously
            # resume (that inflates the pause/resume bookkeeping and, on the first
            # take, resumes a detector nobody paused). Read under the lock so it
            # cannot race the worker's own claim (which pauses under this lock).
            worker_owns_detector = capture.inference_paused
            if worker_owns_detector and cur is not None:
                cur.job.pause()  # honored between frames (INV-4: within one frame)
                app.state.drain_cond.notify_all()

        # --- Phase 2 (lock RELEASED): wait for the drain to actually yield the
        # detector — park at its current frame, or finish/fail/cancel — so a
        # recording predict can never race a drain predict (INV-3). Bounded.
        if worker_owns_detector and cur is not None:
            _wait_for(
                lambda: cur.job.status.state in ("paused", "done", "failed", "cancelled")
            )

        # --- Phase 3 (under lock): reclaim the detector for the live overlay and
        # start recording. On the (guarded, not-expected) chance start_recording
        # raises, roll back: un-park the drain so the queue is never stranded.
        with app.state.recording_lock:
            try:
                if worker_owns_detector:
                    capture.resume_inference()  # reclaim from the parked/finished drain
                capture.start_recording(encoder)
                rec.state = "recording"
                rec.entry_name = name
                rec.entry_dir = entry_dir
                rec.operator_threshold = operator_threshold
                rec.keyframes = set()
                rec.frame_count = 0
            except Exception:
                # start_recording failed after we minted the folder + opened the
                # encoder: release the encoder and remove the orphan entry dir so a
                # failed start leaves nothing behind, and un-park the drain so the
                # queue is never stranded. capture.start_recording assigns the
                # encoder only on success, so releasing it here can't double-close.
                _resume_parked_current_locked()
                app.state.drain_cond.notify_all()
                _release = getattr(encoder, "release", None)
                if _release is not None:
                    try:
                        _release()
                    except Exception:
                        pass
                shutil.rmtree(entry_dir, ignore_errors=True)
                raise
            finally:
                rec.recording_pending = False
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
            # The MP4 is finalized HERE, before enqueue, so the raw recording
            # survives independent of the post-pass (INV-5).
            frames_written = capture.stop_recording()
            # Enqueue the post-pass at the BACK of the queue (FIFO, INV-6). The
            # job carries its own frame_count (== total). We do NOT start it or
            # block on it — the worker drains it when the tool is idle (AC1).
            job = _new_postpass_job(
                entry_dir=rec.entry_dir,
                entry_name=rec.entry_name,
                keyframes=sorted(rec.keyframes),
                frame_count=frames_written,
                operator_threshold=rec.operator_threshold,
            )
            rec.queue.append(
                DrainItem(
                    job=job,
                    entry_name=rec.entry_name,
                    entry_dir=rec.entry_dir,
                    keyframes=sorted(rec.keyframes),
                    frame_count=frames_written,
                    operator_threshold=rec.operator_threshold,
                )
            )
            # Back to idle immediately — never a foreground "processing" state.
            rec.state = "idle"
            rec.entry_name = None
            rec.entry_dir = None
            rec.operator_threshold = None
            rec.keyframes = set()
            rec.frame_count = 0
            # Resume the head drain we parked at /record/start (if any) so it
            # continues from its parked index; otherwise the worker picks up the
            # job just enqueued. Either way, wake the worker.
            _resume_parked_current_locked()
            app.state.drain_cond.notify_all()
        return {"ok": True, "frames_written": frames_written}

    @app.post("/record/discard")
    def record_discard() -> dict:
        rec = app.state.recording
        capture = app.state.capture
        with app.state.recording_lock:
            state = rec.state
            if state == "recording":
                # Abort the live take (as today): stop+release the encoder and
                # delete the half-written Entry — the operator's explicit choice,
                # the one sanctioned delete (INV-5). Inference was NOT owned by
                # the drain while recording, so there is nothing to resume for the
                # take itself; but a drain we PARKED to start this recording must
                # be un-parked so the queue is not stranded.
                capture.stop_recording()
                entry_dir = rec.entry_dir
                rec.state = "idle"
                rec.entry_name = None
                rec.entry_dir = None
                rec.operator_threshold = None
                rec.keyframes = set()
                rec.frame_count = 0
                if entry_dir is not None:
                    shutil.rmtree(entry_dir, ignore_errors=True)
                _resume_parked_current_locked()
                app.state.drain_cond.notify_all()
                return {"ok": True}
            # Idle: discard the CURRENT drain job — the one in drain.current,
            # actively draining OR a failed head (§5). Discarding an arbitrary
            # QUEUED entry is out of scope (U2 §9).
            if rec.current is None:
                raise HTTPException(status_code=409, detail="nothing to discard")
            victim = rec.current
            # Take ownership away from the worker FIRST (clear current + any
            # failed-head error) so its post-run handler won't touch state for
            # this job — this is the single-worker analogue of the old rec.job
            # clear. Then cooperatively cancel so run() stops calling the shared
            # detector and writing files.
            rec.current = None
            rec.error = None
            cancel = getattr(victim.job, "cancel", None)
            if cancel is not None:
                cancel()
            # A parked head (unreachable from idle, but defensive) must be
            # un-parked to observe the cancel.
            if victim.job.status.state == "paused":
                victim.job.resume()
            app.state.drain_cond.notify_all()
        # Lock RELEASED: wait only while the job could still be mid-write — i.e.
        # while run() is actually executing ("running", or "paused" which we just
        # un-parked toward a cancel). Once it leaves those states it performs no
        # further disk writes (cancel is re-checked after predict, before every
        # write), so it is safe to rmtree. A head that never started ("pending" —
        # e.g. a retry job discarded before the worker picked it up) or is already
        # terminal ("cancelled"/"failed"/"done") returns the wait immediately, so
        # discard never burns the timeout on a job that isn't touching disk. This
        # is the single-worker replacement for the old thread join and preserves
        # the "no stray folder after rmtree" guarantee for jobs that WERE running.
        # Bounded; if it ever times out we still proceed (run()'s post-predict
        # cancel re-check keeps the window closed). We do NOT hold the lock here
        # because the worker needs it the instant run() returns.
        if cancel is not None:
            _wait_for(lambda: victim.job.status.state not in ("running", "paused"))
        shutil.rmtree(victim.entry_dir, ignore_errors=True)
        # Advance the queue: wake the worker to start the next job (if any). The
        # detector flag is owned by the worker (it resumes inference itself when
        # nothing is runnable), so we only notify.
        with app.state.recording_lock:
            app.state.drain_cond.notify_all()
        return {"ok": True}

    @app.post("/record/retry")
    def record_retry(body: RecordRetryIn) -> dict:
        rec = app.state.recording
        with app.state.recording_lock:
            # Valid ONLY when the head is a FAILED drain job whose entry_name
            # matches (§5, head-of-line). Because the worker halts on the first
            # failure, at most one entry is ever failed — so this single check is
            # unambiguous.
            if (
                rec.error is None
                or rec.current is None
                or rec.current.entry_name != body.entry_name
            ):
                raise HTTPException(
                    status_code=409, detail="no matching failed drain job to retry"
                )
            item = rec.current
            # Fresh job — idempotent rebuild from the on-disk MP4 + stored
            # keyframes, from frame 0 (TR4 rebuilds every artifact from scratch).
            # Replace the failed job IN PLACE on the head DrainItem so it stays the
            # current head; clearing the error makes the worker pick it up.
            item.job = _new_postpass_job(
                entry_dir=item.entry_dir,
                entry_name=item.entry_name,
                keyframes=sorted(item.keyframes),
                frame_count=item.frame_count,
                operator_threshold=item.operator_threshold,
            )
            rec.error = None
            app.state.drain_cond.notify_all()
        return {"ok": True}

    @app.get("/record/status")
    def record_status() -> dict:
        rec = app.state.recording
        with app.state.recording_lock:
            state = rec.state
            cur = rec.current
            if cur is not None:
                st = cur.job.status
                current = {
                    "entry_name": cur.entry_name,
                    "done": st.done,
                    "total": cur.frame_count,
                }
                remaining = max(cur.frame_count - st.done, 0)
            else:
                current = None
                remaining = 0
            queued = [it.entry_name for it in rec.queue]
            remaining += sum(it.frame_count for it in rec.queue)
            error = rec.error
        # eta = remaining_frames / detect_fps (§6), computed from the CURRENT
        # app.state.detect_fps so a test can inject a known fps and assert the
        # formula. 0 when nothing is draining or queued; div-by-zero guarded.
        detect_fps = app.state.detect_fps
        fps = detect_fps if detect_fps > _ETA_FPS_EPS else _ETA_FPS_EPS
        eta_seconds = remaining / fps if remaining else 0.0
        return {
            "state": state,
            "drain": {"current": current, "queued": queued, "eta_seconds": eta_seconds},
            "error": error,
        }

    return app
