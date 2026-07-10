"""FastAPI layer — wires the capture loop, dataset writer, and renderer into the
dashboard's endpoints (spec §Runtime table).

The threading rules that bite (spec §Threading rules) are load-bearing here:

- ``/flag``, ``/keyframe``, and the ``/record/*`` endpoints are plain ``def``
  (not ``async``). Their blocking work (``cv2.imwrite`` / ``imencode`` + RLE
  encode; the encoder open + reader-thread join; the synchronous keyframe write
  at stop) runs on Starlette's threadpool, off the event loop, so a SPACE press
  never freezes ``/stream``.
- ``/flag`` grabs the ``Latest`` tuple **reference** atomically via
  ``capture.snapshot()`` (the loop rebinds it under its own lock, never mutates in
  place), then does its blocking work on that stable snapshot.
- ``/flag`` and the dataset-mutating half of ``/settings`` serialize on a single
  ``dataset_lock`` so rapid concurrent flags get sequential ``image_id``/``ann_id``
  and never a torn ``annotations.json``.
- Provenance uses the threshold captured **into the snapshot at predict time**
  (``snap.threshold``), never the live slider value at flag time. Video-mode
  keyframes follow the same rule — see ``/keyframe``.
- ``/stream`` paces on ``capture.generation`` so each overlay is sent exactly once.

Recording mode (ADR-0002, the 2026-07-09 simplification): there is no offline
post-pass and no drain queue. ``/record/stop`` saves the finished MP4 and, from
the keyframe detections captured live at SPACE-press time, writes the reviewed
video-project artifacts **synchronously** before returning — the capture loop is
the sole detector caller, so nothing to share or park.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import threading
from pathlib import Path

import cv2
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
    """`/record/start` body. ``entry_base`` is auto-suffixed with a zero-padded
    counter to mint the resolved ``entry_name`` (U1)."""

    entry_base: str


class KeyframeIn(BaseModel):
    """`/keyframe` body. ``generation`` pins the exact displayed frame to mark —
    the same id ``/frame`` echoes and ``/flag`` uses."""

    generation: int


class RecordingController:
    """Foreground recording state (idle ↔ recording only).

    ``keyframes`` maps ``frame_number -> (jpeg, dets, threshold)`` captured live
    at each SPACE press — dict-keyed so a repeat press on the same frame dedups
    for free. ``epoch`` is bumped on every ``/record/start`` so a ``/keyframe``
    whose (unlocked) imencode straddles a Stop→Start can't land its frame in the
    *next* recording's keyframe set. Every field is read/written only under
    ``app.state.recording_lock``.
    """

    def __init__(self) -> None:
        self.state = "idle"  # "idle" | "recording"
        self.epoch = 0  # incremented per recording; identifies the current take

        # Recording-active scratch — meaningful only while state == "recording",
        # then consumed by /record/stop and cleared.
        self.entry_name: str | None = None
        self.entry_dir: Path | None = None
        self.keyframes: dict[int, tuple] = {}


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

    # --- Recording mode -------------------------------------------------------
    # output_path is captured on the /settings success below (recording writes
    # entries under it, same root as image-mode datasets). capture_fps is threaded
    # in from main.py's CLI args; the default here keeps tests that don't set it
    # working. The encoder opener, MP4 probe, and video-project writer are imported
    # but injectable via app.state so tests run with fakes (no real ffmpeg / video).
    from backend.encoder import open_encoder, probe_video
    from backend.video_writer import VideoEntryWriter

    app.state.output_path = None
    app.state.capture_fps = 30.0
    app.state.open_encoder = open_encoder
    app.state.probe_video = probe_video
    app.state.video_writer_factory = VideoEntryWriter
    app.state.recording = RecordingController()
    # recording_lock guards every RecordingController field.
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
        # frame id to pass back to /flag or /keyframe when the operator marks.
        # `after` lets the client skip re-downloading an unchanged frame (204), so
        # polling at display rate costs one tiny request, not a full JPEG, per
        # unchanged tick.
        capture = app.state.capture
        gen, snap = capture.snapshot_with_generation() if capture is not None else (0, None)
        if snap is None:
            raise HTTPException(status_code=503, detail="no frame captured yet")
        if gen == after:
            return Response(status_code=204)
        headers = {"X-Frame-Generation": str(gen), "Cache-Control": "no-store"}
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

        # Recording entries are minted under the same output root.
        app.state.output_path = body.output_path

        return {"ok": True}

    @app.post("/flag")
    def flag(body: FlagIn | None = None) -> dict:
        # Plain def: cv2.imwrite + RLE encode are blocking and must stay off the
        # event loop. Resolve the snapshot ref first (its own lock), then
        # serialize the dataset mutation on dataset_lock.
        # Image-mode stills and recording are mutually exclusive (spec).
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
            # field (frame_number) and a fixed-width unpack would raise. The
            # capture loop is the sole predictor now, so snap.dets is always the
            # live detection for the frozen frame.
            result = writer.flag(snap.frame, snap.dets, snap.threshold)
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

    # --- Recording endpoints --------------------------------------------------
    # Plain def: folder mint, encoder open, the reader-thread join, and the
    # synchronous keyframe write at stop are blocking and must stay off the event
    # loop. There is no background worker and no shared detector — the capture loop
    # is the sole predictor (ADR-0002), so recording_lock guards only the small
    # RecordingController state, never a slow predict.

    @app.post("/record/start")
    def record_start(body: RecordStartIn) -> dict:
        rec = app.state.recording
        capture = app.state.capture
        entry_base = body.entry_base

        with app.state.recording_lock:
            # One recording at a time; 409 only if already recording (AC12
            # serializes concurrent starts on this lock so exactly one wins).
            if rec.state == "recording":
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
            # Mint the video-project layout ourselves: open_encoder is injectable
            # and a fake won't mkdir. The counter scan above proved entry_dir does
            # not pre-exist, so a plain mkdir(parents=True) is safe.
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
            try:
                capture.start_recording(encoder)
            except Exception:
                # start_recording failed after we minted the folder + opened the
                # encoder: release the encoder and remove the orphan entry dir so a
                # failed start leaves nothing behind. capture.start_recording
                # assigns the encoder only on success, so releasing here can't
                # double-close.
                _release = getattr(encoder, "release", None)
                if _release is not None:
                    try:
                        _release()
                    except Exception:
                        pass
                shutil.rmtree(entry_dir, ignore_errors=True)
                raise
            rec.state = "recording"
            rec.epoch += 1
            rec.entry_name = name
            rec.entry_dir = entry_dir
            rec.keyframes = {}
        return {"ok": True, "entry_name": name}

    @app.post("/keyframe")
    def keyframe(body: KeyframeIn) -> dict:
        # Mark the frame the operator froze on as a keyframe, capturing the live
        # detection on it — image-mode /flag's exact plumbing, written into the
        # video-project layout at stop. Pre-encode the frame to JPEG bytes HERE so
        # the stored keyframe dict is bounded (~tens of KB, not ~6 MB/frame) even
        # under SPACE-spamming.
        rec = app.state.recording
        capture = app.state.capture
        with app.state.recording_lock:
            if rec.state != "recording":
                raise HTTPException(status_code=409, detail="not recording")
            epoch = rec.epoch  # pin the take so a Stop→Start during imencode can't leak
        snap = capture.snapshot_at(body.generation) if capture is not None else None
        if snap is None or snap.frame_number is None:
            # Aged out of the ring (or an idle-stamped frame) — non-fatal: the
            # client surfaces it and the operator marks again.
            raise HTTPException(
                status_code=409,
                detail="The frame you marked aged out of the buffer — try again.",
            )
        ok_encode, buf = cv2.imencode(".jpg", snap.frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        jpeg = buf.tobytes() if ok_encode else b""
        with app.state.recording_lock:
            # Re-check the SAME take: if this recording stopped (and possibly a new
            # one started) while we imencoded off-lock, drop the mark rather than
            # writing R1's frame into R2's keyframe set.
            if rec.state != "recording" or rec.epoch != epoch:
                raise HTTPException(status_code=409, detail="not recording")
            # Dict-keyed on frame_number: a repeat press on the same frame dedups.
            rec.keyframes[snap.frame_number] = (jpeg, snap.dets, snap.threshold)
            n_keyframes = len(rec.keyframes)
        return {"generation": body.generation, "n_keyframes": n_keyframes}

    @app.post("/record/stop")
    def record_stop() -> dict:
        rec = app.state.recording
        capture = app.state.capture
        with app.state.recording_lock:
            if rec.state != "recording":
                raise HTTPException(status_code=409, detail="not recording")
            entry_dir = rec.entry_dir
            entry_name = rec.entry_name
            keyframes = dict(rec.keyframes)
            # stop_recording() stops+releases the encoder and RETURNS the final
            # frame count. The MP4 is finalized HERE, before we write anything
            # else, so the raw recording survives independent of the annotation
            # write below (INV-5).
            frames_written = capture.stop_recording()
            # Back to idle immediately — a new recording can start right away, even
            # while the (sub-second) keyframe write below is still running (it
            # touches only this entry's own dir).
            rec.state = "idle"
            rec.entry_name = None
            rec.entry_dir = None
            rec.keyframes = {}

        # Lock released: write the reviewed video-project artifacts from the
        # keyframe detections captured live at press time. selected_frames.json is
        # written FIRST so a mid-write failure leaves exactly the contract's valid
        # PARTIAL entry (video + selected_frames, no annotations → "needs
        # detection"). Recovery for a failed write is "re-detect in the editor",
        # by design — no retry machinery (ADR-0002).
        mp4_path = entry_dir / "video" / f"{entry_name}.mp4"
        probe = app.state.probe_video(str(mp4_path))
        video = {
            "fps": probe.fps,
            "width": probe.width,
            "height": probe.height,
            "frame_count": frames_written,
        }
        writer = app.state.video_writer_factory(
            entry_dir, entry_name, video=video, keyframes=sorted(keyframes)
        )
        writer.write_selected_frames()
        for fn in sorted(keyframes):
            jpeg, dets, threshold = keyframes[fn]
            writer.add_keyframe(fn, jpeg, dets, threshold)
        writer.finalize()

        # An encoder.write exception mid-take stops the reader and freezes the
        # frame count (capture.recording_error — AC8). The MP4 + the keyframes
        # marked before the failure are still finalized above (INV-5: no captured
        # data is thrown away), but the recording is TRUNCATED — report that
        # instead of a clean success, so the operator knows the saved clip is
        # incomplete and re-records rather than trusting it.
        error = getattr(capture, "recording_error", None)
        return {
            "ok": error is None,
            "frames_written": frames_written,
            "error": None if error is None else str(error),
        }

    @app.post("/record/discard")
    def record_discard() -> dict:
        rec = app.state.recording
        capture = app.state.capture
        with app.state.recording_lock:
            if rec.state != "recording":
                raise HTTPException(status_code=409, detail="nothing to discard")
            # Abort the live take: stop+release the encoder and delete the
            # half-written Entry — the operator's explicit choice, the one
            # sanctioned delete (INV-5).
            capture.stop_recording()
            entry_dir = rec.entry_dir
            rec.state = "idle"
            rec.entry_name = None
            rec.entry_dir = None
            rec.keyframes = {}
        if entry_dir is not None:
            shutil.rmtree(entry_dir, ignore_errors=True)
        return {"ok": True}

    @app.get("/record/status")
    def record_status() -> dict:
        rec = app.state.recording
        # Surface an encoder failure that struck mid-take (AC8) while it still
        # matters: during recording a non-None recording_error means the reader
        # has stopped writing frames and the take is doomed — the operator should
        # Discard and restart without waiting for Stop. Idle always reports
        # error=None so a finished take's error can't linger past the next start.
        error = None
        if rec.state == "recording":
            err = getattr(app.state.capture, "recording_error", None)
            if err is not None:
                error = str(err)
        return {"state": rec.state, "error": error}

    return app
