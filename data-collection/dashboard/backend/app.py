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
        return Response(
            content=snap.overlay_jpeg,
            media_type="image/jpeg",
            headers={"X-Frame-Generation": str(gen), "Cache-Control": "no-store"},
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
                    body.output_path, body.dataset_name
                )
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if body.camera_index is not None:
            app.state.capture.set_camera(body.camera_index)

        return {"ok": True}

    @app.post("/flag")
    def flag(body: FlagIn | None = None) -> dict:
        # Plain def: cv2.imwrite + RLE encode are blocking and must stay off the
        # event loop. Resolve the snapshot ref first (its own lock), then
        # serialize the dataset mutation on dataset_lock.
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
            "output_path": str(writer.output_path) if writer is not None else None,
            "n_flagged": writer.n_flagged if writer is not None else 0,
            "capture_health": capture.health if capture is not None else "dead",
            "camera_index": capture.camera_index if capture is not None else None,
        }

    return app
