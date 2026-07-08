"""T03: `CaptureLoop` — the single thread that owns the camera + tracker.update(),
publishing an atomic `Latest` snapshot for HTTP handlers to read (DESIGN D6).
See `app/docs/tasks/T03-capture.md` for the frozen signature and ACs.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, NamedTuple, Protocol

import cv2
import numpy as np

from backend.render import render as _default_render

logger = logging.getLogger(__name__)

# Cap the failure retry rate so a frozen camera / always-throwing tracker never
# busy-spins (AC6): at most ~30 attempts per second.
_RETRY_INTERVAL_S = 1.0 / 30.0


class VideoCaptureLike(Protocol):
    """The subset of `cv2.VideoCapture`'s surface `CaptureLoop` needs —
    satisfied by `cv2.VideoCapture` itself and by `FakeCaptureSource`."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray]: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def get(self, prop_id: int) -> float: ...
    def release(self) -> None: ...


class Latest(NamedTuple):
    overlay_jpeg: bytes            # encoded 1080p overlay for /stream
    present_ids: frozenset[int]    # tracker_ids — BUILTIN int (cast from the
                                   # tracker's np.int64) so /status JSON-encodes;
                                   # T04 must not have to sanitize (see AC12)
    count: int                     # len(present_ids)
    t: float                       # monotonic seconds at capture


OnFrame = Callable[[float, frozenset[int]], None]   # (t, present_ids)


class CaptureLoop:
    def __init__(
        self,
        tracker,
        camera_index: int,
        on_frame: OnFrame | None = None,
        cap_factory: Callable[[int], VideoCaptureLike] = cv2.VideoCapture,
        frame_size: tuple[int, int] = (1920, 1080),
        stale_after_s: float = 2.0,
        render_fn=_default_render,
    ) -> None:
        self._tracker = tracker
        self._camera_index = camera_index
        self._on_frame = on_frame
        self._cap_factory = cap_factory
        self._frame_size = frame_size
        self._stale_after_s = stale_after_s
        self._render_fn = render_fn

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reset_request = threading.Event()
        self._reset_done = threading.Event()

        self._cap: VideoCaptureLike | None = None
        self._thread: threading.Thread | None = None
        self._latest: Latest | None = None
        self._generation = 0
        self._start_time = 0.0
        self._last_publish: float | None = None

    def start(self) -> None:
        cap = self._cap_factory(self._camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"capture {self._camera_index} would not open")
        width, height = self._frame_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._start_time = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reset_tracker(self, timeout_s: float = 2.0) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._reset_done.clear()
        self._reset_request.set()
        if not self._reset_done.wait(timeout_s):
            raise TimeoutError("tracker reset was not applied (capture thread stalled)")

    def snapshot(self) -> Latest | None:
        with self._lock:
            return self._latest

    @property
    def health(self) -> str:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return "dead"
        reference = self._last_publish if self._last_publish is not None else self._start_time
        age = time.monotonic() - reference
        return "ok" if age <= self._stale_after_s else "stale"

    @property
    def generation(self) -> int:
        return self._generation

    def _run(self) -> None:
        cap = self._cap
        try:
            while not self._stop.is_set():
                self._apply_pending_reset()
                try:
                    published = self._capture_tick(cap)
                except (KeyboardInterrupt, SystemExit):
                    # The only escapes: genuine interpreter-level shutdown
                    # signals. These exit the thread (an ABNORMAL thread exit),
                    # surfacing as health "dead" (AC7).
                    raise
                except BaseException:
                    # AC7: ANY other exception a tick can raise — from
                    # tracker.update, render, or encode, whatever its type — is
                    # caught, logged, and the tick skipped so the loop never
                    # crashes. A single blip (given a prior publication) stays
                    # "ok"; a sustained run of exceptions produces no
                    # publications and ages into "stale" (AC6), never "dead".
                    logger.exception("capture tick failed; skipping this frame")
                    published = False
                if not published:
                    # Cap the retry rate on a skipped tick (failed read OR caught
                    # exception) so a frozen camera / always-throwing tracker can
                    # never busy-spin (AC6): at most ~30 attempts per second.
                    self._stop.wait(_RETRY_INTERVAL_S)
        finally:
            if cap is not None:
                cap.release()

    def _apply_pending_reset(self) -> None:
        if not self._reset_request.is_set():
            return
        try:
            self._tracker.reset()
        finally:
            self._reset_request.clear()
            self._reset_done.set()

    def _capture_tick(self, cap: VideoCaptureLike) -> bool:
        """Read → update → render → encode → publish one frame. Returns True on
        a publication, False on a failed read. Raises to signal a skipped tick
        (caught by `_run`); a BaseException propagates to kill the thread."""
        ok, frame = cap.read()
        if not ok:
            return False

        t = time.monotonic()
        dets = self._tracker.update(frame)
        overlay = self._render_fn(frame.copy(), dets)
        _, buffer = cv2.imencode(".jpg", overlay)
        present_ids = frozenset(int(tracker_id) for tracker_id in _tracker_ids(dets))
        latest = Latest(
            overlay_jpeg=buffer.tobytes(),
            present_ids=present_ids,
            count=len(present_ids),
            t=t,
        )
        with self._lock:
            self._latest = latest
            self._generation += 1
            self._last_publish = time.monotonic()

        self._notify(t, present_ids)
        return True

    def _notify(self, t: float, present_ids: frozenset[int]) -> None:
        if self._on_frame is None:
            return
        try:
            self._on_frame(t, present_ids)
        except Exception:
            # AC8: an on_frame exception must not kill the loop. The publication
            # already happened, so this frame still counts.
            logger.exception("on_frame callback raised; continuing")


def _tracker_ids(dets):
    return dets.tracker_id if dets.tracker_id is not None else ()
