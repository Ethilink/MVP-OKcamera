"""Single-threaded camera and tracker loop.

Publishes an atomic ``Latest`` snapshot for HTTP handlers (DESIGN D6).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Protocol

import cv2
import numpy as np

from backend.render import OverlayRenderer

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


# One detection's raw preview material. `mask` is an owned, read-only boolean
# crop aligned to the clamped integer bounds of `xyxy`, rather than the full
# H×W instance mask. Keeping only this small slice makes transparent cutouts
# possible without retaining a large tracker result or doing PNG work on the
# capture thread. It is excluded from dataclass equality because numpy arrays do
# not have scalar equality semantics.
@dataclass(frozen=True)
class DetectionBox:
    tracker_id: int
    xyxy: tuple[float, float, float, float]
    resolving: bool
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    # Experimental (feat/matching-tests): the detector's raw per-box confidence
    # this frame, straight from the tracker's `sv.Detections.confidence` -- a
    # testing aid, not part of the frozen api-contract.
    confidence: float | None = None


class Latest(NamedTuple):
    overlay_jpeg: bytes            # encoded 1080p overlay for /stream
    present_ids: frozenset[int]    # tracker_ids — BUILTIN int (cast from the
                                   # tracker's np.int64) so /status JSON-encodes;
                                   # T04 must not have to sanitize (see AC12)
    count: int                     # len(present_ids)
    t: float                       # monotonic seconds at capture
    frame_bgr: np.ndarray          # the UN-annotated camera frame this snapshot
                                   # owns (a read-only frame.copy() — the camera
                                   # buffer may be reused; see D17). The source
                                   # for lazily-cropped /status thumbnails.
    detections: tuple[DetectionBox, ...]  # id/box/state + bbox-local mask per
                                   # detection, row-aligned with the same frame;
                                   # sorting is left to the consumer
    roster: frozenset[int]         # tracker.roster sampled the SAME tick as
                                   # present_ids (builtin ints) — T10/D8a
    catalog: frozenset[int]        # tracker.catalog sampled the SAME tick as
                                   # present_ids/roster (builtin ints). Constant
                                   # for the tracker's life; carried so /status
                                   # can report the expected set without touching
                                   # the tracker — T11/B1.
    match_debug: dict = {}         # Experimental (feat/matching-tests): tracker.
                                   # match_debug sampled the same tick — last
                                   # matcher score per emitted id, keyed to match
                                   # detections[].tracker_id / instrument ids.
                                   # Empty in fake mode (no real matcher).


OnFrame = Callable[
    [float, frozenset[int], frozenset[int], frozenset[int], frozenset[int]],
    None,
]
"""(t, present_ids, roster, catalog, resolving_ids) — all sampled the same tick."""


class TrackerResetError(RuntimeError):
    """Raised when the capture thread could not reset the tracker."""


@dataclass
class _TrackerCommand:
    """One queued capture-thread tracker mutation (T11/B5). ``confidence is None``
    means a plain reset; otherwise the confidence is applied and THEN the reset,
    together, on the capture thread between frames.

    Each command owns its OWN ``done`` Event and ``error`` slot, so the HTTP↔
    capture handshake is per-command rather than shared state. The capture thread
    applies a command ENTIRELY under ``_reset_lock``; a submitter that times out
    takes that same lock, so it either finds the command already applied (``done``
    set) — or, if the capture thread never reached it, marks it ``cancelled`` so
    the recovering thread drops it. A timed-out command therefore never mutates
    the tracker after its submitter has given up: its confidence can't leak into a
    later plain reset, and it can't fire a reset mid-recording."""

    confidence: float | None
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: bool = False     # a timed-out submitter abandoned it (drop on recovery)
    error: BaseException | None = None


class CaptureLoop:
    def __init__(
        self,
        tracker,
        camera_index: int,
        on_frame: OnFrame | None = None,
        cap_factory: Callable[[int], VideoCaptureLike] = cv2.VideoCapture,
        frame_size: tuple[int, int] = (1920, 1080),
        stale_after_s: float = 2.0,
        render_fn=None,
    ) -> None:
        self._tracker = tracker
        self._camera_index = camera_index
        self._on_frame = on_frame
        self._cap_factory = cap_factory
        self._frame_size = frame_size
        self._stale_after_s = stale_after_s
        self._render_fn = render_fn if render_fn is not None else OverlayRenderer()

        self._lock = threading.Lock()
        self._stop = threading.Event()
        # `_reset_request` is a cheap "there may be a command pending" flag the
        # capture loop checks without taking a lock every tick; `_pending_command`
        # is the actual queued mutation, and both are only ever mutated under
        # `_reset_lock` so a claim/cancel race resolves consistently (T11/B5).
        self._reset_request = threading.Event()
        self._reset_lock = threading.Lock()
        # Public command calls serialize here for their entire submit/wait
        # lifecycle. The one-slot handoff below can therefore never be overwritten
        # by a second caller, even when CaptureLoop is used outside create_app.
        self._command_submission_lock = threading.Lock()
        self._pending_command: _TrackerCommand | None = None

        self._cap: VideoCaptureLike | None = None
        self._thread: threading.Thread | None = None
        self._latest: Latest | None = None
        self._generation = 0
        self._start_time = 0.0
        self._last_publish: float | None = None
        self._forced_dead = False

    def set_on_frame(self, cb: OnFrame | None) -> None:
        """Register (or clear) the per-publication callback. Public so wiring
        code (T04's `create_app`) never has to poke the private attribute."""
        self._on_frame = cb

    def set_render_fn(self, render_fn) -> None:
        """Register the per-frame overlay renderer. Public for the same reason
        as `set_on_frame`: `create_app` owns the stateless `OverlayRenderer`, so
        it needs a supported way to hand that object to the loop."""
        self._render_fn = render_fn

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("capture already running")
        cap = self._cap_factory(self._camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"capture {self._camera_index} would not open")
        width, height = self._frame_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._forced_dead = False
        self._start_time = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # The thread outran the join deadline — almost always wedged in
                # a blocking cv2 read(). A Python thread can't be killed, so do
                # the strongest achievable: mark the loop dead (AC9's observable
                # contract) and release the capture from HERE. Releasing the
                # device typically unblocks the stuck read so the thread can
                # then exit on its own. A release exception must not propagate.
                self._forced_dead = True
                if self._cap is not None:
                    try:
                        self._cap.release()
                    except Exception:
                        logger.exception("capture release during stop() failed")

    def reset_tracker(self, timeout_s: float = 2.0) -> None:
        """Reset the tracker on the capture thread, between frames, returning only
        once it is applied. A no-op before start(); TrackerResetError if the reset
        itself raised.

        `timeout_s` bounds the wait for the capture thread to PICK UP the command
        (the common stall: the thread wedged in `cap.read()`), after which it is
        cancelled and TimeoutError is raised. It is deliberately NOT a hard bound
        once the capture thread has begun applying the command: a mutation already
        in progress cannot be safely abandoned (it would then fire against the
        tracker after this call returned — e.g. a reset landing mid-recording), so
        this call instead waits for it to finish. `tracker.reset()` is a fast
        in-memory operation, so this is bounded in practice; only a genuinely hung
        reset would block longer (and would already have killed the capture loop)."""
        self._submit_tracker_command(None, timeout_s)

    def set_confidence_and_reset(
        self, confidence: float, timeout_s: float = 2.0
    ) -> None:
        """Atomically set the detector confidence and reset the tracker, applied
        together on the CAPTURE thread between frames (never mutate tracker.confidence
        from the HTTP thread mid-update). Serialised with reset_tracker() via the
        same command queue, so two tracker mutations can never overlap. `timeout_s`
        has the same pick-up-bounded semantics as `reset_tracker` (see there)."""
        self._submit_tracker_command(confidence, timeout_s)

    def _submit_tracker_command(
        self, confidence: float | None, timeout_s: float
    ) -> None:
        """Queue one tracker mutation (a plain reset when ``confidence`` is None,
        else set-confidence-then-reset) for the capture thread and block until it
        is applied. Only one command is in flight at a time (main.py's
        mutation_lock also serialises callers).

        On a capture-thread STALL the wait times out; the submitter then takes
        `_reset_lock` (the same lock the capture thread applies a command under).
        If the command has NOT been applied — the capture thread is stuck before
        `_apply_pending_reset` — cancel it, so it is dropped, never applied, when
        the thread recovers, and raise. If it is already being applied, the
        capture thread holds `_reset_lock`, so acquiring it here waits for that
        mutation to finish: the submitter then returns its real result instead of
        abandoning a live command. Either way a timed-out command never mutates
        the tracker after the submitter has given up — the fix for the
        shared-handshake bug (no confidence leak, no reset mid-recording)."""
        with self._command_submission_lock:
            self._submit_serial_tracker_command(confidence, timeout_s)

    def _submit_serial_tracker_command(
        self, confidence: float | None, timeout_s: float
    ) -> None:
        """Submit one command while `_command_submission_lock` gives this caller
        exclusive ownership of the single pending slot."""
        thread = self._thread
        if thread is None:
            # Not running yet: apply a confidence change directly so a later
            # start() uses it; a plain reset has nothing to do.
            if confidence is not None:
                self._tracker.confidence = confidence
            return
        if not thread.is_alive():
            raise TrackerResetError("capture thread is not running")
        command = _TrackerCommand(confidence=confidence)
        with self._reset_lock:
            self._pending_command = command
            self._reset_request.set()
        if not command.done.wait(timeout_s):
            # Acquiring `_reset_lock` serialises us against an in-progress apply
            # (the capture thread holds it for the whole mutation). By the time we
            # hold it, the command is either applied (`done` set) or was never
            # reached (still the pending slot) — never half-applied-and-abandoned.
            with self._reset_lock:
                if not command.done.is_set() and self._pending_command is command:
                    command.cancelled = True
                    self._pending_command = None
                    self._reset_request.clear()
                    raise TimeoutError(
                        "tracker command was not applied (capture thread stalled)"
                    )
                # else: it completed while we were timing out / taking the lock —
                # fall through to report its real outcome.
        if command.error is not None:
            raise TrackerResetError("tracker command failed") from command.error

    def snapshot(self) -> Latest | None:
        with self._lock:
            return self._latest

    @property
    def health(self) -> str:
        if self._forced_dead:
            return "dead"
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
        # Apply the WHOLE mutation under `_reset_lock`, so a timing-out submitter
        # that takes the same lock either sees it fully applied (`done` set) or,
        # if we have not started, can cancel it — never a half-applied command
        # that a recovering thread finishes after its submitter gave up.
        fatal: BaseException | None = None
        with self._reset_lock:
            command = self._pending_command
            if command is None or command.cancelled:
                # Nothing to apply, or a timed-out submitter abandoned it. Drop it
                # untouched — a cancelled command must never mutate the tracker.
                self._pending_command = None
                self._reset_request.clear()
                return
            try:
                # Apply a queued confidence change BEFORE the reset, so the fresh
                # tracker state is built at the new value; a plain reset carries
                # confidence None (pure reset).
                previous_confidence = None
                if command.confidence is not None:
                    previous_confidence = self._tracker.confidence
                    self._tracker.confidence = command.confidence
                self._tracker.reset()
            except BaseException as exc:
                if command.confidence is not None and previous_confidence is not None:
                    try:
                        self._tracker.confidence = previous_confidence
                    except BaseException:
                        logger.exception("failed to roll back tracker confidence")
                command.error = exc
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    fatal = exc  # re-raise AFTER releasing the lock + signalling
                else:
                    logger.exception(
                        "tracker reset/confidence apply failed; capture loop will continue"
                    )
            finally:
                self._pending_command = None
                self._reset_request.clear()
                command.done.set()
        if fatal is not None:
            raise fatal

    def _capture_tick(self, cap: VideoCaptureLike) -> bool:
        """Read → update → render → encode → publish one frame. Returns True on
        a publication, False on a failed read. Raises to signal a skipped tick
        (caught by `_run`); a BaseException propagates to kill the thread."""
        ok, frame = cap.read()
        if not ok:
            return False

        t = time.monotonic()
        dets = self._tracker.update(frame)
        # Sample the roster AND catalog in the SAME tick as the detections
        # (T10/D8a, T11/B1): render, snapshot, session and /status all then judge
        # one instant. Builtin ints, like present_ids, so /status JSON-encodes
        # without sanitizing (AC12). catalog is constant for the tracker's life
        # but sampled here so nothing downstream has to reach into the tracker.
        roster = frozenset(int(session_id) for session_id in self._tracker.roster)
        catalog = frozenset(int(catalog_id) for catalog_id in self._tracker.catalog)
        # Experimental (feat/matching-tests): sampled the same tick, same reason
        # as roster/catalog above. `getattr` tolerates a tracker built before
        # this seam widened (never raises the capture thread).
        match_debug = dict(getattr(self._tracker, "match_debug", {}) or {})
        overlay = self._render_fn(frame.copy(), dets, roster, catalog, t)
        _, buffer = cv2.imencode(".jpg", overlay)
        tracker_ids = _tracker_ids(dets)
        present_ids = frozenset(int(tracker_id) for tracker_id in tracker_ids)
        # Row-aligned resolving flags (safe default False on a missing key).
        # resolving_ids is the present ids still being recognised — readiness
        # counts them as neither known nor unknown yet (T11/B1).
        resolving_flags = _resolving_flags(dets)
        resolving_ids = frozenset(
            int(tracker_id)
            for tracker_id, flag in zip(tracker_ids, resolving_flags)
            if flag
        )

        # Own a clean (un-annotated) copy of the frame for lazy thumbnail crops
        # and mark it read-only: the snapshot is shared, immutable material for
        # any number of concurrent /status readers (D17). `render` mutated its
        # OWN copy above, so this copy is pristine.
        frame_bgr = frame.copy()
        frame_bgr.setflags(write=False)

        latest = Latest(
            overlay_jpeg=buffer.tobytes(),
            present_ids=present_ids,
            count=len(present_ids),
            t=t,
            frame_bgr=frame_bgr,
            detections=_detection_boxes(dets),
            roster=roster,
            catalog=catalog,
            match_debug=match_debug,
        )
        with self._lock:
            self._latest = latest
            self._generation += 1
            self._last_publish = time.monotonic()

        self._notify(t, present_ids, roster, catalog, resolving_ids)
        return True

    def _notify(
        self,
        t: float,
        present_ids: frozenset[int],
        roster: frozenset[int],
        catalog: frozenset[int],
        resolving_ids: frozenset[int],
    ) -> None:
        if self._on_frame is None:
            return
        try:
            self._on_frame(t, present_ids, roster, catalog, resolving_ids)
        except Exception:
            # AC8: an on_frame exception must not kill the loop. The publication
            # already happened, so this frame still counts.
            logger.exception("on_frame callback raised; continuing")


def _tracker_ids(dets):
    return dets.tracker_id if dets.tracker_id is not None else ()


def _resolving_flags(dets) -> list[bool]:
    """Row-aligned resolving flags from `dets.data["resolving"]`, defaulting
    every row to False when the tracker omits the key — the safe default
    (model/docs/tracker-interface.md). Plain Python bools, numpy-free (AC12)."""
    flags = dets.data.get("resolving") if dets.data is not None else None
    n = len(dets.tracker_id) if dets.tracker_id is not None else 0
    if flags is None:
        return [False] * n
    return [bool(f) for f in flags]


def _detection_boxes(dets) -> tuple[DetectionBox, ...]:
    """Build row-aligned preview material without retaining full-frame masks."""
    if dets.tracker_id is None or dets.xyxy is None:
        return ()
    flags = _resolving_flags(dets)
    masks = dets.mask
    confidences = dets.confidence
    return tuple(
        DetectionBox(
            tracker_id=int(tracker_id),
            xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            resolving=bool(flag),
            mask=_local_mask(masks[index], box) if masks is not None else None,
            confidence=float(confidences[index]) if confidences is not None else None,
        )
        for index, (tracker_id, box, flag) in enumerate(
            zip(dets.tracker_id, dets.xyxy, flags)
        )
    )


def _local_mask(mask: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """Own the bbox-local part of one full-frame mask for lazy PNG encoding."""
    if mask.ndim != 2:
        return None
    height, width = mask.shape
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    left = max(0, int(math.floor(min(x1, x2))))
    top = max(0, int(math.floor(min(y1, y2))))
    right = min(width, int(math.ceil(max(x1, x2))))
    bottom = min(height, int(math.ceil(max(y1, y2))))
    if right <= left or bottom <= top:
        return None
    local = np.ascontiguousarray(mask[top:bottom, left:right], dtype=bool)
    local.setflags(write=False)
    return local
