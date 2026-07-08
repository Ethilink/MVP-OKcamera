"""TR1 AC1-AC9: behavior contract for CaptureLoop's recording mode.

Written from ``docs/tasks/TR1-capture-recording.md`` (the frozen SPEC) and
``docs/RECORDING.md`` §Runtime / §The frame-number contract / §Detector
sharing. Blind-TDD: this file is written against the SPEC's behavior
contract, not against any particular implementation — the coder replaces
``backend/capture.py``'s Phase-0 stubs without ever seeing this file.

Fakes only — no real camera, encoder, ffmpeg, or model weights. Reuses T01's
``FakeCapture`` / ``FakeDetector`` (via the shared fixtures in
``conftest.py``) and this task's own ``tests/recording_fakes.py``
(``FakeEncoder`` / ``make_numbered_frames`` / ``decode_frame_index``).

Threading discipline mirrors T03's ``tests/test_capture.py`` (untouched —
the idle-mode regression gate):
- every started ``CaptureLoop`` is stopped via the ``running_loop``
  contextmanager below, so ``.stop()`` runs even if an assertion fails.
- waiting for the background reader/inference threads to reach a state is
  done via ``_wait_until``, a small bounded-poll helper — never a bare or
  unbounded ``while``. ``_collect_until`` is the same idea extended to also
  *sample* intermediate state while waiting (AC4 needs to observe the set of
  distinct published frame_numbers over a window, not just a final state).

Numbered-frame offset note (AC2/AC3): ``CaptureLoop.start()`` performs a
startup validation read, and the background reader thread begins ticking as
soon as it is spawned — both happen in *idle* mode, before a test's
``start_recording()`` call takes effect, and both may consume frames from
the front of a fed sequence depending on scheduling. Rather than assume the
first frame written to the encoder is literally raw index 0 (which would
make these tests flaky under real thread scheduling), tests below verify
frame sequencing *relative to the first frame actually written this
recording* — i.e. exact contiguity (no gaps/dupes/reordering), which is the
substance of the frame-number contract, without depending on how many idle
ticks happened to run first.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest
import supervision as sv

from backend.capture import CaptureLoop
from tests.recording_fakes import FakeEncoder, decode_frame_index, make_numbered_frames

_POLL_INTERVAL_S = 0.005
_POLL_TIMEOUT_S = 2.0

# Pacing for ``_PacedCapture`` below: a real ``cv2.VideoCapture.read()`` blocks
# roughly 1/fps per call (~16ms at 60fps) and always returns a fresh frame.
# 1.5ms is far faster than any real camera but still slow enough, relative to
# pure-Python loop speed, to let a co-scheduled inference thread get GIL turns
# between reads (see ``_PacedCapture`` docstring for why this matters).
_CAMERA_READ_DELAY_S = 0.0015


def _wait_until(predicate, timeout_s: float = _POLL_TIMEOUT_S, interval_s: float = _POLL_INTERVAL_S):
    """Poll ``predicate()`` until truthy or ``timeout_s`` elapses.

    Never hangs: raises a clear AssertionError naming the timeout instead of
    spinning forever, so a stuck loop fails the test fast rather than
    hanging the suite. (Mirrors T03's ``tests/test_capture.py`` helper.)
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    pytest.fail(f"condition not met within {timeout_s}s")


def _collect_until(collector, predicate, timeout_s: float = _POLL_TIMEOUT_S, interval_s: float = _POLL_INTERVAL_S):
    """Like ``_wait_until``, but also calls ``collector()`` on every poll
    (including the first, before checking ``predicate()``), so a caller can
    sample intermediate state while waiting for a bounded condition — e.g.
    the set of distinct ``frame_number`` values published so far, which
    ``_wait_until`` alone (checks-then-discards) cannot observe. Still
    bounded: fails clearly rather than spinning forever.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        collector()
        if predicate():
            return
        time.sleep(interval_s)
    pytest.fail(f"condition not met within {timeout_s}s")


def _identity_render(frame, dets):
    return frame


@contextmanager
def running_loop(*args, **kwargs):
    """Build+start a CaptureLoop and guarantee .stop() even on failure."""
    loop = CaptureLoop(*args, **kwargs)
    loop.start()
    try:
        yield loop
    finally:
        loop.stop()


class _SlowDetector:
    """A detector whose ``predict()`` sleeps briefly so the reader (which,
    against a fake camera, runs about as fast as Python can loop) outruns
    inference and skips frames — the setup AC3/AC4 need to prove the
    published ``frame_number`` tracks what inference actually ran on, not a
    newest-at-publish counter.

    Deliberately NOT a ``FakeDetector`` subclass: it only needs to satisfy
    the detector duck-type the loop depends on (``confidence_threshold``
    attribute, ``predict(frame, confidence_threshold=None)``), and keeping it
    minimal avoids coupling this test to ``FakeDetector``'s internals beyond
    that contract.
    """

    def __init__(self, delay_s: float = 0.02, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold
        self.delay_s = delay_s
        self.seen_frames: list = []

    def predict(self, frame, confidence_threshold=None):
        time.sleep(self.delay_s)
        self.seen_frames.append(frame)
        return sv.Detections.empty()


class _ConcurrencyTrackingCapture:
    """Wraps a ``FakeCapture`` instance to detect overlapping ``read()``
    calls — i.e. two threads inside ``read()`` at the same time — which is
    exactly what AC9 (single-reader semantics) forbids.

    Not a subclass: composes an inner ``FakeCapture`` and forwards every
    other method, so it stays a thin, purpose-built detector rather than
    inheriting (and risking silently relying on) FakeCapture's internals.
    ``read()`` tracks an "active caller" count under its OWN lock —
    independent of any locking inside the wrapped fake — and widens the race
    window with a tiny sleep while a call is "in flight", so a genuine
    double-read is far more likely to be observed than missed.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._concurrency_lock = threading.Lock()
        self._active_readers = 0
        self.max_concurrent_readers = 0

    def read(self):
        with self._concurrency_lock:
            self._active_readers += 1
            self.max_concurrent_readers = max(self.max_concurrent_readers, self._active_readers)
        try:
            time.sleep(0.001)  # widen the window so a real race would show up
            return self._inner.read()
        finally:
            with self._concurrency_lock:
                self._active_readers -= 1

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _PacedCapture:
    """Wraps a ``FakeCapture`` to model a REAL ``cv2.VideoCapture.read()``
    instead of an unrealistic instantaneous one.

    ``FakeCapture.read()`` returns immediately and, once its programmed frame
    queue drains, silently repeats the last frame forever
    (``on_empty="repeat"``). Driving the recording reader thread against that
    unpaced fake makes it busy-loop at millions of reads/sec, which:

    1. GIL-starves whatever else is running (the co-scheduled inference
       thread never gets a turn to publish a ``Latest``), and
    2. outruns any finite frame queue almost instantly, so the drained
       camera hands back a duplicate frame object — something a real camera
       can never do (every ``read()`` blocks until a genuinely new frame is
       available).

    A real camera instead blocks ~1/fps per ``read()`` call and always
    returns a fresh, distinct frame. This wrapper models that: it sleeps a
    small fixed delay per call (yielding the GIL, so other threads make
    progress) and simply forwards to the wrapped ``FakeCapture`` — which, as
    long as it is fed a frame pool large enough that the paced reader cannot
    drain it within a test's timeout window, never needs to fall back to
    repeating a frame either. Composes rather than subclasses (same pattern
    as ``_ConcurrencyTrackingCapture`` above) so it stays a thin, purpose-built
    wrapper instead of depending on ``FakeCapture``'s internals.
    """

    def __init__(self, inner, delay_s: float = _CAMERA_READ_DELAY_S) -> None:
        self._inner = inner
        self._delay_s = delay_s

    def read(self):
        time.sleep(self._delay_s)
        return self._inner.read()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _public_state_snapshot(loop) -> dict:
    """Best-effort snapshot of every non-callable, non-underscore attribute
    on ``loop`` beyond the properties this suite already asserts on
    elsewhere. Used only by the AC8 test: the SPEC explicitly leaves the
    encoder-failure-surfacing *mechanism* open ("propagate on stop, or a
    recording-error flag"), so rather than guess a flag's name, this diffs
    the loop's whole public surface before/after the failure to detect
    ANY new or changed observable state generically.
    """
    known = {"is_recording", "frames_written", "inference_paused", "camera_index", "generation", "health"}
    snap = {}
    for name in dir(loop):
        if name.startswith("_") or name in known:
            continue
        try:
            value = getattr(loop, name)
        except Exception:
            continue
        if callable(value):
            continue
        snap[name] = value
    return snap


# ---------------------------------------------------------------------------
# AC1 (idle regression, focused) — Latest.frame_number is None when idle, and
# after a record -> stop cycle the loop is observably back to idle. The bulk
# of AC1 is defended by the untouched T03 test_capture.py + T05 test_api.py
# suites (run separately, not duplicated here).
# ---------------------------------------------------------------------------


def test_ac01_frame_number_is_none_when_idle(fake_frame, FakeDetector, FakeCapture):
    frame = fake_frame(64, 48)
    cap = FakeCapture(frames=[frame])
    detector = FakeDetector()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        _wait_until(lambda: loop.generation > 0)
        snap = loop.snapshot()

    assert snap is not None
    assert snap.frame_number is None


def test_ac01_record_then_stop_returns_to_idle(FakeDetector, FakeCapture):
    frames = make_numbered_frames(40, 64, 48)
    cap = FakeCapture(frames=frames)
    detector = FakeDetector()
    encoder = FakeEncoder()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        _wait_until(lambda: loop.generation > 0)

        loop.start_recording(encoder)
        _wait_until(lambda: loop.frames_written >= 5)
        loop.stop_recording()

        assert loop.is_recording is False
        assert loop.frames_written == 0

        # Idle snapshots keep publishing after the recording ends, and the
        # published frame_number goes back to None (the idle-mode contract).
        gen_before = loop.generation
        _wait_until(lambda: loop.generation > gen_before)
        snap = loop.snapshot()

    assert snap is not None
    assert snap.frame_number is None


# ---------------------------------------------------------------------------
# AC2 — frame-number contract: the encoder receives frames in order, no
# gaps/dupes, one write per frame.
# ---------------------------------------------------------------------------


def test_ac02_encoder_receives_frames_in_order_no_gaps_no_dupes(FakeDetector, FakeCapture):
    # Frame pool sized well beyond what a ~1.5ms-paced reader can drain within
    # this test's (short) run, so the underlying FakeCapture's on_empty="repeat"
    # fallback is never actually exercised — a repeat here would inject a
    # duplicate frame a real camera could never produce (see _PacedCapture).
    frames = make_numbered_frames(6000, 64, 48)
    cap = _PacedCapture(FakeCapture(frames=frames))
    detector = FakeDetector()
    encoder = FakeEncoder()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        loop.start_recording(encoder)
        _wait_until(lambda: loop.frames_written >= 30)
        total_written = loop.stop_recording()

    written = encoder.written
    assert len(written) == total_written
    assert total_written >= 30

    indices = [decode_frame_index(f) for f in written]
    # Strictly increasing by exactly 1, position-for-position: no gaps, no
    # dupes, no reordering (see module docstring re: why this is relative to
    # indices[0] rather than a hard-coded 0).
    assert indices == list(range(indices[0], indices[0] + len(indices)))
    # The i-th write's own position IS its recording-local frame_number by
    # construction (list index) — assert that position corresponds 1:1 to a
    # unique raw frame (no duplicate content at two different write
    # positions), reinforcing "no dupes" independently of the arithmetic
    # check above.
    assert len(set(indices)) == len(indices)


# ---------------------------------------------------------------------------
# AC3 — the published Latest.frame_number matches the frame inference
# actually ran on (recoverable from that frame's own pixels), not a
# newest-at-publish counter. Proven with a slow detector so the reader
# outruns inference.
# ---------------------------------------------------------------------------


def test_ac03_published_frame_number_matches_frame_actually_detected_on(FakeCapture):
    # A slow detector only paces INFERENCE; the reader still needs a paced,
    # non-repeating camera (see _PacedCapture) so it doesn't busy-loop and
    # GIL-starve the inference thread out of ever publishing. Pool sized well
    # beyond what ~1.5ms/read can drain inside this test's 5s wait window.
    frames = make_numbered_frames(10000, 64, 48)
    cap = _PacedCapture(FakeCapture(frames=frames))
    detector = _SlowDetector(delay_s=0.03)
    encoder = FakeEncoder()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        loop.start_recording(encoder)
        # Let the reader race far ahead of the slow detector.
        _wait_until(lambda: loop.frames_written >= 50, timeout_s=5.0)

        snap = loop.snapshot()
        frames_written_at_check = loop.frames_written
        loop.stop_recording()

    assert snap is not None
    assert snap.frame_number is not None

    written = encoder.written
    assert len(written) > 0
    base_index = decode_frame_index(written[0])  # this recording's frame 0
    content_index = decode_frame_index(snap.frame)
    expected_frame_number = content_index - base_index

    # The published frame_number is derived from the SAME content the
    # snapshot's own frame carries — i.e. it names the frame inference
    # actually detected on.
    assert snap.frame_number == expected_frame_number
    # And it lags the reader's frontier — proving inference did NOT just
    # report "whatever the newest write counter says right now" (a
    # newest-at-publish counter would have raced ahead to
    # frames_written_at_check - 1, not sat behind it under a slow detector).
    assert snap.frame_number < frames_written_at_check


# ---------------------------------------------------------------------------
# AC4 — inference skips frames freely: frames_written (reader rate) exceeds
# the number of distinct published frame_numbers (detector rate) under a
# slow detector.
# ---------------------------------------------------------------------------


def test_ac04_frames_written_exceeds_distinct_published_frame_numbers(FakeCapture):
    # Same rationale as AC3: paced, non-repeating camera so the reader can't
    # GIL-starve inference into never publishing, sized for this test's 5s
    # wait window.
    frames = make_numbered_frames(10000, 64, 48)
    cap = _PacedCapture(FakeCapture(frames=frames))
    detector = _SlowDetector(delay_s=0.02)
    encoder = FakeEncoder()

    seen_frame_numbers: set[int] = set()

    def sample():
        snap = loop.snapshot()
        if snap is not None and snap.frame_number is not None:
            seen_frame_numbers.add(snap.frame_number)

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        loop.start_recording(encoder)
        _collect_until(sample, lambda: loop.frames_written >= 80, timeout_s=5.0)
        frames_written = loop.frames_written
        loop.stop_recording()

    assert frames_written >= 80
    assert len(seen_frame_numbers) > 0
    # The reader wrote every frame at full rate; the slow detector only ever
    # displayed a subset of them.
    assert frames_written > len(seen_frame_numbers)


# ---------------------------------------------------------------------------
# AC5 — stop_recording() returns the exact frame count, release() happens
# exactly once and only after the last write, and idle state is restored.
# ---------------------------------------------------------------------------


def test_ac05_stop_recording_releases_once_after_last_write_and_resets(FakeDetector, FakeCapture):
    frames = make_numbered_frames(60, 64, 48)
    cap = FakeCapture(frames=frames)
    detector = FakeDetector()
    encoder = FakeEncoder()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        loop.start_recording(encoder)
        _wait_until(lambda: loop.frames_written >= 15)
        result = loop.stop_recording()

        assert loop.is_recording is False
        assert loop.frames_written == 0

    # The return value is the authoritative count of frames actually written
    # — not a value grabbed racily before stop_recording() was called.
    assert result == len(encoder.written)
    assert result >= 15

    release_events = [e for e in encoder.events if e[0] == "release"]
    assert len(release_events) == 1

    release_index = encoder.events.index(("release",))
    # Nothing after release() in the log...
    assert encoder.events[release_index + 1 :] == []
    # ...and everything before it was a successful write (no failed writes,
    # no other event types, in this happy-path scenario).
    assert all(e[0] == "write" for e in encoder.events[:release_index])
    assert len(encoder.events[:release_index]) == result


# ---------------------------------------------------------------------------
# AC6 — start_recording while already recording -> RuntimeError;
# stop_recording while not recording -> RuntimeError.
# ---------------------------------------------------------------------------


def test_ac06_start_recording_while_already_recording_raises(FakeDetector, FakeCapture):
    frames = make_numbered_frames(30, 64, 48)
    cap = FakeCapture(frames=frames)
    detector = FakeDetector()
    encoder_a = FakeEncoder()
    encoder_b = FakeEncoder()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        loop.start_recording(encoder_a)
        _wait_until(lambda: loop.frames_written >= 1)

        with pytest.raises(RuntimeError):
            loop.start_recording(encoder_b)

        loop.stop_recording()


def test_ac06_stop_recording_while_not_recording_raises(FakeDetector, FakeCapture):
    frame_list = make_numbered_frames(5, 64, 48)
    cap = FakeCapture(frames=frame_list)
    detector = FakeDetector()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        _wait_until(lambda: loop.generation > 0)

        with pytest.raises(RuntimeError):
            loop.stop_recording()


# ---------------------------------------------------------------------------
# AC7 — pause_inference() stops detector.predict calls while the reader
# stays alive and health stays "ok"; resume_inference() restores predict
# calls. Works in idle mode.
# ---------------------------------------------------------------------------


def test_ac07_pause_inference_stops_predict_calls_in_idle_mode(fake_frame, FakeDetector, FakeCapture):
    frame = fake_frame(64, 48)
    cap = FakeCapture(frames=[frame])
    detector = FakeDetector()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        _wait_until(lambda: len(detector.seen_frames) > 0)

        loop.pause_inference()
        assert loop.inference_paused is True

        seen_before = len(detector.seen_frames)
        reads_before = cap.read_count

        # Proving an ABSENCE of predict() calls needs a bounded observation
        # window (there is no "it happened" event to poll for) — mirrors
        # T03's AC7 tight-spin-bound check, which uses the same pattern for
        # the same reason.
        time.sleep(0.2)

        # The reader keeps reading from the camera (alive, not wedged)...
        assert cap.read_count > reads_before
        # ...but the detector was never called again while paused.
        assert len(detector.seen_frames) == seen_before
        assert loop.health == "ok"

        loop.resume_inference()
        assert loop.inference_paused is False

        _wait_until(lambda: len(detector.seen_frames) > seen_before)


# ---------------------------------------------------------------------------
# AC8 — encoder.write() raising must not crash the loop or silently wedge
# camera health; the failure must be observable (mechanism left open by the
# SPEC: stop_recording() raising, or some recording-error flag).
# ---------------------------------------------------------------------------


def test_ac08_encoder_write_failure_is_surfaced_not_silent(FakeDetector, FakeCapture):
    frames = make_numbered_frames(40, 64, 48)
    cap = FakeCapture(frames=frames)
    detector = FakeDetector()
    encoder = FakeEncoder(raise_on_write=True)

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        before = _public_state_snapshot(loop)

        loop.start_recording(encoder)

        # Give the reader a bounded window to attempt (and fail) at least
        # one write.
        _wait_until(lambda: len(encoder.events) > 0, timeout_s=2.0)

        # The camera-read side must not have wedged/crashed just because
        # writes are failing.
        assert loop.health != "dead"

        raised_on_stop = False
        try:
            loop.stop_recording()
        except Exception:
            raised_on_stop = True

        after = _public_state_snapshot(loop)

    assert loop.health != "dead"
    # No frame was ever counted as successfully written — every write raised.
    assert len(encoder.written) == 0

    changed_or_new_flags = {k: v for k, v in after.items() if before.get(k) != v}
    observed = raised_on_stop or bool(changed_or_new_flags)
    assert observed, (
        "an encoder.write() failure must be surfaced — either stop_recording() "
        "raises, or the loop exposes an observable recording-error flag; "
        "neither was observed"
    )


# ---------------------------------------------------------------------------
# AC9 — single-reader semantics: only the reader thread calls cap.read()
# while recording (no concurrent double-read).
# ---------------------------------------------------------------------------


def test_ac09_single_reader_owns_camera_during_recording(FakeDetector, FakeCapture):
    frames = make_numbered_frames(200, 64, 48)
    inner_cap = FakeCapture(frames=frames)
    cap = _ConcurrencyTrackingCapture(inner_cap)
    detector = FakeDetector()
    encoder = FakeEncoder()

    with running_loop(
        detector, camera_index=0, render_fn=_identity_render, cap_factory=lambda idx: cap
    ) as loop:
        loop.start_recording(encoder)
        _wait_until(lambda: loop.frames_written >= 30, timeout_s=5.0)
        loop.stop_recording()

    assert inner_cap.read_count > 0
    assert cap.max_concurrent_readers == 1
