"""T03 AC1-AC10: behavior contract for backend.capture.CaptureLoop.

Blind-TDD test-writer pass — written from the T03 spec (the SPEC-capture.md
scaffolding file, since deleted, plus DASHBOARD.md). Uses
the T01 fakes (FakeCapture/FakeDetector/fake_frame/fake_dets) so nothing here
touches a real camera or ONNX weights.

Threading discipline:
- every started CaptureLoop is stopped via the `running_loop` contextmanager
  below, which guarantees `.stop()` runs even if an assertion fails.
- waiting for the background thread to publish is done via `_wait_until`, a
  small bounded-poll helper (tiny sleep, hard timeout, clear failure message)
  — never a bare/unbounded `while`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import cv2
import numpy as np
import pytest

from backend.capture import CaptureLoop

_POLL_INTERVAL_S = 0.005
_POLL_TIMEOUT_S = 2.0


def _wait_until(predicate, timeout_s: float = _POLL_TIMEOUT_S, interval_s: float = _POLL_INTERVAL_S):
    """Poll ``predicate()`` until truthy or ``timeout_s`` elapses.

    Never hangs: raises a clear AssertionError naming the timeout instead of
    spinning forever, so a stuck loop fails the test fast rather than hanging
    the suite.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
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


# ---------------------------------------------------------------------------
# AC1 — start() requests the target resolution + BUFFERSIZE=1 via cap.set(),
# in the fake's log
# ---------------------------------------------------------------------------


def test_ac01_start_requests_target_res_and_shallow_buffer(fake_frame, FakeDetector, FakeCapture):
    frame = fake_frame(1920, 1080)
    cap = FakeCapture(frames=[frame])
    detector = FakeDetector()

    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    ) as loop:
        _wait_until(lambda: loop.generation > 0)

    assert (cv2.CAP_PROP_FRAME_WIDTH, 1920) in cap.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 1080) in cap.set_calls
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in cap.set_calls


# ---------------------------------------------------------------------------
# AC2 — a camera that yields no valid first frame -> start() raises
# RuntimeError naming the camera index (the pipeline is resolution-agnostic, so
# there is no resolution to mismatch — the only hard failure is a dead camera)
# ---------------------------------------------------------------------------


def test_ac02_no_valid_first_frame_raises_naming_index(
    fake_frame, FakeDetector, FakeCapture
):
    # A camera that never returns a frame (fail_reads) is the failure start()
    # must catch: no valid first frame at all.
    cap = FakeCapture(frames=[], fail_reads=True)
    detector = FakeDetector()

    loop = CaptureLoop(
        detector,
        camera_index=3,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    )

    with pytest.raises(RuntimeError) as exc_info:
        loop.start()

    message = str(exc_info.value)
    assert "3" in message  # the camera index is named so the operator can act


# ---------------------------------------------------------------------------
# AC3 — detector sees the captured frame as-is (no downscale) and
# snapshot().frame is that same array — one resolution end to end
# ---------------------------------------------------------------------------


def test_ac03_detector_and_snapshot_share_the_captured_frame(fake_frame, FakeDetector, FakeCapture):
    captured = fake_frame(1920, 1080)
    cap = FakeCapture(frames=[captured])
    detector = FakeDetector()

    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    ) as loop:
        _wait_until(lambda: loop.generation > 0)
        _wait_until(lambda: len(detector.seen_frames) > 0)

        snap = loop.snapshot()
        assert snap is not None

    seen = detector.seen_frames[0]
    assert seen.shape == (1080, 1920, 3)  # detected at native res, not downscaled

    assert snap.frame.shape == (1080, 1920, 3)
    assert np.array_equal(snap.frame, captured)


# ---------------------------------------------------------------------------
# AC4 — threshold in the published Latest matches the detector's value AT
# PREDICT TIME for that tick, not a value mutated later
# ---------------------------------------------------------------------------


def test_ac04_threshold_matches_value_at_predict_time(fake_frame, fake_dets, FakeDetector, FakeCapture):
    full_frame = fake_frame(1920, 1080)
    dets_a = fake_dets([[10, 10, 50, 50]], size=(1920, 1080))
    dets_b = fake_dets([[20, 20, 80, 80]], size=(1920, 1080))

    detector = FakeDetector(predictions=[dets_a], confidence_threshold=0.3)
    # The loop free-runs at detector speed (no artificial sleep on the success
    # path per the SPEC), so many ticks can land between two lines of test
    # code. We must not assume a fixed tick index sees our flip — instead we
    # poll the published snapshot until it observably reflects the new value.
    cap = FakeCapture(frames=[full_frame])

    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    ) as loop:
        # Every tick so far used the un-flipped threshold.
        _wait_until(lambda: loop.generation >= 1)
        first_snap = loop.snapshot()
        assert first_snap.threshold == pytest.approx(0.3)

        # Flip the threshold and queue a second detection set; wait for a
        # later Latest whose predict-time threshold reflects the new value.
        detector.confidence_threshold = 0.9
        detector.queue(dets_b)
        _wait_until(lambda: loop.snapshot().threshold == pytest.approx(0.9))
        second_snap = loop.snapshot()

    # The published snapshot after the flip carries the new threshold, not
    # the stale pre-flip one — and every threshold the detector recorded
    # after the flip agrees (never a stale/garbled in-between value).
    assert second_snap.threshold == pytest.approx(0.9)
    assert all(
        t == pytest.approx(0.3) or t == pytest.approx(0.9) for t in detector.seen_thresholds
    )


# ---------------------------------------------------------------------------
# AC5 — render_fn receives a COPY of the frame; mutating it in place
# must not change anything reachable from snapshot() (frame or a later
# snapshot's dets/masks)
# ---------------------------------------------------------------------------


def test_ac05_render_fn_receives_copy_mutation_is_isolated(
    fake_frame, fake_dets, FakeDetector, FakeCapture
):
    full_frame = fake_frame(1920, 1080)
    dets = fake_dets([[10, 10, 50, 50]], size=(1920, 1080))
    detector = FakeDetector(predictions=[dets])
    cap = FakeCapture(frames=[full_frame])

    original_full_copy = full_frame.copy()

    def mutating_render(frame, dets):
        frame[:] = 0  # deliberately corrupt what we were handed
        return frame

    with running_loop(
        detector,
        camera_index=0,
        render_fn=mutating_render,
        cap_factory=lambda idx: cap,
    ) as loop:
        _wait_until(lambda: loop.generation > 0)
        snap = loop.snapshot()
        assert snap is not None

    # frame must be untouched by the render_fn mutation of the frame copy.
    assert np.array_equal(snap.frame, original_full_copy)
    # The FakeCapture's own stored frame must also be untouched (it's the
    # same array object fed into the loop across ticks via on_empty=repeat).
    assert np.array_equal(full_frame, original_full_copy)


# ---------------------------------------------------------------------------
# AC6 — each tick rebinds a brand-new Latest (never mutates the previous one
# in place); generation strictly increases across ticks
# ---------------------------------------------------------------------------


def test_ac06_each_tick_rebinds_new_latest_generation_increases(
    fake_frame, FakeDetector, FakeCapture
):
    full_frame = fake_frame(1920, 1080)
    # No detections programmed: every predict() call returns the empty set,
    # so `count` stays deterministic (0) across ticks regardless of how many
    # ticks the free-running loop squeezes in — this test is only about
    # object identity and generation monotonicity, not detection content.
    detector = FakeDetector()
    cap = FakeCapture(frames=[full_frame])  # repeats -> loop keeps ticking

    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    ) as loop:
        _wait_until(lambda: loop.generation > 0)
        first_gen = loop.generation
        first_snap = loop.snapshot()

        _wait_until(lambda: loop.generation > first_gen)
        second_gen = loop.generation
        second_snap = loop.snapshot()

    assert second_gen > first_gen
    # A new Latest object was published, not a mutation of the same one.
    assert second_snap is not first_snap
    assert first_snap.count == 0
    assert second_snap.count == 0


# ---------------------------------------------------------------------------
# AC7 — failed reads (camera gone) flip health to "stale" after
# stale_after_s, thread stays alive, reads are bounded (no tight-spin); once
# reads succeed again health returns to "ok"
# ---------------------------------------------------------------------------


def test_ac07_failed_reads_go_stale_then_recover(fake_frame, FakeDetector, FakeCapture):
    full_frame = fake_frame(1920, 1080)
    # First frame must succeed so start()'s shape assertion passes; then the
    # capture starts failing every subsequent read.
    cap = FakeCapture(frames=[full_frame])
    detector = FakeDetector()
    stale_after_s = 0.2

    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
        stale_after_s=stale_after_s,
    ) as loop:
        _wait_until(lambda: loop.generation > 0)
        assert loop.health == "ok"

        cap.fail_reads = True
        # Wait past stale_after_s for the heartbeat to age out.
        _wait_until(lambda: loop.health == "stale", timeout_s=stale_after_s + 2.0)

        # Thread must still be alive while stale (not "dead").
        assert loop.health == "stale"

        reads_when_stale_declared = cap.read_count
        time.sleep(stale_after_s * 2)
        reads_after_more_waiting = cap.read_count

        # Bounded: failed reads must not tight-spin. A generous ceiling well
        # under "thousands of reads in a couple hundred ms" catches a busy
        # loop without pinning to a specific retry interval.
        assert reads_after_more_waiting - reads_when_stale_declared < 200

        # Recovery: once reads succeed again, health goes back to "ok".
        cap.fail_reads = False
        _wait_until(lambda: loop.health == "ok")


# ---------------------------------------------------------------------------
# AC8 — stop() joins within ~2s and is a clean exit (not "dead"); an
# abnormal exception escaping the loop body flips health to "dead"
# ---------------------------------------------------------------------------


def test_ac08_stop_joins_quickly_and_is_not_dead(fake_frame, FakeDetector, FakeCapture):
    full_frame = fake_frame(1920, 1080)
    cap = FakeCapture(frames=[full_frame])
    detector = FakeDetector()

    loop = CaptureLoop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    )
    loop.start()
    _wait_until(lambda: loop.generation > 0)

    started_at = time.monotonic()
    loop.stop()
    elapsed = time.monotonic() - started_at

    assert elapsed <= 2.0
    assert loop.health != "dead"
    assert cap.released is True


def test_ac08_abnormal_exception_in_loop_body_marks_dead(fake_frame, FakeCapture):
    full_frame = fake_frame(1920, 1080)
    cap = FakeCapture(frames=[full_frame])

    class ExplodingDetector:
        """Passes start()'s first-frame check, then blows up every predict()."""

        confidence_threshold = 0.5

        def predict(self, frame):
            raise ValueError("boom - simulated detector crash")

    loop = CaptureLoop(
        ExplodingDetector(),
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    )
    loop.start()
    try:
        _wait_until(lambda: loop.health == "dead")
    finally:
        # loop.stop() must be safe to call even on an already-dead thread so
        # this test never leaks a thread; it must not raise or hang.
        loop.stop()


# ---------------------------------------------------------------------------
# AC9 — set_camera(n) makes the loop thread reopen via cap_factory with the
# new index; the old capture is released
# ---------------------------------------------------------------------------


def test_ac09_set_camera_reopens_via_cap_factory_and_releases_old(
    fake_frame, FakeDetector, FakeCapture
):
    full_frame = fake_frame(1920, 1080)
    cap0 = FakeCapture(frames=[full_frame], index=0)
    cap1 = FakeCapture(frames=[full_frame], index=1)
    detector = FakeDetector()

    caps_by_index = {0: cap0, 1: cap1}
    opened_indices: list[int] = []

    def cap_factory(idx):
        opened_indices.append(idx)
        return caps_by_index[idx]

    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=cap_factory,
    ) as loop:
        _wait_until(lambda: loop.generation > 0)
        assert cap0.released is False

        loop.set_camera(1)

        # The reopen happens on the loop thread; wait for cap_factory(1) to
        # have actually been called and the new capture to start producing.
        _wait_until(lambda: 1 in opened_indices)
        _wait_until(lambda: cap0.released is True)
        _wait_until(lambda: cap1.read_count > 0)

    assert opened_indices == [0, 1]
    assert cap0.released is True
    # Re-applies WIDTH/HEIGHT/BUFFERSIZE on the new capture too.
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1920) in cap1.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 1080) in cap1.set_calls
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in cap1.set_calls


# ---------------------------------------------------------------------------
# AC10 — overlay_jpeg decodes (cv2.imdecode) to the frame dims
# ---------------------------------------------------------------------------


def test_ac10_overlay_jpeg_decodes_to_frame_dims(fake_frame, fake_dets, FakeDetector, FakeCapture):
    full_frame = fake_frame(1920, 1080)
    dets = fake_dets([[10, 10, 50, 50]], size=(1920, 1080))
    detector = FakeDetector(predictions=[dets])
    cap = FakeCapture(frames=[full_frame])

    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    ) as loop:
        _wait_until(lambda: loop.generation > 0)
        snap = loop.snapshot()

    assert snap is not None
    decoded = cv2.imdecode(np.frombuffer(snap.overlay_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (1080, 1920, 3)


# ---------------------------------------------------------------------------
# snapshot() before any publish -> None (part of the public contract, used
# by every other test implicitly; kept explicit here as its own case)
# ---------------------------------------------------------------------------


def test_snapshot_returns_none_before_first_publish(FakeDetector, FakeCapture, fake_frame):
    full_frame = fake_frame(1920, 1080)
    cap = FakeCapture(frames=[full_frame])
    detector = FakeDetector()

    loop = CaptureLoop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=lambda idx: cap,
    )

    # Never started: nothing has been published yet.
    assert loop.snapshot() is None


# ---------------------------------------------------------------------------
# Reopen-on-sustained-failure: a wedged/unplugged camera (a handle that keeps
# failing past reopen_after_s) is released and reopened, so a dropped stream
# recovers without a process restart. Retrying read() on a dead handle never
# comes back — on macOS AVFoundation the device must be reopened. (T07 AC6:
# "reconnect recovers".)
# ---------------------------------------------------------------------------


def test_reopen_on_sustained_failure_recovers(fake_frame, FakeDetector, FakeCapture):
    good = fake_frame(1920, 1080)
    # cap A satisfies start()'s first read, then fails every read after (the
    # stream that dropped). cap B is a healthy replacement the reopen picks up.
    cap_a = FakeCapture(frames=[good], on_empty="fail")
    cap_b = FakeCapture(frames=[good], on_empty="repeat")
    made: list = []

    def factory(idx):
        cap = cap_a if not made else cap_b
        made.append(cap)
        return cap

    detector = FakeDetector()
    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=factory,
        stale_after_s=0.2,
        reopen_after_s=0.2,
    ) as loop:
        # cap_a's one frame drains -> reads fail -> health goes stale.
        _wait_until(lambda: loop.health == "stale", timeout_s=2.0)
        # Past reopen_after_s of failures the loop reopens onto cap_b and heals.
        _wait_until(lambda: loop.health == "ok", timeout_s=3.0)

    assert len(made) >= 2  # the camera was reopened, not just re-read
    assert cap_a.released is True  # the wedged handle was released on reopen


def test_reopen_targets_the_switched_index_after_set_camera(
    fake_frame, FakeDetector, FakeCapture
):
    # After set_camera(7), a later reopen-on-failure must reopen index 7, not
    # the original index — the loop tracks the active index across switches.
    good = fake_frame(1920, 1080)
    opened_indices: list[int] = []

    def factory(idx):
        opened_indices.append(idx)
        # Every cap yields one frame then fails, forcing repeated reopens.
        return FakeCapture(frames=[good], on_empty="fail")

    detector = FakeDetector()
    with running_loop(
        detector,
        camera_index=0,
        render_fn=_identity_render,
        cap_factory=factory,
        stale_after_s=0.2,
        reopen_after_s=0.2,
    ) as loop:
        _wait_until(lambda: loop.generation > 0)  # index 0 opened + published
        loop.set_camera(7)
        # Give the switch + at least one reopen-on-failure cycle time to run.
        _wait_until(lambda: opened_indices.count(7) >= 2, timeout_s=3.0)

    assert opened_indices[0] == 0  # started on the original index
    assert 7 in opened_indices  # switched to the requested index
    # Once switched, reopens stay on 7 (never silently revert to 0).
    assert all(i == 7 for i in opened_indices[opened_indices.index(7):])
