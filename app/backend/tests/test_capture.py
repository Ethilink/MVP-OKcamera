"""AC1-AC12 for T03: `CaptureLoop` — the single background thread that owns
the camera + tracker.update(), publishing an atomic `Latest` snapshot.

`CaptureLoop` genuinely spawns a background thread (unlike `Session`, which
is pure/synchronous — DESIGN D11), so this suite's only concession to real
time is thread *synchronization*, never business-timing:

- `_Counter` (a `threading.Condition`) is how a test waits for "N frames have
  published" — CPython's `Condition.wait(timeout=...)` uses the C-level
  monotonic clock, not the Python-level `time.monotonic` name, so it keeps
  working even in tests that patch `backend.capture.time.monotonic`.
- Staleness (AC6/AC7) is driven by patching `backend.capture.time.monotonic`
  with a `_FakeClock` the test advances by hand — no real waiting for
  `stale_after_s` to elapse.
- Two tests can't avoid a short real wall-clock wait because the behavior
  under test IS a real-time property: the retry-rate cap (AC6, "attempts per
  real second") and `stop()`'s join deadline (AC9, "joins ... <= 2s"). Both
  windows are kept small (<= 0.2s).

All fakes/doubles other than `FakeCaptureSource`/`ScenarioTracker` (from
T01's `backend/fakes.py`) are defined locally below — they exist purely to
control failure/exception/blocking behavior the T01 fakes don't provide.
"""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
import pytest
import supervision as sv

import backend.capture as capture_module
from backend.capture import CaptureLoop, Latest
from backend.fakes import FakeCaptureSource, ScenarioTracker

_FRAME_SIZE = (64, 48)  # (width, height) — tiny on purpose, keeps ticks cheap
_WAIT = 1.0             # generous per-test synchronization timeout


class _Counter:
    """Thread-safe call counter with a blocking wait, used to synchronize the
    test thread with `CaptureLoop`'s background thread. Built on
    `threading.Condition` (C-level timing) rather than a `time.monotonic`
    polling loop, so it is unaffected by any `time.monotonic` patch a test
    installs to control `CaptureLoop`'s OWN staleness bookkeeping."""

    def __init__(self) -> None:
        self._n = 0
        self._cv = threading.Condition()

    def hit(self) -> None:
        with self._cv:
            self._n += 1
            self._cv.notify_all()

    def wait_for(self, n: int, timeout: float = _WAIT) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: self._n >= n, timeout=timeout)

    @property
    def value(self) -> int:
        with self._cv:
            return self._n


class _FakeClock:
    """A settable stand-in for `time.monotonic`, so staleness tests can jump
    "now" forward without any real waiting."""

    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, dt: float) -> None:
        with self._lock:
            self._now += dt


def _wait_until(predicate, timeout: float = _WAIT, interval: float = 0.005) -> bool:
    """Real-time polling — used ONLY where no callback exists to hook a
    `_Counter` into (e.g. "the thread died"). Never used in a test that also
    patches `backend.capture.time.monotonic`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _SwitchableCapture:
    """A `VideoCaptureLike` whose reads can be flipped between always-succeed
    and always-fail from the test thread, for AC6's stale/recover cycle."""

    def __init__(self, size: tuple[int, int] = _FRAME_SIZE) -> None:
        self._width, self._height = size
        self._opened = True
        self.failing = False
        self.read_count = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if self.failing:
            return False, None
        return True, np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        return 0.0

    def release(self) -> None:
        self._opened = False


class _BlockingReadCapture:
    """A `VideoCaptureLike` that opens fine but whose `read()` never returns
    until `release()` is called — simulates a camera wedged mid-read, for
    AC11's `reset_tracker()` timeout test."""

    def __init__(self, size: tuple[int, int] = _FRAME_SIZE) -> None:
        self._width, self._height = size
        self._opened = True
        self._unblock = threading.Event()

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        self._unblock.wait()
        return True, np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        return 0.0

    def release(self) -> None:
        self._opened = False
        self._unblock.set()  # let a stuck read() return so nothing leaks


class _NeverOpensCapture:
    """A `VideoCaptureLike` that reports it never opened."""

    def isOpened(self) -> bool:
        return False

    def read(self) -> tuple[bool, np.ndarray | None]:
        return False, None

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        return 0.0

    def release(self) -> None:
        pass


def _single_id_detection(tracker_id: int, frame_shape: tuple[int, int]) -> sv.Detections:
    height, width = frame_shape
    mask = np.zeros((1, height, width), dtype=bool)
    mask[0, 0:5, 0:5] = True
    return sv.Detections(
        xyxy=np.array([[0.0, 0.0, 5.0, 5.0]], dtype=np.float32),
        mask=mask,
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
        tracker_id=np.array([tracker_id], dtype=int),
    )


class _OrderedSingleIdTracker:
    """Each `update()` call publishes present_ids == {call_count} (a single,
    ever-increasing id) — the sequence seen via `on_frame` directly encodes
    call order, for AC2's in-order / one-thread assertion."""

    def __init__(self) -> None:
        self.confidence = 0.5
        self.call_threads: set[int] = set()
        self._n = 0

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    def reset(self) -> None:
        self._n = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        self.call_threads.add(threading.get_ident())
        self._n += 1
        return _single_id_detection(self._n, frame.shape[:2])


class _RetainingTracker:
    """Records a direct reference (no defensive copy) to the frame it's
    given. AC3 is about aliasing, so the tracker deliberately does NOT copy —
    if the capture loop hands `render_fn` the SAME array (bug), a later
    mutation by `render_fn` becomes visible here too."""

    def __init__(self) -> None:
        self.confidence = 0.5
        self.last_frame: np.ndarray | None = None

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    def reset(self) -> None:
        pass

    def update(self, frame: np.ndarray) -> sv.Detections:
        self.last_frame = frame
        return sv.Detections.empty()


_MUTATION_SENTINEL = 111


def _mutating_render(frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
    frame[:] = _MUTATION_SENTINEL
    return frame


class _FlakyTracker:
    """Wraps a real `ScenarioTracker` but raises on specific call numbers
    (`fail_on`) or on every call once `always_fail` is set — for AC6/AC7's
    exception-handling assertions."""

    def __init__(self, base: ScenarioTracker, fail_on: set[int] | None = None) -> None:
        self._base = base
        self._fail_on = fail_on or set()
        self.always_fail = False
        self._n = 0
        self.confidence = base.confidence

    @property
    def class_names(self) -> dict[int, str]:
        return self._base.class_names

    @property
    def model_version(self) -> str:
        return self._base.model_version

    def reset(self) -> None:
        self._n = 0
        self._base.reset()

    def update(self, frame: np.ndarray) -> sv.Detections:
        self._n += 1
        if self.always_fail or self._n in self._fail_on:
            raise RuntimeError("simulated tracker failure")
        return self._base.update(frame)

    @property
    def call_count(self) -> int:
        return self._n


class _CrashingTracker:
    """Raises `KeyboardInterrupt` (a `BaseException`, not an `Exception`) on
    every call — AC7's "abnormal thread exit" must escape any per-tick
    `except Exception` guard and kill the loop."""

    def __init__(self) -> None:
        self.confidence = 0.5

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    def reset(self) -> None:
        pass

    def update(self, frame: np.ndarray) -> sv.Detections:
        raise KeyboardInterrupt("simulated fatal crash")


class _ResetTrackingTracker:
    """present_ids grows 1, 2, 3, ... per call; `reset()` restarts the count
    at 0 and records the calling thread's identity — AC11."""

    def __init__(self) -> None:
        self.confidence = 0.5
        self.reset_call_threads: list[int] = []
        self.update_call_threads: set[int] = set()
        self._n = 0

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    def reset(self) -> None:
        self.reset_call_threads.append(threading.get_ident())
        self._n = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        self.update_call_threads.add(threading.get_ident())
        self._n += 1
        return _growing_ids_detection(self._n, frame.shape[:2])


def _growing_ids_detection(n: int, frame_shape: tuple[int, int]) -> sv.Detections:
    height, width = frame_shape
    mask = np.zeros((n, height, width), dtype=bool)
    mask[:, 0:5, 0:5] = True
    return sv.Detections(
        xyxy=np.array([[0.0, 0.0, 5.0, 5.0]] * n, dtype=np.float32),
        mask=mask,
        confidence=np.full(n, 0.9, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
        tracker_id=np.arange(1, n + 1, dtype=int),
    )


def _make_loop(tracker, cap, **kwargs) -> CaptureLoop:
    kwargs.setdefault("frame_size", _FRAME_SIZE)
    return CaptureLoop(
        tracker=tracker,
        camera_index=0,
        cap_factory=lambda _idx: cap,
        **kwargs,
    )


class TestAC1CapturePropsOnStart:
    """AC1: start() sets CAP_PROP_FRAME_WIDTH/HEIGHT to frame_size and
    CAP_PROP_BUFFERSIZE=1 on the capture."""

    def test_ac1_start_sets_frame_size_and_buffersize_on_the_capture(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids: counter.hit())

        loop.start()
        try:
            assert counter.wait_for(1)
        finally:
            loop.stop()

        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == _FRAME_SIZE[0]
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == _FRAME_SIZE[1]
        assert cap.get(cv2.CAP_PROP_BUFFERSIZE) == 1


class TestStartRaisesWhenCaptureWontOpen:
    """Frozen interface comment on `start()` (not an independently numbered
    AC): "RuntimeError if capture won't open"."""

    def test_start_raises_runtime_error_when_capture_will_not_open(self) -> None:
        loop = _make_loop(ScenarioTracker(), _NeverOpensCapture())

        with pytest.raises(RuntimeError):
            loop.start()


class TestAC2OneThreadInCaptureOrder:
    """AC2: the tracker receives frames in capture order from ONE thread;
    each published Latest.present_ids equals that frame's dets' tracker_ids."""

    def test_ac2_publishes_tracker_ids_in_capture_order_from_one_thread(self) -> None:
        tracker = _OrderedSingleIdTracker()
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        seen: list[frozenset[int]] = []
        counter = _Counter()

        def on_frame(t: float, present_ids: frozenset[int]) -> None:
            seen.append(present_ids)
            counter.hit()

        loop = _make_loop(tracker, cap, on_frame=on_frame)
        loop.start()
        try:
            assert counter.wait_for(5)
        finally:
            loop.stop()

        assert seen[:5] == [frozenset({n}) for n in range(1, 6)]
        assert len(tracker.call_threads) == 1
        call_thread = next(iter(tracker.call_threads))
        assert call_thread != threading.get_ident()


class TestAC3RenderFnReceivesAnIndependentCopy:
    """AC3: render_fn receives a copy — mutating its input does not alter any
    array the tracker sees (aliasing check; see module docstring for the
    "later frames" caveat this suite can't exercise with these fakes)."""

    def test_ac3_render_fn_mutation_does_not_leak_into_the_trackers_frame(self) -> None:
        tracker = _RetainingTracker()
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(
            tracker,
            cap,
            on_frame=lambda t, ids: counter.hit(),
            render_fn=_mutating_render,
        )

        loop.start()
        try:
            assert counter.wait_for(1)
        finally:
            loop.stop()

        assert tracker.last_frame is not None
        assert not np.all(tracker.last_frame == _MUTATION_SENTINEL)


class TestAC4GenerationAndSnapshot:
    """AC4: each published tick rebinds a new Latest; generation strictly
    increases per published frame; snapshot() is None only before the first
    publication."""

    def test_ac4_snapshot_none_before_start_then_populated_after_publish(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids: counter.hit())

        assert loop.snapshot() is None

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        assert isinstance(latest, Latest)

    def test_ac4_generation_strictly_increases_per_published_frame(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        generations_seen: list[int] = []
        counter = _Counter()

        def on_frame(t: float, ids: frozenset[int]) -> None:
            generations_seen.append(loop.generation)
            counter.hit()

        loop = _make_loop(ScenarioTracker(), cap, on_frame=on_frame)
        loop.start()
        try:
            assert counter.wait_for(5)
        finally:
            loop.stop()

        first_five = generations_seen[:5]
        assert first_five == sorted(first_five)
        assert len(set(first_five)) == len(first_five)
        assert first_five[0] >= 1


class TestAC5OverlayJpegDecodesToFrameSize:
    """AC5: overlay_jpeg decodes (cv2.imdecode) to frame_size dimensions."""

    def test_ac5_overlay_jpeg_decodes_to_frame_size_dimensions(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids: counter.hit())

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        decoded = cv2.imdecode(
            np.frombuffer(latest.overlay_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape[:2] == (_FRAME_SIZE[1], _FRAME_SIZE[0])


class TestAC6StaleDetectionAndRecovery:
    """AC6: no publication for > stale_after_s (failed reads OR tracker
    exceptions every tick) flips health to "stale" without killing the
    thread or busy-spinning (<=30 attempts/s); recovery flips back to "ok"."""

    def test_ac6_prolonged_read_failures_flip_health_to_stale_then_recover(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr(capture_module.time, "monotonic", clock)
        cap = _SwitchableCapture()
        counter = _Counter()
        loop = _make_loop(
            ScenarioTracker(),
            cap,
            on_frame=lambda t, ids: counter.hit(),
            stale_after_s=1.0,
        )

        loop.start()
        try:
            assert counter.wait_for(1)
            assert loop.health == "ok"

            cap.failing = True
            clock.advance(2.0)
            assert loop.health == "stale"

            published_before_recovery = counter.value
            cap.failing = False
            assert counter.wait_for(published_before_recovery + 1)
            assert loop.health == "ok"
        finally:
            loop.stop()

    def test_ac6_retry_rate_is_capped_not_busy_spinning(self) -> None:
        cap = _SwitchableCapture()
        cap.failing = True
        loop = _make_loop(ScenarioTracker(), cap, stale_after_s=100.0)

        loop.start()
        try:
            # Real wall-clock window is unavoidable here: this AC is
            # specifically about attempts-PER-REAL-SECOND, not a value an
            # injected clock or frame count can stand in for.
            threading.Event().wait(0.2)
            attempts = cap.read_count
        finally:
            loop.stop()

        # Uncapped busy-spin would rack up thousands of attempts in 200ms;
        # the spec's <=30/s ceiling allows ~6. Generous upper bound absorbs
        # scheduler jitter while still failing hard against a busy-spin.
        assert 0 < attempts <= 30


class TestAC7TrackerExceptionHandling:
    """AC7: a tracker exception on one frame is caught + logged and that
    tick is skipped (no new Latest, no generation bump, no on_frame call);
    the loop continues; an isolated blip leaves health "ok" (given a prior
    publication); a sustained run surfaces as "stale"; an abnormal thread
    exit -> "dead"."""

    def test_ac7_tracker_exception_skips_that_tick_without_crashing_the_loop(
        self,
    ) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        tracker = _FlakyTracker(ScenarioTracker(), fail_on={2})
        counter = _Counter()
        loop = _make_loop(tracker, cap, on_frame=lambda t, ids: counter.hit())

        loop.start()
        try:
            assert counter.wait_for(3)
            health = loop.health
        finally:
            loop.stop()

        # 3 publications despite 1 internal failure needs >= 4 attempts; a
        # buggy "publish anyway" implementation would need only 3.
        assert tracker.call_count >= 4
        assert health != "dead"

    def test_ac7_isolated_exception_blip_still_reads_health_ok(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        tracker = _FlakyTracker(ScenarioTracker(), fail_on={3})
        counter = _Counter()
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids: counter.hit(), stale_after_s=1.0
        )

        loop.start()
        try:
            assert counter.wait_for(5)
            health = loop.health
        finally:
            loop.stop()

        assert health == "ok"

    def test_ac7_sustained_tracker_exceptions_eventually_flip_health_to_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clock = _FakeClock()
        monkeypatch.setattr(capture_module.time, "monotonic", clock)
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        tracker = _FlakyTracker(ScenarioTracker())
        counter = _Counter()
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids: counter.hit(), stale_after_s=1.0
        )

        loop.start()
        try:
            assert counter.wait_for(1)
            tracker.always_fail = True
            clock.advance(2.0)
            assert loop.health == "stale"
        finally:
            loop.stop()

    def test_ac7_an_abnormal_thread_exit_reports_health_dead(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        loop = _make_loop(_CrashingTracker(), cap)

        loop.start()
        try:
            assert _wait_until(lambda: loop.health == "dead")
        finally:
            loop.stop()


class TestAC8OnFrameContract:
    """AC8: on_frame(t, present_ids) fires exactly once per published frame
    with strictly increasing t; an on_frame exception doesn't kill the loop."""

    def test_ac8_on_frame_fires_with_strictly_increasing_t(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        seen_t: list[float] = []
        counter = _Counter()

        def on_frame(t: float, ids: frozenset[int]) -> None:
            seen_t.append(t)
            counter.hit()

        loop = _make_loop(ScenarioTracker(), cap, on_frame=on_frame)
        loop.start()
        try:
            assert counter.wait_for(5)
        finally:
            loop.stop()

        first_five = seen_t[:5]
        assert len(first_five) == 5
        assert first_five == sorted(first_five)
        assert len(set(first_five)) == 5

    def test_ac8_on_frame_exception_does_not_kill_the_loop(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()

        def flaky_on_frame(t: float, ids: frozenset[int]) -> None:
            counter.hit()
            if counter.value == 2:
                raise RuntimeError("boom in on_frame")

        loop = _make_loop(ScenarioTracker(), cap, on_frame=flaky_on_frame)
        loop.start()
        try:
            assert counter.wait_for(6)
            health = loop.health
        finally:
            loop.stop()

        assert health != "dead"


class TestAC9StopJoinsReleasesAndReportsDead:
    """AC9: stop() joins the thread <= 2s, releases the capture, and leaves
    health == "dead"."""

    def test_ac9_stop_joins_releases_capture_and_reports_dead(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids: counter.hit())

        loop.start()
        assert counter.wait_for(1)

        start = time.monotonic()
        loop.stop()
        elapsed = time.monotonic() - start

        assert elapsed <= 2.0
        assert loop.health == "dead"
        assert cap.isOpened() is False


class TestAC11ResetTracker:
    """AC11: reset_tracker() invokes tracker.reset() exactly once, on the
    capture thread, and returns only after it's applied; a no-op before
    start(); TimeoutError when the capture thread is stuck in read()."""

    def test_ac11_reset_runs_on_capture_thread_and_next_frame_reflects_it(
        self,
    ) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        tracker = _ResetTrackingTracker()
        counts: list[int] = []
        counter = _Counter()

        def on_frame(t: float, ids: frozenset[int]) -> None:
            counts.append(len(ids))
            counter.hit()

        loop = _make_loop(tracker, cap, on_frame=on_frame)
        loop.start()
        try:
            assert counter.wait_for(4)
            assert counts[3] == 4  # sanity: unreset count has grown to 4

            loop.reset_tracker(timeout_s=1.0)

            after_reset_index = len(counts)
            assert counter.wait_for(after_reset_index + 1)
        finally:
            loop.stop()

        assert len(tracker.reset_call_threads) == 1
        assert tracker.reset_call_threads[0] != threading.get_ident()
        assert tracker.reset_call_threads[0] in tracker.update_call_threads
        assert counts[after_reset_index] == 1

    def test_ac11_reset_before_start_is_a_safe_noop(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids: counter.hit())

        loop.reset_tracker(timeout_s=0.2)  # must not raise

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None

    def test_ac11_reset_raises_timeout_error_when_capture_thread_is_stuck(
        self,
    ) -> None:
        cap = _BlockingReadCapture()
        loop = _make_loop(ScenarioTracker(), cap)

        loop.start()
        try:
            with pytest.raises(TimeoutError):
                loop.reset_tracker(timeout_s=0.1)
        finally:
            cap.release()  # unblock the stuck read so nothing leaks
            loop.stop()


class TestAC12PresentIdsAreBuiltinInts:
    """AC12: every element of Latest.present_ids is a builtin int, so a raw
    json.dumps of a status dict built from it round-trips."""

    def test_ac12_present_ids_are_builtin_ints_and_json_round_trips(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids: counter.hit())

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        assert len(latest.present_ids) > 0
        assert all(type(x) is int for x in latest.present_ids)
        assert latest.count == len(latest.present_ids)

        payload = {"present_ids": sorted(latest.present_ids), "count": latest.count}
        assert json.loads(json.dumps(payload)) == payload
