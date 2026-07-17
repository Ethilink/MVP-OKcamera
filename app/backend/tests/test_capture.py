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

MIGRATED for T10 (B-C): the seam widened by one value — the roster. Every
local tracker double now carries the `roster` attribute the tracker interface
guarantees, every `on_frame` callback takes `(t, present_ids, roster)`, and
`render_fn` takes `(frame, dets, roster, catalog, t)` since T11/R1 (catalog is
sampled the same tick and threaded to the overlay too). Nothing else about T03's
ACs changed, so their tests are otherwise untouched.
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
from backend.capture import CaptureLoop, Latest, TrackerResetError
from backend.fakes import FakeCaptureSource, ScenarioTracker

_FRAME_SIZE = (64, 48)  # (width, height) — tiny on purpose, keeps ticks cheap
_WAIT = 1.0             # generous per-test synchronization timeout

# Every tracker double answers `roster` — the T10 seam guarantees it (B-C2), so
# `CaptureLoop` reads it without a fallback. The VALUE is irrelevant to T03's
# ACs; only the T10 B-C tests below use a double that controls it.
_DOUBLE_ROSTER = frozenset({1, 2, 3})

# T11 (B1): `_capture_tick` now also samples `tracker.catalog` every tick, so
# every tracker double that publishes a frame must answer `catalog` (a double
# lacking it, like `_RosterlessTracker`, makes the tick raise-and-skip). The
# VALUE is irrelevant to T03/T10's ACs — only T11's new catalog/resolving tests
# assert on it — so the fixed doubles just hand back this constant.
_DOUBLE_CATALOG = frozenset({1, 2, 3})


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

    @property
    def roster(self) -> frozenset[int]:
        return _DOUBLE_ROSTER

    @property
    def catalog(self) -> frozenset[int]:
        return _DOUBLE_CATALOG

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

    @property
    def roster(self) -> frozenset[int]:
        return _DOUBLE_ROSTER

    @property
    def catalog(self) -> frozenset[int]:
        return _DOUBLE_CATALOG

    def reset(self) -> None:
        pass

    def update(self, frame: np.ndarray) -> sv.Detections:
        self.last_frame = frame
        return sv.Detections.empty()


_MUTATION_SENTINEL = 111


def _mutating_render(
    frame: np.ndarray,
    dets: sv.Detections,
    roster: frozenset[int],
    catalog: frozenset[int],
    t: float,
) -> np.ndarray:
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

    @property
    def roster(self) -> frozenset[int]:
        return self._base.roster

    @property
    def catalog(self) -> frozenset[int]:
        return self._base.catalog

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

    @property
    def roster(self) -> frozenset[int]:
        return _DOUBLE_ROSTER

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

    @property
    def roster(self) -> frozenset[int]:
        return _DOUBLE_ROSTER

    @property
    def catalog(self) -> frozenset[int]:
        return _DOUBLE_CATALOG

    def reset(self) -> None:
        self.reset_call_threads.append(threading.get_ident())
        self._n = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        self.update_call_threads.add(threading.get_ident())
        self._n += 1
        return _growing_ids_detection(self._n, frame.shape[:2])


class _FailingResetTracker(_ResetTrackingTracker):
    def reset(self) -> None:
        raise RuntimeError("simulated reset failure")


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
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

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

        def on_frame(
            t: float,
            present_ids: frozenset[int],
            roster: frozenset[int],
            catalog: frozenset[int],
            resolving: frozenset[int],
        ) -> None:
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
            on_frame=lambda t, ids, roster, catalog, resolving: counter.hit(),
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
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

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

        def on_frame(
            t: float,
            ids: frozenset[int],
            roster: frozenset[int],
            catalog: frozenset[int],
            resolving: frozenset[int],
        ) -> None:
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
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

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


class TestLatestCarriesCropMaterial:
    """D17: each published `Latest` carries an owned, read-only camera frame plus
    row-aligned `(tracker_id, xyxy)` boxes as plain int/float — the raw material
    the `/status` setup branch crops lazily. Kept JSON-native so T04 never
    sanitizes it (mirrors the `present_ids` builtin-int contract, AC12)."""

    def test_frame_bgr_is_an_owned_read_only_frame_of_capture_size(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        assert isinstance(latest.frame_bgr, np.ndarray)
        assert latest.frame_bgr.shape == (_FRAME_SIZE[1], _FRAME_SIZE[0], 3)
        assert latest.frame_bgr.flags.writeable is False  # immutability is real

    def test_detections_are_json_native_and_aligned_with_present_ids(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        assert len(latest.detections) == len(latest.present_ids)
        assert {det.tracker_id for det in latest.detections} == set(latest.present_ids)
        for det in latest.detections:
            assert type(det.tracker_id) is int
            assert len(det.xyxy) == 4
            assert all(type(coord) is float for coord in det.xyxy)
            assert det.mask is not None
            assert det.mask.ndim == 2
            assert det.mask.dtype == np.bool_
            assert det.mask.flags.writeable is False
        # the whole detections payload round-trips through JSON unchanged
        payload = [[det.tracker_id, list(det.xyxy)] for det in latest.detections]
        assert json.loads(json.dumps(payload)) == payload


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
            on_frame=lambda t, ids, roster, catalog, resolving: counter.hit(),
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
        loop = _make_loop(tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

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
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit(), stale_after_s=1.0
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
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit(), stale_after_s=1.0
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

        def on_frame(
            t: float,
            ids: frozenset[int],
            roster: frozenset[int],
            catalog: frozenset[int],
            resolving: frozenset[int],
        ) -> None:
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

        def flaky_on_frame(
            t: float,
            ids: frozenset[int],
            roster: frozenset[int],
            catalog: frozenset[int],
            resolving: frozenset[int],
        ) -> None:
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
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

        loop.start()
        assert counter.wait_for(1)

        start = time.monotonic()
        loop.stop()
        elapsed = time.monotonic() - start

        assert elapsed <= 2.0
        assert loop.health == "dead"
        assert cap.isOpened() is False


class TestStartRejectsASecondStart:
    """DESIGN "exactly one capture thread": start() on an already-running loop
    raises rather than spawning a second thread that races on the capture."""

    def test_second_start_while_running_raises_runtime_error(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

        loop.start()
        try:
            assert counter.wait_for(1)
            with pytest.raises(RuntimeError):
                loop.start()
        finally:
            loop.stop()


class TestStopForcesDeadWhenThreadWontJoin:
    """AC9 hardening: if the capture thread is wedged in a blocking read() past
    stop()'s 2s join deadline, stop() still returns promptly, releases the
    capture from the caller thread (which unblocks the stuck read), and leaves
    health == "dead" — the strongest achievable, since a Python thread can't be
    killed."""

    def test_stop_releases_and_reports_dead_when_read_blocks_past_join(
        self,
    ) -> None:
        cap = _BlockingReadCapture()  # read() blocks until release() (AC11 pattern)
        released: list[bool] = []
        original_release = cap.release

        def spy_release() -> None:
            released.append(True)
            original_release()

        cap.release = spy_release  # type: ignore[method-assign]
        loop = _make_loop(ScenarioTracker(), cap)

        loop.start()
        try:
            start = time.monotonic()
            loop.stop()
            elapsed = time.monotonic() - start

            assert elapsed <= 2.5
            assert loop.health == "dead"
            assert released  # stop() released the capture from the caller thread
        finally:
            cap.release()  # ensure the stuck read is unblocked so nothing leaks


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

        def on_frame(
            t: float,
            ids: frozenset[int],
            roster: frozenset[int],
            catalog: frozenset[int],
            resolving: frozenset[int],
        ) -> None:
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
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

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

    def test_ac11_reset_failure_is_reported_without_killing_capture(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(
            _FailingResetTracker(),
            cap,
            on_frame=lambda t, ids, roster, catalog, resolving: counter.hit(),
        )

        loop.start()
        try:
            assert counter.wait_for(1)
            with pytest.raises(TrackerResetError) as exc_info:
                loop.reset_tracker(timeout_s=1.0)
            before = counter.value
            assert counter.wait_for(before + 1)
            assert loop.health != "dead"
        finally:
            loop.stop()

        assert isinstance(exc_info.value.__cause__, RuntimeError)


class _RosterTracker:
    """A tracker whose roster and present_ids are both dictated by the test.
    `roster` deliberately hands back numpy ints — what the real linker's
    frozenset carries — so B-C1's builtin-int cast is observable."""

    def __init__(self, roster: set[int], present: set[int]) -> None:
        self.confidence = 0.5
        self._roster = frozenset(np.int64(i) for i in roster)
        self._present = sorted(present)

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    @property
    def roster(self) -> frozenset:
        return self._roster

    @property
    def catalog(self) -> frozenset:
        return self._roster

    def reset(self) -> None:
        pass

    def update(self, frame: np.ndarray) -> sv.Detections:
        return _growing_ids_detection(len(self._present), frame.shape[:2])


class _EnrollingTracker:
    """Enrols one more instrument per `update()`: the Nth call reports
    present_ids == {N} and roster == {1..N}. The pair is a fingerprint of WHEN
    the roster was sampled — reading it a tick early (before `update()`) or a
    tick late puts the two out of step by exactly one, which no single-frame
    assertion could see."""

    def __init__(self) -> None:
        self.confidence = 0.5
        self._n = 0

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    @property
    def roster(self) -> frozenset[int]:
        return frozenset(range(1, self._n + 1))

    @property
    def catalog(self) -> frozenset[int]:
        return frozenset(range(1, self._n + 1))

    def reset(self) -> None:
        self._n = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        self._n += 1
        return _single_id_detection(self._n, frame.shape[:2])


class _RosterlessTracker:
    """A tracker with NO `roster` attribute at all."""

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
        return _single_id_detection(1, frame.shape[:2])


class TestBC1CaptureSamplesTheRoster:
    """B-C1: `_capture_tick` reads `tracker.roster` once per tick, right after
    `update()`, casts its members to builtin int, stores it in `Latest.roster`,
    and passes it to BOTH the render call and `on_frame`."""

    def test_b_c1_latest_carries_the_trackers_roster_as_builtin_ints(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        tracker = _RosterTracker(roster={3, 5, 7, 9}, present={3, 5})
        loop = _make_loop(tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        assert latest.roster == frozenset({3, 5, 7, 9})
        assert all(type(member) is int for member in latest.roster)
        # ... and it JSON-encodes without T04 having to sanitize it (cf. AC12)
        assert json.loads(json.dumps(sorted(latest.roster))) == [3, 5, 7, 9]

    def test_b_c1_on_frame_receives_the_roster(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        seen: list[frozenset[int]] = []

        def on_frame(
            t: float,
            ids: frozenset[int],
            roster: frozenset[int],
            catalog: frozenset[int],
            resolving: frozenset[int],
        ) -> None:
            seen.append(roster)
            counter.hit()

        loop = _make_loop(_RosterTracker(roster={2, 4}, present={2}), cap, on_frame=on_frame)
        loop.start()
        try:
            assert counter.wait_for(3)
        finally:
            loop.stop()

        assert seen[:3] == [frozenset({2, 4})] * 3

    def test_b_c1_render_fn_receives_the_roster_and_catalog_with_the_frames_own_t(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        calls: list[tuple[frozenset[int], frozenset[int], float]] = []

        def spy_render(frame, dets, roster, catalog, t):
            calls.append((roster, catalog, t))
            return frame

        loop = _make_loop(
            _RosterTracker(roster={2, 4}, present={2}),
            cap,
            on_frame=lambda t, ids, roster, catalog, resolving: counter.hit(),
            render_fn=spy_render,
        )
        loop.start()
        try:
            assert counter.wait_for(2)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        assert calls
        rosters = [roster for roster, _, _ in calls]
        catalogs = [catalog for _, catalog, _ in calls]
        assert rosters[:2] == [frozenset({2, 4})] * 2
        # catalog is sampled and threaded to the renderer too (T11/R1), builtin ints
        assert catalogs[:2] == [frozenset({2, 4})] * 2
        assert all(type(next(iter(catalog))) is int for catalog in catalogs[:2])
        # the render call and the snapshot describe the same instant
        assert latest.t in [t for _, _, t in calls]

    def test_b_c1_roster_is_sampled_in_the_same_tick_as_present_ids(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        pairs: list[tuple[frozenset[int], frozenset[int]]] = []

        def on_frame(
            t: float,
            ids: frozenset[int],
            roster: frozenset[int],
            catalog: frozenset[int],
            resolving: frozenset[int],
        ) -> None:
            pairs.append((ids, roster))
            counter.hit()

        loop = _make_loop(_EnrollingTracker(), cap, on_frame=on_frame)
        loop.start()
        try:
            assert counter.wait_for(5)
        finally:
            loop.stop()

        for present_ids, roster in pairs[:5]:
            enrolled_so_far = frozenset(range(1, max(present_ids) + 1))
            assert roster == enrolled_so_far, f"roster {sorted(roster)} lags {sorted(present_ids)}"


class TestBC2NoRosterFallback:
    """B-C2: the seam guarantees `roster`, so `CaptureLoop` must NOT carry
    defensive fallback code for a tracker that lacks it. An invented empty
    roster is not a safe default — downstream it means "nothing is an
    instrument", turning the whole tray gray and emptying `/status`."""

    def test_b_c2_a_tracker_without_a_roster_is_not_papered_over_with_an_empty_one(
        self,
    ) -> None:
        cap = _SwitchableCapture()
        loop = _make_loop(_RosterlessTracker(), cap)

        loop.start()
        try:
            assert _wait_until(lambda: cap.read_count >= 3)  # ticks are happening
            snapshot = loop.snapshot()
            health = loop.health
        finally:
            loop.stop()

        assert snapshot is None  # the tick raised and was skipped — nothing invented
        assert health != "dead"  # ... and the loop itself soldiers on (AC7)


class TestAC12PresentIdsAreBuiltinInts:
    """AC12: every element of Latest.present_ids is a builtin int, so a raw
    json.dumps of a status dict built from it round-trips."""

    def test_ac12_present_ids_are_builtin_ints_and_json_round_trips(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        loop = _make_loop(ScenarioTracker(), cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit())

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


# --- T11 B1: the snapshot carries same-tick catalog + row-aligned resolving ----


def _ids_detection(ids, frame_shape: tuple[int, int], resolving=None) -> sv.Detections:
    """A detections payload with arbitrary `ids` and (optionally) a row-aligned
    `data["resolving"]` bool array. `resolving=None` omits the key entirely — the
    safe-default path where every row must read False (tracker-interface.md)."""
    ids = list(ids)
    n = len(ids)
    height, width = frame_shape
    mask = np.zeros((n, height, width), dtype=bool)
    mask[:, 0:5, 0:5] = True
    data: dict = {}
    if resolving is not None:
        data["resolving"] = np.array(list(resolving), dtype=bool)
    return sv.Detections(
        xyxy=np.array([[0.0, 0.0, 5.0, 5.0]] * n, dtype=np.float32),
        mask=mask,
        confidence=np.full(n, 0.9, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
        tracker_id=np.array(ids, dtype=int),
        data=data,
    )


class _CatalogResolvingTracker:
    """A tracker double whose `catalog`, present ids, and per-row `resolving`
    flag are all dictated by the test. `catalog` (like `roster`) hands back numpy
    ints — what the real seam carries — so B1's builtin-int cast is observable.
    `emit_resolving=False` returns detections with NO `resolving` key at all, so
    the safe default (every row False) can be exercised."""

    def __init__(
        self,
        *,
        catalog,
        present,
        resolving=(),
        roster=None,
        emit_resolving: bool = True,
    ) -> None:
        self.confidence = 0.5
        self._catalog = frozenset(np.int64(i) for i in catalog)
        self._roster = frozenset(
            np.int64(i) for i in (catalog if roster is None else roster)
        )
        self._present = sorted(present)
        self._resolving = set(resolving)
        self._emit_resolving = emit_resolving

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    @property
    def roster(self) -> frozenset:
        return self._roster

    @property
    def catalog(self) -> frozenset:
        return self._catalog

    def reset(self) -> None:
        pass

    def update(self, frame: np.ndarray) -> sv.Detections:
        flags = (
            [tracker_id in self._resolving for tracker_id in self._present]
            if self._emit_resolving
            else None
        )
        return _ids_detection(self._present, frame.shape[:2], resolving=flags)


class TestB1CaptureSnapshotCarriesCatalogAndResolving:
    """T11 B1 / backend test 1: `_capture_tick` samples `tracker.catalog` the SAME
    tick as present_ids/roster (as builtin, JSON-native ints), derives a
    row-aligned `resolving` flag per `DetectionBox` from `dets.data["resolving"]`
    (safe default False on a missing key), and hands `on_frame` the 5-tuple
    `(t, present_ids, roster, catalog, resolving_ids)` all sampled together."""

    def test_b1_snapshot_catalog_is_the_sampled_catalog_as_json_native_ints(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        tracker = _CatalogResolvingTracker(catalog={3, 5, 7}, present={3, 5, 7}, resolving={5})
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit()
        )

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        assert latest.catalog == frozenset({3, 5, 7})
        assert all(type(member) is int for member in latest.catalog)
        assert json.loads(json.dumps(sorted(latest.catalog))) == [3, 5, 7]

    def test_b1_detection_boxes_carry_a_row_aligned_resolving_flag(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        tracker = _CatalogResolvingTracker(catalog={3, 5, 7}, present={3, 5, 7}, resolving={5})
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit()
        )

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        resolving_by_id = {det.tracker_id: det.resolving for det in latest.detections}
        assert resolving_by_id == {3: False, 5: True, 7: False}
        for det in latest.detections:
            assert type(det.tracker_id) is int
            assert type(det.resolving) is bool

    def test_b1_a_missing_resolving_key_defaults_every_row_to_false(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        tracker = _CatalogResolvingTracker(catalog={1, 2}, present={1, 2}, emit_resolving=False)
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit()
        )

        loop.start()
        try:
            assert counter.wait_for(1)
            latest = loop.snapshot()
        finally:
            loop.stop()

        assert latest is not None
        assert latest.detections  # sanity: there are rows to default
        assert all(det.resolving is False for det in latest.detections)

    def test_b1_on_frame_receives_the_five_tuple_sampled_the_same_tick(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        counter = _Counter()
        seen: list[tuple] = []

        def on_frame(t, ids, roster, catalog, resolving_ids) -> None:
            seen.append((ids, roster, catalog, resolving_ids))
            counter.hit()

        tracker = _CatalogResolvingTracker(
            catalog={3, 5, 7}, present={3, 5}, resolving={5}, roster={3, 5, 7}
        )
        loop = _make_loop(tracker, cap, on_frame=on_frame)

        loop.start()
        try:
            assert counter.wait_for(1)
        finally:
            loop.stop()

        assert seen
        ids, roster, catalog, resolving_ids = seen[0]
        assert ids == frozenset({3, 5})
        assert roster == frozenset({3, 5, 7})
        assert catalog == frozenset({3, 5, 7})
        # resolving_ids: the PRESENT ids whose row flag is True (7 is absent).
        assert resolving_ids == frozenset({5})
        assert all(type(member) is int for member in catalog)
        assert all(type(member) is int for member in resolving_ids)


# --- T11 B5: a timed-out capture command is cancelled, never leaks on recovery -


class _HoldableReadCapture:
    """A `VideoCaptureLike` whose `read()` succeeds until `hold()` is engaged,
    after which each read PARKS on an Event until `resume()` — so a test can wedge
    the capture thread mid-loop deterministically (it is stuck in `read()`, before
    the next `_apply_pending_reset`). `wait_blocked()` synchronises on the thread
    actually being parked."""

    def __init__(self, size: tuple[int, int] = _FRAME_SIZE) -> None:
        self._width, self._height = size
        self._opened = True
        self._held = threading.Event()
        self._resume = threading.Event()
        self._blocked = threading.Event()
        self.read_count = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        if self._held.is_set():
            self._blocked.set()
            self._resume.wait()
        self.read_count += 1
        return True, np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def hold(self) -> None:
        self._resume.clear()
        self._blocked.clear()
        self._held.set()

    def wait_blocked(self, timeout: float = _WAIT) -> bool:
        return self._blocked.wait(timeout)

    def resume(self) -> None:
        self._held.clear()
        self._resume.set()

    def set(self, prop_id: int, value: float) -> bool:
        return True

    def get(self, prop_id: int) -> float:
        return 0.0

    def release(self) -> None:
        self._opened = False
        self._resume.set()  # never leave a parked read wedged


class _ConfidenceRecordingTracker:
    """Records every confidence value assigned to it and each `reset()` — so a
    test can prove a TIMED-OUT confidence change never reaches the tracker, even
    after the capture thread recovers and applies a later plain reset."""

    def __init__(self) -> None:
        self._confidence = 0.5
        self.confidence_history: list[float] = []
        self.reset_count = 0
        self._n = 0

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self._confidence = value
        self.confidence_history.append(value)

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    @property
    def roster(self) -> frozenset[int]:
        return _DOUBLE_ROSTER

    @property
    def catalog(self) -> frozenset[int]:
        return _DOUBLE_CATALOG

    def reset(self) -> None:
        self.reset_count += 1
        self._n = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        self._n += 1
        return _single_id_detection(self._n, frame.shape[:2])


class _FailingConfidenceResetTracker(_ConfidenceRecordingTracker):
    """Applies the requested confidence, then fails reset so rollback is tested
    against the real CaptureLoop transaction rather than the API's view state."""

    def reset(self) -> None:
        self.reset_count += 1
        raise RuntimeError("simulated reset failure")


class TestB5FailedConfidenceChangeRollsBack:
    def test_failed_reset_restores_the_previous_tracker_confidence(self) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        tracker = _FailingConfidenceResetTracker()
        counter = _Counter()
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit()
        )

        loop.start()
        try:
            assert counter.wait_for(1)
            with pytest.raises(TrackerResetError):
                loop.set_confidence_and_reset(0.7, timeout_s=1.0)
        finally:
            loop.stop()

        assert tracker.confidence == 0.5
        assert tracker.confidence_history[-2:] == [0.7, 0.5]


class TestB5TimedOutCommandIsCancelledNotLeaked:
    """T11/B5 (Codex blocker): the capture-command handshake is per-command. When
    a `set_confidence_and_reset` times out against a stalled capture thread, that
    command is CANCELLED — so when the thread later recovers it is dropped, never
    applied. Its confidence can neither leak into a subsequent plain reset nor
    fire a tracker mutation after the fact."""

    def test_b5_timed_out_confidence_does_not_leak_into_a_later_reset(self) -> None:
        cap = _HoldableReadCapture()
        tracker = _ConfidenceRecordingTracker()
        counter = _Counter()
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit()
        )

        loop.start()
        try:
            assert counter.wait_for(2)  # the loop is alive and publishing

            # Wedge the capture thread inside read(), before _apply_pending_reset.
            cap.hold()
            assert cap.wait_blocked()

            # A confidence change now cannot be applied (thread stalled) -> times
            # out. The command must be cancelled, not left pending.
            with pytest.raises(TimeoutError):
                loop.set_confidence_and_reset(0.7, timeout_s=0.1)

            # Recover the thread and issue a PLAIN reset. It must apply as a pure
            # reset — the abandoned confidence must not ride along with it.
            cap.resume()
            loop.reset_tracker(timeout_s=1.0)
        finally:
            loop.stop()

        assert tracker.reset_count >= 1          # the plain reset really ran
        assert 0.7 not in tracker.confidence_history  # the timed-out value never applied
        assert tracker.confidence == 0.5         # confidence is still the original

    def test_b5_a_timed_out_command_does_not_fire_when_the_thread_recovers(
        self,
    ) -> None:
        # Bug #2: a timed-out command must not mutate the tracker at all once the
        # capture thread recovers (it would otherwise reset mid-recording).
        cap = _HoldableReadCapture()
        tracker = _ConfidenceRecordingTracker()
        counter = _Counter()
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit()
        )

        loop.start()
        try:
            assert counter.wait_for(2)
            cap.hold()
            assert cap.wait_blocked()

            with pytest.raises(TimeoutError):
                loop.reset_tracker(timeout_s=0.1)

            resets_at_timeout = tracker.reset_count
            cap.resume()
            # Let the loop run several more frames after recovery.
            assert counter.wait_for(counter.value + 3)
        finally:
            loop.stop()

        # The abandoned reset must NOT have fired on recovery.
        assert tracker.reset_count == resets_at_timeout


class _GatedResetTracker:
    """`reset()` blocks on a gate until the test releases it — lets a test hold
    the capture thread mid-apply (while it owns `_reset_lock`) to prove a command
    whose apply is IN FLIGHT when the submitter times out is waited out, never
    abandoned to fire after the submitter returned (the residual T11/B5 bug)."""

    def __init__(self) -> None:
        self._confidence = 0.5
        self.confidence_history: list[float] = []
        self.reset_started = threading.Event()
        self.reset_gate = threading.Event()
        self.reset_count = 0
        self._n = 0

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self._confidence = value
        self.confidence_history.append(value)

    @property
    def class_names(self) -> dict[int, str]:
        return {0: "surgical_instrument"}

    @property
    def model_version(self) -> str:
        return "test-0.1"

    @property
    def roster(self) -> frozenset[int]:
        return _DOUBLE_ROSTER

    @property
    def catalog(self) -> frozenset[int]:
        return _DOUBLE_CATALOG

    def reset(self) -> None:
        self.reset_started.set()
        self.reset_gate.wait(5.0)
        self.reset_count += 1
        self._n = 0

    def update(self, frame: np.ndarray) -> sv.Detections:
        self._n += 1
        return _single_id_detection(self._n, frame.shape[:2])


class TestB5InFlightCommandIsWaitedOutNotAbandoned:
    """T11/B5 (Codex follow-up): if a command's apply is ALREADY in flight when
    the submitter's timeout elapses, the submitter must not abandon it (return an
    error while the tracker mutates behind its back). Because the capture thread
    applies the whole mutation under `_reset_lock`, the submitter's timeout path
    blocks on that lock until the mutation completes, then returns its real
    result — the mutation is fully applied BEFORE the call returns."""

    def test_b5_a_command_applying_when_the_timeout_elapses_completes_before_return(
        self,
    ) -> None:
        cap = FakeCaptureSource(size=_FRAME_SIZE, fps=None)
        tracker = _GatedResetTracker()
        counter = _Counter()
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit()
        )

        result: dict[str, object] = {}

        def submit() -> None:
            try:
                loop.set_confidence_and_reset(0.7, timeout_s=0.1)
                result["ok"] = True
            except BaseException as exc:  # noqa: BLE001 - record whatever it raises
                result["error"] = exc

        loop.start()
        try:
            assert counter.wait_for(2)

            submitter = threading.Thread(target=submit)
            submitter.start()
            # The capture thread has entered reset() and is holding _reset_lock.
            assert tracker.reset_started.wait(_WAIT)
            # The submitter's 0.1 s timeout has long elapsed, yet it must be BLOCKED
            # on _reset_lock (the mutation is in flight) — not returned.
            threading.Event().wait(0.2)
            assert result == {}, "submitter abandoned an in-flight command"

            tracker.reset_gate.set()  # let the mutation complete
            submitter.join(_WAIT)
        finally:
            tracker.reset_gate.set()
            loop.stop()

        # The submitter returned SUCCESS (not a timeout), and the mutation it
        # requested was fully applied before it returned.
        assert result.get("ok") is True
        assert "error" not in result
        assert 0.7 in tracker.confidence_history
        assert tracker.reset_count >= 1


class TestB5ConcurrentCommandsAreSerialisedByCaptureLoop:
    def test_second_command_waits_instead_of_overwriting_the_first(self) -> None:
        cap = _HoldableReadCapture()
        tracker = _ConfidenceRecordingTracker()
        counter = _Counter()
        loop = _make_loop(
            tracker, cap, on_frame=lambda t, ids, roster, catalog, resolving: counter.hit()
        )
        results: list[str] = []

        def submit(value: float) -> None:
            loop.set_confidence_and_reset(value, timeout_s=1.0)
            results.append(f"{value:.1f}")

        loop.start()
        try:
            assert counter.wait_for(2)
            cap.hold()
            assert cap.wait_blocked()

            first = threading.Thread(target=submit, args=(0.6,))
            second = threading.Thread(target=submit, args=(0.7,))
            first.start()
            threading.Event().wait(0.05)  # first owns the pending command slot
            second.start()

            threading.Event().wait(0.1)
            assert 0.7 not in tracker.confidence_history
            assert second.is_alive()

            cap.resume()
            first.join(_WAIT)
            second.join(_WAIT)
        finally:
            cap.resume()
            loop.stop()

        assert results == ["0.6", "0.7"]
        assert tracker.confidence_history[-2:] == [0.6, 0.7]
        assert tracker.reset_count == 2
