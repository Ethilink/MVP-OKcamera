"""TR4 AC1-AC7: behavior contract for ``backend.postpass.PostPassJob``.

Written from ``docs/tasks/TR4-postpass.md`` (the frozen SPEC/Public interface)
and ``docs/RECORDING.md`` §Post-pass / §Post-pass specifics. Blind-TDD: this
file is written against the SPEC's behavior contract, not against any
particular implementation — the coder replaces ``backend/postpass.py``'s
Phase-0 stub without ever seeing this file.

No real ffmpeg/video decode/detector/camera. Everything the job touches is
injected:

- ``NumberedFrameReader`` — a ``cv2.VideoCapture``-shaped fake modeled on
  TR1's ``make_numbered_frames``/``decode_frame_index`` (``tests/
  recording_fakes.py``): ``read()`` yields the numbered frames in order, then
  ``(False, None)`` once exhausted (no repeat-last-frame, unlike ``tests.
  fakes.FakeCapture`` — postpass must see real EOF at ``frame_count``).
  ``_make_cap_factory`` wraps a zero-arg reader builder as the
  ``cap_factory(path)`` callable the job calls.
- ``_make_probe_fn`` — wraps a fixed ``VideoProbe`` as the injected
  ``probe_fn(path)`` callable.
- ``FakeDetector`` (T01, ``tests/fakes.py``) — queued ``sv.Detections``,
  records every frame + the ``confidence_threshold`` it was called with.
  ``RaisingDetector`` is this file's own variant that raises once its queue
  drains, to drive the AC5/AC6 mid-run-failure scenarios.
- ``RecordingVideoWriter`` / ``make_recording_writer_factory`` — a
  call-sequencing double matching ``VideoEntryWriter``'s exact constructor
  shape, logging ``init``/``write_selected_frames``/``add_frame``/
  ``finalize`` calls in order (used for AC1's call-sequencing assertions, AC3,
  AC4, AC7 — where the point is *what got called, in what order*, not the
  on-disk artifacts). AC1's sidecar-completeness assertion, AC2, AC5, and AC6
  use the REAL ``VideoEntryWriter`` (TR3) for genuine end-to-end,
  file-on-disk assertions.
- ``DetectorAccessSpy`` — wraps a detector and records every attribute name
  accessed on it (AC7: ``run()`` must touch the detector solely through
  ``.predict``).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import pytest
import supervision as sv

from backend.encoder import VideoProbe
from backend.postpass import PostPassJob
from backend.video_writer import VideoEntryWriter
from tests.fakes import FakeDetector, make_fake_dets
from tests.recording_fakes import decode_frame_index, make_numbered_frames

ENTRY_NAME = "entry1"
FPS = 24.0
W, H = 16, 16


def _wait_for(predicate, timeout=3.0, interval=0.02) -> bool:
    """Bounded poll (mirrors tests/test_recording_api.py's helper) -- never a
    bare/unbounded wait; a stuck job fails the assertion fast instead of
    hanging the suite."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Local fakes/doubles
# ---------------------------------------------------------------------------


class NumberedFrameReader:
    """``cv2.VideoCapture``-shaped fake yielding TR1's numbered frames in
    order, then ``(False, None)`` once exhausted — modeling a finished MP4
    with exactly ``len(frames)`` frames. Unlike ``tests.fakes.FakeCapture``,
    it does NOT repeat the last frame on drain: postpass must stop at real
    EOF, not read stale duplicates past ``frame_count``.
    """

    def __init__(self, frames):
        self._frames = list(frames)
        self._index = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API name
        return True

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _make_cap_factory(reader_builder):
    """Wrap a zero-arg reader builder as a ``cap_factory(path)`` callable,
    recording every path it was invoked with."""
    calls: list = []

    def factory(path):
        calls.append(path)
        return reader_builder()

    return factory, calls


def _make_probe_fn(probe: VideoProbe):
    """Wrap a fixed ``VideoProbe`` as the injected ``probe_fn`` callable,
    tolerant of either ``probe_fn(path)`` or ``probe_fn(path, **kwargs)``."""
    calls: list = []

    def probe_fn(*args, **kwargs):
        calls.append((args, kwargs))
        return probe

    return probe_fn, calls


class RaisingDetector:
    """Detector double that returns queued detections for its first calls,
    then raises once the queue drains — the AC5/AC6 mid-run failure fake."""

    def __init__(self, dets_before_raise, raise_message: str = "simulated detector failure"):
        self._queue = list(dets_before_raise)
        self._raise_message = raise_message
        self.seen_frames: list = []
        self.seen_thresholds: list = []

    def predict(self, frame, confidence_threshold=None):
        if not self._queue:
            raise RuntimeError(self._raise_message)
        self.seen_frames.append(frame)
        self.seen_thresholds.append(confidence_threshold)
        return self._queue.pop(0)


class DetectorAccessSpy:
    """Wraps a detector-like object, recording every attribute name accessed
    on the wrapper — so a test can assert ``run()`` touches the detector
    solely through ``.predict`` (AC7): no pause/resume/confidence_threshold
    attribute access from inside the job itself (TR5 brackets those around
    ``run()``, not ``run()`` itself)."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "accessed", set())

    def __getattr__(self, name):
        self.accessed.add(name)
        return getattr(self._inner, name)


class RecordingVideoWriter:
    """Call-sequencing double matching ``VideoEntryWriter``'s exact
    constructor shape. Logs every call (in order) into a shared list instead
    of writing real files — used where the ACs care about *what got called,
    with what args, in what order*, not on-disk artifacts."""

    def __init__(
        self,
        entry_dir,
        entry_name,
        model_version,
        *,
        video,
        keyframes,
        operator_threshold,
        mining_threshold,
        model_name="nf-detector",
        _log,
    ):
        self.entry_dir = Path(entry_dir)
        self.entry_name = entry_name
        self.model_version = model_version
        self.video = video
        self.keyframes = keyframes
        self.operator_threshold = operator_threshold
        self.mining_threshold = mining_threshold
        self.model_name = model_name
        self._log = _log
        self._log.append(
            (
                "init",
                {
                    "video": video,
                    "keyframes": keyframes,
                    "operator_threshold": operator_threshold,
                    "mining_threshold": mining_threshold,
                    "model_name": model_name,
                },
            )
        )

    def write_selected_frames(self) -> None:
        self._log.append(("write_selected_frames",))

    def add_frame(self, frame_number, frame, dets) -> None:
        self._log.append(("add_frame", frame_number, decode_frame_index(frame)))

    def finalize(self) -> None:
        self._log.append(("finalize",))


def make_recording_writer_factory():
    log: list = []

    def factory(*args, **kwargs):
        return RecordingVideoWriter(*args, _log=log, **kwargs)

    return factory, log


class ProgressObservingDetector:
    """Detector double that snapshots the job's ``status`` (state, done) at
    the moment of each ``predict()`` call, via a mutable holder assigned
    after both the detector and the job exist (AC3)."""

    def __init__(self, job_holder, dets_per_frame):
        self._job_holder = job_holder
        self._queue = list(dets_per_frame)
        self.observed: list = []

    def predict(self, frame, confidence_threshold=None):
        job = self._job_holder["job"]
        self.observed.append((job.status.state, job.status.done))
        return self._queue.pop(0) if self._queue else sv.Detections.empty()


class PausingObserverDetector:
    """U2 addition: detector double that calls ``job.pause()`` (from INSIDE a
    ``predict()`` call, on the worker thread) the moment a target number of
    frames have been predicted, then keeps predicting normally once resumed.
    This is what lets a test drive ``PostPassJob.pause()/resume()`` exactly
    like the real caller (TR5's ``/record/start`` handler) would: pause() only
    *requests* the yield, and ``run()`` itself parks on its own next
    read/predict/write check (never mid-predict) -- ``run()`` must therefore
    execute on a background thread so the parking doesn't deadlock the thread
    that called ``pause()``. Records every frame it saw (in order) so a test
    can prove, after resume, that the run processed 0..frame_count-1 exactly
    once each -- the decisive proof that resume continues from the PARKED
    index rather than restarting at 0 (INV-4)."""

    def __init__(self, job_holder, dets_per_frame, pause_after: int):
        self._job_holder = job_holder
        self._queue = list(dets_per_frame)
        self._pause_after = pause_after
        self.seen_frames: list = []
        self.paused_once = False

    def predict(self, frame, confidence_threshold=None):
        self.seen_frames.append(frame)
        if len(self.seen_frames) == self._pause_after and not self.paused_once:
            self.paused_once = True
            self._job_holder["job"].pause()
        return self._queue.pop(0) if self._queue else sv.Detections.empty()


# ---------------------------------------------------------------------------
# AC1 — every frame processed: predict() called frame_count times at
# mining_threshold, add_frame called once per frame_number 0..frame_count-1
# in order; with the real writer, full_frame_detections.json has all frames.
# ---------------------------------------------------------------------------


def test_ac1_predicts_every_frame_in_order_at_mining_threshold(tmp_path):
    frame_count = 6
    mining_threshold = 0.33
    detector = FakeDetector(predictions=[sv.Detections.empty()] * frame_count)
    writer_factory, log = make_recording_writer_factory()
    cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    probe_fn, _ = _make_probe_fn(VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H))

    job = PostPassJob(
        tmp_path / ENTRY_NAME,
        ENTRY_NAME,
        detector,
        keyframes=[],
        frame_count=frame_count,
        mining_threshold=mining_threshold,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=writer_factory,
    )

    job.run()

    assert len(detector.seen_frames) == frame_count
    assert detector.seen_thresholds == [mining_threshold] * frame_count
    assert [decode_frame_index(f) for f in detector.seen_frames] == list(range(frame_count))

    add_frame_records = [(entry[1], entry[2]) for entry in log if entry[0] == "add_frame"]
    assert add_frame_records == [(i, i) for i in range(frame_count)]


def test_ac1_full_frame_detections_json_has_every_processed_frame(tmp_path):
    frame_count = 5
    entry_dir = tmp_path / ENTRY_NAME
    detector = FakeDetector(predictions=[sv.Detections.empty()] * frame_count)
    cap_factory, cap_calls = _make_cap_factory(
        lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H))
    )
    probe_fn, probe_calls = _make_probe_fn(VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H))

    job = PostPassJob(
        entry_dir,
        ENTRY_NAME,
        detector,
        keyframes=[],
        frame_count=frame_count,
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=VideoEntryWriter,
    )

    job.run()

    assert job.status.state == "done"

    # Both the reader and the probe must be pointed at the recorded MP4 at the
    # exact §video-file path — a wrong path only breaks against real cv2, so gate
    # it here where cap_factory/probe_fn are fakes that ignore the path.
    expected_video_path = entry_dir / "video" / f"{ENTRY_NAME}.mp4"
    assert cap_calls == [str(expected_video_path)]
    assert len(probe_calls) == 1
    probe_args, _probe_kwargs = probe_calls[0]
    assert probe_args[0] == str(expected_video_path)

    sidecar = json.loads(
        (entry_dir / "annotations" / "metadata" / "full_frame_detections.json").read_text()
    )
    assert [f["frame_number"] for f in sidecar["frames"]] == list(range(frame_count))


# ---------------------------------------------------------------------------
# AC2 — keyframe outputs: JPEG + COCO records filtered at operator_threshold;
# non-keyframes contribute only to the sidecar.
# ---------------------------------------------------------------------------


def test_ac2_keyframe_gets_jpeg_and_operator_filtered_coco_record(tmp_path):
    frame_count = 3
    keyframe = 1
    operator_threshold = 0.5
    mining_threshold = 0.2
    model_version = "model-v2"
    entry_dir = tmp_path / ENTRY_NAME

    # Keyframe carries three detections at distinct boxes so each is
    # identifiable by bbox: 0.9 (above operator), 0.3 (in the mining↔operator
    # GAP), 0.1 (below mining). The gap detection is the crux: it must be
    # EXCLUDED from annotations.json (proves the keyframe filter is the
    # operator threshold, not the mining threshold — a threshold-swap bug that
    # fed mining into the operator slot would wrongly keep 0.3), yet PRESENT in
    # the sidecar (proves the mining-path det still reached the writer).
    kf_09_box = [2, 2, 10, 10]   # -> bbox [2, 2, 8, 8]
    kf_03_box = [3, 3, 12, 12]   # -> bbox [3, 3, 9, 9]  (in the gap)
    kf_01_box = [4, 4, 9, 9]     # -> bbox [4, 4, 5, 5]

    dets_by_frame = []
    for fn in range(frame_count):
        if fn == keyframe:
            dets_by_frame.append(
                make_fake_dets(
                    [kf_09_box, kf_03_box, kf_01_box],
                    confidences=[0.9, 0.3, 0.1],
                    size=(W, H),
                )
            )
        else:
            dets_by_frame.append(make_fake_dets([[1, 1, 5, 5]], confidences=[0.3], size=(W, H)))

    detector = FakeDetector(predictions=dets_by_frame)
    cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    probe_fn, _ = _make_probe_fn(VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H))

    job = PostPassJob(
        entry_dir,
        ENTRY_NAME,
        detector,
        keyframes=[keyframe],
        frame_count=frame_count,
        mining_threshold=mining_threshold,
        operator_threshold=operator_threshold,
        model_version=model_version,
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=VideoEntryWriter,
    )

    job.run()

    assert job.status.state == "done"

    jpg_path = entry_dir / "images" / f"{ENTRY_NAME}_f{keyframe:06d}.jpg"
    assert jpg_path.is_file()
    saved = cv2.imread(str(jpg_path))
    assert saved is not None
    assert saved.shape == (H, W, 3)

    jpg_names = {p.name for p in (entry_dir / "images").glob("*.jpg")}
    assert jpg_names == {f"{ENTRY_NAME}_f{keyframe:06d}.jpg"}, "only the keyframe gets a JPEG"

    ann_data = json.loads((entry_dir / "annotations" / "annotations.json").read_text())
    assert [img["frame_number"] for img in ann_data["images"]] == [keyframe]
    image_id = ann_data["images"][0]["id"]
    anns = [a for a in ann_data["annotations"] if a["image_id"] == image_id]
    # Only the 0.9 detection clears the operator threshold; the 0.3 gap
    # detection must NOT appear (this is what fails a mining/operator swap).
    assert len(anns) == 1, "only the above-operator-threshold detection survives in annotations.json"
    assert anns[0]["bbox"] == pytest.approx([2.0, 2.0, 8.0, 8.0])
    ann_bboxes = [tuple(a["bbox"]) for a in anns]
    assert (3.0, 3.0, 9.0, 9.0) not in ann_bboxes, "the mining↔operator gap det must be excluded from annotations"

    sidecar = json.loads(
        (entry_dir / "annotations" / "metadata" / "full_frame_detections.json").read_text()
    )

    # model block flows only into the sidecar — gate it here.
    assert sidecar["model"]["conf_threshold"] == mining_threshold
    assert sidecar["model"]["version"] == model_version

    kf_entry = next(f for f in sidecar["frames"] if f["frame_number"] == keyframe)
    kf_bboxes = [tuple(d["bbox"]) for d in kf_entry["detections"]]
    # The mining-path detections reached the writer: both the operator-kept 0.9
    # AND the gap 0.3 are in the keyframe's sidecar entry (superset check — the
    # 0.1 below-mining det may or may not survive real mining filtering).
    assert (2.0, 2.0, 8.0, 8.0) in kf_bboxes, "the operator-kept det is also in the sidecar"
    assert (3.0, 3.0, 9.0, 9.0) in kf_bboxes, "the mining↔operator gap det must be present in the sidecar"

    non_keyframes = [fn for fn in range(frame_count) if fn != keyframe]
    for fn in non_keyframes:
        frame_entry = next(f for f in sidecar["frames"] if f["frame_number"] == fn)
        assert len(frame_entry["detections"]) == 1, "non-keyframe detections land only in the sidecar"


# ---------------------------------------------------------------------------
# AC3 — progress: status.done advances 0 -> frame_count during the run;
# status.state goes pending -> running -> done.
# ---------------------------------------------------------------------------


def test_ac3_progress_advances_and_state_transitions(tmp_path):
    frame_count = 4
    job_holder: dict = {}
    detector = ProgressObservingDetector(job_holder, [sv.Detections.empty()] * frame_count)
    writer_factory, _log = make_recording_writer_factory()
    cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    probe_fn, _ = _make_probe_fn(VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H))

    job = PostPassJob(
        tmp_path / ENTRY_NAME,
        ENTRY_NAME,
        detector,
        keyframes=[],
        frame_count=frame_count,
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=writer_factory,
    )
    job_holder["job"] = job

    assert job.status.state == "pending"

    job.run()

    assert detector.observed == [("running", i) for i in range(frame_count)]
    assert job.status.state == "done"
    assert job.status.done == frame_count
    assert job.status.total == frame_count


# ---------------------------------------------------------------------------
# AC4 — frame_count reconciliation: a probe/authoritative mismatch is
# resolved in favor of the authoritative constructor arg, and the run still
# processes exactly frame_count frames.
# ---------------------------------------------------------------------------


def test_ac4_frame_count_reconciliation_prefers_authoritative_value(tmp_path):
    authoritative_frame_count = 7
    probe_frame_count = 5  # deliberately mismatched (re-encode gotcha)
    detector = FakeDetector(predictions=[sv.Detections.empty()] * authoritative_frame_count)
    writer_factory, log = make_recording_writer_factory()
    cap_factory, _ = _make_cap_factory(
        lambda: NumberedFrameReader(make_numbered_frames(authoritative_frame_count, W, H))
    )
    probe_fn, _ = _make_probe_fn(VideoProbe(fps=30.0, frame_count=probe_frame_count, width=W, height=H))

    job = PostPassJob(
        tmp_path / ENTRY_NAME,
        ENTRY_NAME,
        detector,
        keyframes=[],
        frame_count=authoritative_frame_count,
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=writer_factory,
    )

    job.run()

    init_calls = [entry for entry in log if entry[0] == "init"]
    assert len(init_calls) == 1
    video_block = init_calls[0][1]["video"]
    assert video_block["frame_count"] == authoritative_frame_count
    assert video_block["fps"] == 30.0
    assert video_block["width"] == W
    assert video_block["height"] == H

    add_frame_records = [(entry[1], entry[2]) for entry in log if entry[0] == "add_frame"]
    assert add_frame_records == [(i, i) for i in range(authoritative_frame_count)]

    assert job.status.total == authoritative_frame_count
    assert job.status.state == "done"


# ---------------------------------------------------------------------------
# AC5 — failure -> partial: a mid-run detector.predict exception does not
# propagate; status is "failed" with the error message, selected_frames.json
# exists, annotations.json does not.
# ---------------------------------------------------------------------------


def test_ac5_failure_leaves_valid_partial_project(tmp_path):
    frame_count = 5
    keyframes = [1]
    entry_dir = tmp_path / ENTRY_NAME
    raise_message = "boom: detector exploded"

    detector = RaisingDetector(
        dets_before_raise=[
            make_fake_dets([], size=(W, H)),
            make_fake_dets([[1, 1, 5, 5]], confidences=[0.9], size=(W, H)),
        ],
        raise_message=raise_message,
    )
    cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    probe_fn, _ = _make_probe_fn(VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H))

    job = PostPassJob(
        entry_dir,
        ENTRY_NAME,
        detector,
        keyframes=keyframes,
        frame_count=frame_count,
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=VideoEntryWriter,
    )

    job.run()  # must not raise

    assert job.status.state == "failed"
    assert job.status.error is not None
    assert raise_message in job.status.error

    assert (entry_dir / "annotations" / "metadata" / "selected_frames.json").is_file()
    assert not (entry_dir / "annotations" / "annotations.json").exists()


# ---------------------------------------------------------------------------
# AC6 (spec AC7) — idempotent retry: a fresh PostPassJob after a failed run,
# run to completion, reproduces an uninterrupted run's output byte-for-byte.
# ---------------------------------------------------------------------------


def test_ac6_retry_after_failure_reproduces_uninterrupted_output(tmp_path):
    frame_count = 6
    keyframes = [1, 4]
    operator_threshold = 0.5
    mining_threshold = 0.2
    model_version = "model-v1"
    probe = VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H)

    def _dets_sequence():
        seq = []
        for fn in range(frame_count):
            if fn in keyframes:
                seq.append(make_fake_dets([[2, 2, 10, 10]], confidences=[0.9], size=(W, H)))
            else:
                seq.append(make_fake_dets([], size=(W, H)))
        return seq

    # --- uninterrupted reference run ---
    direct_dir = tmp_path / "direct" / ENTRY_NAME
    direct_cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    direct_probe_fn, _ = _make_probe_fn(probe)
    direct_job = PostPassJob(
        direct_dir,
        ENTRY_NAME,
        FakeDetector(predictions=_dets_sequence()),
        keyframes=keyframes,
        frame_count=frame_count,
        mining_threshold=mining_threshold,
        operator_threshold=operator_threshold,
        model_version=model_version,
        probe_fn=direct_probe_fn,
        cap_factory=direct_cap_factory,
        video_writer_factory=VideoEntryWriter,
    )
    direct_job.run()
    assert direct_job.status.state == "done"

    ann_direct = (direct_dir / "annotations" / "annotations.json").read_bytes()
    side_direct = (direct_dir / "annotations" / "metadata" / "full_frame_detections.json").read_bytes()
    jpgs_direct = {
        fn: (direct_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").read_bytes() for fn in keyframes
    }

    # --- interrupted run, then a fresh retry on the SAME entry_dir ---
    retry_dir = tmp_path / "retry" / ENTRY_NAME
    failing_cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    failing_probe_fn, _ = _make_probe_fn(probe)
    failing_job = PostPassJob(
        retry_dir,
        ENTRY_NAME,
        RaisingDetector(dets_before_raise=_dets_sequence()[:3], raise_message="boom"),
        keyframes=keyframes,
        frame_count=frame_count,
        mining_threshold=mining_threshold,
        operator_threshold=operator_threshold,
        model_version=model_version,
        probe_fn=failing_probe_fn,
        cap_factory=failing_cap_factory,
        video_writer_factory=VideoEntryWriter,
    )
    failing_job.run()
    assert failing_job.status.state == "failed"

    retry_cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    retry_probe_fn, _ = _make_probe_fn(probe)
    retry_job = PostPassJob(
        retry_dir,
        ENTRY_NAME,
        FakeDetector(predictions=_dets_sequence()),
        keyframes=keyframes,
        frame_count=frame_count,
        mining_threshold=mining_threshold,
        operator_threshold=operator_threshold,
        model_version=model_version,
        probe_fn=retry_probe_fn,
        cap_factory=retry_cap_factory,
        video_writer_factory=VideoEntryWriter,
    )
    retry_job.run()
    assert retry_job.status.state == "done"

    ann_retry = (retry_dir / "annotations" / "annotations.json").read_bytes()
    side_retry = (retry_dir / "annotations" / "metadata" / "full_frame_detections.json").read_bytes()
    jpgs_retry = {
        fn: (retry_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").read_bytes() for fn in keyframes
    }

    assert ann_direct == ann_retry
    assert side_direct == side_retry
    assert jpgs_direct == jpgs_retry


# ---------------------------------------------------------------------------
# AC7 — detector-only interaction: run() touches the detector solely through
# predict(...); no other attribute (pause/resume/etc.) is accessed.
# ---------------------------------------------------------------------------


def test_ac7_run_touches_detector_only_through_predict(tmp_path):
    frame_count = 3
    inner = FakeDetector(predictions=[sv.Detections.empty()] * frame_count, confidence_threshold=0.5)
    spy = DetectorAccessSpy(inner)
    writer_factory, _log = make_recording_writer_factory()
    cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    probe_fn, _ = _make_probe_fn(VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H))

    job = PostPassJob(
        tmp_path / ENTRY_NAME,
        ENTRY_NAME,
        spy,
        keyframes=[],
        frame_count=frame_count,
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=writer_factory,
    )

    job.run()

    assert spy.accessed == {"predict"}
    assert len(inner.seen_frames) == frame_count


# ---------------------------------------------------------------------------
# U2 additions — PostPassJob.pause()/resume() (docs/tasks/U2-queue.md §4
# "PostPassJob additions", INV-4). Additive: cancel()/run()/AC7 semantics
# above are unchanged by these tests. ``run()`` executes on a background
# thread in every test below because pause() only *requests* the yield --
# the park happens inside run()'s own loop, on ITS thread, so the caller that
# invokes pause()/resume() must be a different thread (exactly mirroring the
# real caller: TR5's ``/record/start`` handler pausing the drain worker).
# ---------------------------------------------------------------------------


def test_pause_resume_continues_from_parked_index_byte_identical_to_uninterrupted(tmp_path):
    """INV-4's core claim: a job paused mid-run and resumed produces output
    byte-identical to an uninterrupted run, because it resumes from the
    PARKED frame index rather than restarting at 0. Proven two ways: (1) the
    detector saw frames 0..frame_count-1 exactly once each, in order (a
    restart-from-0 bug would duplicate frames 0..pause_after-1); (2) the
    on-disk artifacts (annotations.json, sidecar, keyframe JPEGs) are
    byte-for-byte identical to a reference job given the same inputs.
    """
    frame_count = 8
    keyframes = [2, 5]
    operator_threshold = 0.5
    mining_threshold = 0.2
    model_version = "model-pause"
    probe = VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H)

    def _dets_sequence():
        seq = []
        for fn in range(frame_count):
            if fn in keyframes:
                seq.append(make_fake_dets([[2, 2, 10, 10]], confidences=[0.9], size=(W, H)))
            else:
                seq.append(make_fake_dets([], size=(W, H)))
        return seq

    # --- uninterrupted reference run ---
    direct_dir = tmp_path / "direct" / ENTRY_NAME
    direct_cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    direct_probe_fn, _ = _make_probe_fn(probe)
    direct_job = PostPassJob(
        direct_dir,
        ENTRY_NAME,
        FakeDetector(predictions=_dets_sequence()),
        keyframes=keyframes,
        frame_count=frame_count,
        mining_threshold=mining_threshold,
        operator_threshold=operator_threshold,
        model_version=model_version,
        probe_fn=direct_probe_fn,
        cap_factory=direct_cap_factory,
        video_writer_factory=VideoEntryWriter,
    )
    direct_job.run()
    assert direct_job.status.state == "done"

    ann_direct = (direct_dir / "annotations" / "annotations.json").read_bytes()
    side_direct = (direct_dir / "annotations" / "metadata" / "full_frame_detections.json").read_bytes()
    jpgs_direct = {
        fn: (direct_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").read_bytes() for fn in keyframes
    }

    # --- paused-and-resumed run, on the SAME entry_dir shape ---
    paused_dir = tmp_path / "paused" / ENTRY_NAME
    paused_cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    paused_probe_fn, _ = _make_probe_fn(probe)
    job_holder: dict = {}
    detector = PausingObserverDetector(job_holder, _dets_sequence(), pause_after=3)
    job = PostPassJob(
        paused_dir,
        ENTRY_NAME,
        detector,
        keyframes=keyframes,
        frame_count=frame_count,
        mining_threshold=mining_threshold,
        operator_threshold=operator_threshold,
        model_version=model_version,
        probe_fn=paused_probe_fn,
        cap_factory=paused_cap_factory,
        video_writer_factory=VideoEntryWriter,
    )
    job_holder["job"] = job

    worker = threading.Thread(target=job.run, daemon=True)
    worker.start()

    # Parked after exactly 3 frames processed (frame_number 0,1,2 written);
    # bounded wait, never sleep-and-hope.
    assert _wait_for(lambda: job.status.state == "paused", timeout=3.0)
    assert job.status.done == 3

    # Prove it really is parked: done does NOT creep forward while paused.
    assert _wait_for(lambda: job.status.done != 3, timeout=0.3) is False

    job.resume()
    worker.join(timeout=5.0)
    assert not worker.is_alive()

    assert job.status.state == "done"
    assert job.status.done == frame_count

    # (1) exactly frame_count predicts, each frame decoded exactly once, in
    # order -- a restart-from-0 bug would show frames 0,1,2 twice.
    assert [decode_frame_index(f) for f in detector.seen_frames] == list(range(frame_count))

    # (2) byte-identical on-disk artifacts vs the uninterrupted reference.
    ann_paused = (paused_dir / "annotations" / "annotations.json").read_bytes()
    side_paused = (paused_dir / "annotations" / "metadata" / "full_frame_detections.json").read_bytes()
    jpgs_paused = {
        fn: (paused_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").read_bytes() for fn in keyframes
    }
    assert ann_paused == ann_direct
    assert side_paused == side_direct
    assert jpgs_paused == jpgs_direct


def test_pause_state_transitions_pending_running_paused_running_done():
    """``status.state`` gains "paused" while parked; existing states
    (pending/running/done) are otherwise unchanged -- a plain state-string
    contract test, independent of the byte-identical proof above."""
    frame_count = 4
    job_holder: dict = {}
    detector = PausingObserverDetector(
        job_holder, [sv.Detections.empty()] * frame_count, pause_after=2
    )
    cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    probe_fn, _ = _make_probe_fn(VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H))
    writer_factory, _log = make_recording_writer_factory()

    job = PostPassJob(
        Path("unused"),
        ENTRY_NAME,
        detector,
        keyframes=[],
        frame_count=frame_count,
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=writer_factory,
    )
    job_holder["job"] = job

    assert job.status.state == "pending"

    worker = threading.Thread(target=job.run, daemon=True)
    worker.start()
    assert _wait_for(lambda: job.status.state == "paused", timeout=3.0)

    job.resume()
    # Immediately after resume, run() must still be alive and finish on its
    # own (never re-pending, never re-raise) -- state eventually settles on
    # "done" without a second pause (pause_after fires only once per detector).
    worker.join(timeout=3.0)
    assert not worker.is_alive()
    assert job.status.state == "done"
    assert job.status.done == frame_count


def test_cancel_while_paused_stops_the_job_without_completing(tmp_path):
    """A job discarded WHILE paused (operator starts a new take, then
    discards it before it ever goes back to idle) must still cancel cleanly --
    park must not swallow a concurrent cancel() (INV-5 / discard robustness)."""
    frame_count = 6
    job_holder: dict = {}
    detector = PausingObserverDetector(
        job_holder, [sv.Detections.empty()] * frame_count, pause_after=2
    )
    cap_factory, _ = _make_cap_factory(lambda: NumberedFrameReader(make_numbered_frames(frame_count, W, H)))
    probe_fn, _ = _make_probe_fn(VideoProbe(fps=FPS, frame_count=frame_count, width=W, height=H))
    writer_factory, _log = make_recording_writer_factory()

    job = PostPassJob(
        tmp_path / ENTRY_NAME,
        ENTRY_NAME,
        detector,
        keyframes=[],
        frame_count=frame_count,
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="v1",
        probe_fn=probe_fn,
        cap_factory=cap_factory,
        video_writer_factory=writer_factory,
    )
    job_holder["job"] = job

    worker = threading.Thread(target=job.run, daemon=True)
    worker.start()
    assert _wait_for(lambda: job.status.state == "paused", timeout=3.0)

    job.cancel()
    worker.join(timeout=3.0)
    assert not worker.is_alive()
    assert job.status.state == "cancelled"
    assert job.status.done == 2  # bailed at the parked index, never resumed forward
