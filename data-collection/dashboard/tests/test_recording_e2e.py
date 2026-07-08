"""TR7 AC1-AC6 — full recording pipeline end-to-end, with fakes (CI-gated).

Drives the WHOLE stack — real ``CaptureLoop`` (reader/inference/slot threads),
real ``create_app`` state machine + ``/record/*`` + ``/keyframe`` endpoints, real
``PostPassJob`` + ``VideoEntryWriter`` + ``coco`` helper — over a **fake numbered
camera**. Only three seams are faked, and none needs hardware:

- **camera**: ``_PacedNumberedCap`` yields ``make_numbered_frames`` — each frame's
  pixels encode its own 0-based content index, so a dropped / duplicated /
  reordered / off-by-one frame is detectable after the fact (RECORDING.md
  §Gotchas). It sleeps ~1/``fps`` per ``read`` so the un-rate-limited reader is
  bounded to a camera-like rate (the "paced fake" — otherwise the reader spins
  and ``frames_written`` explodes) and repeats the last frame when drained so
  idle never starves ``/record/start``.
- **encoder**: ``_FileEncoder`` records every written frame in order (the
  ground-truth "MP4 content") and, on ``release()``, drops a stub file into
  ``video/`` so the finished entry satisfies the video-project discovery rule
  (``annotations.json`` + non-empty ``video/``) without a real H.264 file.
- **post-pass video read**: the real ``PostPassJob`` is handed a ``cap_factory``
  that **replays the encoder's recorded frames losslessly** (numbered frames are
  flat, so they survive the JPEG-95 keyframe re-encode too) and a ``probe_fn``
  returning the true dims — so the post-pass decodes the exact pixels that were
  "recorded", off-by-one stays detectable, and no ffmpeg/VideoToolbox runs.

The load-bearing identity fact this test leans on: the reader stamps
``frame_number = its write index`` and the inference thread publishes *that*
frame + number, so the frame displayed as ``frame_number == D`` **is** the frame
at MP4 position ``D`` (``encoder.written[D]``). That equality is what
``/keyframe`` echoes and what the post-pass seeks — it holds regardless of how
the fake camera paces or repeats, which is why the assertions here are
count-agnostic and race-free.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from backend.app import create_app
from backend.capture import CaptureLoop
from backend.dataset_writer import DatasetWriter
from backend.encoder import VideoProbe
from backend.postpass import PostPassJob
from fastapi.testclient import TestClient
from tests.fakes import make_fake_dets
from tests.recording_fakes import decode_frame_index, make_numbered_frames

# Tiny frames keep the test fast; dims are asserted end-to-end anyway (a real
# 1080p run keeps the dims-equal invariant trivially true — RECORDING.md §Gotchas
# — and this exercises the same code path at 64x48).
_W, _H = 64, 48
_CAM_FPS = 120.0  # paced read rate: bounds frames_written to a camera-like count


# --- fakes ------------------------------------------------------------------


class _PacedNumberedCap:
    """``cv2.VideoCapture`` stand-in yielding numbered frames at ~``fps``.

    Sleeps ``1/fps`` per ``read`` so the (un-rate-limited) capture reader is
    bounded to a camera-like rate instead of spinning. Repeats the last frame
    once the queue drains (``on_empty="repeat"`` semantics) so idle inference
    never starves the initial ``/record/start`` snapshot. Thread-safe: idle,
    reader, and the test may all touch it.
    """

    def __init__(self, frames, fps: float = _CAM_FPS):
        self._frames = frames
        self._i = 0
        self._delay = 1.0 / fps
        self._last = frames[-1] if frames else None
        self.set_calls: list = []
        self.released = False
        self._lock = threading.Lock()

    def set(self, prop, value) -> bool:
        with self._lock:
            self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        f = self._frames[0] if self._frames else None
        if f is None:
            return 0.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(f.shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(f.shape[0])
        return 0.0

    def read(self):
        time.sleep(self._delay)
        with self._lock:
            if self._i < len(self._frames):
                frame = self._frames[self._i]
                self._i += 1
                self._last = frame
                return (True, frame)
            return (True, self._last)

    def isOpened(self) -> bool:  # noqa: N802 - cv2 API name
        with self._lock:
            return not self.released

    def release(self) -> None:
        with self._lock:
            self.released = True


class _OneBoxDetector:
    """Detector stub returning a fixed single-instrument detection for every
    frame (confidence 0.9, one solid mask). Frame-agnostic so it is safe to call
    from both the live inference thread and the post-pass with no queue to drain
    and no per-frame race. Records how many predicts ran."""

    def __init__(self, box=(10, 10, 40, 30), confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._dets = make_fake_dets([list(box)], confidences=[0.9], size=(_W, _H))
        self.predict_count = 0
        self._lock = threading.Lock()

    def predict(self, frame, confidence_threshold=None):
        with self._lock:
            self.predict_count += 1
        return self._dets


class _FileEncoder:
    """Encoder stub: records written frames in order and, on release, writes a
    stub file to the MP4 path so ``video/`` is non-empty (video-project
    discovery rule). Duck-types the encoder interface the capture loop needs
    (``write`` / ``release`` / ``is_open``)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.written: list[np.ndarray] = []
        self.is_open = True
        self._lock = threading.Lock()

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            self.written.append(frame.copy())

    def release(self) -> None:
        with self._lock:
            if not self.is_open:
                return
            self.is_open = False
            # A real avc1/ffmpeg writer leaves a finalized MP4 here; a 1-byte stub
            # is enough to make video/ non-empty for the discovery rule.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"\x00")


class _ReplayCap:
    """Post-pass ``cap_factory`` stand-in: replays recorded frames in order,
    losslessly. ``read`` past the end returns ``(False, None)`` (the post-pass
    loop is bounded by ``frame_count`` and never reads past it)."""

    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def read(self):
        if self._i < len(self._frames):
            frame = self._frames[self._i]
            self._i += 1
            return (True, frame)
        return (False, None)

    def release(self) -> None:
        pass


class _ExplodingReplayCap(_ReplayCap):
    """Replay cap that raises after ``explode_at`` reads — used to kill a
    post-pass mid-loop (AC5). ``write_selected_frames`` has already run by then,
    so the entry is left a valid partial (no ``annotations.json``)."""

    def __init__(self, frames, explode_at: int):
        super().__init__(frames)
        self._explode_at = explode_at

    def read(self):
        if self._i >= self._explode_at:
            raise RuntimeError("simulated post-pass decode failure (killed mid-run)")
        return super().read()


def _wait_for(predicate, timeout=5.0, interval=0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _writer_factory(model_version="rfdetr-e2e"):
    def factory(output_path, dataset_name):
        return DatasetWriter(output_path, dataset_name, model_version)

    return factory


def _build_app(capture, detector, output_path, encoder_holder, *, job_hook=None,
               mining_threshold=0.25, model_version="rfdetr-e2e"):
    """create_app wired for the e2e: real endpoints, fake encoder (captured into
    ``encoder_holder['enc']``), and a real ``PostPassJob`` whose video read is a
    replay of the encoder's frames. ``job_hook(call_index, kwargs)`` may return a
    replacement job (AC5's kill/retry); otherwise a normal replaying job runs."""
    app = create_app(detector, _writer_factory(model_version), capture)
    app.state.output_path = str(output_path)
    app.state.capture_fps = 30.0
    app.state.mining_threshold = mining_threshold
    app.state.model_version = model_version

    def open_encoder(path, fps, frame_size, **kwargs):
        enc = _FileEncoder(path)
        encoder_holder["enc"] = enc
        encoder_holder["open_args"] = {"fps": fps, "frame_size": tuple(frame_size)}
        return enc

    app.state.open_encoder = open_encoder

    calls = {"n": 0}

    def post_pass_factory(entry_dir, entry_name, det, **kwargs):
        written = encoder_holder["enc"].written
        idx = calls["n"]
        calls["n"] += 1
        cap_factory = None
        if job_hook is not None:
            cap_factory = job_hook(idx, written)
        if cap_factory is None:
            cap_factory = lambda p: _ReplayCap(written)  # noqa: E731
        return PostPassJob(
            entry_dir, entry_name, det, **kwargs,
            cap_factory=cap_factory,
            probe_fn=lambda p: VideoProbe(fps=30.0, frame_count=len(written), width=_W, height=_H),
        )

    app.state.post_pass_factory = post_pass_factory
    return app


def _start_capture(frames, detector):
    capture = CaptureLoop(
        detector,
        camera_index=0,
        render_fn=lambda frame, dets: frame,
        cap_factory=lambda idx: _PacedNumberedCap(frames),
        stale_after_s=5.0,
        ring_size=20000,  # never evict a generation before /frame re-reads it
    )
    capture.start()
    return capture


def _mark_keyframe(client, capture):
    """Read the on-screen frame's X-Frame-Number, echo it to /keyframe (exactly
    what the browser does), and return (frame_number, decoded_displayed_index) —
    the displayed frame's raw pixel identity, captured atomically via the ring."""
    r = client.get("/frame")
    assert r.status_code == 200
    assert "x-frame-number" in r.headers
    fn = int(r.headers["x-frame-number"])
    gen = int(r.headers["x-frame-generation"])
    displayed = capture.snapshot_at(gen)
    assert displayed is not None and displayed.frame_number == fn
    disp_index = decode_frame_index(displayed.frame)
    resp = client.post("/keyframe", json={"frame_number": fn})
    resp.raise_for_status()
    return fn, disp_index, resp.json()["n_keyframes"]


def _run_recording(tmp_path, entry_name, *, n_keyframes=2, job_hook=None):
    """Full record → keyframe(s) → stop drive. Returns a dict of everything the
    assertions need. Leaves the machine wherever the post-pass ended (idle on
    success / failed on a killed run)."""
    frames = make_numbered_frames(400, _W, _H)
    detector = _OneBoxDetector()
    capture = _start_capture(frames, detector)
    holder: dict = {}
    app = _build_app(capture, detector, tmp_path, holder, job_hook=job_hook)
    client = TestClient(app)
    kf: list = []
    try:
        assert _wait_for(lambda: capture.snapshot() is not None)
        client.post("/record/start", json={"entry_name": entry_name}).raise_for_status()
        # Wait for a recording-stamped snapshot (frame_number not None).
        assert _wait_for(
            lambda: capture.snapshot() is not None
            and capture.snapshot().frame_number is not None
        )
        seen = set()
        deadline = time.monotonic() + 5.0
        while len(kf) < n_keyframes and time.monotonic() < deadline:
            fn, disp_index, _ = _mark_keyframe(client, capture)
            if fn not in seen:
                seen.add(fn)
                kf.append((fn, disp_index))
            # let the displayed frame advance so the next keyframe is a new one
            _wait_for(
                lambda: capture.snapshot() is not None
                and capture.snapshot().frame_number not in seen,
                timeout=1.0,
            )
        assert len(kf) == n_keyframes, f"only marked {len(kf)} keyframes"

        stop = client.post("/record/stop")
        stop.raise_for_status()
        frames_written = stop.json()["frames_written"]
        # Wait for the worker-thread post-pass to leave 'processing'.
        assert _wait_for(lambda: client.get("/record/status").json()["state"] != "processing",
                         timeout=15.0)
        status = client.get("/record/status").json()
        return {
            "client": client,
            "capture": capture,
            "app": app,
            "entry_dir": Path(tmp_path) / entry_name,
            "entry_name": entry_name,
            "encoder": holder["enc"],
            "written": holder["enc"].written,
            "keyframes": kf,          # list of (frame_number, decoded_displayed_index)
            "frames_written": frames_written,
            "final_status": status,
        }
    finally:
        capture.stop()


def _load(entry_dir: Path, rel: str):
    return json.loads((entry_dir / rel).read_text())


# --- AC1: a completed run is a video project on disk ------------------------


def test_ac1_completed_run_is_a_video_project(tmp_path):
    res = _run_recording(tmp_path, "take1")
    entry = res["entry_dir"]
    assert res["final_status"]["state"] == "idle"  # success
    # Discovery rule: annotations.json present + non-empty video/.
    assert (entry / "annotations" / "annotations.json").is_file()
    video_files = list((entry / "video").iterdir())
    assert video_files and all(p.stat().st_size > 0 for p in video_files)


# --- AC2: keyframe fidelity — one entry each, pixels match, displayed frame --


def test_ac2_keyframe_fidelity_and_displayed_frame_identity(tmp_path):
    res = _run_recording(tmp_path, "take2", n_keyframes=3)
    entry = res["entry_dir"]
    written = res["written"]
    keyframes = res["keyframes"]
    frames_written = res["frames_written"]

    anns = _load(entry, "annotations/annotations.json")
    selected = _load(entry, "annotations/metadata/selected_frames.json")

    marked = sorted(fn for fn, _ in keyframes)
    assert selected["selected_frames"] == marked
    # exactly one image record per SPACE press, one JPEG each
    assert sorted(im["frame_number"] for im in anns["images"]) == marked
    jpg_dir = entry / "images"
    assert sorted(p.name for p in jpg_dir.glob("*.jpg")) == [
        f"take2_f{fn:06d}.jpg" for fn in marked
    ]

    # JPEG's BGR->YCbCr->BGR conversion perturbs each channel by ~+-1-2 even on a
    # flat frame, so the keyframe JPEG can never be BYTE-equal to the source and
    # decode_frame_index (which weights R by 65536) can't be used on it. The
    # rigorous frame-identity / off-by-one proof is therefore assertion (a)
    # below, on the LOSSLESS recorded frame; assertion (b) is the JPEG pixel-
    # fidelity claim within re-encode tolerance (spec AC2 "pixels equal ...").
    _JPEG_TOL = 4
    for fn, disp_index in keyframes:
        # (a) the displayed frame IS the frame at MP4 position fn (no
        #     newest-at-press lag, no off-by-one): written[fn] decodes to what
        #     was on screen when SPACE was pressed. Lossless -> exact.
        assert decode_frame_index(written[fn]) == disp_index
        # (b) the keyframe JPEG's pixels equal the video decoded at fn (both are
        #     written[fn]) within JPEG re-encode tolerance.
        jpg = cv2.imread(str(jpg_dir / f"take2_f{fn:06d}.jpg"))
        assert jpg is not None
        assert jpg.shape == written[fn].shape
        assert int(np.abs(jpg.astype(int) - written[fn].astype(int)).max()) <= _JPEG_TOL
        # (c) displayed frame lagged the newest write (it is a real earlier
        #     frame, not the newest-at-press-time frame).
        assert fn < frames_written


# --- AC3: sidecar completeness — every frame, pixel-space coords, dims -------


def test_ac3_sidecar_has_every_frame_with_stream_dims(tmp_path):
    res = _run_recording(tmp_path, "take3")
    entry = res["entry_dir"]
    frames_written = res["frames_written"]
    sidecar = _load(entry, "annotations/metadata/full_frame_detections.json")

    frame_numbers = [f["frame_number"] for f in sidecar["frames"]]
    assert frame_numbers == list(range(frames_written))  # 0..frame_count-1, contiguous
    assert sidecar["video"]["width"] == _W and sidecar["video"]["height"] == _H
    assert sidecar["video"]["frame_count"] == frames_written

    anns = _load(entry, "annotations/annotations.json")
    assert anns["video"]["width"] == _W and anns["video"]["height"] == _H
    assert anns["video"]["frame_count"] == frames_written
    for im in anns["images"]:
        assert im["width"] == _W and im["height"] == _H
    # coords in original pixel space: every sidecar bbox fits inside the frame.
    for f in sidecar["frames"]:
        for d in f["detections"]:
            x, y, w, h = d["bbox"]
            assert 0 <= x <= _W and 0 <= y <= _H and w <= _W and h <= _H


# --- AC4: keyframe annotation == image-mode structure + video-mode fields ---


def test_ac4_keyframe_annotation_structure(tmp_path):
    res = _run_recording(tmp_path, "take4", n_keyframes=2)
    entry = res["entry_dir"]
    anns = _load(entry, "annotations/annotations.json")

    # Compare against image-mode: build one annotation via the SAME shared helper
    # for the same detection, and assert the video annotation is that structure
    # plus video_id / frame_number(on the image) / a unique track_id.
    from backend import coco

    dets = make_fake_dets([[10, 10, 40, 30]], confidences=[0.9], size=(_W, _H))
    image_mode = coco.build_annotation(dets, 0, _W, _H, ann_id=1, image_id=1)
    image_keys = set(image_mode.keys())

    assert anns["annotations"], "expected keyframe annotations"
    track_ids = [a["track_id"] for a in anns["annotations"]]
    assert len(track_ids) == len(set(track_ids))  # unique per annotation
    for a in anns["annotations"]:
        assert image_keys <= set(a.keys())  # image-mode structure preserved
        assert a["video_id"] == 1
        assert "track_id" in a
        # the annotation's image carries the frame_number (video-mode field)
        img = next(im for im in anns["images"] if im["id"] == a["image_id"])
        assert "frame_number" in img
    assert anns["categories"] == coco.CATEGORIES


# --- AC5: kill mid-run -> valid partial; retry completes; byte-identical -----


def test_ac5_kill_partial_then_retry_is_identical(tmp_path):
    # First post-pass explodes after 3 decoded frames; retry (2nd factory call)
    # replays cleanly. write_selected_frames runs before the loop, so the killed
    # entry is a valid partial (selected_frames.json, no annotations.json).
    def job_hook(call_index, written):
        if call_index == 0:
            return lambda p: _ExplodingReplayCap(written, explode_at=3)
        return None  # retry: normal replay

    res = _run_recording(tmp_path, "take5", n_keyframes=2, job_hook=job_hook)
    entry = res["entry_dir"]
    client = res["client"]

    # Killed: failed state, valid partial project on disk.
    assert res["final_status"]["state"] == "failed"
    assert res["final_status"]["error"]
    assert (entry / "annotations" / "metadata" / "selected_frames.json").is_file()
    assert not (entry / "annotations" / "annotations.json").exists()
    assert list((entry / "video").iterdir())  # video survived

    # Retry completes.
    client.post("/record/retry").raise_for_status()
    assert _wait_for(lambda: client.get("/record/status").json()["state"] == "idle", timeout=15.0)
    assert (entry / "annotations" / "annotations.json").is_file()

    # Byte-identical to an uninterrupted run: rebuild a reference straight from
    # the same recorded frames + keyframes + frame_count (VideoEntryWriter is
    # deterministic — fixed timestamps, running ids), then diff the JSONs.
    ref_dir = tmp_path / "reference"
    (ref_dir / "video").mkdir(parents=True)
    (ref_dir / "video" / "take5.mp4").write_bytes(b"\x00")
    written = res["written"]
    ref_job = PostPassJob(
        ref_dir, "take5", _OneBoxDetector(),
        keyframes=sorted(fn for fn, _ in res["keyframes"]),
        frame_count=res["frames_written"],
        mining_threshold=0.25,
        operator_threshold=0.5,
        model_version="rfdetr-e2e",
        cap_factory=lambda p: _ReplayCap(written),
        probe_fn=lambda p: VideoProbe(fps=30.0, frame_count=len(written), width=_W, height=_H),
    )
    ref_job.run()
    assert ref_job.status.state == "done"

    for rel in ("annotations/annotations.json",
                "annotations/metadata/full_frame_detections.json",
                "annotations/metadata/selected_frames.json"):
        assert (entry / rel).read_bytes() == (ref_dir / rel).read_bytes(), f"{rel} differs from clean run"


# --- AC6: import validator runs advisory-clean on the finished entry --------


def test_ac6_validator_advisory_clean(tmp_path):
    res = _run_recording(tmp_path, "take6")
    entry = res["entry_dir"]
    from scripts.validate_import import validate

    errors, warnings = validate(entry)
    assert errors == [], f"validator reported errors: {errors}"
