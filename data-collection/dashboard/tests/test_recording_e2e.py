"""Full recording pipeline end-to-end, keyframe-only (ADR-0002), with fakes.

The 2026-07-09 simplification dropped the all-frames post-pass and the drain
queue. A recording is now: record the MP4 live, mark a handful of keyframes with
SPACE (each capturing the live detection on the frame the operator saw), then
``/record/stop`` writes the reviewed video project **synchronously** from those
stored keyframes and returns to idle.

Drives the WHOLE stack — real ``CaptureLoop`` (reader/inference/slot threads),
real ``create_app`` + ``/record/*`` + ``/keyframe`` endpoints, real
``VideoEntryWriter`` + ``coco`` helper — over a **fake numbered camera**. Only
two seams are faked, and neither needs hardware:

- **camera**: ``_PacedNumberedCap`` yields ``make_numbered_frames`` — each
  frame's pixels encode its own 0-based content index, so a dropped / duplicated
  / reordered / off-by-one frame is detectable after the fact. It sleeps
  ~1/``fps`` per ``read`` so the un-rate-limited reader is bounded to a
  camera-like rate, and repeats the last frame when drained so idle never
  starves ``/record/start``.
- **encoder**: ``_FileEncoder`` records every written frame in order (the
  ground-truth "MP4 content") and, on ``release()``, drops a stub file into
  ``video/`` so the finished entry satisfies the video-project discovery rule
  (``annotations.json`` + non-empty ``video/``) without a real H.264 file.
- **MP4 probe**: faked to return the true dims (no real file to probe), so the
  writer's ``video`` block and image records carry the stream dims.

The load-bearing identity fact: the reader stamps ``frame_number = its write
index`` and the inference thread publishes *that* frame + number, so the frame
displayed as ``frame_number == D`` **is** the frame at MP4 position ``D``
(``encoder.written[D]``). ``/keyframe`` echoes the on-screen *generation*; the
endpoint resolves it to that frame via the ring and pre-encodes it to the
keyframe JPEG at press time (the live pre-encode frame — NOT an MP4 decode, per
the settled AC2 relaxation).
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
from fastapi.testclient import TestClient
from tests.fakes import make_fake_dets
from tests.recording_fakes import decode_frame_index, make_numbered_frames

# Tiny frames keep the test fast; dims are asserted end-to-end anyway (a real
# 1080p run keeps the dims-equal invariant trivially true).
_W, _H = 64, 48
_CAM_FPS = 120.0  # paced read rate: bounds frames_written to a camera-like count


# --- fakes ------------------------------------------------------------------


class _PacedNumberedCap:
    """``cv2.VideoCapture`` stand-in yielding numbered frames at ~``fps``.

    Sleeps ``1/fps`` per ``read`` so the (un-rate-limited) capture reader is
    bounded to a camera-like rate instead of spinning. Repeats the last frame
    once the queue drains so idle inference never starves the initial
    ``/record/start`` snapshot. Thread-safe."""

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
    """Detector stub returning a fixed single-instrument detection (confidence
    0.9) for every frame. Frame-agnostic and thread-safe."""

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
    stub file to the MP4 path so ``video/`` is non-empty (discovery rule)."""

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
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(b"\x00")


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


def _build_app(capture, detector, output_path, encoder_holder):
    """create_app wired for the e2e: real endpoints + real VideoEntryWriter,
    fake encoder (captured into ``encoder_holder['enc']``), fake probe."""
    app = create_app(detector, _writer_factory(), capture)
    app.state.output_path = str(output_path)
    app.state.capture_fps = 30.0

    def open_encoder(path, fps, frame_size, **kwargs):
        enc = _FileEncoder(path)
        encoder_holder["enc"] = enc
        encoder_holder["open_args"] = {"fps": fps, "frame_size": tuple(frame_size)}
        return enc

    app.state.open_encoder = open_encoder
    app.state.probe_video = lambda p, **kw: VideoProbe(
        fps=30.0, frame_count=0, width=_W, height=_H
    )
    return app


def _start_capture(frames, detector):
    capture = CaptureLoop(
        detector,
        camera_index=0,
        render_fn=lambda frame, dets: frame,
        cap_factory=lambda idx: _PacedNumberedCap(frames),
        stale_after_s=5.0,
        ring_size=20000,  # never evict a generation before /keyframe re-reads it
    )
    capture.start()
    return capture


def _mark_keyframe(client, capture):
    """Read the on-screen frame's generation, echo it to /keyframe (exactly what
    the browser does), and return (frame_number, decoded_displayed_index,
    displayed_frame) — the displayed frame's identity, captured atomically via
    the ring."""
    r = client.get("/frame")
    assert r.status_code == 200
    gen = int(r.headers["x-frame-generation"])
    displayed = capture.snapshot_at(gen)
    assert displayed is not None and displayed.frame_number is not None
    fn = displayed.frame_number
    disp_index = decode_frame_index(displayed.frame)
    disp_frame = displayed.frame.copy()
    resp = client.post("/keyframe", json={"generation": gen})
    resp.raise_for_status()
    assert resp.json()["generation"] == gen
    return fn, disp_index, disp_frame


def _run_recording(tmp_path, entry_base, *, n_keyframes=2):
    """Full record → keyframe(s) → stop drive. Returns everything the assertions
    need; leaves the machine idle (synchronous stop)."""
    frames = make_numbered_frames(400, _W, _H)
    detector = _OneBoxDetector()
    capture = _start_capture(frames, detector)
    holder: dict = {}
    app = _build_app(capture, detector, tmp_path, holder)
    client = TestClient(app)
    kf: list = []
    try:
        assert _wait_for(lambda: capture.snapshot() is not None)
        start_resp = client.post("/record/start", json={"entry_base": entry_base})
        start_resp.raise_for_status()
        entry_name = start_resp.json()["entry_name"]  # resolved <base>_NNN (U1)
        # Wait for a recording-stamped snapshot (frame_number not None).
        assert _wait_for(
            lambda: capture.snapshot() is not None
            and capture.snapshot().frame_number is not None
        )
        seen = set()
        deadline = time.monotonic() + 5.0
        while len(kf) < n_keyframes and time.monotonic() < deadline:
            fn, disp_index, disp_frame = _mark_keyframe(client, capture)
            if fn not in seen:
                seen.add(fn)
                kf.append((fn, disp_index, disp_frame))
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
        assert stop.json()["ok"] is True
        # Synchronous stop: back to idle, entry fully on disk (AC1: no drain).
        assert client.get("/record/status").json()["state"] == "idle"
        return {
            "client": client,
            "capture": capture,
            "app": app,
            "entry_dir": Path(tmp_path) / "videos" / entry_name,
            "entry_name": entry_name,
            "encoder": holder["enc"],
            "written": holder["enc"].written,
            "keyframes": kf,  # list of (frame_number, disp_index, disp_frame)
            "frames_written": frames_written,
        }
    finally:
        capture.stop()


def _load(entry_dir: Path, rel: str):
    return json.loads((entry_dir / rel).read_text())


# --- AC1: a completed run is a video project on disk ------------------------


def test_ac1_completed_run_is_a_video_project(tmp_path):
    res = _run_recording(tmp_path, "take1")
    entry = res["entry_dir"]
    # Discovery rule: annotations.json present + non-empty video/.
    assert (entry / "annotations" / "annotations.json").is_file()
    video_files = list((entry / "video").iterdir())
    assert video_files and all(p.stat().st_size > 0 for p in video_files)


# --- AC2 (relaxed) + AC3: keyframe fidelity + displayed-frame identity -------


def test_ac2_keyframe_fidelity_and_displayed_frame_identity(tmp_path):
    res = _run_recording(tmp_path, "take2", n_keyframes=3)
    entry = res["entry_dir"]
    resolved = res["entry_name"]  # resolved <base>_NNN (U1) — names artifacts
    written = res["written"]
    keyframes = res["keyframes"]
    frames_written = res["frames_written"]

    anns = _load(entry, "annotations/annotations.json")
    selected = _load(entry, "annotations/metadata/selected_frames.json")

    marked = sorted(fn for fn, _, _ in keyframes)
    assert selected["selected_frames"] == marked
    # exactly one image record per SPACE press, one JPEG each
    assert sorted(im["frame_number"] for im in anns["images"]) == marked
    jpg_dir = entry / "images"
    assert sorted(p.name for p in jpg_dir.glob("*.jpg")) == [
        f"{resolved}_f{fn:06d}.jpg" for fn in marked
    ]

    _JPEG_TOL = 4
    for fn, disp_index, disp_frame in keyframes:
        # (a) writer index == MP4 index (no off-by-one): the frame written at MP4
        #     position fn decodes to the same content that was on screen when
        #     SPACE was pressed. Lossless -> exact.
        assert decode_frame_index(written[fn]) == disp_index
        # (b) AC2 relaxed: the keyframe JPEG is the LIVE pre-encode frame captured
        #     at press time (not an MP4 decode) — its pixels equal the displayed
        #     frame within JPEG re-encode tolerance.
        jpg = cv2.imread(str(jpg_dir / f"{resolved}_f{fn:06d}.jpg"))
        assert jpg is not None
        assert jpg.shape == disp_frame.shape
        assert int(np.abs(jpg.astype(int) - disp_frame.astype(int)).max()) <= _JPEG_TOL
        # (c) the displayed frame lagged the newest write (a real earlier frame,
        #     not newest-at-press-time).
        assert fn < frames_written


# --- AC5: keyframe annotation == image-mode structure + video-mode fields ---


def test_ac5_keyframe_annotation_structure(tmp_path):
    res = _run_recording(tmp_path, "take5", n_keyframes=2)
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
        img = next(im for im in anns["images"] if im["id"] == a["image_id"])
        assert "frame_number" in img
    assert anns["categories"] == coco.CATEGORIES


# --- AC6: import validator runs advisory-clean on the finished entry --------


def test_ac6_validator_advisory_clean(tmp_path):
    res = _run_recording(tmp_path, "take6")
    entry = res["entry_dir"]
    from scripts.validate_import import validate

    errors, warnings = validate(entry)
    assert errors == [], f"validator reported errors: {errors}"
