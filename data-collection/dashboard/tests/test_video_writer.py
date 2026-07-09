"""Behavior contract for backend.video_writer.VideoEntryWriter (keyframe-only).

Since the 2026-07-09 simplification (ADR-0002) the writer produces the three
reviewed artifacts only — annotations.json (COCO-VID, keyframes), one JPEG per
keyframe, selected_frames.json — from keyframes captured live at SPACE-press
time. No all-frames post-pass, no full_frame_detections.json sidecar.

``add_keyframe(frame_number, jpeg_bytes, dets, threshold)`` is fed pre-encoded
JPEG bytes (the frame the app imencoded at press time) plus that keyframe's own
snapshot threshold — no detector, no camera, no MP4 on disk. The surviving
contracts guarded here: folder layout, the §3 filename contract, COCO-VID
image/annotation shape (video_id / frame_number / unique track_id), the
per-keyframe threshold gate, selected_frames.json, the partial-project state,
and idempotent rebuild.

AC1-AC3 (the shared coco.build_annotation helper) live in ``tests/test_coco.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.video_writer import VideoEntryWriter
from tests.fakes import make_fake_dets, make_fake_frame

ENTRY_NAME = "entry1"
W, H = 640, 480
FPS = 24.0


def _video_dict(frame_count: int) -> dict:
    return {
        "file_name": f"{ENTRY_NAME}.mp4",
        "width": W,
        "height": H,
        "fps": FPS,
        "frame_count": frame_count,
    }


def _make_writer(root, keyframes, frame_count: int):
    entry_dir = Path(root) / ENTRY_NAME
    video = _video_dict(frame_count)
    writer = VideoEntryWriter(entry_dir, ENTRY_NAME, video=video, keyframes=keyframes)
    return writer, video


def _jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def _add_keyframe(writer, fn, frame, dets, threshold=0.5) -> None:
    writer.add_keyframe(fn, _jpeg(frame), dets, threshold)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Folder layout: annotations.json + selected_frames.json, and exactly one JPEG
# per keyframe named per the §3 filename contract. No sidecar.
# ---------------------------------------------------------------------------


def test_folder_layout_after_finalize(tmp_path):
    keyframes = [0, 2]
    writer, video = _make_writer(tmp_path, keyframes, frame_count=3)

    writer.write_selected_frames()
    for fn in keyframes:
        _add_keyframe(writer, fn, make_fake_frame(W, H),
                      make_fake_dets([[10, 10, 60, 60]], size=(W, H)))
    writer.finalize()

    entry_dir = writer.entry_dir
    assert (entry_dir / "annotations" / "annotations.json").is_file()
    assert (entry_dir / "annotations" / "metadata" / "selected_frames.json").is_file()
    # The sidecar is gone (ADR-0002).
    assert not (entry_dir / "annotations" / "metadata" / "full_frame_detections.json").exists()

    for fn in keyframes:
        assert (entry_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").is_file()

    jpg_names = {p.name for p in (entry_dir / "images").glob("*.jpg")}
    assert jpg_names == {f"{ENTRY_NAME}_f{fn:06d}.jpg" for fn in keyframes}


# ---------------------------------------------------------------------------
# images[] is exactly the (deduped+sorted) keyframe set, one record each, with
# video_id/frame_number/dims/review_status; top-level `video` block is the
# passed dict plus id:1 + source_type:"video".
# ---------------------------------------------------------------------------


def test_images_are_exactly_keyframes_with_fields(tmp_path):
    keyframes_input = [212, 104, 104, 0]  # unsorted + duplicate
    expected_keyframes = [0, 104, 212]
    writer, video = _make_writer(tmp_path, keyframes_input, frame_count=300)

    writer.write_selected_frames()
    for fn in expected_keyframes:
        _add_keyframe(writer, fn, make_fake_frame(W, H),
                      make_fake_dets([[10, 10, 60, 60]], size=(W, H)))
    writer.finalize()

    data = _read_json(writer.entry_dir / "annotations" / "annotations.json")

    assert [img["frame_number"] for img in data["images"]] == expected_keyframes
    assert len(data["images"]) == len(expected_keyframes)
    for img in data["images"]:
        assert img["video_id"] == 1
        assert img["width"] == W
        assert img["height"] == H
        assert img["review_status"] == "pending"
        # The §3 filename contract must hold in the RECORD itself, not just on
        # disk — a wrong in-record name silently orphans the JPEG.
        assert img["file_name"] == f"{ENTRY_NAME}_f{img['frame_number']:06d}.jpg"

    assert data["video"] == {**video, "id": 1, "source_type": "video"}


# ---------------------------------------------------------------------------
# Every annotation across all keyframes has a distinct track_id (running
# counter) and video_id==1.
# ---------------------------------------------------------------------------


def test_every_annotation_has_unique_track_id_and_video_id(tmp_path):
    keyframes = [0, 1, 2]
    writer, video = _make_writer(tmp_path, keyframes, frame_count=3)

    writer.write_selected_frames()
    for fn in keyframes:
        _add_keyframe(writer, fn, make_fake_frame(W, H),
                      make_fake_dets([[10, 10, 60, 60], [100, 100, 160, 160]], size=(W, H)))
    writer.finalize()

    data = _read_json(writer.entry_dir / "annotations" / "annotations.json")
    anns = data["annotations"]

    assert len(anns) == 6  # 2 detections x 3 keyframes
    track_ids = [a["track_id"] for a in anns]
    assert len(set(track_ids)) == len(track_ids)  # all distinct
    assert all(a["video_id"] == 1 for a in anns)


# ---------------------------------------------------------------------------
# Per-keyframe threshold gate: only dets with confidence >= the keyframe's own
# snapshot threshold become annotations; an all-below-threshold keyframe is a
# first-class empty image record (zero annotations), not an error. Two keyframes
# with DIFFERENT thresholds prove it is per-keyframe, not take-wide.
# ---------------------------------------------------------------------------


def test_per_keyframe_threshold_filters_annotations(tmp_path):
    keyframes = [0, 1]
    writer, video = _make_writer(tmp_path, keyframes, frame_count=2)
    writer.write_selected_frames()

    frame = make_fake_frame(W, H)
    # frame 0 at threshold 0.5: one det above, one below -> only the above kept.
    dets0 = make_fake_dets(
        [[10, 10, 60, 60], [100, 100, 160, 160]], confidences=[0.9, 0.1], size=(W, H)
    )
    _add_keyframe(writer, 0, frame, dets0, threshold=0.5)

    # frame 1 at threshold 0.5: every det below -> zero annotations, record stays.
    dets1 = make_fake_dets([[20, 20, 80, 80]], confidences=[0.2], size=(W, H))
    _add_keyframe(writer, 1, frame, dets1, threshold=0.5)

    writer.finalize()

    data = _read_json(writer.entry_dir / "annotations" / "annotations.json")
    assert len(data["images"]) == 2  # both keyframes present, even the empty one

    image_id_frame0 = next(img["id"] for img in data["images"] if img["frame_number"] == 0)
    image_id_frame1 = next(img["id"] for img in data["images"] if img["frame_number"] == 1)

    anns_by_image: dict[int, list] = {}
    for a in data["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    assert len(anns_by_image.get(image_id_frame0, [])) == 1
    assert anns_by_image.get(image_id_frame1, []) == []


def test_threshold_is_per_keyframe_not_take_wide(tmp_path):
    # The SAME detection (conf 0.3) is a keyframe annotation under a low
    # threshold and dropped under a high one — proving each keyframe filters at
    # its own captured snapshot threshold.
    keyframes = [0, 1]
    writer, video = _make_writer(tmp_path, keyframes, frame_count=2)
    writer.write_selected_frames()

    dets = make_fake_dets([[10, 10, 60, 60]], confidences=[0.3], size=(W, H))
    _add_keyframe(writer, 0, make_fake_frame(W, H), dets, threshold=0.2)  # kept
    _add_keyframe(writer, 1, make_fake_frame(W, H), dets, threshold=0.5)  # dropped
    writer.finalize()

    data = _read_json(writer.entry_dir / "annotations" / "annotations.json")
    id0 = next(img["id"] for img in data["images"] if img["frame_number"] == 0)
    id1 = next(img["id"] for img in data["images"] if img["frame_number"] == 1)
    per_image = {id0: 0, id1: 0}
    for a in data["annotations"]:
        per_image[a["image_id"]] += 1
    assert per_image[id0] == 1
    assert per_image[id1] == 0


# ---------------------------------------------------------------------------
# The keyframe JPEG on disk is the exact bytes handed to add_keyframe (the app
# pre-encodes at press time; the writer writes them verbatim, no re-encode).
# ---------------------------------------------------------------------------


def test_keyframe_jpeg_is_the_bytes_passed_verbatim(tmp_path):
    writer, video = _make_writer(tmp_path, [0], frame_count=1)

    frame = np.full((H, W, 3), (30, 180, 90), dtype=np.uint8)
    jpeg = _jpeg(frame)
    writer.add_keyframe(0, jpeg, make_fake_dets([[10, 10, 60, 60]], size=(W, H)), 0.5)

    jpg_path = writer.entry_dir / "images" / f"{ENTRY_NAME}_f000000.jpg"
    assert jpg_path.read_bytes() == jpeg  # written verbatim
    # And it still decodes to the frame dims within JPEG tolerance.
    saved = cv2.imread(str(jpg_path))
    assert saved is not None and saved.shape == frame.shape
    assert np.abs(saved.astype(int) - frame.astype(int)).max() <= 2


# ---------------------------------------------------------------------------
# selected_frames.json: manual_review true, selected_frames == sorted deduped
# keyframes, fps from `video`, selected_frames_with_time == frame/fps, and it
# matches the frame_numbers present in annotations.json images[].
# ---------------------------------------------------------------------------


def test_selected_frames_json_fields(tmp_path):
    keyframes_input = [212, 104, 104, 0]
    expected = [0, 104, 212]
    writer, video = _make_writer(tmp_path, keyframes_input, frame_count=300)

    writer.write_selected_frames()
    for fn in expected:
        _add_keyframe(writer, fn, make_fake_frame(W, H),
                      make_fake_dets([[10, 10, 60, 60]], size=(W, H)))
    writer.finalize()

    data = _read_json(writer.entry_dir / "annotations" / "metadata" / "selected_frames.json")

    assert data["manual_review"] is True
    assert data["selected_frames"] == expected
    assert data["fps"] == pytest.approx(video["fps"])

    seconds_by_frame = {d["frame"]: d["seconds"] for d in data["selected_frames_with_time"]}
    assert set(seconds_by_frame.keys()) == set(expected)
    for fn in expected:
        assert seconds_by_frame[fn] == pytest.approx(fn / video["fps"])

    ann_data = _read_json(writer.entry_dir / "annotations" / "annotations.json")
    ann_frame_numbers = sorted({img["frame_number"] for img in ann_data["images"]})
    assert data["selected_frames"] == ann_frame_numbers


# ---------------------------------------------------------------------------
# After write_selected_frames() only (no keyframes, no finalize) the entry has
# selected_frames.json but NO annotations.json — the "needs detection" partial
# state a mid-write failure at /record/stop leaves behind.
# ---------------------------------------------------------------------------


def test_write_selected_frames_only_leaves_partial_project(tmp_path):
    keyframes = [0, 2]
    writer, video = _make_writer(tmp_path, keyframes, frame_count=5)

    writer.write_selected_frames()

    assert (writer.entry_dir / "annotations" / "metadata" / "selected_frames.json").is_file()
    assert not (writer.entry_dir / "annotations" / "annotations.json").exists()


# ---------------------------------------------------------------------------
# Idempotent rebuild: a second full pass (fresh writer, same entry_dir, same
# inputs) reproduces byte-identical annotations.json / JPEGs — never appended.
# ---------------------------------------------------------------------------


def test_two_full_passes_are_byte_identical(tmp_path):
    keyframes = [0, 3]
    frame = np.full((H, W, 3), (10, 20, 30), dtype=np.uint8)
    jpeg = _jpeg(frame)
    dets = make_fake_dets([[10, 10, 60, 60]], confidences=[0.9], size=(W, H))

    def _run():
        writer, video = _make_writer(tmp_path, keyframes, frame_count=4)
        writer.write_selected_frames()
        for fn in keyframes:
            writer.add_keyframe(fn, jpeg, dets, 0.5)
        writer.finalize()
        return writer

    writer1 = _run()
    ann1 = (writer1.entry_dir / "annotations" / "annotations.json").read_bytes()
    jpgs1 = {
        fn: (writer1.entry_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").read_bytes()
        for fn in keyframes
    }

    writer2 = _run()  # fresh instance, same entry_dir, identical inputs
    ann2 = (writer2.entry_dir / "annotations" / "annotations.json").read_bytes()
    jpgs2 = {
        fn: (writer2.entry_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").read_bytes()
        for fn in keyframes
    }

    assert ann1 == ann2
    assert jpgs1 == jpgs2
