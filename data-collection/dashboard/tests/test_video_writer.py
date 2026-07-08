"""TR3 AC4-AC13: behavior contract for backend.video_writer.VideoEntryWriter.

Blind-TDD test-writer pass — written from TR3-video-writer.md against the frozen
``VideoEntryWriter`` interface (``__init__`` + ``write_selected_frames`` /
``add_frame`` / ``finalize`` / ``.entry_dir``), plus the on-disk contract in
IMPORT_FORMAT_VIDEO.md §§1-5 and RECORDING.md §Post-pass. No real video decode —
``add_frame`` is fed synthetic frames + ``make_fake_dets`` directly (no detector,
no camera, no MP4 on disk: VideoEntryWriter never opens the video itself).

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


def _make_writer(
    root,
    keyframes,
    frame_count: int,
    *,
    operator_threshold: float = 0.5,
    mining_threshold: float = 0.25,
    model_version: str = "model-v1",
    model_name: str = "nf-detector",
):
    entry_dir = Path(root) / ENTRY_NAME
    video = _video_dict(frame_count)
    writer = VideoEntryWriter(
        entry_dir,
        ENTRY_NAME,
        model_version,
        video=video,
        keyframes=keyframes,
        operator_threshold=operator_threshold,
        mining_threshold=mining_threshold,
        model_name=model_name,
    )
    return writer, video


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# AC4 — folder layout after finalize: annotations.json, metadata sidecars, and
# exactly one JPEG per keyframe named per the §3 filename contract.
# ---------------------------------------------------------------------------


def test_ac04_folder_layout_after_finalize(tmp_path):
    frame_count = 3
    keyframes = [0, 2]
    writer, video = _make_writer(tmp_path, keyframes, frame_count)

    for fn in range(frame_count):
        frame = make_fake_frame(W, H)
        dets = (
            make_fake_dets([[10, 10, 60, 60]], size=(W, H))
            if fn in keyframes
            else make_fake_dets([], size=(W, H))
        )
        writer.add_frame(fn, frame, dets)
    writer.finalize()

    entry_dir = writer.entry_dir
    assert (entry_dir / "annotations" / "annotations.json").is_file()
    assert (entry_dir / "annotations" / "metadata" / "selected_frames.json").is_file()
    assert (entry_dir / "annotations" / "metadata" / "full_frame_detections.json").is_file()

    for fn in keyframes:
        assert (entry_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").is_file()

    jpg_names = {p.name for p in (entry_dir / "images").glob("*.jpg")}
    assert jpg_names == {f"{ENTRY_NAME}_f{fn:06d}.jpg" for fn in keyframes}


# ---------------------------------------------------------------------------
# AC5 — images[] is exactly the (deduped+sorted) keyframe set, one record each,
# with video_id/frame_number/dims/review_status; top-level `video` block is the
# passed dict plus id:1 + source_type:"video".
# ---------------------------------------------------------------------------


def test_ac05_images_are_exactly_keyframes_with_fields(tmp_path):
    frame_count = 300
    keyframes_input = [212, 104, 104, 0]  # unsorted + duplicate, per the spec
    expected_keyframes = [0, 104, 212]
    writer, video = _make_writer(tmp_path, keyframes_input, frame_count)

    for fn in range(frame_count):
        frame = make_fake_frame(W, H)
        dets = (
            make_fake_dets([[10, 10, 60, 60]], size=(W, H))
            if fn in expected_keyframes
            else make_fake_dets([], size=(W, H))
        )
        writer.add_frame(fn, frame, dets)
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
# AC6 — every annotation across all keyframes has a distinct track_id (running
# counter) and video_id==1.
# ---------------------------------------------------------------------------


def test_ac06_every_annotation_has_unique_track_id_and_video_id(tmp_path):
    frame_count = 3
    keyframes = [0, 1, 2]
    writer, video = _make_writer(tmp_path, keyframes, frame_count)

    for fn in range(frame_count):
        frame = make_fake_frame(W, H)
        dets = make_fake_dets([[10, 10, 60, 60], [100, 100, 160, 160]], size=(W, H))
        writer.add_frame(fn, frame, dets)
    writer.finalize()

    data = _read_json(writer.entry_dir / "annotations" / "annotations.json")
    anns = data["annotations"]

    assert len(anns) == 6  # 2 detections x 3 keyframes
    track_ids = [a["track_id"] for a in anns]
    assert len(set(track_ids)) == len(track_ids)  # all distinct
    assert all(a["video_id"] == 1 for a in anns)


# ---------------------------------------------------------------------------
# AC7 — operator threshold gate: only dets with confidence >= operator_threshold
# become keyframe annotations; an all-below-threshold keyframe is a first-class
# empty image record (zero annotations), not an error.
# ---------------------------------------------------------------------------


def test_ac07_operator_threshold_filters_keyframe_annotations(tmp_path):
    frame_count = 2
    keyframes = [0, 1]
    writer, video = _make_writer(tmp_path, keyframes, frame_count, operator_threshold=0.5)

    frame = make_fake_frame(W, H)
    # frame 0: one det above threshold, one below -> only the above one is kept.
    dets0 = make_fake_dets(
        [[10, 10, 60, 60], [100, 100, 160, 160]],
        confidences=[0.9, 0.1],
        size=(W, H),
    )
    writer.add_frame(0, frame, dets0)

    # frame 1: every det below threshold -> zero annotations, image record stays.
    dets1 = make_fake_dets([[20, 20, 80, 80]], confidences=[0.2], size=(W, H))
    writer.add_frame(1, frame, dets1)

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


# ---------------------------------------------------------------------------
# AC8 — full_frame_detections.json: one entry per processed frame (incl.
# empty), box-only {bbox,score,label}, schema_version==1, model/video blocks.
# ---------------------------------------------------------------------------


def test_ac08_sidecar_completeness(tmp_path):
    frame_count = 4
    keyframes = [1]
    model_version = "model-vX"
    mining_threshold = 0.2
    writer, video = _make_writer(
        tmp_path,
        keyframes,
        frame_count,
        mining_threshold=mining_threshold,
        model_version=model_version,
    )

    for fn in range(frame_count):
        frame = make_fake_frame(W, H)
        if fn == 1:
            dets = make_fake_dets([[10, 10, 60, 60]], confidences=[0.9], size=(W, H))
        elif fn == 2:
            dets = make_fake_dets([[20, 20, 80, 80]], confidences=[0.3], size=(W, H))
        else:
            dets = make_fake_dets([], size=(W, H))
        writer.add_frame(fn, frame, dets)
    writer.finalize()

    data = _read_json(writer.entry_dir / "annotations" / "metadata" / "full_frame_detections.json")

    assert data["schema_version"] == 1
    assert data["video"] == {
        "file_name": video["file_name"],
        "width": video["width"],
        "height": video["height"],
        "fps": video["fps"],
        "frame_count": video["frame_count"],
    }
    assert data["model"] == {
        "name": "nf-detector",
        "version": model_version,
        "conf_threshold": mining_threshold,
    }

    frames = data["frames"]
    assert [f["frame_number"] for f in frames] == list(range(frame_count))

    frame0 = frames[0]
    assert frame0["detections"] == []

    frame1 = frames[1]
    assert len(frame1["detections"]) == 1
    det = frame1["detections"][0]
    assert set(det.keys()) == {"bbox", "score", "label"}
    assert det["bbox"] == pytest.approx([10.0, 10.0, 50.0, 50.0])
    assert det["score"] == pytest.approx(0.9, abs=1e-6)
    assert det["label"] == "surgical_instrument"

    # frame 2 is a NON-keyframe with a real detection (conf 0.3, above the 0.2
    # mining threshold) — the sidecar must carry EVERY frame's real detections,
    # not just keyframes, or frame-mining loses its candidate pool.
    frame2 = frames[2]
    assert len(frame2["detections"]) == 1
    det2 = frame2["detections"][0]
    assert set(det2.keys()) == {"bbox", "score", "label"}  # xywh abs px, no segmentation
    assert det2["bbox"] == pytest.approx([20.0, 20.0, 60.0, 60.0])
    assert det2["score"] == pytest.approx(0.3, abs=1e-6)
    assert det2["label"] == "surgical_instrument"


# ---------------------------------------------------------------------------
# AC9 — a keyframe's detections appear in BOTH annotations.json (operator
# threshold) and the sidecar (mining threshold) — not deduplicated away.
# ---------------------------------------------------------------------------


def test_ac09_keyframe_detections_appear_in_both_files(tmp_path):
    frame_count = 3
    keyframes = [1]
    writer, video = _make_writer(tmp_path, keyframes, frame_count)

    for fn in range(frame_count):
        frame = make_fake_frame(W, H)
        dets = (
            make_fake_dets([[10, 10, 60, 60]], confidences=[0.9], size=(W, H))
            if fn == 1
            else make_fake_dets([], size=(W, H))
        )
        writer.add_frame(fn, frame, dets)
    writer.finalize()

    ann_data = _read_json(writer.entry_dir / "annotations" / "annotations.json")
    sidecar_data = _read_json(
        writer.entry_dir / "annotations" / "metadata" / "full_frame_detections.json"
    )

    assert 1 in [img["frame_number"] for img in ann_data["images"]]
    assert 1 in [f["frame_number"] for f in sidecar_data["frames"]]

    sidecar_frame1_dets = next(f["detections"] for f in sidecar_data["frames"] if f["frame_number"] == 1)
    assert len(sidecar_frame1_dets) == 1

    image_id_frame1 = next(img["id"] for img in ann_data["images"] if img["frame_number"] == 1)
    ann_frame1 = [a for a in ann_data["annotations"] if a["image_id"] == image_id_frame1]
    assert len(ann_frame1) == 1


# ---------------------------------------------------------------------------
# AC10 — the keyframe JPEG decodes to the exact frame array handed to
# add_frame (shape + pixels), at video dims, no resize. Written by add_frame
# itself (finalize's docstring: JPEGs are "already written by add_frame").
# ---------------------------------------------------------------------------


def test_ac10_keyframe_jpeg_matches_frame_pixels_exactly(tmp_path):
    frame_count = 1
    keyframes = [0]
    writer, video = _make_writer(tmp_path, keyframes, frame_count)

    # Flat, constant-color frame so lossy JPEG round-trips near-exactly.
    frame = np.full((H, W, 3), (30, 180, 90), dtype=np.uint8)
    dets = make_fake_dets([[10, 10, 60, 60]], size=(W, H))
    writer.add_frame(0, frame, dets)  # no finalize() — JPEG is add_frame's job

    jpg_path = writer.entry_dir / "images" / f"{ENTRY_NAME}_f000000.jpg"
    saved = cv2.imread(str(jpg_path))
    assert saved is not None
    assert saved.shape == frame.shape

    diff = np.abs(saved.astype(int) - frame.astype(int))
    assert diff.max() <= 2


# ---------------------------------------------------------------------------
# AC11 — selected_frames.json: manual_review true, selected_frames == sorted
# deduped keyframes, fps from `video`, selected_frames_with_time == frame/fps.
# ---------------------------------------------------------------------------


def test_ac11_selected_frames_json_fields(tmp_path):
    frame_count = 300
    keyframes_input = [212, 104, 104, 0]
    expected = [0, 104, 212]
    writer, video = _make_writer(tmp_path, keyframes_input, frame_count)

    # Full pass so annotations.json exists too — AC11 requires selected_frames to
    # equal the frame_numbers actually present in annotations.json images[].
    for fn in range(frame_count):
        frame = make_fake_frame(W, H)
        dets = (
            make_fake_dets([[10, 10, 60, 60]], size=(W, H))
            if fn in expected
            else make_fake_dets([], size=(W, H))
        )
        writer.add_frame(fn, frame, dets)
    writer.finalize()

    data = _read_json(writer.entry_dir / "annotations" / "metadata" / "selected_frames.json")

    assert data["manual_review"] is True
    assert data["selected_frames"] == expected
    assert data["fps"] == pytest.approx(video["fps"])

    seconds_by_frame = {d["frame"]: d["seconds"] for d in data["selected_frames_with_time"]}
    assert set(seconds_by_frame.keys()) == set(expected)
    for fn in expected:
        assert seconds_by_frame[fn] == pytest.approx(fn / video["fps"])

    # Cross-check directly against annotations.json (not just the constant): the
    # import contract requires selected_frames == the set of frame_numbers there.
    ann_data = _read_json(writer.entry_dir / "annotations" / "annotations.json")
    ann_frame_numbers = sorted({img["frame_number"] for img in ann_data["images"]})
    assert data["selected_frames"] == ann_frame_numbers


# ---------------------------------------------------------------------------
# AC12 — after write_selected_frames() only (no finalize), the entry has
# selected_frames.json but NO annotations.json — the "needs detection" state.
# ---------------------------------------------------------------------------


def test_ac12_write_selected_frames_only_leaves_partial_project(tmp_path):
    frame_count = 5
    keyframes = [0, 2]
    writer, video = _make_writer(tmp_path, keyframes, frame_count)

    writer.write_selected_frames()

    assert (writer.entry_dir / "annotations" / "metadata" / "selected_frames.json").is_file()
    assert not (writer.entry_dir / "annotations" / "annotations.json").exists()


# ---------------------------------------------------------------------------
# AC13 — idempotent rebuild: a second full pass (fresh writer, same entry_dir,
# same inputs) reproduces byte-identical annotations.json / sidecar / JPEGs —
# never appended/duplicated (this is what makes /record/retry safe).
# ---------------------------------------------------------------------------


def test_ac13_two_full_passes_are_byte_identical(tmp_path):
    frame_count = 4
    keyframes = [0, 3]

    def _run():
        writer, video = _make_writer(tmp_path, keyframes, frame_count)
        for fn in range(frame_count):
            frame = np.full((H, W, 3), (10, 20, 30), dtype=np.uint8)
            dets = (
                make_fake_dets([[10, 10, 60, 60]], confidences=[0.9], size=(W, H))
                if fn in keyframes
                else make_fake_dets([], size=(W, H))
            )
            writer.add_frame(fn, frame, dets)
        writer.finalize()
        return writer

    writer1 = _run()
    ann1 = (writer1.entry_dir / "annotations" / "annotations.json").read_bytes()
    side1 = (writer1.entry_dir / "annotations" / "metadata" / "full_frame_detections.json").read_bytes()
    jpgs1 = {
        fn: (writer1.entry_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").read_bytes()
        for fn in keyframes
    }

    writer2 = _run()  # fresh instance, same entry_dir, identical inputs
    ann2 = (writer2.entry_dir / "annotations" / "annotations.json").read_bytes()
    side2 = (writer2.entry_dir / "annotations" / "metadata" / "full_frame_detections.json").read_bytes()
    jpgs2 = {
        fn: (writer2.entry_dir / "images" / f"{ENTRY_NAME}_f{fn:06d}.jpg").read_bytes()
        for fn in keyframes
    }

    assert ann1 == ann2
    assert side1 == side2
    assert jpgs1 == jpgs2
