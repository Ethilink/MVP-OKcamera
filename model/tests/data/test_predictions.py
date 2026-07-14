from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from orc_model.data.predictions import PredictedClip, load_cached_detections


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _crop_bounds(box_xyxy: tuple[float, float, float, float], image_height: int, image_width: int):
    """Same floor/ceil clamping the decoder itself uses (`predictions.py`),
    so the fixture packs masks the same way the real cache does."""
    x1, y1, x2, y2 = box_xyxy
    x1 = min(max(int(np.floor(x1)), 0), image_width)
    y1 = min(max(int(np.floor(y1)), 0), image_height)
    x2 = min(max(int(np.ceil(x2)), x1), image_width)
    y2 = min(max(int(np.ceil(y2)), y1), image_height)
    return x1, y1, x2, y2


def _build_cache(
    image_height: int,
    image_width: int,
    entries: list[tuple[int, tuple[float, float, float, float], float, int, np.ndarray]],
) -> dict:
    """entries: (frame_index, box_xyxy, confidence, class_id, full_frame_mask)."""
    frame_indices, xyxy, confidence, class_id, mask_bits, mask_bit_counts = [], [], [], [], [], []

    for frame_index, box, conf, cid, mask_full in entries:
        x1, y1, x2, y2 = _crop_bounds(box, image_height, image_width)
        cropped = mask_full[y1:y2, x1:x2]
        frame_indices.append(frame_index)
        xyxy.append(box)
        confidence.append(conf)
        class_id.append(cid)
        mask_bits.append(np.packbits(cropped))
        mask_bit_counts.append(cropped.size)

    return {
        "frame_indices": np.array(frame_indices, dtype=np.int32),
        "xyxy": np.array(xyxy, dtype=np.float32).reshape(-1, 4),
        "confidence": np.array(confidence, dtype=np.float32),
        "class_id": np.array(class_id, dtype=np.int32),
        "mask_bits": np.concatenate(mask_bits) if mask_bits else np.empty(0, dtype=np.uint8),
        "mask_bit_counts": np.array(mask_bit_counts, dtype=np.int64),
        "image_height": np.int32(image_height),
        "image_width": np.int32(image_width),
    }


def _write_synthetic_video(video_dir: Path, n_frames: int, height: int, width: int) -> Path:
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "CLIP.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    for i in range(n_frames):
        writer.write(np.full((height, width, 3), i * 20, dtype=np.uint8))
    writer.release()
    return video_path


# ---------------------------------------------------------------------------
# load_cached_detections
# ---------------------------------------------------------------------------


def test_load_cached_detections_decodes_masks_and_groups_by_frame():
    h, w = 10, 10
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[1:4, 1:4] = True
    mask_b = np.zeros((h, w), dtype=bool)
    mask_b[5:8, 5:8] = True

    cache = _build_cache(
        h,
        w,
        [
            (0, (1.0, 1.0, 4.0, 4.0), 0.9, 0, mask_a),
            (0, (5.0, 5.0, 8.0, 8.0), 0.7, 0, mask_b),
            (2, (1.0, 1.0, 4.0, 4.0), 0.5, 0, mask_a),
        ],
    )

    detections_by_frame = load_cached_detections(cache)

    assert set(detections_by_frame) == {0, 2}

    frame0 = detections_by_frame[0]
    assert isinstance(frame0, sv.Detections)
    assert len(frame0) == 2
    np.testing.assert_allclose(frame0.confidence, [0.9, 0.7], atol=1e-5)
    assert np.array_equal(frame0.mask[0], mask_a)
    assert np.array_equal(frame0.mask[1], mask_b)

    frame2 = detections_by_frame[2]
    assert len(frame2) == 1
    assert np.array_equal(frame2.mask[0], mask_a)


def test_load_cached_detections_empty_cache():
    cache = _build_cache(10, 10, [])
    assert load_cached_detections(cache) == {}


# ---------------------------------------------------------------------------
# PredictedClip.from_cache
# ---------------------------------------------------------------------------


def test_predicted_clip_from_cache(tmp_path: Path):
    h, w = 10, 10
    clip_dir = tmp_path / "SYNTH_CLIP"
    _write_synthetic_video(clip_dir / "video", n_frames=5, height=h, width=w)

    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[1:4, 1:4] = True
    cache = _build_cache(h, w, [(2, (1.0, 1.0, 4.0, 4.0), 0.9, 0, mask_a)])
    np.savez(clip_dir / "detections_cache.npz", **cache)

    predicted = PredictedClip.from_cache("SYNTH_CLIP", data_dir=tmp_path)

    assert predicted.name == "SYNTH_CLIP"
    assert predicted.video_path == clip_dir / "video" / "CLIP.mp4"
    assert predicted.frame_indices == [2]
    assert len(predicted) == 1

    frame = predicted[2]
    assert frame.frame_index == 2
    assert np.array_equal(frame.to_detections().mask[0], mask_a)
    assert frame.load_image().shape == (h, w, 3)


def test_predicted_clip_iterates_frames_in_order(tmp_path: Path):
    h, w = 10, 10
    clip_dir = tmp_path / "SYNTH_CLIP"
    _write_synthetic_video(clip_dir / "video", n_frames=5, height=h, width=w)

    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[1:4, 1:4] = True
    cache = _build_cache(
        h,
        w,
        [
            (3, (1.0, 1.0, 4.0, 4.0), 0.9, 0, mask_a),
            (1, (1.0, 1.0, 4.0, 4.0), 0.9, 0, mask_a),
        ],
    )
    np.savez(clip_dir / "detections_cache.npz", **cache)

    predicted = PredictedClip.from_cache("SYNTH_CLIP", data_dir=tmp_path)

    assert [frame.frame_index for frame in predicted] == [1, 3]


def test_predicted_clip_getitem_missing_frame_raises_keyerror(tmp_path: Path):
    h, w = 10, 10
    clip_dir = tmp_path / "SYNTH_CLIP"
    _write_synthetic_video(clip_dir / "video", n_frames=3, height=h, width=w)

    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[1:4, 1:4] = True
    cache = _build_cache(h, w, [(0, (1.0, 1.0, 4.0, 4.0), 0.9, 0, mask_a)])
    np.savez(clip_dir / "detections_cache.npz", **cache)

    predicted = PredictedClip.from_cache("SYNTH_CLIP", data_dir=tmp_path)

    with pytest.raises(KeyError, match="frame 1"):
        predicted[1]


def test_predicted_clip_from_cache_requires_exactly_one_video_file(tmp_path: Path):
    clip_dir = tmp_path / "SYNTH_CLIP"
    (clip_dir / "video").mkdir(parents=True)
    cache = _build_cache(10, 10, [])
    np.savez(clip_dir / "detections_cache.npz", **cache)

    with pytest.raises(FileNotFoundError):
        PredictedClip.from_cache("SYNTH_CLIP", data_dir=tmp_path)
