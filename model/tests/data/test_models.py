import io
import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import pytest
import supervision as sv
from PIL import Image

from orc_model.data.models import BBox, Clip, Frame, InstrumentAnnotation


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------


def test_bbox_xyxy():
    bbox = BBox(x=10.0, y=20.0, width=5.0, height=8.0)
    assert bbox.xyxy == (10.0, 20.0, 15.0, 28.0)


def test_bbox_is_frozen():
    bbox = BBox(x=0.0, y=0.0, width=1.0, height=1.0)
    with pytest.raises(Exception):
        bbox.x = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# InstrumentAnnotation
# ---------------------------------------------------------------------------


def _make_mask(h: int = 10, w: int = 10) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[2:5, 3:7] = True
    return mask


def _encode_mask(mask: np.ndarray) -> dict:
    rle = maskUtils.encode(np.asfortranarray(mask))
    # pycocotools returns bytes for 'counts'; JSON-friendliness isn't required
    # here since we pass the dict straight through, but keep it realistic.
    return rle


def test_instrument_annotation_to_mask_roundtrips():
    original_mask = _make_mask()
    rle = _encode_mask(original_mask)

    annotation = InstrumentAnnotation(
        track_id=1,
        instance_id=1300001,
        description="long metallic drill bit",
        bbox=BBox(x=3.0, y=2.0, width=4.0, height=3.0),
        manually_edited=False,
        propagated_from=None,
        segmentation=rle,
    )

    decoded_mask = annotation.to_mask()

    assert decoded_mask.dtype == bool
    assert np.array_equal(decoded_mask, original_mask)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


def _make_annotation(track_id: int, description: str, bbox: BBox, h: int, w: int) -> InstrumentAnnotation:
    mask = np.zeros((h, w), dtype=bool)
    mask[1:3, 1:3] = True
    return InstrumentAnnotation(
        track_id=track_id,
        instance_id=track_id,
        description=description,
        bbox=bbox,
        manually_edited=False,
        propagated_from=None,
        segmentation=_encode_mask(mask),
    )


def test_frame_load_image(tmp_path: Path):
    image_path = tmp_path / "synthetic.jpg"
    synthetic_image = np.zeros((20, 30, 3), dtype=np.uint8)
    synthetic_image[:, :, 1] = 255  # green image, arbitrary
    cv2.imwrite(str(image_path), synthetic_image)

    frame = Frame(
        frame_number=0,
        file_path=image_path,
        width=30,
        height=20,
        annotations=[],
    )

    loaded = frame.load_image()
    assert loaded.shape == (20, 30, 3)


def test_frame_image_returns_pil_image(tmp_path: Path):
    image_path = tmp_path / "synthetic.jpg"
    synthetic_image = np.zeros((20, 30, 3), dtype=np.uint8)
    synthetic_image[:, :, 1] = 255  # green image, arbitrary
    cv2.imwrite(str(image_path), synthetic_image)

    frame = Frame(
        frame_number=0,
        file_path=image_path,
        width=30,
        height=20,
        annotations=[],
    )

    assert isinstance(frame.image, Image.Image)


def test_frame_image_size_matches_loaded_pixel_dimensions(tmp_path: Path):
    image_path = tmp_path / "synthetic.jpg"
    synthetic_image = np.zeros((20, 30, 3), dtype=np.uint8)
    synthetic_image[:, :, 1] = 255  # green image, arbitrary
    cv2.imwrite(str(image_path), synthetic_image)

    frame = Frame(
        frame_number=0,
        file_path=image_path,
        width=30,
        height=20,
        annotations=[],
    )

    # PIL's .size is (width, height).
    assert frame.image.size == (30, 20)


def test_frame_image_converts_bgr_to_rgb(tmp_path: Path):
    image_path = tmp_path / "synthetic.png"
    # cv2.imwrite writes an array whose channel order is BGR. Use a
    # distinct, non-symmetric color so a channel swap would be caught:
    # pure blue in BGR terms is (255, 0, 0), which should show up as
    # (0, 0, 255) in RGB terms.
    synthetic_image = np.zeros((10, 10, 3), dtype=np.uint8)
    synthetic_image[5, 5] = (255, 0, 0)  # BGR: pure blue
    cv2.imwrite(str(image_path), synthetic_image)

    frame = Frame(
        frame_number=0,
        file_path=image_path,
        width=10,
        height=10,
        annotations=[],
    )

    pixel = frame.image.getpixel((5, 5))
    assert pixel == (0, 0, 255)  # RGB: pure blue


def test_frame_repr_png_returns_valid_roundtrippable_png(tmp_path: Path):
    image_path = tmp_path / "synthetic.png"
    synthetic_image = np.zeros((10, 10, 3), dtype=np.uint8)
    synthetic_image[5, 5] = (255, 0, 0)  # BGR: pure blue
    cv2.imwrite(str(image_path), synthetic_image)

    frame = Frame(
        frame_number=0,
        file_path=image_path,
        width=10,
        height=10,
        annotations=[],
    )

    result = frame._repr_png_()
    assert isinstance(result, bytes)

    decoded = Image.open(io.BytesIO(result))
    decoded.load()
    expected = frame.image
    assert decoded.size == expected.size
    assert list(decoded.getdata()) == list(expected.getdata())


def test_frame_to_detections_with_annotations():
    h, w = 10, 10
    annotations = [
        _make_annotation(1, "rongeur", BBox(x=1.0, y=1.0, width=3.0, height=3.0), h, w),
        _make_annotation(2, "scalpel", BBox(x=4.0, y=4.0, width=2.0, height=2.0), h, w),
    ]
    frame = Frame(
        frame_number=0,
        file_path=Path("unused.jpg"),
        width=w,
        height=h,
        annotations=annotations,
    )

    detections = frame.to_detections()

    assert isinstance(detections, sv.Detections)
    assert len(detections) == 2
    assert detections.xyxy.shape == (2, 4)
    np.testing.assert_allclose(detections.xyxy[0], annotations[0].bbox.xyxy)
    np.testing.assert_allclose(detections.xyxy[1], annotations[1].bbox.xyxy)
    assert detections.mask is not None
    assert detections.mask.shape == (2, h, w)
    assert np.array_equal(detections.mask[0], annotations[0].to_mask())
    assert np.array_equal(detections.mask[1], annotations[1].to_mask())
    assert list(detections.data["description"]) == ["rongeur", "scalpel"]


def test_frame_to_detections_empty():
    frame = Frame(
        frame_number=0,
        file_path=Path("unused.jpg"),
        width=10,
        height=10,
        annotations=[],
    )

    detections = frame.to_detections()

    empty = sv.Detections.empty()
    assert isinstance(detections, sv.Detections)
    assert len(detections) == 0
    assert detections.xyxy.shape == empty.xyxy.shape


# ---------------------------------------------------------------------------
# Clip.from_directory
# ---------------------------------------------------------------------------


def _write_synthetic_clip(clip_dir: Path) -> dict:
    """Build a small synthetic COCO-with-extras dataset on disk, mimicking the
    real clip_dir/{annotations,images,video} layout, and return the raw dict
    that was written (for cross-checking in assertions)."""
    (clip_dir / "annotations").mkdir(parents=True)
    (clip_dir / "images").mkdir(parents=True)
    (clip_dir / "video").mkdir(parents=True)

    # Video placeholder file - only its path is used, never opened by from_directory.
    (clip_dir / "video" / "CLIP.mp4").write_bytes(b"")

    # Real synthetic images, written so width/height are self-consistent
    # (from_directory trusts the COCO json's declared width/height, but let's
    # keep them real for good measure / potential future tests).
    for fname in ("f0.jpg", "f1.jpg", "f2.jpg"):
        cv2.imwrite(str(clip_dir / "images" / fname), np.zeros((10, 10, 3), dtype=np.uint8))

    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[1:4, 1:4] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[5:8, 5:8] = True

    def _encode_for_json(mask: np.ndarray) -> dict:
        # Real annotations.json store RLE 'counts' as a str (compressed RLE),
        # not raw bytes, since it goes through json.dump. Match that here.
        rle = maskUtils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        return rle

    coco_dict = {
        "info": {},
        "video": {
            "id": 42,
            "file_name": "CLIP.mp4",
            "width": 10,
            "height": 10,
            "fps": 30.0,
            "frame_count": 300,
        },
        "categories": [
            {"id": 1, "name": "surgical_instrument", "supercategory": "instrument"}
        ],
        "images": [
            {
                "id": 1001,
                "video_id": 42,
                "frame_number": 5,
                "width": 10,
                "height": 10,
                "file_name": "f1.jpg",
                "review_status": "reviewed",
            },
            {
                "id": 1000,
                "video_id": 42,
                "frame_number": 0,
                "width": 10,
                "height": 10,
                "file_name": "f0.jpg",
                "review_status": "reviewed",
            },
            {
                # Zero-annotation image, deliberately out of frame_number order.
                "id": 1002,
                "video_id": 42,
                "frame_number": 10,
                "width": 10,
                "height": 10,
                "file_name": "f2.jpg",
                "review_status": "reviewed",
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1000,
                "video_id": 42,
                "category_id": 1,
                "track_id": 1,
                "bbox": [1.0, 1.0, 3.0, 3.0],
                "area": 9,
                "iscrowd": 0,
                "segmentation": _encode_for_json(mask_a),
                "gemini_bbox": [1, 1, 4, 4],
                "gemini_points": [[2, 2]],
                "instance_id": 420001,
                "gemini_description": "scalpel",
                "attributes": {"manually_edited": True},
            },
            {
                "id": 2,
                "image_id": 1001,
                "video_id": 42,
                "category_id": 1,
                "track_id": 2,
                "bbox": [5.0, 5.0, 3.0, 3.0],
                "area": 9,
                "iscrowd": 0,
                "segmentation": _encode_for_json(mask_b),
                "gemini_bbox": [5, 5, 8, 8],
                "gemini_points": [[6, 6]],
                "instance_id": 420002,
                "gemini_description": "forceps",
                # no "attributes" key at all
            },
        ],
    }

    with open(clip_dir / "annotations" / "annotations.json", "w") as f:
        json.dump(coco_dict, f)

    return coco_dict


def test_clip_from_directory(tmp_path: Path):
    clip_dir = tmp_path / "SYNTH_CLIP"
    _write_synthetic_clip(clip_dir)

    clip = Clip.from_directory(clip_dir)

    assert clip.name == "SYNTH_CLIP"
    assert clip.video_path == clip_dir / "video" / "CLIP.mp4"
    assert clip.fps == 30.0
    assert clip.frame_count == 300
    assert len(clip) == 3

    # Sorted by frame_number: f0 (0), f1 (5), f2 (10).
    assert [frame.frame_number for frame in clip.frames] == [0, 5, 10]
    assert clip[0].file_path == clip_dir / "images" / "f0.jpg"
    assert clip[1].file_path == clip_dir / "images" / "f1.jpg"
    assert clip[2].file_path == clip_dir / "images" / "f2.jpg"

    # Per-frame annotation counts.
    assert len(clip[0].annotations) == 1
    assert len(clip[1].annotations) == 1
    assert len(clip[2].annotations) == 0  # zero-annotation edge case

    ann_a = clip[0].annotations[0]
    assert ann_a.track_id == 1
    assert ann_a.instance_id == 420001
    assert ann_a.description == "scalpel"
    assert ann_a.bbox == BBox(x=1.0, y=1.0, width=3.0, height=3.0)
    assert ann_a.manually_edited is True
    assert ann_a.propagated_from is None
    mask_a_expected = np.zeros((10, 10), dtype=bool)
    mask_a_expected[1:4, 1:4] = True
    assert np.array_equal(ann_a.to_mask(), mask_a_expected)

    ann_b = clip[1].annotations[0]
    assert ann_b.track_id == 2
    assert ann_b.instance_id == 420002
    assert ann_b.description == "forceps"
    # Default when "attributes" key is entirely absent.
    assert ann_b.manually_edited is False
    assert ann_b.propagated_from is None
    mask_b_expected = np.zeros((10, 10), dtype=bool)
    mask_b_expected[5:8, 5:8] = True
    assert np.array_equal(ann_b.to_mask(), mask_b_expected)


def test_clip_from_directory_suppresses_pycocotools_stdout(tmp_path: Path, capsys):
    clip_dir = tmp_path / "SYNTH_CLIP_STDOUT"
    _write_synthetic_clip(clip_dir)

    Clip.from_directory(clip_dir)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_instrument_annotation_propagated_from_via_clip(tmp_path: Path):
    clip_dir = tmp_path / "SYNTH_CLIP2"
    coco_dict = _write_synthetic_clip(clip_dir)
    # Add a "propagated_from" attributes case directly to the written json,
    # to exercise that branch distinctly from "manually_edited".
    coco_dict["annotations"][1]["attributes"] = {"propagated_from": 7}
    with open(clip_dir / "annotations" / "annotations.json", "w") as f:
        json.dump(coco_dict, f)

    clip = Clip.from_directory(clip_dir)
    ann_b = clip[1].annotations[0]
    assert ann_b.manually_edited is False
    assert ann_b.propagated_from == 7
