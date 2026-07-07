import json
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import pytest

from orc_model.data.dataset import ClipDataset
from orc_model.data.models import Clip


# ---------------------------------------------------------------------------
# Shared helper — writes a small synthetic COCO-with-extras clip directory,
# same style/fields as tests/data/test_models.py's _write_synthetic_clip.
# ---------------------------------------------------------------------------


def _write_valid_clip(clip_dir: Path, *, video_id: int = 1) -> None:
    (clip_dir / "annotations").mkdir(parents=True)
    (clip_dir / "images").mkdir(parents=True)
    (clip_dir / "video").mkdir(parents=True)

    (clip_dir / "video" / "CLIP.mp4").write_bytes(b"")
    cv2.imwrite(str(clip_dir / "images" / "f0.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True
    rle = maskUtils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")

    coco_dict = {
        "info": {},
        "video": {
            "id": video_id,
            "file_name": "CLIP.mp4",
            "width": 10,
            "height": 10,
            "fps": 30.0,
            "frame_count": 100,
        },
        "categories": [
            {"id": 1, "name": "surgical_instrument", "supercategory": "instrument"}
        ],
        "images": [
            {
                "id": video_id * 1000,
                "video_id": video_id,
                "frame_number": 0,
                "width": 10,
                "height": 10,
                "file_name": "f0.jpg",
                "review_status": "reviewed",
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": video_id * 1000,
                "video_id": video_id,
                "category_id": 1,
                "track_id": 1,
                "bbox": [1.0, 1.0, 3.0, 3.0],
                "area": 9,
                "iscrowd": 0,
                "segmentation": rle,
                "gemini_bbox": [1, 1, 4, 4],
                "gemini_points": [[2, 2]],
                "instance_id": video_id * 10000 + 1,
                "gemini_description": "scalpel",
                "attributes": {"manually_edited": True},
            },
        ],
    }

    with open(clip_dir / "annotations" / "annotations.json", "w") as f:
        json.dump(coco_dict, f)


# ---------------------------------------------------------------------------
# ClipDataset.from_data_dir
# ---------------------------------------------------------------------------


def test_from_data_dir_skips_incomplete_clips(tmp_path: Path):
    _write_valid_clip(tmp_path / "CLIP_B", video_id=2)
    _write_valid_clip(tmp_path / "CLIP_A", video_id=1)

    # Missing images/ entirely.
    missing_images_dir = tmp_path / "CLIP_MISSING_IMAGES"
    (missing_images_dir / "annotations").mkdir(parents=True)
    (missing_images_dir / "annotations" / "annotations.json").write_text("{}")

    # Missing annotations/annotations.json entirely (dir exists, file doesn't).
    missing_annotations_dir = tmp_path / "CLIP_MISSING_ANNOTATIONS"
    (missing_annotations_dir / "annotations").mkdir(parents=True)
    (missing_annotations_dir / "images").mkdir(parents=True)

    with pytest.warns(UserWarning):
        dataset = ClipDataset.from_data_dir(tmp_path)

    assert len(dataset) == 2
    assert [clip.name for clip in dataset] == ["CLIP_A", "CLIP_B"]


def test_from_data_dir_ignores_non_directory_entries(tmp_path: Path):
    _write_valid_clip(tmp_path / "CLIP_A", video_id=1)
    (tmp_path / "stray_file.txt").write_text("not a clip")

    dataset = ClipDataset.from_data_dir(tmp_path)

    assert len(dataset) == 1
    assert dataset[0].name == "CLIP_A"


def test_from_data_dir_default_resolves_to_real_model_data_dir():
    import orc_model.data.dataset as dataset_module

    expected = Path(__file__).resolve().parents[2] / "data"
    actual = dataset_module._default_data_dir()

    assert actual == expected


# ---------------------------------------------------------------------------
# ClipDataset — constructed directly from a list of Clips
# ---------------------------------------------------------------------------


@pytest.fixture
def two_clips(tmp_path: Path) -> list[Clip]:
    _write_valid_clip(tmp_path / "CLIP_A", video_id=1)
    _write_valid_clip(tmp_path / "CLIP_B", video_id=2)
    return [
        Clip.from_directory(tmp_path / "CLIP_A"),
        Clip.from_directory(tmp_path / "CLIP_B"),
    ]


def test_len(two_clips: list[Clip]):
    dataset = ClipDataset(clips=two_clips)
    assert len(dataset) == 2


def test_iter(two_clips: list[Clip]):
    dataset = ClipDataset(clips=two_clips)
    assert list(dataset) == two_clips


def test_get_clip_found(two_clips: list[Clip]):
    dataset = ClipDataset(clips=two_clips)
    clip = dataset.get_clip("CLIP_B")
    assert clip.name == "CLIP_B"


def test_get_clip_not_found_lists_available_names(two_clips: list[Clip]):
    dataset = ClipDataset(clips=two_clips)
    with pytest.raises(KeyError) as exc_info:
        dataset.get_clip("NOPE")

    message = str(exc_info.value)
    assert "CLIP_A" in message
    assert "CLIP_B" in message


def test_getitem_str_key(two_clips: list[Clip]):
    dataset = ClipDataset(clips=two_clips)
    assert dataset["CLIP_A"].name == "CLIP_A"


def test_getitem_int_key(two_clips: list[Clip]):
    dataset = ClipDataset(clips=two_clips)
    assert dataset[0].name == "CLIP_A"
    assert dataset[1].name == "CLIP_B"
