"""TR3 AC1-AC3 (coco-helper half): behavior contract for backend.coco.build_annotation.

Blind-TDD test-writer pass — written from TR3-video-writer.md + IMPORT_FORMAT_VIDEO.md
§§2-5, against the frozen ``build_annotation`` signature. ``build_annotation`` is the
per-detection ``sv.Detections[i] -> COCO annotation`` logic extracted verbatim from
``DatasetWriter.flag`` (see backend/dataset_writer.py's inline block, read as the
source-of-truth for the clip/drop-degenerate/RLE/box-only-fallback math this helper
must reproduce exactly).

AC2's image-mode END-TO-END regression (DatasetWriter still produces T02's exact
annotations.json shape) is covered by T02's untouched ``tests/test_dataset_writer.py``
— NOT re-implemented here. This file only exercises the shared helper directly:
AC1 (geometry equivalence + degenerate-box -> None), the helper half of AC2 (extra
shallow-merge, image mode), and AC3 (image vs. video mode share geometry, differ only
in the per-mode ``extra`` fields).
"""

from __future__ import annotations

import numpy as np
import pycocotools.mask as mask_utils
import pytest

from backend.coco import build_annotation
from tests.fakes import make_fake_dets

# ---------------------------------------------------------------------------
# AC1 — helper geometry equivalence (the "one code path" guarantee): clip to
# bounds, drop degenerate (bw<=0 or bh<=0) -> None, RLE-encode the mask, all
# float()/int() casts, category_id==1, iscrowd==0.
# ---------------------------------------------------------------------------


def test_ac01_helper_geometry_equivalence():
    W, H = 1920, 1080
    # index 0: box runs off the right/bottom edge -> clips to [1800,1000,1920,1080]
    #          -> non-degenerate (120x80) -> kept.
    # index 1: box entirely outside the frame -> clips to zero area -> dropped.
    boxes = [
        [1800.0, 1000.0, 2000.0, 1200.0],
        [-500.0, -500.0, -100.0, -100.0],
    ]
    mask = np.zeros((2, H, W), dtype=bool)
    mask[0, 1000:1080, 1800:1920] = True
    dets = make_fake_dets(boxes, masks=mask, size=(W, H))

    ann = build_annotation(dets, 0, W, H, ann_id=1, image_id=1)
    assert ann is not None
    assert ann["id"] == 1
    assert ann["image_id"] == 1
    assert ann["category_id"] == 1
    assert ann["iscrowd"] == 0

    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = 1800.0, 1000.0, 1920.0, 1080.0
    bw, bh = clipped_x2 - clipped_x1, clipped_y2 - clipped_y1
    assert ann["bbox"] == pytest.approx([clipped_x1, clipped_y1, bw, bh], rel=1e-6)
    assert type(ann["bbox"][0]) is float

    assert "segmentation" in ann
    seg = ann["segmentation"]
    assert seg["size"] == [H, W]
    rle_for_decode = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
    decoded = mask_utils.decode(rle_for_decode)
    assert np.array_equal(decoded.astype(bool), mask[0])

    expected_area = int(mask_utils.area(rle_for_decode))
    assert ann["area"] == expected_area
    assert type(ann["area"]) is int

    # The fully-out-of-frame box clips to a zero-area box -> degenerate -> None.
    degenerate = build_annotation(dets, 1, W, H, ann_id=2, image_id=1)
    assert degenerate is None


def test_ac01_degenerate_box_from_zero_width_returns_none():
    # A box with equal x1/x2 (zero width, non-negative coords) is degenerate
    # independent of clipping — must still drop to None, not raise.
    W, H = 800, 600
    dets = make_fake_dets([[100.0, 100.0, 100.0, 400.0]], size=(W, H))

    ann = build_annotation(dets, 0, W, H, ann_id=1, image_id=1)
    assert ann is None


def test_ac01_empty_mask_falls_back_to_box_only_area():
    W, H = 1920, 1080
    box = [100.0, 100.0, 300.0, 300.0]
    zero_mask = np.zeros((1, H, W), dtype=bool)
    dets = make_fake_dets([box], masks=zero_mask, size=(W, H))

    ann = build_annotation(dets, 0, W, H, ann_id=7, image_id=2)
    assert ann is not None
    assert "segmentation" not in ann

    bw, bh = 200.0, 200.0
    assert ann["area"] == int(bw * bh)
    assert type(ann["area"]) is int
    assert ann["bbox"] == pytest.approx([100.0, 100.0, bw, bh], rel=1e-6)


# ---------------------------------------------------------------------------
# AC2 (helper half) — image-mode `extra` shallow-merges AFTER the core geometry
# keys; the merged dict still carries the geometry alongside the provenance
# fields. (The DatasetWriter end-to-end regression is T02's untouched suite.)
# ---------------------------------------------------------------------------


def test_ac02_image_mode_extra_merges_alongside_geometry():
    W, H = 1920, 1080
    box = [100.0, 100.0, 300.0, 300.0]
    dets = make_fake_dets([box], confidences=[0.83], size=(W, H))

    extra = {
        "confidence": float(dets.confidence[0]),
        "model_version": "model-explicit-v9",
        "confidence_threshold": 0.42,
    }
    ann = build_annotation(dets, 0, W, H, ann_id=5, image_id=3, extra=extra)

    assert ann is not None
    assert ann["id"] == 5
    assert ann["image_id"] == 3
    assert ann["category_id"] == 1
    assert ann["iscrowd"] == 0
    assert "bbox" in ann
    assert "area" in ann

    assert ann["confidence"] == pytest.approx(0.83, abs=1e-6)
    assert ann["model_version"] == "model-explicit-v9"
    assert ann["confidence_threshold"] == pytest.approx(0.42, abs=1e-9)


# ---------------------------------------------------------------------------
# AC3 — byte-identical keyframe structure: same sv.Detections through the
# helper with image-mode extra vs. video-mode extra yields EQUAL geometry keys;
# the video annotation carries video_id/track_id and none of the image-mode
# provenance keys, and vice versa.
# ---------------------------------------------------------------------------


def test_ac03_image_and_video_mode_share_geometry_differ_in_extra():
    W, H = 1920, 1080
    box = [50.0, 60.0, 250.0, 260.0]
    dets = make_fake_dets([box], confidences=[0.77], size=(W, H))

    image_extra = {
        "confidence": float(dets.confidence[0]),
        "model_version": "model-v1",
        "confidence_threshold": 0.5,
    }
    video_extra = {"video_id": 1, "track_id": 7}

    image_ann = build_annotation(dets, 0, W, H, ann_id=1, image_id=1, extra=image_extra)
    video_ann = build_annotation(dets, 0, W, H, ann_id=1, image_id=1, extra=video_extra)

    assert image_ann is not None
    assert video_ann is not None

    for key in ("bbox", "area", "category_id", "iscrowd"):
        assert image_ann[key] == video_ann[key]
    assert ("segmentation" in image_ann) == ("segmentation" in video_ann)
    if "segmentation" in image_ann:
        assert image_ann["segmentation"] == video_ann["segmentation"]

    assert video_ann["video_id"] == 1
    assert video_ann["track_id"] == 7
    for key in ("confidence", "model_version", "confidence_threshold"):
        assert key not in video_ann

    for key in ("video_id", "track_id"):
        assert key not in image_ann
