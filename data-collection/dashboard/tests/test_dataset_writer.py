"""T02 AC1-AC13: behavior contract for backend.dataset_writer.DatasetWriter.

Blind-TDD test-writer pass — written from the T02 spec (the SPEC-dataset-writer.md
scaffolding file, since deleted, plus DASHBOARD.md). Asserts the on-disk COCO-VID shape, the (unscaled, single-resolution) box/mask math, atomic rewrite,
counters, and the validate_import.py gate. Does NOT assert on any internal
attribute/field beyond the frozen public interface (FlagResult fields,
DatasetWriter.flag/.n_flagged/.dataset_dir).
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_utils
import pytest

from backend.dataset_writer import DatasetWriter, FlagResult

# ---------------------------------------------------------------------------
# validate() — loaded directly from the script path per the task brief so we
# don't need a package/sys.path entry for docs/scripts.
# ---------------------------------------------------------------------------
_VP = Path(__file__).resolve().parents[2] / "docs" / "scripts" / "validate_import.py"
_spec = importlib.util.spec_from_file_location("validate_import", _VP)
validate_import = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_import)
validate = validate_import.validate


def _read_json(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "annotations" / "annotations.json").read_text())


# ---------------------------------------------------------------------------
# AC1 — constructor validation & laziness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    ["", ".hidden", "a/b", "a\\b", ".", "..", "sub/dir/name"],
)
def test_ac01_constructor_rejects_bad_dataset_names(tmp_path, bad_name):
    with pytest.raises(ValueError):
        DatasetWriter(tmp_path, bad_name, "model-v1")


def test_ac01_constructor_rejects_existing_collision(tmp_path):
    (tmp_path / "already_here").mkdir()
    with pytest.raises(FileExistsError):
        DatasetWriter(tmp_path, "already_here", "model-v1")


def test_ac01_constructor_rejects_collision_with_a_file(tmp_path):
    (tmp_path / "already_here").write_text("not a dataset")
    with pytest.raises(FileExistsError):
        DatasetWriter(tmp_path, "already_here", "model-v1")


def test_ac01_constructor_creates_no_directories(tmp_path):
    writer = DatasetWriter(tmp_path, "lazy_set", "model-v1")
    assert not (tmp_path / "lazy_set").exists()
    assert writer.n_flagged == 0
    assert writer.dataset_dir == tmp_path / "lazy_set"


def test_reuse_existing_dataset_appends_and_preserves_prior_data(
    tmp_path, fake_frame, fake_dets
):
    # Reusing a name RESUMES the on-disk dataset (operator request 2026-07-10):
    # prior images/annotations load so ids continue and old frames survive.
    frame = fake_frame(1920, 1080)
    dets = fake_dets([[100, 100, 300, 300]], size=(1920, 1080))

    first = DatasetWriter(tmp_path, "reused", "model-v1")
    first.flag(frame, dets, threshold=0.5)
    first.flag(frame, dets, threshold=0.5)
    assert first.resumed is False
    assert first.n_flagged == 2

    second = DatasetWriter(tmp_path, "reused", "model-v1")
    assert second.resumed is True
    assert second.n_flagged == 2  # loaded the two prior frames

    result = second.flag(frame, dets, threshold=0.5)
    assert result.file_name == "frame_00003.jpg"  # id continues, no overwrite
    assert (tmp_path / "reused" / "images" / "frame_00001.jpg").is_file()
    assert (tmp_path / "reused" / "images" / "frame_00003.jpg").is_file()

    doc = _read_json(tmp_path / "reused")
    assert [img["id"] for img in doc["images"]] == [1, 2, 3]
    # Undo removes only THIS writer's capture, never the two it resumed with.
    assert second.discard_last() == 3
    with pytest.raises(IndexError):
        second.discard_last()


# ---------------------------------------------------------------------------
# AC2 — first-flag folder creation, n=1
# ---------------------------------------------------------------------------


def test_ac02_first_flag_creates_folders_and_is_frame_one(tmp_path, fake_frame, fake_dets):
    writer = DatasetWriter(tmp_path, "set_ac02", "model-v1")
    frame = fake_frame(1920, 1080)
    dets = fake_dets([[100, 100, 300, 300]], size=(1920, 1080))

    result = writer.flag(frame, dets, threshold=0.5)

    dataset_dir = tmp_path / "set_ac02"
    assert dataset_dir.is_dir()
    assert (dataset_dir / "images").is_dir()
    assert (dataset_dir / "annotations").is_dir()
    assert (dataset_dir / "annotations" / "annotations.json").is_file()

    assert isinstance(result, FlagResult)
    assert result.image_id == 1
    assert result.file_name == "frame_00001.jpg"
    assert (dataset_dir / "images" / "frame_00001.jpg").is_file()
    assert writer.n_flagged == 1


# ---------------------------------------------------------------------------
# AC3 — saved JPEG == raw frame (shape + sampled pixels, JPEG-lossy tolerant)
# ---------------------------------------------------------------------------


def test_ac03_saved_jpeg_matches_raw_frame(tmp_path, fake_frame, fake_dets):
    writer = DatasetWriter(tmp_path, "set_ac03", "model-v1")
    frame = fake_frame(1920, 1080)
    dets = fake_dets([[10, 10, 50, 50]], size=(1920, 1080))

    result = writer.flag(frame, dets, threshold=0.5)

    saved_path = writer.dataset_dir / "images" / result.file_name
    saved = cv2.imread(str(saved_path))
    assert saved is not None
    assert saved.shape == frame.shape

    # Sample a grid of pixels; JPEG is lossy so allow a generous tolerance,
    # but this catches gross errors like overlay drawing, resizing, or
    # channel/orientation swaps.
    ys = np.linspace(0, frame.shape[0] - 1, 8, dtype=int)
    xs = np.linspace(0, frame.shape[1] - 1, 8, dtype=int)
    for y in ys:
        for x in xs:
            diff = np.abs(saved[y, x].astype(int) - frame[y, x].astype(int))
            assert diff.max() <= 12, f"pixel ({x},{y}) drifted too far: {diff}"


# ---------------------------------------------------------------------------
# AC4 — image record dims + review_status:"pending" + id==n
# ---------------------------------------------------------------------------


def test_ac04_image_record_dims_and_status(tmp_path, fake_frame, fake_dets):
    writer = DatasetWriter(tmp_path, "set_ac04", "model-v1")
    frame = fake_frame(1920, 1080)
    dets = fake_dets([[10, 10, 50, 50]], size=(1920, 1080))

    result = writer.flag(frame, dets, threshold=0.5)

    data = _read_json(writer.dataset_dir)
    assert len(data["images"]) == 1
    img = data["images"][0]
    assert img["id"] == result.image_id == 1
    assert img["file_name"] == "frame_00001.jpg"
    assert img["width"] == 1920
    assert img["height"] == 1080
    assert img["review_status"] == "pending"


# ---------------------------------------------------------------------------
# AC5 — boxes & masks written UNSCALED in the frame's own pixel coordinates
# (dets share the saved frame's resolution), RLE round-trips the mask as-is,
# segmentation.size == [H, W]. A non-square frame catches any x/y axis swap.
# ---------------------------------------------------------------------------


def test_ac05_bbox_and_mask_written_unscaled_with_rle_roundtrip(tmp_path, fake_frame, fake_dets):
    W, H = 1920, 1080  # non-square, so an x/y axis swap bug would be caught
    frame = fake_frame(W, H)
    box = [100.0, 100.0, 500.0, 400.0]  # fully inside bounds, no clipping
    mask = np.zeros((1, H, W), dtype=bool)
    mask[0, 200:500, 300:900] = True  # arbitrary region, independent of the box
    dets = fake_dets([box], masks=mask, size=(W, H))

    writer = DatasetWriter(tmp_path, "set_ac05", "model-v1")
    result = writer.flag(frame, dets, threshold=0.5)
    assert result.n_annotations == 1

    data = _read_json(writer.dataset_dir)
    ann = data["annotations"][0]

    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    expected_bbox = [x1, y1, bw, bh]  # no scaling
    assert ann["bbox"] == pytest.approx(expected_bbox, rel=1e-6)

    assert "segmentation" in ann
    seg = ann["segmentation"]
    assert seg["size"] == [H, W]

    rle_for_decode = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
    decoded = mask_utils.decode(rle_for_decode)
    assert np.array_equal(decoded.astype(bool), mask[0])  # round-trips as-is, no resize

    expected_area = int(mask_utils.area(rle_for_decode))
    assert ann["area"] == expected_area
    assert expected_area > 0


# ---------------------------------------------------------------------------
# AC6 — boxes are clipped to the frame; a fully-out-of-frame box -> no annotation
# ---------------------------------------------------------------------------


def test_ac06_out_of_frame_box_dropped_in_frame_kept(tmp_path, fake_frame, fake_dets):
    W, H = 960, 540
    frame = fake_frame(W, H)

    # Box 0: entirely beyond the frame -> clips to zero-area -> dropped.
    # Box 1: fully inside the frame -> kept.
    boxes = [[3000.0, 3000.0, 3100.0, 3100.0], [50.0, 50.0, 150.0, 150.0]]
    mask = np.zeros((2, H, W), dtype=bool)
    mask[1, 50:150, 50:150] = True
    dets = fake_dets(boxes, masks=mask, size=(W, H))

    writer = DatasetWriter(tmp_path, "set_ac06", "model-v1")
    result = writer.flag(frame, dets, threshold=0.5)

    assert result.n_annotations == 1
    data = _read_json(writer.dataset_dir)
    assert len(data["annotations"]) == 1

    ann = data["annotations"][0]
    assert ann["bbox"] == pytest.approx([50, 50, 100, 100], rel=1e-6)  # unscaled


def test_ac06_fully_out_of_frame_only_box_yields_zero_annotations(
    tmp_path, fake_frame, fake_dets
):
    W, H = 960, 540
    frame = fake_frame(W, H)
    dets = fake_dets([[-500.0, -500.0, -100.0, -100.0]], size=(W, H))

    writer = DatasetWriter(tmp_path, "set_ac06b", "model-v1")
    result = writer.flag(frame, dets, threshold=0.5)

    assert result.n_annotations == 0
    data = _read_json(writer.dataset_dir)
    # The image record is still written (this is a detection set with a
    # dropped box, not the empty-frame case) but carries no annotations.
    assert len(data["images"]) == 1
    assert len(data["annotations"]) == 0


# ---------------------------------------------------------------------------
# AC7 — all-zero mask -> no segmentation, area == int(bw*bh) box fallback
# ---------------------------------------------------------------------------


def test_ac07_all_zero_mask_falls_back_to_box_area(tmp_path, fake_frame, fake_dets):
    W, H = 1920, 1080
    frame = fake_frame(W, H)
    box = [100.0, 100.0, 300.0, 300.0]
    zero_mask = np.zeros((1, H, W), dtype=bool)
    dets = fake_dets([box], masks=zero_mask, size=(W, H))

    writer = DatasetWriter(tmp_path, "set_ac07", "model-v1")
    result = writer.flag(frame, dets, threshold=0.5)
    assert result.n_annotations == 1

    data = _read_json(writer.dataset_dir)
    ann = data["annotations"][0]

    assert "segmentation" not in ann
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    expected_area = int(bw * bh)  # no scaling
    assert ann["area"] == expected_area


# ---------------------------------------------------------------------------
# AC8 — JSON loads with stdlib json, no numpy scalars leaked
# ---------------------------------------------------------------------------


def test_ac08_json_round_trips_with_stdlib_and_plain_python_types(
    tmp_path, fake_frame, fake_dets
):
    frame = fake_frame(1920, 1080)
    dets = fake_dets([[10, 10, 60, 60]], confidences=[0.83], size=(1920, 1080))

    writer = DatasetWriter(tmp_path, "set_ac08", "model-v1")
    # If any numpy scalar leaked into the payload, stdlib json.dump would have
    # raised TypeError inside flag() already, so this call succeeding is part
    # of the assertion.
    writer.flag(frame, dets, threshold=0.4)

    raw = (writer.dataset_dir / "annotations" / "annotations.json").read_text()
    data = json.loads(raw)  # stdlib json, not a lenient/tolerant parser

    ann = data["annotations"][0]
    assert isinstance(ann["bbox"], list)
    for v in ann["bbox"]:
        assert type(v) is float
    assert type(ann["confidence"]) is float
    assert type(ann["area"]) is int
    assert type(data["images"][0]["width"]) is int
    assert type(data["images"][0]["height"]) is int


# ---------------------------------------------------------------------------
# AC9 — counters across 3 -> 0 -> 2 dets
# ---------------------------------------------------------------------------


def test_ac09_counters_across_multiple_flags(tmp_path, fake_frame, fake_dets):
    frame = fake_frame(1920, 1080)
    writer = DatasetWriter(tmp_path, "set_ac09", "model-v1")

    dets1 = fake_dets(
        [[10, 10, 60, 60], [70, 70, 140, 140], [200, 200, 260, 260]],
        size=(1920, 1080),
    )
    r1 = writer.flag(frame, dets1, threshold=0.5)
    assert r1.image_id == 1
    assert r1.n_annotations == 3

    dets2 = fake_dets([], size=(1920, 1080))
    r2 = writer.flag(frame, dets2, threshold=0.5)
    assert r2.image_id == 2
    assert r2.n_annotations == 0

    dets3 = fake_dets([[30, 30, 90, 90], [300, 300, 360, 360]], size=(1920, 1080))
    r3 = writer.flag(frame, dets3, threshold=0.5)
    assert r3.image_id == 3
    assert r3.n_annotations == 2

    assert writer.n_flagged == 3

    data = _read_json(writer.dataset_dir)
    assert [img["id"] for img in data["images"]] == [1, 2, 3]

    ann_ids = [a["id"] for a in data["annotations"]]
    assert ann_ids == [1, 2, 3, 4, 5]  # continuous across all frames

    image_ids_by_ann = [a["image_id"] for a in data["annotations"]]
    assert image_ids_by_ann == [1, 1, 1, 3, 3]  # none reference the empty frame (id 2)


# ---------------------------------------------------------------------------
# AC10 — atomic rewrite, no .tmp left after a successful flag
# ---------------------------------------------------------------------------


def test_ac10_no_tmp_file_remains_after_flags(tmp_path, fake_frame, fake_dets):
    frame = fake_frame(1920, 1080)
    writer = DatasetWriter(tmp_path, "set_ac10", "model-v1")

    for _ in range(3):
        dets = fake_dets([[10, 10, 60, 60]], size=(1920, 1080))
        writer.flag(frame, dets, threshold=0.5)

        ann_dir = writer.dataset_dir / "annotations"
        names = os.listdir(ann_dir)
        assert "annotations.json.tmp" not in names
        assert "annotations.json" in names

    # Final file must still be valid, complete JSON (rewritten wholesale each
    # time, not appended-to).
    data = _read_json(writer.dataset_dir)
    assert len(data["images"]) == 3


# ---------------------------------------------------------------------------
# AC11 — per-annotation provenance: confidence, model_version, threshold
# ---------------------------------------------------------------------------


def test_ac11_annotation_carries_provenance(tmp_path, fake_frame, fake_dets):
    frame = fake_frame(1920, 1080)
    dets = fake_dets([[10, 10, 60, 60]], confidences=[0.77], size=(1920, 1080))

    writer = DatasetWriter(tmp_path, "set_ac11", "model-explicit-v9")
    writer.flag(frame, dets, threshold=0.42)

    data = _read_json(writer.dataset_dir)
    ann = data["annotations"][0]

    assert ann["confidence"] == pytest.approx(0.77, abs=1e-6)
    assert ann["model_version"] == "model-explicit-v9"
    assert ann["confidence_threshold"] == pytest.approx(0.42, abs=1e-9)


# ---------------------------------------------------------------------------
# AC12 — exact categories, info.description == dataset_name, iso8601 date_created
# ---------------------------------------------------------------------------


def test_ac12_categories_info_and_stable_date_created(tmp_path, fake_frame, fake_dets):
    frame = fake_frame(1920, 1080)
    writer = DatasetWriter(tmp_path, "my_precious_set", "model-v1")

    dets = fake_dets([[10, 10, 60, 60]], size=(1920, 1080))
    writer.flag(frame, dets, threshold=0.5)
    data_after_first = _read_json(writer.dataset_dir)

    assert data_after_first["categories"] == [
        {"id": 1, "name": "surgical_instrument", "supercategory": ""}
    ]
    assert data_after_first["info"]["description"] == "my_precious_set"

    date_created_1 = data_after_first["info"]["date_created"]
    # Must not raise: ISO-8601 parseable.
    datetime.fromisoformat(date_created_1)

    writer.flag(frame, dets, threshold=0.5)
    data_after_second = _read_json(writer.dataset_dir)
    date_created_2 = data_after_second["info"]["date_created"]

    # Captured once at construction time, reused unchanged on every rewrite.
    assert date_created_2 == date_created_1
    assert data_after_second["categories"] == data_after_first["categories"]


# ---------------------------------------------------------------------------
# Regression — a failed cv2.imwrite must raise, not silently record a missing
# image; and the writer must stay retryable (counters untouched) so the next
# flag can succeed. cv2.imwrite returns False on failure instead of raising.
# ---------------------------------------------------------------------------


def test_failed_imwrite_raises_and_leaves_writer_retryable(
    tmp_path, fake_frame, fake_dets, monkeypatch
):
    frame = fake_frame(1920, 1080)
    dets = fake_dets([[10, 10, 60, 60]], size=(1920, 1080))
    writer = DatasetWriter(tmp_path, "set_imwrite_fail", "model-v1")

    import backend.dataset_writer as dw

    monkeypatch.setattr(dw.cv2, "imwrite", lambda *a, **k: False)
    with pytest.raises(OSError):
        writer.flag(frame, dets, threshold=0.5)

    # No state advanced: no image record, no annotations, no annotations.json,
    # counter still zero — the flag is as if it never happened.
    assert writer.n_flagged == 0
    assert writer.images == []
    assert writer.annotations == []
    assert not (writer.dataset_dir / "annotations" / "annotations.json").exists()

    # Recover: with imwrite working again the next flag succeeds as frame 1
    # (the earlier failure did not consume the id or trip on the existing dirs).
    monkeypatch.undo()
    result = writer.flag(frame, dets, threshold=0.5)
    assert result.image_id == 1
    assert result.file_name == "frame_00001.jpg"
    assert (writer.dataset_dir / "images" / "frame_00001.jpg").is_file()
    assert writer.n_flagged == 1


# ---------------------------------------------------------------------------
# Regression — discard_last() undoes the most recent flag: its image record, its
# annotations (the contiguous tail), and its JPEG; the freed id is cleanly reused.
# ---------------------------------------------------------------------------


def test_discard_last_removes_tail_and_reuses_id(tmp_path, fake_frame, fake_dets):
    frame = fake_frame(1920, 1080)
    writer = DatasetWriter(tmp_path, "set_discard", "model-v1")

    writer.flag(frame, fake_dets([[10, 10, 60, 60]], size=(1920, 1080)), threshold=0.5)
    writer.flag(
        frame,
        fake_dets([[20, 20, 80, 80], [100, 100, 160, 160]], size=(1920, 1080)),
        threshold=0.5,
    )
    assert writer.n_flagged == 2

    removed = writer.discard_last()
    assert removed == 2
    assert writer.n_flagged == 1
    assert not (writer.dataset_dir / "images" / "frame_00002.jpg").exists()

    data = _read_json(writer.dataset_dir)
    assert [i["id"] for i in data["images"]] == [1]
    # image 2's two annotations are gone; image 1's single annotation remains.
    assert [a["image_id"] for a in data["annotations"]] == [1]
    assert [a["id"] for a in data["annotations"]] == [1]

    # The freed id is reused with no gap or duplicate on the next flag.
    r = writer.flag(frame, fake_dets([[30, 30, 90, 90]], size=(1920, 1080)), threshold=0.5)
    assert r.image_id == 2
    assert r.file_name == "frame_00002.jpg"
    assert (writer.dataset_dir / "images" / "frame_00002.jpg").is_file()

    # And the result still validates cleanly (ids continuous, no dupes).
    errors, _warnings = validate(writer.dataset_dir)
    assert errors == []


def test_discard_last_on_empty_raises(tmp_path):
    writer = DatasetWriter(tmp_path, "set_discard_empty", "model-v1")
    with pytest.raises(IndexError):
        writer.discard_last()


# ---------------------------------------------------------------------------
# AC13 — produced dataset passes validate() with zero errors
# ---------------------------------------------------------------------------


def test_ac13_produced_dataset_passes_validate(tmp_path, fake_frame, fake_dets):
    frame = fake_frame(1920, 1080)
    writer = DatasetWriter(tmp_path, "set_ac13", "model-v1")

    # A frame with a masked detection...
    dets1 = fake_dets(
        [[100, 100, 400, 400]],
        masks=None,
        confidences=[0.9],
        size=(1920, 1080),
    )
    writer.flag(frame, dets1, threshold=0.5)

    # ...an empty frame (false-negative capture)...
    dets2 = fake_dets([], size=(1920, 1080))
    writer.flag(frame, dets2, threshold=0.5)

    # ...and a frame with a box-only (all-zero mask) detection.
    zero_mask = np.zeros((1, 1080, 1920), dtype=bool)
    dets3 = fake_dets(
        [[50, 50, 250, 250]], masks=zero_mask, confidences=[0.6], size=(1920, 1080)
    )
    writer.flag(frame, dets3, threshold=0.3)

    errors, _warnings = validate(writer.dataset_dir)
    assert errors == []
