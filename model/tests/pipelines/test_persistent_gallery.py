"""Behavioural tests for `persistent_gallery` — SPEC §B-L (loader).

Contract: `docs/wayfinder/session-linker/assets/T08-T10/T08-SPEC.md`.

These tests synthesise tiny COCO specimen folders on disk and inject a
recording fake matcher, so `ChampionMethod` (DINOv2-B) never loads.

Do NOT hard-assert `DEFAULT_INSTRUMENTS_DIR`'s value or any crop-quality
internal — they are tunables (SPEC "Tunable parameters"). What is asserted
here is behaviour: which folders are scanned, how the dict is keyed, that
embedding happens exactly once per specimen, and that every malformed input
degrades to a skip rather than an exception.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from orc_model.pipelines.persistent_gallery import (
    DEFAULT_INSTRUMENTS_DIR,
    load_persistent_galleries,
)

# --------------------------------------------------------------------------
# COCO specimen-folder synthesis
# --------------------------------------------------------------------------

IMG_H = 64
IMG_W = 64

# x, y, w, h -> a 32-row x 24-col crop, comfortably above B-L2's w/h >= 4 gate.
BBOX = (16, 8, 24, 32)

# Deliberately asymmetric: an implementation that loads BGR instead of RGB
# recovers (60, 40, 200) here, so the channel order is observable.
SPECIMEN_RGB = (200, 40, 60)
BACKGROUND_RGB = (8, 8, 8)


def _rle(box, size=(IMG_H, IMG_W)) -> dict:
    """A real COCO RLE ({"size", "counts"}) for a solid box — the exact shape
    `pycocotools.mask.decode` consumes, counts as a str like the real files."""
    h, w = size
    x, y, bw, bh = box
    m = np.zeros((h, w), dtype=np.uint8)
    m[y : y + bh, x : x + bw] = 1
    encoded = coco_mask.encode(np.asfortranarray(m))
    return {"size": [h, w], "counts": encoded["counts"].decode("ascii")}


def _write_image(path: Path, box, rgb) -> None:
    arr = np.full((IMG_H, IMG_W, 3), BACKGROUND_RGB, dtype=np.uint8)
    x, y, bw, bh = box
    arr[y : y + bh, x : x + bw] = rgb
    # quality=100 + no chroma subsampling: JPEG stays within a few counts of
    # the painted colour, so the RGB assertion below has real signal.
    Image.fromarray(arr).save(path, quality=100, subsampling=0)


def write_specimen(
    instruments_dir: Path,
    n: int,
    *,
    n_images: int = 2,
    box=BBOX,
    extra_boxes=(),
    rgb=SPECIMEN_RGB,
    repr_strings: bool = False,
    folder_name: str | None = None,
    corrupt_images: bool = False,
    annotations_text: str | None = None,
) -> Path:
    """Write `instrument{n}/{images,annotations}` in the layout B-L1 describes.

    `extra_boxes` adds further annotations to the FIRST image (multi-crop
    frame). `repr_strings` stores bbox/segmentation as python-repr strings
    instead of real JSON values — B-L2 says both must parse.
    """
    root = instruments_dir / (folder_name or f"instrument{n}")
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    for i in range(n_images):
        file_name = f"frame_{i + 1:05d}.jpg"
        path = root / "images" / file_name
        if corrupt_images:
            path.write_bytes(b"not a jpeg at all")
        else:
            _write_image(path, box, rgb)
        images.append(
            {"id": i + 1, "width": IMG_W, "height": IMG_H, "file_name": file_name}
        )
        boxes = [box, *extra_boxes] if i == 0 else [box]
        for b in boxes:
            seg = _rle(b)
            bbox_value = [float(v) for v in b]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": 1,
                    "bbox": repr(bbox_value) if repr_strings else bbox_value,
                    "segmentation": repr(seg) if repr_strings else seg,
                    "iscrowd": 0,
                    "area": int(b[2] * b[3]),
                }
            )
            ann_id += 1

    text = annotations_text
    if text is None:
        text = json.dumps(
            {
                "info": {},
                "categories": [{"id": 1, "name": "surgical_instrument"}],
                "images": images,
                "annotations": annotations,
            }
        )
    (root / "annotations" / "annotations.json").write_text(text)
    return root


# --------------------------------------------------------------------------
# Fake matcher — records build_gallery, returns an opaque gallery object
# --------------------------------------------------------------------------


class FakeGallery:
    """Opaque stand-in: `.views` (n, D) ndarray + `.identity` (SPEC
    "Dependencies / reuse")."""

    def __init__(self, views: np.ndarray, identity=None):
        self.views = views
        self.identity = identity


class RecordingMatcher:
    """Only `build_gallery` is exercised by the loader (B-L3)."""

    def __init__(self):
        self.build_gallery_calls: list[dict] = []

    def build_gallery(self, crops, masks, meta):
        self.build_gallery_calls.append(
            {
                "crops": [np.array(c, copy=True) for c in crops],
                "masks": [np.array(m, copy=True) for m in masks],
                "meta": dict(meta),
            }
        )
        return FakeGallery(
            views=np.zeros((max(len(crops), 1), 4), dtype=np.float32),
            identity=meta.get("identity"),
        )


@pytest.fixture
def instruments_dir(tmp_path) -> Path:
    d = tmp_path / "instruments"
    d.mkdir()
    return d


def calls_by_identity(matcher: RecordingMatcher) -> dict:
    return {c["meta"].get("identity"): c for c in matcher.build_gallery_calls}


# --------------------------------------------------------------------------
# B-L1 / B-L5 — scanning and the returned dict
# --------------------------------------------------------------------------


def test_b_l1_scans_instrument_folders_and_keys_by_the_parsed_specimen_number(
    instruments_dir,
):
    write_specimen(instruments_dir, 3)
    write_specimen(instruments_dir, 7)
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {3, 7}, (
        "the specimen number is parsed from the folder name (instrument3 -> 3), "
        "not assigned by enumeration order"
    )


def test_b_l5_returns_int_keyed_galleries_carrying_their_specimen_identity(
    instruments_dir,
):
    write_specimen(instruments_dir, 5, n_images=3)
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {5}
    assert all(isinstance(key, int) for key in galleries), "keys are specimen numbers"
    gallery = galleries[5]
    assert gallery.views.ndim == 2, "an embedded gallery exposes (n, D) views"
    assert gallery.views.shape[0] >= 1, "a usable specimen must contribute views"


def test_b_l1_accepts_a_str_path(instruments_dir):
    write_specimen(instruments_dir, 2)
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, str(instruments_dir))

    assert set(galleries) == {2}


def test_b_l1_default_instruments_dir_is_resolved_from_the_module_not_the_cwd(
    tmp_path, monkeypatch
):
    """SPEC: resolved relative to this module's file, NOT the process cwd."""
    assert DEFAULT_INSTRUMENTS_DIR.is_absolute()
    resolved_before = Path(DEFAULT_INSTRUMENTS_DIR)

    monkeypatch.chdir(tmp_path)

    assert Path(DEFAULT_INSTRUMENTS_DIR) == resolved_before
    assert DEFAULT_INSTRUMENTS_DIR.exists(), (
        "the default must point at the real shipped specimen photos regardless "
        "of where the process was started from"
    )


# --------------------------------------------------------------------------
# B-L3 — embed once, one build_gallery call per specimen
# --------------------------------------------------------------------------


def test_b_l3_embeds_each_specimen_exactly_once(instruments_dir):
    write_specimen(instruments_dir, 3, n_images=3)
    write_specimen(instruments_dir, 7, n_images=2)
    matcher = RecordingMatcher()

    load_persistent_galleries(matcher, instruments_dir)

    assert len(matcher.build_gallery_calls) == 2, (
        "exactly one build_gallery() call per specimen — embedding happens here, "
        "once (SPEC Trap 5)"
    )
    by_identity = calls_by_identity(matcher)
    assert set(by_identity) == {3, 7}, "each call is tagged with its specimen number"
    assert len(by_identity[3]["crops"]) == 3, "all of a specimen's views go in one call"
    assert len(by_identity[7]["crops"]) == 2


def test_b_l3_passes_one_mask_per_crop(instruments_dir):
    write_specimen(instruments_dir, 4, n_images=3)
    matcher = RecordingMatcher()

    load_persistent_galleries(matcher, instruments_dir)

    by_identity = calls_by_identity(matcher)
    assert 4 in by_identity, "expected specimen 4 to be embedded"
    call = by_identity[4]
    assert len(call["crops"]) == len(call["masks"]) == 3


# --------------------------------------------------------------------------
# B-L2 — COCO handling
# --------------------------------------------------------------------------


def test_b_l2_parses_python_repr_string_bbox_and_segmentation(instruments_dir):
    """The real files store these as python-repr strings, not JSON values."""
    write_specimen(instruments_dir, 6, n_images=2, repr_strings=True)
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {6}, "repr-string bbox/segmentation must parse, not skip"
    assert len(calls_by_identity(matcher)[6]["crops"]) == 2


def test_b_l2_crop_is_the_bbox_cropped_rgb_image_with_an_aligned_mask(instruments_dir):
    write_specimen(instruments_dir, 1, n_images=1)
    matcher = RecordingMatcher()

    load_persistent_galleries(matcher, instruments_dir)

    by_identity = calls_by_identity(matcher)
    assert 1 in by_identity, "expected specimen 1 to be embedded"
    call = by_identity[1]
    crop = call["crops"][0]
    mask = call["masks"][0]
    _x, _y, box_w, box_h = BBOX

    assert crop.shape == (box_h, box_w, 3), "crop is cropped to its bbox"
    assert mask.shape[:2] == (box_h, box_w), "the mask is aligned to the crop"
    assert mask.astype(bool).all(), (
        "this specimen's RLE covers exactly its bbox, so every crop pixel is masked in"
    )

    centre = crop[box_h // 2, box_w // 2]
    assert np.allclose(centre, SPECIMEN_RGB, atol=20), (
        f"expected the crop's centre pixel to be RGB {SPECIMEN_RGB}, got "
        f"{tuple(int(v) for v in centre)} — a BGR read would give "
        f"{SPECIMEN_RGB[::-1]}"
    )


def test_b_l2_skips_annotations_with_a_degenerate_bbox(instruments_dir):
    """B-L2: skip annotations with bbox w < 4 or h < 4."""
    write_specimen(
        instruments_dir,
        2,
        n_images=1,
        extra_boxes=[(40, 40, 3, 20), (44, 44, 20, 3)],  # w < 4, then h < 4
    )
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {2}
    assert len(calls_by_identity(matcher)[2]["crops"]) == 1, (
        "only the one non-degenerate annotation may become a crop"
    )


# --------------------------------------------------------------------------
# B-L4 — robustness: skip and log, never raise
# --------------------------------------------------------------------------


def test_b_l4_missing_directory_returns_empty_and_never_raises(tmp_path):
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, tmp_path / "does-not-exist")

    assert galleries == {}
    assert not matcher.build_gallery_calls, "nothing to embed when there is no dir"


def test_b_l4_empty_directory_returns_empty(instruments_dir):
    matcher = RecordingMatcher()

    assert load_persistent_galleries(matcher, instruments_dir) == {}


def test_b_l4_unparseable_annotations_skip_only_that_specimen(instruments_dir):
    write_specimen(instruments_dir, 3)
    write_specimen(instruments_dir, 4, annotations_text="{ this is not json")
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {3}, "a broken specimen is skipped; its siblings survive"


def test_b_l4_missing_annotations_file_skips_only_that_specimen(instruments_dir):
    write_specimen(instruments_dir, 3)
    root = write_specimen(instruments_dir, 4)
    (root / "annotations" / "annotations.json").unlink()
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {3}


def test_b_l4_unreadable_images_skip_only_that_specimen(instruments_dir):
    write_specimen(instruments_dir, 3)
    write_specimen(instruments_dir, 4, corrupt_images=True)
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {3}, (
        "a specimen whose images cannot be decoded yields no usable crops and is "
        "skipped — never an exception"
    )


@pytest.mark.parametrize("folder_name", ["notes", "instrument", "instrumentX", "instrument-3"])
def test_b_l4_folders_that_are_not_instrument_n_are_ignored(instruments_dir, folder_name):
    write_specimen(instruments_dir, 3)
    write_specimen(instruments_dir, 99, folder_name=folder_name)
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {3}, f"{folder_name!r} is not instrument{{N}} — skip it"


def test_b_l4_stray_files_beside_the_specimen_folders_are_ignored(instruments_dir):
    write_specimen(instruments_dir, 3)
    (instruments_dir / "instrument9.txt").write_text("not a folder")
    (instruments_dir / "README.md").write_text("notes")
    matcher = RecordingMatcher()

    galleries = load_persistent_galleries(matcher, instruments_dir)

    assert set(galleries) == {3}


def test_b_l4_zero_usable_specimens_returns_empty(instruments_dir):
    write_specimen(instruments_dir, 3, annotations_text="{ broken")
    write_specimen(instruments_dir, 4, corrupt_images=True)
    matcher = RecordingMatcher()

    assert load_persistent_galleries(matcher, instruments_dir) == {}, (
        "no usable specimen -> {} (the linker then degrades to session-only)"
    )
