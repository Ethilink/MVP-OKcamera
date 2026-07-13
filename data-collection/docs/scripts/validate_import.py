#!/usr/bin/env python3
"""Validate a pre-annotated image dataset against the dashboard import format.

Checks a folder laid out per docs/IMPORT_FORMAT.md before it is dropped into
data/processed/ — so ID mismatches, missing images, bad bboxes, and mask/size
disagreements are caught up front instead of surfacing as broken editing.

Usage:
    python annotation_tool/scripts/validate_import.py <dataset_folder>

Exit code 0 = valid (warnings allowed), 1 = errors found, 2 = bad invocation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _load_dims(path: Path) -> tuple[int, int] | None:
    """Return (width, height) of an image, or None if unreadable."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        return None


def validate(root: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for the dataset folder at ``root``."""
    errors: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        return ([f"not a directory: {root}"], [])

    if root.name.startswith("."):
        errors.append(f"dataset name must not start with '.': {root.name!r}")
    if "/" in root.name or "\\" in root.name:
        errors.append(f"dataset name must be a single path component: {root.name!r}")

    images_dir = root / "images"
    ann_path = root / "annotations" / "annotations.json"

    if not images_dir.is_dir():
        errors.append(f"missing images/ directory: {images_dir}")
    if not ann_path.is_file():
        errors.append(f"missing annotations/annotations.json: {ann_path}")
    if errors:
        return (errors, warnings)

    # Flat images/ only; build the set of available basenames.
    on_disk: dict[str, Path] = {}
    for p in images_dir.iterdir():
        if p.is_dir():
            warnings.append(f"images/ should be flat; ignoring subfolder: {p.name}")
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in _IMAGE_EXTS:
            warnings.append(f"non-image file in images/: {p.name}")
            continue
        on_disk[p.name] = p

    try:
        data = json.loads(ann_path.read_text())
    except Exception as exc:
        return (errors + [f"annotations.json is not valid JSON: {exc}"], warnings)

    for key in ("categories", "images", "annotations"):
        if key not in data:
            errors.append(f"annotations.json missing top-level key: {key!r}")
    if errors:
        return (errors, warnings)

    # ---- categories -------------------------------------------------------
    cat_ids: set[int] = set()
    for i, c in enumerate(data["categories"]):
        cid = c.get("id")
        if cid is None:
            errors.append(f"categories[{i}] missing 'id'")
        elif cid in cat_ids:
            errors.append(f"duplicate category id: {cid}")
        else:
            cat_ids.add(cid)
        if not c.get("name"):
            warnings.append(f"categories[{i}] has empty 'name'")

    # ---- images -----------------------------------------------------------
    image_ids: set[int] = set()
    dims_by_id: dict[int, tuple[int, int]] = {}       # id -> (w, h) declared
    referenced_files: set[str] = set()
    for i, img in enumerate(data["images"]):
        iid = img.get("id")
        if iid is None:
            errors.append(f"images[{i}] missing 'id'")
        elif iid in image_ids:
            errors.append(f"duplicate image id: {iid}")
        else:
            image_ids.add(iid)

        fn = img.get("file_name")
        if not fn:
            errors.append(f"images[{i}] (id={iid}) missing 'file_name'")
        else:
            if "/" in fn or "\\" in fn:
                errors.append(f"images[{i}] file_name must be a basename: {fn!r}")
            referenced_files.add(fn)
            if fn not in on_disk:
                errors.append(f"image referenced but not on disk: images/{fn}")

        w, h = img.get("width"), img.get("height")
        if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
            errors.append(f"images[{i}] (id={iid}) needs positive int width/height")
        elif iid is not None:
            dims_by_id[iid] = (w, h)
            # Cross-check against the actual JPEG when Pillow is available.
            if fn in on_disk:
                actual = _load_dims(on_disk[fn])
                if actual is not None and actual != (w, h):
                    errors.append(
                        f"images[{i}] ({fn}) declares {w}x{h} but file is "
                        f"{actual[0]}x{actual[1]} — fix dims / strip EXIF rotation"
                    )

        rs = img.get("review_status", "pending")
        if rs == "reviewed":
            warnings.append(
                f"images[{i}] ({fn}) is pre-marked 'reviewed' — machine output "
                f"should start as 'pending' so a human reviews it"
            )

    # Orphan files on disk not referenced by the JSON.
    for name in sorted(set(on_disk) - referenced_files):
        warnings.append(f"image on disk not referenced by annotations.json: {name}")

    # ---- annotations ------------------------------------------------------
    ann_ids: set[int] = set()
    n_with_mask = 0
    for i, a in enumerate(data["annotations"]):
        aid = a.get("id")
        if aid is None:
            errors.append(f"annotations[{i}] missing 'id'")
        elif aid in ann_ids:
            errors.append(f"duplicate annotation id: {aid}")
        else:
            ann_ids.add(aid)

        img_id = a.get("image_id")
        if img_id not in image_ids:
            errors.append(f"annotations[{i}] (id={aid}) image_id={img_id} has no matching image")

        if a.get("category_id") not in cat_ids:
            errors.append(f"annotations[{i}] (id={aid}) category_id has no matching category")

        bbox = a.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(isinstance(v, (int, float)) for v in bbox)):
            errors.append(f"annotations[{i}] (id={aid}) bbox must be [x, y, w, h] numbers")
        elif bbox[2] <= 0 or bbox[3] <= 0:
            errors.append(f"annotations[{i}] (id={aid}) bbox has non-positive width/height")

        seg = a.get("segmentation")
        if seg is None:
            warnings.append(
                f"annotations[{i}] (id={aid}) has no segmentation — box-only "
                f"(dashboard can fill masks with SAM3 later)"
            )
        elif isinstance(seg, dict):
            n_with_mask += 1
            size = seg.get("size")
            counts = seg.get("counts")
            if (not isinstance(size, list) or len(size) != 2
                    or not all(isinstance(v, int) for v in size)):
                errors.append(f"annotations[{i}] (id={aid}) RLE 'size' must be [height, width] ints")
            elif img_id in dims_by_id:
                w, h = dims_by_id[img_id]
                if size != [h, w]:
                    errors.append(
                        f"annotations[{i}] (id={aid}) RLE size {size} != image "
                        f"[height,width] [{h},{w}]"
                    )
            if not isinstance(counts, (str, bytes)):
                errors.append(f"annotations[{i}] (id={aid}) RLE 'counts' must be a string")
        elif isinstance(seg, list):
            n_with_mask += 1  # polygon — accepted, converted on load
        else:
            errors.append(f"annotations[{i}] (id={aid}) segmentation must be an RLE dict or polygon list")

    warnings.append(
        f"summary: {len(image_ids)} images, {len(ann_ids)} annotations "
        f"({n_with_mask} with masks), {len(cat_ids)} categories"
    )
    return (errors, warnings)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1]).resolve()
    errors, warnings = validate(root)

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    if errors:
        print(f"\n✗ {len(errors)} error(s) — not import-ready.")
        return 1
    print(f"\n✓ Valid. {root.name} is import-ready — copy it into data/processed/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
