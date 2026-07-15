#!/usr/bin/env python
"""
FROZEN — data loader. NEVER edit from a method/ variant.

Reads (read-only, never written to):
  model/data/instruments/instrumentN/         — 8 demo instruments x ~15 frames
  model/data/other_objects/                   — foreign-object negatives
  matching/data/testing/09-07-26-00{1..4}/    — videos + frames (available, see
                                                 `load_testing_video_index` note)

Hands the method RAW bbox-cropped pixels + a RAW binary mask (same H,W) — never
a pre-masked crop. Masking strategy (fill/dilate/canonicalize) is a method-level
choice per program.md §6 ("embedding-space tricks").
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

REPO = Path(__file__).resolve().parents[3]  # .../MVP-OKcamera
INSTRUMENTS_DIR = REPO / "model" / "data" / "instruments"
FOREIGN_DIR = REPO / "model" / "data" / "other_objects"
TESTING_DIR = REPO / "matching" / "data" / "testing"


def _parse_field(v):
    """bbox / segmentation are stored as python-repr strings in these files."""
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return ast.literal_eval(v)


def _decode_rle(seg):
    seg = _parse_field(seg)
    counts = seg["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return coco_mask.decode({"size": seg["size"], "counts": counts})


@dataclass(frozen=True)
class RawItem:
    """One detector-style crop: bbox-cropped RGB pixels + aligned binary mask.
    `identity` is the enrolled instrument name for instrument data, None for
    foreign objects (there is no identity to leak — they're never a candidate).
    `group_id` is the FRAME-level grouping key: a CV/holdout split must never
    put two crops from the same `group_id` on both sides (guards against a
    future multi-crop-per-frame dataset leaking a frame across gallery/query
    even though today's data is 1 crop/frame, so groups are already singletons).
    `order` is a stable within-identity ordering (by file name) used to build
    return-window chunks for multi-frame voting.
    """
    image: np.ndarray   # (H, W, 3) uint8, bbox-cropped, UNMASKED
    mask: np.ndarray     # (H, W) uint8 {0,1}, aligned to `image`
    identity: str | None
    group_id: str         # source frame file name (unique per source image)
    item_id: str           # unique per crop (group_id, or group_id#ann_id if >1/frame)
    order: int               # stable sort key within an identity/source pool


def _load_coco_crops(root: Path, identity: str | None) -> list[RawItem]:
    coco = json.loads((root / "annotations" / "annotations.json").read_text())
    fn_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    img_cache: dict[str, np.ndarray] = {}
    # stable file-name order -> stable `order` index, robust to annotation order
    file_order = {fn: i for i, fn in enumerate(sorted(set(fn_by_id.values())))}
    per_file_seen: dict[str, int] = {}
    items: list[RawItem] = []
    for a in sorted(coco["annotations"], key=lambda a: (fn_by_id[a["image_id"]], a.get("id", 0))):
        fn = fn_by_id[a["image_id"]]
        if fn not in img_cache:
            img_cache[fn] = np.array(Image.open(root / "images" / fn).convert("RGB"))
        m = _decode_rle(a["segmentation"])
        x, y, w, h = [int(v) for v in _parse_field(a["bbox"])]
        if w < 4 or h < 4:
            continue
        img_crop = img_cache[fn][y:y + h, x:x + w]
        mask_crop = m[y:y + h, x:x + w]
        seen = per_file_seen.get(fn, 0)
        per_file_seen[fn] = seen + 1
        item_id = fn if seen == 0 else f"{fn}#{seen}"
        items.append(RawItem(
            image=img_crop, mask=mask_crop, identity=identity,
            group_id=fn, item_id=item_id, order=file_order[fn],
        ))
    return items


def load_instruments() -> list[RawItem]:
    """All 8 instruments x ~15 frames. One RawItem per detector annotation."""
    dirs = sorted(
        [d for d in INSTRUMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("instrument")],
        key=lambda p: int("".join(c for c in p.name if c.isdigit())),
    )
    items: list[RawItem] = []
    for d in dirs:
        items.extend(_load_coco_crops(d, identity=d.name))
    return items


def load_foreign() -> list[RawItem]:
    """model/data/other_objects — adversarially tool-like negatives (stage-1)."""
    return _load_coco_crops(FOREIGN_DIR, identity=None)


def instrument_names() -> list[str]:
    dirs = sorted(
        [d for d in INSTRUMENTS_DIR.iterdir() if d.is_dir() and d.name.startswith("instrument")],
        key=lambda p: int("".join(c for c in p.name if c.isdigit())),
    )
    return [d.name for d in dirs]


def testing_video_ids() -> list[str]:
    """matching/data/testing/09-07-26-00{1..4} — video ids available read-only.

    NOTE (scoping decision, see TRIED.md): these videos carry detection
    annotations but no per-instrument RETURN ground truth (no track-identity
    labels tying a video detection back to one of the 8 demo instruments), so
    they cannot drive a *labelled* multi-frame-voting re-ID metric today. This
    loader exposes them read-only for a future challenger/T04 harness; the
    champion baseline's window-voting is evaluated on windowed chunks of the
    labelled instrument crops instead (see frozen/cv_split.py).
    """
    if not TESTING_DIR.exists():
        return []
    return sorted(p.name for p in TESTING_DIR.iterdir() if p.is_dir())
