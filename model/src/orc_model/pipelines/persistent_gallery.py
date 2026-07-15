"""Persistent specimen galleries — pre-captured photo sets embedded once.

Each physical demo instrument was photographed from ~15 angles and annotated
(`model/data/instruments/instrument{N}/`). `load_persistent_galleries()` turns
each folder into ONE embedded `ChampionGallery`, keyed by its specimen number,
so the session linker can bind an enrolled identity to its real specimen at the
enrolment freeze (`linker-design.md` §3, wayfinder T08).

Embedding happens HERE, exactly once per process — `load_tracker()` calls this
eagerly at startup and hands the result to `SessionLinker`, which then only ever
reads the cached vectors. Re-embedding 15 views per identity per event would
grow the enrolment freeze from ~260 ms toward seconds.

Nothing in here raises on bad data: an absent directory, an unparseable
annotation file, an unreadable image or a folder that isn't `instrument{N}` is
skipped with a warning, and a run with no usable specimens returns `{}` — the
linker then degrades to session-only linking.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

if TYPE_CHECKING:
    from orc_model.pipelines.matching import ChampionGallery

DEFAULT_INSTRUMENTS_DIR = Path(__file__).resolve().parents[3] / "data" / "instruments"

_SPECIMEN_DIR_RE = re.compile(r"^instrument(\d+)$")
_MIN_BBOX_PX = 4

_log = logging.getLogger("orc_model.persistent_gallery")


def load_persistent_galleries(
    matcher,
    instruments_dir: str | Path | None = None,
) -> dict[int, ChampionGallery]:
    """specimen_number -> embedded gallery. {} when nothing usable is found."""
    root = Path(instruments_dir) if instruments_dir is not None else DEFAULT_INSTRUMENTS_DIR
    if not root.is_dir():
        _log.warning("persistent galleries: %s is not a directory -- none loaded", root)
        return {}

    galleries: dict[int, ChampionGallery] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        match = _SPECIMEN_DIR_RE.match(folder.name)
        if match is None or int(match.group(1)) < 1:
            _log.warning(
                "persistent galleries: skipping %s -- not an instrument{N} folder", folder.name
            )
            continue
        try:
            crops, masks = _load_coco_crops(folder)
        except Exception:
            _log.warning(
                "persistent galleries: skipping %s -- unreadable annotations", folder.name,
                exc_info=True,
            )
            continue
        if not crops:
            _log.warning(
                "persistent galleries: skipping %s -- no usable annotated crops", folder.name
            )
            continue
        specimen = int(match.group(1))
        galleries[specimen] = matcher.build_gallery(crops, masks, {"identity": specimen})

    _log.info(
        "persistent galleries: %d specimen(s) loaded from %s: %s",
        len(galleries), root,
        {specimen: int(g.views.shape[0]) for specimen, g in sorted(galleries.items())},
    )
    return galleries


# -- COCO loading ---------------------------------------------------------
# Logic vendored from experiments/matcher-autoresearch/frozen/loader.py
# (`_parse_field` / `_decode_rle` / `_load_coco_crops`), which is a research
# workspace and not an installable package. Kept behaviourally identical so
# production embeds exactly the pixels the champion was benchmarked on.


def _parse_field(v):
    """bbox / segmentation are real JSON in some files, python-repr strings in
    others."""
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return ast.literal_eval(v)


def _decode_rle(seg) -> np.ndarray:
    seg = _parse_field(seg)
    counts = seg["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return coco_mask.decode({"size": seg["size"], "counts": counts})


def _load_coco_crops(root: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One bbox-cropped RGB image + aligned binary mask per usable annotation."""
    coco = json.loads((root / "annotations" / "annotations.json").read_text())
    fn_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    img_cache: dict[str, np.ndarray | None] = {}
    crops: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for a in sorted(coco["annotations"], key=lambda a: (fn_by_id[a["image_id"]], a.get("id", 0))):
        fn = fn_by_id[a["image_id"]]
        if fn not in img_cache:
            img_cache[fn] = _read_rgb(root / "images" / fn)
        image = img_cache[fn]
        if image is None:
            continue
        x, y, w, h = (int(v) for v in _parse_field(a["bbox"]))
        if w < _MIN_BBOX_PX or h < _MIN_BBOX_PX:
            continue
        mask = _decode_rle(a["segmentation"])
        crops.append(image[y : y + h, x : x + w])
        masks.append(mask[y : y + h, x : x + w])
    return crops, masks


def _read_rgb(path: Path) -> np.ndarray | None:
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        _log.warning("persistent galleries: skipping unreadable image %s", path)
        return None
