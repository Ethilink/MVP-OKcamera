"""Pure, camera-free, FastAPI-free writer that turns one detection snapshot
into a COCO-VID dataset increment on disk.

``dets`` and the saved ``frame`` share one resolution — the detector ran on the
same frame we save — so boxes and masks are written in the frame's own pixel
coordinates with no scaling. Not thread-safe by design — the caller serializes
concurrent flags with its own lock.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from pycocotools import mask as mask_utils

_CATEGORIES = [{"id": 1, "name": "surgical_instrument", "supercategory": ""}]


@dataclass
class FlagResult:
    image_id: int
    n_annotations: int
    file_name: str


class DatasetWriter:
    def __init__(self, output_path: Path, dataset_name: str, model_version: str):
        if not dataset_name or "/" in dataset_name or "\\" in dataset_name or dataset_name.startswith("."):
            raise ValueError(f"invalid dataset_name: {dataset_name!r}")

        dataset_dir = Path(output_path) / dataset_name
        if dataset_dir.exists():
            raise FileExistsError(f"dataset already exists: {dataset_dir}")

        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        self.model_version = model_version
        self.images: list = []
        self.annotations: list = []
        self._n_flagged = 0
        self._date_created = datetime.now().isoformat()
        self._dataset_dir = dataset_dir

    @property
    def n_flagged(self) -> int:
        return self._n_flagged

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    def flag(self, frame: np.ndarray, dets: sv.Detections, threshold: float) -> FlagResult:
        if self.n_flagged == 0:
            (self._dataset_dir / "images").mkdir(parents=True)
            (self._dataset_dir / "annotations").mkdir(parents=True)

        image_id = len(self.images) + 1
        file_name = f"frame_{image_id:05d}.jpg"

        W, H = int(frame.shape[1]), int(frame.shape[0])
        cv2.imwrite(str(self._dataset_dir / "images" / file_name), frame)

        n_annotations_before = len(self.annotations)
        if len(dets):
            # dets share the saved frame's resolution — write boxes and masks in
            # the frame's own pixel coordinates, no scaling. Boxes still get
            # clipped: the detector emits boxes that can run off-frame (no clamp
            # in decode_predictions) and the import validator only checks w>0/h>0.
            for i in range(len(dets)):
                x1, y1, x2, y2 = dets.xyxy[i]
                x1 = min(max(float(x1), 0.0), W)
                x2 = min(max(float(x2), 0.0), W)
                y1 = min(max(float(y1), 0.0), H)
                y2 = min(max(float(y2), 0.0), H)
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                bbox = [x1, y1, bw, bh]

                rle = mask_utils.encode(np.asfortranarray(dets.mask[i].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                mask_area = int(mask_utils.area(rle))

                ann_id = len(self.annotations) + 1
                ann = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "confidence": float(dets.confidence[i]),
                    "model_version": self.model_version,
                    "confidence_threshold": threshold,
                }
                if mask_area > 0:
                    ann["segmentation"] = {"size": rle["size"], "counts": rle["counts"]}
                    ann["area"] = mask_area
                else:
                    ann["area"] = int(bw * bh)

                self.annotations.append(ann)

        self.images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": W,
                "height": H,
                "review_status": "pending",
            }
        )
        self._n_flagged += 1

        self._write_annotations()

        return FlagResult(
            image_id=image_id,
            n_annotations=len(self.annotations) - n_annotations_before,
            file_name=file_name,
        )

    def _write_annotations(self) -> None:
        document = {
            "info": {"description": self.dataset_name, "date_created": self._date_created},
            "categories": _CATEGORIES,
            "images": self.images,
            "annotations": self.annotations,
        }
        annotations_dir = self._dataset_dir / "annotations"
        tmp_path = annotations_dir / "annotations.json.tmp"
        final_path = annotations_dir / "annotations.json"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(document, f)
            os.replace(tmp_path, final_path)
        except BaseException:
            # A failed write must not leave the previous good file torn (os.replace
            # is atomic) nor a stale .tmp behind.
            tmp_path.unlink(missing_ok=True)
            raise
