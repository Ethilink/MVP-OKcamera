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

from backend import coco

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
            # exist_ok so a flag that raises AFTER these dirs are made (e.g. a
            # failed imwrite below) leaves the writer retryable — n_flagged stays
            # 0, so the next flag re-enters this block and must not trip on the
            # dirs already present.
            (self._dataset_dir / "images").mkdir(parents=True, exist_ok=True)
            (self._dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)

        image_id = len(self.images) + 1
        file_name = f"frame_{image_id:05d}.jpg"

        W, H = int(frame.shape[1]), int(frame.shape[0])
        # imwrite returns False on failure (bad path, full disk, unwritable dir)
        # instead of raising. Unchecked, the image record + counters below would
        # still advance and /flag would report success referencing a file that
        # isn't on disk. Fail loudly before mutating any writer state.
        if not cv2.imwrite(str(self._dataset_dir / "images" / file_name), frame):
            raise OSError(
                f"failed to write image {file_name} to {self._dataset_dir / 'images'}"
            )

        n_annotations_before = len(self.annotations)
        if len(dets):
            # dets share the saved frame's resolution — write boxes and masks in
            # the frame's own pixel coordinates, no scaling. Boxes still get
            # clipped: the detector emits boxes that can run off-frame (no clamp
            # in decode_predictions) and the import validator only checks w>0/h>0.
            for i in range(len(dets)):
                ann = coco.build_annotation(
                    dets,
                    i,
                    W,
                    H,
                    ann_id=len(self.annotations) + 1,
                    image_id=image_id,
                    extra={
                        "confidence": float(dets.confidence[i]),
                        "model_version": self.model_version,
                        "confidence_threshold": threshold,
                    },
                )
                if ann is None:
                    continue

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

    def discard_last(self) -> int:
        """Undo the most recent flag: drop its image record, its annotations, and
        its JPEG, then rewrite ``annotations.json``. Returns the removed image id.

        Powers the capture-preview "Discard" affordance. Safe to interleave with
        ``flag`` only under the caller's ``dataset_lock`` (this class is not
        thread-safe by itself). Because ``image_id``/``file_name`` derive from
        ``len(self.images)``, removing the tail makes the next flag cleanly reuse
        the freed id — no gap, no duplicate. Raises ``IndexError`` if empty.
        """
        if not self.images:
            raise IndexError("nothing to discard")

        removed = self.images.pop()
        removed_id = removed["id"]
        # The last flag's annotations are exactly those referencing its image_id
        # (annotations are appended per flag, so they are the contiguous tail).
        self.annotations = [a for a in self.annotations if a["image_id"] != removed_id]
        (self._dataset_dir / "images" / removed["file_name"]).unlink(missing_ok=True)
        self._n_flagged -= 1

        self._write_annotations()
        return removed_id

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
