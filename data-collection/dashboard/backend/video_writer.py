"""VideoEntryWriter — turns a finished recording (MP4 already on disk) + a
keyframe list + per-frame detections into the four video-project artifacts
(IMPORT_FORMAT_VIDEO.md §§2-5): annotations.json (COCO-VID, keyframes only),
keyframe JPEGs, selected_frames.json, and the full_frame_detections.json
sidecar.

Camera-free and detector-free by design — the post-pass job decodes the MP4
and runs the detector, handing decoded frames + detections to ``add_frame``
one at a time, in increasing ``frame_number`` order. Not thread-safe — the
caller serializes calls to one writer with its own lock.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from backend import coco

# AC13 (idempotent rebuild): annotations.json and full_frame_detections.json
# must be byte-identical across two separate writer instances fed the same
# add_frame sequence, so no wall-clock timestamp may leak into either file.
_FIXED_TIMESTAMP = "1970-01-01T00:00:00"


class VideoEntryWriter:
    def __init__(
        self,
        entry_dir: Path,
        entry_name: str,
        model_version: str,
        *,
        video: dict,
        keyframes,
        operator_threshold: float,
        mining_threshold: float,
        model_name: str = "nf-detector",
    ):
        self._entry_dir = Path(entry_dir)
        self._entry_name = entry_name
        self._model_version = model_version
        self._video = video
        self._keyframes = sorted(set(keyframes))
        self._keyframe_set = set(self._keyframes)
        self._operator_threshold = operator_threshold
        self._mining_threshold = mining_threshold
        self._model_name = model_name

        self._sidecar_frames: dict = {}
        self._images: list = []
        self._annotations: list = []
        self._next_image_id = 1
        self._next_ann_id = 1
        self._next_track_id = 1

    def write_selected_frames(self) -> None:
        fps = self._video["fps"]
        metadata_dir = self._entry_dir / "annotations" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "selected_frames": self._keyframes,
            "manual_review": True,
            "timestamp": _FIXED_TIMESTAMP,
            "fps": fps,
            "selected_frames_with_time": [{"frame": f, "seconds": f / fps} for f in self._keyframes],
        }
        with open(metadata_dir / "selected_frames.json", "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)

    def add_frame(self, frame_number: int, frame: np.ndarray, dets: "sv.Detections") -> None:
        detections = []
        for i in range(len(dets)):
            x1, y1, x2, y2 = dets.xyxy[i]
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(dets.confidence[i]),
                    "label": coco.CATEGORIES[0]["name"],
                }
            )
        self._sidecar_frames[frame_number] = {"frame_number": frame_number, "detections": detections}

        if frame_number not in self._keyframe_set:
            return

        images_dir = self._entry_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{self._entry_name}_f{frame_number:06d}.jpg"
        cv2.imwrite(str(images_dir / file_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        W, H = int(self._video["width"]), int(self._video["height"])
        image_id = self._next_image_id
        self._next_image_id += 1
        self._images.append(
            {
                "id": image_id,
                "video_id": 1,
                "frame_number": frame_number,
                "file_name": file_name,
                "width": W,
                "height": H,
                "review_status": "pending",
            }
        )

        keyframe_dets = dets[dets.confidence >= self._operator_threshold]
        for i in range(len(keyframe_dets)):
            ann = coco.build_annotation(
                keyframe_dets,
                i,
                W,
                H,
                ann_id=self._next_ann_id,
                image_id=image_id,
                extra={"video_id": 1, "track_id": self._next_track_id},
            )
            if ann is None:
                continue
            self._annotations.append(ann)
            self._next_ann_id += 1
            self._next_track_id += 1

    def finalize(self) -> None:
        # `file_name` = the basename inside video/ (IMPORT_FORMAT_VIDEO.md §2,
        # required on the annotations.json video block; §5 sidecar carries a
        # copy). Consumers fall back to this when an image record lacks its own
        # file_name — omitting it KeyErrors that path in the annotation tool.
        mp4_name = f"{self._entry_name}.mp4"
        video_block = dict(self._video)
        video_block["id"] = 1
        video_block["file_name"] = mp4_name
        video_block["source_type"] = "video"

        document = {
            "info": {"description": self._entry_name, "date_created": _FIXED_TIMESTAMP},
            "categories": coco.CATEGORIES,
            "video": video_block,
            "images": sorted(self._images, key=lambda im: im["frame_number"]),
            "annotations": self._annotations,
        }
        annotations_dir = self._entry_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        with open(annotations_dir / "annotations.json", "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)

        metadata_dir = annotations_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        sidecar = {
            "schema_version": 1,
            "video": {**dict(self._video), "file_name": mp4_name},
            "model": {
                "name": self._model_name,
                "version": self._model_version,
                "conf_threshold": self._mining_threshold,
            },
            "frames": [self._sidecar_frames[fn] for fn in sorted(self._sidecar_frames)],
        }
        with open(metadata_dir / "full_frame_detections.json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)

        self.write_selected_frames()

    @property
    def entry_dir(self) -> Path:
        return self._entry_dir
