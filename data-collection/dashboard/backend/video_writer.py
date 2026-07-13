"""VideoEntryWriter — turns a finished recording (MP4 already on disk) + the
keyframes marked live during the take into the three reviewed video-project
artifacts (IMPORT_FORMAT_VIDEO.md §§2-4): annotations.json (COCO-VID, keyframes
only), one keyframe JPEG per mark, and selected_frames.json.

Since the 2026-07-09 simplification (ADR-0002) there is no all-frames post-pass
and no ``full_frame_detections.json`` sidecar — each keyframe carries the live
detection captured at SPACE-press time (frame pre-encoded to JPEG bytes,
detections, and the live threshold), so this writer never decodes the MP4 and
never runs a detector. The caller feeds keyframes one at a time via
``add_keyframe`` in increasing ``frame_number`` order; not thread-safe (the
caller serializes with its own lock).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend import coco

# annotations.json / selected_frames.json must be reproducible byte-for-byte
# from the same keyframe inputs (idempotent rebuild), so no wall-clock timestamp
# may leak into either file.
_FIXED_TIMESTAMP = "1970-01-01T00:00:00"


def _atomic_write_json(path: Path, document: dict) -> None:
    """Write ``document`` as JSON via a temp file + ``os.replace`` so the target
    is never a torn/partial file. This is what keeps the partial-entry contract
    honest: ``annotations.json`` is either fully present or absent — a crash
    mid-``json.dump`` can only leave the ``.tmp``, never a half-written project
    the discovery rule would try to open (mirrors ``DatasetWriter``)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    os.replace(tmp, path)


class VideoEntryWriter:
    def __init__(
        self,
        entry_dir: Path,
        entry_name: str,
        *,
        video: dict,
        keyframes,
    ):
        self._entry_dir = Path(entry_dir)
        self._entry_name = entry_name
        self._video = video
        self._keyframes = sorted(set(keyframes))

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
        _atomic_write_json(metadata_dir / "selected_frames.json", document)

    def add_keyframe(self, frame_number: int, jpeg: bytes, dets: "sv.Detections", threshold: float) -> None:
        """Write one keyframe's JPEG (the pre-encoded bytes captured live at
        SPACE-press time) and accumulate its image record + annotations, filtered
        at ``threshold`` — the live per-frame snapshot value at press time
        (image-mode's exact provenance rule), not a take-wide operator threshold.
        """
        images_dir = self._entry_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{self._entry_name}_f{frame_number:06d}.jpg"
        with open(images_dir / file_name, "wb") as f:
            f.write(jpeg)

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

        keyframe_dets = dets[dets.confidence >= threshold]
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
        # required on the annotations.json video block). Consumers fall back to
        # this when an image record lacks its own file_name — omitting it KeyErrors
        # that path in the annotation tool.
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
        _atomic_write_json(annotations_dir / "annotations.json", document)

    @property
    def entry_dir(self) -> Path:
        return self._entry_dir
