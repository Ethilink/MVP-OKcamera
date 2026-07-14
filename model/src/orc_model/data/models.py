"""Plain-dataclass ground-truth models for annotated surgical-instrument clips.

Loads COCO-format annotations (with a few project-specific extra fields) via
`pycocotools`, and bridges to `supervision.Detections` for visualization.
"""

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import supervision as sv
from pycocotools.coco import COCO
from PIL import Image


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class InstrumentAnnotation:
    track_id: int
    instance_id: int
    description: str
    bbox: BBox
    manually_edited: bool
    propagated_from: int | None
    segmentation: dict

    def to_mask(self) -> np.ndarray:
        """`maskUtils.decode` reads width/height off the RLE's own 'size'
        field, so no separate width/height is needed here."""
        return maskUtils.decode(self.segmentation).astype(bool)


@dataclass(frozen=True)
class Frame:
    frame_number: int
    file_path: Path
    width: int
    height: int
    annotations: list[InstrumentAnnotation]

    def load_image(self) -> np.ndarray:
        """Lazy image load — not an eagerly-loaded field."""
        return cv2.imread(str(self.file_path))

    @property
    def image(self) -> Image.Image:
        """Plain, unannotated frame image as RGB PIL image, for notebook
        display. Uncached — re-reads/reconverts on every access, which is
        fine for lightweight interactive tooling over small clips."""
        return Image.fromarray(cv2.cvtColor(self.load_image(), cv2.COLOR_BGR2RGB))

    def _repr_png_(self) -> bytes:
        """IPython/Jupyter rich-display hook: makes a bare `Frame` (e.g.
        `clip[0]`) auto-render as its image when it's a cell's output."""
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        return buffer.getvalue()

    def to_detections(self) -> sv.Detections:
        """Bridge to `supervision.Detections`, for visualization only."""
        if not self.annotations:
            return sv.Detections.empty()

        xyxy = np.array([ann.bbox.xyxy for ann in self.annotations], dtype=np.float32)
        mask = np.stack([ann.to_mask() for ann in self.annotations])
        descriptions = np.array([ann.description for ann in self.annotations])
        tracker_id = np.array([ann.track_id for ann in self.annotations])

        return sv.Detections(
            xyxy=xyxy, mask=mask, tracker_id=tracker_id, data={"description": descriptions}
        )


@dataclass(frozen=True)
class Clip:
    name: str
    video_path: Path
    fps: float
    frame_count: int
    resolution: tuple[int, int]
    frames: list[Frame]

    @classmethod
    def from_directory(cls, clip_dir: Path) -> "Clip":
        annotations_path = clip_dir / "annotations" / "annotations.json"
        # pycocotools.coco.COCO's __init__/createIndex() print progress
        # messages unconditionally (no verbosity flag exists), which spams
        # notebook cell output when loading multiple clips. Suppress just
        # this construction call, not the whole method.
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(annotations_path))

        assert len(coco.getCatIds()) == 1, "expected exactly one category"

        video_meta = coco.dataset["video"]

        image_ids = coco.getImgIds()
        image_infos = coco.loadImgs(image_ids)
        image_infos.sort(key=lambda info: info["frame_number"])

        frames = []
        for image_info in image_infos:
            image_id = image_info["id"]
            raw_annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

            instrument_annotations = []
            for raw_ann in raw_annotations:
                attrs = raw_ann.get("attributes", {})
                bbox = raw_ann["bbox"]
                instrument_annotations.append(
                    InstrumentAnnotation(
                        track_id=raw_ann["track_id"],
                        instance_id=raw_ann["instance_id"],
                        description=raw_ann["gemini_description"],
                        bbox=BBox(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3]),
                        manually_edited=attrs.get("manually_edited", False),
                        propagated_from=attrs.get("propagated_from"),
                        segmentation=raw_ann["segmentation"],
                    )
                )

            frames.append(
                Frame(
                    frame_number=image_info["frame_number"],
                    file_path=clip_dir / "images" / image_info["file_name"],
                    width=image_info["width"],
                    height=image_info["height"],
                    annotations=instrument_annotations,
                )
            )

        return cls(
            name=clip_dir.name,
            video_path=clip_dir / "video" / video_meta["file_name"],
            fps=video_meta["fps"],
            frame_count=video_meta["frame_count"],
            resolution=(video_meta["width"], video_meta["height"]),
            frames=frames,
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Frame:
        return self.frames[index]
