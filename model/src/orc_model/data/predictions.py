"""Cached model predictions per clip (`detections_cache.npz`), decoded into a
`Clip`-shaped but independent view.

Deliberately not `Clip`/`Frame`: predictions have no `InstrumentAnnotation`
fields (`description` / human-assigned `track_id` / `manually_edited`) and no
pre-extracted still to `cv2.imread` — the cache only exists as a video seek
position. See https://github.com/Ethilink/MVP-OKcamera/issues/8 for the full
rationale for keeping this separate from `Clip`/`Frame`.
"""

import io
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from orc_model.data.dataset import _default_data_dir


def load_cached_detections(cache: np.lib.npyio.NpzFile) -> dict[int, sv.Detections]:
    """Frame indices absent from the returned dict had zero detections —
    treat a missing key as `sv.Detections.empty()`.

    Masks are stored bit-packed and cropped to each detection's own box (to
    keep the file small); decoding means unpacking bits and pasting them back
    into a full-frame canvas at the box's location.
    """
    frame_indices = cache["frame_indices"]
    xyxy = cache["xyxy"]
    confidence = cache["confidence"]
    class_id = cache["class_id"]
    mask_bits = cache["mask_bits"]
    mask_bit_counts = cache["mask_bit_counts"]
    image_height, image_width = int(cache["image_height"]), int(cache["image_width"])

    byte_counts = np.ceil(mask_bit_counts / 8).astype(np.int64)
    byte_offsets = np.concatenate([[0], np.cumsum(byte_counts)])

    detections_by_frame = {}
    for frame_index in np.unique(frame_indices):
        det_indices = np.where(frame_indices == frame_index)[0]
        masks = np.zeros((len(det_indices), image_height, image_width), dtype=bool)
        for out_i, det_i in enumerate(det_indices):
            x1, y1, x2, y2 = xyxy[det_i]
            x1 = min(max(int(np.floor(x1)), 0), image_width)
            y1 = min(max(int(np.floor(y1)), 0), image_height)
            x2 = min(max(int(np.ceil(x2)), x1), image_width)
            y2 = min(max(int(np.ceil(y2)), y1), image_height)
            n_bits = int(mask_bit_counts[det_i])
            packed = mask_bits[byte_offsets[det_i] : byte_offsets[det_i + 1]]
            cropped = np.unpackbits(packed, count=n_bits).astype(bool).reshape(y2 - y1, x2 - x1)
            masks[out_i, y1:y2, x1:x2] = cropped

        detections_by_frame[int(frame_index)] = sv.Detections(
            xyxy=xyxy[det_indices],
            mask=masks,
            confidence=confidence[det_indices],
            class_id=class_id[det_indices],
        )
    return detections_by_frame


class PredictedFrame:
    """Read-only view of one cached-prediction frame — mirrors `Frame`'s
    display/detections interface (`.to_detections()`, `.image`, auto-render),
    backed by the decoded cache + a video seek instead of COCO annotations +
    a pre-extracted still."""

    def __init__(self, frame_index: int, detections: sv.Detections, clip: "PredictedClip") -> None:
        self.frame_index = frame_index
        self.detections = detections
        self._clip = clip

    def load_image(self) -> np.ndarray:
        """Lazy image load, via a video seek — not an eagerly-loaded field."""
        return self._clip._read_frame(self.frame_index)

    @property
    def image(self) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(self.load_image(), cv2.COLOR_BGR2RGB))

    def _repr_png_(self) -> bytes:
        """IPython/Jupyter rich-display hook, same as `Frame`'s."""
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        return buffer.getvalue()

    def to_detections(self) -> sv.Detections:
        return self.detections


class PredictedClip:
    """Same shape as `Clip` (indexable, `.to_detections()` per frame), backed
    by a cached-detections `.npz` + video seek instead of COCO annotations +
    extracted stills.

    Indexed by the cache's own `frame_index` (native video decode position),
    NOT by list position like `Clip` — those aren't the same set of frames
    (the cache covers ~every decoded frame; `Clip` only the sparse
    hand-annotated subset), so keeping the indexing distinct is the point,
    not an oversight.
    """

    def __init__(self, name: str, video_path: Path, detections_by_frame: dict[int, sv.Detections]) -> None:
        self.name = name
        self.video_path = video_path
        self.detections_by_frame = detections_by_frame
        self.frame_indices = sorted(detections_by_frame)
        self._capture: cv2.VideoCapture | None = None

    @classmethod
    def from_cache(cls, clip_name: str, data_dir: Path | str | None = None) -> "PredictedClip":
        if data_dir is None:
            data_dir = _default_data_dir()
        data_dir = Path(data_dir)
        clip_dir = data_dir / clip_name

        cache = np.load(clip_dir / "detections_cache.npz")
        detections_by_frame = load_cached_detections(cache)

        video_paths = list((clip_dir / "video").glob("*"))
        if len(video_paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one video file in {clip_dir / 'video'}, found {len(video_paths)}"
            )

        return cls(name=clip_name, video_path=video_paths[0], detections_by_frame=detections_by_frame)

    def _read_frame(self, frame_index: int) -> np.ndarray:
        if self._capture is None:
            self._capture = cv2.VideoCapture(str(self.video_path))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._capture.read()
        assert ok, f"couldn't read frame {frame_index} from {self.video_path}"
        return frame

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __iter__(self):
        return (self[i] for i in self.frame_indices)

    def __getitem__(self, frame_index: int) -> PredictedFrame:
        if frame_index not in self.detections_by_frame:
            raise KeyError(
                f"No cached detections for frame {frame_index} in {self.name!r} "
                "(indexed by native video frame position — see `.frame_indices` "
                "for which frames have cached detections)."
            )
        return PredictedFrame(frame_index, self.detections_by_frame[frame_index], self)
