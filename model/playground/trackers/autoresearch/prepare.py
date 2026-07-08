"""One-time, expensive step: decode every native-fps frame of every clip and
run the detector once, caching detections (at CONFIDENCE_THRESHOLD) to disk
per clip. Lets the experimentation loop re-run tracking logic over cached
detections without re-running the (slow) detector on every tuning iteration.

Segmentation masks are cached too, cropped to each detection's own bounding
box and bit-packed -- a mask is all-False outside its box by construction, so
storing the full frame-sized array per detection would be almost entirely
wasted space (e.g. ~140GB for one clip at CONFIDENCE_THRESHOLD=0.0, versus a
few hundred MB cropped at 0.5).
"""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from orc_model.components.detector import Detector
from orc_model.data import ClipDataset
from orc_model.data.models import Clip

MODEL_DIR = Path(__file__).resolve().parents[3]
WEIGHTS_PATH = MODEL_DIR / "weights" / "rf-detr-seg-2xl-v1.onnx"
CONFIDENCE_THRESHOLD = 0.5


def cache_path_for(clip_name: str) -> Path:
    return MODEL_DIR / "data" / clip_name / "detections_cache.npz"


def _mask_crop_bounds(box_xyxy: np.ndarray, image_height: int, image_width: int) -> tuple[int, int, int, int]:
    """Integer pixel bounds to crop/place a detection's mask against its own
    box. Shared by build (crop) and load (reconstruct) so the two stay in
    sync without persisting the bounds separately."""
    x1, y1, x2, y2 = box_xyxy
    x1 = min(max(int(np.floor(x1)), 0), image_width)
    y1 = min(max(int(np.floor(y1)), 0), image_height)
    x2 = min(max(int(np.ceil(x2)), x1), image_width)
    y2 = min(max(int(np.ceil(y2)), y1), image_height)
    return x1, y1, x2, y2


def build_cache_for_clip(clip: Clip, detector: Detector) -> None:
    frame_indices = []
    xyxy = []
    confidence = []
    class_id = []
    mask_bits = []  # packbits of each detection's box-cropped mask, concatenated
    mask_bit_counts = []  # bit (pixel) count per detection, to unpack without padding

    cap = cv2.VideoCapture(str(clip.video_path))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.predict(frame)
        count = len(detections)
        if count:
            image_height, image_width = frame.shape[:2]
            frame_indices.append(np.full(count, index, dtype=np.int32))
            xyxy.append(detections.xyxy.astype(np.float32))
            confidence.append(detections.confidence.astype(np.float32))
            class_id.append(detections.class_id.astype(np.int32))
            for box, mask in zip(detections.xyxy, detections.mask, strict=False):
                x1, y1, x2, y2 = _mask_crop_bounds(box, image_height, image_width)
                cropped = mask[y1:y2, x1:x2]
                mask_bits.append(np.packbits(cropped))
                mask_bit_counts.append(cropped.size)
        index += 1
    cap.release()

    cache_path = cache_path_for(clip.name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        frame_indices=np.concatenate(frame_indices) if frame_indices else np.empty(0, dtype=np.int32),
        xyxy=np.concatenate(xyxy) if xyxy else np.empty((0, 4), dtype=np.float32),
        confidence=np.concatenate(confidence) if confidence else np.empty(0, dtype=np.float32),
        class_id=np.concatenate(class_id) if class_id else np.empty(0, dtype=np.int32),
        mask_bits=np.concatenate(mask_bits) if mask_bits else np.empty(0, dtype=np.uint8),
        mask_bit_counts=np.array(mask_bit_counts, dtype=np.int64),
        image_height=np.int32(clip.resolution[1]),
        image_width=np.int32(clip.resolution[0]),
    )


def load_cached_detections(clip_name: str) -> dict[int, sv.Detections]:
    """Frame indices absent from the returned dict had zero detections —
    callers should treat a missing key as `sv.Detections.empty()`."""
    cache_path = cache_path_for(clip_name)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No detections cache for clip {clip_name!r} at {cache_path}. "
            "Run prepare.py first."
        )

    data = np.load(cache_path)
    frame_indices = data["frame_indices"]
    xyxy = data["xyxy"]
    confidence = data["confidence"]
    class_id = data["class_id"]
    mask_bits = data["mask_bits"]
    mask_bit_counts = data["mask_bit_counts"]
    image_height = int(data["image_height"])
    image_width = int(data["image_width"])

    byte_counts = np.ceil(mask_bit_counts / 8).astype(np.int64)
    byte_offsets = np.concatenate([[0], np.cumsum(byte_counts)])

    detections_by_frame = {}
    for frame_index in np.unique(frame_indices):
        det_indices = np.where(frame_indices == frame_index)[0]
        masks = np.zeros((len(det_indices), image_height, image_width), dtype=bool)
        for out_i, det_i in enumerate(det_indices):
            x1, y1, x2, y2 = _mask_crop_bounds(xyxy[det_i], image_height, image_width)
            n_bits = int(mask_bit_counts[det_i])
            packed = mask_bits[byte_offsets[det_i]:byte_offsets[det_i + 1]]
            cropped = np.unpackbits(packed, count=n_bits).astype(bool).reshape(y2 - y1, x2 - x1)
            masks[out_i, y1:y2, x1:x2] = cropped

        detections_by_frame[int(frame_index)] = sv.Detections(
            xyxy=xyxy[det_indices],
            mask=masks,
            confidence=confidence[det_indices],
            class_id=class_id[det_indices],
        )
    return detections_by_frame


if __name__ == "__main__":
    dataset = ClipDataset.from_data_dir()
    detector = Detector(WEIGHTS_PATH, confidence_threshold=CONFIDENCE_THRESHOLD)

    for clip in dataset:
        cache_path = cache_path_for(clip.name)
        if cache_path.exists():
            print(f"{clip.name}: skip (already cached)")
            continue

        build_cache_for_clip(clip, detector)
        total_detections = len(np.load(cache_path)["confidence"])
        print(f"{clip.name}: cached {total_detections} detections across {clip.frame_count} frames")
