"""One-time, expensive step: decode every native-fps frame of every clip and
run the detector once, caching detections (at CONFIDENCE_THRESHOLD) to disk
per clip as `data/<clip>/detections_cache.npz`. Lets tracker experiments
(`playground/trackers/*.ipynb`, via `PredictedClip.from_cache`) re-run
tracking logic without re-running the (slow) detector on every iteration.

Segmentation masks are cached cropped to each detection's own bounding box
and bit-packed -- a mask is all-False outside its box by construction, so
storing the full frame-sized array per detection would be almost entirely
wasted space.

Usage: `python playground/build_detections_cache.py` from `model/`. Clips
that already have a cache are skipped -- delete the .npz to force a rebuild.
"""

from pathlib import Path

import cv2
import numpy as np

from orc_model.components.detector import Detector
from orc_model.data import ClipDataset
from orc_model.data.models import Clip

MODEL_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = MODEL_DIR / "weights" / "rf-detr-seg-2xl-v1.onnx"
CONFIDENCE_THRESHOLD = 0.5


def cache_path_for(clip_name: str) -> Path:
    return MODEL_DIR / "data" / clip_name / "detections_cache.npz"


def _mask_crop_bounds(box_xyxy: np.ndarray, image_height: int, image_width: int) -> tuple[int, int, int, int]:
    """Integer pixel bounds to crop a detection's mask against its own box --
    matches the floor/ceil clamping `load_cached_detections` uses to decode."""
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


def main() -> None:
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


if __name__ == "__main__":
    main()
