"""Fixed scoring/logging logic for the tracker experimentation harness.

Owns all scoring and logging. Must never be edited by the experimentation
loop (`experiment.py`) — only `experiment.py`'s config/tracker-construction
logic is meant to change between runs.

Ports the checkpoint-based switch/miss/merge methodology proven out on
synthetic data in `playground/trackers/experiments/evals.ipynb` onto real
annotated clips: every annotated frame in a clip is a GT checkpoint (a
cumulative snapshot of every instrument added so far), and a tracker's live
output is IoU-matched against those checkpoints to count identity switches,
misses, and merges.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import supervision as sv
from scipy.optimize import linear_sum_assignment
from supervision.detection.utils.boxes import box_iou_batch

from orc_model.data.models import Clip

IOU_MATCH_THRESHOLD = 0.3

TRAIN_CLIPS = ["IMG_2044", "IMG_2047", "IMG_2076", "IMG_2081", "IMG_2104"]
HOLDOUT_CLIP = "IMG_2112"


def merged_frame_indices(
    native_fps: float, frame_count: int, target_fps: float, required_frames: list[int]
) -> list[int]:
    """Evenly-spaced indices approximating `target_fps`, plus every frame in
    `required_frames` merged in so GT checkpoints are never missed."""
    step = max(round(native_fps / target_fps), 1)
    sampled = range(0, frame_count, step)
    return sorted(set(sampled) | set(required_frames))


def match_gt_to_predictions(
    gt_items: list[tuple[int, np.ndarray]], tracked: sv.Detections
) -> dict[int, int | None]:
    """Returns {gt_track_id: matched_tracker_id_or_None} for one checkpoint."""
    gt_ids = [track_id for track_id, _ in gt_items]
    matches: dict[int, int | None] = dict.fromkeys(gt_ids)
    if not gt_items:
        return matches

    # tracker_id == -1 is this library's sentinel for immature/unconfirmed
    # tracks, not a real identity -- exclude before matching, otherwise two
    # different immature tracklets would look like "the same ID"
    live = tracked.tracker_id != -1
    pred_xyxy = tracked.xyxy[live]
    pred_ids = tracked.tracker_id[live]
    if len(pred_xyxy) == 0:
        return matches

    gt_xyxy = np.array([bbox for _, bbox in gt_items], dtype=np.float32)
    iou = box_iou_batch(gt_xyxy, pred_xyxy)
    gt_rows, pred_cols = linear_sum_assignment(-iou)

    for gt_row, pred_col in zip(gt_rows, pred_cols, strict=False):
        if iou[gt_row, pred_col] >= IOU_MATCH_THRESHOLD:
            matches[gt_ids[gt_row]] = int(pred_ids[pred_col])
    return matches


def per_track_sequences(
    checkpoint_matches: dict[int, dict[int, int | None]],
) -> dict[int, list[tuple[int, int | None]]]:
    sequences: dict[int, list[tuple[int, int | None]]] = {}
    for frame_index in sorted(checkpoint_matches):
        for gt_track_id, pred_id in checkpoint_matches[frame_index].items():
            sequences.setdefault(gt_track_id, []).append((frame_index, pred_id))
    return sequences


def _count_errors(
    sequences: dict[int, list[tuple[int, int | None]]],
) -> tuple[int, int, dict[int, set[int]]]:
    """Switch/miss/merge counts for a set of per-GT-track sequences."""
    total_switches = total_misses = 0
    gt_tracks_by_pred_id: dict[int, set[int]] = {}

    for gt_track_id, points in sorted(sequences.items()):
        matched_sequence = [pred_id for _, pred_id in points if pred_id is not None]
        total_switches += sum(1 for a, b in zip(matched_sequence, matched_sequence[1:]) if a != b)
        total_misses += sum(1 for _, pred_id in points if pred_id is None)
        for pred_id in matched_sequence:
            gt_tracks_by_pred_id.setdefault(pred_id, set()).add(gt_track_id)

    merges = {pred_id: gt_ids for pred_id, gt_ids in gt_tracks_by_pred_id.items() if len(gt_ids) > 1}
    return total_switches, total_misses, merges


def checkpoint_frames(clip: Clip) -> list[int]:
    return sorted(frame.frame_number for frame in clip.frames)


def gt_checkpoint_boxes(clip: Clip, frame_number: int) -> list[tuple[int, np.ndarray]]:
    frame = next(f for f in clip.frames if f.frame_number == frame_number)
    return [(ann.track_id, np.array(ann.bbox.xyxy, dtype=np.float32)) for ann in frame.annotations]


@dataclass(frozen=True)
class ClipScore:
    clip_name: str
    total_checkpoints: int
    switches: int
    misses: int
    merges: dict[int, set[int]]

    @property
    def errors(self) -> int:
        return self.switches + self.misses + sum(len(gt_ids) - 1 for gt_ids in self.merges.values())

    @property
    def score(self) -> float:
        if self.total_checkpoints > 0:
            return self.errors / self.total_checkpoints
        return float("nan")


def score_clip(clip: Clip, tracked_by_frame: dict[int, sv.Detections]) -> ClipScore:
    checkpoint_matches: dict[int, dict[int, int | None]] = {}
    for frame_number in checkpoint_frames(clip):
        checkpoint_matches[frame_number] = match_gt_to_predictions(
            gt_checkpoint_boxes(clip, frame_number), tracked_by_frame[frame_number]
        )

    sequences = per_track_sequences(checkpoint_matches)
    switches, misses, merges = _count_errors(sequences)
    total_checkpoints = sum(len(points) for points in sequences.values())

    return ClipScore(
        clip_name=clip.name,
        total_checkpoints=total_checkpoints,
        switches=switches,
        misses=misses,
        merges=merges,
    )


def aggregate_score(clip_scores: dict[str, ClipScore]) -> float:
    """Pooled error rate across all given clips, not a mean of per-clip rates."""
    total_errors = sum(cs.errors for cs in clip_scores.values())
    total_checkpoints = sum(cs.total_checkpoints for cs in clip_scores.values())
    return total_errors / total_checkpoints


def log_run(
    config: dict,
    clip_scores: dict[str, ClipScore],
    notes: str,
    log_path: Path | None = None,
) -> None:
    if log_path is None:
        log_path = Path(__file__).parent / "runs.jsonl"

    per_clip = {
        name: {
            "score": cs.score,
            "switches": cs.switches,
            "misses": cs.misses,
            "merges": {str(pred_id): sorted(gt_ids) for pred_id, gt_ids in cs.merges.items()},
            "total_checkpoints": cs.total_checkpoints,
        }
        for name, cs in clip_scores.items()
    }

    train_scores = {name: cs for name, cs in clip_scores.items() if name in TRAIN_CLIPS}
    holdout_score = clip_scores[HOLDOUT_CLIP].score if HOLDOUT_CLIP in clip_scores else None

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config": config,
        "clips": per_clip,
        "aggregate_score": aggregate_score(train_scores) if train_scores else None,
        "holdout_score": holdout_score,
        "notes": notes,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    # equivalent of the notebook's hand-crafted `example_sequences`:
    #   GT track 1: matched to pred id 10, then switches to 99
    #   GT track 2: missed at its last checkpoint
    #   GT track 3: also matched to pred id 99 -> merge with track 1
    checkpoint_matches: dict[int, dict[int, int | None]] = {
        0: {1: 10},
        60: {1: 10, 2: 20},
        150: {1: 99, 2: 20, 3: 99},
        250: {1: 99, 2: None},
    }

    sequences = per_track_sequences(checkpoint_matches)
    switches, misses, merges = _count_errors(sequences)

    assert switches == 1, f"expected 1 switch, got {switches}"
    assert misses == 1, f"expected 1 miss, got {misses}"
    assert merges == {99: {1, 3}}, f"expected merge {{99: {{1, 3}}}}, got {merges}"

    print("eval.py self-test passed: 1 switch, 1 miss, merge {99: {1, 3}}")
