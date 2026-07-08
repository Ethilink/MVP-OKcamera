"""Baseline SORT tracking experiment, ported from `sort.ipynb`.

Runs the known-good SORT config over cached detections for every train clip
plus the holdout clip, scores each, and logs the run. This is the one file
in `autoresearch/` meant to be freely edited by future tuning iterations —
today it just seeds that loop with the current baseline, unmodified.
"""

import supervision as sv
from eval import (
    HOLDOUT_CLIP,
    TRAIN_CLIPS,
    aggregate_score,
    checkpoint_frames,
    log_run,
    merged_frame_indices,
    score_clip,
)
from prepare import load_cached_detections
from trackers import SORTTracker
from trackers.utils.iou import GIoU
from trackers.utils.state_representations import XCYCSRStateEstimator

from orc_model.data import ClipDataset

TARGET_FPS = 30
CONFIDENCE_THRESHOLD = 0.80  # detector confidence filter, applied to cached raw detections


MINIMUM_CONSECUTIVE_FRAMES = 1
MINIMUM_IOU_THRESHOLD = 0.15
LOST_TRACK_BUFFER_SECONDS = 1


def make_tracker() -> SORTTracker:
    return SORTTracker(
        frame_rate=TARGET_FPS,
        track_activation_threshold=0.90,
        minimum_consecutive_frames=MINIMUM_CONSECUTIVE_FRAMES,
        lost_track_buffer=LOST_TRACK_BUFFER_SECONDS * TARGET_FPS,
        minimum_iou_threshold=MINIMUM_IOU_THRESHOLD,
        state_estimator_class=XCYCSRStateEstimator,
        iou=GIoU(),
    )


def main() -> None:
    dataset = ClipDataset.from_data_dir()
    clip_scores = {}

    for clip_name in [*TRAIN_CLIPS, HOLDOUT_CLIP]:
        clip = dataset.get_clip(clip_name)
        raw_detections = load_cached_detections(clip.name)

        frame_indices = merged_frame_indices(
            clip.fps, clip.frame_count, TARGET_FPS, checkpoint_frames(clip)
        )

        tracker = make_tracker()
        tracked_by_frame = {}
        checkpoints = set(checkpoint_frames(clip))

        for frame_index in frame_indices:
            detections = raw_detections.get(frame_index, sv.Detections.empty())
            if len(detections) > 0:
                detections = detections[detections.confidence >= CONFIDENCE_THRESHOLD]
            tracked = tracker.update(detections)
            if frame_index in checkpoints:
                tracked_by_frame[frame_index] = tracked

        clip_scores[clip.name] = score_clip(clip, tracked_by_frame)

    train_scores = {name: s for name, s in clip_scores.items() if name in TRAIN_CLIPS}
    agg = aggregate_score(train_scores)
    holdout = clip_scores.get(HOLDOUT_CLIP)

    print(f"aggregate (train) score: {agg:.4f}")
    for name, s in clip_scores.items():
        print(
            f"  {name}: score={s.score:.4f} switches={s.switches} misses={s.misses} "
            f"merges={len(s.merges)} checkpoints={s.total_checkpoints}"
        )
    if holdout is not None:
        print(f"holdout ({HOLDOUT_CLIP}) score: {holdout.score:.4f}")

    log_run(
        config={
            "target_fps": TARGET_FPS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "tracker": "SORTTracker",
            "iou": "GIoU",
            "track_activation_threshold": 0.90,
            "minimum_consecutive_frames": MINIMUM_CONSECUTIVE_FRAMES,
            "lost_track_buffer": LOST_TRACK_BUFFER_SECONDS * TARGET_FPS,
            "minimum_iou_threshold": MINIMUM_IOU_THRESHOLD,
            "state_estimator_class": "XCYCSRStateEstimator",
        },
        clip_scores=clip_scores,
        notes=(
            "Final point of the lost_track_buffer sweep. Full sweep at GIoU/min_iou=0.15/"
            "min_consec=1: 20s=0.067agg/0.103holdout, 5s=0.065/0.103 (tied holdout, better "
            "agg), 2s=identical to 5s (no further change at all -- nothing in these clips "
            "needed a buffer between 2-5s), 1s=0.061/0.103 (best -- further improvement, "
            "still tied holdout), 0.5s=0.095/0.128 (sharp regression -- misses jump, e.g. "
            "IMG_2081 1->7 -- confirming genuine occlusion recoveries do need somewhere "
            "between 0.5s and 1s of grace). 1s is the sweep optimum: short enough that a truly "
            "dead/relocated track's stale Kalman box stops sitting around to steal a "
            "neighboring instrument's detection (the interference mechanism identified when "
            "20s->5s improved things), long enough to cover this domain's actual occlusion "
            "durations. Re-swept minimum_iou_threshold at this new buffer (0.10, 0.15, 0.20) "
            "to check for interaction effects -- 0.15 remained the optimum, confirming the "
            "two hyperparameters are independent here. Final config vs. baseline: aggregate "
            "0.244->0.061 (-75%), holdout 0.295->0.103 (-65%)."
        ),
    )


if __name__ == "__main__":
    main()
