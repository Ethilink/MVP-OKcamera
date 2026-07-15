import numpy as np
import supervision as sv

from .cmc import SparseFlowCMC
from .embedder import Dinov2Embedder
from .ocsort import OCSort


class DeepOCSortTracker:
    """`sv.Detections`-in/-out wrapper around `OCSort`, matching the
    `tracker.update(detections)` shape the sibling SORT/ByteTrack/OC-SORT
    notebooks use (`roboflow/trackers`), so this notebook's loop looks the
    same as theirs.

    OC-SORT only ever emits a box for a detection matched to a track this
    frame (never a coasted/predicted-only box), so every output row maps
    back to exactly one input `detections` row -- that's how mask/class_id
    get carried through untouched below.
    """

    def __init__(
        self,
        det_thresh: float,
        frame_rate: int = 30,
        max_age_seconds: float = 1.0,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        embedding_off: bool = False,
        cmc_off: bool = False,
        mask_crop: bool = False,
    ):
        self.ocsort = OCSort(
            det_thresh=det_thresh,
            embedder=Dinov2Embedder(mask_crop=mask_crop),
            cmc=SparseFlowCMC(),
            max_age=round(max_age_seconds * frame_rate),
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            embedding_off=embedding_off,
            cmc_off=cmc_off,
        )

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        if len(detections) == 0:
            dets = np.empty((0, 5))
            masks = None
        else:
            dets = np.concatenate([detections.xyxy, detections.confidence[:, None]], axis=1)
            masks = detections.mask

        tracked, det_indices = self.ocsort.update(dets, frame, masks=masks)
        if len(tracked) == 0:
            return sv.Detections.empty()

        result = detections[det_indices]
        result.xyxy = tracked[:, :4]
        result.tracker_id = tracked[:, 4].astype(int)
        return result
