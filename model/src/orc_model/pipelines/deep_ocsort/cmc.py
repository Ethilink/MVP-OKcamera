"""On-the-fly camera-motion compensation (CMC): sparse-feature optical flow +
RANSAC affine fit, run fresh every frame.

Adapted from Deep-OC-SORT's `CMCComputer._affine_sparse_flow` -- upstream
defaults to a `file` mode instead, replaying precomputed BoT-SORT GMC arrays
keyed by MOT sequence name, which doesn't apply to arbitrary clips.
"""

import cv2
import numpy as np


class SparseFlowCMC:
    def __init__(self, minimum_features=10):
        self.minimum_features = minimum_features
        self.prev_gray = None
        self.prev_points = None
        self.feature_params = dict(
            maxCorners=3000,
            qualityLevel=0.01,
            minDistance=1,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04,
        )

    def compute_affine(self, frame_bgr, bboxes_xyxy):
        """Returns a 2x3 affine mapping points in the *previous* frame to
        this one (identity on the first call, or whenever tracking fails).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # mask out detection boxes so tracked keypoints come from the
        # (presumably static) background, not from moving objects
        mask = np.ones_like(gray, dtype=np.uint8)
        if len(bboxes_xyxy) > 0:
            boxes = np.round(bboxes_xyxy).astype(np.int32)
            boxes[boxes < 0] = 0
            for x1, y1, x2, y2 in boxes:
                mask[y1:y2, x1:x2] = 0

        keypoints = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        affine = np.eye(2, 3)

        if self.prev_gray is None or self.prev_points is None or keypoints is None:
            self.prev_gray, self.prev_points = gray, keypoints
            return affine

        matched, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_points, None)
        status = status.reshape(-1).astype(bool)
        prev_points = self.prev_points.reshape(-1, 2)[status]
        curr_points = matched.reshape(-1, 2)[status]

        if prev_points.shape[0] > self.minimum_features:
            estimated, _ = cv2.estimateAffinePartial2D(prev_points, curr_points, method=cv2.RANSAC)
            if estimated is not None:
                affine = estimated

        self.prev_gray, self.prev_points = gray, keypoints
        return affine
