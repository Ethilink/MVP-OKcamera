"""Adapted from Deep-OC-SORT's `trackers/integrated_ocsort_embedding/ocsort.py`.

Kept close to the original: the Kalman filter parameterization, the
Observation-Centric Momentum/Recovery (OCM/OCR) association, and the
embedding-fused cost matrix are unchanged. Differences from upstream:
  - `OCSort.__init__` takes `embedder`/`cmc` objects directly instead of
    building MOT-checkpoint-specific ones from `argparse` args.
  - `update()` takes detections already in the frame's own pixel
    coordinates (no YOLOX-tensor-space rescale) and a frame image instead of
    an MOT sequence `tag` string.
  - `KalmanBoxTracker` additionally remembers which detection row it was
    last matched to (`last_det_index`), so `update()` can return that
    alongside each output row -- upstream doesn't need this since its output
    rows are written straight to a results file.
  - `KalmanBoxTracker` also remembers the *raw* per-frame appearance
    embedding of that same matched detection (`last_det_emb`), separate from
    `self.emb` (the track's EMA-smoothed embedding used for association).
    `update()` returns `last_det_emb` per output row alongside
    `last_det_index` -- downstream tracklet refinement (GTA-Link-style
    split/connect) needs the raw per-frame feature, not the smoothed one.
"""

import numpy as np

from .association import (
    associate,
    ciou_batch,
    ct_dist,
    diou_batch,
    giou_batch,
    iou_batch,
    linear_assignment,
)


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    return observations[max(observations.keys())]


def convert_bbox_to_z_new(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = bbox[0] + w / 2.0, bbox[1] + h / 2.0
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox_new(x):
    x, y, w, h = x.reshape(-1)[:4]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
    return np.diag(((p * w) ** 2, (p * h) ** 2, (p * w) ** 2, (p * h) ** 2, (v * w) ** 2, (v * h) ** 2, (v * w) ** 2, (v * h) ** 2))


def new_kf_measurement_noise(w, h, m=1 / 20):
    w_var, h_var = (m * w) ** 2, (m * h) ** 2
    return np.diag((w_var, h_var, w_var, h_var))


class KalmanBoxTracker:
    """Internal state of an individual tracked object, observed as a bbox.

    Only the `new_kf` (x, y, w, h + velocities) parameterization from
    upstream is kept -- the `orig` flag selecting the original SORT
    `filterpy`-based (x, y, s, r) filter is dropped since this playground
    never exercises it.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, emb=None):
        from .kalmanfilter import KalmanFilterNew

        self.kf = KalmanFilterNew(dim_x=8, dim_z=4)
        self.kf.F = np.array(
            [
                # x  y  w  h  x' y' w' h'
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )
        _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
        self.kf.P = new_kf_process_noise(w, h)
        self.kf.P[:4, :4] *= 4
        self.kf.P[4:, 4:] *= 100
        self.kf.x[:4] = convert_bbox_to_z_new(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # [-1,-1,-1,-1,-1] is the "no observation" placeholder throughout,
        # so k_previous_obs/associate can build fixed-shape observation arrays
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.observations = {}
        self.velocity = None
        self.delta_t = delta_t

        self.emb = emb
        self.frozen = False
        self.last_det_index = None
        self.last_det_emb = emb

    def update(self, bbox, det_index=None, det_emb=None):
        if bbox is not None:
            self.frozen = False
            self.last_det_index = det_index
            self.last_det_emb = det_emb

            if self.last_observation.sum() >= 0:
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                self.velocity = speed_direction(previous_box, bbox)

            self.last_observation = bbox
            self.observations[self.age] = bbox

            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
            R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
            self.kf.update(convert_bbox_to_z_new(bbox), R=R, new_kf=True)
        else:
            self.kf.update(bbox, new_kf=True)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        m, t = affine[:, :2], affine[:, 2].reshape(2, 1)
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        self.kf.apply_affine_correction(m, t, new_kf=True)

    def predict(self):
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] = 0
        if self.kf.x[3] + self.kf.x[7] <= 0:
            self.kf.x[7] = 0
        if self.frozen:
            self.kf.x[6] = self.kf.x[7] = 0

        Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return convert_x_to_bbox_new(self.kf.x)

    def get_state(self):
        return convert_x_to_bbox_new(self.kf.x)


ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


class OCSort:
    def __init__(
        self,
        det_thresh,
        embedder,
        cmc,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.75,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        KalmanBoxTracker.count = 0

        self.embedder = embedder
        self.cmc = cmc
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off

    def update(
        self, dets: np.ndarray, frame: np.ndarray, masks: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Params:
          dets: (N, 5) array of [x1, y1, x2, y2, score] in `frame`'s own pixel
            coordinates. Call once per frame even with `dets` empty.
          frame: the frame's BGR image, for the embedder and CMC.
          masks: one full-frame boolean mask per row of `dets`, or None.
            Forwarded to the embedder -- only used if it was built with
            `mask_crop=True`.
        Returns:
          (tracked, det_indices, det_embeddings): `tracked` is an (M, 5)
          array of [x1, y1, x2, y2, track_id]; `det_indices` is the
          input-`dets` row each tracked box came from (only detections
          matched to a track this frame are ever returned -- OC-SORT doesn't
          emit boxes for coasted/predicted-only tracks); `det_embeddings` is
          an (M, D) array, row-aligned with `tracked`/`det_indices`, of the
          *raw per-frame* appearance embedding of the detection each track
          matched this frame (not the track's EMA-smoothed `trk.emb` used
          internally for association -- tracklet refinement downstream needs
          the unsmoothed per-frame feature). When `embedding_off=True`,
          `det_embeddings` is a dummy `(M, 1)` array of ones (same
          pass-through as `dets_embs` internally) -- callers must not treat
          it as a real embedding in that case.
        """
        self.frame_count += 1
        remain_inds = dets[:, 4] > self.det_thresh
        dets = dets[remain_inds]
        # maps a row in the filtered `dets` back to a row in the original
        # input, so returned det_indices are meaningful to the caller
        original_indices = np.where(remain_inds)[0]
        masks = masks[remain_inds] if masks is not None else None

        dets_embs = np.ones((dets.shape[0], 1))
        if not self.embedding_off and dets.shape[0] != 0:
            dets_embs = self.embedder.compute_embedding(frame, dets[:, :4], masks=masks)

        if not self.cmc_off:
            transform = self.cmc.compute_affine(frame, dets[:, :4])
            for trk in self.trackers:
                trk.apply_affine_correction(transform)

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)  # -> 1 as detector confidence drops

        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trk_embs = np.array(trk_embs)
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        # first round: IoU + OCM (velocity-consistency) + embedding cost
        matched, unmatched_dets, unmatched_trks = associate(
            dets,
            trks,
            dets_embs,
            trk_embs,
            self.iou_threshold,
            velocities,
            k_observations,
            self.inertia,
            self.w_association_emb,
            self.aw_off,
            self.aw_param,
            self.embedding_off,
            grid_off=True,
        )
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(
                dets[det_idx, :], det_index=int(original_indices[det_idx]), det_emb=dets_embs[det_idx]
            )
            self.trackers[trk_idx].update_emb(dets_embs[det_idx], alpha=dets_alpha[det_idx])

        # second round: OCR -- re-try unmatched dets against unmatched tracks'
        # *last observed* (not predicted) box, plain IoU only
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = np.array(self.asso_func(left_dets, left_trks))
            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices, to_remove_trk_indices = [], []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(
                        dets[det_ind, :], det_index=int(original_indices[det_ind]), det_emb=dets_embs[det_ind]
                    )
                    self.trackers[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for trk_idx in unmatched_trks:
            self.trackers[trk_idx].update(None)

        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(dets[det_idx, :], delta_t=self.delta_t, emb=dets_embs[det_idx])
            trk.last_det_index = int(original_indices[det_idx])
            self.trackers.append(trk)

        ret, ret_det_indices, ret_embeddings = [], [], []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # a track spawned this frame has time_since_update == 0 too,
                # but was never run through update() -- last_observation is
                # still its `-1` placeholder, so fall back to the Kalman
                # state (seeded from the spawning bbox in __init__)
                d = trk.get_state()[0] if trk.last_observation.sum() < 0 else trk.last_observation[:4]
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
                ret_det_indices.append(trk.last_det_index)
                ret_embeddings.append(trk.last_det_emb)
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return (
                np.concatenate(ret),
                np.array(ret_det_indices, dtype=int),
                np.array(ret_embeddings),
            )
        return (
            np.empty((0, 5)),
            np.empty((0,), dtype=int),
            np.empty((0, self.embedder.embedding_dim)),
        )
