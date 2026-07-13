"""Trimmed from Deep-OC-SORT's `trackers/integrated_ocsort_embedding/kalmanfilter.py`
(itself adapted from `filterpy.kalman.KalmanFilter`) -- keeps only the
predict/update/affine-correction/unfreeze path `ocsort.py` actually calls,
dropping the steady-state, RTS-smoother, and log-likelihood machinery so this
module needs nothing beyond numpy.
"""

from copy import deepcopy

import numpy as np
from numpy import dot, zeros, eye, isscalar


def reshape_z(z, dim_z, ndim):
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if z.shape[0] != dim_z:
        raise ValueError(f"z must have shape ({dim_z},) or ({dim_z}, 1)")
    return z.reshape(dim_z, 1) if ndim == 2 else z.reshape(dim_z)


class KalmanFilterNew:
    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))
        self.P = eye(dim_x)
        self.Q = eye(dim_x)
        self.B = None
        self.F = eye(dim_x)
        self.H = zeros((dim_z, dim_x))
        self.R = eye(dim_z)
        self._alpha_sq = 1.0
        self.z = np.array([[None] * self.dim_z]).T

        self.K = np.zeros((dim_x, dim_z))
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))
        self.SI = np.zeros((dim_z, dim_z))

        self._I = np.eye(dim_x)
        self.inv = np.linalg.inv

        # keep all observations, used by `unfreeze` to reconstruct a virtual
        # (constant-velocity) trajectory across an out-of-sight gap
        self.history_obs = []
        self.attr_saved = None
        self.observed = False
        self.last_measurement = None

    def predict(self, F=None, Q=None):
        F = self.F if F is None else F
        Q = self.Q if Q is None else (eye(self.dim_x) * Q if isscalar(Q) else Q)

        self.x = dot(F, self.x)
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

    def freeze(self):
        """Save the parameters before a non-observation (coasted) forward step."""
        self.attr_saved = deepcopy(self.__dict__)

    def apply_affine_correction(self, m, t, new_kf):
        """Apply a camera-motion affine correction to the state (and, if
        frozen, to the saved pre-freeze state so `unfreeze` stays consistent).
        """
        if new_kf:
            big_m = np.kron(np.eye(4, dtype=float), m)
            self.x = big_m @ self.x
            self.x[:2] += t
            self.P = big_m @ self.P @ big_m.T

            if not self.observed and self.attr_saved is not None:
                self.attr_saved["x"] = big_m @ self.attr_saved["x"]
                self.attr_saved["x"][:2] += t
                self.attr_saved["P"] = big_m @ self.attr_saved["P"] @ big_m.T
                self.attr_saved["last_measurement"][:2] = m @ self.attr_saved["last_measurement"][:2] + t
                self.attr_saved["last_measurement"][2:] = m @ self.attr_saved["last_measurement"][2:]
        else:
            self.x[:2] = m @ self.x[:2] + t
            self.x[4:6] = m @ self.x[4:6]
            self.P[:2, :2] = m @ self.P[:2, :2] @ m.T
            self.P[4:6, 4:6] = m @ self.P[4:6, 4:6] @ m.T

            if not self.observed and self.attr_saved is not None:
                self.attr_saved["x"][:2] = m @ self.attr_saved["x"][:2] + t
                self.attr_saved["x"][4:6] = m @ self.attr_saved["x"][4:6]
                self.attr_saved["P"][:2, :2] = m @ self.attr_saved["P"][:2, :2] @ m.T
                self.attr_saved["P"][4:6, 4:6] = m @ self.attr_saved["P"][4:6, 4:6] @ m.T
                self.attr_saved["last_measurement"][:2] = m @ self.attr_saved["last_measurement"][:2] + t

    def unfreeze(self, new_kf):
        if self.attr_saved is None:
            return
        new_history = deepcopy(self.history_obs)
        self.__dict__ = self.attr_saved
        self.history_obs = self.history_obs[:-1]
        occur = [int(d is None) for d in new_history]
        indices = np.where(np.array(occur) == 0)[0]
        index1, index2 = indices[-2], indices[-1]

        box1 = self.last_measurement
        box2 = new_history[index2]
        if new_kf:
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
        else:
            x1, y1, s1, r1 = box1
            w1, h1 = np.sqrt(s1 * r1), np.sqrt(s1 / r1)
            x2, y2, s2, r2 = box2
            w2, h2 = np.sqrt(s2 * r2), np.sqrt(s2 / r2)

        time_gap = index2 - index1
        dx, dy = (x2 - x1) / time_gap, (y2 - y1) / time_gap
        dw, dh = (w2 - w1) / time_gap, (h2 - h1) / time_gap
        for i in range(index2 - index1):
            # linear (constant-velocity) virtual trajectory across the gap
            x, y = x1 + (i + 1) * dx, y1 + (i + 1) * dy
            w, h = w1 + (i + 1) * dw, h1 + (i + 1) * dh
            if new_kf:
                new_box = np.array([x, y, w, h]).reshape((4, 1))
            else:
                new_box = np.array([x, y, w * h, w / float(h)]).reshape((4, 1))
            self.update(new_box)
            if i != (index2 - index1 - 1):
                self.predict()

    def update(self, z, R=None, H=None, new_kf=False):
        self.history_obs.append(z)

        if z is None:
            if self.observed:
                self.last_measurement = self.history_obs[-2]
                self.freeze()
            self.observed = False
            self.z = np.array([[None] * self.dim_z]).T
            self.y = zeros((self.dim_z, 1))
            return

        if not self.observed:
            self.unfreeze(new_kf)
        self.observed = True

        R = self.R if R is None else (eye(self.dim_z) * R if isscalar(R) else R)
        if H is None:
            z = reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H

        self.y = z - dot(H, self.x)
        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        self.K = dot(PHT, self.SI)
        self.x = self.x + dot(self.K, self.y)

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        self.z = deepcopy(z)

    def md_for_measurement(self, z):
        """Mahalanobis distance for a candidate measurement. Run after `predict()`."""
        y = z - self.H @ self.x
        return float(dot(dot(y.T, self.SI), y))
