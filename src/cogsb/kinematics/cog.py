from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from cogsb.reconstruction.body_segments import SEGMENT_TABLE, all_segment_mass, segment_center


class COGEstimator:
    def __init__(self, gravity: float = 9.80665):
        self.gravity = gravity
        self.mass_profile = all_segment_mass()
        # lightweight person-specific correction factors
        self.adjustment = {k: 1.0 for k in self.mass_profile}

    def update_mass_profile(self, factor_overrides: Dict[str, float]) -> None:
        for k, v in factor_overrides.items():
            if k in self.adjustment and v > 0:
                self.adjustment[k] = float(v)

    def compute(self, joints_world: List[Tuple[float, float, float]]) -> np.ndarray:
        total_mass = 0.0
        weighted = np.zeros(3, dtype=np.float64)
        pts = joints_world

        for seg in SEGMENT_TABLE:
            p = segment_center(pts, seg)
            key = seg.name
            m = self.mass_profile.get(key, 0.0) * self.adjustment.get(key, 1.0)
            weighted += m * np.array(p)
            total_mass += m

        if total_mass <= 0:
            return np.zeros(3)
        return weighted / total_mass


def smooth_time_derivative(vec_prev: List[Tuple[float, float, float]], vec_curr: List[Tuple[float, float, float]], dt: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    a = np.array(vec_curr, dtype=np.float64)
    b = np.array(vec_prev, dtype=np.float64)
    if dt <= 0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    raw_vel = np.mean(a - b, axis=0) / dt
    vel = (float(raw_vel[0]), float(raw_vel[1]), float(raw_vel[2]))
    # accel cannot be computed without two previous steps here; return zeros.
    acc = (0.0, 0.0, 0.0)
    return vel, acc
