from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
try:
    from scipy.optimize import least_squares  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    least_squares = None

from cogsb.core.types import PoseFrame
from cogsb.reconstruction.body_segments import SEGMENT_TABLE


class SkeletonReconstructor:
    def __init__(self, damping: float = 0.15, length_weight: float = 4.0, smooth_weight: float = 1.0):
        self.damping = damping
        self.length_weight = length_weight
        self.smooth_weight = smooth_weight
        self._segment_lengths: Dict[str, float] = {}

    def _pose_to_array(self, pose_frame: PoseFrame, fallback_world_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        if pose_frame.world_landmarks:
            raw = pose_frame.world_landmarks
            pts = np.array([[p.x, p.y, p.z] for p in raw], dtype=np.float64)
            conf = np.array([max(0.0, min(1.0, float(p.visibility))) for p in raw], dtype=np.float64)
            return pts, conf

        h, w = pose_frame.shape[1], pose_frame.shape[0]
        if w == 0 or h == 0:
            w, h = 1, 1
        lm = pose_frame.landmarks
        pts = np.array([[l.x * w * fallback_world_scale, l.y * h * fallback_world_scale, l.z * fallback_world_scale] for l in lm], dtype=np.float64)
        conf = np.array([max(0.0, min(1.0, float(l.visibility))) for l in lm], dtype=np.float64)
        return pts, conf

    def _segment_target_lengths(self, pts: np.ndarray, confidences: np.ndarray) -> Dict[str, float]:
        targets = {}
        for seg in SEGMENT_TABLE:
            start = self._mean_point(pts, seg.start)
            end = self._mean_point(pts, seg.end)
            start_conf = np.mean([confidences[i] for i in seg.start])
            end_conf = np.mean([confidences[i] for i in seg.end])
            if start_conf < 0.15 or end_conf < 0.15:
                continue
            d = float(np.linalg.norm(start - end))
            if d > 1e-6:
                targets[seg.name] = d
        if not targets and self._segment_lengths:
            return self._segment_lengths.copy()
        if targets:
            self._segment_lengths.update(targets)
        return self._segment_lengths.copy()

    def _mean_point(self, pts: np.ndarray, idxs: Sequence[int]) -> np.ndarray:
        return np.mean(pts[np.array(idxs, dtype=np.int32)], axis=0)

    def initialize_segment_lengths(self, pose_frame: PoseFrame, fallback_world_scale: float = 1.0) -> None:
        pts, conf = self._pose_to_array(pose_frame, fallback_world_scale)
        self._segment_lengths = self._segment_target_lengths(pts, conf)

    def reconstruct(
        self,
        pose_frame: PoseFrame,
        prev_joints: Optional[Sequence[Tuple[float, float, float]]] = None,
        fallback_world_scale: float = 1.0,
    ) -> Tuple[List[Tuple[float, float, float]], float, float]:
        if len(pose_frame.landmarks) == 0:
            return [], 0.0, 1.0

        joints, confidences = self._pose_to_array(pose_frame, fallback_world_scale)
        if self._segment_lengths:
            segment_targets = self._segment_lengths
        else:
            segment_targets = self._segment_target_lengths(joints, confidences)

        x0 = joints.flatten()
        n = joints.shape[0]

        def unpack(x: np.ndarray) -> np.ndarray:
            return x.reshape((n, 3))

        if least_squares is None:
            refined = joints
            residual = 0.0
            scale = float(np.linalg.norm(refined - joints))
            return [tuple(float(v) for v in row) for row in refined], residual, scale

        prev_np = np.array(prev_joints, dtype=np.float64) if prev_joints is not None else None

        def residuals(x: np.ndarray) -> np.ndarray:
            p = unpack(x)
            res = []

            # observation residual
            obs = (p - joints).reshape(-1)
            obs_w = np.repeat(confidences, 3)
            res.append(np.atleast_1d(self.damping * np.sqrt(obs_w) * obs))

            # segment length residuals
            for seg_name, length in segment_targets.items():
                seg = next((s for s in SEGMENT_TABLE if s.name == seg_name), None)
                if seg is None:
                    continue
                pa = self._mean_point(p, seg.start)
                pb = self._mean_point(p, seg.end)
                cur_len = np.linalg.norm(pb - pa)
                if cur_len <= 1e-8:
                    continue
                res.append(np.atleast_1d((cur_len - length) * self.length_weight))

            # temporal smoothness
            if prev_np is not None and prev_np.shape == p.shape:
                res.append(np.atleast_1d(self.smooth_weight * (p - prev_np).reshape(-1)))
            return np.concatenate([r for r in res if r is not None])

        try:
            result = least_squares(  # type: ignore[misc]
                residuals,
                x0,
                method="trf",
                max_nfev=40,
                xtol=1e-4,
                ftol=1e-4,
                gtol=1e-4,
            )
            refined = result.x.reshape((n, 3))
            residual = float(np.mean(np.square(result.fun)))
            scale = float(np.linalg.norm(refined - joints))
            return [tuple(float(v) for v in row) for row in refined], residual, scale
        except Exception:
            return [tuple(float(v) for v in row) for row in joints], 0.0, 0.0


class Smoother:
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.prev = None

    def update(self, values: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        arr = np.array(values, dtype=np.float64)
        if self.prev is None:
            self.prev = arr.copy()
            return values
        sm = self.alpha * arr + (1 - self.alpha) * self.prev
        self.prev = sm
        return [tuple(float(v) for v in row) for row in sm]
