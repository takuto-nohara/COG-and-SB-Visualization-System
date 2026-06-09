from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class SMPLFitter:
    def __init__(
        self,
        enabled: bool = False,
        model_path: Optional[str] = None,
        gender: str = "neutral",
        use_gpu: bool = False,
    ) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.gender = gender
        self.use_gpu = use_gpu
        self._impl_available = False
        self._model = None

        if not enabled or model_path is None:
            return

        try:
            import torch
            import smplx

            self.torch = torch
            self.smplx = smplx
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            self._model = smplx.create(
                model_folder=model_path,
                model_type="smpl",
                gender=gender,
                use_face_contour=False,
                num_betas=10,
                ext="npz",
            ).to(self.device)
            self._impl_available = True
        except Exception:
            self._impl_available = False

    @property
    def available(self) -> bool:
        return self._impl_available and self._model is not None

    def fit(self, world_joints):
        if not self.available:
            return {
                "status": "skipped",
                "reason": "smpl model not available",
                "joints": world_joints,
            }

        try:
            import torch

            joints = torch.tensor(np.asarray(world_joints, dtype=np.float32), device=self.device)
            # placeholder fit: no optimization, use zero pose and betas as warm start.
            betas = torch.zeros((1, self._model.num_betas), device=self.device)
            body_pose = torch.zeros((1, self._model.num_body_pose), device=self.device)
            global_orient = torch.zeros((1, 3), device=self.device)

            out = self._model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=False,
            )
            verts = out.vertices.detach().cpu().numpy()
            return {
                "status": "ok",
                "verts_mean": float(np.mean(verts)),
                "joint_count": len(world_joints),
            }
        except Exception as exc:
            return {
                "status": "error",
                "reason": str(exc),
            }
