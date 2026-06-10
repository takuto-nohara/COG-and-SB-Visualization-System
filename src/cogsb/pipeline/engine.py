from __future__ import annotations

from typing import Callable, List, Optional, cast
import threading
import time

import numpy as np

from cogsb.core import COGState, PipelineConfig, PipelineMode, PoseFrame, ReconstructedFrame
from cogsb.core.types import BOSState, FrameOutput
from cogsb.core.util import new_session_id
from cogsb.kinematics import COPEstimator, COPInput, COGEstimator, compute_bos
from cogsb.pipeline.writer import FrameWriter
from cogsb.reconstruction import Smoother, SkeletonReconstructor
from cogsb.smpl.smpl_fitter import SMPLFitter
from cogsb.kinematics.bos import SupportFrame


class AnalysisEngine:
    def __init__(self, config: PipelineConfig, smpl_fitter: Optional[SMPLFitter] = None):
        self.config = config
        self.reconstructor = SkeletonReconstructor()
        self.smoother = Smoother(alpha=0.30)
        self.cog_estimator = COGEstimator(gravity=config.gravity)
        self.cop_estimator = COPEstimator(default_mass_kg=70.0)
        self.smpl_fitter = smpl_fitter
        self.prev_joints: Optional[list[tuple[float, float, float]]] = None
        self.prev_cog: Optional[np.ndarray] = None
        self.prev_cog_vel: Optional[np.ndarray] = None
        self.prev_joint_velocity: Optional[list[tuple[float, float, float]]] = None
        self.prev_bos_polygon: list[tuple[float, float]] = []
        self.prev_cop: Optional[tuple[float, float]] = None
        self._prev_support_frame: Optional[SupportFrame] = None
        self.stats_processed = 0
        self.stats_dropped = 0
        self.latency_ms = 0.0
        self._prev_ts: Optional[float] = None

    @staticmethod
    def _to_points(pose_frame: PoseFrame) -> list[tuple[float, float, float]]:
        if pose_frame.world_landmarks:
            return [(p.x, p.y, p.z) for p in pose_frame.world_landmarks]
        return [(float(p.x), float(p.y), float(p.z)) for p in pose_frame.landmarks]

    def _estimate_pose_derivative(self, cog: np.ndarray, ts: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self.prev_cog is None or self._prev_ts is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        dt = ts - self._prev_ts
        if dt <= 1e-6:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        v = (cog - self.prev_cog) / dt
        velocity = (float(v[0]), float(v[1]), float(v[2]))
        if self.prev_cog_vel is None:
            return velocity, (0.0, 0.0, 0.0)

        a = (v - self.prev_cog_vel) / dt
        acceleration = (float(a[0]), float(a[1]), float(a[2]))
        return velocity, acceleration

    def run(
        self,
        source,
        estimator,
        max_frames: Optional[int] = None,
        *,
        on_frame: Optional[Callable[[object, FrameOutput], bool | None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> List[FrameOutput]:
        session_id = new_session_id("session")
        writer = FrameWriter(self.config.output_dir)
        outputs: List[FrameOutput] = []
        prev_source_ts: Optional[float] = None

        try:
            for frame in source.iter_frames(max_frames=max_frames):
                if stop_event is not None and stop_event.is_set():
                    break
                start_ms = time.perf_counter()
                pose_frame = estimator.estimate(frame.frame, frame.frame_idx, frame.source_timestamp)

                if not pose_frame.landmarks:
                    self.stats_dropped += 1
                    self.prev_joint_velocity = None
                    self.prev_joints = None
                    out = FrameOutput(
                        session_id=session_id,
                        frame_idx=frame.frame_idx,
                        source_timestamp=frame.source_timestamp,
                        ingest_timestamp=frame.ingest_timestamp,
                        source_type=source.source_type,
                        mode=self.config.mode,
                        pose=pose_frame,
                        source_flags=["no_landmarks"],
                    )
                    outputs.append(out)
                    writer.write_frame(out)
                    if on_frame is not None:
                        try:
                            should_continue = on_frame(frame, out)
                        except Exception:
                            should_continue = False
                        if should_continue is False:
                            break
                    continue

                if frame.frame_idx == 0:
                    self.reconstructor.initialize_segment_lengths(pose_frame)

                dt = 1.0 / (source.metadata.fps or 30.0)
                if prev_source_ts is not None:
                    dt = max(frame.source_timestamp - prev_source_ts, 1e-6)
                prev_source_ts = frame.source_timestamp

                joints_recon, residual, scale = self.reconstructor.reconstruct(
                    pose_frame,
                    prev_joints=self.prev_joints,
                    prev_velocity=self.prev_joint_velocity,
                    dt=dt,
                    fallback_world_scale=1.0,
                )
                joints_smooth = self.smoother.update(joints_recon)

                cog_vec = self.cog_estimator.compute(joints_smooth)
                cog_vec_3d = (
                    float(cog_vec[0]),
                    float(cog_vec[1]),
                    float(cog_vec[2]),
                )
                cog_np = np.array(cog_vec_3d, dtype=np.float64)
                v, a = self._estimate_pose_derivative(cog_np, frame.source_timestamp)
                self._prev_ts = frame.source_timestamp

                self.prev_cog = cog_np
                self.prev_cog_vel = np.array(v)

                if np.linalg.norm(cog_vec) > 0:
                    conf = 1.0 / (1.0 + scale + residual)
                else:
                    conf = 0.0

                cog_state = COGState(
                    cog=cog_vec_3d,
                    velocity=v,
                    acceleration=a,
                    confidence=float(conf),
                    frame_idx=frame.frame_idx,
                )

                bos_result, support_frame = compute_bos(
                    world_landmarks=joints_smooth,
                    prev_landmarks=self.prev_joints,
                    prev_polygon=self.prev_bos_polygon,
                    dt=dt,
                    cog_point=cog_vec_3d,
                    prev_support_frame=self._prev_support_frame,
                    return_support_frame=True,
                )
                if support_frame is not None:
                    self._prev_support_frame = support_frame

                polygon = bos_result["polygon"]
                if len(polygon) < 3 and self.prev_bos_polygon:
                    polygon = self.prev_bos_polygon
                polygon_world = bos_result["polygon_world"]
                support_point_world = bos_result["support_point_world"]
                left_contact = bos_result["left_contact"]
                right_contact = bos_result["right_contact"]
                support_area = bos_result["area"]
                inside_cog = bos_result["inside"]
                stability_margin = bos_result["margin"]

                bos_state = BOSState(
                    polygon=polygon,
                    polygon_world=polygon_world,
                    support_point_world=support_point_world,
                    left_contact=left_contact,
                    right_contact=right_contact,
                    support_area=support_area,
                    inside_cog=inside_cog,
                    stability_margin=stability_margin,
                )
                self.prev_bos_polygon = polygon

                cop_state = self.cop_estimator.estimate(
                    COPInput(
                        cog=cog_vec_3d,
                        acceleration=a,
                        mass_kg=70.0,
                        polygon=polygon,
                        prev_cop=self.prev_cop,
                        friction_mu=self.config.friction_mu,
                        gravity=self.config.gravity,
                    )
                )
                self.prev_cop = cop_state.cop

                reconstructed = ReconstructedFrame(
                    pose_frame=pose_frame,
                    joints_world=joints_recon,
                    joints_smooth=joints_smooth,
                    segment_scale=scale,
                    optimization_residual=residual,
                )

                out_overlays = []
                if self.smpl_fitter and self.config.smpl_enabled:
                    smpl_info = self.smpl_fitter.fit(joints_smooth)
                    out_overlays.append({
                        "type": "smpl",
                        "result": {
                            "status": smpl_info.get("status", "skipped") if isinstance(smpl_info, dict) else "unknown",
                            "reason": smpl_info.get("reason") if isinstance(smpl_info, dict) else None,
                        },
                    })

                out = FrameOutput(
                    session_id=session_id,
                    frame_idx=frame.frame_idx,
                    source_timestamp=frame.source_timestamp,
                    ingest_timestamp=frame.ingest_timestamp,
                    source_type=source.source_type,
                    mode=self.config.mode,
                    pose=pose_frame,
                    reconstructed=reconstructed,
                    cog=cog_state,
                    bos=bos_state,
                    cop=cop_state,
                    overlays=out_overlays,
                )

                if self.prev_joints is not None and len(self.prev_joints) == len(joints_recon):
                    prev_arr = np.array(self.prev_joints, dtype=np.float64)
                    cur_arr = np.array(joints_recon, dtype=np.float64)
                    if prev_arr.shape == cur_arr.shape:
                        vel = (cur_arr - prev_arr) / max(dt, 1e-6)
                        if np.all(np.isfinite(vel)):
                            self.prev_joint_velocity = [
                                (float(v[0]), float(v[1]), float(v[2])) for v in vel
                            ]
                        else:
                            self.prev_joint_velocity = None
                    else:
                        self.prev_joint_velocity = None
                else:
                    self.prev_joint_velocity = None

                self.prev_joints = joints_recon
                self.stats_processed += 1
                self.latency_ms = (time.perf_counter() - start_ms) * 1000.0

                outputs.append(out)
                writer.write_frame(out)
                if on_frame is not None:
                    try:
                        should_continue = on_frame(frame, out)
                    except Exception:
                        should_continue = False
                    if should_continue is False:
                        break

                if self.config.mode == PipelineMode.REALTIME and self.config.max_frames and self.stats_processed >= self.config.max_frames:
                    break
        finally:
            writer.close()
            if hasattr(source, "close"):
                source.close()

        return outputs
