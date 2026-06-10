from __future__ import annotations

from typing import Callable, List, Optional
import threading
import time

import numpy as np

from cogsb.core import COGState, PipelineConfig, PipelineMode, PoseFrame, ReconstructedFrame
from cogsb.core.types import BOSState, FrameOutput
from cogsb.core.util import new_session_id
from cogsb.kinematics import COPEstimator, COPInput, COGEstimator, compute_bos, inside_bos
from cogsb.pipeline.writer import FrameWriter
from cogsb.reconstruction import Smoother, SkeletonReconstructor
from cogsb.smpl.smpl_fitter import SMPLFitter


class AnalysisEngine:
    def __init__(self, config: PipelineConfig, smpl_fitter: Optional[SMPLFitter] = None):
        self.config = config
        self.reconstructor = SkeletonReconstructor()
        self.smoother = Smoother(alpha=0.30)
        self.cog_estimator = COGEstimator(gravity=config.gravity)
        self.cop_estimator = COPEstimator(default_mass_kg=70.0)
        self.smpl_fitter = smpl_fitter
        self.prev_joints: Optional[list[tuple[float, float, float]]] = None
        self.prev_cog = None
        self.prev_cog_vel = None
        self.prev_joint_velocity: Optional[list[tuple[float, float, float]]] = None
        self.prev_bos_polygon = []
        self.prev_cop = None
        self.stats_processed = 0
        self.stats_dropped = 0
        self.latency_ms = 0.0
        self._prev_ts = None

    @staticmethod
    def _to_points(pose_frame: PoseFrame):
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
        if self.prev_cog_vel is None:
            return tuple(float(x) for x in v), (0.0, 0.0, 0.0)

        a = (v - self.prev_cog_vel) / dt
        return tuple(float(x) for x in v), tuple(float(x) for x in a)

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
        prev_source_ts = None

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
                cog_vec = (float(cog_vec[0]), float(cog_vec[1]), float(cog_vec[2]))
                cog_np = np.array(cog_vec, dtype=np.float64)
                v, a = self._estimate_pose_derivative(cog_np, frame.source_timestamp)
                self._prev_ts = frame.source_timestamp

                self.prev_cog = cog_np
                self.prev_cog_vel = np.array(v)

                if np.linalg.norm(cog_vec) > 0:
                    conf = 1.0 / (1.0 + scale + residual)
                else:
                    conf = 0.0

                cog_state = COGState(
                    cog=tuple(float(x) for x in cog_vec),
                    velocity=v,
                    acceleration=a,
                    confidence=float(conf),
                    frame_idx=frame.frame_idx,
                )

                bos_dict = compute_bos(
                    world_landmarks=joints_smooth,
                    prev_landmarks=self.prev_joints,
                    prev_polygon=self.prev_bos_polygon,
                    dt=dt,
                )

                polygon = bos_dict["polygon"]
                inside, margin = inside_bos((float(cog_vec[0]), float(cog_vec[1])), polygon)
                bos_state = BOSState(
                    polygon=polygon,
                    left_contact=bos_dict["left_contact"],
                    right_contact=bos_dict["right_contact"],
                    support_area=bos_dict["area"],
                    inside_cog=inside,
                    stability_margin=margin,
                )
                self.prev_bos_polygon = polygon

                cop_state = self.cop_estimator.estimate(
                    COPInput(
                        cog=(float(cog_vec[0]), float(cog_vec[1]), float(cog_vec[2])),
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
