"""Core configuration and shared data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SourceType(str, Enum):
    LIVE = "live"
    FILE = "file"
    IMAGE = "image"


class PipelineMode(str, Enum):
    REALTIME = "realtime"
    OFFLINE = "offline"


@dataclass
class SourceMetadata:
    source_type: SourceType
    width: int
    height: int
    fps: float
    camera_intrinsics: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    mode: PipelineMode = PipelineMode.REALTIME
    source_type: SourceType = SourceType.FILE
    output_dir: str = "outputs"
    max_frames: Optional[int] = None
    realtime_mode_drop: bool = True
    smoothing_window: int = 5
    gravity: float = 9.80665
    friction_mu: float = 0.7
    min_landmark_confidence: float = 0.25
    smpl_enabled: bool = False
    smpl_model_path: Optional[str] = None
    smpl_gender: str = "neutral"
    skip_display: bool = True
    vis_2d: bool = True


@dataclass
class VideoSourceState:
    frame_idx: int = 0
    frame_count: Optional[int] = None
    source_fps: Optional[float] = None


@dataclass
class TimestampedFrame:
    source_timestamp: float
    ingest_timestamp: float
    frame_idx: int
    frame: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pose2D:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseFrame:
    frame_idx: int
    source_timestamp: float
    landmarks: List[Pose2D]
    world_landmarks: Optional[List[Pose2D]] = None
    shape: Tuple[int, int] = (0, 0)
    pose_3d: Optional[Any] = None
    pose3d_confidence: float = 1.0
    quality_flags: List[str] = field(default_factory=list)


@dataclass
class ReconstructedFrame:
    pose_frame: PoseFrame
    joints_world: List[Tuple[float, float, float]]
    joints_smooth: List[Tuple[float, float, float]]
    segment_scale: float = 1.0
    optimization_residual: float = 0.0


@dataclass
class COGState:
    cog: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    acceleration: Tuple[float, float, float]
    confidence: float
    frame_idx: int
    segment_scale: Dict[str, float] = field(default_factory=dict)


@dataclass
class BOSState:
    polygon: List[Tuple[float, float]]
    polygon_world: Optional[List[Tuple[float, float, float]]]
    support_point_world: Optional[Tuple[float, float, float]]
    left_contact: float
    right_contact: float
    support_area: float
    inside_cog: Optional[bool]
    stability_margin: Optional[float]


@dataclass
class COPState:
    cop: Optional[Tuple[float, float]]
    force: Optional[Tuple[float, float, float]]
    confidence: float
    within_bos: bool
    residual: float
    method: str


@dataclass
class FrameOutput:
    session_id: str
    frame_idx: int
    source_timestamp: float
    ingest_timestamp: float
    source_type: SourceType
    mode: PipelineMode
    pose: Optional[PoseFrame] = None
    reconstructed: Optional[ReconstructedFrame] = None
    cog: Optional[COGState] = None
    bos: Optional[BOSState] = None
    cop: Optional[COPState] = None
    overlays: List[Dict[str, Any]] = field(default_factory=list)
    source_flags: List[str] = field(default_factory=list)
