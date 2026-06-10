from __future__ import annotations

from typing import List, Optional

import cv2
try:
    import mediapipe as mp  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    mp = None
import numpy as np

from cogsb.core.types import Pose2D, PoseFrame


class MediaPipePoseEstimator:
    def __init__(
        self,
        model_complexity: int = 1,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
        output_world: bool = True,
        output_rgb: bool = False,
    ) -> None:
        if mp is None:
            raise RuntimeError("mediapipe がインストールされていません。pip install mediapipe を実行してください。")
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=smooth_landmarks,
        )
        self.output_world = output_world
        self.output_rgb = output_rgb
        self.draw = mp.solutions.drawing_utils  # type: ignore[assignment]

    def close(self) -> None:
        self.pose.close()

    def estimate(self, frame, frame_idx: int, source_ts: float) -> PoseFrame:
        if frame is None:
            return PoseFrame(
                frame_idx=frame_idx,
                source_timestamp=source_ts,
                landmarks=[],
                world_landmarks=[],
                shape=(0, 0),
                quality_flags=["empty_frame"],
            )

        if self.output_rgb:
            rgb = frame
        else:
            if len(frame.shape) == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame

        results = self.pose.process(rgb)
        h, w = rgb.shape[:2]

        if not results.pose_landmarks:
            return PoseFrame(
                frame_idx=frame_idx,
                source_timestamp=source_ts,
                landmarks=[],
                shape=(w, h),
                quality_flags=["no_pose"],
            )

        raw = results.pose_landmarks.landmark
        landmarks = [
            Pose2D(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
            for lm in raw
        ]

        if self.output_world and results.pose_world_landmarks:
            wraw = results.pose_world_landmarks.landmark
            world = [
                Pose2D(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=float(lm.visibility) if hasattr(lm, "visibility") else 1.0,
                )
                for lm in wraw
            ]
        else:
            world = None

        return PoseFrame(
            frame_idx=frame_idx,
            source_timestamp=source_ts,
            landmarks=landmarks,
            world_landmarks=world,
            shape=(w, h),
            pose3d_confidence=1.0,
            quality_flags=[],
        )
