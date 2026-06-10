from __future__ import annotations

from pathlib import Path
import os
import urllib.request
from typing import Any, List, Optional, cast

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

        solutions_lib = getattr(mp, "solutions", None)
        pose_module = getattr(solutions_lib, "pose", None) if solutions_lib is not None else None
        pose_factory = getattr(pose_module, "Pose", None) if pose_module is not None else None
        drawing_utils = getattr(solutions_lib, "drawing_utils", None) if solutions_lib is not None else None
        if callable(pose_factory):
            self.pose = pose_factory(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                smooth_landmarks=smooth_landmarks,
            )
            self.draw = drawing_utils
            self._task_mode = False
        else:
            try:
                from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
                from mediapipe.tasks.python.core import base_options as base_options_lib
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
                from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "mediapipe.solutions が利用できないため、Tasks API へ切り替えを試みましたが依存関係の準備が不完全です。"
                ) from exc

            model_path = self._ensure_task_model()
            options = PoseLandmarkerOptions(
                base_options=base_options_lib.BaseOptions(model_asset_path=model_path),
                running_mode=VisionTaskRunningMode.IMAGE,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.pose = PoseLandmarker.create_from_options(options)
            self._Image = Image
            self._ImageFormat = ImageFormat.SRGB
            self._task_mode = True

            # For drawing utilities we only need compatibility with legacy output path;
            # tasks API does not expose an identical drawing helper object.
            self.draw = None  # type: ignore[assignment]

        self.output_world = output_world
        self.output_rgb = output_rgb
        self._min_detection_confidence = min_detection_confidence
        self._smooth_landmarks = smooth_landmarks

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
        h, w = rgb.shape[:2]

        if self._task_mode:
            mp_image = self._Image(image_format=self._ImageFormat.SRGB, data=rgb)
            results = self.pose.detect(mp_image)
            if not results.pose_landmarks:
                return PoseFrame(
                    frame_idx=frame_idx,
                    source_timestamp=source_ts,
                    landmarks=[],
                    shape=(w, h),
                    quality_flags=["no_pose"],
                )

            raw = results.pose_landmarks[0]
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
                wraw = results.pose_world_landmarks[0]
                world = [
                    Pose2D(
                        x=float(lm.x),
                        y=float(lm.y),
                        z=float(lm.z),
                        visibility=float(getattr(lm, "visibility", 1.0)),
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

        results = self.pose.process(rgb)

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

    @staticmethod
    def _ensure_task_model() -> str:
        # NOTE: mediapipe>=0.10.35 exposes Tasks API as default on Python 3.13.
        # The legacy solutions API package layout does not provide `solutions` anymore.
        override = os.getenv("COGSB_POSE_LANDMARKER_TASK_PATH")
        if override:
            task_path = Path(override).expanduser().resolve()
            if not task_path.exists():
                raise RuntimeError(f"指定されたモデルファイルが見つかりません: {task_path}")
            return str(task_path)

        cache_dir = Path.home() / ".cache" / "cogsb"
        model_path = cache_dir / "pose_landmarker_lite.task"
        if model_path.exists() and model_path.stat().st_size > 0:
            return str(model_path)

        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(model_url, timeout=30) as response:
                payload = response.read()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "mediapipe.tasks を利用するための pose_landmarker モデル(.task)を取得できませんでした。\n"
                "COGSB_POSE_LANDMARKER_TASK_PATH を有効なモデルファイル(.task)に設定するか、ネットワーク環境を確認してください。"
            ) from exc

        model_path.write_bytes(payload)
        return str(model_path)
