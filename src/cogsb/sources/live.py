from __future__ import annotations

from datetime import datetime
from typing import Generator, Optional

import cv2
import numpy as np

from cogsb.core import SourceMetadata, SourceType, TimestampedFrame
from cogsb.sources.base import VideoSource


class LiveCameraSource(VideoSource):
    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        output_rgb: bool = False,
    ) -> None:
        super().__init__(SourceType.LIVE, fps)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps_hint = fps
        self.output_rgb = output_rgb

        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"ライブカメラを開けませんでした: index={camera_index}")

        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not self._source_fps or self._source_fps <= 1:
            self._source_fps = fps or 30.0

        self._metadata = SourceMetadata(
            source_type=SourceType.LIVE,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._source_fps,
            extra={"camera_index": camera_index},
        )

    @property
    def metadata(self) -> SourceMetadata:
        return self._metadata

    def iter_frames(self, max_frames: Optional[int] = None) -> Generator[TimestampedFrame, None, None]:
        frame_idx = 0
        last_frame_ts = datetime.now().timestamp()
        dropped = 0

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            ok, frame = self._cap.read()
            if not ok:
                dropped += 1
                if dropped > 10:
                    break
                continue

            ingest_ts = datetime.now().timestamp()
            dt = 1.0 / self._source_fps
            source_ts = last_frame_ts + dt
            last_frame_ts = source_ts

            if self.output_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield TimestampedFrame(
                source_timestamp=source_ts,
                ingest_timestamp=ingest_ts,
                frame_idx=frame_idx,
                frame=frame,
                metadata={"source": "live", "frame_dropped": dropped},
            )
            frame_idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
