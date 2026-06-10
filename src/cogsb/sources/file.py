from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import cv2

from cogsb.core import SourceMetadata, SourceType, TimestampedFrame
from cogsb.sources.base import VideoSource


class RecordedVideoSource(VideoSource):
    def __init__(self, video_path: str, output_rgb: bool = False):
        super().__init__(SourceType.FILE)
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        self.video_path = path
        self.output_rgb = output_rgb

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"動画を開けませんでした: {path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = count / fps if fps > 0 else None

        self._metadata = SourceMetadata(
            source_type=SourceType.FILE,
            width=width,
            height=height,
            fps=fps,
            extra={
                "path": str(path),
                "frame_count": count,
                "duration_sec": duration_sec,
            },
        )

    @property
    def metadata(self) -> SourceMetadata:
        return self._metadata

    def iter_frames(self, max_frames: Optional[int] = None) -> Generator[TimestampedFrame, None, None]:
        fps = self._metadata.fps
        frame_idx = 0
        period = 1.0 / fps if fps and fps > 0 else 0.0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            ok, frame = self._cap.read()
            if not ok:
                break

            ingest_ts = datetime.now().timestamp()
            source_ts = frame_idx / fps if fps and fps > 0 else frame_idx * period
            if self.output_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield TimestampedFrame(
                source_timestamp=source_ts,
                ingest_timestamp=ingest_ts,
                frame_idx=frame_idx,
                frame=frame,
                metadata={"source": "file", "path": str(self.video_path)},
            )
            frame_idx += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
