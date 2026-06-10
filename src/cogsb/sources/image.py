from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import cv2

from cogsb.core import SourceMetadata, SourceType, TimestampedFrame
from cogsb.sources.base import VideoSource


class RecordedImageSource(VideoSource):
    def __init__(self, image_path: str, output_rgb: bool = False):
        super().__init__(SourceType.IMAGE)
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"画像を開けませんでした: {path}")

        self.image_path = path
        self.output_rgb = output_rgb
        self._frame = frame
        height, width = frame.shape[:2]
        self._metadata = SourceMetadata(
            source_type=SourceType.IMAGE,
            width=width,
            height=height,
            fps=1.0,
            extra={
                "path": str(path),
                "source": "image",
            },
        )

    @property
    def metadata(self) -> SourceMetadata:
        return self._metadata

    def iter_frames(self, max_frames: Optional[int] = None) -> Generator[TimestampedFrame, None, None]:
        if max_frames is not None and max_frames <= 0:
            return

        ingest_ts = datetime.now().timestamp()
        frame = self._frame.copy()
        if self.output_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield TimestampedFrame(
            source_timestamp=0.0,
            ingest_timestamp=ingest_ts,
            frame_idx=0,
            frame=frame,
            metadata={"source": "image", "path": str(self.image_path)},
        )

    def close(self) -> None:
        return None
