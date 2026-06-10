from __future__ import annotations

import abc
from typing import Generator, Optional

from cogsb.core.types import TimestampedFrame, SourceMetadata


class VideoSource(abc.ABC):
    """Common frame source interface for live and recorded inputs."""

    def __init__(self, source_type, source_fps: Optional[float] = None):
        self.source_type = source_type
        self.source_fps = source_fps

    @property
    @abc.abstractmethod
    def metadata(self) -> SourceMetadata:
        raise NotImplementedError

    @abc.abstractmethod
    def iter_frames(self, max_frames: Optional[int] = None) -> Generator[TimestampedFrame, None, None]:
        raise NotImplementedError
