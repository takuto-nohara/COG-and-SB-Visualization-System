from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from cogsb.core.types import FrameOutput


class FrameWriter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "frames.jsonl"

    def write_frame(self, frame_output: FrameOutput) -> None:
        def _normalize(v: Any):
            if isinstance(v, tuple):
                return list(v)
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dict):
                return {k: _normalize(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_normalize(val) for val in v]
            if hasattr(v, "__dict__"):
                return _normalize(v.__dict__)
            return v

        payload = {
            "session_id": frame_output.session_id,
            "frame_idx": frame_output.frame_idx,
            "source_timestamp": frame_output.source_timestamp,
            "ingest_timestamp": frame_output.ingest_timestamp,
            "source_type": frame_output.source_type.value,
            "mode": frame_output.mode.value,
            "pose": _normalize(frame_output.pose),
            "cog": _normalize(frame_output.cog),
            "bos": _normalize(frame_output.bos),
            "cop": _normalize(frame_output.cop),
            "overlays": _normalize(frame_output.overlays),
            "source_flags": frame_output.source_flags,
        }

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self) -> None:
        return None
