from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2

from cogsb.visualization import draw_frame_overlay


def load_overlay_json(path: str) -> Dict[int, dict]:
    overlays = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            idx = payload.get("frame_idx")
            if idx is None:
                continue
            overlays[int(idx)] = payload
    return overlays


def render_video(video_path: str, jsonl_path: str, output_path: Optional[str] = None, window_name: str = "cogsb"):
    overlays = load_overlay_json(jsonl_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    writer = None
    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 30.0
        fourcc_func = getattr(cv2, "VideoWriter_fourcc", None)
        if callable(fourcc_func):
            fourcc = int(fourcc_func(*"mp4v"))
        else:
            fourcc = int(ord("m") | (ord("p") << 8) | (ord("4") << 16) | (ord("v") << 24))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx in overlays:
            frame = draw_frame_overlay(frame, overlays[idx])

        if writer is not None:
            writer.write(frame)
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
