from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


_POSE_EDGES = [
    (11, 13), (13, 15), (15, 17), (17, 19), (12, 14), (14, 16),
    (16, 18), (18, 20), (11, 12), (11, 23), (12, 24), (23, 25), (25, 27),
    (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32),
]


def _as_float(v: object) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def draw_frame_overlay(frame, overlay):
    if overlay is None:
        return frame

    pose = overlay.get("pose") or {}
    landmarks = pose.get("landmarks") or []
    shape = pose.get("shape", [frame.shape[1], frame.shape[0]])
    if len(shape) >= 2:
        w, h = int(shape[0]), int(shape[1])
    else:
        h, w = frame.shape[:2]

    # draw landmarks
    for idx, p in enumerate(landmarks):
        x = _as_float(p.get("x"))
        y = _as_float(p.get("y"))
        vis = _as_float(p.get("visibility"))
        if x is None or y is None:
            continue
        if vis is not None and vis < 0.2:
            continue
        px = int(min(max(0.0, x * w), max(1.0, w - 1)))
        py = int(min(max(0.0, y * h), max(1.0, h - 1)))
        cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
        if idx < 10:
            cv2.circle(frame, (px, py), 3, (255, 0, 0), 1)

    for a, b in _POSE_EDGES:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        pa = landmarks[a]
        pb = landmarks[b]
        xa = _as_float(pa.get("x"))
        ya = _as_float(pa.get("y"))
        xb = _as_float(pb.get("x"))
        yb = _as_float(pb.get("y"))
        va = _as_float(pa.get("visibility"))
        vb = _as_float(pb.get("visibility"))
        if None in (xa, ya, xb, yb):
            continue
        if (va is not None and va < 0.2) or (vb is not None and vb < 0.2):
            continue
        p1 = (int(xa * w), int(ya * h))
        p2 = (int(xb * w), int(yb * h))
        cv2.line(frame, p1, p2, (0, 255, 255), 1)

    cop = overlay.get("cop") or {}
    if cop.get("cop"):
        cp = cop.get("cop")
        if isinstance(cp, list) and len(cp) >= 2 and isinstance(cp[0], (int, float)) and isinstance(cp[1], (int, float)):
            # world frame to image frame conversion (best effort fallback)
            cx = int(cp[0] * w + w / 2)
            cy = int(cp[1] * h + h / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    return frame
