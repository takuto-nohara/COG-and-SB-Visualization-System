from __future__ import annotations

from typing import Any, Optional, Mapping, Tuple

import cv2


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


def _as_shape(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    w = _as_float(value[0])
    h = _as_float(value[1])
    if w is None or h is None:
        return None
    return int(w), int(h)


def _pose_point(point: Any) -> Optional[Tuple[float, float, Optional[float]]]:
    if not isinstance(point, Mapping):
        return None
    x = _as_float(point.get("x"))
    y = _as_float(point.get("y"))
    if x is None or y is None:
        return None
    return x, y, _as_float(point.get("visibility"))


def _cop_point(value: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _as_float(value[0])
    y = _as_float(value[1])
    if x is None or y is None:
        return None
    return x, y


def draw_frame_overlay(frame, overlay):
    if overlay is None:
        return frame

    pose = overlay.get("pose") or {}
    if isinstance(pose, Mapping):
        landmarks = pose.get("landmarks") if isinstance(pose.get("landmarks"), list) else []
        shape = _as_shape(pose.get("shape"))
    else:
        landmarks = []
        shape = None

    if shape is not None:
        w, h = shape
    else:
        h, w = int(frame.shape[0]), int(frame.shape[1])

    # draw landmarks
    for idx, p in enumerate(landmarks):
        point = _pose_point(p)
        if point is None:
            continue
        x, y, vis = point
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
        pose_a = _pose_point(pa)
        pose_b = _pose_point(pb)
        if pose_a is None or pose_b is None:
            continue
        xa, ya, va = pose_a
        xb, yb, vb = pose_b
        if (va is not None and va < 0.2) or (vb is not None and vb < 0.2):
            continue
        p1 = (int(xa * w), int(ya * h))
        p2 = (int(xb * w), int(yb * h))
        cv2.line(frame, p1, p2, (0, 255, 255), 1)

    cop = overlay.get("cop") or {}
    if isinstance(cop, Mapping) and cop.get("cop"):
        cp = _cop_point(cop.get("cop"))
        if cp is not None:
            # world frame to image frame conversion (best effort fallback)
            cp0, cp1 = cp
            cx = int(cp0 * w + w / 2)
            cy = int(cp1 * h + h / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    return frame
