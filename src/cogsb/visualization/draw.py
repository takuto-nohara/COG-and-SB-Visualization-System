from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

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


def _as_shape(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    w = _as_float(value[0])
    h = _as_float(value[1])
    if w is None or h is None:
        return None
    return int(w), int(h)


def _as_point(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, Mapping):
        x = _as_float(value.get("x"))
        y = _as_float(value.get("y"))
        if x is None or y is None:
            return None
        return x, y
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _as_float(value[0])
    y = _as_float(value[1])
    if x is None or y is None:
        return None
    return x, y


def _world_to_pixel(point: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    x, y = point
    px = int(round((x * width) + (width / 2)))
    py = int(round((y * height) + (height / 2)))
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    return px, py


def _world_to_pixel_norm(
    point: Tuple[float, float],
    width: int,
    height: int,
    bounds: Optional[Tuple[float, float, float, float]],
) -> Tuple[int, int]:
    x, y = point
    if bounds is None:
        return _world_to_pixel(point, width, height)

    min_x, max_x, min_y, max_y = bounds
    x_range = max_x - min_x
    y_range = max_y - min_y
    if x_range <= 0:
        x_range = 1e-9
    if y_range <= 0:
        y_range = 1e-9

    xn = (x - min_x) / x_range
    yn = (y - min_y) / y_range
    xn = max(0.0, min(1.0, xn))
    yn = max(0.0, min(1.0, yn))

    px = int(round(xn * (width - 1)))
    py = int(round(yn * (height - 1)))
    return px, py


def _fit_world_to_pixel_affine(
    pose_landmarks: Any,
    world_landmarks: Any,
    width: int,
    height: int,
    min_points: int = 3,
) -> Optional[np.ndarray]:
    if not isinstance(pose_landmarks, list) or not isinstance(world_landmarks, list):
        return None

    src = []
    dst = []
    limit = min(len(pose_landmarks), len(world_landmarks))
    for idx in range(limit):
        pose_pt = _pose_point(pose_landmarks[idx])
        world_pt = _pose_point(world_landmarks[idx])
        if pose_pt is None or world_pt is None:
            continue
        x, y, vis = pose_pt
        if vis is not None and vis < 0.2:
            continue
        wx, wy, _ = world_pt
        dst.append([x * width, y * height])
        src.append([wx, wy])

    if len(src) < min_points:
        return None

    src_np = np.asarray(src, dtype=np.float32)
    dst_np = np.asarray(dst, dtype=np.float32)

    matrix, _ = cv2.estimateAffinePartial2D(
        src_np,
        dst_np,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=1000,
        confidence=0.99,
    )
    if matrix is not None:
        return matrix

    matrix, _ = cv2.estimateAffinePartial2D(src_np, dst_np)
    return matrix


def _world_to_pixel_mapped(
    point: Tuple[float, float],
    width: int,
    height: int,
    transform: Optional[np.ndarray],
    bounds: Optional[Tuple[float, float, float, float]],
) -> Tuple[int, int]:
    if transform is not None:
        x, y = point
        try:
            mapped = np.dot(transform, np.array([x, y, 1.0], dtype=np.float64))
            px = int(round(float(mapped[0])))
            py = int(round(float(mapped[1])))
            return max(0, min(width - 1, px)), max(0, min(height - 1, py))
        except Exception:
            pass

    return _world_to_pixel_norm(point, width, height, bounds)


def _world_bounds(points: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(points, list) or not points:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for raw in points:
        if isinstance(raw, Mapping):
            vis = _as_float(raw.get("visibility"))
            if vis is not None and vis < 0.2:
                continue
        p = _as_point(raw)
        if p is None:
            continue
        xs.append(p[0])
        ys.append(p[1])

    if not xs or not ys:
        return None

    if len(xs) >= 12:
        x_min = float(np.quantile(xs, 0.05))
        x_max = float(np.quantile(xs, 0.95))
        y_min = float(np.quantile(ys, 0.05))
        y_max = float(np.quantile(ys, 0.95))
    else:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

    if x_min >= x_max:
        x_min = x_max - 1e-9
    if y_min >= y_max:
        y_min = y_max - 1e-9

    return x_min, x_max, y_min, y_max


def _pose_point(point: Any) -> Optional[Tuple[float, float, Optional[float]]]:
    if not isinstance(point, Mapping):
        return None
    x = _as_float(point.get("x"))
    y = _as_float(point.get("y"))
    if x is None or y is None:
        return None
    return x, y, _as_float(point.get("visibility"))


def _cop_point(value: Any) -> Optional[Tuple[float, float]]:
    return _as_point(value)


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
    world_points = []
    if isinstance(overlay, Mapping):
        world_points = overlay.get("pose", {}).get("world_landmarks", [])
    world_bounds = _world_bounds(world_points)
    world_transform = _fit_world_to_pixel_affine(landmarks, world_points, w, h)

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

    # COG
    cog = overlay.get("cog")
    if isinstance(cog, Mapping):
        cog_xy = _as_point(cog.get("point") or cog.get("cog"))
    else:
        cog_xy = _as_point(cog)
    if cog_xy is not None:
        cx, cy = _world_to_pixel_mapped(cog_xy, w, h, world_transform, world_bounds)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(frame, (cx, cy), 6, (0, 200, 255), -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
            conf = _as_float(cog.get("confidence")) if isinstance(cog, Mapping) else None
            if conf is not None:
                cv2.putText(
                    frame,
                    f"COG conf={conf:.2f}",
                    (cx + 8, max(0, cy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 255),
                    1,
                )

    # BOS
    bos = overlay.get("bos")
    if isinstance(bos, Mapping):
        polygon = []
        for raw_point in bos.get("polygon", []) or []:
            p = _as_point(raw_point)
            if p is None:
                continue
            polygon.append(_world_to_pixel_mapped(p, w, h, world_transform, world_bounds))

        if len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
            inside = bos.get("inside") if "inside" in bos else bos.get("inside_cog")
            if inside is not None:
                text = "BOS (in)" if bool(inside) else "BOS (out)"
                cv2.putText(
                    frame,
                    text,
                    (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 200, 0),
                    2,
                )

        elif len(polygon) == 2:
            cv2.line(frame, polygon[0], polygon[1], (255, 200, 0), 2)

        area = _as_float(bos.get("support_area"))
        if area is not None and area >= 0:
            cv2.putText(
                frame,
                f"area={area:.4f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                1,
            )

    # COP
    cop = overlay.get("cop") or {}
    if isinstance(cop, Mapping):
        cp = _cop_point(cop.get("cop"))
        if cp is not None:
            cp0, cp1 = cp
            cx, cy = _world_to_pixel_mapped((cp0, cp1), w, h, world_transform, world_bounds)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    return frame
