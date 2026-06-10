from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np

from cogsb.reconstruction.body_segments import SEGMENT_TABLE


_POSE_EDGES = [
    (11, 13), (13, 15), (15, 17), (17, 19), (12, 14), (14, 16),
    (16, 18), (18, 20), (11, 12), (11, 23), (12, 24), (23, 25), (25, 27),
    (27, 29), (29, 31), (24, 26), (26, 28), (28, 30), (30, 32),
]

RENDER_MODE_OVERLAY = "overlay"
RENDER_MODE_SPACE3D = "space3d"
RENDER_MODE_OPTIONS = {RENDER_MODE_OVERLAY, RENDER_MODE_SPACE3D}


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


def _as_point3(value: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(value, Mapping):
        x = _as_float(value.get("x"))
        y = _as_float(value.get("y"))
        z = _as_float(value.get("z"))
        if x is None or y is None or z is None:
            return None
        return x, y, z
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    x = _as_float(value[0])
    y = _as_float(value[1])
    z = _as_float(value[2])
    if x is None or y is None or z is None:
        return None
    return x, y, z


def _as_point3_with_visibility(value: Any) -> Optional[Tuple[float, float, float, Optional[float]]]:
    point = _as_point3(value)
    if point is None:
        return None
    visibility = None
    if isinstance(value, Mapping):
        visibility = _as_float(value.get("visibility"))
    return point[0], point[1], point[2], visibility


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


def _collect_world_points_for_3d(
    overlay: Mapping[str, Any],
) -> list[Optional[Tuple[float, float, float, Optional[float]]]]:
    pose = overlay.get("pose") if isinstance(overlay.get("pose"), Mapping) else None
    if pose is None:
        return []

    raw_world = pose.get("world_landmarks")
    if isinstance(raw_world, list) and raw_world:
        points: list[Optional[Tuple[float, float, float, Optional[float]]]] = []
        for item in raw_world:
            point = _as_point3_with_visibility(item)
            points.append(point)
        return points

    raw_landmarks = pose.get("landmarks")
    points2 = []
    if isinstance(raw_landmarks, list):
        for item in raw_landmarks:
            point = _as_point(item)
            if point is None:
                points2.append(None)
                continue
            visibility = None
            if isinstance(item, Mapping):
                visibility = _as_float(item.get("visibility"))
            points2.append((point[0], point[1], 0.0, visibility))
    return points2


def _resolve_world_point(
    points: list[Optional[Tuple[float, float, float, Optional[float]]]],
    idx: int,
    *,
    require_visible: bool = False,
) -> Optional[Tuple[float, float, float]]:
    if idx < 0 or idx >= len(points):
        return None
    item = points[idx]
    if item is None:
        return None
    x, y, z, visibility = item
    if require_visible and visibility is not None and visibility < 0.2:
        return None
    return x, y, z


def _average_points(
    points: list[Optional[Tuple[float, float, float, Optional[float]]]],
    indices: Tuple[int, ...],
) -> Optional[Tuple[float, float, float]]:
    values = [p for p in (_resolve_world_point(points, idx, require_visible=False) for idx in indices) if p is not None]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())


def _segment_center_point(
    points: list[Optional[Tuple[float, float, float, Optional[float]]]],
    segment: Any,
) -> Optional[Tuple[float, float, float]]:
    start = _average_points(points, segment.start)
    end = _average_points(points, segment.end)
    if start is None or end is None:
        return None
    return (
        start[0] + segment.alpha * (end[0] - start[0]),
        start[1] + segment.alpha * (end[1] - start[1]),
        start[2] + segment.alpha * (end[2] - start[2]),
    )


def _rotate_point(point: Tuple[float, float, float], yaw: float, pitch: float) -> Tuple[float, float, float]:
    x, y, z = point
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x2 = x * cos_y + z * sin_y
    z2 = -x * sin_y + z * cos_y
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    y2 = y * cos_p - z2 * sin_p
    z2 = y * sin_p + z2 * cos_p
    return x2, y2, z2


def _scene_transform(points: list[Tuple[float, float, float]], width: int, height: int) -> Tuple[Tuple[float, float, float], float]:
    if not points:
        return (0.0, 0.0, 0.0), 1.0
    arr = np.asarray(points, dtype=np.float64)
    min_v = np.min(arr, axis=0)
    max_v = np.max(arr, axis=0)
    center = ((min_v + max_v) / 2.0).tolist()
    span = float(max(np.max(max_v - min_v), 1e-6))
    scale = min(width, height) * 0.35 / span
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (float(center[0]), float(center[1]), float(center[2])), scale


def _coerce_render_state(
    render_state: Optional[Mapping[str, Any]],
    key: str,
    default: float,
) -> float:
    value = default
    if render_state is not None:
        raw = render_state.get(key)
        parsed = _as_float(raw)
        if parsed is not None:
            value = parsed
    return value


def _project_scene_point(
    point: Tuple[float, float, float],
    width: int,
    height: int,
    center: Tuple[float, float, float],
    scale: float,
    yaw_deg: float,
    pitch_deg: float,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> Tuple[Tuple[int, int], float]:
    cx, cy, cz = center
    x, y, z = point
    x -= cx
    y -= cy
    z -= cz

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    x2, y2, z2 = _rotate_point((x, y, z), yaw, pitch)

    cx_px = width * 0.5
    cy_px = height * 0.5
    px = cx_px + (x2 + z2 * 0.35) * (scale * zoom) + pan_x
    py = cy_px + (y2 - z2 * 0.20) * (scale * zoom) + pan_y
    return (
        (
            max(0, min(width - 1, int(round(px)))),
            max(0, min(height - 1, int(round(py)))),
        ),
        float(z2),
    )


def _segment_color(index: int) -> Tuple[int, int, int]:
    palette = [
        (0, 180, 255),
        (180, 0, 255),
        (255, 102, 0),
        (64, 255, 64),
        (255, 255, 0),
        (0, 255, 255),
        (255, 150, 150),
    ]
    return palette[index % len(palette)]


def _draw_frame_space3d(frame, overlay: Mapping[str, Any], render_state: Optional[Mapping[str, Any]] = None):
    if overlay is None:
        return frame

    frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
    scene = np.full((frame_h, frame_w, 3), 18, dtype=np.uint8)

    yaw = _coerce_render_state(render_state, "yaw", -40.0)
    pitch = _coerce_render_state(render_state, "pitch", 22.0)
    zoom = _coerce_render_state(render_state, "zoom", 1.0)
    pan_x = _coerce_render_state(render_state, "pan_x", 0.0)
    pan_y = _coerce_render_state(render_state, "pan_y", 0.0)
    zoom = max(0.08, min(6.0, zoom))

    world_points = _collect_world_points_for_3d(overlay)
    if not any(point is not None for point in world_points):
        return frame

    visible_points = [
        (point[0], point[1], point[2])
        for point in world_points
        if point is not None and (point[3] is None or point[3] >= 0.2)
    ]

    # Add COG / COP / BOS for scene context
    cog = overlay.get("cog")
    if isinstance(cog, Mapping):
        cog_xy = _as_point(cog.get("point") or cog.get("cog"))
        if cog_xy is not None:
            visible_points.append((cog_xy[0], cog_xy[1], 0.0))

    cop = overlay.get("cop")
    if isinstance(cop, Mapping):
        cop_xy = _as_point(cop.get("cop"))
        if cop_xy is not None:
            visible_points.append((cop_xy[0], cop_xy[1], 0.0))

    bos = overlay.get("bos")
    if isinstance(bos, Mapping):
        for raw_point in bos.get("polygon", []) or []:
            point = _as_point(raw_point)
            if point is None:
                continue
            visible_points.append((point[0], point[1], 0.0))

    for segment in SEGMENT_TABLE:
        center_point = _segment_center_point(world_points, segment)
        if center_point is not None:
            visible_points.append(center_point)

    if not visible_points:
        return frame

    center, scale = _scene_transform(visible_points, frame_w, frame_h)

    def project(point: Tuple[float, float, float]) -> Tuple[Tuple[int, int], float]:
        return _project_scene_point(
            point,
            frame_w,
            frame_h,
            center,
            scale,
            yaw_deg=yaw,
            pitch_deg=pitch,
            zoom=zoom,
            pan_x=pan_x,
            pan_y=pan_y,
        )

    # Axis
    axis_scale = max(1.0, min(frame_w, frame_h) / 6.0) / scale
    axes = [
        ((-axis_scale, 0.0, 0.0), (axis_scale, 0.0, 0.0), (255, 80, 80)),
        ((0.0, -axis_scale, 0.0), (0.0, axis_scale, 0.0), (80, 255, 80)),
        ((0.0, 0.0, -axis_scale), (0.0, 0.0, axis_scale), (80, 80, 255)),
    ]
    for axis_start, axis_end, color in axes:
        p0, z0 = project(axis_start)
        p1, z1 = project(axis_end)
        cv2.line(scene, p0, p1, color, 1, cv2.LINE_AA)
        cv2.putText(scene, str(abs(round(axis_scale))), (p1[0] + 2, p1[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Segments (far to near)
    segments_draw: list[tuple[float, Tuple[int, int], Tuple[int, int], Tuple[int, int, int], int]] = []
    for idx, segment in enumerate(SEGMENT_TABLE):
        start = _average_points(world_points, segment.start)
        end = _average_points(world_points, segment.end)
        if start is None or end is None:
            continue
        p0, z0 = project(start)
        p1, z1 = project(end)
        segments_draw.append((0.5 * (z0 + z1), p0, p1, _segment_color(idx), 2 if segment.mass >= 6.0 else 1))

    segments_draw.sort(key=lambda item: item[0])
    for _, p0, p1, color, thickness in segments_draw:
        cv2.line(scene, p0, p1, color, thickness, cv2.LINE_AA)

    # Segment centroids
    for segment in SEGMENT_TABLE:
        center_point = _segment_center_point(world_points, segment)
        if center_point is None:
            continue
        p, _ = project(center_point)
        cv2.circle(scene, p, 4, (255, 255, 255), 1, cv2.LINE_AA)

    # Landmarks
    for item in world_points:
        if item is None:
            continue
        x, y, z, visibility = item
        if visibility is not None and visibility < 0.2:
            continue
        p, depth = project((x, y, z))
        radius = max(1, min(4, int(round(3.5 - 0.1 * max(0.0, depth)))))
        cv2.circle(scene, p, radius, (230, 230, 230), -1, cv2.LINE_AA)

    # COG
    if isinstance(cog, Mapping):
        cog_xy = _as_point(cog.get("point") or cog.get("cog"))
        if cog_xy is not None:
            c, _ = project((cog_xy[0], cog_xy[1], 0.0))
            cv2.circle(scene, c, 7, (0, 180, 255), -1, cv2.LINE_AA)
            conf = _as_float(cog.get("confidence"))
            if conf is not None:
                cv2.putText(scene, f"COG {conf:.2f}", (c[0] + 8, max(0, c[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)

    # COP
    if isinstance(cop, Mapping):
        cop_xy = _as_point(cop.get("cop"))
        if cop_xy is not None:
            c, _ = project((cop_xy[0], cop_xy[1], 0.0))
            cv2.circle(scene, c, 5, (255, 255, 0), -1, cv2.LINE_AA)

    # BOS
    if isinstance(bos, Mapping):
        bos_points = []
        for raw_point in bos.get("polygon", []) or []:
            point = _as_point(raw_point)
            if point is None:
                continue
            bos_points.append(project((point[0], point[1], 0.0))[0])

        if len(bos_points) >= 3:
            pts = np.array(bos_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(scene, [pts], isClosed=True, color=(120, 255, 120), thickness=2, lineType=cv2.LINE_AA)

    cv2.putText(scene, "3D Mode", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (170, 170, 170), 1, cv2.LINE_AA)
    return scene


def draw_frame_overlay(
    frame,
    overlay,
    render_mode: str = RENDER_MODE_OVERLAY,
    render_state: Optional[Mapping[str, Any]] = None,
):
    if render_mode not in RENDER_MODE_OPTIONS:
        render_mode = RENDER_MODE_OVERLAY
    if render_mode == RENDER_MODE_SPACE3D:
        return _draw_frame_space3d(frame, overlay, render_state=render_state)
    return _draw_frame_overlay_2d(frame, overlay)


def _draw_frame_overlay_2d(frame, overlay):
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

    cop = overlay.get("cop") or {}
    if isinstance(cop, Mapping):
        cp = _cop_point(cop.get("cop"))
        if cp is not None:
            cp0, cp1 = cp
            cx, cy = _world_to_pixel_mapped((cp0, cp1), w, h, world_transform, world_bounds)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    return frame
