from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Optional, Tuple, TypedDict, cast

import numpy as np

try:
    from shapely.geometry import MultiPoint as _ShapelyMultiPoint  # type: ignore[import-not-found]
    from shapely.geometry import Point as _ShapelyPoint  # type: ignore[import-not-found]
    from shapely.geometry import Polygon as _ShapelyPolygon  # type: ignore[import-not-found]
    from shapely.errors import TopologicalError  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    _ShapelyMultiPoint: Any | None = None
    _ShapelyPoint: Any | None = None
    _ShapelyPolygon: Any | None = None
    TopologicalError = Exception
    _SHAPELY_AVAILABLE = False
else:
    _SHAPELY_AVAILABLE = True


class BOSResult(TypedDict):
    polygon: List[Tuple[float, float]]
    polygon_world: List[Tuple[float, float, float]]
    left_contact: float
    right_contact: float
    inside: bool
    margin: Optional[float]
    area: float
    support_point_xy: Tuple[float, float]
    support_point_world: Tuple[float, float, float]
    support_points: List[Tuple[float, float, float]]


SupportFrame = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return 0.0, 0.0
    xs, ys = zip(*points)
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        return 0.0, 0.0
    return float(sum(xs) / len(points)), float(sum(ys) / len(points))


def _sort_polygon_ccw(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points[:]
    cx, cy = _polygon_centroid(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _match_vertices(
    points: List[Tuple[float, float]],
    reference: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    if not points or not reference or len(points) != len(reference):
        return points[:]

    used = [False] * len(reference)
    matched: list[Optional[tuple[float, float]]] = [None] * len(reference)
    for px, py in points:
        best_idx = -1
        best_distance = float("inf")
        for j, (rx, ry) in enumerate(reference):
            if used[j]:
                continue
            d = (px - rx) * (px - rx) + (py - ry) * (py - ry)
            if d < best_distance:
                best_distance = d
                best_idx = j
        if best_idx < 0:
            return points[:]
        used[best_idx] = True
        matched[best_idx] = (px, py)

    if any(point is None for point in matched):
        return points[:]

    return [point for point in matched if point is not None]


def _stabilize_polygon(
    polygon: List[Tuple[float, float]],
    prev_polygon: List[Tuple[float, float]],
    *,
    blend_ratio: float = 0.25,
) -> List[Tuple[float, float]]:
    if len(polygon) < 3 or len(prev_polygon) < 3:
        return polygon
    if len(polygon) != len(prev_polygon):
        return polygon

    ordered_current = _sort_polygon_ccw(polygon)
    ordered_prev = _sort_polygon_ccw(prev_polygon)

    curr_area = _polygon_area(ordered_current)
    prev_area = _polygon_area(ordered_prev)
    if curr_area <= 0.0 or prev_area <= 0.0:
        return ordered_current

    curr_cx, curr_cy = _polygon_centroid(ordered_current)
    prev_cx, prev_cy = _polygon_centroid(ordered_prev)
    shift = float(np.hypot(curr_cx - prev_cx, curr_cy - prev_cy))
    if not np.isfinite(shift):
        return ordered_current

    stable_ratio = curr_area / max(prev_area, 1e-12)
    scale = math.sqrt(max(prev_area, 1e-12))
    blend = float(np.clip(blend_ratio, 0.0, 1.0))
    if shift > 0.06:
        blend = 0.0
    elif shift > 0.03:
        blend = min(blend, 0.15)
    elif stable_ratio < 0.2 or stable_ratio > 2.5:
        blend = min(blend, 0.1)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    if shift > 0.25 * scale:
        blend = min(blend, 0.2)

    if blend <= 0.0:
        return ordered_prev
    if blend >= 1.0:
        return ordered_current

    ordered_prev = _match_vertices(ordered_prev, ordered_current)
    if len(ordered_prev) != len(ordered_current):
        return ordered_current
    return [
        ((1.0 - blend) * curr_x + blend * prev_x, (1.0 - blend) * curr_y + blend * prev_y)
        for (curr_x, curr_y), (prev_x, prev_y) in zip(ordered_current, ordered_prev)
    ]


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) <= 1:
        return points[:]
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return hull


def _to_float_pair(point: np.ndarray) -> Optional[Tuple[float, float]]:
    if point.shape != (3,):
        return None
    return float(point[0]), float(point[1])


def _normalize_vec(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return vec / norm


def _blend_support_frame(
    frame: SupportFrame,
    prev_frame: SupportFrame,
    blend_ratio: float,
) -> SupportFrame:
    if blend_ratio <= 0.0:
        return prev_frame
    if blend_ratio >= 1.0:
        return frame

    alpha = float(np.clip(blend_ratio, 0.0, 1.0))
    prev_origin, prev_x_axis, prev_y_axis, prev_up = prev_frame
    origin, x_axis, y_axis, up = frame

    # Keep axis orientation consistent with previous frame.
    if np.dot(up, prev_up) < 0.0:
        up = -up
    x_axis = x_axis - np.dot(x_axis, up) * up
    normalized_x = _normalize_vec(x_axis)
    if normalized_x is not None:
        x_axis = normalized_x
    if np.dot(x_axis, prev_x_axis) < 0.0:
        x_axis = -x_axis
    y_axis = np.cross(up, x_axis)
    normalized_y = _normalize_vec(y_axis)
    if normalized_y is not None:
        y_axis = normalized_y
    if np.dot(y_axis, prev_y_axis) < 0.0:
        y_axis = -y_axis

    origin = (1.0 - alpha) * prev_origin + alpha * origin
    blended_up = _normalize_vec((1.0 - alpha) * prev_up + alpha * up)
    if blended_up is None:
        blended_up = prev_up
    blended_x = _normalize_vec((1.0 - alpha) * prev_x_axis + alpha * x_axis)
    if blended_x is None:
        blended_x = prev_x_axis
    blended_y = _normalize_vec(np.cross(blended_up, blended_x))
    if blended_y is None:
        blended_y = prev_y_axis
    if np.dot(blended_y, prev_y_axis) < 0.0:
        blended_y = -blended_y

    return origin, blended_x, blended_y, blended_up


def _estimate_support_frame(
    points: List[Tuple[float, float, float]],
    prev_frame: Optional[SupportFrame] = None,
    blend_ratio: float = 0.25,
) -> Optional[SupportFrame]:
    if len(points) < 3:
        return None

    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 3:
        return None
    if not np.isfinite(arr).all():
        return None

    # Robust 2D frame on the estimated floor/contact plane:
    # - origin: centroid of supporting landmarks
    # - up: smallest-variance eigen-direction
    # - x/y: orthonormal basis on the plane
    origin = arr.mean(axis=0)
    centered = arr - origin
    if not np.isfinite(centered).all():
        return None

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape != (3, 3) or singular_values.shape[0] < 3:
        return None
    planarity = float(singular_values[-1] / (float(singular_values.sum()) + 1e-12))
    if not np.isfinite(planarity):
        return None
    up = vh[-1]
    up = _normalize_vec(up)
    if up is None:
        return None

    # Choose x direction from the dominant in-plane PCA direction.
    x_axis = _normalize_vec(vh[0] - np.dot(vh[0], up) * up)
    if x_axis is None:
        return None
    y_axis = np.cross(up, x_axis)
    y_axis = _normalize_vec(y_axis)
    if y_axis is None:
        return None

    frame = (origin, x_axis, y_axis, up)
    if prev_frame is None:
        return frame

    if planarity > 0.30:
        return prev_frame
    dynamic_blend_ratio = blend_ratio
    if planarity > 0.22:
        dynamic_blend_ratio = 0.0
    elif planarity > 0.18:
        dynamic_blend_ratio = 0.05
    elif planarity > 0.14:
        dynamic_blend_ratio = 0.12
    elif planarity > 0.10:
        dynamic_blend_ratio = min(0.20, blend_ratio)
    else:
        dynamic_blend_ratio = min(0.28, max(0.12, blend_ratio))

    return _blend_support_frame(frame, prev_frame, dynamic_blend_ratio)


def _project_point_to_support(point: Tuple[float, float, float], frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[float, float]:
    if frame is None:
        return float(point[0]), float(point[1])
    origin, x_axis, y_axis, _ = frame
    rel = np.asarray(point, dtype=np.float64) - origin
    return float(np.dot(rel, x_axis)), float(np.dot(rel, y_axis))


def _project_points_to_support(points: List[Tuple[float, float, float]], frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> List[Tuple[float, float]]:
    return [_project_point_to_support(point, frame) for point in points]


def _local_point_to_world(point_xy: Tuple[float, float], frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[float, float, float]:
    if frame is None:
        return point_xy[0], point_xy[1], 0.0
    origin, x_axis, y_axis, _ = frame
    x, y = point_xy
    return (
        float(origin[0] + x * x_axis[0] + y * y_axis[0]),
        float(origin[1] + x * x_axis[1] + y * y_axis[1]),
        float(origin[2] + x * x_axis[2] + y * y_axis[2]),
    )


def _height_along_support(point: Tuple[float, float, float], frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> float:
    if frame is None:
        return float(point[1])
    origin, _, _, up = frame
    return float(np.dot(np.asarray(point, dtype=np.float64) - origin, up))


def _point_to_segment_distance(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = a
    x2, y2 = b
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0.0:
        return float(np.hypot(wx, wy))
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return float(np.hypot(px - proj_x, py - proj_y))


def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        if intersects:
            inside = not inside
    return inside


def _project_point_to_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not polygon:
        return point
    if len(polygon) == 1:
        return polygon[0]
    if len(polygon) == 2:
        return polygon[0]

    best_point = polygon[0]
    best_distance = _point_to_segment_distance(point, polygon[0], polygon[1])
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        d = _point_to_segment_distance(point, a, b)
        if d <= best_distance:
            best_distance = d
            x1, y1 = a
            x2, y2 = b
            vx, vy = x2 - x1, y2 - y1
            wx, wy = point[0] - x1, point[1] - y1
            seg_len2 = vx * vx + vy * vy
            if seg_len2 == 0.0:
                best_point = a
            else:
                t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
                best_point = (x1 + t * vx, y1 + t * vy)

    return best_point


@dataclass
class BOSInput:
    world_landmarks: List[Tuple[float, float, float]]
    prev_left: float = 0.0
    prev_right: float = 0.0
    dt: float = 1 / 30


def _norm(v: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return min(1.0, max(0.0, (v - lower) / (upper - lower)))


def _foot_contacts(
    pts: np.ndarray,
    dt: float,
    prev_pts: Optional[np.ndarray] = None,
    support_frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[float, float]:
    # key landmarks: heel/ankle/foot_index
    LANKLE, RANKLE = 27, 28
    LHEEL, RHEEL = 29, 30
    LFOOT, RFOOT = 31, 32

    left_h = _height_along_support((float(pts[LHEEL][0]), float(pts[LHEEL][1]), float(pts[LHEEL][2])), support_frame)
    right_h = _height_along_support((float(pts[RHEEL][0]), float(pts[RHEEL][1]), float(pts[RHEEL][2])), support_frame)
    base = min(left_h, right_h)
    h0 = base + 0.08

    # normalized heights against ankles; normalized for numerical stability
    left_score = _norm(h0 - left_h, -0.02, 0.08)
    right_score = _norm(h0 - right_h, -0.02, 0.08)

    # low speed means stable contact
    if prev_pts is None:
        return left_score, right_score

    l_speed = float(np.linalg.norm(pts[LANKLE] - prev_pts[LANKLE])) / max(dt, 1e-3)
    r_speed = float(np.linalg.norm(pts[RANKLE] - prev_pts[RANKLE])) / max(dt, 1e-3)
    lv = _norm(0.15 - l_speed, -0.1, 0.3)
    rv = _norm(0.15 - r_speed, -0.1, 0.3)
    return 0.7 * left_score + 0.3 * lv, 0.7 * right_score + 0.3 * rv


def _foot_polygon(points: List[Tuple[float, float]], prev_polygon: Optional[List[Tuple[float, float]]]) -> Tuple[List[Tuple[float, float]], float]:
    if len(points) >= 3:
        if _SHAPELY_AVAILABLE:
            try:
                if _ShapelyMultiPoint is not None and callable(_ShapelyMultiPoint):
                    mp = cast(Any, _ShapelyMultiPoint)(points)
                    hull = mp.convex_hull
                    if hasattr(hull, "exterior"):
                        hull_polygon = cast(Any, hull)
                        polygon_raw = list(cast(Any, hull_polygon.exterior.coords))
                        if polygon_raw and polygon_raw[0] == polygon_raw[-1]:
                            polygon_raw = polygon_raw[:-1]
                        current_polygon = [(float(x), float(y)) for x, y in polygon_raw]
                        if prev_polygon is not None:
                            current_polygon = _stabilize_polygon(current_polygon, prev_polygon, blend_ratio=0.25)
                        return current_polygon, _polygon_area(current_polygon)
            except Exception:
                pass
        fallback_polygon = _convex_hull(points)
        if len(fallback_polygon) >= 3:
            current_polygon = fallback_polygon
            if prev_polygon is not None:
                current_polygon = _stabilize_polygon(current_polygon, prev_polygon, blend_ratio=0.25)
            return current_polygon, _polygon_area(current_polygon)

    # fallback to line segment / empty polygon
    if len(points) == 2:
        return points, 0.0
    if len(points) == 1:
        return points, 0.0
    if prev_polygon and len(prev_polygon) >= 3:
        return prev_polygon, 0.0
    return [], 0.0


def compute_bos(
    world_landmarks: List[Tuple[float, float, float]],
    prev_landmarks: Optional[List[Tuple[float, float, float]]] = None,
    prev_polygon: Optional[List[Tuple[float, float]]] = None,
    dt: float = 1 / 30,
    cog_point: Optional[Tuple[float, float, float]] = None,
    prev_support_frame: Optional[SupportFrame] = None,
    return_support_frame: bool = False,
) -> Tuple[BOSResult, Optional[SupportFrame]]:
    pts = np.array(world_landmarks, dtype=np.float64) if len(world_landmarks) >= 33 else None
    support_frame: Optional[SupportFrame] = None
    if pts is None or len(world_landmarks) < 33:
        result = BOSResult(
            polygon=prev_polygon or [],
            polygon_world=[],
            left_contact=0.0,
            right_contact=0.0,
            inside=False,
            margin=None,
            area=0.0,
            support_point_xy=(0.0, 0.0),
            support_point_world=(0.0, 0.0, 0.0),
            support_points=[],
        )
        return result, None

    prev = np.array(prev_landmarks, dtype=np.float64) if prev_landmarks is not None else None
    support_frame = _estimate_support_frame([
        (float(pts[27][0]), float(pts[27][1]), float(pts[27][2])),
        (float(pts[28][0]), float(pts[28][1]), float(pts[28][2])),
        (float(pts[29][0]), float(pts[29][1]), float(pts[29][2])),
        (float(pts[30][0]), float(pts[30][1]), float(pts[30][2])),
        (float(pts[31][0]), float(pts[31][1]), float(pts[31][2])),
        (float(pts[32][0]), float(pts[32][1]), float(pts[32][2])),
    ], prev_frame=prev_support_frame)

    left_conf, right_conf = _foot_contacts(pts, dt, prev, support_frame)

    foot_points = [
        (float(pts[27][0]), float(pts[27][1]), float(pts[27][2])),
        (float(pts[29][0]), float(pts[29][1]), float(pts[29][2])),
        (float(pts[31][0]), float(pts[31][1]), float(pts[31][2])),
        (float(pts[28][0]), float(pts[28][1]), float(pts[28][2])),
        (float(pts[30][0]), float(pts[30][1]), float(pts[30][2])),
        (float(pts[32][0]), float(pts[32][1]), float(pts[32][2])),
    ]
    active = []
    if left_conf >= 0.05:
        active.extend([
            (float(pts[27][0]), float(pts[27][1]), float(pts[27][2])),
            (float(pts[29][0]), float(pts[29][1]), float(pts[29][2])),
            (float(pts[31][0]), float(pts[31][1]), float(pts[31][2])),
        ])
    if right_conf >= 0.05:
        active.extend([
            (float(pts[28][0]), float(pts[28][1]), float(pts[28][2])),
            (float(pts[30][0]), float(pts[30][1]), float(pts[30][2])),
            (float(pts[32][0]), float(pts[32][1]), float(pts[32][2])),
        ])
    if not active:
        active = foot_points

    support_points_xy = _project_points_to_support(active, support_frame)
    polygon_xy, area = _foot_polygon(support_points_xy, prev_polygon)
    polygon_world = [
        _local_point_to_world(point_xy, support_frame) for point_xy in polygon_xy
    ] if support_frame is not None else []

    support_point = (float(pts[0][0]), float(pts[0][1]), float(pts[0][2]))
    if cog_point is not None:
        try:
            candidate_point = (float(cog_point[0]), float(cog_point[1]), float(cog_point[2]))
            if np.isfinite(candidate_point).all():
                support_point = candidate_point
        except (TypeError, ValueError, IndexError):
            pass
    support_point_xy = _project_point_to_support(support_point, support_frame)
    if support_frame is not None:
        projected_support_point_world = _local_point_to_world(support_point_xy, support_frame)
        support_point_world = support_point
        if np.isfinite(projected_support_point_world).all():
            delta = np.linalg.norm(
                np.asarray(projected_support_point_world, dtype=np.float64)
                - np.asarray(support_point, dtype=np.float64)
            )
            if np.isfinite(delta) and delta <= 0.15:
                support_point_world = projected_support_point_world
    else:
        support_point_world = support_point

    inside, margin = inside_bos(support_point_xy, polygon_xy)
    result = BOSResult(
        polygon=polygon_xy,
        polygon_world=polygon_world,
        left_contact=left_conf,
        right_contact=right_conf,
        inside=inside,
        margin=margin,
        area=area,
        support_point_xy=support_point_xy,
        support_point_world=support_point_world,
        support_points=foot_points,
    )
    if return_support_frame:
        return result, support_frame
    return result, support_frame


def inside_bos(point_xy: Tuple[float, float], polygon_xy: List[Tuple[float, float]]) -> Tuple[bool, Optional[float]]:
    if len(polygon_xy) < 3:
        if len(polygon_xy) == 2:
            return False, None
        if not polygon_xy:
            return False, None
        d = float(np.hypot(point_xy[0] - polygon_xy[0][0], point_xy[1] - polygon_xy[0][1]))
        return d < 1e-6, 0.0

    if _SHAPELY_AVAILABLE:
        try:
            if _ShapelyPolygon is not None and _ShapelyPoint is not None:
                if callable(_ShapelyPolygon) and callable(_ShapelyPoint):
                    poly = cast(Any, _ShapelyPolygon)(polygon_xy)
                    p = cast(Any, _ShapelyPoint)(point_xy)
                    inside = poly.contains(p) or poly.touches(p)
                    margin = float(poly.boundary.distance(p)) if inside else -float(poly.distance(p))
                    return inside, margin
        except TopologicalError:
            pass

    inside = _point_in_polygon(point_xy, polygon_xy)
    margin = min(
        _point_to_segment_distance(point_xy, polygon_xy[i], polygon_xy[(i + 1) % len(polygon_xy)])
        for i in range(len(polygon_xy))
    )
    if not inside:
        margin = -(abs(margin) if margin is not None else 0.0)
    return inside, margin
