from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon


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


def _foot_contacts(pts: np.ndarray, dt: float, prev_pts: Optional[np.ndarray] = None) -> Tuple[float, float]:
    # key landmarks: heel/ankle/foot_index
    LANKLE, RANKLE = 27, 28
    LHEEL, RHEEL = 29, 30
    LFOOT, RFOOT = 31, 32

    left_h = pts[LHEEL][1]
    right_h = pts[RHEEL][1]
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
        mp = MultiPoint(points)
        hull = mp.convex_hull
        if isinstance(hull, Polygon):
            polygon = list(hull.exterior.coords)
            if polygon and polygon[0] == polygon[-1]:
                polygon = polygon[:-1]
            return [(float(x), float(y)) for x, y in polygon], float(hull.area)

    # fallback to line segment / empty polygon
    if len(points) == 2:
        return points, 0.0
    if len(points) == 1:
        return points, 0.0
    if prev_polygon and len(prev_polygon) >= 3:
        return prev_polygon, 0.0
    return [], 0.0


def compute_bos(world_landmarks: List[Tuple[float, float, float]], prev_landmarks: Optional[List[Tuple[float, float, float]]] = None, prev_polygon: Optional[List[Tuple[float, float]]] = None, dt: float = 1 / 30) -> Dict[str, object]:
    pts = np.array(world_landmarks, dtype=np.float64) if len(world_landmarks) >= 33 else None
    if pts is None or len(world_landmarks) < 33:
        return {
            "polygon": prev_polygon or [],
            "left_contact": 0.0,
            "right_contact": 0.0,
            "inside": False,
            "margin": None,
            "area": 0.0,
            "support_points": [],
        }

    prev = np.array(prev_landmarks, dtype=np.float64) if prev_landmarks is not None else None
    left_conf, right_conf = _foot_contacts(pts, dt, prev)

    foot_points = [
        (pts[27][0], pts[27][1]),
        (pts[29][0], pts[29][1]),
        (pts[31][0], pts[31][1]),
        (pts[28][0], pts[28][1]),
        (pts[30][0], pts[30][1]),
        (pts[32][0], pts[32][1]),
    ]

    active = []
    if left_conf >= 0.05:
        active.extend([(pts[27][0], pts[27][1]), (pts[29][0], pts[29][1]), (pts[31][0], pts[31][1])])
    if right_conf >= 0.05:
        active.extend([(pts[28][0], pts[28][1]), (pts[30][0], pts[30][1]), (pts[32][0], pts[32][1])])
    if not active:
        active = foot_points

    polygon_xy, area = _foot_polygon(active, prev_polygon)
    return {
        "polygon": polygon_xy,
        "left_contact": left_conf,
        "right_contact": right_conf,
        "inside": False,
        "margin": None,
        "area": area,
        "support_points": foot_points,
    }


def inside_bos(point_xy: Tuple[float, float], polygon_xy: List[Tuple[float, float]]) -> Tuple[bool, Optional[float]]:
    if len(polygon_xy) < 3:
        if len(polygon_xy) == 2:
            return False, None
        if not polygon_xy:
            return False, None
        d = float(np.hypot(point_xy[0] - polygon_xy[0][0], point_xy[1] - polygon_xy[0][1]))
        return d < 1e-6, 0.0

    poly = Polygon(polygon_xy)
    p = Point(point_xy)
    inside = poly.contains(p) or poly.touches(p)
    margin = float(poly.boundary.distance(p)) if inside else -float(poly.distance(p))
    return inside, margin
