from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np

try:
    from shapely.geometry import Point as _SHAPELY_POINT  # type: ignore[import-not-found]
    from shapely.geometry import Polygon as _SHAPELY_POLYGON  # type: ignore[import-not-found]
    from shapely.ops import nearest_points as _SHAPELY_NEAREST_POINTS  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover
    _SHAPELY_POINT: Any | None = None
    _SHAPELY_POLYGON: Any | None = None
    _SHAPELY_NEAREST_POINTS: Any | None = None
    _SHAPELY_AVAILABLE = False
else:
    _SHAPELY_AVAILABLE = True

from cogsb.core.types import COPState


@dataclass
class COPInput:
    cog: Tuple[float, float, float]
    acceleration: Tuple[float, float, float]
    mass_kg: float
    polygon: List[Tuple[float, float]]
    prev_cop: Optional[Tuple[float, float]] = None
    friction_mu: float = 0.7
    gravity: float = 9.80665


class COPEstimator:
    def __init__(self, default_mass_kg: float = 70.0):
        self.default_mass = default_mass_kg

    def _coerce_poly(self, polygon: List[Tuple[float, float]]) -> Optional[Any]:
        if not _SHAPELY_AVAILABLE or not polygon or len(polygon) < 3:
            return None
        if _SHAPELY_POLYGON is None or not callable(_SHAPELY_POLYGON):
            return None
        poly = cast(Any, _SHAPELY_POLYGON)(polygon)
        if poly is None:
            return None
        if poly.is_valid and not poly.is_empty:
            return poly
        return None

    @staticmethod
    def _point_in_polygon(point_xy: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        x, y = point_xy
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside

    @staticmethod
    def _distance_to_segment(
        point_xy: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        px, py = point_xy
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

    @staticmethod
    def _project_to_polygon(point_xy: Tuple[float, float], polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
        best_point = polygon[0]
        best_distance = COPEstimator._distance_to_segment(point_xy, polygon[0], polygon[1])
        for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]
            dist = COPEstimator._distance_to_segment(point_xy, a, b)
            if dist < best_distance:
                best_distance = dist
                x1, y1 = a
                x2, y2 = b
                vx, vy = x2 - x1, y2 - y1
                wx, wy = point_xy[0] - x1, point_xy[1] - y1
                seg_len2 = vx * vx + vy * vy
                if seg_len2 == 0.0:
                    best_point = a
                else:
                    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
                    best_point = (x1 + t * vx, y1 + t * vy)
        return best_point

    def estimate(self, data: COPInput, method: str = "physical_projection") -> COPState:
        if data.mass_kg <= 1e-3:
            mass = self.default_mass
        else:
            mass = data.mass_kg

        ax, ay, az = data.acceleration
        x, y, z = data.cog

        Fz = mass * (data.gravity + az)
        Fz = max(1.0, float(Fz))
        Fx = mass * float(ax)
        Fy = mass * float(ay)

        # Ground reaction and moment consistency (2D approximation):
        cx = x - z * Fx / Fz
        cy = y + z * Fy / Fz
        cand = (float(cx), float(cy))

        poly = self._coerce_poly(data.polygon)
        within = False
        residual = 0.0
        used = cand

        if (
            poly is not None
            and _SHAPELY_AVAILABLE
            and _SHAPELY_POINT is not None
            and _SHAPELY_NEAREST_POINTS is not None
            and callable(_SHAPELY_POINT)
            and callable(_SHAPELY_NEAREST_POINTS)
        ):
            poly_shape = cast(Any, poly)
            point_obj = cast(Any, _SHAPELY_POINT)(cand)
            if poly_shape.contains(point_obj) or poly_shape.touches(point_obj):
                used = cand
                within = True
                residual = 0.0
            else:
                nearest = cast(Any, _SHAPELY_NEAREST_POINTS)(point_obj, poly_shape)[1]
                used = (float(nearest.x), float(nearest.y))
                residual = float(point_obj.distance(poly_shape))
        else:
            if self._point_in_polygon(cand, data.polygon):
                used = cand
                within = True
                residual = 0.0
            else:
                used = self._project_to_polygon(cand, data.polygon)
                residual = min(
                    self._distance_to_segment(cand, data.polygon[i], data.polygon[(i + 1) % len(data.polygon)])
                    for i in range(len(data.polygon))
                )

        # friction residual (soft)
        if abs(Fx) > data.friction_mu * Fz + 1e-6:
            residual += float((abs(Fx) - data.friction_mu * Fz) / (data.friction_mu * Fz))
        if abs(Fy) > data.friction_mu * Fz + 1e-6:
            residual += float((abs(Fy) - data.friction_mu * Fz) / (data.friction_mu * Fz))

        if residual == 0.0 and data.prev_cop is not None:
            # smoothness prior
            residual = float(np.hypot(used[0] - data.prev_cop[0], used[1] - data.prev_cop[1]))

        conf = 1.0
        if poly is None:
            conf = 0.35
        elif residual > 0.0:
            conf = max(0.25, 1.0 / (1.0 + residual))

        return COPState(
            cop=used,
            force=(float(Fx), float(Fy), float(Fz)),
            confidence=conf,
            within_bos=within,
            residual=float(residual),
            method=method,
        )
