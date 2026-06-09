from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

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

    def _coerce_poly(self, polygon: List[Tuple[float, float]]) -> Optional[Polygon]:
        if polygon and len(polygon) >= 3:
            poly = Polygon(polygon)
            if poly.is_valid and not poly.is_empty:
                return poly
        return None

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

        if poly is None:
            if data.prev_cop is not None:
                used = data.prev_cop
            residual = 0.0
        else:
            p = Point(cand)
            if poly.contains(p) or poly.touches(p):
                used = cand
                within = True
            else:
                proj = nearest_points(p, poly)[1]
                used = (float(proj.x), float(proj.y))
                residual = float(p.distance(poly))

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
