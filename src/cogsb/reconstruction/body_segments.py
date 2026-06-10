from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass(frozen=True)
class SegmentDef:
    name: str
    start: Tuple[int, ...]
    end: Tuple[int, ...]
    alpha: float
    mass: float


def _avg_point(pts: List[Tuple[float, float, float]], idxs: Tuple[int, ...]) -> Tuple[float, float, float]:
    if not idxs:
        return (0.0, 0.0, 0.0)
    xs, ys, zs = zip(*[pts[i] for i in idxs])
    n = len(idxs)
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


# Mediapipe 33 landmarks index-based mapping.
# start/end can be multi-index so that pelvis/shoulder centers can be represented.
SEGMENT_TABLE = [
    SegmentDef("head", (0,), (7, 8), 0.506, 6.94),
    SegmentDef("torso", (11, 12), (23, 24), 0.5, 29.5),
    SegmentDef("upper_arm_l", (11,), (13,), 0.436, 2.71),
    SegmentDef("upper_arm_r", (12,), (14,), 0.436, 2.71),
    SegmentDef("forearm_l", (13,), (15,), 0.430, 1.62),
    SegmentDef("forearm_r", (14,), (16,), 0.430, 1.62),
    SegmentDef("hand_l", (15,), (19,), 0.500, 0.61),
    SegmentDef("hand_r", (16,), (20,), 0.500, 0.61),
    SegmentDef("thigh_l", (23,), (25,), 0.433, 10.34),
    SegmentDef("thigh_r", (24,), (26,), 0.433, 10.34),
    SegmentDef("shank_l", (25,), (27,), 0.430, 4.33),
    SegmentDef("shank_r", (26,), (28,), 0.430, 4.33),
    SegmentDef("foot_l", (29,), (31,), 0.500, 1.45),
    SegmentDef("foot_r", (30,), (32,), 0.500, 1.45),
]


def segment_center(pts: List[Tuple[float, float, float]], segment: SegmentDef) -> Tuple[float, float, float]:
    p0 = _avg_point(pts, segment.start)
    p1 = _avg_point(pts, segment.end)
    return (
        p0[0] + segment.alpha * (p1[0] - p0[0]),
        p0[1] + segment.alpha * (p1[1] - p0[1]),
        p0[2] + segment.alpha * (p1[2] - p0[2]),
    )


def all_segment_mass() -> dict:
    total = sum(item.mass for item in SEGMENT_TABLE)
    return {item.name: item.mass / total for item in SEGMENT_TABLE}
