"""
trunk_lean_mean — mean angle (degrees) between shoulder–hip line and vertical.

Definition: 肩-髋连线相对垂线的夹角均值
Output: degree

Uses mid-shoulder and mid-hip in normalized image coordinates; vertical = (0, 1) (y down).
Angle = arccos(clamp(dot(v, e_y), -1, 1)) with v = unit vector from hip to shoulder.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from app.services.metrics.constants import (
    L_HIP,
    L_SHOULDER,
    POSE_NUM_LANDMARKS,
    R_HIP,
    R_SHOULDER,
)
from app.services.metrics.errors import MetricComputationError
from app.services.pose_extractor import FrameResult, LandmarkPoint


def _trunk_lean_deg_frame(fr: FrameResult) -> float | None:
    if not fr.has_pose or len(fr.pose_landmarks) < POSE_NUM_LANDMARKS:
        return None
    lm: List[LandmarkPoint] = fr.pose_landmarks
    for idx in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        if lm[idx].visibility < 0.3:
            return None
    sx = (lm[L_SHOULDER].x + lm[R_SHOULDER].x) * 0.5
    sy = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) * 0.5
    hx = (lm[L_HIP].x + lm[R_HIP].x) * 0.5
    hy = (lm[L_HIP].y + lm[R_HIP].y) * 0.5
    dx, dy = sx - hx, sy - hy
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        return None
    # Smallest angle between hip→shoulder segment and vertical (y axis in image).
    theta_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(theta_rad)


def trunk_lean_series_deg(frames: Sequence[FrameResult]) -> List[float]:
    """All per-frame trunk lean angles (degrees) where landmarks are visible."""
    vals: List[float] = []
    for fr in frames:
        a = _trunk_lean_deg_frame(fr)
        if a is not None:
            vals.append(a)
    return vals


def compute_trunk_lean_mean(frames: Sequence[FrameResult]) -> float:
    vals = trunk_lean_series_deg(frames)
    if len(vals) < 3:
        raise MetricComputationError(
            "trunk_lean_mean",
            "Insufficient frames with visible shoulders and hips for trunk lean.",
        )
    return float(sum(vals) / len(vals))
