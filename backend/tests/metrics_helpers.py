"""Synthetic pose frames for metric unit tests."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from app.services.metrics.constants import (
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
)
from app.services.pose_extractor import FrameResult, LandmarkPoint


def make_landmarks(
    overrides: Optional[Dict[int, Tuple[float, float, float]]] = None,
) -> List[LandmarkPoint]:
    """33 landmarks; default center with high visibility. Override (x, y, visibility)."""
    lm = [LandmarkPoint(0.5, 0.5, 0.0, 0.95) for _ in range(33)]
    if overrides:
        for idx, (x, y, vis) in overrides.items():
            lm[idx] = LandmarkPoint(x, y, 0.0, vis)
    return lm


def make_frame(
    frame_idx: int,
    timestamp_ms: int,
    overrides: Optional[Dict[int, Tuple[float, float, float]]] = None,
) -> FrameResult:
    return FrameResult(
        frame_idx=frame_idx,
        timestamp_ms=timestamp_ms,
        has_pose=True,
        pose_landmarks=make_landmarks(overrides),
    )


def synthetic_running_segment(
    num_frames: int,
    fps: float = 30.0,
    left_period_frames: int = 12,
    right_phase: float = 0.5,
) -> List[FrameResult]:
    """
    Oscillating ankle y to produce periodic minima (foot-strike proxy).
    right_phase in [0,1] shifts right leg phase vs left.
    """
    frames: List[FrameResult] = []
    for i in range(num_frames):
        t = 2 * math.pi * i / left_period_frames
        ly = 0.65 + 0.08 * math.sin(t)
        ry = 0.65 + 0.08 * math.sin(t + right_phase * 2 * math.pi)
        ov = {
            L_ANKLE: (0.45, ly, 0.95),
            R_ANKLE: (0.55, ry, 0.95),
            L_HIP: (0.46, 0.78, 0.95),
            R_HIP: (0.54, 0.78, 0.95),
            L_SHOULDER: (0.46, 0.35, 0.95),
            R_SHOULDER: (0.54, 0.35, 0.95),
            L_ELBOW: (0.42, 0.48, 0.95),
            R_ELBOW: (0.58, 0.48, 0.95),
            L_WRIST: (0.40, 0.58, 0.95),
            R_WRIST: (0.60, 0.58, 0.95),
        }
        ts = int(i * (1000.0 / fps))
        frames.append(make_frame(i, ts, ov))
    return frames


def synthetic_arm_swing_segment(num_frames: int, fps: float = 30.0) -> List[FrameResult]:
    """Elbow angles modulate via wrist y oscillation."""
    frames: List[FrameResult] = []
    for i in range(num_frames):
        t = 2 * math.pi * i / 10
        wy_l = 0.55 + 0.06 * math.sin(t)
        wy_r = 0.55 + 0.06 * math.sin(t + 1.0)
        ov = {
            L_SHOULDER: (0.46, 0.35, 0.95),
            R_SHOULDER: (0.54, 0.35, 0.95),
            L_ELBOW: (0.44, 0.45, 0.95),
            R_ELBOW: (0.56, 0.45, 0.95),
            L_WRIST: (0.42, wy_l, 0.95),
            R_WRIST: (0.58, wy_r, 0.95),
            L_HIP: (0.46, 0.78, 0.95),
            R_HIP: (0.54, 0.78, 0.95),
            L_ANKLE: (0.45, 0.88, 0.95),
            R_ANKLE: (0.55, 0.88, 0.95),
        }
        ts = int(i * (1000.0 / fps))
        frames.append(make_frame(i, ts, ov))
    return frames
