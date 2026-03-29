"""
step_rate — steps per second in the valid analysis interval.

Definition (metric_definition.md): 有效分析区间内的步数 / 时间
Output: steps per second

Heuristic: detect foot-strike–like minima on smoothed ankle y (image space, y down).
Each minimum on left ankle + right ankle counts as one step event (one foot contact).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from app.services.metrics.constants import L_ANKLE, R_ANKLE, POSE_NUM_LANDMARKS
from app.services.metrics.errors import MetricComputationError
from app.services.pose_extractor import FrameResult, LandmarkPoint
from app.services.metrics.signal_utils import (
    local_minima_indices,
    moving_average,
    pick_strikes_with_min_separation,
)


def _ankle_y_series(frames: Sequence[FrameResult], ankle_idx: int) -> List[float]:
    ys: List[float] = []
    for fr in frames:
        if not fr.has_pose or len(fr.pose_landmarks) < POSE_NUM_LANDMARKS:
            ys.append(float("nan"))
            continue
        lm: LandmarkPoint = fr.pose_landmarks[ankle_idx]
        if lm.visibility < 0.2:
            ys.append(float("nan"))
        else:
            ys.append(float(lm.y))
    return ys


def _fill_nan_ankle_series(arr: List[float]) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if np.all(np.isnan(a)):
        raise MetricComputationError("step_rate", "Ankle landmarks missing in segment.")
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    if good.sum() < 2:
        raise MetricComputationError("step_rate", "Insufficient ankle visibility.")
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def ankle_strike_indices_per_leg(
    frames: Sequence[FrameResult],
    fps: float,
    min_strike_gap_sec: float = 0.22,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (left_strikes, right_strikes, merged) frame indices within *frames* slice.
    Per-leg separation enforces one-foot minimum gap; merged re-applies a
    shorter cross-leg separation to remove side-view overlap duplicates.
    """
    if len(frames) < 5:
        raise MetricComputationError("step_rate", "Not enough frames in valid segment.")

    ly = _ankle_y_series(frames, L_ANKLE)
    ry = _ankle_y_series(frames, R_ANKLE)
    l_arr = _fill_nan_ankle_series(ly)
    r_arr = _fill_nan_ankle_series(ry)

    min_sep = max(3, int(fps * min_strike_gap_sec))
    sm_l = moving_average(l_arr, 5)
    sm_r = moving_average(r_arr, 5)
    strikes_l = pick_strikes_with_min_separation(local_minima_indices(sm_l.tolist()), min_sep)
    strikes_r = pick_strikes_with_min_separation(local_minima_indices(sm_r.tolist()), min_sep)

    raw_merged = sorted(set(strikes_l) | set(strikes_r))
    min_sep_merged = max(2, int(fps * 0.10))
    merged = pick_strikes_with_min_separation(raw_merged, min_sep_merged)
    return strikes_l, strikes_r, merged


def compute_step_rate(
    frames: Sequence[FrameResult],
    duration_sec: float,
    fps: float,
    min_strike_gap_sec: float = 0.22,
) -> float:
    """
    :param frames: frames inside valid_segment (inclusive slice)
    :param duration_sec: (end_ms - start_ms) / 1000 of the segment
    :param fps: video fps
    """
    if duration_sec <= 0:
        raise MetricComputationError("step_rate", "Segment duration must be positive.")

    _, _, merged = ankle_strike_indices_per_leg(frames, fps, min_strike_gap_sec)

    if len(merged) < 2:
        raise MetricComputationError(
            "step_rate",
            "Not enough step events detected in valid segment (need at least 2 strikes).",
        )

    return float(len(merged)) / duration_sec
