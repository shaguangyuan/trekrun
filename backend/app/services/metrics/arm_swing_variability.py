"""
arm_swing_variability — normalized score from shoulder–elbow–wrist amplitude + period variability.

Definition: 肩-肘-腕轨迹振幅和周期波动的组合指标
Output: normalized score in [0, 1] (higher = more variability / less consistent swing)
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from app.services.metrics.constants import (
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    POSE_NUM_LANDMARKS,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
)
from app.services.metrics.errors import MetricComputationError
from app.services.metrics.signal_utils import moving_average
from app.services.pose_extractor import FrameResult, LandmarkPoint


def _angle_at_elbow(lm: List[LandmarkPoint], sh: int, el: int, wr: int) -> float | None:
    if len(lm) <= max(sh, el, wr):
        return None
    if lm[sh].visibility < 0.25 or lm[el].visibility < 0.25 or lm[wr].visibility < 0.25:
        return None
    # Vectors from elbow to shoulder and elbow to wrist
    v1x = lm[sh].x - lm[el].x
    v1y = lm[sh].y - lm[el].y
    v2x = lm[wr].x - lm[el].x
    v2y = lm[wr].y - lm[el].y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    c = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(c))


def _series_for_arm(frames: Sequence[FrameResult], sh: int, el: int, wr: int) -> np.ndarray:
    vals: List[float] = []
    for fr in frames:
        if not fr.has_pose or len(fr.pose_landmarks) < POSE_NUM_LANDMARKS:
            vals.append(float("nan"))
            continue
        a = _angle_at_elbow(fr.pose_landmarks, sh, el, wr)
        vals.append(float(a) if a is not None else float("nan"))
    arr = np.asarray(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return arr
    idx = np.arange(len(arr))
    good = ~np.isnan(arr)
    if good.sum() < 2:
        return arr
    arr[~good] = np.interp(idx[~good], idx[good], arr[good])
    return arr


def _peak_period_cv(angles: np.ndarray) -> float:
    sm = moving_average(angles, 3)
    # peaks in swing: local maxima on smoothed signal
    peaks: List[int] = []
    s = sm.tolist()
    for i in range(1, len(s) - 1):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1]:
            peaks.append(i)
    if len(peaks) < 3:
        return 0.0
    gaps = [peaks[j] - peaks[j - 1] for j in range(1, len(peaks))]
    g = np.asarray(gaps, dtype=float)
    m = float(np.mean(g))
    if m < 1e-6:
        return 0.0
    return float(np.std(g) / m)


def compute_arm_swing_variability(frames: Sequence[FrameResult]) -> float:
    left = _series_for_arm(frames, L_SHOULDER, L_ELBOW, L_WRIST)
    right = _series_for_arm(frames, R_SHOULDER, R_ELBOW, R_WRIST)
    if np.all(np.isnan(left)) and np.all(np.isnan(right)):
        raise MetricComputationError(
            "arm_swing_variability",
            "Arm landmarks not visible enough in valid segment.",
        )
    # Amplitude fluctuation: rolling std of angle (normalized)
    def amp_part(a: np.ndarray) -> float:
        if np.all(np.isnan(a)):
            return 0.0
        if np.nanstd(a) < 1e-6:
            return 0.0
        sm = moving_average(np.nan_to_num(a, nan=np.nanmean(a)), 5)
        resid = a - sm
        return float(np.nanstd(resid) / (np.nanmax(a) - np.nanmin(a) + 5.0))

    al = amp_part(left)
    ar = amp_part(right)
    pl = _peak_period_cv(np.nan_to_num(left, nan=np.nanmean(left)))
    pr = _peak_period_cv(np.nan_to_num(right, nan=np.nanmean(right)))
    raw = 0.5 * (al + ar) + 0.5 * (0.5 * (pl + pr))
    # Map to roughly [0, 1]
    score = float(max(0.0, min(1.0, raw)))
    if score == 0.0 and al + ar + pl + pr == 0.0:
        raise MetricComputationError(
            "arm_swing_variability",
            "Could not derive arm swing variability (no angular motion detected).",
        )
    return score
