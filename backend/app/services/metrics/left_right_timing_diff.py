"""
left_right_timing_diff — percentage difference between left vs right step intervals.

Definition: 左右步时或关键相位间隔差异
Output: percentage (0–100+ scale; higher = larger asymmetry)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.services.metrics.errors import MetricComputationError
from app.services.metrics.signal_utils import intervals_from_indices
from app.services.metrics.step_rate import ankle_strike_indices_per_leg
from app.services.pose_extractor import FrameResult


def compute_left_right_timing_diff(
    frames: Sequence[FrameResult],
    fps: float,
    min_strike_gap_sec: float = 0.22,
) -> float:
    strikes_l, strikes_r, _ = ankle_strike_indices_per_leg(frames, fps, min_strike_gap_sec)
    il = intervals_from_indices(strikes_l, fps)
    ir = intervals_from_indices(strikes_r, fps)
    if len(il) < 1 or len(ir) < 1:
        raise MetricComputationError(
            "left_right_timing_diff",
            "Need at least two strikes per leg to compare left/right timing (intervals missing).",
        )
    m_l = float(np.mean(il))
    m_r = float(np.mean(ir))
    mean_d = 0.5 * (m_l + m_r)
    if mean_d < 1e-6:
        raise MetricComputationError(
            "left_right_timing_diff",
            "Mean step interval too small to compute timing difference.",
        )
    return float(abs(m_l - m_r) / mean_d * 100.0)
