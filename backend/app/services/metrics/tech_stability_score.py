"""
tech_stability_score — 0–100 composite from step, trunk, arm, L/R timing variability.

Definition: 基于步频变化、躯干波动、摆臂波动、左右差异综合加权的分数
Output: 0–100 (higher = more stable / consistent movement in this heuristic)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from app.services.metrics.errors import MetricComputationError
from app.services.metrics.signal_utils import intervals_from_indices
from app.services.metrics.step_rate import ankle_strike_indices_per_leg
from app.services.metrics.trunk_lean_mean import trunk_lean_series_deg
from app.services.pose_extractor import FrameResult


def _norm_clamp(x: float, scale: float) -> float:
    """Map x/scale into ~[0,1], clamped."""
    if scale <= 0:
        return 0.0
    return float(max(0.0, min(1.0, x / scale)))


def compute_tech_stability_score(
    frames: Sequence[FrameResult],
    fps: float,
    arm_swing_variability: float,
    left_right_timing_diff_pct: float,
    min_strike_gap_sec: float = 0.12,
) -> float:
    """
    :param arm_swing_variability: output of compute_arm_swing_variability (0–1)
    :param left_right_timing_diff_pct: output of compute_left_right_timing_diff
    """
    if len(frames) < 5:
        raise MetricComputationError(
            "tech_stability_score",
            "Not enough frames to compute stability score.",
        )

    _, _, merged = ankle_strike_indices_per_leg(frames, fps, min_strike_gap_sec)
    ivals = intervals_from_indices(merged, fps)
    if len(ivals) >= 2:
        cv_step = float(np.std(ivals) / (np.mean(ivals) + 1e-6))
    elif len(ivals) == 1:
        cv_step = 0.0
    else:
        cv_step = 1.0  # penalize if no intervals

    t_list: List[float] = trunk_lean_series_deg(frames)
    if len(t_list) >= 3:
        cv_trunk = float(np.std(t_list) / (abs(np.mean(t_list)) + 1e-3))
    else:
        cv_trunk = 1.0

    # Normalize components to roughly [0,1]
    n_step = _norm_clamp(cv_step, 0.35)
    n_trunk = _norm_clamp(cv_trunk, 0.25)
    n_arm = float(max(0.0, min(1.0, arm_swing_variability)))
    n_lr = _norm_clamp(left_right_timing_diff_pct, 25.0)

    raw_penalty = 0.25 * n_step + 0.25 * n_trunk + 0.25 * n_arm + 0.25 * n_lr
    score = 100.0 * (1.0 - raw_penalty)
    return float(max(0.0, min(100.0, score)))
