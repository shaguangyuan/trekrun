"""
Compute all 5 sprint metrics for a valid segment.

Input: successful ExtractionResult + ValidSegment from video_quality_segment.
Output: dict with keys matching metric_definition.md names.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.metrics.arm_swing_variability import compute_arm_swing_variability
from app.services.metrics.errors import MetricComputationError
from app.services.metrics.left_right_timing_diff import compute_left_right_timing_diff
from app.services.metrics.step_rate import compute_step_rate
from app.services.metrics.tech_stability_score import compute_tech_stability_score
from app.services.metrics.trunk_lean_mean import compute_trunk_lean_mean
from app.services.pose_extractor import ExtractionResult
from app.services.video_quality_segment import ValidSegment


def compute_sprint_metrics(
    extraction: ExtractionResult,
    segment: ValidSegment,
) -> Dict[str, Any]:
    """
    Returns:
        step_rate, trunk_lean_mean, arm_swing_variability,
        left_right_timing_diff, tech_stability_score

    Raises MetricComputationError if any metric cannot be computed.
    """
    if not extraction.success:
        raise MetricComputationError("pipeline", "Extraction was not successful.")
    if segment.end_frame_idx < segment.start_frame_idx:
        raise MetricComputationError("pipeline", "Invalid valid_segment frame range.")

    frames = extraction.frames[segment.start_frame_idx : segment.end_frame_idx + 1]
    duration_sec = (segment.end_ms - segment.start_ms) / 1000.0
    fps = float(extraction.fps)

    step_rate = compute_step_rate(frames, duration_sec, fps)
    trunk_lean_mean = compute_trunk_lean_mean(frames)
    arm_sw = compute_arm_swing_variability(frames)
    lr_diff = compute_left_right_timing_diff(frames, fps)
    tech = compute_tech_stability_score(frames, fps, arm_sw, lr_diff)

    return {
        "step_rate": step_rate,
        "trunk_lean_mean": trunk_lean_mean,
        "arm_swing_variability": arm_sw,
        "left_right_timing_diff": lr_diff,
        "tech_stability_score": tech,
    }
