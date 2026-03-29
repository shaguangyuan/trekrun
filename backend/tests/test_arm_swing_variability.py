import pytest

from app.services.metrics.arm_swing_variability import compute_arm_swing_variability
from app.services.metrics.errors import MetricComputationError
from app.services.pose_extractor import FrameResult, LandmarkPoint
from tests.metrics_helpers import make_frame, make_landmarks, synthetic_arm_swing_segment


def test_arm_swing_positive_on_oscillating_wrists() -> None:
    frames = synthetic_arm_swing_segment(80, fps=30.0)
    score = compute_arm_swing_variability(frames)
    assert 0.0 <= score <= 1.0
    assert score > 0.01


def test_arm_swing_rejects_missing_arms() -> None:
    lm = [LandmarkPoint(0.5, 0.5, 0.0, 0.05) for _ in range(33)]
    frames = [FrameResult(i, i * 33, True, lm) for i in range(20)]
    with pytest.raises(MetricComputationError) as ei:
        compute_arm_swing_variability(frames)
    assert ei.value.code == "arm_swing_variability"


def test_arm_swing_single_visible_frame_insufficient() -> None:
    f = make_frame(
        0,
        0,
        {
            11: (0.46, 0.35, 0.95),
            12: (0.54, 0.35, 0.95),
            13: (0.44, 0.45, 0.95),
            14: (0.56, 0.45, 0.95),
            15: (0.42, 0.55, 0.95),
            16: (0.58, 0.55, 0.95),
            23: (0.46, 0.78, 0.95),
            24: (0.54, 0.78, 0.95),
            27: (0.45, 0.88, 0.95),
            28: (0.55, 0.88, 0.95),
        },
    )
    with pytest.raises(MetricComputationError):
        compute_arm_swing_variability([f])
