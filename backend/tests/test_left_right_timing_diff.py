import pytest

from app.services.metrics.errors import MetricComputationError
from app.services.metrics.left_right_timing_diff import compute_left_right_timing_diff
from tests.metrics_helpers import synthetic_running_segment


def test_left_right_timing_diff_computable_on_synthetic() -> None:
    fps = 30.0
    frames = synthetic_running_segment(120, fps=fps, right_phase=0.5)
    pct = compute_left_right_timing_diff(frames, fps=fps)
    assert pct >= 0.0
    assert pct < 500.0


def test_left_right_timing_diff_symmetric_legs_small_diff() -> None:
    """Same phase on both ankles → intervals should be similar → low % diff."""
    fps = 30.0
    frames = synthetic_running_segment(120, fps=fps, right_phase=0.0)
    pct = compute_left_right_timing_diff(frames, fps=fps)
    assert pct < 80.0


def test_left_right_timing_diff_too_short() -> None:
    fps = 30.0
    frames = synthetic_running_segment(8, fps=fps)
    with pytest.raises(MetricComputationError) as ei:
        compute_left_right_timing_diff(frames, fps=fps)
    assert ei.value.code in ("left_right_timing_diff", "step_rate")
