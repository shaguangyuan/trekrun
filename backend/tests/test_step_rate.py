import pytest

from app.services.metrics.errors import MetricComputationError
from app.services.metrics.step_rate import compute_step_rate
from tests.metrics_helpers import synthetic_running_segment


def test_step_rate_positive_on_synthetic_gait() -> None:
    fps = 30.0
    frames = synthetic_running_segment(120, fps=fps)
    duration_sec = (frames[-1].timestamp_ms - frames[0].timestamp_ms) / 1000.0
    sr = compute_step_rate(frames, duration_sec=duration_sec, fps=fps)
    assert sr > 0.5
    assert sr < 20.0


def test_step_rate_rejects_zero_duration() -> None:
    frames = synthetic_running_segment(20, fps=30.0)
    with pytest.raises(MetricComputationError) as ei:
        compute_step_rate(frames, duration_sec=0.0, fps=30.0)
    assert ei.value.code == "step_rate"


def test_step_rate_rejects_too_few_frames() -> None:
    frames = synthetic_running_segment(3, fps=30.0)
    with pytest.raises(MetricComputationError) as ei:
        compute_step_rate(frames, duration_sec=0.1, fps=30.0)
    assert ei.value.code == "step_rate"
