import math

import pytest

from app.services.metrics.errors import MetricComputationError
from app.services.metrics.trunk_lean_mean import compute_trunk_lean_mean, trunk_lean_series_deg
from app.services.pose_extractor import FrameResult
from tests.metrics_helpers import make_frame, make_landmarks


def test_trunk_lean_mean_upright_near_zero() -> None:
    frames = []
    for i in range(10):
        lm = make_landmarks(
            {
                11: (0.5, 0.35, 0.95),
                12: (0.5, 0.35, 0.95),
                23: (0.5, 0.75, 0.95),
                24: (0.5, 0.75, 0.95),
            }
        )
        frames.append(
            FrameResult(
                frame_idx=i,
                timestamp_ms=i * 33,
                has_pose=True,
                pose_landmarks=lm,
            )
        )
    mean_lean = compute_trunk_lean_mean(frames)
    assert mean_lean < 2.0


def test_trunk_lean_mean_forward_lean() -> None:
    frames = []
    for i in range(10):
        lm = make_landmarks(
            {
                11: (0.55, 0.35, 0.95),
                12: (0.55, 0.35, 0.95),
                23: (0.5, 0.75, 0.95),
                24: (0.5, 0.75, 0.95),
            }
        )
        frames.append(FrameResult(frame_idx=i, timestamp_ms=i * 33, has_pose=True, pose_landmarks=lm))
    mean_lean = compute_trunk_lean_mean(frames)
    expected = math.degrees(math.atan2(0.05, 0.4))
    assert abs(mean_lean - expected) < 0.5


def test_trunk_lean_mean_insufficient_visibility() -> None:
    lm = make_landmarks({11: (0.5, 0.35, 0.1), 12: (0.5, 0.35, 0.1), 23: (0.5, 0.75, 0.1), 24: (0.5, 0.75, 0.1)})
    frames = [FrameResult(0, 0, True, lm)] * 2
    with pytest.raises(MetricComputationError) as ei:
        compute_trunk_lean_mean(frames)
    assert ei.value.code == "trunk_lean_mean"


def test_trunk_lean_series_deg_non_empty() -> None:
    f = make_frame(
        0,
        0,
        {
            11: (0.5, 0.35, 0.95),
            12: (0.5, 0.35, 0.95),
            23: (0.5, 0.75, 0.95),
            24: (0.5, 0.75, 0.95),
        },
    )
    assert len(trunk_lean_series_deg([f])) == 1
