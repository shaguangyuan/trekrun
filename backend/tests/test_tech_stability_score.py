from app.services.metrics.arm_swing_variability import compute_arm_swing_variability
from app.services.metrics.left_right_timing_diff import compute_left_right_timing_diff
from app.services.metrics.tech_stability_score import compute_tech_stability_score
from tests.metrics_helpers import synthetic_arm_swing_segment, synthetic_running_segment


def test_tech_stability_score_in_range() -> None:
    fps = 30.0
    frames = synthetic_running_segment(120, fps=fps, right_phase=0.5)
    # Mix in arm motion so arm metric is non-degenerate
    arm_frames = synthetic_arm_swing_segment(120, fps=fps)
    for i, fr in enumerate(frames):
        fr.pose_landmarks[15] = arm_frames[i].pose_landmarks[15]
        fr.pose_landmarks[16] = arm_frames[i].pose_landmarks[16]
        fr.pose_landmarks[13] = arm_frames[i].pose_landmarks[13]
        fr.pose_landmarks[14] = arm_frames[i].pose_landmarks[14]
    arm_sw = compute_arm_swing_variability(frames)
    lr = compute_left_right_timing_diff(frames, fps=fps)
    score = compute_tech_stability_score(frames, fps, arm_sw, lr)
    assert 0.0 <= score <= 100.0
