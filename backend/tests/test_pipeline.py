from app.services.metrics.pipeline import compute_sprint_metrics
from app.services.pose_extractor import ExtractionResult
from app.services.video_quality_segment import ValidSegment
from tests.metrics_helpers import synthetic_arm_swing_segment, synthetic_running_segment


def test_pipeline_returns_five_metric_keys() -> None:
    fps = 30.0
    frames = synthetic_running_segment(120, fps=fps, right_phase=0.5)
    arm_frames = synthetic_arm_swing_segment(120, fps=fps)
    for i, fr in enumerate(frames):
        fr.pose_landmarks[15] = arm_frames[i].pose_landmarks[15]
        fr.pose_landmarks[16] = arm_frames[i].pose_landmarks[16]
        fr.pose_landmarks[13] = arm_frames[i].pose_landmarks[13]
        fr.pose_landmarks[14] = arm_frames[i].pose_landmarks[14]

    extraction = ExtractionResult(
        video_id="test-vid",
        fps=fps,
        total_frames=len(frames),
        duration_ms=frames[-1].timestamp_ms,
        width=1280,
        height=720,
        frames=frames,
        frames_with_pose=len(frames),
        success=True,
    )
    seg = ValidSegment(
        start_frame_idx=0,
        end_frame_idx=len(frames) - 1,
        start_ms=frames[0].timestamp_ms,
        end_ms=frames[-1].timestamp_ms,
        frame_count=len(frames),
    )
    out = compute_sprint_metrics(extraction, seg)
    assert set(out.keys()) == {
        "step_rate",
        "trunk_lean_mean",
        "arm_swing_variability",
        "left_right_timing_diff",
        "tech_stability_score",
    }
    assert out["tech_stability_score"] >= 0.0
