from app.services.report_explainer import build_natural_language_explanation


def _base_overview() -> dict:
    return {
        "duration_ms": 4200.0,
        "fps": 30.0,
        "pose_summary": {
            "total_frames": 126,
            "frames_with_pose": 118,
            "pose_ratio": 0.9365,
        },
        "qc_summary": {"quality_level": "usable_for_all_metrics"},
        "selected_segment": {"start_ms": 900, "end_ms": 3600, "duration_ms": 2700, "reason": "最长连续片段"},
    }


def test_explainer_done_generates_sections() -> None:
    out = build_natural_language_explanation(
        video_id="v1",
        analysis_state="done",
        analysis_overview=_base_overview(),
        metric_details=[
            {"key": "step_rate", "available": True, "confidence": 0.82, "used_frames": 81, "used_joints": ["left_ankle", "right_ankle"]},
        ],
        warnings=[],
        suggested_fix=None,
    )
    assert "summary" in out
    assert "process_steps" in out
    assert "metric_explanations" in out
    assert len(out["process_steps"]) >= 3


def test_explainer_partial_contains_unavailable_metric_text() -> None:
    out = build_natural_language_explanation(
        video_id="v2",
        analysis_state="partial",
        analysis_overview=_base_overview(),
        metric_details=[
            {"key": "left_right_timing_diff", "available": False, "confidence": 0.1, "used_frames": 50, "warning": "intervals missing"},
        ],
        warnings=["keypoint pass ratio low"],
        suggested_fix="请保持全身连续入镜",
    )
    row = out["metric_explanations"][0]
    assert row["available"] is False
    assert "未成功计算" in row["explanation"]
    assert len(out["warnings"]) == 1


def test_explainer_failed_structure_consistent() -> None:
    out = build_natural_language_explanation(
        video_id="v3",
        analysis_state="failed",
        analysis_overview={"duration_ms": 0, "fps": 0, "pose_summary": {}},
        metric_details=[],
        warnings=["模型文件缺失"],
        suggested_fix="下载模型后重试",
    )
    assert set(out.keys()) == {
        "summary",
        "process_steps",
        "selected_segment_explanation",
        "metric_explanations",
        "warnings",
        "suggested_fix",
        "video_id",
    }
