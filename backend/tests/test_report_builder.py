from app.services.report_builder import build_report_document

CUR = {
    "step_rate": 4.2,
    "trunk_lean_mean": 8.3,
    "arm_swing_variability": 0.72,
    "left_right_timing_diff": 3.5,
    "tech_stability_score": 78.0,
}

PREV = {
    "step_rate": 4.0,
    "trunk_lean_mean": 8.5,
    "arm_swing_variability": 0.70,
    "left_right_timing_diff": 4.0,
    "tech_stability_score": 75.0,
}


def test_nested_report_json_shape() -> None:
    doc = build_report_document(
        video_id="v1",
        athlete_id="A1",
        session_type="normal",
        fatigue_state="no",
        event_group="100",
        created_at="2026-03-23",
        metric_values=CUR,
        previous_metric_values=PREV,
        duration_ms=3000.0,
        fps=30.0,
    )
    d = doc.to_nested_dict()
    assert set(d.keys()) == {
        "video_info",
        "metric_values",
        "comparison_values",
        "metrics_detail",
        "coach_summary",
        "warnings",
        "analysis_overview",
        "natural_language",
        "raw_feature_summary",
        "feature_groups",
        "metric_confidence",
        "used_frames",
        "used_joints",
        "failure_stage",
        "failure_reason",
        "pose_summary",
        "qc_summary",
        "metrics_available",
        "suggested_fix",
        "saved_to_history",
        "saved_at",
    }
    assert d["metric_values"]["step_rate"] == 4.2
    assert abs(d["comparison_values"]["step_rate"] - 0.2) < 1e-6


def test_no_previous_graceful_comparison_zeros() -> None:
    doc = build_report_document(
        video_id="v1",
        athlete_id="A1",
        session_type="normal",
        fatigue_state="no",
        event_group="100",
        created_at="2026-03-23",
        metric_values=CUR,
        previous_metric_values=None,
    )
    api = doc.to_api_response()
    assert api.comparison.tech_stability_score == 0.0
    assert any("暂无" in w for w in api.warnings)


def test_api_response_preserves_legacy_flat_fields() -> None:
    doc = build_report_document(
        video_id="v9",
        athlete_id="A9",
        session_type="pre_test",
        fatigue_state="yes",
        event_group="200",
        created_at="2026-01-01",
        metric_values=CUR,
        previous_metric_values=PREV,
    )
    r = doc.to_api_response()
    assert r.video_id == "v9"
    assert r.metrics.step_rate == CUR["step_rate"]
    assert r.comparison.step_rate == CUR["step_rate"] - PREV["step_rate"]
    assert r.video_info.athlete_id == "A9"
    assert len(r.coach_summary) > 0
    assert isinstance(r.metrics_detail, list)


def test_coach_summary_avoids_medical_keywords() -> None:
    doc = build_report_document(
        video_id="v1",
        athlete_id="A1",
        session_type="normal",
        fatigue_state="no",
        event_group="100",
        created_at="2026-03-23",
        metric_values=CUR,
        previous_metric_values=PREV,
    )
    text = doc.coach_summary
    banned = ("伤病", "损伤", "医疗", "诊断", "处方", "治疗")
    for w in banned:
        assert w not in text
