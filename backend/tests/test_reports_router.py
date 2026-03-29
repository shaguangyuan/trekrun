from __future__ import annotations

from typing import Any

from app.routers import reports


def _mock_job() -> dict[str, Any]:
    return {
        "athlete_id": "athlete-001",
        "session_type": "train",
        "fatigue_state": "normal",
        "event_group": "100m",
        "created_at": "2026-03-26T10:00:00Z",
        "finished_at": "2026-03-26T10:01:00Z",
        "status": "done",
        "analysis_state": "done",
        "metric_details": [],
        "analysis_overview": {},
        "warnings": [],
        "metrics_available": [
            "step_rate",
            "trunk_lean_mean",
            "arm_swing_variability",
            "left_right_timing_diff",
            "tech_stability_score",
        ],
    }


def _mock_metrics() -> dict[str, float]:
    return {
        "step_rate": 4.2,
        "trunk_lean_mean": 12.5,
        "arm_swing_variability": 0.18,
        "left_right_timing_diff": 3.1,
        "tech_stability_score": 82.0,
    }


def test_get_report_returns_without_ai_cache(monkeypatch) -> None:
    monkeypatch.setattr(reports, "read_job", lambda video_id: _mock_job())
    monkeypatch.setattr(reports, "read_metrics", lambda video_id: _mock_metrics())
    monkeypatch.setattr(reports, "get_previous_metrics", lambda athlete_id, video_id: None)
    monkeypatch.setattr(reports, "read_ai_analysis", lambda video_id: None)
    monkeypatch.setattr(
        reports,
        "build_natural_language_explanation",
        lambda **kwargs: {"metric_explanations": [], "warnings": []},
    )

    result = reports.get_report("video-001")

    assert result.video_id == "video-001"
    assert result.metrics.tech_stability_score == 82.0
    assert result.ai_analysis is not None
    assert result.ai_analysis.is_fallback is True


def test_get_report_uses_cached_ai_analysis(monkeypatch) -> None:
    monkeypatch.setattr(reports, "read_job", lambda video_id: _mock_job())
    monkeypatch.setattr(reports, "read_metrics", lambda video_id: _mock_metrics())
    monkeypatch.setattr(reports, "get_previous_metrics", lambda athlete_id, video_id: None)
    monkeypatch.setattr(
        reports,
        "read_ai_analysis",
        lambda video_id: {
            "report_title": "Cached AI",
            "ai_summary": "cached summary",
            "evidence_trace": ["cached evidence"],
            "_metadata": {
                "generated_at": "2026-03-26T10:02:00Z",
                "data_quality_grade": "high",
                "is_fallback": True,
            },
        },
    )
    monkeypatch.setattr(
        reports,
        "build_natural_language_explanation",
        lambda **kwargs: {"metric_explanations": [], "warnings": []},
    )

    result = reports.get_report("video-001")

    assert result.ai_analysis is not None
    assert result.ai_analysis.report_title == "Cached AI"
    assert result.ai_analysis.ai_summary == "cached summary"
    assert result.ai_analysis.is_fallback is True
