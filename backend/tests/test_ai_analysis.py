"""
Tests for AI analysis layer (DeepSeek integration).

Tests cover:
- Input builder compression
- Fallback generation when DeepSeek unavailable
- AI result structure validation
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import pytest

from app.services.ai.ai_input_builder import build_ai_input, build_user_prompt
from app.services.ai.ai_report_analyzer import (
    _build_fallback_analysis,
    _estimate_data_quality,
    run_ai_analysis,
)
from app.services.ai.deepseek_client import DeepSeekClient, is_configured


def test_build_ai_input_structure() -> None:
    """Test that AI input builder creates correct 4-layer structure."""
    feature_groups = {
        "pose_geometry": {
            "trunk_lean_angle": 8.5,
            "left_knee_angle": 145.0,
            "right_knee_angle": 142.0,
        },
        "temporal": {
            "cadence": 185.0,
            "gait_cycle_duration": 0.65,
        },
        "symmetry": {
            "left_right_angle_difference": {"knee": 3.0},
            "left_right_timing_difference": 5.2,
        },
        "qc": {
            "total_frames": 180,
            "valid_pose_frames": 156,
            "pose_coverage": 0.87,
            "core_joint_valid_ratio": 0.82,
            "candidate_segments_count": 2,
            "selected_segment_duration": 1200,
            "interpolated_frame_ratio": 0.05,
            "jitter_score": 0.03,
        },
    }

    result = build_ai_input(
        video_id="test-video-001",
        feature_groups=feature_groups,
        raw_feature_summary={
            "total_frames": 180,
            "pose_detected_frames": 156,
            "pose_coverage": 0.87,
            "core_joint_valid_ratio": 0.82,
            "fps": 30.0,
            "duration_ms": 6000,
        },
        metrics_available=["step_rate", "trunk_lean_mean", "arm_swing_variability"],
        analysis_state="done",
    )

    # Verify 4-layer structure
    assert "layer_1_mediapipe_raw" in result
    assert "layer_2_derived_features" in result
    assert "layer_3_quality_control" in result
    assert "layer_4_explanation" in result

    # Verify layer 1 content
    l1 = result["layer_1_mediapipe_raw"]
    assert l1["total_frames"] == 180
    assert l1["pose_coverage"] == 0.87
    assert l1["fps"] == 30.0

    # Verify layer 3 quality grade
    l3 = result["layer_3_quality_control"]
    assert "data_quality_grade" in l3
    assert l3["qc_status"] == "passed"

    # Verify video ID preserved
    assert result["video_id"] == "test-video-001"


def test_build_user_prompt_content() -> None:
    """Test that user prompt contains all 4 layers in text."""
    ai_input = {
        "video_id": "test-video",
        "analysis_state": "done",
        "layer_1_mediapipe_raw": {"pose_coverage": 0.85},
        "layer_2_derived_features": {"core_running_metrics": []},
        "layer_3_quality_control": {"data_quality_grade": "high"},
        "layer_4_explanation": {"analysis_summary": "Test summary"},
    }

    prompt = build_user_prompt(ai_input)

    # Verify prompt contains layer headers
    assert "第一层" in prompt or "layer_1" in prompt
    assert "第二层" in prompt or "layer_2" in prompt
    assert "第三层" in prompt or "layer_3" in prompt
    assert "第四层" in prompt or "layer_4" in prompt

    # Verify video ID in prompt
    assert "test-video" in prompt


def test_estimate_data_quality_high() -> None:
    """Test data quality estimation for high quality data."""
    grade = _estimate_data_quality(
        {"pose_coverage": 0.85, "core_joint_valid_ratio": 0.75},
        {"step_rate": 0.9, "trunk_lean_mean": 0.88},
        ["step_rate", "trunk_lean_mean", "arm_swing_variability", "left_right_timing_diff", "tech_stability_score"],
    )
    assert grade == "high"


def test_estimate_data_quality_insufficient() -> None:
    """Test data quality estimation for insufficient data."""
    grade = _estimate_data_quality(
        {"pose_coverage": 0.2, "core_joint_valid_ratio": 0.15},
        {},
        [],
    )
    assert grade == "insufficient"


def test_estimate_data_quality_from_frame_counts() -> None:
    """pose_coverage can be derived from pose_detected_frames / total_frames when explicit key missing."""
    grade = _estimate_data_quality(
        {
            "total_frames": 100,
            "pose_detected_frames": 85,
            "core_joint_valid_ratio": 0.75,
        },
        {},
        [
            "step_rate",
            "trunk_lean_mean",
            "arm_swing_variability",
            "left_right_timing_diff",
            "tech_stability_score",
        ],
    )
    assert grade == "high"


def test_fallback_analysis_structure() -> None:
    """Test that fallback analysis has all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock settings
        import app.config
        original_dir = app.config.settings.upload_dir
        app.config.settings.upload_dir = tmpdir

        try:
            result = _build_fallback_analysis(
                video_id="test-fallback",
                analysis_state="partial",
                warnings=["Low coverage"],
                data_quality="medium",
            )

            # Verify required fields
            assert "ai_summary" in result
            assert "evidence_trace" in result
            assert "key_findings" in result
            assert "metric_interpretations" in result
            assert "risk_flags" in result
            assert "limitations" in result
            assert "suggestions" in result
            assert "confidence_statement" in result
            assert "recommended_next_steps" in result
            assert "_metadata" in result
            assert "report_title" in result
            assert "report_text" in result
            assert isinstance(result.get("report_json"), dict)
            assert "summary" in result["report_json"]

            # Verify metadata
            meta = result["_metadata"]
            assert meta["video_id"] == "test-fallback"
            assert meta["data_quality_grade"] == "medium"
            assert meta["is_fallback"] is True

            # Verify file saved
            assert os.path.exists(os.path.join(tmpdir, "test-fallback_ai_analysis.json"))

        finally:
            app.config.settings.upload_dir = original_dir


def test_fallback_analysis_done_state() -> None:
    """Test fallback analysis for 'done' state."""
    result = _build_fallback_analysis(
        video_id="test-done",
        analysis_state="done",
        warnings=[],
        data_quality="high",
    )

    # Should have positive tone
    assert "完成" in result["ai_summary"]
    assert len(result["key_findings"]) > 0


def test_fallback_analysis_failed_state() -> None:
    """Test fallback analysis for 'failed' state."""
    result = _build_fallback_analysis(
        video_id="test-failed",
        analysis_state="failed",
        warnings=["Error occurred"],
        data_quality="insufficient",
    )

    # Should indicate failure
    assert "未能" in result["ai_summary"] or "失败" in result["ai_summary"]
    assert len(result["risk_flags"]) > 0


def test_deepseek_client_is_configured_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that is_configured returns False when settings have no API key."""
    import app.config as app_config

    monkeypatch.setattr(app_config.settings, "deepseek_api_key", "")
    assert is_configured() is False


def test_deepseek_client_creation() -> None:
    """Test DeepSeek client can be created with explicit config."""
    client = DeepSeekClient(
        api_key="sk-test-key",
        base_url="https://test.example.com",
        model="test-model",
        timeout=30,
    )

    assert client.api_key == "sk-test-key"
    assert client.base_url == "https://test.example.com"
    assert client.model == "test-model"
    assert client.timeout == 30


def test_run_ai_analysis_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that AI analysis falls back gracefully when not configured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        import app.config
        original_dir = app.config.settings.upload_dir
        app.config.settings.upload_dir = tmpdir
        monkeypatch.setattr(app.config.settings, "deepseek_api_key", "")

        try:
            result = run_ai_analysis(
                video_id="test-no-config",
                feature_groups={},
                analysis_state="done",
            )

            # Should return fallback
            assert result["_metadata"]["is_fallback"] is True
            assert result["_metadata"]["model_name"] == "fallback"

        finally:
            app.config.settings.upload_dir = original_dir
