"""
AI Input Builder: Compress 4-layer analysis data into structured prompt for DeepSeek.

Layers:
1. MediaPipe raw summary (no per-frame landmarks)
2. Derived features (geometry, temporal, symmetry, QC)
3. Quality control layer
4. Explanation layer

Output: Structured, compressed JSON suitable for LLM consumption.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def build_ai_input(
    *,
    video_id: str,
    feature_groups: Optional[Mapping[str, Any]] = None,
    raw_feature_summary: Optional[Mapping[str, Any]] = None,
    analysis_overview: Optional[Mapping[str, Any]] = None,
    natural_language: Optional[Mapping[str, Any]] = None,
    metric_confidence: Optional[Mapping[str, float]] = None,
    used_frames: Optional[Mapping[str, int]] = None,
    used_joints: Optional[Mapping[str, List[str]]] = None,
    metrics_available: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    suggested_fix: Optional[str] = None,
    analysis_state: str = "done",
) -> Dict[str, Any]:
    """
    Build structured AI input from all 4 analysis layers.

    This compresses large data structures into summary statistics
    suitable for LLM prompt context.
    """
    fg = dict(feature_groups or {})
    ov = dict(analysis_overview or {})
    nl = dict(natural_language or {})
    rfs = dict(raw_feature_summary or {})

    pose_resolution = dict((pose_summary := dict(ov.get("pose_summary") or {})).get("resolution") or {})
    video_meta_from_pose_summary: Dict[str, Any] = {
        "width": int(pose_resolution.get("width") or 0),
        "height": int(pose_resolution.get("height") or 0),
        "aspect_ratio_wh": float((pose_resolution.get("width") or 0)) / float((pose_resolution.get("height") or 1)),
    }

    # Layer 1: MediaPipe Raw Summary
    pose_summary = dict(ov.get("pose_summary") or {})
    layer1 = {
        "total_frames": int(pose_summary.get("total_frames") or rfs.get("total_frames") or 0),
        "valid_pose_frames": int(pose_summary.get("frames_with_pose") or rfs.get("pose_detected_frames") or 0),
        # analysis_overview uses pose_ratio (feature_extractor + analysis_runner), keep backward compatible with pose_coverage.
        "pose_coverage": float(
            pose_summary.get("pose_coverage")
            or pose_summary.get("pose_ratio")
            or rfs.get("pose_coverage")
            or 0.0
        ),
        "core_joint_valid_ratio": float(pose_summary.get("core_joint_valid_ratio") or rfs.get("core_joint_valid_ratio") or 0.0),
        "fps": float(rfs.get("fps") or ov.get("fps") or 0.0),
        "duration_sec": float(rfs.get("duration_ms") or ov.get("duration_ms") or 0) / 1000.0,
        "resolution": video_meta_from_pose_summary,
        "selected_segment": ov.get("selected_segment") or {},
    }

    # Layer 2: Derived Features (selective, compressed)
    geometry = dict(fg.get("pose_geometry") or {})
    temporal = dict(fg.get("temporal") or {})
    symmetry = dict(fg.get("symmetry") or {})

    temporal_summary = _summarize_temporal(temporal)
    symmetry_summary = _summarize_symmetry(symmetry)
    joint_angles_summary = _summarize_joint_angles(geometry)

    layer2 = {
        "core_running_metrics": _extract_core_metrics(fg, metric_confidence, used_frames, used_joints),
        "joint_angles_summary": joint_angles_summary,
        "temporal_summary": temporal_summary,
        "symmetry_summary": symmetry_summary,
    }

    # Layer 3: Quality Control
    qc = dict(fg.get("qc") or {})
    qc_summary = dict(ov.get("qc_summary") or {})

    # Select important joints for QC summary
    per_joint_valid = dict(qc.get("per_joint_valid_ratio") or {})
    core_joints = [
        "LEFT_SHOULDER",
        "RIGHT_SHOULDER",
        "LEFT_HIP",
        "RIGHT_HIP",
        "LEFT_KNEE",
        "RIGHT_KNEE",
        "LEFT_ANKLE",
        "RIGHT_ANKLE",
        # foot/strike proxy landmarks
        "LEFT_HEEL",
        "RIGHT_HEEL",
        "LEFT_FOOT_INDEX",
        "RIGHT_FOOT_INDEX",
    ]
    core_valid_ratios = {k: float(per_joint_valid.get(k, 0.0)) for k in core_joints if k in per_joint_valid}

    keypoint_qc_summary = dict(qc_summary.get("keypoint_qc_summary") or {})
    low_confidence_metrics = _find_low_confidence_metrics(metric_confidence, threshold=0.5)

    layer3 = {
        "qc_status": "passed" if analysis_state in ("done", "partial") else "failed",
        "quality_level": qc_summary.get("quality_level") or "unknown",
        "data_quality_grade": _compute_quality_grade(layer1["pose_coverage"], layer1["core_joint_valid_ratio"], len(metrics_available or [])),
        "candidate_segments_count": int(qc.get("candidate_segments_count") or 0),
        "selected_segment_duration_sec": float(qc.get("selected_segment_duration") or 0) / 1000.0,
        "interpolated_frame_ratio": float(qc.get("interpolated_frame_ratio") or 0.0),
        "pose_detected_frames": int(keypoint_qc_summary.get("frames_passing_keypoint_qc") or 0),
        "low_visibility_ratio_mean": float(keypoint_qc_summary.get("mean_low_visibility_ratio") or 0.0),
        "missing_ratio_mean": float(keypoint_qc_summary.get("mean_missing_ratio") or 0.0),
        "pass_ratio": float(keypoint_qc_summary.get("pass_ratio") or 0.0),
        "max_qc_gap_frames": int(keypoint_qc_summary.get("max_qc_gap_frames") or 0),
        "core_joint_valid_ratios": core_valid_ratios,
        "jitter_score": float(qc.get("jitter_score") or 0.0),
        "low_confidence_metrics": low_confidence_metrics,
        "warning_count": len(warnings or []),
        "data_quality_anomaly_checks": _build_data_quality_anomaly_checks(
            layer1=layer1,
            temporal_summary=temporal_summary,
            symmetry_summary=symmetry_summary,
            joint_angles_summary=joint_angles_summary,
            metric_confidence=metric_confidence,
            low_confidence_metrics=low_confidence_metrics,
            interpolated_frame_ratio=float(qc.get("interpolated_frame_ratio") or 0.0),
            jitter_score=float(qc.get("jitter_score") or 0.0),
            keypoint_qc_summary=keypoint_qc_summary,
        ),
    }

    # Layer 4: Explanation Context
    metric_explanations = list(nl.get("metric_explanations") or [])
    layer4 = {
        "analysis_summary": nl.get("summary") or "",
        "selected_segment_explanation": nl.get("selected_segment_explanation") or "",
        "process_steps": list(nl.get("process_steps") or []),
        "metric_explanations": [
            {
                "key": m.get("key"),
                "label": m.get("label"),
                "confidence": m.get("confidence"),
                "available": m.get("available"),
            }
            for m in metric_explanations
        ],
        "warnings": list(warnings or []),
        "suggested_fix": suggested_fix,
    }

    return {
        "video_id": video_id,
        "analysis_state": analysis_state,
        "metrics_available": list(metrics_available or []),
        "layer_1_mediapipe_raw": layer1,
        "layer_2_derived_features": layer2,
        "layer_3_quality_control": layer3,
        "layer_4_explanation": layer4,
    }


def _build_data_quality_anomaly_checks(
    *,
    layer1: Dict[str, Any],
    temporal_summary: Dict[str, Any],
    symmetry_summary: Dict[str, Any],
    joint_angles_summary: Dict[str, Any],
    metric_confidence: Optional[Mapping[str, float]],
    low_confidence_metrics: List[str],
    interpolated_frame_ratio: float,
    jitter_score: float,
    keypoint_qc_summary: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Produce deterministic "anomaly checks" for the LLM.

    IMPORTANT: This only flags potential issues; it never proves data errors.
    """
    out: List[Dict[str, Any]] = []

    pose_coverage = float(layer1.get("pose_coverage") or 0.0)
    core_joint_valid_ratio = float(layer1.get("core_joint_valid_ratio") or 0.0)
    cadence = float(temporal_summary.get("cadence") or 0.0)
    trunk_lean_mean = float(joint_angles_summary.get("trunk_lean_mean") or 0.0)
    timing_diff_percent = float(symmetry_summary.get("left_right_timing_diff_percent") or 0.0)

    conf_step_rate = float(metric_confidence.get("step_rate") or 0.0) if metric_confidence else None
    conf_trunk = float(metric_confidence.get("trunk_lean_mean") or 0.0) if metric_confidence else None
    conf_timing = float(metric_confidence.get("left_right_timing_diff") or 0.0) if metric_confidence else None

    # Data quality / landmark stability
    out.append(
        {
            "category": "data_quality",
            "metric": "pose_coverage",
            "value": pose_coverage,
            "rule": "pose_coverage >= 0.8 (high), >=0.5 (medium)",
            "severity": "high" if pose_coverage < 0.45 else ("medium" if pose_coverage < 0.7 else "low"),
            "out_of_range": pose_coverage < 0.5,
        }
    )
    out.append(
        {
            "category": "data_quality",
            "metric": "core_joint_valid_ratio",
            "value": core_joint_valid_ratio,
            "rule": "core_joint_valid_ratio >= 0.7 (high), >=0.5 (medium)",
            "severity": "high" if core_joint_valid_ratio < 0.4 else ("medium" if core_joint_valid_ratio < 0.6 else "low"),
            "out_of_range": core_joint_valid_ratio < 0.5,
        }
    )

    out.append(
        {
            "category": "data_quality",
            "metric": "interpolated_frame_ratio_proxy",
            "value": interpolated_frame_ratio,
            "rule": "interpolated_frame_ratio (higher => more unstable / missing frames)",
            "severity": "high" if interpolated_frame_ratio > 0.25 else ("medium" if interpolated_frame_ratio > 0.1 else "low"),
            "out_of_range": interpolated_frame_ratio > 0.2,
        }
    )
    out.append(
        {
            "category": "data_quality",
            "metric": "jitter_score_proxy",
            "value": jitter_score,
            "rule": "jitter_score (higher => more trunk-line instability)",
            "severity": "high" if jitter_score > 10 else ("medium" if jitter_score > 3 else "low"),
            "out_of_range": jitter_score > 10,
        }
    )
    out.append(
        {
            "category": "data_quality",
            "metric": "mean_missing_ratio",
            "value": float(keypoint_qc_summary.get("mean_missing_ratio") or 0.0),
            "rule": "mean_missing_ratio should be small; missing keypoints harm temporal event detection",
            "severity": "high" if float(keypoint_qc_summary.get("mean_missing_ratio") or 0.0) > 0.15 else "low",
            "out_of_range": float(keypoint_qc_summary.get("mean_missing_ratio") or 0.0) > 0.15,
        }
    )

    # Metric anomalies (potential physiological implausibility or unit error or counting errors)
    out.append(
        {
            "category": "temporal_or_metric_consistency",
            "metric": "cadence",
            "value": cadence,
            "unit": "步/分钟",
            "rule": "short-sprint typical ~160-200; <150 or >220 => anomaly",
            "severity": "high" if cadence < 150 or cadence > 220 else "low",
            "out_of_range": cadence < 150 or cadence > 220,
            "metric_confidence": conf_step_rate,
            "low_confidence_suspected": ("cadence" in low_confidence_metrics) if low_confidence_metrics else False,
        }
    )
    out.append(
        {
            "category": "pose_angle_range",
            "metric": "trunk_lean_mean",
            "value": trunk_lean_mean,
            "unit": "度",
            "rule": "typical range for sprint acceleration proxy ~5-15; <3 or >20 => anomaly",
            "severity": "high" if trunk_lean_mean < 3 or trunk_lean_mean > 20 else "low",
            "out_of_range": trunk_lean_mean < 3 or trunk_lean_mean > 20,
            "metric_confidence": conf_trunk,
            "low_confidence_suspected": ("trunk_lean_mean" in low_confidence_metrics) if low_confidence_metrics else False,
        }
    )
    out.append(
        {
            "category": "symmetry_timing_consistency",
            "metric": "left_right_timing_diff_percent",
            "value": timing_diff_percent,
            "unit": "%",
            "rule": "symmetry timing diff should be small; >15% => anomaly / internal contradiction",
            "severity": "high" if timing_diff_percent > 15 else ("medium" if timing_diff_percent > 10 else "low"),
            "out_of_range": timing_diff_percent > 15,
            "metric_confidence": conf_timing,
            "low_confidence_suspected": ("left_right_timing_diff" in low_confidence_metrics) if low_confidence_metrics else False,
        }
    )

    return out


def _extract_core_metrics(
    feature_groups: Mapping[str, Any],
    metric_confidence: Optional[Mapping[str, float]],
    used_frames: Optional[Mapping[str, int]],
    used_joints: Optional[Mapping[str, List[str]]],
) -> List[Dict[str, Any]]:
    """Extract core running metrics with metadata from feature groups."""
    geometry = dict(feature_groups.get("pose_geometry") or {})
    temporal = dict(feature_groups.get("temporal") or {})
    symmetry = dict(feature_groups.get("symmetry") or {})

    metrics: List[Dict[str, Any]] = []

    # Step rate / cadence
    cadence = temporal.get("cadence")
    if cadence is not None:
        metrics.append({
            "name": "cadence",
            "label": "步频",
            "value": float(cadence),
            "unit": "步/分钟",
            "reference_range": {"min": 160, "max": 200, "note": "短跑通常较高"},
            "confidence": metric_confidence.get("step_rate") if metric_confidence else None,
            "used_frames": used_frames.get("step_rate") if used_frames else None,
            "used_joints": used_joints.get("step_rate") if used_joints else ["left_ankle", "right_ankle"],
        })

    # Trunk lean
    trunk_lean = geometry.get("trunk_lean_angle")
    if trunk_lean is not None:
        metrics.append({
            "name": "trunk_lean",
            "label": "躯干前倾",
            "value": float(trunk_lean),
            "unit": "度",
            "reference_range": {"min": 5, "max": 15, "note": "短跑加速阶段常见范围"},
            "confidence": metric_confidence.get("trunk_lean_mean") if metric_confidence else None,
            "used_frames": used_frames.get("trunk_lean_mean") if used_frames else None,
            "used_joints": used_joints.get("trunk_lean_mean") if used_joints else ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        })

    # ROM summary
    rom = dict(temporal.get("rom") or {})
    for joint, value in rom.items():
        if value is not None and isinstance(value, (int, float)):
            metrics.append({
                "name": f"{joint}_rom",
                "label": f"{joint}关节活动度",
                "value": float(value),
                "unit": "度",
                "reference_range": None,
                "confidence": None,
                "used_frames": None,
                "used_joints": [joint],
            })

    # Symmetry timing diff
    timing_diff = symmetry.get("left_right_timing_difference")
    if timing_diff is not None:
        metrics.append({
            "name": "left_right_timing_diff",
            "label": "左右节律差",
            "value": float(timing_diff),
            "unit": "%",
            "reference_range": {"min": 0, "max": 10, "note": "越低越对称"},
            "confidence": metric_confidence.get("left_right_timing_diff") if metric_confidence else None,
            "used_frames": used_frames.get("left_right_timing_diff") if used_frames else None,
            "used_joints": used_joints.get("left_right_timing_diff") if used_joints else ["left_ankle", "right_ankle"],
        })

    return metrics


def _summarize_joint_angles(geometry: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize joint angle data from geometry features."""
    angles = {}
    for key in ["left_elbow_angle", "right_elbow_angle", "left_hip_angle", "right_hip_angle",
                "left_knee_angle", "right_knee_angle", "left_ankle_angle", "right_ankle_angle"]:
        val = geometry.get(key)
        if val is not None:
            angles[key] = round(float(val), 2)

    return {
        "mean_joint_angles": angles,
        "trunk_lean_mean": round(float(geometry.get("trunk_lean_angle") or 0), 2),
        "trunk_side_tilt": round(float(geometry.get("trunk_side_tilt") or 0), 2),
        # Foot progression proxy (image-plane line angle) - used as an overstride / strike heuristic.
        "left_foot_progression_proxy": round(float(geometry.get("left_foot_progression_proxy") or 0), 2),
        "right_foot_progression_proxy": round(float(geometry.get("right_foot_progression_proxy") or 0), 2),
    }


def _summarize_temporal(temporal: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize temporal features."""
    rom = dict(temporal.get("rom") or {})
    stance_swing = dict(temporal.get("stance_swing_proxy") or {})

    gait_cycle_duration_sec = round(float(temporal.get("gait_cycle_duration") or 0), 3)
    stance_ratio = stance_swing.get("stance_ratio")
    swing_ratio = stance_swing.get("swing_ratio")

    contact_time_proxy_sec = None
    flight_time_proxy_sec = None
    if stance_ratio is not None and gait_cycle_duration_sec:
        contact_time_proxy_sec = round(float(stance_ratio) * float(gait_cycle_duration_sec), 3)
    if swing_ratio is not None and gait_cycle_duration_sec:
        flight_time_proxy_sec = round(float(swing_ratio) * float(gait_cycle_duration_sec), 3)

    return {
        "cadence": round(float(temporal.get("cadence") or 0), 2),
        "gait_cycle_duration_sec": gait_cycle_duration_sec,
        "stance_ratio": stance_ratio,
        "swing_ratio": swing_ratio,
        # Not ground truth contact/flight events (no force plates); purely proxy based on stance/swing ratio.
        "contact_time_proxy_sec": contact_time_proxy_sec,
        "flight_time_proxy_sec": flight_time_proxy_sec,
        "peak_flexion": dict(temporal.get("peak_flexion") or {}),
        "peak_extension": dict(temporal.get("peak_extension") or {}),
        "stride_to_stride_variability": round(float(temporal.get("stride_to_stride_variability") or 0), 4),
    }


def _summarize_symmetry(symmetry: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize symmetry features."""
    angle_diffs = dict(symmetry.get("left_right_angle_difference") or {})

    return {
        "left_right_angle_differences": {k: round(float(v), 2) for k, v in angle_diffs.items() if v is not None},
        "left_right_timing_diff_percent": round(float(symmetry.get("left_right_timing_difference") or 0), 2),
        "waveform_similarity_proxy": round(float(symmetry.get("waveform_similarity_correlation_proxy") or 0), 3),
    }


def _compute_quality_grade(pose_coverage: float, core_valid_ratio: float, metrics_count: int) -> str:
    """Compute overall data quality grade."""
    if pose_coverage >= 0.8 and core_valid_ratio >= 0.7 and metrics_count >= 5:
        return "high"
    elif pose_coverage >= 0.5 and core_valid_ratio >= 0.5 and metrics_count >= 3:
        return "medium"
    elif pose_coverage >= 0.3 and metrics_count >= 1:
        return "low"
    else:
        return "insufficient"


def _find_low_confidence_metrics(metric_confidence: Optional[Mapping[str, float]], threshold: float = 0.5) -> List[str]:
    """Find metrics with confidence below threshold."""
    if not metric_confidence:
        return []
    return [k for k, v in metric_confidence.items() if v is not None and float(v) < threshold]


def build_user_prompt(ai_input: Dict[str, Any]) -> str:
    """
    Convert AI input structure into user-facing prompt text.

    This prompt is designed to guide the LLM to produce structured,
    evidence-based analysis without hallucination.
    """
    return f"""你将收到一份已压缩的结构化短跑视频分析输入（四层：检测摘要/派生特征/质量控制/过程说明）。

请注意：报告主视角必须是“短跑技术分析”，不是算法质检说明。请优先输出技术问题、可能机制、训练建议；数据质量只作为最后的辅助提示（除非数据几乎不可解释）。

请对以下短跑视频分析数据进行专业解读。

视频ID: {ai_input.get("video_id")}
分析状态: {ai_input.get("analysis_state")}
可用指标数: {len(ai_input.get("metrics_available", []))}

【第一层：视频与姿态检测摘要】
{json.dumps(ai_input.get("layer_1_mediapipe_raw"), ensure_ascii=False, indent=2)}

【第二层：派生跑姿特征】
核心指标：
{json.dumps(ai_input.get("layer_2_derived_features", {}).get("core_running_metrics"), ensure_ascii=False, indent=2)}

关节角度摘要：
{json.dumps(ai_input.get("layer_2_derived_features", {}).get("joint_angles_summary"), ensure_ascii=False, indent=2)}

时序特征：
{json.dumps(ai_input.get("layer_2_derived_features", {}).get("temporal_summary"), ensure_ascii=False, indent=2)}

对称性特征：
{json.dumps(ai_input.get("layer_2_derived_features", {}).get("symmetry_summary"), ensure_ascii=False, indent=2)}

【第三层：质量控制信息】
{json.dumps(ai_input.get("layer_3_quality_control"), ensure_ascii=False, indent=2)}

【第四层：分析过程说明】
摘要：{ai_input.get("layer_4_explanation", {}).get("analysis_summary", "")}

选中片段说明：{ai_input.get("layer_4_explanation", {}).get("selected_segment_explanation", "")}

警告列表：{ai_input.get("layer_4_explanation", {}).get("warnings", [])}

请基于以上数据生成结构化分析结果。注意：
1. 所有结论必须有数据支持，不得自由发挥
2. 数据质量有限时必须明确提示不确定性
3. 不输出医学诊断，只提供训练参考
4. 主要发现要使用短跑教练语言（如支撑效率、摆动节奏、单侧控制、推进效率）
5. 每条主要发现尽量包含：动作现象 + 短跑含义 + 表现影响
6. 建议要和发现对应，优先给出专项练习（A/B skip、ankling、小步快跑、加速跑、单腿RDL、分腿蹲等）
7. 若存在“数据质量明显不足”或“核心指标明显冲突且不可解释”，才可提升数据限制优先级，并使用“当前视频对动作判断的支持有限，因此以下结论仅供参考”
8. 拍摄/视频建议最多 1 条，且不允许占据前两条建议
9. 不使用markdown代码块包裹JSON输出
"""


import json  # noqa: F811
