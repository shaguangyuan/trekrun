"""
Assemble coach-facing report JSON (training feedback only).

Outputs a logical document with:
  video_info, metric_values, comparison_values, coach_summary, warnings

Maps to API ReportResponse for backward-compatible flat fields (metrics, comparison, ...).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from app.schemas.report import AIAnalysisItem, Comparison, MetricExplainItem, Metrics, ReportResponse, VideoInfo

# Keys must match metric_definition.md / frontend METRIC_META + tech_stability_score
METRIC_KEYS: Sequence[str] = (
    "step_rate",
    "trunk_lean_mean",
    "arm_swing_variability",
    "left_right_timing_diff",
    "tech_stability_score",
)


class ReportDocument(BaseModel):
    """Canonical nested report JSON (e.g. for storage or logs)."""

    video_info: VideoInfo
    metric_values: Metrics
    comparison_values: Comparison
    metrics_detail: List[MetricExplainItem] = Field(default_factory=list)
    coach_summary: str = Field(..., max_length=2000)
    warnings: List[str] = Field(default_factory=list)
    analysis_overview: Optional[Dict[str, Any]] = None
    natural_language: Optional[Dict[str, Any]] = None
    raw_feature_summary: Optional[Dict[str, Any]] = None
    feature_groups: Optional[Dict[str, Any]] = None
    metric_confidence: Optional[Dict[str, Any]] = None
    used_frames: Optional[Dict[str, Any]] = None
    used_joints: Optional[Dict[str, Any]] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    pose_summary: Optional[Dict[str, Any]] = None
    qc_summary: Optional[Dict[str, Any]] = None
    metrics_available: List[str] = Field(default_factory=list)
    suggested_fix: Optional[str] = None
    saved_to_history: Optional[bool] = None
    saved_at: Optional[str] = None
    ai_analysis: Optional[AIAnalysisItem] = None

    def to_nested_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_info.video_id,
            "athlete_id": self.video_info.athlete_id,
            "session_type": self.video_info.session_type,
            "fatigue_state": self.video_info.fatigue_state,
            "event_group": self.video_info.event_group,
            "created_at": self.video_info.created_at,
            "video_info": self.video_info.model_dump(),
            "metric_values": self.metric_values.model_dump(),
            "comparison_values": self.comparison_values.model_dump(),
            "metrics_detail": [m.model_dump() for m in self.metrics_detail],
            "coach_summary": self.coach_summary,
            "warnings": list(self.warnings),
            "analysis_overview": self.analysis_overview,
            "natural_language": self.natural_language,
            "raw_feature_summary": self.raw_feature_summary,
            "feature_groups": self.feature_groups,
            "metric_confidence": self.metric_confidence,
            "used_frames": self.used_frames,
            "used_joints": self.used_joints,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "pose_summary": self.pose_summary,
            "qc_summary": self.qc_summary,
            "metrics_available": list(self.metrics_available),
            "suggested_fix": self.suggested_fix,
            "saved_to_history": self.saved_to_history,
            "saved_at": self.saved_at,
            "ai_analysis": self.ai_analysis.model_dump() if self.ai_analysis else None,
        }

    def to_api_response(self) -> ReportResponse:
        """Flat response shape expected by the mini-program report page."""
        vi = self.video_info
        return ReportResponse(
            video_id=vi.video_id,
            athlete_id=vi.athlete_id,
            session_type=vi.session_type,
            fatigue_state=vi.fatigue_state,
            event_group=vi.event_group,
            created_at=vi.created_at,
            metrics=self.metric_values,
            comparison=self.comparison_values,
            metrics_detail=self.metrics_detail,
            video_info=vi,
            coach_summary=self.coach_summary,
            warnings=list(self.warnings),
            analysis_overview=self.analysis_overview,
            natural_language=self.natural_language,
            raw_feature_summary=self.raw_feature_summary,
            feature_groups=self.feature_groups,
            metric_confidence=self.metric_confidence,
            used_frames=self.used_frames,
            used_joints=self.used_joints,
            failure_stage=self.failure_stage,
            failure_reason=self.failure_reason,
            pose_summary=self.pose_summary,
            qc_summary=self.qc_summary,
            metrics_available=list(self.metrics_available),
            suggested_fix=self.suggested_fix,
            saved_to_history=self.saved_to_history,
            saved_at=self.saved_at,
            ai_analysis=self.ai_analysis,
        )


def _metrics_from_mapping(m: Mapping[str, Any]) -> Metrics:
    return Metrics(
        step_rate=float(m["step_rate"]),
        trunk_lean_mean=float(m["trunk_lean_mean"]),
        arm_swing_variability=float(m["arm_swing_variability"]),
        left_right_timing_diff=float(m["left_right_timing_diff"]),
        tech_stability_score=float(m["tech_stability_score"]),
    )


def _comparison_deltas(
    current: Mapping[str, float],
    previous: Optional[Mapping[str, float]],
) -> tuple[Comparison, List[str], bool]:
    """
    Returns (comparison_deltas, warnings, has_previous).
    If no previous: deltas are 0 and a neutral warning is added.
    """
    warns: List[str] = []
    if previous is None:
        warns.append("暂无同运动员上一次分析记录，变化量暂按 0 展示；下次上传后可对比本次。")
        return (
            Comparison(
                step_rate=0.0,
                trunk_lean_mean=0.0,
                arm_swing_variability=0.0,
                left_right_timing_diff=0.0,
                tech_stability_score=0.0,
            ),
            warns,
            False,
        )
    missing = [k for k in METRIC_KEYS if k not in previous]
    if missing:
        warns.append("上次记录缺少部分指标，变化量仅按可比对项计算；缺失项按 0 展示。")
    deltas = {}
    for k in METRIC_KEYS:
        if k in previous:
            deltas[k] = float(current[k]) - float(previous[k])
        else:
            deltas[k] = 0.0
    return Comparison(**deltas), warns, True  # type: ignore[arg-type]


def _training_warnings(
    metric_values: Metrics,
    has_previous: bool,
) -> List[str]:
    """Non-medical, non-diagnostic training hints only."""
    out: List[str] = []
    if metric_values.left_right_timing_diff > 18.0:
        out.append("左右步时差异偏大，可在慢跑或走跑交替中分步体会两侧节奏。")
    if metric_values.arm_swing_variability > 0.55:
        out.append("摆臂波动偏大，可配合原地摆臂练习，关注幅度与节奏稳定。")
    if has_previous and metric_values.tech_stability_score < 50.0:
        out.append("综合稳定性分数偏低，建议固定机位与拍摄距离，便于连续对比。")
    return out


def _coach_summary(metric_values: Metrics, comp: Comparison, has_previous: bool) -> str:
    """
    Short training feedback only. No medical or injury language.
    """
    ts = metric_values.tech_stability_score
    parts: List[str] = [
        f"本次技术稳定性综合分约 {ts:.0f} 分，可作为后续训练的参照基线。",
    ]
    if not has_previous:
        parts.append("建议保持侧面、全身入镜的拍摄方式，方便下次对照。")
        return " ".join(parts)

    d = comp.tech_stability_score
    if d > 0.5:
        parts.append("综合分较上次略有提升，可延续当前训练安排并继续记录。")
    elif d < -0.5:
        parts.append("综合分较上次略低，可回看视频关注摆臂与左右节奏是否更一致。")
    else:
        parts.append("综合分与上次接近，可继续巩固途中跑技术细节。")

    if comp.step_rate > 0.05:
        parts.append("步频较上次略快，注意在可控范围内完成。")
    elif comp.step_rate < -0.05:
        parts.append("步频较上次略慢，可在热身充分后再录一段对照。")

    return " ".join(parts)


def build_report_document(
    *,
    video_id: str,
    athlete_id: str,
    session_type: str,
    fatigue_state: str,
    event_group: str,
    created_at: str,
    metric_values: Mapping[str, Any],
    previous_metric_values: Optional[Mapping[str, Any]] = None,
    duration_ms: Optional[float] = None,
    fps: Optional[float] = None,
    metrics_detail: Optional[List[Dict[str, Any]]] = None,
    analysis_overview: Optional[Dict[str, Any]] = None,
    natural_language: Optional[Dict[str, Any]] = None,
    raw_feature_summary: Optional[Dict[str, Any]] = None,
    feature_groups: Optional[Dict[str, Any]] = None,
    metric_confidence: Optional[Dict[str, Any]] = None,
    used_frames: Optional[Dict[str, Any]] = None,
    used_joints: Optional[Dict[str, Any]] = None,
    failure_stage: Optional[str] = None,
    failure_reason: Optional[str] = None,
    pose_summary: Optional[Dict[str, Any]] = None,
    qc_summary: Optional[Dict[str, Any]] = None,
    metrics_available: Optional[List[str]] = None,
    suggested_fix: Optional[str] = None,
    saved_to_history: Optional[bool] = None,
    saved_at: Optional[str] = None,
    ai_analysis: Optional[AIAnalysisItem] = None,
) -> ReportDocument:
    """
    Build nested report document + flat API mapping via to_api_response().

    :param previous_metric_values: Last successful metrics for same athlete; None = no comparison.
    """
    cur = _metrics_from_mapping(metric_values)
    prev_model: Optional[Metrics] = None
    if previous_metric_values is not None:
        try:
            prev_model = _metrics_from_mapping(previous_metric_values)
        except (KeyError, TypeError, ValueError):
            prev_model = None

    prev_map = prev_model.model_dump() if prev_model else None
    comp, warn_cmp, has_previous = _comparison_deltas(cur.model_dump(), prev_map)

    vi = VideoInfo(
        video_id=video_id,
        athlete_id=athlete_id,
        session_type=session_type,
        fatigue_state=fatigue_state,
        event_group=event_group,
        created_at=created_at,
        duration_ms=duration_ms,
        fps=fps,
    )

    warnings: List[str] = []
    warnings.extend(warn_cmp)
    warnings.extend(_training_warnings(cur, has_previous))

    summary = _coach_summary(cur, comp, has_previous)

    details: List[MetricExplainItem] = []
    raw_detail = {str((d or {}).get("key")): d for d in (metrics_detail or [])}
    for key in METRIC_KEYS:
        d = raw_detail.get(key, {})
        details.append(
            MetricExplainItem(
                key=key,
                value=float(cur.model_dump()[key]),
                explanation=str(d.get("explanation") or ""),
                confidence=float(d.get("confidence") or 0.0),
                used_frames=int(d.get("used_frames") or 0),
                used_joints=list(d.get("used_joints") or []),
                available=bool(d.get("available", True)),
            )
        )

    return ReportDocument(
        video_info=vi,
        metric_values=cur,
        comparison_values=comp,
        metrics_detail=details,
        coach_summary=summary,
        warnings=warnings,
        analysis_overview=analysis_overview,
        natural_language=natural_language,
        raw_feature_summary=raw_feature_summary,
        feature_groups=feature_groups,
        metric_confidence=metric_confidence,
        used_frames=used_frames,
        used_joints=used_joints,
        failure_stage=failure_stage,
        failure_reason=failure_reason,
        pose_summary=pose_summary,
        qc_summary=qc_summary,
        metrics_available=list(metrics_available or []),
        suggested_fix=suggested_fix,
        saved_to_history=saved_to_history,
        saved_at=saved_at,
        ai_analysis=ai_analysis,
    )


def build_report_response(**kwargs: Any) -> ReportResponse:
    """Convenience: directly return API-shaped model."""
    return build_report_document(**kwargs).to_api_response()
