from typing import List, Optional

from pydantic import BaseModel, Field


class Metrics(BaseModel):
    step_rate: float
    trunk_lean_mean: float
    arm_swing_variability: float
    left_right_timing_diff: float
    tech_stability_score: float


class MetricExplainItem(BaseModel):
    key: str
    value: float
    explanation: str
    confidence: float
    used_frames: int
    used_joints: List[str] = Field(default_factory=list)
    available: bool = True


class Comparison(BaseModel):
    step_rate: float
    trunk_lean_mean: float
    arm_swing_variability: float
    left_right_timing_diff: float
    tech_stability_score: float


class VideoInfo(BaseModel):
    """Session / video context for the report (training use only)."""

    video_id: str
    athlete_id: str
    session_type: str
    fatigue_state: str
    event_group: str
    created_at: str
    duration_ms: Optional[float] = None
    fps: Optional[float] = None


class AIAnalysisItem(BaseModel):
    """AI analysis result for a single video."""

    report_title: Optional[str] = None
    report_text: Optional[str] = None
    report_json: Optional[dict] = None
    ai_summary: str = ""
    evidence_trace: List[str] = Field(default_factory=list)
    key_findings: List[dict] = Field(default_factory=list)
    metric_interpretations: List[dict] = Field(default_factory=list)
    risk_flags: List[dict] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    confidence_statement: str = ""
    recommended_next_steps: List[str] = Field(default_factory=list)
    generated_at: Optional[str] = None
    data_quality_grade: Optional[str] = None
    is_fallback: bool = False


class ReportResponse(BaseModel):
    """
    Flat shape for GET /api/reports/{video_id} (mini-program compatible).

    Includes nested video_info + coach_summary + warnings for Task 4 report layer.
    Field names metrics / comparison are unchanged for existing clients.
    """

    video_id: str
    athlete_id: str
    session_type: str
    fatigue_state: str
    event_group: str
    created_at: str
    metrics: Metrics
    comparison: Comparison
    metrics_detail: List[MetricExplainItem] = Field(default_factory=list)
    video_info: VideoInfo
    coach_summary: str = Field(..., max_length=2000)
    warnings: List[str] = Field(default_factory=list)
    analysis_overview: Optional[dict] = None
    natural_language: Optional[dict] = None
    raw_feature_summary: Optional[dict] = None
    feature_groups: Optional[dict] = None
    metric_confidence: Optional[dict] = None
    used_frames: Optional[dict] = None
    used_joints: Optional[dict] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    pose_summary: Optional[dict] = None
    qc_summary: Optional[dict] = None
    metrics_available: List[str] = Field(default_factory=list)
    suggested_fix: Optional[str] = None
    saved_to_history: Optional[bool] = None
    saved_at: Optional[str] = None
    ai_analysis: Optional[AIAnalysisItem] = None
