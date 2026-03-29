"""
AI Analysis Router: Endpoints for DeepSeek-powered sprint video analysis.

Provides:
- GET /ai-analysis/{video_id}: Retrieve AI analysis (cached or generate)
- POST /ai-analysis/{video_id}/refresh: Force regenerate AI analysis
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.services.ai import read_ai_analysis, run_ai_analysis
from app.services.job_store import read_job

router = APIRouter(tags=["ai-analysis"])


def _load_report_data(video_id: str) -> Dict[str, Any]:
    """Load all necessary data for AI analysis."""
    job = read_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown video_id.")

    status = job.get("status")
    if status in ("queued", "processing"):
        raise HTTPException(status_code=404, detail="Analysis not complete yet.")

    # Return job data directly - AI analysis doesn't need ReportDocument
    return {
        "job": job,
        "feature_groups": job.get("feature_groups") or {},
        "raw_feature_summary": job.get("raw_feature_summary") or {},
        "analysis_overview": job.get("analysis_overview") or {},
        "natural_language": job.get("natural_language") or {},
        "metric_confidence": job.get("metric_confidence") or {},
        "used_frames": job.get("used_frames") or {},
        "used_joints": job.get("used_joints") or {},
        "metrics_available": job.get("metrics_available") or [],
        "warnings": job.get("warnings") or [],
        "suggested_fix": job.get("suggested_fix"),
    }


def _format_ai_response(video_id: str, ai_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build clean response dict from raw AI result."""
    return {
        "video_id": video_id,
        "report_title": ai_result.get("report_title"),
        "report_text": ai_result.get("report_text"),
        "report_json": ai_result.get("report_json") if isinstance(ai_result.get("report_json"), dict) else None,
        "ai_summary": ai_result.get("ai_summary", ""),
        "evidence_trace": ai_result.get("evidence_trace", []),
        "key_findings": ai_result.get("key_findings", []),
        "metric_interpretations": ai_result.get("metric_interpretations", []),
        "risk_flags": ai_result.get("risk_flags", []),
        "limitations": ai_result.get("limitations", []),
        "suggestions": ai_result.get("suggestions", []),
        "confidence_statement": ai_result.get("confidence_statement", ""),
        "recommended_next_steps": ai_result.get("recommended_next_steps", []),
        "_meta": {
            "generated_at": ai_result.get("_metadata", {}).get("generated_at"),
            "data_quality_grade": ai_result.get("_metadata", {}).get("data_quality_grade"),
            "is_fallback": ai_result.get("_metadata", {}).get("is_fallback", False),
        },
    }


@router.get("/ai-analysis/{video_id}")
def get_ai_analysis(video_id: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get AI analysis for a video.

    - If cached analysis exists and force_refresh=False, return cached
    - Otherwise, generate new analysis using DeepSeek
    """
    # Fast path: return cached result without touching DeepSeek at all.
    if not force_refresh:
        cached = read_ai_analysis(video_id)
        if cached:
            return _format_ai_response(video_id, cached)

    data = _load_report_data(video_id)
    job = data["job"]

    analysis_state = str(job.get("analysis_state") or "done")

    ai_result = run_ai_analysis(
        video_id=video_id,
        feature_groups=data["feature_groups"],
        raw_feature_summary=data["raw_feature_summary"],
        analysis_overview=data["analysis_overview"],
        natural_language=data["natural_language"],
        metric_confidence=data["metric_confidence"],
        used_frames=data["used_frames"],
        used_joints=data["used_joints"],
        metrics_available=data["metrics_available"],
        warnings=data["warnings"],
        suggested_fix=data["suggested_fix"],
        analysis_state=analysis_state,
        force_refresh=force_refresh,
    )

    return _format_ai_response(video_id, ai_result)


@router.post("/ai-analysis/{video_id}/refresh")
def refresh_ai_analysis(video_id: str) -> Dict[str, Any]:
    """Force regenerate AI analysis (clear cache and rerun)."""
    return get_ai_analysis(video_id, force_refresh=True)


@router.get("/ai-analysis/{video_id}/raw")
def get_ai_analysis_raw(video_id: str) -> Dict[str, Any]:
    """Get raw AI analysis including internal metadata (for debugging)."""
    data = _load_report_data(video_id)
    job = data["job"]

    analysis_state = str(job.get("analysis_state") or "done")

    ai_result = run_ai_analysis(
        video_id=video_id,
        feature_groups=data["feature_groups"],
        raw_feature_summary=data["raw_feature_summary"],
        analysis_overview=data["analysis_overview"],
        natural_language=data["natural_language"],
        metric_confidence=data["metric_confidence"],
        used_frames=data["used_frames"],
        used_joints=data["used_joints"],
        metrics_available=data["metrics_available"],
        warnings=data["warnings"],
        suggested_fix=data["suggested_fix"],
        analysis_state=analysis_state,
        force_refresh=False,
    )

    return ai_result
