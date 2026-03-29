import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from app.schemas.report import AIAnalysisItem, ReportResponse
from app.services.ai import read_ai_analysis
from app.services.ai.ai_report_analyzer import _build_fallback_analysis, _estimate_data_quality

logger = logging.getLogger(__name__)
from app.services.job_store import get_previous_metrics, read_job, read_metrics, save_analysis_to_history
from app.services.report_explainer import build_natural_language_explanation
from app.services.report_builder import build_report_response

router = APIRouter(tags=["reports"])


@router.get("/reports/{video_id}", response_model=ReportResponse)
def get_report(video_id: str) -> ReportResponse:
    job = read_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown video_id.")

    status = job.get("status")
    if status in ("queued", "processing"):
        raise HTTPException(status_code=404, detail="Analysis not complete yet.")
    if status == "failed":
        stage = job.get("failure_stage")
        err = job.get("failure_reason") or job.get("error") or "Analysis failed."
        if stage:
            err = f"[{stage}] {err}"
        raise HTTPException(status_code=404, detail=err)

    metrics = read_metrics(video_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not found.")

    athlete_id = str(job["athlete_id"])
    prev = get_previous_metrics(athlete_id, video_id)
    finished = job.get("finished_at") or job.get("created_at") or ""
    created_at = str(finished)[:10] if finished else ""

    # Prefer persisted artifacts; fallback to job inline fields.
    feature_artifact = _read_upload_json(f"{video_id}_features.json")
    explanation_artifact = _read_upload_json(f"{video_id}_explanation.json")

    # Read cached AI analysis if available; otherwise build a local fallback
    # so the report always includes ai_analysis without a second request.
    ai_raw = read_ai_analysis(video_id)
    if not ai_raw:
        ai_raw = _build_fallback_analysis(
            video_id=video_id,
            analysis_state=str(job.get("analysis_state") or "done"),
            warnings=job.get("warnings") or [],
            data_quality=_estimate_data_quality(
                feature_artifact.get("raw_feature_summary") or job.get("raw_feature_summary"),
                feature_artifact.get("metric_confidence") or job.get("metric_confidence"),
                job.get("metrics_available"),
            ),
        )
    ai_item = _safe_build_ai_item(ai_raw)

    return build_report_response(
        video_id=video_id,
        athlete_id=athlete_id,
        session_type=str(job["session_type"]),
        fatigue_state=str(job["fatigue_state"]),
        event_group=str(job["event_group"]),
        created_at=created_at,
        metric_values=metrics,
        previous_metric_values=prev,
        duration_ms=job.get("duration_ms"),
        fps=job.get("fps"),
        metrics_detail=_build_metric_detail_with_explanation(
            metrics,
            job.get("metric_details") or [],
            build_natural_language_explanation(
                video_id=video_id,
                analysis_state=str(job.get("analysis_state") or "done"),
                analysis_overview=job.get("analysis_overview") or {},
                metric_details=job.get("metric_details") or [],
                warnings=job.get("warnings") or [],
                suggested_fix=job.get("suggested_fix"),
            ),
        ),
        analysis_overview=feature_artifact.get("analysis_overview") or job.get("analysis_overview"),
        natural_language=explanation_artifact or build_natural_language_explanation(
            video_id=video_id,
            analysis_state=str(job.get("analysis_state") or "done"),
            analysis_overview=(feature_artifact.get("analysis_overview") or job.get("analysis_overview") or {}),
            metric_details=job.get("metric_details") or [],
            warnings=job.get("warnings") or [],
            suggested_fix=job.get("suggested_fix"),
        ),
        raw_feature_summary=feature_artifact.get("raw_feature_summary") or job.get("raw_feature_summary"),
        feature_groups=feature_artifact.get("feature_groups") or job.get("feature_groups"),
        metric_confidence=feature_artifact.get("metric_confidence") or job.get("metric_confidence"),
        used_frames=feature_artifact.get("used_frames") or job.get("used_frames"),
        used_joints=feature_artifact.get("used_joints") or job.get("used_joints"),
        failure_stage=job.get("failure_stage"),
        failure_reason=job.get("failure_reason"),
        pose_summary=job.get("pose_summary"),
        qc_summary=job.get("qc_summary"),
        metrics_available=job.get("metrics_available") or [],
        suggested_fix=job.get("suggested_fix"),
        saved_to_history=bool(job.get("saved_to_history")),
        saved_at=job.get("saved_at"),
        ai_analysis=ai_item,
    )


@router.post("/reports/{video_id}/save")
def save_report_to_history(video_id: str) -> Dict[str, Any]:
    """Manual save: only now write this run into athlete history/trend."""
    result = save_analysis_to_history(video_id)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("message", "Save failed."))
    return result


def _coerce_list_of_dicts(items: Any) -> list:
    """Ensure items is a list of dicts; convert strings to wrapper dicts."""
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"text": item})
    return out


def _safe_build_ai_item(ai_data: Optional[Dict[str, Any]]) -> Optional[AIAnalysisItem]:
    """Build AIAnalysisItem from cached data, tolerating schema mismatches."""
    if not ai_data:
        return None
    try:
        return AIAnalysisItem(
            report_title=ai_data.get("report_title"),
            report_text=ai_data.get("report_text"),
            report_json=ai_data.get("report_json") if isinstance(ai_data.get("report_json"), dict) else None,
            ai_summary=ai_data.get("ai_summary", ""),
            evidence_trace=ai_data.get("evidence_trace", []),
            key_findings=_coerce_list_of_dicts(ai_data.get("key_findings")),
            metric_interpretations=_coerce_list_of_dicts(ai_data.get("metric_interpretations")),
            risk_flags=_coerce_list_of_dicts(ai_data.get("risk_flags")),
            limitations=ai_data.get("limitations", []),
            suggestions=ai_data.get("suggestions", []),
            confidence_statement=ai_data.get("confidence_statement", ""),
            recommended_next_steps=ai_data.get("recommended_next_steps", []),
            generated_at=ai_data.get("_metadata", {}).get("generated_at"),
            data_quality_grade=ai_data.get("_metadata", {}).get("data_quality_grade"),
            is_fallback=ai_data.get("_metadata", {}).get("is_fallback", False),
        )
    except Exception as exc:
        logger.warning("Failed to build AIAnalysisItem from cache, skipping: %s", exc)
        return None


def _build_metric_detail_with_explanation(
    metrics: dict,
    raw_details: list,
    nl: dict,
) -> list:
    text_map = {str(m.get("key")): m for m in (nl.get("metric_explanations") or [])}
    out = []
    by_key = {str(d.get("key")): d for d in raw_details}
    for k in ("step_rate", "trunk_lean_mean", "arm_swing_variability", "left_right_timing_diff", "tech_stability_score"):
        d = by_key.get(k, {})
        t = text_map.get(k, {})
        out.append(
            {
                "key": k,
                "value": float(metrics.get(k, 0.0)),
                "explanation": str(t.get("explanation") or ""),
                "confidence": float(d.get("confidence", t.get("confidence", 0.0))),
                "used_frames": int(d.get("used_frames", t.get("used_frames", 0))),
                "used_joints": list(d.get("used_joints", t.get("used_joints", []))),
                "available": bool(d.get("available", t.get("available", True))),
            }
        )
    return out


def _read_upload_json(filename: str) -> Dict[str, Any]:
    path = os.path.join("uploads", filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        return {}
    return {}
