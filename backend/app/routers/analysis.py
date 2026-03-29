from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.services.job_store import read_job

router = APIRouter(tags=["analysis"])


@router.get("/analysis/{video_id}/status")
def get_analysis_status(video_id: str) -> Dict[str, Any]:
    """
    返回纯 dict，避免 response_model + Pydantic 在部分环境下省略可选字段，
    导致小程序只收到 video_id / task_id / status 三条。
    """
    job = read_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown video_id.")

    status = str(job.get("status") or "queued")
    out: Dict[str, Any] = {
        "video_id": video_id,
        "task_id": str(job.get("task_id") or ""),
        "status": status,
        "analysis_state": job.get("analysis_state") or status,
        "failure_stage": job.get("failure_stage"),
        "failure_reason": job.get("failure_reason"),
        "pose_summary": job.get("pose_summary"),
        "qc_summary": job.get("qc_summary"),
        "metrics_available": job.get("metrics_available") or [],
        "metric_details": job.get("metric_details") or [],
        "analysis_overview": job.get("analysis_overview"),
        "raw_feature_summary": job.get("raw_feature_summary"),
        "feature_groups": job.get("feature_groups"),
        "metric_confidence": job.get("metric_confidence"),
        "used_frames": job.get("used_frames"),
        "used_joints": job.get("used_joints"),
        "warnings": job.get("warnings") or [],
        "suggested_fix": job.get("suggested_fix"),
    }

    if status == "failed":
        raw = job.get("error")
        text = (str(raw).strip() if raw is not None else "") or (
            "分析失败，请查看后端 uploads 目录下同名 .job.json 中的 error 字段。"
        )
        # 多字段冗余：部分客户端对 key 名敏感
        out["hint"] = text
        out["failure_reason"] = text
        out["error"] = text

    return out
