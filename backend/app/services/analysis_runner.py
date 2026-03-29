"""
Run pose extraction → QC / valid segment → 5 metrics after upload.

Executed via FastAPI BackgroundTasks (same process). Not for production scale.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Tuple

from app.config import settings
from app.services.feature_extractor import extract_feature_groups
from app.services.report_explainer import build_natural_language_explanation

logger = logging.getLogger(__name__)
from app.services.job_store import fail_job, read_job, update_job, write_metrics
from app.services.metrics.arm_swing_variability import compute_arm_swing_variability
from app.services.metrics.errors import MetricComputationError
from app.services.metrics.left_right_timing_diff import compute_left_right_timing_diff
from app.services.metrics.step_rate import compute_step_rate
from app.services.metrics.tech_stability_score import compute_tech_stability_score
from app.services.metrics.trunk_lean_mean import compute_trunk_lean_mean
from app.services.pose_extractor import extract_landmarks
from app.services.qc_thresholds import QCThresholds
from app.services.video_quality_segment import build_valid_segment


def run_full_analysis(
    video_id: str,
    task_id: str,
    dest_path: str,
    athlete_id: str,
    session_type: str,
    fatigue_state: str,
    event_group: str,
) -> None:
    """Background entry point."""
    job = read_job(video_id)
    if not job:
        return

    update_job(video_id, status="processing", task_id=task_id)

    try:
        extraction = extract_landmarks(
            dest_path,
            video_id,
            settings.upload_dir,
        )
        if not extraction.success:
            msg = extraction.error or "Pose extraction failed."
            logger.warning("analysis failed video_id=%s stage=pose_extract: %s", video_id, msg)
            update_job(
                video_id,
                failure_stage="pose_extract",
                failure_reason=msg,
                pose_summary=None,
                qc_summary=None,
                metrics_available=[],
                metric_details=[],
                analysis_overview={
                    "duration_ms": extraction.duration_ms,
                    "fps": extraction.fps,
                    "pose_summary": {},
                    "qc_summary": {},
                    "candidate_segments": [],
                    "selected_segment": {},
                },
                warnings=[],
                suggested_fix="请先确认已下载模型文件，并保证视频可正常解码。",
                analysis_state="failed",
            )
            fail_job(video_id, msg)
            return

        qc = build_valid_segment(extraction, QCThresholds(), output_dir=settings.upload_dir)
        if not qc.success or qc.valid_segment is None:
            msg = qc.error or "Video quality check failed."
            logger.warning("analysis failed video_id=%s stage=qc: %s", video_id, msg)
            update_job(
                video_id,
                failure_stage="qc",
                failure_reason=msg,
                pose_summary={
                    "frames": extraction.total_frames,
                    "frames_with_pose": extraction.frames_with_pose,
                    "pose_ratio": extraction.pose_ratio,
                    "core_joint_valid_ratio": extraction.core_joint_valid_ratio,
                },
                qc_summary={
                    "quality_level": qc.quality_level,
                    "video_meta": qc.video_meta,
                    "keypoint_qc_summary": qc.keypoint_qc_summary,
                    "warnings": qc.warnings,
                },
                metrics_available=[],
                metric_details=[],
                analysis_overview={
                    "duration_ms": extraction.duration_ms,
                    "fps": extraction.fps,
                    "pose_summary": {
                        "total_frames": extraction.total_frames,
                        "frames_with_pose": extraction.frames_with_pose,
                        "pose_ratio": extraction.pose_ratio,
                        "core_joint_valid_ratio": extraction.core_joint_valid_ratio,
                        "resolution": {"width": extraction.width, "height": extraction.height},
                    },
                    "qc_summary": {
                        "quality_level": qc.quality_level,
                        "video_meta": qc.video_meta,
                        "keypoint_qc_summary": qc.keypoint_qc_summary,
                    },
                    "candidate_segments": (qc.segment_debug or {}).get("runs_after_merge", []),
                    "selected_segment": {},
                },
                warnings=qc.warnings,
                suggested_fix=_suggested_fix(qc.error or ""),
                analysis_state="failed",
            )
            fail_job(video_id, msg)
            return

        metrics, available, metric_errors, metric_details = _compute_partial_metrics(extraction, qc.valid_segment)
        warnings: List[str] = list(qc.warnings)
        warnings.extend([f"{k}: {v}" for k, v in metric_errors.items()])
        if not available:
            msg = "No metrics available from current segment."
            logger.warning("analysis failed video_id=%s stage=metrics: %s", video_id, msg)
            update_job(
                video_id,
                failure_stage="metrics",
                failure_reason=msg,
                pose_summary={
                    "frames": extraction.total_frames,
                    "frames_with_pose": extraction.frames_with_pose,
                    "pose_ratio": extraction.pose_ratio,
                    "core_joint_valid_ratio": extraction.core_joint_valid_ratio,
                },
                qc_summary={
                    "quality_level": qc.quality_level,
                    "video_meta": qc.video_meta,
                    "keypoint_qc_summary": qc.keypoint_qc_summary,
                    "warnings": qc.warnings,
                },
                metrics_available=[],
                metric_details=metric_details,
                analysis_overview={
                    "duration_ms": extraction.duration_ms,
                    "fps": extraction.fps,
                    "pose_summary": {
                        "total_frames": extraction.total_frames,
                        "frames_with_pose": extraction.frames_with_pose,
                        "pose_ratio": extraction.pose_ratio,
                        "core_joint_valid_ratio": extraction.core_joint_valid_ratio,
                        "resolution": {"width": extraction.width, "height": extraction.height},
                    },
                    "qc_summary": {
                        "quality_level": qc.quality_level,
                        "video_meta": qc.video_meta,
                        "keypoint_qc_summary": qc.keypoint_qc_summary,
                    },
                    "candidate_segments": (qc.segment_debug or {}).get("runs_after_merge", []),
                    "selected_segment": {
                        "start_ms": qc.valid_segment.start_ms,
                        "end_ms": qc.valid_segment.end_ms,
                        "duration_ms": qc.valid_segment.end_ms - qc.valid_segment.start_ms,
                        "reason": "最长且连续的有效片段",
                    },
                },
                warnings=warnings,
                suggested_fix="请保证侧向全身连续入镜，提升有效片段时长。",
                analysis_state="failed",
            )
            fail_job(video_id, msg)
            return

        feature_bundle = extract_feature_groups(
            extraction=extraction,
            qc=qc,
            metrics=metrics,
            metric_details=metric_details,
            warnings=warnings,
        )
        natural_language = build_natural_language_explanation(
            video_id=video_id,
            analysis_state="partial" if len(available) < 5 else "done",
            analysis_overview=feature_bundle.get("analysis_overview"),
            metric_details=metric_details,
            warnings=warnings,
            suggested_fix=None,
        )

        write_metrics(video_id, metrics)
        finished_at = datetime.now(timezone.utc).isoformat()
        update_job(
            video_id,
            status="done",
            error=None,
            finished_at=finished_at,
            fps=extraction.fps,
            duration_ms=extraction.duration_ms,
            failure_stage=None,
            failure_reason=None,
            pose_summary={
                "frames": extraction.total_frames,
                "frames_with_pose": extraction.frames_with_pose,
                "pose_ratio": extraction.pose_ratio,
                "core_joint_valid_ratio": extraction.core_joint_valid_ratio,
            },
            qc_summary={
                "quality_level": qc.quality_level,
                "video_meta": qc.video_meta,
                "keypoint_qc_summary": qc.keypoint_qc_summary,
                "warnings": qc.warnings,
                "segment": asdict(qc.valid_segment),
            },
            metrics_available=available,
            metric_details=metric_details,
            analysis_overview={
                "duration_ms": extraction.duration_ms,
                "fps": extraction.fps,
                "pose_summary": {
                    "total_frames": extraction.total_frames,
                    "frames_with_pose": extraction.frames_with_pose,
                    "pose_ratio": extraction.pose_ratio,
                    "core_joint_valid_ratio": extraction.core_joint_valid_ratio,
                    "resolution": {"width": extraction.width, "height": extraction.height},
                },
                "qc_summary": {
                    "quality_level": qc.quality_level,
                    "video_meta": qc.video_meta,
                    "keypoint_qc_summary": qc.keypoint_qc_summary,
                },
                "candidate_segments": (qc.segment_debug or {}).get("runs_after_merge", []),
                "selected_segment": {
                    "start_ms": qc.valid_segment.start_ms,
                    "end_ms": qc.valid_segment.end_ms,
                    "duration_ms": qc.valid_segment.end_ms - qc.valid_segment.start_ms,
                    "reason": "合并短缺口后最长连续片段",
                },
            },
            warnings=warnings,
            suggested_fix=None,
            analysis_state="partial" if len(available) < 5 else "done",
            raw_feature_summary=feature_bundle.get("raw_feature_summary"),
            feature_groups=feature_bundle.get("feature_groups"),
            metric_confidence=feature_bundle.get("metric_confidence"),
            used_frames=feature_bundle.get("used_frames"),
            used_joints=feature_bundle.get("used_joints"),
            natural_language=natural_language,
        )
        _save_json(f"{video_id}_features.json", feature_bundle)
        _save_json(
            f"{video_id}_qc.json",
            {
                "video_id": video_id,
                "qc_summary": feature_bundle.get("feature_groups", {}).get("qc", {}),
                "warnings": warnings,
                "analysis_overview": feature_bundle.get("analysis_overview", {}),
            },
        )
        _save_json(
            f"{video_id}_explanation.json",
            {
                "video_id": video_id,
                "summary": natural_language.get("summary"),
                "process_steps": natural_language.get("process_steps", []),
                "selected_segment_explanation": natural_language.get("selected_segment_explanation"),
                "metric_explanations": natural_language.get("metric_explanations", []),
                "warnings": natural_language.get("warnings", []),
                "suggested_fix": natural_language.get("suggested_fix"),
            },
        )

        # History is now user-driven: only saved via POST /api/reports/{video_id}/save.
        update_job(video_id, saved_to_history=False, saved_at=None)
    except Exception as exc:  # noqa: BLE001
        logger.exception("analysis failed video_id=%s stage=unexpected", video_id)
        fail_job(video_id, f"Unexpected analysis error: {exc}")


def _compute_partial_metrics(
    extraction: Any,
    seg: Any,
) -> Tuple[Dict[str, float], List[str], Dict[str, str], List[Dict[str, Any]]]:
    frames = extraction.frames[seg.start_frame_idx : seg.end_frame_idx + 1]
    duration_sec = (seg.end_ms - seg.start_ms) / 1000.0
    fps = float(extraction.fps)
    metric_values: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    detail_map: Dict[str, Dict[str, Any]] = {}
    used_frames = len(frames)

    def _run(name: str, fn: Any, used_joints: List[str]) -> None:
        try:
            metric_values[name] = float(fn())
            detail_map[name] = {
                "key": name,
                "available": True,
                "used_frames": used_frames,
                "used_joints": used_joints,
                "confidence": _estimate_metric_confidence(name, extraction, seg, True, ""),
                "warning": "",
            }
        except MetricComputationError as exc:
            errors[name] = exc.message
            detail_map[name] = {
                "key": name,
                "available": False,
                "used_frames": used_frames,
                "used_joints": used_joints,
                "confidence": _estimate_metric_confidence(name, extraction, seg, False, exc.message),
                "warning": exc.message,
            }

    _run("step_rate", lambda: compute_step_rate(frames, duration_sec, fps), ["left_ankle", "right_ankle"])
    _run(
        "trunk_lean_mean",
        lambda: compute_trunk_lean_mean(frames),
        ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    )
    _run(
        "arm_swing_variability",
        lambda: compute_arm_swing_variability(frames),
        ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"],
    )
    _run("left_right_timing_diff", lambda: compute_left_right_timing_diff(frames, fps), ["left_ankle", "right_ankle"])

    if "arm_swing_variability" in metric_values and "left_right_timing_diff" in metric_values:
        _run(
            "tech_stability_score",
            lambda: compute_tech_stability_score(
                frames,
                fps,
                metric_values["arm_swing_variability"],
                metric_values["left_right_timing_diff"],
            ),
            ["ankle", "shoulder", "hip", "elbow", "wrist"],
        )
    else:
        errors["tech_stability_score"] = "需要 arm_swing_variability 和 left_right_timing_diff。"
        detail_map["tech_stability_score"] = {
            "key": "tech_stability_score",
            "available": False,
            "used_frames": used_frames,
            "used_joints": ["ankle", "shoulder", "hip", "elbow", "wrist"],
            "confidence": _estimate_metric_confidence(
                "tech_stability_score", extraction, seg, False, errors["tech_stability_score"]
            ),
            "warning": errors["tech_stability_score"],
        }

    for k in ("step_rate", "trunk_lean_mean", "arm_swing_variability", "left_right_timing_diff", "tech_stability_score"):
        metric_values.setdefault(k, 0.0)
    available = [k for k, v in metric_values.items() if not (v == 0.0 and k in errors)]
    metric_details = [detail_map[k] for k in ("step_rate", "trunk_lean_mean", "arm_swing_variability", "left_right_timing_diff", "tech_stability_score")]
    return metric_values, available, errors, metric_details


def _estimate_metric_confidence(name: str, extraction: Any, seg: Any, ok: bool, warning: str) -> float:
    if not ok:
        return 0.15
    base = 0.85
    pose_ratio = float(getattr(extraction, "pose_ratio", 0.0))
    base *= max(0.35, min(1.0, pose_ratio + 0.2))
    seg_frames = int(seg.frame_count)
    if seg_frames < 12:
        base *= 0.65
    if "insufficient" in warning.lower():
        base *= 0.6
    if name == "tech_stability_score":
        base *= 0.9
    return round(max(0.1, min(0.99, base)), 3)


def _suggested_fix(message: str) -> str:
    msg = message.lower()
    if "duration" in msg:
        return "拍摄 3-10 秒稳定侧向跑步片段。"
    if "landscape" in msg or "aspect" in msg:
        return "建议横屏侧拍并保证全身入镜。"
    if "segment" in msg or "关键点" in msg:
        return "后退半步并保持全身连续入镜，减少遮挡。"
    return "请保证光线充足、机位稳定、侧向全身入镜。"


def _save_json(filename: str, data: Dict[str, Any]) -> None:
    os.makedirs(settings.upload_dir, exist_ok=True)
    path = os.path.join(settings.upload_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
