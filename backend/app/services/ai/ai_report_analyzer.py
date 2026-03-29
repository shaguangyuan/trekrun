"""
AI Report Analyzer: DeepSeek-powered secondary analysis for sprint video results.

Provides:
- Layered formal report (data quality first, then metrics, cross-analysis, limitations)
- Parsed structured report_json for optional frontend use
- Risk flagging without medical diagnosis

Output: Structured JSON saved to uploads/<video_id>_ai_analysis.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services.ai.ai_input_builder import build_ai_input, build_user_prompt
from app.services.ai.deepseek_client import DeepSeekError, get_client, is_configured

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v3.0"

# Schema hint for the model (soft contract; parsing still tolerates minor omissions).
AI_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "report_title": {"type": "string"},
        "report_text": {"type": "string", "description": "Full formal report in Chinese"},
        "report_json": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "major_technical_issues": {"type": "array"},
                "training_recommendations": {"type": "array"},
                "data_reliability_note": {"type": "string"},
                "confidence": {"type": "object"},
                "limitations": {"type": "array"},
            },
        },
        "ai_summary": {
            "type": "string",
            "description": "3-5 sentences executive summary in Chinese (analysis credibility, top findings, anomalies)",
        },
        "evidence_trace": {"type": "array", "items": {"type": "string"}},
        "key_findings": {"type": "array"},
        "metric_interpretations": {"type": "array"},
        "risk_flags": {"type": "array"},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
        "confidence_statement": {"type": "string"},
        "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "report_title",
        "report_text",
        "report_json",
        "ai_summary",
        "evidence_trace",
        "key_findings",
        "risk_flags",
        "limitations",
        "confidence_statement",
    ],
}

SYSTEM_PROMPT = """你是一名同时具备运动人体科学背景、短跑生物力学分析能力和短跑训练执教经验的专家。
你的核心任务是：把结构化指标转译为短跑技术问题、可能机制和训练建议，而不是解释算法本身。
你只能基于输入字段推理，不得编造未提供数值。

## 内容优先级（硬规则）
1) 第一优先级：短跑技术问题（主叙事）
2) 第二优先级：可能机制（力量、髋伸控制、踝刚度、单侧稳定、核心骨盆控制、节奏模式等）
3) 第三优先级：训练建议（要具体到练习）
4) 第四优先级：数据可靠性提示（仅辅助）

## 强制规则
- 若同时存在技术发现与数据质量发现：主要发现必须优先输出技术发现。
- “抖动/拍摄/指标矛盾”不得进入前三条主要发现，除非整体可解释性为 weak 或数据不可用。
- 建议模块前两条必须是训练建议，不得是拍摄建议。
- 拍摄/视频建议最多 1 条，单列为“数据可靠性提示”。
- 每条主要发现必须包含三层：动作现象 + 短跑含义 + 表现影响。
- 禁止只复述指标名或纯数值句式。
- 禁止工程报错语气，少用“算法/污染/异常值”等词。

## 指标转译要求
- left_right_timing_diff：转译为左右支撑-摆动节奏不均、单侧发力节律差，影响加速效率/直线推进。
- left_right_angle_differences：转译为左右动作幅度与链条不一致，关联单侧稳定/柔韧/力量控制。
- contact_time_proxy_sec：转译触地-离地转换效率；过长提示缓冲偏多、离地偏慢、刚度不足；过短需谨慎。
- cadence：解释步频与推进匹配，避免只说“快/慢”。
- ankle_rom：结合触地与节律解释踝刚度/缓冲控制，不做孤立结论。
- jitter_score：仅用于“数据可靠性提示”，不作为主要技术问题标题。

## 输出结构（必须）
- report_title: 字符串
- report_text: 字符串，分 3 节：
  1) 主要技术问题
  2) 训练建议
  3) 数据可靠性提示（仅必要时）
- report_json: 对象，至少包含：
  - summary: 字符串
  - major_technical_issues: 数组，每项包含 title, description, priority(high|medium|low), confidence(high|medium|low), related_metrics
  - training_recommendations: 数组，每项包含 name, purpose, target_issue, drills
  - data_reliability_note: 字符串（可空）
  - confidence: 对象（overall, interpretability_level, per_metric_notes）
  - limitations: 字符串数组
- ai_summary: 3-5 句
- evidence_trace: 字符串数组（引用输入字段和值）
- key_findings: 与 major_technical_issues 一致或兼容
- risk_flags: 仅保留必要的技术风险/数据可靠性提示
- suggestions: 训练建议字符串数组（优先训练，不要拍摄建议占前两条）
- confidence_statement: 字符串

若数据质量低到无法动作判断，允许先写“当前视频对动作判断的支持有限，因此以下结论仅供参考”，并下调技术结论强度。"""


def _pose_coverage_from_rfs(rfs: Dict[str, Any]) -> float:
    """Normalize pose coverage from raw_feature_summary (pose_ratio / computed ratio)."""
    explicit = rfs.get("pose_coverage")
    if explicit is not None and float(explicit) > 0:
        return float(explicit)
    total = int(rfs.get("total_frames") or 0)
    if total <= 0:
        return 0.0
    detected = float(rfs.get("pose_detected_frames") or 0)
    return detected / float(total)


def _normalize_ai_result(ai_result: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure new fields exist for backward compatibility and API consumers."""
    rj = ai_result.get("report_json")
    if not isinstance(rj, dict):
        rj = {
            "summary": str(ai_result.get("ai_summary") or ""),
            "major_technical_issues": list(ai_result.get("key_findings") or []),
            "training_recommendations": [],
            "data_reliability_note": "",
            "confidence": {"overall": "unknown", "interpretability_level": "weak"},
            "limitations": list(ai_result.get("limitations") or []),
        }
        ai_result["report_json"] = rj
    rj.setdefault("major_technical_issues", list(ai_result.get("key_findings") or []))
    rj.setdefault("training_recommendations", [])
    rj.setdefault("data_reliability_note", "")
    if not ai_result.get("report_title"):
        ai_result["report_title"] = "跑步动作分析报告"
    if not ai_result.get("report_text"):
        ai_result["report_text"] = str(ai_result.get("ai_summary") or "")
    return ai_result


def run_ai_analysis(
    video_id: str,
    feature_groups: Optional[Dict[str, Any]] = None,
    raw_feature_summary: Optional[Dict[str, Any]] = None,
    analysis_overview: Optional[Dict[str, Any]] = None,
    natural_language: Optional[Dict[str, Any]] = None,
    metric_confidence: Optional[Dict[str, float]] = None,
    used_frames: Optional[Dict[str, int]] = None,
    used_joints: Optional[Dict[str, List[str]]] = None,
    metrics_available: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    suggested_fix: Optional[str] = None,
    analysis_state: str = "done",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Run AI analysis on sprint video data using DeepSeek.

    Args:
        video_id: Video identifier
        feature_groups: Feature groups from feature_extractor
        raw_feature_summary: Raw feature summary
        analysis_overview: Analysis overview dict
        natural_language: Natural language explanation
        metric_confidence: Per-metric confidence scores
        used_frames: Per-metric used frame counts
        used_joints: Per-metric used joints
        metrics_available: List of successfully computed metrics
        warnings: Warning messages
        suggested_fix: Suggested fix message
        analysis_state: Overall analysis state
        force_refresh: If True, regenerate even if cache exists

    Returns:
        AI analysis result dict
    """
    cache_path = _ai_analysis_path(video_id)
    if not force_refresh and os.path.exists(cache_path):
        logger.info("AI analysis cache hit for video_id=%s", video_id)
        with open(cache_path, encoding="utf-8") as fh:
            cached = json.load(fh)
            return _normalize_ai_result(cached)

    if not is_configured():
        logger.warning("DeepSeek not configured, returning fallback analysis for video_id=%s", video_id)
        return _build_fallback_analysis(
            video_id=video_id,
            analysis_state=analysis_state,
            warnings=warnings,
            data_quality=_estimate_data_quality(raw_feature_summary, metric_confidence, metrics_available),
        )

    ai_input = build_ai_input(
        video_id=video_id,
        feature_groups=feature_groups,
        raw_feature_summary=raw_feature_summary,
        analysis_overview=analysis_overview,
        natural_language=natural_language,
        metric_confidence=metric_confidence,
        used_frames=used_frames,
        used_joints=used_joints,
        metrics_available=metrics_available,
        warnings=warnings,
        suggested_fix=suggested_fix,
        analysis_state=analysis_state,
    )

    user_prompt = build_user_prompt(ai_input)
    client = get_client()
    try:
        logger.info("Calling DeepSeek for video_id=%s", video_id)
        ai_result = client.generate_structured(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema_hint=AI_OUTPUT_SCHEMA,
            temperature=0.25,
        )
        ai_result = _normalize_ai_result(ai_result)

        ai_result["_metadata"] = {
            "video_id": video_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": client.model,
            "prompt_version": PROMPT_VERSION,
            "data_quality_grade": ai_input.get("layer_3_quality_control", {}).get("data_quality_grade"),
        }

        _save_ai_analysis(video_id, ai_result)
        logger.info("AI analysis completed and saved for video_id=%s", video_id)

        return ai_result

    except DeepSeekError as exc:
        logger.error("DeepSeek error for video_id=%s: %s", video_id, exc)
        return _build_fallback_analysis(
            video_id=video_id,
            analysis_state=analysis_state,
            warnings=warnings,
            data_quality=_estimate_data_quality(raw_feature_summary, metric_confidence, metrics_available),
            error_message=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected error during AI analysis for video_id=%s", video_id)
        return _build_fallback_analysis(
            video_id=video_id,
            analysis_state=analysis_state,
            warnings=warnings,
            data_quality=_estimate_data_quality(raw_feature_summary, metric_confidence, metrics_available),
            error_message=f"Analysis error: {exc}",
        )


def read_ai_analysis(video_id: str) -> Optional[Dict[str, Any]]:
    """Read cached AI analysis for video_id."""
    path = _ai_analysis_path(video_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return _normalize_ai_result(json.load(fh))
    except Exception:
        return None


def _ai_analysis_path(video_id: str) -> str:
    return os.path.join(settings.upload_dir, f"{video_id}_ai_analysis.json")


def _save_ai_analysis(video_id: str, data: Dict[str, Any]) -> None:
    os.makedirs(settings.upload_dir, exist_ok=True)
    path = _ai_analysis_path(video_id)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _estimate_data_quality(
    raw_feature_summary: Optional[Dict[str, Any]],
    metric_confidence: Optional[Dict[str, float]],
    metrics_available: Optional[List[str]],
) -> str:
    """Estimate overall data quality grade."""
    rfs = dict(raw_feature_summary or {})
    pose_coverage = _pose_coverage_from_rfs(rfs)
    core_valid = float(rfs.get("core_joint_valid_ratio") or 0)
    metrics_count = len(metrics_available or [])

    if pose_coverage >= 0.8 and core_valid >= 0.7 and metrics_count >= 5:
        return "high"
    elif pose_coverage >= 0.5 and core_valid >= 0.5 and metrics_count >= 3:
        return "medium"
    elif pose_coverage >= 0.3 and metrics_count >= 1:
        return "low"
    return "insufficient"


def _build_fallback_analysis(
    video_id: str,
    analysis_state: str,
    warnings: Optional[List[str]] = None,
    data_quality: str = "unknown",
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Build fallback analysis when DeepSeek is unavailable or fails."""
    is_partial = analysis_state == "partial"
    is_failed = analysis_state == "failed"
    warns = list(warnings or [])

    if is_failed:
        summary = (
            "本次视频分析未能完成，无法生成基于指标的可靠分层报告。"
            " 在流程未成功产出有效片段与数值前，不建议对动作强弱做解释。"
        )
        evidence = [
            "分析流程未完成：缺少与指标计算绑定的有效连续片段或必要字段。",
            "无法区分数据缺失、检测失败与真实动作表现。",
        ]
        findings: List[Dict[str, Any]] = []
        risk_flags = [{"type": "data_quality", "message": "当前视频对动作判断的支持有限，因此以下结论仅供参考。"}]
        strong_warn = "不建议强解释：分析未成功完成。"
    elif is_partial:
        summary = (
            "本次仅部分指标可用，分层报告仅能基于已提供字段做保守描述；"
            " 结论的不确定性高于完整分析，需谨慎引用。"
        )
        evidence = [
            "部分指标成功计算，其余缺失或置信度受限。",
            "解读应优先标注数据缺口，而不是推断未经验证的跑动模式。",
        ]
        findings = [
            {
                "title": "部分指标可用",
                "description": "可用子集可能无法代表全程技术稳定状态。",
                "related_metrics": [],
                "confidence": "low",
            }
        ]
        risk_flags = [{"type": "data_quality", "message": "部分指标可用性不足，技术结论仅限保守解释。"}]
        strong_warn = "不建议强解释：指标未完整覆盖五项核心输出。"
    else:
        summary = "本次已完成指标计算，以下为以短跑技术问题为主的占位分析（非大模型推理全文）。"
        evidence = [
            "基于 MediaPipe 派生特征与 QC 摘要的占位输出（非 LLM 生成）。",
            f"数据质量等级（启发式）：{data_quality}。",
        ]
        findings = [
            {
                "title": "技术趋势需结合下一次复测确认",
                "description": "当前为占位解读，建议将本次作为基线，重点观察左右节奏一致性与触地转换效率。",
                "related_metrics": [],
                "confidence": "medium",
            }
        ]
        risk_flags = []
        strong_warn = "" if data_quality in ("high", "medium") else "不建议强解释：数据质量等级偏低。"

    limitations = [
        "单目侧向视频不能替代实验室动作捕捉；若干时空量为算法 proxy。",
        "占位报告不包含模型驱动的交叉验证与矛盾检测全文。",
    ]
    if data_quality in ("low", "insufficient"):
        limitations.append("本次数据质量有限，任何动作层面的结论均应降级为假设。")

    report_title = "短跑技术分析报告（离线/占位）" if is_failed or error_message else "短跑技术分析报告"

    report_text_sections = "\n\n".join(
        [
            f"1. {report_title}",
            f"2. 分析摘要\n{summary}",
            "3. 主要技术问题\n当前为 fallback，占位输出不做强推断。优先关注：左右节奏一致性、触地到离地转换效率、踝关节支撑回弹表现。",
            (
                "4. 训练建议\n"
                "- 单侧稳定与节奏一致性：单腿RDL、分腿蹲、A-skip分侧强化。\n"
                "- 快速触地转换：ankling、小步快跑、10-30米加速跑。\n"
                "- 踝刚度与下肢弹性：连续小步弹跳、提踵变式。"
            ),
            f"5. 数据可靠性提示\n启发式质量等级：{data_quality}。{strong_warn or '当前可用于初步技术趋势观察。'}",
        ]
    )

    report_json: Dict[str, Any] = {
        "summary": summary,
        "major_technical_issues": findings,
        "training_recommendations": [
            {
                "name": "单侧支撑稳定与节奏一致性训练",
                "purpose": "改善左右支撑-摆动转换一致性",
                "target_issue": "左右节律不一致",
                "drills": ["单腿RDL", "分腿蹲", "A-skip分侧练习"],
            },
            {
                "name": "快速触地-离地转换训练",
                "purpose": "提升触地效率与推进效率",
                "target_issue": "触地阶段拖沓或离地不够干脆",
                "drills": ["ankling", "小步快跑", "10-30米加速跑"],
            },
        ],
        "data_reliability_note": "；".join(warns[:1]) if warns else "",
        "confidence": {
            "overall": "low" if is_failed or is_partial else "medium",
            "interpretability_level": "weak" if is_failed or is_partial else "moderate",
            "per_metric_notes": "fallback：无模型分项注释",
        },
        "limitations": limitations,
    }

    result: Dict[str, Any] = {
        "report_title": report_title,
        "report_text": report_text_sections,
        "report_json": report_json,
        "ai_summary": summary,
        "evidence_trace": evidence,
        "key_findings": findings,
        "metric_interpretations": [],
        "risk_flags": risk_flags,
        "limitations": limitations,
        "suggestions": [
            "针对左右节奏差异，优先增加单侧支撑稳定与单侧节奏控制练习（单腿RDL、分腿蹲、A-skip分侧强化）。",
            "针对触地-离地转换效率，加入ankling、小步快跑与10-30米加速跑，提升快速支撑与离地能力。",
            "针对踝部支撑弹性，加入连续小步弹跳与提踵变式，提升踝刚度和回弹效率。",
            "视频稳定性可作为辅助优化项，建议固定机位后复测以提高细节指标稳定性。",
        ],
        "confidence_statement": (
            f"本次为离线占位/错误回退输出，基于{data_quality}启发式质量估计；"
            " 不构成实验结论，仅用于流程与接口占位。"
        ),
        "recommended_next_steps": [
            "配置 DEEPSEEK_API_KEY 后刷新 AI 解读。",
            "若持续失败，检查上传视频与 QC 告警文本。",
        ],
        "_metadata": {
            "video_id": video_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": "fallback",
            "prompt_version": PROMPT_VERSION,
            "data_quality_grade": data_quality,
            "is_fallback": True,
            "error": error_message,
        },
    }

    _save_ai_analysis(video_id, result)
    return result
