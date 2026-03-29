"""
Template-based natural language explanation for sprint analysis.

No LLM usage. Explanations are evidence-grounded and concise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


METRIC_RULES: Dict[str, Dict[str, Any]] = {
    "step_rate": {
        "label": "步频",
        "used_joints": ["left_ankle", "right_ankle"],
        "rule": "依据左右踝点轨迹中的落地节律事件，按有效片段时长计算步/秒。",
    },
    "trunk_lean_mean": {
        "label": "躯干前倾均值",
        "used_joints": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "rule": "依据肩髋连线相对垂线角度，统计有效片段内均值。",
    },
    "arm_swing_variability": {
        "label": "摆臂波动",
        "used_joints": ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"],
        "rule": "依据肘关节角度振幅和周期稳定性，输出归一化波动分数。",
    },
    "left_right_timing_diff": {
        "label": "左右节律差",
        "used_joints": ["left_ankle", "right_ankle"],
        "rule": "比较左右步间隔均值差异，输出百分比。",
    },
    "tech_stability_score": {
        "label": "技术稳定性综合分",
        "used_joints": ["ankle", "shoulder", "hip", "elbow", "wrist"],
        "rule": "综合步频波动、躯干稳定、摆臂波动和左右节律差得到 0-100 分。",
    },
}


def build_natural_language_explanation(
    *,
    video_id: str,
    analysis_state: str,
    analysis_overview: Optional[Mapping[str, Any]],
    metric_details: List[Dict[str, Any]],
    warnings: Optional[List[str]] = None,
    suggested_fix: Optional[str] = None,
) -> Dict[str, Any]:
    warnings = list(warnings or [])
    ov = dict(analysis_overview or {})
    pose = dict(ov.get("pose_summary") or {})
    qc = dict(ov.get("qc_summary") or {})
    seg = dict(ov.get("selected_segment") or {})

    total_frames = int(pose.get("total_frames") or 0)
    pose_frames = int(pose.get("frames_with_pose") or 0)
    pose_ratio = float(pose.get("pose_ratio") or 0.0)
    fps = float(ov.get("fps") or 0.0)
    duration_ms = float(ov.get("duration_ms") or 0.0)

    state_text = {
        "done": "本次视频已完成指标计算。",
        "partial": "本次视频完成了部分指标计算。",
        "failed": "本次视频未能产出可用指标。",
    }.get(analysis_state, "分析已结束。")

    summary = (
        f"{state_text} 视频时长约 {duration_ms/1000.0:.1f}s，FPS 约 {fps:.1f}，"
        f"共 {total_frames} 帧，其中检测到人体骨架 {pose_frames} 帧（{pose_ratio:.1%}）。"
    )

    process_steps = [
        "步骤1：读取视频并逐帧进行 MediaPipe Pose VIDEO 模式检测。",
        "步骤2：对短缺口帧做插值并对核心关节做平滑，降低抖动影响。",
        "步骤3：根据关键点覆盖率与时序连续性筛选候选片段，并合并短暂断裂。",
        "步骤4：在最终片段上逐项计算指标；单项失败不会直接导致整单失败。",
    ]

    selected_segment_explanation = (
        f"最终选择片段：{int(seg.get('start_ms') or 0)}ms 到 {int(seg.get('end_ms') or 0)}ms，"
        f"时长约 {float(seg.get('duration_ms') or 0):.0f}ms。"
    )
    reason = seg.get("reason")
    if reason:
        selected_segment_explanation += f" 选中原因：{reason}"
    elif qc.get("quality_level"):
        selected_segment_explanation += f" 片段质量级别：{qc.get('quality_level')}。"

    metric_explanations: List[Dict[str, Any]] = []
    for m in metric_details:
        key = str(m.get("key") or "")
        meta = METRIC_RULES.get(key, {})
        label = meta.get("label", key)
        rule = meta.get("rule", "依据有效片段内关键点时序规则计算。")
        conf = float(m.get("confidence") or 0.0)
        used_frames = int(m.get("used_frames") or 0)
        used_joints = list(m.get("used_joints") or meta.get("used_joints") or [])
        available = bool(m.get("available"))
        note = str(m.get("warning") or "")
        tone = "可信度较高" if conf >= 0.75 else ("可信度中等" if conf >= 0.45 else "可信度偏低")
        text = f"{label}：{rule} 本次使用约 {used_frames} 帧，{tone}（{conf:.2f}）。"
        if not available:
            text = f"{label}：本次未成功计算。{note or '关键点条件不足。'}"
        elif note:
            text += f" 注意：{note}"
        metric_explanations.append(
            {
                "key": key,
                "label": label,
                "explanation": text,
                "confidence": conf,
                "used_frames": used_frames,
                "used_joints": used_joints,
                "available": available,
            }
        )

    return {
        "summary": summary,
        "process_steps": process_steps,
        "selected_segment_explanation": selected_segment_explanation,
        "metric_explanations": metric_explanations,
        "warnings": warnings,
        "suggested_fix": suggested_fix,
        "video_id": video_id,
    }
