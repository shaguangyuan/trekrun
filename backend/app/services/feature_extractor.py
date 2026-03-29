"""
Derived feature extraction from MediaPipe landmarks.

This layer explicitly separates:
- direct_from_mediapipe: fields directly emitted by MediaPipe
- derived_from_landmarks: deterministic calculations from landmarks
- inferred_proxy: heuristic proxy metrics (not ground-truth biomechanics)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.services.pose_extractor import ExtractionResult, FrameResult, LandmarkPoint
from app.services.video_quality_segment import QualitySegmentOutcome, ValidSegment

LANDMARK_NAMES: List[str] = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

IDX = {k: i for i, k in enumerate(LANDMARK_NAMES)}
CORE_JOINTS = [
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
]


def _lm(fr: FrameResult, name: str) -> Optional[LandmarkPoint]:
    idx = IDX[name]
    if not fr.has_pose or len(fr.pose_landmarks) <= idx:
        return None
    return fr.pose_landmarks[idx]


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return float((sum((v - m) * (v - m) for v in vals) / (len(vals) - 1)) ** 0.5)


def _angle(a: LandmarkPoint, b: LandmarkPoint, c: LandmarkPoint) -> Optional[float]:
    v1x, v1y = a.x - b.x, a.y - b.y
    v2x, v2y = c.x - b.x, c.y - b.y
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cosv = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
    return math.degrees(math.acos(cosv))


def _line_angle(p1: LandmarkPoint, p2: LandmarkPoint) -> float:
    return math.degrees(math.atan2((p2.y - p1.y), (p2.x - p1.x)))


def _collect_joint_visibility(extraction: ExtractionResult) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in LANDMARK_NAMES:
        vals: List[float] = []
        for fr in extraction.frames:
            p = _lm(fr, name)
            if p is not None:
                vals.append(float(p.visibility))
        out[name] = _mean(vals) if vals else 0.0
    return out


def _collect_joint_valid_ratio(extraction: ExtractionResult, threshold: float = 0.3) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = len(extraction.frames)
    for name in LANDMARK_NAMES:
        cnt = 0
        for fr in extraction.frames:
            p = _lm(fr, name)
            if p is not None and p.visibility >= threshold:
                cnt += 1
        out[name] = (cnt / total) if total > 0 else 0.0
    return out


def _segment_frames(extraction: ExtractionResult, seg: Optional[ValidSegment]) -> List[FrameResult]:
    if seg is None:
        return extraction.frames
    return extraction.frames[seg.start_frame_idx : seg.end_frame_idx + 1]


def extract_feature_groups(
    *,
    extraction: ExtractionResult,
    qc: QualitySegmentOutcome,
    metrics: Dict[str, float],
    metric_details: List[Dict[str, Any]],
    warnings: List[str],
) -> Dict[str, Any]:
    frames = _segment_frames(extraction, qc.valid_segment)
    dt = 1.0 / extraction.fps if extraction.fps > 0 else 0.033

    trunk_lean_vals: List[float] = []
    trunk_side_tilt_vals: List[float] = []
    shoulder_line_vals: List[float] = []
    hip_line_vals: List[float] = []
    l_elbow, r_elbow, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle = ([] for _ in range(8))
    left_foot_prog, right_foot_prog = [], []

    for fr in frames:
        ls, rs = _lm(fr, "LEFT_SHOULDER"), _lm(fr, "RIGHT_SHOULDER")
        lh, rh = _lm(fr, "LEFT_HIP"), _lm(fr, "RIGHT_HIP")
        lk, rk = _lm(fr, "LEFT_KNEE"), _lm(fr, "RIGHT_KNEE")
        la, ra = _lm(fr, "LEFT_ANKLE"), _lm(fr, "RIGHT_ANKLE")
        lheel, rheel = _lm(fr, "LEFT_HEEL"), _lm(fr, "RIGHT_HEEL")
        lfoot, rfoot = _lm(fr, "LEFT_FOOT_INDEX"), _lm(fr, "RIGHT_FOOT_INDEX")
        le, re = _lm(fr, "LEFT_ELBOW"), _lm(fr, "RIGHT_ELBOW")
        lw, rw = _lm(fr, "LEFT_WRIST"), _lm(fr, "RIGHT_WRIST")

        if ls and rs and lh and rh:
            shoulder_mid = LandmarkPoint((ls.x + rs.x) * 0.5, (ls.y + rs.y) * 0.5, 0.0, 1.0)
            hip_mid = LandmarkPoint((lh.x + rh.x) * 0.5, (lh.y + rh.y) * 0.5, 0.0, 1.0)
            dx = shoulder_mid.x - hip_mid.x
            dy = shoulder_mid.y - hip_mid.y
            trunk_lean_vals.append(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-8)))
            trunk_side_tilt_vals.append(_line_angle(ls, rs))
            shoulder_line_vals.append(_line_angle(ls, rs))
            hip_line_vals.append(_line_angle(lh, rh))

        if ls and le and lw:
            a = _angle(ls, le, lw)
            if a is not None:
                l_elbow.append(a)
        if rs and re and rw:
            a = _angle(rs, re, rw)
            if a is not None:
                r_elbow.append(a)
        if ls and lh and lk:
            a = _angle(ls, lh, lk)
            if a is not None:
                l_hip.append(a)
        if rs and rh and rk:
            a = _angle(rs, rh, rk)
            if a is not None:
                r_hip.append(a)
        if lh and lk and la:
            a = _angle(lh, lk, la)
            if a is not None:
                l_knee.append(a)
        if rh and rk and ra:
            a = _angle(rh, rk, ra)
            if a is not None:
                r_knee.append(a)
        if lk and la and lfoot:
            a = _angle(lk, la, lfoot)
            if a is not None:
                l_ankle.append(a)
        if rk and ra and rfoot:
            a = _angle(rk, ra, rfoot)
            if a is not None:
                r_ankle.append(a)
        if lheel and lfoot:
            left_foot_prog.append(_line_angle(lheel, lfoot))
        if rheel and rfoot:
            right_foot_prog.append(_line_angle(rheel, rfoot))

    def _velocity(vals: List[float]) -> List[float]:
        return [(vals[i] - vals[i - 1]) / dt for i in range(1, len(vals))]

    def _acc(vals: List[float]) -> List[float]:
        v = _velocity(vals)
        return [(v[i] - v[i - 1]) / dt for i in range(1, len(v))]

    def _rom(vals: List[float]) -> float:
        return (max(vals) - min(vals)) if vals else 0.0

    joint_visibility_mean = _collect_joint_visibility(extraction)
    joint_valid_ratio = _collect_joint_valid_ratio(extraction, threshold=0.3)
    mean_core_valid = _mean([joint_valid_ratio.get(j, 0.0) for j in CORE_JOINTS])

    pose_detected_by_frame = [1.0 if fr.has_pose else 0.0 for fr in extraction.frames]
    jitter_score = _std(_velocity(trunk_lean_vals)) if len(trunk_lean_vals) >= 3 else 0.0
    candidate_segments = (qc.segment_debug or {}).get("runs_after_merge", [])
    seg_duration = 0.0
    if qc.valid_segment is not None:
        seg_duration = float(qc.valid_segment.end_ms - qc.valid_segment.start_ms)

    metric_conf = {d.get("key"): float(d.get("confidence", 0.0)) for d in metric_details}
    metric_used_frames = {d.get("key"): int(d.get("used_frames", 0)) for d in metric_details}
    metric_used_joints = {d.get("key"): list(d.get("used_joints", [])) for d in metric_details}

    geometry = {
        "trunk_lean_angle": _mean(trunk_lean_vals),
        "trunk_side_tilt": _mean(trunk_side_tilt_vals),
        "shoulder_line_angle": _mean(shoulder_line_vals),
        "hip_line_angle": _mean(hip_line_vals),
        "left_shoulder_angle": _mean(shoulder_line_vals),
        "right_shoulder_angle": _mean(shoulder_line_vals),
        "left_elbow_angle": _mean(l_elbow),
        "right_elbow_angle": _mean(r_elbow),
        "left_hip_angle": _mean(l_hip),
        "right_hip_angle": _mean(r_hip),
        "left_knee_angle": _mean(l_knee),
        "right_knee_angle": _mean(r_knee),
        "left_ankle_angle": _mean(l_ankle),
        "right_ankle_angle": _mean(r_ankle),
        "left_foot_progression_proxy": _mean(left_foot_prog),
        "right_foot_progression_proxy": _mean(right_foot_prog),
    }

    temporal = {
        "joint_angular_velocity": {
            "left_knee": _mean(_velocity(l_knee)),
            "right_knee": _mean(_velocity(r_knee)),
            "left_elbow": _mean(_velocity(l_elbow)),
            "right_elbow": _mean(_velocity(r_elbow)),
        },
        "joint_angular_acceleration": {
            "left_knee": _mean(_acc(l_knee)),
            "right_knee": _mean(_acc(r_knee)),
            "left_elbow": _mean(_acc(l_elbow)),
            "right_elbow": _mean(_acc(r_elbow)),
        },
        "rom": {
            "shoulder": _rom(shoulder_line_vals),
            "elbow": _mean([_rom(l_elbow), _rom(r_elbow)]),
            "hip": _mean([_rom(l_hip), _rom(r_hip)]),
            "knee": _mean([_rom(l_knee), _rom(r_knee)]),
            "ankle": _mean([_rom(l_ankle), _rom(r_ankle)]),
        },
        "peak_flexion": {
            "knee": min(l_knee + r_knee) if (l_knee + r_knee) else 0.0,
            "elbow": min(l_elbow + r_elbow) if (l_elbow + r_elbow) else 0.0,
        },
        "peak_extension": {
            "knee": max(l_knee + r_knee) if (l_knee + r_knee) else 0.0,
            "elbow": max(l_elbow + r_elbow) if (l_elbow + r_elbow) else 0.0,
        },
        "cadence": float(metrics.get("step_rate", 0.0) * 60.0),
        "gait_cycle_duration": (60.0 / (float(metrics.get("step_rate", 0.0) * 2.0))) if float(metrics.get("step_rate", 0.0)) > 0 else 0.0,
        "stance_swing_proxy": {
            "stance_ratio": 0.6 if float(metrics.get("step_rate", 0.0)) > 0 else 0.0,
            "swing_ratio": 0.4 if float(metrics.get("step_rate", 0.0)) > 0 else 0.0,
        },
        "stride_to_stride_variability": _std(_velocity(trunk_lean_vals)),
    }

    symmetry = {
        "left_right_angle_difference": {
            "elbow": abs(_mean(l_elbow) - _mean(r_elbow)),
            "hip": abs(_mean(l_hip) - _mean(r_hip)),
            "knee": abs(_mean(l_knee) - _mean(r_knee)),
            "ankle": abs(_mean(l_ankle) - _mean(r_ankle)),
        },
        "left_right_timing_difference": float(metrics.get("left_right_timing_diff", 0.0)),
        "left_right_peak_difference": {
            "knee_peak_extension_diff": abs((max(l_knee) if l_knee else 0.0) - (max(r_knee) if r_knee else 0.0)),
        },
        "waveform_similarity_correlation_proxy": max(0.0, 1.0 - min(1.0, abs(float(metrics.get("left_right_timing_diff", 0.0)) / 100.0))),
    }

    qc_group = {
        "total_frames": extraction.total_frames,
        "valid_pose_frames": extraction.frames_with_pose,
        "pose_coverage": extraction.pose_ratio,
        "per_joint_visibility_mean": joint_visibility_mean,
        "per_joint_valid_ratio": joint_valid_ratio,
        "core_joint_valid_ratio": mean_core_valid,
        "jitter_score": jitter_score,
        "candidate_segments_count": len(candidate_segments),
        "candidate_segments": candidate_segments,
        "selected_segment_duration": seg_duration,
        "interpolated_frame_ratio": 1.0 - extraction.pose_ratio,
        "per_metric_confidence": metric_conf,
        "per_metric_used_frames": metric_used_frames,
        "per_metric_used_joints": metric_used_joints,
    }

    source_boundary = {
        "direct_from_mediapipe": [
            "per-frame landmarks",
            "pose detection flags",
            "visibility scores",
        ],
        "derived_from_landmarks": [
            "joint angles",
            "rom",
            "line angles",
            "pose coverage",
            "left-right differences",
        ],
        "inferred_proxy": [
            "stance/swing proxy",
            "gait cycle duration proxy",
            "waveform similarity proxy",
            "fatigue is not directly inferred",
        ],
        "not_direct_from_mediapipe": [
            "ground reaction force",
            "joint torque",
            "muscle activation",
            "exact stride length ground truth",
            "physiological fatigue state",
        ],
    }

    return {
        "schema_version": "feature_v1",
        "video_id": extraction.video_id,
        "feature_groups": {
            "pose_geometry": geometry,
            "temporal": temporal,
            "symmetry": symmetry,
            "qc": qc_group,
        },
        "raw_feature_summary": {
            "duration_ms": extraction.duration_ms,
            "fps": extraction.fps,
            "total_frames": extraction.total_frames,
            "pose_detected_frames": extraction.frames_with_pose,
            "core_joint_valid_ratio": extraction.core_joint_valid_ratio,
            "candidate_segments_count": len(candidate_segments),
            "selected_segment_duration_ms": seg_duration,
        },
        "metric_confidence": metric_conf,
        "used_frames": metric_used_frames,
        "used_joints": metric_used_joints,
        "warnings": list(warnings),
        "analysis_overview": {
            "pose_summary": {
                "total_frames": extraction.total_frames,
                "frames_with_pose": extraction.frames_with_pose,
                "pose_coverage": extraction.pose_ratio,
                "core_joint_valid_ratio": extraction.core_joint_valid_ratio,
            },
            "qc_summary": {
                "quality_level": qc.quality_level,
                "warnings": qc.warnings,
                "candidate_segments": candidate_segments,
                "selected_segment_duration_ms": seg_duration,
            },
        },
        "source_boundary": source_boundary,
    }
