"""
Video quality checks + valid mid-clip segment selection.

Inputs: successful ExtractionResult from pose_extractor (all frames + landmarks).

Outputs:
- Video-level QC (duration, fps, landscape, single-person note)
- Per-frame keypoint QC (low-visibility ratio, drop ratio, full-body gate)
- Longest contiguous valid_segment for downstream metric code

Does NOT compute the 5 sprint metrics or training conclusions.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.services.pose_extractor import ExtractionResult, FrameResult, LandmarkPoint
from app.services.qc_thresholds import QCThresholds


@dataclass
class ValidSegment:
    """Contiguous frame range suitable for metric computation."""

    start_frame_idx: int
    end_frame_idx: int  # inclusive
    start_ms: int
    end_ms: int
    frame_count: int


@dataclass
class QualitySegmentOutcome:
    success: bool
    error: Optional[str]
    video_id: str
    thresholds_version: str  # fixed string for traceability
    video_meta: Dict[str, Any]
    keypoint_qc_summary: Dict[str, Any]
    valid_segment: Optional[ValidSegment]
    single_person_note: str
    quality_level: str = "usable_for_preview"
    warnings: List[str] = field(default_factory=list)
    segment_debug: Dict[str, Any] = field(default_factory=dict)


THRESHOLDS_VERSION = "qc_v1"


def build_valid_segment(
    extraction: ExtractionResult,
    thresholds: QCThresholds,
    output_dir: Optional[str] = None,
) -> QualitySegmentOutcome:
    """
    Run video + keypoint QC and pick the longest valid segment.

    If output_dir is set, writes {video_id}_valid_segment.json there.
    """
    if not extraction.success:
        return _outcome_fail(
            extraction.video_id,
            thresholds,
            "Extraction did not succeed; cannot run QC.",
        )

    video_id = extraction.video_id
    duration_sec = extraction.duration_ms / 1000.0
    aspect = extraction.width / max(extraction.height, 1)

    video_meta: Dict[str, Any] = {
        "duration_ms": extraction.duration_ms,
        "duration_sec": duration_sec,
        "fps": extraction.fps,
        "width": extraction.width,
        "height": extraction.height,
        "aspect_ratio_wh": aspect,
        "single_person_enforced_by_extractor": True,
        "duration_ok": thresholds.min_duration_sec <= duration_sec <= thresholds.max_duration_sec,
        "fps_ok": thresholds.min_fps <= extraction.fps <= thresholds.max_fps,
        "landscape_ok": aspect >= thresholds.min_landscape_aspect_ratio,
    }

    warnings: List[str] = []
    if not video_meta["duration_ok"]:
        warnings.append(
            f"Duration {duration_sec:.2f}s outside [{thresholds.min_duration_sec}, {thresholds.max_duration_sec}]s, "
            "but analysis continues."
        )
    if not video_meta["fps_ok"]:
        warnings.append(
            f"FPS {extraction.fps:.2f} outside [{thresholds.min_fps}, {thresholds.max_fps}], "
            "metric stability may degrade."
        )
    if not video_meta["landscape_ok"]:
        warnings.append(
            f"Aspect ratio {aspect:.3f} < {thresholds.min_landscape_aspect_ratio}; portrait accepted with lower confidence."
        )

    required_idx = QCThresholds.full_body_landmark_indices()
    frame_ok: List[bool] = []
    low_vis_ratios: List[float] = []
    missing_ratios: List[float] = []

    for fr in extraction.frames:
        ok, low_r, miss_r = _frame_passes(fr, thresholds, required_idx)
        frame_ok.append(ok)
        low_vis_ratios.append(low_r)
        missing_ratios.append(miss_r)

    n = len(frame_ok)
    passed_frames = sum(frame_ok)
    keypoint_qc_summary = {
        "frames_total": n,
        "frames_passing_keypoint_qc": passed_frames,
        "pass_ratio": passed_frames / n if n else 0.0,
        "mean_low_visibility_ratio": sum(low_vis_ratios) / n if n else 0.0,
        "mean_missing_ratio": sum(missing_ratios) / n if n else 0.0,
        "max_qc_gap_frames": thresholds.max_qc_gap_frames,
    }

    # 合并检测抖动造成的短间隙，再取长连续段（时间跨度含间隙内帧）
    frame_ok_merged = _bridge_short_gaps(frame_ok, thresholds.max_qc_gap_frames)
    runs_before = _all_true_runs(frame_ok)
    runs_after = _all_true_runs(frame_ok_merged)
    seg = _longest_contiguous_true(frame_ok_merged)
    if seg is None:
        total_f = keypoint_qc_summary["frames_total"]
        passed_f = keypoint_qc_summary["frames_passing_keypoint_qc"]
        if passed_f == 0:
            qc_err = (
                f"关键点质检：共 {total_f} 帧，0 帧同时满足「全身关键点可见且在画面安全边内」。"
                "竖屏时脚尖/头顶易贴边；请后退、人置中、全身入镜，或横屏侧向拍摄。"
            )
        else:
            qc_err = "No contiguous segment met keypoint QC and minimum length."
        outcome = QualitySegmentOutcome(
            success=False,
            error=qc_err,
            video_id=video_id,
            thresholds_version=THRESHOLDS_VERSION,
            video_meta=video_meta,
            keypoint_qc_summary=keypoint_qc_summary,
            valid_segment=None,
            quality_level="usable_for_preview",
            warnings=warnings,
            segment_debug={"runs_before_merge": runs_before, "runs_after_merge": runs_after},
            single_person_note=(
                "Single-person assumption: MediaPipe PoseLandmarker is configured with num_poses=1."
            ),
        )
        _maybe_save(outcome, output_dir)
        return outcome

    start_i, end_i = seg
    start_fr = extraction.frames[start_i]
    end_fr = extraction.frames[end_i]
    seg_duration_ms = end_fr.timestamp_ms - start_fr.timestamp_ms
    seg_frames = end_i - start_i + 1

    if seg_duration_ms < thresholds.min_segment_duration_ms:
        outcome = QualitySegmentOutcome(
            success=False,
            error=(
                f"合格动作连续段过短：最长约 {seg_duration_ms:.0f}ms，需要 ≥{thresholds.min_segment_duration_ms:.0f}ms。"
                "请录稍长、稳定侧身跑步片段，或人再远一点保证全身多帧都过检。"
            ),
            video_id=video_id,
            thresholds_version=THRESHOLDS_VERSION,
            video_meta=video_meta,
            keypoint_qc_summary=keypoint_qc_summary,
            valid_segment=None,
            quality_level="usable_for_preview",
            warnings=warnings,
            segment_debug={"runs_before_merge": runs_before, "runs_after_merge": runs_after},
            single_person_note=(
                "Single-person assumption: MediaPipe PoseLandmarker is configured with num_poses=1."
            ),
        )
        _maybe_save(outcome, output_dir)
        return outcome

    if seg_frames < thresholds.min_segment_frames:
        outcome = QualitySegmentOutcome(
            success=False,
            error=(
                f"Longest valid segment has {seg_frames} frames "
                f"< minimum {thresholds.min_segment_frames}."
            ),
            video_id=video_id,
            thresholds_version=THRESHOLDS_VERSION,
            video_meta=video_meta,
            keypoint_qc_summary=keypoint_qc_summary,
            valid_segment=None,
            quality_level="usable_for_preview",
            warnings=warnings,
            segment_debug={"runs_before_merge": runs_before, "runs_after_merge": runs_after},
            single_person_note=(
                "Single-person assumption: MediaPipe PoseLandmarker is configured with num_poses=1."
            ),
        )
        _maybe_save(outcome, output_dir)
        return outcome

    valid = ValidSegment(
        start_frame_idx=start_i,
        end_frame_idx=end_i,
        start_ms=start_fr.timestamp_ms,
        end_ms=end_fr.timestamp_ms,
        frame_count=seg_frames,
    )

    quality_level = "usable_for_all_metrics"
    pass_ratio = float(keypoint_qc_summary.get("pass_ratio", 0.0))
    if pass_ratio < 0.45:
        quality_level = "usable_for_metrics"
        warnings.append("Keypoint pass ratio is low; some metrics may be unstable.")

    outcome = QualitySegmentOutcome(
        success=True,
        error=None,
        video_id=video_id,
        thresholds_version=THRESHOLDS_VERSION,
        video_meta=video_meta,
        keypoint_qc_summary=keypoint_qc_summary,
        valid_segment=valid,
        quality_level=quality_level,
        warnings=warnings,
        segment_debug={"runs_before_merge": runs_before, "runs_after_merge": runs_after},
        single_person_note=(
            "Single-person assumption: MediaPipe PoseLandmarker is configured with num_poses=1."
        ),
    )
    _maybe_save(outcome, output_dir)
    return outcome


def _frame_passes(
    fr: FrameResult,
    t: QCThresholds,
    required_idx: tuple[int, ...],
) -> tuple[bool, float, float]:
    """Returns (passes, low_visibility_ratio, missing_ratio)."""
    if not fr.has_pose or not fr.pose_landmarks:
        return False, 1.0, 1.0

    lms: List[LandmarkPoint] = fr.pose_landmarks
    n = len(lms)
    if n == 0:
        return False, 1.0, 1.0

    low = sum(1 for lm in lms if lm.visibility < t.min_landmark_visibility)
    missing = sum(1 for lm in lms if lm.visibility < t.missing_visibility_cutoff)
    low_r = low / n
    miss_r = missing / n

    if low_r > t.max_low_visibility_ratio:
        return False, low_r, miss_r
    if miss_r > t.max_missing_ratio:
        return False, low_r, miss_r

    m = t.in_frame_margin
    for idx in required_idx:
        if idx >= n:
            return False, low_r, miss_r
        lm = lms[idx]
        if lm.visibility < t.full_body_min_visibility:
            return False, low_r, miss_r
        if not (m <= lm.x <= 1.0 - m and m <= lm.y <= 1.0 - m):
            return False, low_r, miss_r

    return True, low_r, miss_r


def _bridge_short_gaps(flags: List[bool], max_gap: int) -> List[bool]:
    """
    若两段 True 之间仅有不超过 max_gap 个 False，则把中间填成 True，
    便于在姿态抖动时仍得到足够长的连续时间跨度。
    """
    if max_gap <= 0 or not flags:
        return list(flags)
    s = list(flags)
    merged_any = True
    # 多轮直到无法再合并（处理链式多段）
    while merged_any:
        merged_any = False
        n = len(s)
        i = 0
        while i < n:
            while i < n and not s[i]:
                i += 1
            if i >= n:
                break
            j = i
            while j < n and s[j]:
                j += 1
            k = j
            while k < n and not s[k]:
                k += 1
            gap = k - j
            if k < n and s[k] and 0 < gap <= max_gap:
                for t in range(j, k):
                    s[t] = True
                merged_any = True
                break
            i = k if k > j else j + 1
    return s


def _longest_contiguous_true(flags: List[bool]) -> Optional[tuple[int, int]]:
    """Return (start_idx, end_idx inclusive) for longest True run; require at least one True."""
    best_len = 0
    best: Optional[tuple[int, int]] = None
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        length = j - i
        if length > best_len:
            best_len = length
            best = (i, j - 1)
        i = j
    return best


def _outcome_fail(
    video_id: str,
    thresholds: QCThresholds,
    error: str,
) -> QualitySegmentOutcome:
    return QualitySegmentOutcome(
        success=False,
        error=error,
        video_id=video_id,
        thresholds_version=THRESHOLDS_VERSION,
        video_meta={},
        keypoint_qc_summary={},
        valid_segment=None,
        quality_level="usable_for_preview",
        warnings=[],
        segment_debug={},
        single_person_note=(
            "Single-person assumption: MediaPipe PoseLandmarker is configured with num_poses=1."
        ),
    )


def _outcome_to_dict(o: QualitySegmentOutcome) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "success": o.success,
        "error": o.error,
        "video_id": o.video_id,
        "thresholds_version": o.thresholds_version,
        "video_meta": o.video_meta,
        "keypoint_qc_summary": o.keypoint_qc_summary,
        "quality_level": o.quality_level,
        "warnings": list(o.warnings or []),
        "segment_debug": o.segment_debug or {},
        "single_person_note": o.single_person_note,
        "valid_segment": None,
    }
    if o.valid_segment:
        d["valid_segment"] = asdict(o.valid_segment)
    return d


def _maybe_save(outcome: QualitySegmentOutcome, output_dir: Optional[str]) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{outcome.video_id}_valid_segment.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_outcome_to_dict(outcome), fh, ensure_ascii=False, indent=2)
    debug_path = os.path.join(output_dir, f"{outcome.video_id}_qc_debug.json")
    with open(debug_path, "w", encoding="utf-8") as fh:
        json.dump(_outcome_to_dict(outcome), fh, ensure_ascii=False, indent=2)


def _all_true_runs(flags: List[bool]) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        out.append({"start": i, "end": j - 1, "length": j - i})
        i = j
    return out
