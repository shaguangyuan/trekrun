"""
Pose landmark extraction using MediaPipe Pose Landmarker in VIDEO mode.

Responsibilities:
- Open a video file with OpenCV
- Process each frame sequentially with strictly-increasing timestamps
- Return per-frame pose_landmarks and pose_world_landmarks
- Save raw result to {output_dir}/{video_id}_landmarks.json
- Return a structured ExtractionResult; success=False on any failure

NOT responsible for:
- Metric calculation
- Quality gating / segment detection
- Multi-person analysis
- Medical or injury judgement
"""

from __future__ import annotations

import json
import logging
import os
import ctypes
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from app.services.qc_thresholds import PoseRuntimeSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model path — download with backend/scripts/download_model.py
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_THIS_DIR, "..", "..", "models")
DEFAULT_MODEL_PATH = os.path.join(_MODELS_DIR, "pose_landmarker_full.task")
LANDMARK_NAMES = [
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

CORE_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)

# Minimum fraction of frames that must have a detected pose.
# Extraction fails if fewer frames than this contain landmarks.
MIN_POSE_FRAME_RATIO = 0.1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class FrameResult:
    frame_idx: int
    timestamp_ms: int
    has_pose: bool
    pose_landmarks: List[LandmarkPoint] = field(default_factory=list)
    pose_world_landmarks: List[LandmarkPoint] = field(default_factory=list)


@dataclass
class ExtractionResult:
    video_id: str
    fps: float
    total_frames: int
    duration_ms: float
    width: int
    height: int
    frames: List[FrameResult] = field(default_factory=list)
    frames_with_pose: int = 0
    pose_ratio: float = 0.0
    core_joint_valid_ratio: float = 0.0
    pose_runtime: Dict[str, float | str | int] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_landmarks(
    video_path: str,
    video_id: str,
    output_dir: str,
    model_path: str = DEFAULT_MODEL_PATH,
) -> ExtractionResult:
    """
    Extract pose landmarks for every frame of *video_path*.

    Saves raw result to *output_dir*/{video_id}_landmarks.json.
    Returns ExtractionResult; inspect .success and .error on failure.

    Failure conditions:
    - video file not found
    - OpenCV cannot open the file
    - video has 0 frames or fps == 0
    - model file not found at model_path
    - MediaPipe raises an exception during processing
    - fewer than MIN_POSE_FRAME_RATIO of frames contain a detected pose
    """

    # --- 1. Validate inputs ---
    if not os.path.exists(video_path):
        return _fail(video_id, "Video file not found.")

    pose_cfg = PoseRuntimeSettings()
    resolved_model_path = model_path if model_path != DEFAULT_MODEL_PATH else _resolve_model_path(pose_cfg.model_variant)

    if not os.path.exists(resolved_model_path):
        return _fail(
            video_id,
            f"MediaPipe model file not found at '{resolved_model_path}'. "
            "Run backend/scripts/download_model.py to download it.",
        )

    # --- 2. Open video ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _fail(video_id, "OpenCV cannot open the video file. File may be corrupt or unsupported.")

    fps_raw = float(cap.get(cv2.CAP_PROP_FPS))
    # 部分手机/编码在 OpenCV 下会报 fps=0 或荒谬值，导致整段分析失败。
    fps: float = fps_raw if 0 < fps_raw <= 240 else 30.0
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # total_frames_meta 可能为 0 但逐帧可读，真正以读完的帧数为准。

    # --- 3. Build MediaPipe landmarker (VIDEO mode) ---
    delegate_requested = "GPU" if _env_flag("MEDIAPIPE_USE_GPU") else "CPU"
    delegate_attempts = _build_delegate_attempts(delegate_requested)
    frames: List[FrameResult] = []
    selected_delegate = "CPU"
    last_error = ""

    try:
        for delegate_name in delegate_attempts:
            try:
                frames = _extract_frames_with_delegate(
                    cap=cap,
                    fps=fps,
                    resolved_model_path=resolved_model_path,
                    pose_cfg=pose_cfg,
                    delegate_name=delegate_name,
                )
                selected_delegate = delegate_name
                if delegate_name == "CPU" and delegate_requested == "GPU":
                    logger.warning(
                        "MediaPipe GPU requested but fell back to CPU. video_id=%s reason=%s",
                        video_id,
                        last_error or "GPU attempt failed",
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "pose_extract failed on delegate=%s video_id=%s",
                    delegate_name,
                    video_id,
                )
                if delegate_name != "CPU":
                    # Retry on CPU when non-CPU path fails.
                    continue
                return _fail(
                    video_id,
                    f"MediaPipe processing error at pose_extract ({delegate_name}): {last_error}",
                )
    finally:
        cap.release()

    # --- 4. Validate pose coverage ---
    frames_with_pose = sum(1 for f in frames if f.has_pose)
    actual_frames = len(frames)

    if actual_frames == 0:
        return _fail(video_id, "No frames were read from the video.")

    duration_ms: float = (actual_frames / fps) * 1000.0 if fps > 0 else 0.0

    _interpolate_missing_frames(frames, max_gap=pose_cfg.max_interp_gap_frames)
    _smooth_core_joints(frames, alpha=pose_cfg.smoothing_alpha)
    pose_ratio = frames_with_pose / actual_frames
    core_joint_valid_ratio = _core_joint_valid_ratio(frames)
    if pose_ratio < MIN_POSE_FRAME_RATIO:
        return _fail(
            video_id,
            f"Pose detected in only {pose_ratio:.1%} of frames "
            f"(minimum required: {MIN_POSE_FRAME_RATIO:.0%}). "
            "Ensure the video is side-view, single person, full-body in frame.",
        )

    # --- 5. Build and save result ---
    result = ExtractionResult(
        video_id=video_id,
        fps=fps,
        total_frames=actual_frames,
        duration_ms=duration_ms,
        width=width,
        height=height,
        frames=frames,
        frames_with_pose=frames_with_pose,
        pose_ratio=pose_ratio,
        core_joint_valid_ratio=core_joint_valid_ratio,
        pose_runtime={
            "model_variant": pose_cfg.model_variant,
            "model_path": resolved_model_path,
            "delegate_requested": delegate_requested,
            "delegate_used": selected_delegate,
            "min_pose_detection_confidence": pose_cfg.min_pose_detection_confidence,
            "min_pose_presence_confidence": pose_cfg.min_pose_presence_confidence,
            "min_tracking_confidence": pose_cfg.min_tracking_confidence,
            "max_interp_gap_frames": pose_cfg.max_interp_gap_frames,
            "smoothing_alpha": pose_cfg.smoothing_alpha,
        },
        success=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}_landmarks.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, ensure_ascii=False)
    _save_pose_debug_json(output_dir, result)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(video_id: str, error: str) -> ExtractionResult:
    return ExtractionResult(
        video_id=video_id,
        fps=0.0,
        total_frames=0,
        duration_ms=0.0,
        width=0,
        height=0,
        success=False,
        error=error,
    )


def _resolve_model_path(variant: str) -> str:
    return os.path.join(_MODELS_DIR, f"pose_landmarker_{variant}.task")


def _env_flag(name: str) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _build_delegate_attempts(requested: str) -> List[str]:
    if requested != "GPU":
        return ["CPU"]

    gpu_ok, gpu_reason = _gpu_runtime_available()
    if not gpu_ok:
        logger.warning("GPU disabled by runtime self-check: %s", gpu_reason)
        return ["CPU"]
    return ["GPU", "CPU"]


def _gpu_runtime_available() -> tuple[bool, str]:
    try:
        ctypes.CDLL("libGLESv2.so.2")
    except OSError as exc:
        return False, f"libGLESv2 missing ({exc})"

    try:
        ctypes.CDLL("libEGL.so.1")
    except OSError as exc:
        return False, f"libEGL missing ({exc})"

    return True, "ok"


def _base_options_with_delegate(model_path: str, delegate_name: str) -> mp_python.BaseOptions:
    delegate = mp_python.BaseOptions.Delegate.CPU
    if delegate_name == "GPU":
        delegate = mp_python.BaseOptions.Delegate.GPU
    return mp_python.BaseOptions(model_asset_path=model_path, delegate=delegate)


def _extract_frames_with_delegate(
    *,
    cap: cv2.VideoCapture,
    fps: float,
    resolved_model_path: str,
    pose_cfg: PoseRuntimeSettings,
    delegate_name: str,
) -> List[FrameResult]:
    base_options = _base_options_with_delegate(resolved_model_path, delegate_name)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,  # single person only — do not change
        min_pose_detection_confidence=pose_cfg.min_pose_detection_confidence,
        min_pose_presence_confidence=pose_cfg.min_pose_presence_confidence,
        min_tracking_confidence=pose_cfg.min_tracking_confidence,
    )

    frames: List[FrameResult] = []
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * (1000.0 / fps))
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection.pose_landmarks:
                raw_lm = detection.pose_landmarks[0]
                raw_wlm = detection.pose_world_landmarks[0]
                frame_result = FrameResult(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    has_pose=True,
                    pose_landmarks=[LandmarkPoint(lm.x, lm.y, lm.z, lm.visibility) for lm in raw_lm],
                    pose_world_landmarks=[LandmarkPoint(lm.x, lm.y, lm.z, lm.visibility) for lm in raw_wlm],
                )
            else:
                frame_result = FrameResult(
                    frame_idx=frame_idx,
                    timestamp_ms=timestamp_ms,
                    has_pose=False,
                )

            frames.append(frame_result)
            frame_idx += 1

    return frames


def _save_pose_debug_json(output_dir: str, result: ExtractionResult) -> None:
    pose_by_frame = [1 if f.has_pose else 0 for f in result.frames]
    core_valid_per_frame = []
    for fr in result.frames:
        core_valid = 0
        for idx in CORE_JOINTS:
            if fr.has_pose and len(fr.pose_landmarks) > idx and fr.pose_landmarks[idx].visibility >= 0.2:
                core_valid += 1
        core_valid_per_frame.append(core_valid / float(len(CORE_JOINTS)))
    payload = {
        "video_id": result.video_id,
        "fps": result.fps,
        "total_frames": result.total_frames,
        "duration_ms": result.duration_ms,
        "width": result.width,
        "height": result.height,
        "frames_with_pose": result.frames_with_pose,
        "pose_ratio": result.pose_ratio,
        "core_joint_valid_ratio": result.core_joint_valid_ratio,
        "pose_runtime": result.pose_runtime,
        "pose_detected_per_frame": pose_by_frame,
        "core_joint_coverage_per_frame": core_valid_per_frame,
    }
    with open(os.path.join(output_dir, f"{result.video_id}_pose_debug.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    _save_landmarks_jsonl(output_dir, result)


def _interpolate_missing_frames(frames: List[FrameResult], max_gap: int) -> None:
    if max_gap <= 0 or not frames:
        return
    n = len(frames)
    i = 0
    while i < n:
        if frames[i].has_pose:
            i += 1
            continue
        start = i
        while i < n and not frames[i].has_pose:
            i += 1
        end = i - 1
        gap = end - start + 1
        left = start - 1
        right = i
        if gap > max_gap or left < 0 or right >= n:
            continue
        if (not frames[left].has_pose) or (not frames[right].has_pose):
            continue
        if len(frames[left].pose_landmarks) != len(frames[right].pose_landmarks):
            continue
        for k in range(gap):
            t = (k + 1) / float(gap + 1)
            src_l = frames[left].pose_landmarks
            src_r = frames[right].pose_landmarks
            src_wl = frames[left].pose_world_landmarks
            src_wr = frames[right].pose_world_landmarks
            interp_lm = []
            interp_wlm = []
            for j in range(len(src_l)):
                a, b = src_l[j], src_r[j]
                interp_lm.append(
                    LandmarkPoint(
                        x=(1.0 - t) * a.x + t * b.x,
                        y=(1.0 - t) * a.y + t * b.y,
                        z=(1.0 - t) * a.z + t * b.z,
                        visibility=(1.0 - t) * a.visibility + t * b.visibility,
                    )
                )
            for j in range(len(src_wl)):
                a, b = src_wl[j], src_wr[j]
                interp_wlm.append(
                    LandmarkPoint(
                        x=(1.0 - t) * a.x + t * b.x,
                        y=(1.0 - t) * a.y + t * b.y,
                        z=(1.0 - t) * a.z + t * b.z,
                        visibility=(1.0 - t) * a.visibility + t * b.visibility,
                    )
                )
            tgt = frames[start + k]
            tgt.has_pose = True
            tgt.pose_landmarks = interp_lm
            tgt.pose_world_landmarks = interp_wlm


def _smooth_core_joints(frames: List[FrameResult], alpha: float) -> None:
    if not frames:
        return
    for idx in CORE_JOINTS:
        prev = None
        for fr in frames:
            if not fr.has_pose or len(fr.pose_landmarks) <= idx:
                continue
            cur = fr.pose_landmarks[idx]
            if prev is None:
                prev = LandmarkPoint(cur.x, cur.y, cur.z, cur.visibility)
                continue
            v = max(cur.visibility, 0.05)
            a = alpha * (0.3 + 0.7 * v)
            if abs(cur.x - prev.x) > 0.25 or abs(cur.y - prev.y) > 0.25:
                a *= 0.15
            cur.x = a * cur.x + (1.0 - a) * prev.x
            cur.y = a * cur.y + (1.0 - a) * prev.y
            cur.z = a * cur.z + (1.0 - a) * prev.z
            cur.visibility = alpha * cur.visibility + (1.0 - alpha) * prev.visibility
            prev = LandmarkPoint(cur.x, cur.y, cur.z, cur.visibility)


def _core_joint_valid_ratio(frames: List[FrameResult]) -> float:
    if not frames:
        return 0.0
    valid = 0
    total = 0
    for fr in frames:
        for idx in CORE_JOINTS:
            total += 1
            if fr.has_pose and len(fr.pose_landmarks) > idx and fr.pose_landmarks[idx].visibility >= 0.2:
                valid += 1
    return valid / float(total) if total > 0 else 0.0


def _save_landmarks_jsonl(output_dir: str, result: ExtractionResult) -> None:
    path = os.path.join(output_dir, f"{result.video_id}_landmarks.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for fr in result.frames:
            line = {
                "frame_index": fr.frame_idx,
                "timestamp_ms": fr.timestamp_ms,
                "image_width": result.width,
                "image_height": result.height,
                "pose_detected": fr.has_pose,
                "pose_landmarks": [
                    {
                        "name": LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"LM_{i}",
                        "x": p.x,
                        "y": p.y,
                        "z": p.z,
                        "visibility": p.visibility,
                    }
                    for i, p in enumerate(fr.pose_landmarks)
                ],
                "pose_world_landmarks": [
                    {
                        "name": LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"WLM_{i}",
                        "x": p.x,
                        "y": p.y,
                        "z": p.z,
                        "visibility": p.visibility,
                    }
                    for i, p in enumerate(fr.pose_world_landmarks)
                ],
                "segmentation_mask": {
                    "enabled": False,
                    "available": False,
                    "summary": "PoseLandmarker segmentation output not enabled in current runtime.",
                },
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
