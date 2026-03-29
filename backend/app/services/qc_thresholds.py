"""
Configurable thresholds for video QC and valid-segment detection.

Override any field via environment variable with prefix QC_ (uppercase snake).
Example: QC_MIN_DURATION_SEC=2.5

Loaded from .env next to the working directory when running uvicorn from backend/.
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QCThresholds(BaseSettings):
    """All QC / segmentation thresholds in one place (env-overridable)."""

    model_config = SettingsConfigDict(
        env_prefix="QC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 默认放宽：仅 3–5s 会把常见手机录像全部判失败；分析会在整段里截取合格连续片段。
    min_duration_sec: float = Field(default=2.0, ge=0.1)
    max_duration_sec: float = Field(default=120.0, ge=0.1)
    min_fps: float = Field(default=15.0, ge=1.0)
    max_fps: float = Field(default=120.0, ge=1.0)
    # 默认允许常见竖屏录像（720×960 → 0.75）；侧向跑姿仍以横屏侧拍为理想，可在 .env 提高到 1.05
    min_landscape_aspect_ratio: float = Field(
        default=0.65,
        ge=0.3,
        le=3.0,
        description="width/height gate; <1 accepts portrait phone video.",
    )

    # 竖屏/手机远景时脚易贴边、部分点 visibility 偏低，略放宽以减少「0 帧通过」
    min_landmark_visibility: float = Field(default=0.45, ge=0.0, le=1.0)
    max_low_visibility_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    missing_visibility_cutoff: float = Field(default=0.1, ge=0.0, le=1.0)
    max_missing_ratio: float = Field(default=0.28, ge=0.0, le=1.0)
    in_frame_margin: float = Field(default=0.008, ge=0.0, le=0.25)

    full_body_min_visibility: float = Field(default=0.32, ge=0.0, le=1.0)

    # 最短连续合格段；姿态检测抖动会产生大量短 True 段，配合 max_qc_gap_frames 合并
    min_segment_duration_ms: float = Field(default=600.0, ge=100.0)
    min_segment_frames: int = Field(default=8, ge=1)
    # 两段合格帧之间的 False 帧数 ≤ 此值时视为同一段（闭合短间隙，约 30fps 下 10 帧≈330ms）
    max_qc_gap_frames: int = Field(default=10, ge=0, le=60)

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "QCThresholds":
        if self.min_duration_sec > self.max_duration_sec:
            raise ValueError("min_duration_sec must be <= max_duration_sec")
        if self.min_fps > self.max_fps:
            raise ValueError("min_fps must be <= max_fps")
        return self

    @staticmethod
    def full_body_landmark_indices() -> tuple[int, ...]:
        """Indices used for 'full body in frame' gate (33-landmark pose)."""
        return (0, 11, 12, 15, 16, 23, 24, 27, 28)


class PoseRuntimeSettings(BaseSettings):
    """MediaPipe runtime settings (env-overridable)."""

    model_config = SettingsConfigDict(
        env_prefix="POSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    model_variant: str = Field(default="full", pattern="^(lite|full|heavy)$")
    min_pose_detection_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    min_pose_presence_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    max_interp_gap_frames: int = Field(default=5, ge=0, le=30)
    smoothing_alpha: float = Field(default=0.50, ge=0.0, le=1.0)


class UploadValidationSettings(BaseSettings):
    """Minimal upload-stage checks only (do not gate by posture quality)."""

    model_config = SettingsConfigDict(
        env_prefix="UPLOAD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    min_short_side_px: int = Field(default=240, ge=64, le=4096)
    min_duration_sec: float = Field(default=0.4, ge=0.1, le=30.0)
    max_duration_sec: float = Field(default=300.0, ge=5.0, le=3600.0)
