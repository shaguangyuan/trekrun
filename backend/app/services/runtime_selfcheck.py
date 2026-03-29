from __future__ import annotations

import ctypes
import logging
import os

logger = logging.getLogger(__name__)


def _can_load(lib_name: str) -> tuple[bool, str]:
    try:
        ctypes.CDLL(lib_name)
        return True, "ok"
    except OSError as exc:
        return False, str(exc)


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def run_startup_selfcheck() -> None:
    """Emit runtime diagnostics for cloud deployment troubleshooting."""
    gles_ok, gles_reason = _can_load("libGLESv2.so.2")
    egl_ok, egl_reason = _can_load("libEGL.so.1")
    gl_ok, gl_reason = _can_load("libGL.so.1")
    gpu_requested = _env_flag("MEDIAPIPE_USE_GPU")

    logger.info(
        "startup selfcheck libs: libGLESv2=%s (%s), libEGL=%s (%s), libGL=%s (%s)",
        gles_ok,
        gles_reason,
        egl_ok,
        egl_reason,
        gl_ok,
        gl_reason,
    )

    try:
        import cv2  # noqa: F401

        logger.info("startup selfcheck import: cv2=ok")
    except Exception as exc:  # noqa: BLE001
        logger.exception("startup selfcheck import: cv2 failed: %s", exc)

    try:
        import mediapipe as mp  # noqa: F401

        logger.info("startup selfcheck import: mediapipe=ok")
    except Exception as exc:  # noqa: BLE001
        logger.exception("startup selfcheck import: mediapipe failed: %s", exc)

    effective_delegate = "CPU"
    if gpu_requested and gles_ok and egl_ok:
        effective_delegate = "GPU"

    logger.info(
        "startup selfcheck delegate: requested=%s effective=%s",
        "GPU" if gpu_requested else "CPU",
        effective_delegate,
    )
