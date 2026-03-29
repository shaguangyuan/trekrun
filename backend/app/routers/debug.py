"""Debug endpoint: lightweight pose landmark extraction for frontend overlay."""

import os
import shutil
import tempfile
import uuid
from typing import Any, Dict

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.pose_extractor import extract_landmarks

router = APIRouter(tags=["debug"])

_ALLOWED_EXT = {".mp4", ".mov", ".avi", ".m4v"}


@router.post("/debug/extract-landmarks")
async def extract_debug_landmarks(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Accept a video, run MediaPipe Pose extraction, return compact per-frame
    landmarks for the front-end debug overlay canvas.

    Response shape::

        {
          "fps": 30.0,
          "total_frames": 900,
          "duration_ms": 30000.0,
          "width": 1920,
          "height": 1080,
          "frames_with_pose": 880,
          "frames": [
            { "t": 0,  "lm": [{"x":0.5,"y":0.3,"z":-0.1,"v":0.99}, ...] },
            { "t": 33, "lm": [...] },
            ...
          ]
        }
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    contents = await file.read()
    tmp_dir = tempfile.mkdtemp(prefix="pose_dbg_")
    video_id = f"dbg_{uuid.uuid4().hex[:8]}"
    tmp_path = os.path.join(tmp_dir, f"{video_id}{ext}")

    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        result = await run_in_threadpool(
            extract_landmarks, tmp_path, video_id, tmp_dir
        )

        if not result.success:
            raise HTTPException(
                status_code=422, detail=result.error or "Landmark extraction failed"
            )

        frames = []
        for fr in result.frames:
            if not fr.has_pose:
                continue
            frames.append(
                {
                    "t": fr.timestamp_ms,
                    "lm": [
                        {
                            "x": round(p.x, 5),
                            "y": round(p.y, 5),
                            "z": round(p.z, 5),
                            "v": round(p.visibility, 3),
                        }
                        for p in fr.pose_landmarks
                    ],
                }
            )

        return {
            "fps": result.fps,
            "total_frames": result.total_frames,
            "duration_ms": result.duration_ms,
            "width": result.width,
            "height": result.height,
            "frames_with_pose": result.frames_with_pose,
            "frames": frames,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
