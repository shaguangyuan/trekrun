import os
import uuid

import cv2
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.schemas.video import VideoUploadResponse
from app.services.analysis_runner import run_full_analysis
from app.services.job_store import write_initial_job
from app.services.qc_thresholds import UploadValidationSettings

router = APIRouter(tags=["videos"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}


@router.post("/videos/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    athlete_id: str = Form(...),
    session_type: str = Form(...),
    fatigue_state: str = Form(...),
    event_group: str = Form(...),
) -> VideoUploadResponse:
    upload_cfg = UploadValidationSettings()
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File size {size_mb:.1f} MB exceeds limit of {settings.max_upload_size_mb} MB.",
        )

    video_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    os.makedirs(settings.upload_dir, exist_ok=True)
    dest_path = os.path.join(settings.upload_dir, f"{video_id}{ext}")
    with open(dest_path, "wb") as f:
        f.write(contents)
    _validate_uploaded_video(dest_path, upload_cfg)

    write_initial_job(
        video_id=video_id,
        task_id=task_id,
        athlete_id=athlete_id,
        session_type=session_type,
        fatigue_state=fatigue_state,
        event_group=event_group,
        file_path=dest_path,
    )

    async def _run_analysis() -> None:
        await run_in_threadpool(
            run_full_analysis,
            video_id,
            task_id,
            dest_path,
            athlete_id,
            session_type,
            fatigue_state,
            event_group,
        )

    background_tasks.add_task(_run_analysis)

    return VideoUploadResponse(video_id=video_id, task_id=task_id, status="queued")


def _validate_uploaded_video(path: str, cfg: UploadValidationSettings) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail="视频文件不可读，请更换编码后重试。")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if min(width, height) < cfg.min_short_side_px:
        raise HTTPException(status_code=400, detail=f"视频分辨率过低，短边至少需 {cfg.min_short_side_px}px。")
    if fps <= 0 or frame_count <= 0:
        return
    duration_sec = frame_count / fps
    if duration_sec < cfg.min_duration_sec:
        raise HTTPException(status_code=400, detail=f"视频过短，至少 {cfg.min_duration_sec:.1f}s。")
    if duration_sec > cfg.max_duration_sec:
        raise HTTPException(status_code=400, detail=f"视频过长，最多 {cfg.max_duration_sec:.0f}s。")
