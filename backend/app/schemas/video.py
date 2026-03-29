from typing import Literal

from pydantic import BaseModel


class VideoUploadResponse(BaseModel):
    video_id: str
    task_id: str
    status: Literal["queued"] = "queued"
