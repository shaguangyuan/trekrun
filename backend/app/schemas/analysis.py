from typing import Literal, Optional

from pydantic import BaseModel


class AnalysisStatusResponse(BaseModel):
    video_id: str
    task_id: str
    status: Literal["queued", "processing", "done", "failed"]
    # 失败原因：部分环境下 JSON 字段名 `error` 会被小程序侧丢弃，故同时提供 `failure_reason`（内容一致）。
    error: Optional[str] = None
    failure_reason: Optional[str] = None
