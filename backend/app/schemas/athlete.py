from typing import List

from pydantic import BaseModel


class HistoryItem(BaseModel):
    video_id: str
    created_at: str
    tech_stability_score: float


class AthleteHistoryResponse(BaseModel):
    athlete_id: str
    history: List[HistoryItem]
