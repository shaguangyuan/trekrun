from fastapi import APIRouter

from app.schemas.athlete import AthleteHistoryResponse, HistoryItem
from app.services.job_store import read_athlete_history_records

router = APIRouter(tags=["athletes"])


def _history_payload(athlete_id: str) -> AthleteHistoryResponse:
    records = read_athlete_history_records(athlete_id)
    items = [
        HistoryItem(
            video_id=str(r.get("video_id", "")),
            created_at=str(r.get("created_at", "")),
            tech_stability_score=float(r.get("tech_stability_score", 0.0)),
        )
        for r in records
    ]
    return AthleteHistoryResponse(athlete_id=athlete_id, history=items)


@router.get("/athletes/{athlete_id}/history", response_model=AthleteHistoryResponse)
def get_athlete_history(athlete_id: str) -> AthleteHistoryResponse:
    return _history_payload(athlete_id)


@router.get("/athletes/{athlete_id}/trend", response_model=AthleteHistoryResponse)
def get_athlete_trend(athlete_id: str) -> AthleteHistoryResponse:
    """Same payload as /history — used by the mini-program trend page."""
    return _history_payload(athlete_id)
