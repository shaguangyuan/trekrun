"""
Local JSON persistence for analysis jobs, per-video metrics, and athlete history.

No database — suitable for MVP / single-machine dev. All writes are guarded by a lock.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings

_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(video_id: str) -> str:
    return os.path.join(settings.upload_dir, f"{video_id}.job.json")


def _metrics_path(video_id: str) -> str:
    return os.path.join(settings.upload_dir, f"{video_id}.metrics.json")


def _athlete_history_path(athlete_id: str) -> str:
    safe = athlete_id.replace(os.sep, "_").replace("/", "_")
    return os.path.join(settings.upload_dir, f"athlete_{safe}_history.json")


def _atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def write_initial_job(
    *,
    video_id: str,
    task_id: str,
    athlete_id: str,
    session_type: str,
    fatigue_state: str,
    event_group: str,
    file_path: str,
) -> None:
    payload = {
        "video_id": video_id,
        "task_id": task_id,
        "athlete_id": athlete_id,
        "session_type": session_type,
        "fatigue_state": fatigue_state,
        "event_group": event_group,
        "file_path": file_path,
        "status": "queued",
        "error": None,
        "created_at": _utc_now_iso(),
        "finished_at": None,
        "fps": None,
        "duration_ms": None,
        "saved_to_history": False,
        "saved_at": None,
    }
    with _lock:
        _atomic_write_json(_job_path(video_id), payload)


def read_job(video_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(video_id)
    if not os.path.exists(path):
        return None
    with _lock:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)


def update_job(video_id: str, **fields: Any) -> None:
    with _lock:
        path = _job_path(video_id)
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for k, v in fields.items():
            data[k] = v
        _atomic_write_json(path, data)


def fail_job(video_id: str, message: str) -> None:
    with _lock:
        path = _job_path(video_id)
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        data["status"] = "failed"
        data["error"] = message
        data["finished_at"] = _utc_now_iso()
        _atomic_write_json(path, data)


def write_metrics(video_id: str, metrics: Dict[str, Any]) -> None:
    with _lock:
        _atomic_write_json(_metrics_path(video_id), metrics)


def read_metrics(video_id: str) -> Optional[Dict[str, Any]]:
    path = _metrics_path(video_id)
    if not os.path.exists(path):
        return None
    with _lock:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)


def append_athlete_history(
    athlete_id: str,
    *,
    video_id: str,
    created_at: str,
    tech_stability_score: float,
    metrics: Dict[str, Any],
) -> None:
    """Prepend record (newest first)."""
    path = _athlete_history_path(athlete_id)
    with _lock:
        records: List[Dict[str, Any]] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
                records = list(blob.get("records", []))
        record = {
            "video_id": video_id,
            "created_at": created_at,
            "tech_stability_score": tech_stability_score,
            "metrics": metrics,
        }
        records.insert(0, record)
        _atomic_write_json(path, {"records": records})


def save_analysis_to_history(video_id: str) -> Dict[str, Any]:
    """
    Save completed analysis to athlete history once (manual action).
    Returns status dict for API responses.
    """
    job = read_job(video_id)
    if not job:
        return {"ok": False, "code": "not_found", "message": "Unknown video_id."}
    if str(job.get("status")) != "done":
        return {"ok": False, "code": "not_done", "message": "Analysis is not done yet."}
    if bool(job.get("saved_to_history")):
        return {"ok": True, "code": "already_saved", "message": "Already saved.", "saved_to_history": True}

    metrics = read_metrics(video_id)
    if not isinstance(metrics, dict):
        return {"ok": False, "code": "metrics_missing", "message": "Metrics not found."}

    athlete_id = str(job.get("athlete_id") or "")
    if not athlete_id:
        return {"ok": False, "code": "athlete_missing", "message": "Missing athlete_id."}
    finished = job.get("finished_at") or job.get("created_at") or ""
    created_at = str(finished)[:10] if finished else ""
    append_athlete_history(
        athlete_id,
        video_id=video_id,
        created_at=created_at,
        tech_stability_score=float(metrics.get("tech_stability_score", 0.0)),
        metrics=metrics,
    )
    now = _utc_now_iso()
    update_job(video_id, saved_to_history=True, saved_at=now)
    return {
        "ok": True,
        "code": "saved",
        "message": "Saved to history.",
        "saved_to_history": True,
        "saved_at": now,
        "athlete_id": athlete_id,
    }


def read_athlete_history_records(athlete_id: str) -> List[Dict[str, Any]]:
    path = _athlete_history_path(athlete_id)
    if not os.path.exists(path):
        return []
    with _lock:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        return list(blob.get("records", []))


def get_previous_metrics(athlete_id: str, video_id: str) -> Optional[Dict[str, Any]]:
    """Newest-first history: previous upload is the next record after *video_id*."""
    records = read_athlete_history_records(athlete_id)
    for i, r in enumerate(records):
        if r.get("video_id") == video_id:
            if i + 1 < len(records):
                prev = records[i + 1].get("metrics")
                if isinstance(prev, dict):
                    return prev
            return None
    return None
