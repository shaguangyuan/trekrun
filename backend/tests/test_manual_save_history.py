from app.services import job_store


def test_manual_save_only_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(job_store.settings, "upload_dir", str(tmp_path))
    vid = "v-save-1"
    aid = "A100"
    job_store.write_initial_job(
        video_id=vid,
        task_id="t1",
        athlete_id=aid,
        session_type="normal",
        fatigue_state="no",
        event_group="100",
        file_path="x.mp4",
    )
    job_store.update_job(vid, status="done", finished_at="2026-03-23T00:00:00+00:00")
    job_store.write_metrics(
        vid,
        {
            "step_rate": 4.0,
            "trunk_lean_mean": 8.0,
            "arm_swing_variability": 0.3,
            "left_right_timing_diff": 5.0,
            "tech_stability_score": 70.0,
        },
    )
    r1 = job_store.save_analysis_to_history(vid)
    assert r1["ok"] is True
    records = job_store.read_athlete_history_records(aid)
    assert len(records) == 1
    r2 = job_store.save_analysis_to_history(vid)
    assert r2["ok"] is True
    assert r2["code"] == "already_saved"
    records2 = job_store.read_athlete_history_records(aid)
    assert len(records2) == 1


def test_not_saved_when_not_done(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(job_store.settings, "upload_dir", str(tmp_path))
    vid = "v-save-2"
    job_store.write_initial_job(
        video_id=vid,
        task_id="t2",
        athlete_id="A101",
        session_type="normal",
        fatigue_state="no",
        event_group="100",
        file_path="x.mp4",
    )
    out = job_store.save_analysis_to_history(vid)
    assert out["ok"] is False
    assert out["code"] == "not_done"
