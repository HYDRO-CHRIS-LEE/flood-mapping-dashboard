"""Flappy Bird API — RF classifier-based gameplay."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.rf_game_engine import (
    STAGES,
    get_model_status,
    run_rf_game,
)
from backend.services.flappy_engine import (
    get_unlocked_stages,
    execute_race,
    FLAPPY_LB_PATH,
)
from utils.flappy_leaderboard import add_entry as lb_add

router = APIRouter(prefix="/api/flappy", tags=["flappy"])


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("/stages")
def stages():
    """Return stage definitions."""
    stage_data = {}
    for sid, s in STAGES.items():
        stage_data[sid] = {"label": s["label"], "pass_avg": s["pass_avg"]}
    return {"ok": True, "data": {"stages": stage_data}}


@router.get("/model-status/{team_id}")
def model_status(team_id: str):
    """Check if a team has a persisted RF model."""
    status = get_model_status(team_id)
    return {"ok": True, "data": status}


@router.get("/unlocked/{team_id}")
def unlocked(team_id: str):
    """Return all stage IDs — all stages are available once a model is submitted."""
    return {"ok": True, "data": sorted(STAGES.keys())}


# ── Request bodies ──────────────────────────────────────────────────


class PlayRequest(BaseModel):
    team_id: str
    stage_id: int
    mode: str = "practice"  # "practice" or "leaderboard"


class SaveResultRequest(BaseModel):
    team_id: str
    team_name: str
    stage_id: int
    avg_score: float
    max_score: int
    episode_scores: list[float]
    passed: bool


class RaceBody(BaseModel):
    stage_id: int
    admin_password: str


# ── Play / Save / Race ─────────────────────────────────────────────


@router.post("/play")
def play(body: PlayRequest):
    """Run an RF-based game session (10 episodes)."""
    try:
        result = run_rf_game(body.team_id, body.stage_id, body.mode)
    except ValueError as e:
        parts = str(e).split("|", 1)
        code = parts[0] if len(parts) == 2 else "GAME_ERROR"
        msg = parts[1] if len(parts) == 2 else str(e)
        status = 404 if code == "MODEL_NOT_FOUND" else 400
        raise HTTPException(status_code=status, detail={
            "ok": False, "error": code, "message": msg,
        })
    return {"ok": True, "data": result}


@router.post("/save-result")
def save_result(body: SaveResultRequest):
    """Save a game result to the leaderboard."""
    try:
        lb_add(
            FLAPPY_LB_PATH,
            team_name=body.team_name,
            stage_id=body.stage_id,
            avg_score=body.avg_score,
            max_score=body.max_score,
            survival_steps_avg=0,
            episode_scores=body.episode_scores,
            passed=body.passed,
            race_id="",
            submission_timestamp="",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={
            "ok": False, "error": "LEADERBOARD_SAVE_FAILED", "message": str(exc),
        })
    return {"ok": True, "data": {"message": "Result saved to leaderboard"}}


@router.post("/race")
def race(body: RaceBody):
    """Run a race for all submissions on the given stage (admin-only)."""
    result, err = execute_race(body.stage_id, body.admin_password)
    if err:
        if "password" in err.lower():
            raise HTTPException(status_code=403, detail={
                "ok": False, "error": "INVALID_PASSWORD", "message": err,
            })
        raise HTTPException(status_code=400, detail={
            "ok": False, "error": "RACE_FAILED", "message": err,
        })
    return {
        "ok": True,
        "data": {
            "race_id": result["race_id"],
            "results": result["results"],
            "replay": result["replay"],
        },
    }
