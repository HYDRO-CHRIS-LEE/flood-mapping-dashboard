"""Flappy Bird API — RF classifier-based gameplay + competition."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.rf_game_engine import (
    STAGES,
    get_model_status,
    run_rf_game,
)
from backend.services.competition_engine import (
    submit_for_competition,
    list_submitted_teams,
    run_competition,
)

router = APIRouter(prefix="/api/flappy", tags=["flappy"])


# ── Info endpoints ─────────────────────────────────────────────────


@router.get("/stages")
def stages():
    stage_data = {}
    for sid, s in STAGES.items():
        stage_data[sid] = {"label": s["label"], "pass_avg": s["pass_avg"]}
    return {"ok": True, "data": {"stages": stage_data}}


@router.get("/model-status/{team_id}")
def model_status(team_id: str):
    status = get_model_status(team_id)
    return {"ok": True, "data": status}


@router.get("/unlocked/{team_id}")
def unlocked(team_id: str):
    return {"ok": True, "data": sorted(STAGES.keys())}


# ── Student: Practice ──────────────────────────────────────────────


class PlayRequest(BaseModel):
    team_id: str
    stage_id: int


@router.post("/play")
def play(body: PlayRequest):
    """Student practice mode — single team, no leaderboard save."""
    try:
        result = run_rf_game(body.team_id, body.stage_id, mode="practice")
    except ValueError as e:
        parts = str(e).split("|", 1)
        code = parts[0] if len(parts) == 2 else "GAME_ERROR"
        msg = parts[1] if len(parts) == 2 else str(e)
        status = 404 if code == "MODEL_NOT_FOUND" else 400
        raise HTTPException(status_code=status, detail={
            "ok": False, "error": code, "message": msg,
        })
    return {"ok": True, "data": result}


# ── Student: Submit model for competition ──────────────────────────


class SubmitModelRequest(BaseModel):
    team_id: str
    team_name: str


@router.post("/submit-model")
def submit_model(body: SubmitModelRequest):
    """Student submits their classifier for official competition."""
    try:
        submit_for_competition(body.team_id, body.team_name)
    except ValueError as e:
        parts = str(e).split("|", 1)
        code = parts[0] if len(parts) == 2 else "SUBMIT_ERROR"
        msg = parts[1] if len(parts) == 2 else str(e)
        raise HTTPException(status_code=400, detail={
            "ok": False, "error": code, "message": msg,
        })
    return {"ok": True, "data": {"message": f"Model submitted for competition by {body.team_name}"}}


@router.get("/submitted-teams")
def submitted_teams():
    """List all teams that have submitted models for competition."""
    teams = list_submitted_teams()
    return {"ok": True, "data": teams}


# ── Admin: Run competition ─────────────────────────────────────────


class CompetitionRequest(BaseModel):
    admin_password: str


@router.post("/competition")
def competition(body: CompetitionRequest):
    """Admin: run full sequential competition through all stages."""
    try:
        result = run_competition(body.admin_password)
    except ValueError as e:
        parts = str(e).split("|", 1)
        code = parts[0] if len(parts) == 2 else "COMPETITION_ERROR"
        msg = parts[1] if len(parts) == 2 else str(e)
        if code == "INVALID_PASSWORD":
            raise HTTPException(status_code=403, detail={"ok": False, "error": code, "message": msg})
        raise HTTPException(status_code=400, detail={"ok": False, "error": code, "message": msg})
    return {"ok": True, "data": result}
