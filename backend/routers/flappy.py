"""
Flappy Bird API router — model upload, submission, race execution.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.services.flappy_engine import (
    ALLOWED_ARCHITECTURES,
    execute_race,
    get_stages,
    get_unlocked_stages,
    submit_model,
    validate_upload,
)

router = APIRouter(prefix="/api/flappy", tags=["flappy"])

# In-memory staging area for validated (but not yet submitted) models.
# Keys are UUID strings, values are (state_dict_bytes, metadata_dict).
_model_store: dict[str, tuple[bytes, dict]] = {}


# ── Request bodies ──────────────────────────────────────────────────


class SubmitBody(BaseModel):
    team_name: str
    model_id: str
    stage_id: int


class RaceBody(BaseModel):
    stage_id: int
    admin_password: str


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("/stages")
def stages():
    """Return stage definitions and allowed architectures."""
    return {
        "ok": True,
        "data": {
            "stages": get_stages(),
            "architectures": sorted(ALLOWED_ARCHITECTURES),
        },
    }


@router.get("/unlocked/{team_name}")
def unlocked(team_name: str):
    """Return the stage IDs unlocked by *team_name*."""
    return {"ok": True, "data": get_unlocked_stages(team_name)}


@router.post("/upload")
async def upload(
    state_dict: UploadFile = File(...),
    metadata: UploadFile = File(...),
):
    """Upload and validate a model state_dict + metadata JSON.

    Stores the validated artefacts in memory so that a subsequent
    ``/submit`` call can persist them to disk.
    """
    # Validate file extensions
    if not (state_dict.filename or "").endswith(".pt"):
        raise HTTPException(status_code=400, detail="state_dict must be a .pt file")
    if not (metadata.filename or "").endswith(".json"):
        raise HTTPException(status_code=400, detail="metadata must be a .json file")

    sd_bytes = await state_dict.read()
    meta_bytes = await metadata.read()

    try:
        meta_dict = json.loads(meta_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")

    model, _meta, err = validate_upload(sd_bytes, meta_dict)
    if err:
        return {"ok": False, "error": "VALIDATION_FAILED", "message": err}

    model_id = str(uuid.uuid4())
    _model_store[model_id] = (sd_bytes, meta_dict)

    return {"ok": True, "data": {"model_id": model_id, "message": "Model validated"}}


@router.post("/submit")
def submit(body: SubmitBody):
    """Persist a previously-uploaded model to the submission store."""
    entry = _model_store.pop(body.model_id, None)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail="model_id not found — upload first via /api/flappy/upload",
        )

    sd_bytes, meta_dict = entry
    team_dir = submit_model(body.team_name, body.stage_id, sd_bytes, meta_dict)

    return {
        "ok": True,
        "data": {
            "team_name": body.team_name,
            "stage_id": body.stage_id,
            "team_dir": team_dir,
            "message": "Submission saved",
        },
    }


@router.post("/race")
def race(body: RaceBody):
    """Run a race for all submissions on the given stage (admin-only)."""
    result, err = execute_race(body.stage_id, body.admin_password)

    if err:
        if "password" in err.lower():
            raise HTTPException(status_code=403, detail=err)
        return {"ok": False, "error": "RACE_FAILED", "message": err}

    return {
        "ok": True,
        "data": {
            "race_id": result["race_id"],
            "results": result["results"],
            "replay": result["replay"],
        },
    }
