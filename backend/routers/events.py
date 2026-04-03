"""
Events API router.
Provides endpoints for listing and retrieving flood event metadata.
"""

import os
from fastapi import APIRouter, HTTPException

from backend.config import ALL_EVENTS, DATA_ROOT

router = APIRouter(prefix="/api", tags=["events"])


def _event_has_data(key: str) -> bool:
    """Check whether the event directory exists and contains at least one file."""
    d = os.path.join(DATA_ROOT, key)
    return os.path.isdir(d) and len(os.listdir(d)) > 0


@router.get("/events")
def list_events():
    """Return all events that have data directories."""
    data = []
    for key, meta in ALL_EVENTS.items():
        if _event_has_data(key):
            data.append({"key": key, **meta})
    return {"ok": True, "data": data}


@router.get("/events/{key}")
def get_event(key: str):
    """Return a single event by key, or 404 if not found."""
    if key not in ALL_EVENTS:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "EVENT_NOT_FOUND", "message": f"No event with key '{key}'"},
        )
    meta = ALL_EVENTS[key]
    return {"ok": True, "data": {"key": key, **meta}}
