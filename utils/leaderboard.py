"""JSON-file-based leaderboard with atomic writes and team dedup."""

import json
import os
import tempfile
from datetime import datetime


def load_leaderboard(path: str) -> dict:
    """Load leaderboard JSON. Returns empty structure if file missing."""
    if not os.path.exists(path):
        return {"held_out_event": "", "entries": []}
    with open(path, "r") as f:
        return json.load(f)


def save_leaderboard(path: str, data: dict) -> None:
    """Atomic write: write to temp file, then os.replace."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _normalize_team(name: str) -> str:
    return name.strip().lower()


def add_entry(
    path: str,
    held_out_event: str,
    *,
    team: str,
    f1: float,
    accuracy: float,
    precision: float,
    recall: float,
    features: list[str],
    n_trees: int,
    max_depth: int,
) -> None:
    """Add or update a team's entry. Keeps only the best F1 per team."""
    data = load_leaderboard(path)
    data["held_out_event"] = held_out_event

    norm_name = _normalize_team(team)

    existing_idx = None
    for i, entry in enumerate(data["entries"]):
        if _normalize_team(entry["team"]) == norm_name:
            existing_idx = i
            break

    new_entry = {
        "team": team if existing_idx is None else data["entries"][existing_idx]["team"],
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "features": features,
        "n_trees": n_trees,
        "max_depth": max_depth,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if existing_idx is not None:
        if f1 > data["entries"][existing_idx]["f1"]:
            data["entries"][existing_idx] = new_entry
    else:
        data["entries"].append(new_entry)

    save_leaderboard(path, data)


def get_sorted(path: str) -> list[dict]:
    """Return entries sorted by F1 descending."""
    data = load_leaderboard(path)
    return sorted(data["entries"], key=lambda e: e["f1"], reverse=True)
