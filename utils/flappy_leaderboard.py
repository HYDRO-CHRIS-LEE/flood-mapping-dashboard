"""JSON-file-based Flappy Bird competition leaderboard with atomic writes and team dedup."""

import json
import os
import tempfile


def load_leaderboard(path: str) -> dict:
    """Load leaderboard JSON. Returns empty structure if file missing."""
    if not os.path.exists(path):
        return {
            "competition_version": "flappy_competition_v1",
            "eval_protocol_version": "flappy_eval_v1",
            "entries": [],
        }
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


def _normalize_team_name(name: str) -> str:
    return name.strip().lower()


def add_entry(
    path: str,
    *,
    team_name: str,
    stage_id: int,
    avg_score: float,
    max_score: int,
    survival_steps_avg: float,
    episode_scores: list[float],
    passed: bool,
    race_id: str,
    submission_timestamp: str,
    status: str = "success",
) -> None:
    """Add or update a team's entry. Keeps only the best avg_score per team_name + stage_id."""
    data = load_leaderboard(path)

    norm_name = _normalize_team_name(team_name)

    existing_idx = None
    for i, entry in enumerate(data["entries"]):
        if (
            _normalize_team_name(entry["team_name"]) == norm_name
            and entry["stage_id"] == stage_id
        ):
            existing_idx = i
            break

    new_entry = {
        "team_name": team_name if existing_idx is None else data["entries"][existing_idx]["team_name"],
        "stage_id": stage_id,
        "avg_score": avg_score,
        "max_score": max_score,
        "survival_steps_avg": survival_steps_avg,
        "episode_scores": episode_scores,
        "passed": passed,
        "race_id": race_id,
        "submission_timestamp": submission_timestamp,
        "status": status,
    }

    if existing_idx is not None:
        if avg_score > data["entries"][existing_idx]["avg_score"]:
            data["entries"][existing_idx] = new_entry
    else:
        data["entries"].append(new_entry)

    save_leaderboard(path, data)


def _sort_key(entry: dict) -> tuple:
    """Sort key: avg_score desc, max_score desc, survival_steps_avg desc, earlier timestamp asc."""
    return (
        -entry["avg_score"],
        -entry["max_score"],
        -entry["survival_steps_avg"],
        entry["submission_timestamp"],
    )


def get_sorted_by_stage(path: str, stage_id: int) -> list[dict]:
    """Return entries for a given stage, sorted by ranking criteria."""
    data = load_leaderboard(path)
    stage_entries = [e for e in data["entries"] if e["stage_id"] == stage_id]
    return sorted(stage_entries, key=_sort_key)


def team_passed_stage(path: str, team_name: str, stage_id: int) -> bool:
    """Check whether a team has a passing entry for a given stage."""
    data = load_leaderboard(path)
    norm_name = _normalize_team_name(team_name)
    for entry in data["entries"]:
        if (
            _normalize_team_name(entry["team_name"]) == norm_name
            and entry["stage_id"] == stage_id
            and entry.get("passed", False)
        ):
            return True
    return False
