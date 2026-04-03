"""
Leaderboard API router — classifier and flappy bird leaderboards from SQLite.
"""

import json

from fastapi import APIRouter

from backend.db import get_connection

router = APIRouter(prefix="/api/leaderboard", tags=["leaderboard"])


@router.get("/classifier")
def classifier_leaderboard():
    """Top 20 classifier entries by F1 score."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT team_id, team_name, f1, accuracy, precision_val, recall,
                      features, n_trees, max_depth, submitted_at
               FROM classifier_leaderboard
               ORDER BY f1 DESC
               LIMIT 20"""
        ).fetchall()
    finally:
        conn.close()

    entries = []
    for r in rows:
        features = None
        if r["features"]:
            try:
                features = json.loads(r["features"])
            except (json.JSONDecodeError, TypeError):
                features = r["features"]
        entries.append({
            "team_id": r["team_id"],
            "team_name": r["team_name"],
            "f1": r["f1"],
            "accuracy": r["accuracy"],
            "precision_val": r["precision_val"],
            "recall": r["recall"],
            "features": features,
            "n_trees": r["n_trees"],
            "max_depth": r["max_depth"],
            "submitted_at": r["submitted_at"],
        })

    return {"ok": True, "data": entries}


@router.get("/flappy/{stage_id}")
def flappy_leaderboard(stage_id: int):
    """Top 20 flappy bird entries by avg_score for a given stage."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT team_id, team_name, stage_id, avg_score, max_score,
                      survival_steps_avg, episode_scores, passed, race_id, submitted_at
               FROM flappy_leaderboard
               WHERE stage_id = ?
               ORDER BY avg_score DESC
               LIMIT 20""",
            (stage_id,),
        ).fetchall()
    finally:
        conn.close()

    entries = []
    for r in rows:
        episode_scores = None
        if r["episode_scores"]:
            try:
                episode_scores = json.loads(r["episode_scores"])
            except (json.JSONDecodeError, TypeError):
                episode_scores = r["episode_scores"]
        entries.append({
            "team_id": r["team_id"],
            "team_name": r["team_name"],
            "stage_id": r["stage_id"],
            "avg_score": r["avg_score"],
            "max_score": r["max_score"],
            "survival_steps_avg": r["survival_steps_avg"],
            "episode_scores": episode_scores,
            "passed": bool(r["passed"]),
            "race_id": r["race_id"],
            "submitted_at": r["submitted_at"],
        })

    return {"ok": True, "data": entries}
