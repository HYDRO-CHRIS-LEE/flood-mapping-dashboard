"""
SQLite database initialization and connection management.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

from backend.config import DATA_ROOT, DB_PATH

_CREATE_CLASSIFIER_LEADERBOARD = """
CREATE TABLE IF NOT EXISTS classifier_leaderboard (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id         TEXT NOT NULL,
    team_name       TEXT NOT NULL,
    f1              REAL,
    accuracy        REAL,
    precision_val   REAL,
    recall          REAL,
    features        TEXT,
    n_trees         INTEGER,
    max_depth       INTEGER,
    test_events     TEXT,
    submitted_at    TEXT NOT NULL
);
"""

_CREATE_FLAPPY_LEADERBOARD = """
CREATE TABLE IF NOT EXISTS flappy_leaderboard (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id             TEXT NOT NULL,
    team_name           TEXT NOT NULL,
    stage_id            INTEGER,
    avg_score           REAL,
    max_score           REAL,
    survival_steps_avg  REAL,
    episode_scores      TEXT,
    passed              INTEGER,
    race_id             TEXT,
    submitted_at        TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    """Return a sqlite3 connection with Row factory and WAL journal mode."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _migrate_classifier_json(conn: sqlite3.Connection) -> int:
    """Migrate legacy leaderboard.json into classifier_leaderboard table."""
    path = os.path.join(DATA_ROOT, "leaderboard.json")
    if not os.path.isfile(path):
        return 0

    with open(path) as f:
        data = json.load(f)

    entries = data.get("entries", data if isinstance(data, list) else [])
    count = 0
    for entry in entries:
        conn.execute(
            """INSERT INTO classifier_leaderboard
               (team_id, team_name, f1, accuracy, precision_val, recall,
                features, n_trees, max_depth, test_events, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.get("team_id", entry.get("team_name", "unknown")),
                entry.get("team_name", "unknown"),
                entry.get("f1"),
                entry.get("accuracy"),
                entry.get("precision_val", entry.get("precision")),
                entry.get("recall"),
                json.dumps(entry.get("features")) if entry.get("features") else None,
                entry.get("n_trees"),
                entry.get("max_depth"),
                json.dumps(entry.get("test_events")) if entry.get("test_events") else None,
                entry.get("submission_timestamp", datetime.now(timezone.utc).isoformat()),
            ),
        )
        count += 1
    conn.commit()
    return count


def _migrate_flappy_json(conn: sqlite3.Connection) -> int:
    """Migrate legacy flappy_leaderboard.json into flappy_leaderboard table."""
    path = os.path.join(DATA_ROOT, "flappy_leaderboard.json")
    if not os.path.isfile(path):
        return 0

    with open(path) as f:
        data = json.load(f)

    entries = data.get("entries", data if isinstance(data, list) else [])
    count = 0
    for entry in entries:
        conn.execute(
            """INSERT INTO flappy_leaderboard
               (team_id, team_name, stage_id, avg_score, max_score,
                survival_steps_avg, episode_scores, passed, race_id, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.get("team_id", entry.get("team_name", "unknown")),
                entry.get("team_name", "unknown"),
                entry.get("stage_id"),
                entry.get("avg_score"),
                entry.get("max_score"),
                entry.get("survival_steps_avg"),
                json.dumps(entry.get("episode_scores")) if entry.get("episode_scores") else None,
                1 if entry.get("passed") else 0,
                entry.get("race_id"),
                entry.get("submission_timestamp", datetime.now(timezone.utc).isoformat()),
            ),
        )
        count += 1
    conn.commit()
    return count


def init_db() -> None:
    """Create tables if needed and migrate legacy JSON files when tables are empty."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    try:
        conn.execute(_CREATE_CLASSIFIER_LEADERBOARD)
        conn.execute(_CREATE_FLAPPY_LEADERBOARD)
        conn.commit()

        # Migrate legacy JSON only if tables are empty
        row = conn.execute("SELECT COUNT(*) FROM classifier_leaderboard").fetchone()
        if row[0] == 0:
            _migrate_classifier_json(conn)

        row = conn.execute("SELECT COUNT(*) FROM flappy_leaderboard").fetchone()
        if row[0] == 0:
            _migrate_flappy_json(conn)
    finally:
        conn.close()
