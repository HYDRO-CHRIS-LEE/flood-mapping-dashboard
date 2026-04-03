"""
Flappy Bird engine service — thin wrapper around utils/flappy_*.

Delegates all heavy lifting to:
  - utils.flappy_submission  (validate, save, load, list)
  - utils.flappy_eval        (STAGES, run_race)
  - utils.flappy_leaderboard (add_entry, get_sorted_by_stage, team_passed_stage)
"""

from __future__ import annotations

import os
import sys

# Headless SDL — must be set BEFORE any pygame / pyglet import
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# Ensure the flappy_bird package is importable
_FLAPPY_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "flappy_bird",
)
if _FLAPPY_ROOT not in sys.path:
    sys.path.insert(0, _FLAPPY_ROOT)

from backend.config import ADMIN_PASSWORD, DATA_ROOT
from utils.flappy_eval import STAGES, run_race
from utils.flappy_leaderboard import (
    add_entry as lb_add,
    get_sorted_by_stage,
    team_passed_stage,
)
from utils.flappy_submission import (
    ALLOWED_ARCHITECTURES,
    list_stage_submissions,
    load_submission,
    save_submission,
    validate_and_load,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FLAPPY_LB_PATH = os.path.join(DATA_ROOT, "flappy_leaderboard.json")
_LOCK_PATH = os.path.join(DATA_ROOT, ".flappy_race.lock")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_stages() -> dict:
    """Return a copy of the stage configuration dict."""
    return {k: dict(v) for k, v in STAGES.items()}


def get_unlocked_stages(team_name: str) -> list[int]:
    """Return the list of stage IDs that *team_name* has unlocked.

    Stage 1 is always unlocked.  Stage N is unlocked when the team has
    passed stage N-1.
    """
    unlocked = [1]
    for stage_id in sorted(STAGES.keys()):
        if stage_id == 1:
            continue
        if team_passed_stage(FLAPPY_LB_PATH, team_name, stage_id - 1):
            unlocked.append(stage_id)
    return unlocked


def validate_upload(
    sd_bytes: bytes, metadata: dict
) -> tuple:
    """Validate an uploaded state_dict.  Delegates to ``validate_and_load``.

    Returns (model, metadata, error_message).
    """
    return validate_and_load(sd_bytes, metadata)


def submit_model(
    team_name: str,
    stage_id: int,
    sd_bytes: bytes,
    metadata: dict,
) -> str:
    """Save a validated submission to disk.

    Returns the submission directory path.
    """
    return save_submission(DATA_ROOT, stage_id, team_name, sd_bytes, metadata)


def execute_race(
    stage_id: int, admin_password: str
) -> tuple[dict | None, str | None]:
    """Run a race for all submissions on *stage_id*.

    1. Verify admin password.
    2. Acquire a file-based lock.
    3. Load all submissions for the stage.
    4. Run the race.
    5. Update the leaderboard with each result.
    6. Return (race_result, None) on success or (None, error_msg) on failure.
    """
    if admin_password != ADMIN_PASSWORD:
        return None, "Invalid admin password."

    # Acquire lock -----------------------------------------------------------
    if os.path.exists(_LOCK_PATH):
        return None, "A race is already in progress."

    try:
        os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
        with open(_LOCK_PATH, "w") as f:
            f.write(str(os.getpid()))

        # Load submissions ---------------------------------------------------
        subs = list_stage_submissions(DATA_ROOT, stage_id)
        if not subs:
            return None, f"No submissions found for stage {stage_id}."

        teams: list[dict] = []
        for sub in subs:
            model, meta, err = load_submission(sub["team_dir"])
            if err:
                continue  # skip broken submissions
            teams.append({
                "team_name": sub["team_name"],
                "model": model,
                "submission_timestamp": sub.get("saved_at", ""),
            })

        if not teams:
            return None, "All submissions failed validation."

        # Run race -----------------------------------------------------------
        race_result = run_race(teams, stage_id, DATA_ROOT)

        # Update leaderboard -------------------------------------------------
        for entry in race_result["results"]["results"]:
            if entry["status"] != "success":
                continue
            # Find the submission timestamp for this team
            sub_ts = ""
            for t in teams:
                if t["team_name"] == entry["team_name"]:
                    sub_ts = t["submission_timestamp"]
                    break
            lb_add(
                FLAPPY_LB_PATH,
                team_name=entry["team_name"],
                stage_id=stage_id,
                avg_score=entry["avg_score"],
                max_score=entry["max_score"],
                survival_steps_avg=entry["survival_steps_avg"],
                episode_scores=entry["episode_scores"],
                passed=entry["passed"],
                race_id=race_result["race_id"],
                submission_timestamp=sub_ts,
            )

        return race_result, None

    except Exception as exc:
        return None, f"Race failed: {exc}"

    finally:
        # Release lock -------------------------------------------------------
        if os.path.exists(_LOCK_PATH):
            os.unlink(_LOCK_PATH)
