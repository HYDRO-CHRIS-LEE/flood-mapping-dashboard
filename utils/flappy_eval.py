"""
Deterministic Flappy Bird evaluation engine.

Runs fixed-seed episodes against the FlappyBirdEnv, collects scores and
optional per-frame replay data, and orchestrates multi-team "races".

Protocol version: flappy_eval_v1
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Headless pygame / pyglet -- must be set BEFORE any env import
# ---------------------------------------------------------------------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

# Ensure the flappy_bird package is importable
_FLAPPY_ROOT = str(Path(__file__).resolve().parent.parent / "flappy_bird")
if _FLAPPY_ROOT not in sys.path:
    sys.path.insert(0, _FLAPPY_ROOT)

from env_flappybird.flappybird_env import FlappyBirdEnv
from utils.flappy_replay import build_replay_json

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
EVAL_PROTOCOL_VERSION = "flappy_eval_v1"
ENV_VERSION = "flappy_env_v1"

# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------
SPEED_TO_DT = {
    "slow": 0.06,
    "normal": 0.05,
    "fast": 0.04,
}

STAGES = {
    1: {"label": "First Flight",            "gap_size": 170, "speed": "slow",   "pass_avg": 3},
    2: {"label": "Getting Steady",          "gap_size": 155, "speed": "slow",   "pass_avg": 5},
    3: {"label": "Tighter Gaps",            "gap_size": 140, "speed": "normal", "pass_avg": 10},
    4: {"label": "Under Pressure",          "gap_size": 125, "speed": "fast",   "pass_avg": 15},
    5: {"label": "Survival of the Fittest", "gap_size": 120, "speed": "fast",   "pass_avg": None},
}

STAGE_SEEDS = {
    1: [1031, 1049, 1063, 1091, 1103, 1129, 1151, 1181, 1213, 1237],
    2: [2039, 2053, 2069, 2081, 2099, 2111, 2129, 2141, 2153, 2161],
    3: [3037, 3049, 3061, 3079, 3089, 3109, 3119, 3137, 3163, 3181],
    4: [4007, 4019, 4027, 4049, 4057, 4073, 4091, 4099, 4111, 4127],
    5: [5003, 5009, 5021, 5039, 5051, 5059, 5077, 5081, 5099, 5107],
}

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    """Set random, numpy, and torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------


def _run_episode(
    model: torch.nn.Module,
    env: FlappyBirdEnv,
    seed: int,
    gap_size: int,
    dt: float,
    max_steps: int,
    record_frames: bool,
) -> dict:
    """
    Run one episode with a fixed seed.

    Returns a dict with:
        seed, score, survival_steps, and optionally frames (list of dicts).
    """
    _set_seeds(seed)

    obs = env.reset(gap_size=gap_size, is_random_gap=False)
    state = torch.FloatTensor(obs).unsqueeze(0)  # shape (1, 4)

    frames: list[dict] = []
    step_count = 0

    for t in range(max_steps):
        # Greedy action: argmax(Q(s)), no exploration
        with torch.no_grad():
            q_values = model(state)
            action = int(q_values.argmax(dim=1).item())

        if record_frames:
            # Capture pipe info from env walls
            pipes = []
            for w in env.walls:
                pipes.append({
                    "x": float(w.pos_x),
                    "gap_y": float(w.gap_y),
                    "gap_size": float(w.gap_size),
                })
            frames.append({
                "t": t,
                "bird_y": float(env.player.pos_y),
                "alive": True,
                "score": int(env.score),
                "action": action,
                "pipes": pipes,
            })

        obs, reward, done, info = env.step(action, gap_size=gap_size, dt=dt)
        state = torch.FloatTensor(obs).unsqueeze(0)
        step_count = t + 1

        if done:
            if record_frames:
                # Record the death frame
                pipes = []
                for w in env.walls:
                    pipes.append({
                        "x": float(w.pos_x),
                        "gap_y": float(w.gap_y),
                        "gap_size": float(w.gap_size),
                    })
                frames.append({
                    "t": t + 1,
                    "bird_y": float(env.player.pos_y),
                    "alive": False,
                    "score": int(env.score),
                    "action": -1,
                    "pipes": pipes,
                })
            break

    result = {
        "seed": seed,
        "score": int(env.score),
        "survival_steps": step_count,
    }
    if record_frames:
        result["frames"] = frames
    return result


# ---------------------------------------------------------------------------
# evaluate_model -- public API
# ---------------------------------------------------------------------------


def evaluate_model(
    model: torch.nn.Module,
    stage_id: int,
    team_name: str,
    *,
    max_steps: int = 5000,
    record_frames: bool = False,
) -> dict:
    """
    Evaluate a single model on all seeds for the given stage.

    Parameters
    ----------
    model : torch.nn.Module
        The DQN model to evaluate.
    stage_id : int
        Stage identifier (1-5).
    team_name : str
        Name of the team.
    max_steps : int
        Maximum steps per episode before forced termination.
    record_frames : bool
        If True, record per-frame data for replay generation.

    Returns
    -------
    dict
        team_name, status, episode_scores, avg_score, max_score,
        survival_steps_avg, passed, and optionally episodes (frame data).
    """
    if stage_id not in STAGES:
        raise ValueError(f"Unknown stage_id: {stage_id}")

    stage_def = STAGES[stage_id]
    seeds = STAGE_SEEDS[stage_id]
    gap_size = stage_def["gap_size"]
    dt = SPEED_TO_DT[stage_def["speed"]]
    pass_avg = stage_def["pass_avg"]

    # Put model in eval mode (greedy, no dropout / batchnorm training stats)
    model.eval()

    # Create a fresh env for this evaluation
    env = FlappyBirdEnv()

    episodes: list[dict] = []
    episode_scores: list[int] = []
    survival_steps_list: list[int] = []

    for seed in seeds:
        ep_result = _run_episode(
            model=model,
            env=env,
            seed=seed,
            gap_size=gap_size,
            dt=dt,
            max_steps=max_steps,
            record_frames=record_frames,
        )
        episode_scores.append(ep_result["score"])
        survival_steps_list.append(ep_result["survival_steps"])
        if record_frames:
            episodes.append(ep_result)

    env.close()

    avg_score = float(np.mean(episode_scores))
    max_score = int(np.max(episode_scores))
    survival_steps_avg = float(np.mean(survival_steps_list))

    # Determine pass/fail
    if pass_avg is None:
        # Stage 5: no pass threshold, always "passed"
        passed = True
    else:
        passed = avg_score >= pass_avg

    result = {
        "team_name": team_name,
        "status": "success",
        "episode_scores": episode_scores,
        "avg_score": avg_score,
        "max_score": max_score,
        "survival_steps_avg": survival_steps_avg,
        "passed": passed,
    }

    if record_frames:
        result["episodes"] = episodes

    return result


# ---------------------------------------------------------------------------
# run_race -- public API
# ---------------------------------------------------------------------------


def run_race(
    teams: list[dict],
    stage_id: int,
    data_root: str,
) -> dict:
    """
    Run a race: evaluate all teams on the given stage, rank, save artifacts.

    Parameters
    ----------
    teams : list[dict]
        Each dict must contain:
            - team_name : str
            - model : torch.nn.Module
            - submission_timestamp : str
    stage_id : int
        Stage identifier (1-5).
    data_root : str
        Root directory for saving race artifacts.

    Returns
    -------
    dict
        race_id, race_dir, manifest, results, replay.
    """
    if stage_id not in STAGES:
        raise ValueError(f"Unknown stage_id: {stage_id}")

    # Create race_id from current timestamp
    race_id = datetime.now(timezone.utc).strftime("race_%Y%m%d_%H%M%S")

    race_dir = os.path.join(data_root, "flappy_races", f"stage_{stage_id}", race_id)
    os.makedirs(race_dir, exist_ok=True)

    # Build and save immutable manifest
    manifest = {
        "race_id": race_id,
        "stage_id": stage_id,
        "stage_label": STAGES[stage_id]["label"],
        "gap_size": STAGES[stage_id]["gap_size"],
        "speed": STAGES[stage_id]["speed"],
        "dt": SPEED_TO_DT[STAGES[stage_id]["speed"]],
        "seeds": STAGE_SEEDS[stage_id],
        "pass_avg": STAGES[stage_id]["pass_avg"],
        "max_steps": 5000,
        "protocol_version": EVAL_PROTOCOL_VERSION,
        "env_version": ENV_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teams": [
            {
                "team_name": t["team_name"],
                "submission_timestamp": t.get("submission_timestamp", ""),
            }
            for t in teams
        ],
    }

    manifest_path = os.path.join(race_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Evaluate each team
    eval_runs: list[dict] = []

    for team_entry in teams:
        team_name = team_entry["team_name"]
        model = team_entry["model"]

        try:
            run_result = evaluate_model(
                model=model,
                stage_id=stage_id,
                team_name=team_name,
                max_steps=5000,
                record_frames=True,
            )
            eval_runs.append(run_result)
        except Exception:
            eval_runs.append({
                "team_name": team_name,
                "status": "evaluation_failed",
                "error": traceback.format_exc(),
                "episode_scores": [],
                "avg_score": 0.0,
                "max_score": 0,
                "survival_steps_avg": 0.0,
                "passed": False,
            })

    # Rank successful teams by avg_score descending, then max_score descending
    successful = [r for r in eval_runs if r["status"] == "success"]
    successful.sort(key=lambda r: (-r["avg_score"], -r["max_score"]))

    for rank, run in enumerate(successful, start=1):
        run["rank"] = rank

    # Failed teams get no rank
    for run in eval_runs:
        if run["status"] != "success":
            run["rank"] = None

    # Build results summary (strip frame data for the results file)
    results_summary = []
    for run in eval_runs:
        entry = {
            "team_name": run["team_name"],
            "status": run["status"],
            "rank": run.get("rank"),
            "episode_scores": run.get("episode_scores", []),
            "avg_score": run.get("avg_score", 0.0),
            "max_score": run.get("max_score", 0),
            "survival_steps_avg": run.get("survival_steps_avg", 0.0),
            "passed": run.get("passed", False),
        }
        if run["status"] == "evaluation_failed":
            entry["error"] = run.get("error", "Unknown error")
        results_summary.append(entry)

    results = {
        "race_id": race_id,
        "stage_id": stage_id,
        "protocol_version": EVAL_PROTOCOL_VERSION,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "results": results_summary,
    }

    results_path = os.path.join(race_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Generate replay JSON
    replay = build_replay_json(
        race_id=race_id,
        stage_id=stage_id,
        eval_runs=eval_runs,
        stages=STAGES,
    )

    replay_path = os.path.join(race_dir, "replay.json")
    with open(replay_path, "w", encoding="utf-8") as f:
        json.dump(replay, f, ensure_ascii=False, indent=2)

    return {
        "race_id": race_id,
        "race_dir": race_dir,
        "manifest": manifest,
        "results": results,
        "replay": replay,
    }
