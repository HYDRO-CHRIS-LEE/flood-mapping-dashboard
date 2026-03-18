"""
Serialize Flappy Bird evaluation trajectories into replay JSON for the Canvas renderer.

This module is SEPARATE from flappy_eval.py. The eval engine calls this module
to generate replay data after evaluation completes.
"""

from __future__ import annotations

# Environment constants (match the Flappy Bird env used by the eval engine)
WORLD_WIDTH = 420
WORLD_HEIGHT = 580
GROUND_HEIGHT = 60


def build_replay_json(
    race_id: str,
    stage_id: int,
    eval_runs: list[dict],
    stages: dict,
) -> dict:
    """
    Build replay JSON from evaluation results.

    Parameters
    ----------
    race_id : str
        Unique identifier for this race.
    stage_id : int
        Which stage (key into *stages*) is being replayed.
    eval_runs : list[dict]
        Per-team evaluation results. Each dict contains:
            - team_name: str
            - status: "success" | "evaluation_failed"
            - episodes: list of episode dicts (present only when status == "success"),
              each with:
                - seed: int
                - frames: list of frame dicts with keys
                    t (int), bird_y (float), alive (bool), score (int),
                    action (int: 0=no flap, 1=flap, -1=dead),
                    pipes (list of dicts with x, gap_y, gap_size)
    stages : dict
        Stage definitions keyed by stage_id. Each value has gap_size, speed, label.

    Returns
    -------
    dict
        Replay payload ready to be serialised to JSON for the Canvas renderer.
    """
    stage_def = stages[stage_id]

    # Collect only successful runs
    successful_runs = [r for r in eval_runs if r.get("status") == "success"]

    # Build a seed -> {team_name -> episode_frames} index
    seed_index: dict[int, dict[str, list[dict]]] = {}
    for run in successful_runs:
        team = run["team_name"]
        for episode in run.get("episodes", []):
            seed = episode["seed"]
            if seed not in seed_index:
                seed_index[seed] = {}
            seed_index[seed][team] = episode["frames"]

    # Ordered list of team names (stable across episodes)
    team_names = [r["team_name"] for r in successful_runs]

    # Merge per-team frames into a unified timeline for each episode/seed
    episodes: list[dict] = []
    for seed in sorted(seed_index):
        teams_frames = seed_index[seed]
        merged_frames = _merge_episode_frames(team_names, teams_frames)
        episodes.append({
            "seed": seed,
            "frames": merged_frames,
        })

    return {
        "race_id": race_id,
        "stage_id": stage_id,
        "gap_size": stage_def["gap_size"],
        "speed": stage_def["speed"],
        "world_width": WORLD_WIDTH,
        "world_height": WORLD_HEIGHT,
        "ground_height": GROUND_HEIGHT,
        "episodes": episodes,
    }


def _merge_episode_frames(
    team_names: list[str],
    teams_frames: dict[str, list[dict]],
) -> list[dict]:
    """
    Merge per-team frame lists into a single list of unified frames.

    Dead teams retain their last known position with alive=False for every
    subsequent frame in the timeline.
    """
    # Determine the length of the longest trajectory
    max_t = 0
    for frames in teams_frames.values():
        if frames:
            max_t = max(max_t, max(f["t"] for f in frames))

    # Build a t-indexed lookup per team for O(1) access
    team_frame_map: dict[str, dict[int, dict]] = {}
    for team, frames in teams_frames.items():
        team_frame_map[team] = {f["t"]: f for f in frames}

    merged: list[dict] = []
    # Track last known state per team (for dead-bird carry-forward)
    last_state: dict[str, dict] = {}

    for t in range(max_t + 1):
        # Use pipe data from the first team that has a frame at this timestep
        pipes = None
        for team in team_names:
            fmap = team_frame_map.get(team, {})
            if t in fmap and pipes is None:
                pipes = fmap[t].get("pipes", [])
                break
        if pipes is None:
            pipes = []

        birds: list[dict] = []
        for team in team_names:
            fmap = team_frame_map.get(team, {})
            if t in fmap:
                f = fmap[t]
                bird = {
                    "team_name": team,
                    "y": f["bird_y"],
                    "alive": f["alive"],
                    "score": f["score"],
                    "action": f["action"],
                }
                last_state[team] = bird
            elif team in last_state:
                # Dead or missing — carry forward last position, mark dead
                bird = {
                    "team_name": team,
                    "y": last_state[team]["y"],
                    "alive": False,
                    "score": last_state[team]["score"],
                    "action": -1,
                }
            else:
                # Team had no frames at all for this episode (shouldn't happen
                # for successful runs, but be defensive)
                bird = {
                    "team_name": team,
                    "y": WORLD_HEIGHT / 2,
                    "alive": False,
                    "score": 0,
                    "action": -1,
                }
            birds.append(bird)

        merged.append({
            "t": t,
            "pipes": pipes,
            "birds": birds,
        })

    return merged
