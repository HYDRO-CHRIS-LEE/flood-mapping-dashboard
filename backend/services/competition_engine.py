"""Multi-team competition engine.

Runs all submitted models simultaneously through sequential stages.
Score >= 100 to survive to next stage. Eliminated teams ranked by
(stage_eliminated, score). Final ranking by last-stage score.
"""

import os
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from backend.config import DATA_ROOT, ADMIN_PASSWORD
from backend.services.rf_game_engine import (
    STAGES, STAGE_SEEDS, MODELS_DIR, PLAYABLE_HEIGHT, BIRD_START_Y,
    GRAVITY, PIPE_SPACING_X,
    _load_test_samples, _select_stage_samples,
)
from backend.services.rf_engine import ALL_FEATURES

COMPETITION_MODELS_DIR = os.path.join(DATA_ROOT, "competition_models")
SURVIVE_THRESHOLD = 100  # score needed to advance to next stage
COMPETITION_MAX_PIPES = 200

# Stage-specific physics for competition (escalating difficulty)
COMPETITION_STAGE_PHYSICS = {
    1: {"frames_per_pipe": 24, "gap_size": 160},   # easy
    2: {"frames_per_pipe": 20, "gap_size": 140},   # medium
    3: {"frames_per_pipe": 16, "gap_size": 120},   # hard
    4: {"frames_per_pipe": 12, "gap_size": 100},   # very hard
    5: {"frames_per_pipe": 9,  "gap_size": 80},    # extreme
}


def _select_stage_samples_simple(test_df: pd.DataFrame, stage_id: int) -> pd.DataFrame:
    """Select samples for competition stage without model-dependent confidence.

    Uses label balance as difficulty proxy:
    Stage 1-2: balanced (50/50 flood/non-flood) — easier
    Stage 3: slightly more non-flood — harder (more tricky negatives)
    Stage 4-5: random mix from all samples — hardest
    """
    pool_config = STAGES.get(stage_id, {}).get("confidence_pool", "all")

    if pool_config == "top_40":
        # Easy: balanced, clear-cut samples — oversample to ensure enough
        flood = test_df[test_df["label"] == 1]
        nonflood = test_df[test_df["label"] == 0]
        n = min(len(flood), len(nonflood), COMPETITION_MAX_PIPES // 2)
        if n > 0:
            return pd.concat([flood.sample(n=n, random_state=42),
                              nonflood.sample(n=n, random_state=42)])
        return test_df

    elif pool_config == "top_60":
        flood = test_df[test_df["label"] == 1]
        nonflood = test_df[test_df["label"] == 0]
        nf = min(len(flood), COMPETITION_MAX_PIPES // 3)
        nn = min(len(nonflood), COMPETITION_MAX_PIPES * 2 // 3)
        if nf > 0 and nn > 0:
            return pd.concat([flood.sample(n=nf, random_state=42),
                              nonflood.sample(n=nn, random_state=42)])
        return test_df

    else:  # top_80, all, bottom_50 — use all samples
        return test_df


def list_submitted_teams() -> list[dict]:
    """List all teams that have submitted models for competition."""
    if not os.path.isdir(COMPETITION_MODELS_DIR):
        return []
    teams = []
    for team_id in os.listdir(COMPETITION_MODELS_DIR):
        path = os.path.join(COMPETITION_MODELS_DIR, team_id, "model_artifact.pkl")
        if os.path.isfile(path):
            try:
                artifact = joblib.load(path)
                teams.append({
                    "team_id": team_id,
                    "team_name": artifact.get("team_name", team_id),
                    "features": artifact.get("features", []),
                    "f1": artifact.get("metrics", {}).get("f1", 0),
                    "submitted_at": artifact.get("created_at", ""),
                })
            except Exception:
                continue
    return sorted(teams, key=lambda t: t["team_name"])


def submit_for_competition(team_id: str, team_name: str) -> str:
    """Copy a team's trained model to competition directory."""
    src = os.path.join(MODELS_DIR, team_id, "model_artifact.pkl")
    if not os.path.isfile(src):
        raise ValueError("MODEL_NOT_FOUND|No trained classifier found. Train and save first.")

    dst_dir = os.path.join(COMPETITION_MODELS_DIR, team_id)
    os.makedirs(dst_dir, exist_ok=True)

    artifact = joblib.load(src)
    artifact["team_name"] = team_name
    artifact["competition_submitted_at"] = datetime.now(timezone.utc).isoformat()
    joblib.dump(artifact, os.path.join(dst_dir, "model_artifact.pkl"))

    return dst_dir


def _run_stage_multi(
    teams: list[dict],
    stage_id: int,
    test_df: pd.DataFrame,
) -> dict:
    """Run one stage for multiple teams simultaneously.

    Returns:
        {
            "stage_id": int,
            "survivors": [team_id, ...],
            "eliminated": [{team_id, team_name, score, pipe_index}, ...],
            "scores": {team_id: score, ...},
            "frames": [{t, pipes, birds: [{team_id, team_name, y, alive, score}, ...]}, ...],
            "pipe_map": [{world_x, gap_y, gap_size}, ...],
        }
    """
    if not teams:
        return {"stage_id": stage_id, "survivors": [], "eliminated": [],
                "scores": {}, "frames": [], "pipe_map": []}

    seed = STAGE_SEEDS.get(stage_id, [42])[0]
    rng = random.Random(seed)

    # Stage-specific physics
    phys = COMPETITION_STAGE_PHYSICS.get(stage_id, {"frames_per_pipe": 20, "gap_size": 130})
    gap_size = phys["gap_size"]
    fps = phys["frames_per_pipe"]
    half_gap = gap_size // 2
    bird_speed_x = PIPE_SPACING_X / fps

    # Select samples for this stage — use ALL features for reference model
    # to ensure fair sample selection across teams
    stage_samples = _select_stage_samples_simple(test_df, stage_id)
    if len(stage_samples) < COMPETITION_MAX_PIPES:
        # Recycle samples
        base_idx = list(stage_samples.index)
        while len(base_idx) < COMPETITION_MAX_PIPES:
            extra = list(stage_samples.index)
            rng.shuffle(extra)
            base_idx.extend(extra)
        sample_indices = base_idx[:COMPETITION_MAX_PIPES]
    else:
        sample_indices = list(stage_samples.index)
        rng.shuffle(sample_indices)
        sample_indices = sample_indices[:COMPETITION_MAX_PIPES]

    # Pre-lay pipes
    pipe_map = []
    for i in range(COMPETITION_MAX_PIPES):
        pipe_world_x = (i + 1) * PIPE_SPACING_X
        gap_y = rng.randint(half_gap + 20, PLAYABLE_HEIGHT - half_gap - 20)
        pipe_map.append({"world_x": pipe_world_x, "gap_y": gap_y, "gap_size": gap_size})

    # Pre-classify all pipes for all teams
    team_correct = {}  # team_id -> [bool, bool, ...]
    for team in teams:
        a = team["artifact"]
        model = a["model"]
        scaler = a.get("scaler")
        features = a["features"]

        correct_list = []
        for idx in sample_indices:
            row = stage_samples.loc[idx]
            # Use only features this model was trained on
            available = [f for f in features if f in row.index]
            if len(available) != len(features):
                correct_list.append(False)
                continue
            X = row[features].values.reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)
            pred = int(model.predict(X)[0])
            true = int(row["label"])
            correct_list.append(pred == true)
        team_correct[team["team_id"]] = correct_list

    # Generate synchronized frames
    frames = []
    bird_states = {}
    for team in teams:
        bird_states[team["team_id"]] = {
            "y": BIRD_START_Y,
            "alive": True,
            "score": 0,
            "team_name": team["team_name"],
        }

    eliminated = []
    scores = {}

    for pipe_idx in range(COMPETITION_MAX_PIPES):
        pipe = pipe_map[pipe_idx]
        target_y = float(pipe["gap_y"])

        # For each alive team, determine outcome at this pipe
        for team in teams:
            tid = team["team_id"]
            bs = bird_states[tid]
            if not bs["alive"]:
                continue

            if team_correct[tid][pipe_idx]:
                bs["score"] += 1
            else:
                bs["alive"] = False
                eliminated.append({
                    "team_id": tid,
                    "team_name": bs["team_name"],
                    "score": bs["score"],
                    "pipe_index": pipe_idx,
                })

        # Generate transition frames for this pipe
        for f in range(fps):
            t_ratio = (f + 1) / fps
            t_smooth = t_ratio * t_ratio * (3.0 - 2.0 * t_ratio)

            birds_frame = []
            for team in teams:
                tid = team["team_id"]
                bs = bird_states[tid]

                if bs["alive"]:
                    # Smooth lerp toward gap
                    start_y = bs.get("_start_y", bs["y"])
                    new_y = start_y + (target_y - start_y) * t_smooth
                    new_y = max(0, min(new_y, PLAYABLE_HEIGHT))
                    bs["_draw_y"] = new_y
                elif not bs.get("_crashed"):
                    # Just died at this pipe — falling animation
                    fall_y = bs.get("_draw_y", bs["y"]) + GRAVITY * 3 * (f + 1)
                    fall_y = min(fall_y, PLAYABLE_HEIGHT)
                    bs["_draw_y"] = fall_y
                    if f == fps - 1:
                        bs["_crashed"] = True

                birds_frame.append({
                    "team_id": tid,
                    "team_name": bs["team_name"],
                    "y": round(bs.get("_draw_y", bs["y"]), 1),
                    "alive": bs["alive"],
                    "score": bs["score"],
                })

            bird_x = (pipe_idx * PIPE_SPACING_X) + bird_speed_x * (f + 1)
            frames.append({
                "t": len(frames),
                "bird_x": round(bird_x, 1),
                "birds": birds_frame,
            })

        # Update start positions for next pipe
        for team in teams:
            tid = team["team_id"]
            bs = bird_states[tid]
            if bs["alive"]:
                bs["y"] = target_y
                bs["_start_y"] = target_y

        # Check if all eliminated
        alive_count = sum(1 for bs in bird_states.values() if bs["alive"])
        if alive_count == 0:
            break

    # Record final scores for survivors
    survivors = []
    for team in teams:
        tid = team["team_id"]
        bs = bird_states[tid]
        scores[tid] = bs["score"]
        if bs["alive"]:
            survivors.append(tid)

    return {
        "stage_id": stage_id,
        "survivors": survivors,
        "eliminated": eliminated,
        "scores": scores,
        "frames": frames,
        "pipe_map": pipe_map,
    }


def generate_demo_teams(n_teams: int = 20) -> list[dict]:
    """Generate N fake teams with varying RF models for testing."""
    from sklearn.ensemble import RandomForestClassifier
    from backend.services.normalization import validate_events, normalize_by_event

    test_df = _load_test_samples()
    if test_df is None:
        raise ValueError("NO_TEST_DATA|No test samples available.")

    # Load raw training data
    available = []
    for key in os.listdir(DATA_ROOT):
        rf_path = os.path.join(DATA_ROOT, key, "RF_training_samples.csv")
        if os.path.isfile(rf_path):
            available.append(key)

    from backend.services.rf_engine import load_training_data, HELD_OUT_EVENTS
    raw_df = load_training_data(available)
    if raw_df is None:
        raise ValueError("NO_TEST_DATA|No training data available.")

    keep_cols = [c for c in raw_df.columns if c in ALL_FEATURES + ["label", "event"]]
    raw_df = raw_df[keep_cols].dropna()
    valid_df, _ = validate_events(raw_df)
    df = normalize_by_event(valid_df)

    from backend.services.rf_engine import event_based_split
    train_df, test_split = event_based_split(df, HELD_OUT_EVENTS)

    TEAM_NAMES = [
        "FloodHunters", "AquaMinds", "DataDeluge", "TidalForce", "StormChasers",
        "WaveRiders", "RainMakers", "DamBreakers", "CurrentFlow", "DeepWaters",
        "PeakSurge", "RiverBots", "MonsoonAI", "FlashFlood", "CycloneNet",
        "TsunamiLab", "DeltaForce", "BayWatch", "SeaLevel", "ArcticMelt",
    ]

    teams = []
    feature_combos = [
        ["NDWI"],
        ["NDWI", "elevation"],
        ["NDWI", "slope"],
        ["NDWI", "elevation", "slope"],
        ["NDWI", "MNDWI"],
        ["NDWI", "MNDWI", "elevation"],
        ["NDWI", "MNDWI", "elevation", "slope"],
        ["NDWI", "elevation", "slope", "permanent_water"],
        ["NDWI", "MNDWI", "elevation", "slope", "permanent_water"],
        ["elevation", "slope"],
        ["MNDWI", "elevation"],
        ["NDWI", "MNDWI", "slope"],
        ["NDWI", "permanent_water"],
        ["slope", "permanent_water"],
        ["NDWI", "elevation", "permanent_water"],
        ["MNDWI", "slope", "permanent_water"],
        ["NDWI", "MNDWI", "elevation", "permanent_water"],
        ["elevation", "slope", "permanent_water"],
        ["NDWI", "MNDWI", "slope", "permanent_water"],
        ["MNDWI", "elevation", "slope"],
    ]

    for i in range(min(n_teams, len(TEAM_NAMES))):
        features = feature_combos[i % len(feature_combos)]
        n_trees = random.choice([30, 50, 80, 100, 150])
        max_depth = random.choice([3, 5, 7, 10])

        available_features = [f for f in features if f in train_df.columns]
        if not available_features:
            continue

        X_tr = train_df[available_features].values
        y_tr = train_df["label"].values

        clf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=max_depth,
            random_state=42 + i, n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)

        team_id = f"demo-team-{i:03d}"
        team_name = TEAM_NAMES[i]

        # Save to competition models dir
        dst_dir = os.path.join(COMPETITION_MODELS_DIR, team_id)
        os.makedirs(dst_dir, exist_ok=True)
        artifact = {
            "model": clf,
            "features": available_features,
            "scaler": None,
            "hyperparameters": {"n_trees": n_trees, "max_depth": max_depth},
            "metrics": {"f1": 0, "accuracy": 0},
            "held_out_events": HELD_OUT_EVENTS,
            "class_labels": {0: "non-flood", 1: "flood"},
            "artifact_version": "rf_artifact_v1",
            "team_id": team_id,
            "team_name": team_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(artifact, os.path.join(dst_dir, "model_artifact.pkl"))
        teams.append({"team_id": team_id, "team_name": team_name, "features": available_features})

    return teams


def run_competition(admin_password: str) -> dict:
    """Run full sequential competition through all stages.

    Returns:
        {
            "stages": [{stage results}, ...],
            "final_ranking": [{team_id, team_name, final_score, eliminated_stage}, ...],
            "replay": {stage_id -> {frames, pipe_map}},
        }
    """
    if admin_password != ADMIN_PASSWORD:
        raise ValueError("INVALID_PASSWORD|Invalid admin password.")

    submitted = list_submitted_teams()
    if not submitted:
        raise ValueError("NO_TEAMS|No teams have submitted models for competition.")

    # Load all artifacts
    teams = []
    for t in submitted:
        path = os.path.join(COMPETITION_MODELS_DIR, t["team_id"], "model_artifact.pkl")
        artifact = joblib.load(path)
        teams.append({
            "team_id": t["team_id"],
            "team_name": t["team_name"],
            "artifact": artifact,
        })

    test_df = _load_test_samples()
    if test_df is None:
        raise ValueError("NO_TEST_DATA|No test samples available.")

    stage_ids = sorted(STAGES.keys())
    stage_results = []
    replay = {}
    all_eliminated = []  # ordered by elimination
    active_teams = list(teams)

    for stage_id in stage_ids:
        if not active_teams:
            break

        result = _run_stage_multi(active_teams, stage_id, test_df)
        stage_results.append({
            "stage_id": stage_id,
            "teams_entered": len(active_teams),
            "survivors": result["survivors"],
            "eliminated": result["eliminated"],
            "scores": result["scores"],
        })

        phys = COMPETITION_STAGE_PHYSICS.get(stage_id, {"gap_size": 130})
        replay[stage_id] = {
            "frames": result["frames"],
            "pipe_map": result["pipe_map"],
            "gap_size": phys["gap_size"],
        }

        # Record eliminated teams (earliest eliminated first)
        for elim in result["eliminated"]:
            all_eliminated.append({
                "team_id": elim["team_id"],
                "team_name": elim["team_name"],
                "eliminated_stage": stage_id,
                "score": elim["score"],
            })

        # Filter to survivors only
        survivor_set = set(result["survivors"])

        # Survivors need score >= threshold to advance (except last stage)
        if stage_id < max(stage_ids):
            advancing = []
            for t in active_teams:
                tid = t["team_id"]
                if tid in survivor_set and result["scores"].get(tid, 0) >= SURVIVE_THRESHOLD:
                    advancing.append(t)
                elif tid in survivor_set:
                    # Survived but didn't reach threshold
                    all_eliminated.append({
                        "team_id": tid,
                        "team_name": t["team_name"],
                        "eliminated_stage": stage_id,
                        "score": result["scores"].get(tid, 0),
                    })
            active_teams = advancing
        else:
            active_teams = [t for t in active_teams if t["team_id"] in survivor_set]

    # Build final ranking
    # 1. Last-stage survivors ranked by score (descending)
    # 2. Then eliminated teams in reverse order (last eliminated = higher rank)
    final_ranking = []

    # Survivors of the last played stage
    if stage_results:
        last = stage_results[-1]
        survivor_scores = []
        for tid in last["survivors"]:
            for t in teams:
                if t["team_id"] == tid:
                    survivor_scores.append({
                        "team_id": tid,
                        "team_name": t["team_name"],
                        "final_score": last["scores"].get(tid, 0),
                        "eliminated_stage": None,
                    })
        survivor_scores.sort(key=lambda x: -x["final_score"])
        final_ranking.extend(survivor_scores)

    # Eliminated teams (reverse order — last eliminated first)
    for elim in reversed(all_eliminated):
        final_ranking.append({
            "team_id": elim["team_id"],
            "team_name": elim["team_name"],
            "final_score": elim["score"],
            "eliminated_stage": elim["eliminated_stage"],
        })

    return {
        "stage_results": stage_results,
        "final_ranking": final_ranking,
        "replay": replay,
        "total_teams": len(teams),
        "stages_played": len(stage_results),
    }
