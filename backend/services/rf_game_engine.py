"""RF-based Flappy Bird game engine.

Each pipe encounter = one real flood/non-flood test sample.
Correct classification = bird passes through gap.
Wrong classification = bird crashes into pipe.
"""

import os
import random

import joblib
import numpy as np
import pandas as pd

from backend.config import DATA_ROOT
from backend.services.rf_engine import (
    HELD_OUT_EVENTS,
    ALL_FEATURES,
    load_training_data,
)
from backend.services.normalization import validate_events, normalize_by_event

MODELS_DIR = os.path.join(DATA_ROOT, "rf_models")

# Game constants
WORLD_HEIGHT = 580
GROUND_HEIGHT = 60
PLAYABLE_HEIGHT = WORLD_HEIGHT - GROUND_HEIGHT  # 520
BIRD_START_Y = PLAYABLE_HEIGHT / 2  # 260
GRAVITY = 0.5
FLAP_VELOCITY = -8.0
MAX_PIPES_PRACTICE = 50
MAX_PIPES_LEADERBOARD = 200
PIPE_SPACING_X = 250

# ── Physics difficulty: two axes ──
#
# Axis 1: Feature efficiency (fewer features = harder)
#   feature_ratio = n_used / n_available (0.0 ~ 1.0)
#   Affects base speed and base gap size.
#
# Axis 2: Stage progression (higher stage = harder)
#   Each stage multiplies speed and shrinks gap further.
#
# Final: frames_per_pipe = base_frames * stage_speed_mult
#        gap_size        = base_gap    * stage_gap_mult

# ── Base physics from feature ratio ──
BASE_FRAMES_FAST = 12    # fewest features → fastest base
BASE_FRAMES_SLOW = 28    # all features → slowest base
BASE_GAP_NARROW = 100    # fewest features → narrowest base
BASE_GAP_WIDE = 160      # all features → widest base

# ── Stage multipliers (applied on top of base) ──
# speed_mult < 1.0 = faster (fewer frames per pipe)
# gap_mult < 1.0 = narrower gap
STAGES = {
    1: {
        "label": "First Flight",
        "confidence_pool": "top_40",
        "pass_avg": 8,
        "speed_mult": 1.0,     # base speed
        "gap_mult": 1.0,       # base gap
    },
    2: {
        "label": "Getting Steady",
        "confidence_pool": "top_60",
        "pass_avg": 6,
        "speed_mult": 0.85,    # 15% faster
        "gap_mult": 0.90,      # 10% narrower
    },
    3: {
        "label": "Tighter Gaps",
        "confidence_pool": "top_80",
        "pass_avg": 5,
        "speed_mult": 0.70,    # 30% faster
        "gap_mult": 0.80,      # 20% narrower
    },
    4: {
        "label": "Under Pressure",
        "confidence_pool": "all",
        "pass_avg": 4,
        "speed_mult": 0.55,    # 45% faster
        "gap_mult": 0.70,      # 30% narrower
    },
    5: {
        "label": "Survival of the Fittest",
        "confidence_pool": "bottom_50",
        "pass_avg": None,
        "speed_mult": 0.45,    # 55% faster
        "gap_mult": 0.60,      # 40% narrower
    },
}


def _physics_for_stage(feature_ratio: float, stage_id: int) -> tuple[int, int]:
    """Return (frames_per_pipe, gap_size) combining feature ratio + stage difficulty."""
    r = max(0.0, min(1.0, feature_ratio))
    stage = STAGES.get(stage_id, STAGES[1])

    # Base from feature ratio
    base_frames = BASE_FRAMES_FAST + r * (BASE_FRAMES_SLOW - BASE_FRAMES_FAST)
    base_gap = BASE_GAP_NARROW + r * (BASE_GAP_WIDE - BASE_GAP_NARROW)

    # Apply stage multipliers
    frames = max(6, int(base_frames * stage["speed_mult"]))
    gap = max(60, int(base_gap * stage["gap_mult"]))

    return frames, gap

STAGE_SEEDS = {
    1: [1031, 1049, 1063, 1091, 1103, 1129, 1151, 1181, 1213, 1237],
    2: [2039, 2053, 2069, 2081, 2099, 2111, 2129, 2141, 2153, 2161],
    3: [3037, 3049, 3061, 3079, 3089, 3109, 3119, 3137, 3163, 3181],
    4: [4007, 4019, 4027, 4049, 4057, 4073, 4091, 4099, 4111, 4127],
    5: [5003, 5009, 5021, 5039, 5051, 5059, 5077, 5081, 5099, 5107],
}


def load_model_artifact(team_id: str) -> dict | None:
    """Load persisted RF model artifact from disk."""
    path = os.path.join(MODELS_DIR, team_id, "model_artifact.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def get_model_status(team_id: str) -> dict:
    """Check if a team has a persisted model and return its metadata."""
    artifact = load_model_artifact(team_id)
    if artifact is None:
        return {"has_model": False}
    return {
        "has_model": True,
        "team_id": artifact.get("team_id", team_id),
        "team_name": artifact.get("team_name", ""),
        "features": artifact.get("features", []),
        "f1": artifact.get("metrics", {}).get("f1", 0),
        "accuracy": artifact.get("metrics", {}).get("accuracy", 0),
        "precision": artifact.get("metrics", {}).get("precision", 0),
        "recall": artifact.get("metrics", {}).get("recall", 0),
        "hyperparameters": artifact.get("hyperparameters", {}),
        "trained_at": artifact.get("created_at", ""),
    }


def _load_test_samples() -> pd.DataFrame | None:
    """Load held-out test samples for game use."""
    available = []
    for key in os.listdir(DATA_ROOT):
        rf_path = os.path.join(DATA_ROOT, key, "RF_training_samples.csv")
        if os.path.isfile(rf_path):
            available.append(key)

    raw_df = load_training_data(available)
    if raw_df is None:
        return None

    keep_cols = [c for c in raw_df.columns if c in ALL_FEATURES + ["label", "event"]]
    raw_df = raw_df[keep_cols].dropna()

    valid_df, _ = validate_events(raw_df)
    if len(valid_df) == 0:
        return None

    df = normalize_by_event(valid_df)

    # Only held-out events
    test_df = df[df["event"].isin(HELD_OUT_EVENTS)].copy()
    return test_df if len(test_df) > 0 else None


def _select_stage_samples(
    model, scaler, features: list[str],
    test_df: pd.DataFrame, stage_id: int,
) -> pd.DataFrame:
    """Select samples for a stage based on reference confidence ranking."""
    X = test_df[features].values
    if scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)
    confidence = np.max(proba, axis=1)
    test_df = test_df.copy()
    test_df["_confidence"] = confidence

    pool = STAGES[stage_id]["confidence_pool"]
    if pool == "top_40":
        threshold = np.percentile(confidence, 60)
        return test_df[test_df["_confidence"] >= threshold]
    elif pool == "top_60":
        threshold = np.percentile(confidence, 40)
        return test_df[test_df["_confidence"] >= threshold]
    elif pool == "top_80":
        threshold = np.percentile(confidence, 20)
        return test_df[test_df["_confidence"] >= threshold]
    elif pool == "bottom_50":
        threshold = np.percentile(confidence, 50)
        return test_df[test_df["_confidence"] <= threshold]
    else:  # "all"
        return test_df


def _run_episode(
    model, scaler, features: list[str],
    samples: pd.DataFrame, seed: int,
    frames_per_pipe: int = 30, gap_size: int = 150,
    max_pipes: int = 50,
) -> dict:
    """Run one game episode.

    The bird moves rightward through a pre-laid pipe map.
    The camera follows the bird. Multiple pipes visible at once.

    Frame format:
        bird_x: world x position of bird (increases each frame)
        bird_y: world y position of bird
        alive: bool
        score: pipes passed so far
    All pipe positions are returned once in ``pipe_map`` (not per-frame).
    """
    rng = random.Random(seed)
    base_indices = list(samples.index)
    rng.shuffle(base_indices)
    sample_indices = base_indices.copy()
    while len(sample_indices) < max_pipes:
        extra = base_indices.copy()
        rng.shuffle(extra)
        sample_indices.extend(extra)

    half_gap = gap_size // 2

    # ── Pre-lay all pipes at fixed world-x positions ──
    pipe_map = []
    for i in range(min(max_pipes, len(sample_indices))):
        pipe_world_x = (i + 1) * PIPE_SPACING_X
        gap_y = rng.randint(half_gap + 20, PLAYABLE_HEIGHT - half_gap - 20)
        pipe_map.append({
            "world_x": pipe_world_x,
            "gap_y": gap_y,
            "gap_size": gap_size,
        })

    # ── Pre-classify all pipes ──
    classification_events = []
    pipe_correct = []
    total_correct = 0
    n_seen = 0

    for i, pipe in enumerate(pipe_map):
        sample_row = samples.loc[sample_indices[i]]
        X_sample = sample_row[features].values.reshape(1, -1)
        if scaler is not None:
            X_sample = scaler.transform(X_sample)

        prediction = int(model.predict(X_sample)[0])
        true_label = int(sample_row["label"])
        correct = prediction == true_label
        n_seen += 1
        if correct:
            total_correct += 1
        pipe_correct.append(correct)

        classification_events.append({
            "pipe_index": i,
            "true_label": true_label,
            "predicted": prediction,
            "correct": correct,
        })

    # ── Generate frames: bird flies rightward through the map ──
    frames = []
    bird_x = 0.0
    bird_y = BIRD_START_Y
    score = 0
    alive = True
    next_pipe_idx = 0
    bird_speed_x = PIPE_SPACING_X / frames_per_pipe  # pixels per frame

    while alive and next_pipe_idx < len(pipe_map):
        pipe = pipe_map[next_pipe_idx]
        target_y = float(pipe["gap_y"])
        start_y = bird_y
        start_x = bird_x

        if pipe_correct[next_pipe_idx]:
            # Safe: smooth lerp to gap center over frames_per_pipe frames
            for f in range(frames_per_pipe):
                t = (f + 1) / frames_per_pipe
                t_smooth = t * t * (3.0 - 2.0 * t)  # cubic ease
                bird_y = start_y + (target_y - start_y) * t_smooth
                bird_y = max(0, min(bird_y, PLAYABLE_HEIGHT))
                bird_x = start_x + bird_speed_x * (f + 1)

                frames.append({
                    "t": len(frames),
                    "bird_x": round(bird_x, 1),
                    "bird_y": round(bird_y, 1),
                    "alive": True,
                    "score": score,
                })

            bird_y = target_y
            score += 1
            next_pipe_idx += 1
        else:
            # Fail: bird drifts and crashes into this pipe
            crash_frames = 20
            vel = 0.0
            for f in range(crash_frames):
                vel += GRAVITY * 1.5
                bird_y += vel
                bird_y = max(0, min(bird_y, PLAYABLE_HEIGHT))
                bird_x = start_x + bird_speed_x * (f + 1) * 0.3  # slow forward

                is_alive = f < crash_frames - 3
                frames.append({
                    "t": len(frames),
                    "bird_x": round(bird_x, 1),
                    "bird_y": round(bird_y, 1),
                    "alive": is_alive,
                    "score": score,
                })

            alive = False

    episode_accuracy = total_correct / n_seen if n_seen > 0 else 0

    return {
        "seed": seed,
        "final_score": score,
        "episode_accuracy": round(episode_accuracy, 4),
        "total_correct": total_correct,
        "n_samples_seen": n_seen,
        "classification_events": classification_events,
        "pipe_map": pipe_map,
        "frames": frames,
    }


def run_rf_game(
    team_id: str, stage_id: int, mode: str = "practice",
) -> dict:
    """Run a full game session (10 episodes) for a team.

    Returns replay data + scores + metrics.
    Raises ValueError on errors (caller should convert to HTTP errors).
    """
    artifact = load_model_artifact(team_id)
    if artifact is None:
        raise ValueError(
            "MODEL_NOT_FOUND|No classifier submitted yet. "
            "Train and submit on the Classifier page first."
        )

    model = artifact["model"]
    scaler = artifact.get("scaler")
    features = artifact["features"]

    if stage_id not in STAGES:
        raise ValueError(f"INVALID_STAGE|Stage {stage_id} does not exist.")

    test_df = _load_test_samples()
    if test_df is None:
        raise ValueError("NO_TEST_DATA|No test samples available.")

    # Check all features exist in test data
    missing = [f for f in features if f not in test_df.columns]
    if missing:
        raise ValueError(
            f"ARTIFACT_INCOMPATIBLE|Features missing from test data: {missing}"
        )

    # Compute feature efficiency ratio → physics difficulty
    n_available = len(ALL_FEATURES)
    n_used = len(features)
    feature_ratio = n_used / n_available if n_available > 0 else 1.0
    frames_per_pipe, gap_size = _physics_for_stage(feature_ratio, stage_id)

    stage_samples = _select_stage_samples(model, scaler, features, test_df, stage_id)
    if len(stage_samples) < 10:
        raise ValueError(
            f"NO_ELIGIBLE_SAMPLES|Not enough test samples for "
            f"Stage {stage_id} (found {len(stage_samples)})."
        )

    # Leaderboard: 1 long episode (200 pipes). Practice: 10 short episodes (50 pipes).
    if mode == "leaderboard":
        max_pipes = MAX_PIPES_LEADERBOARD
        seeds = [STAGE_SEEDS.get(stage_id, [42])[0]]  # single seed
    else:
        max_pipes = MAX_PIPES_PRACTICE
        seeds = STAGE_SEEDS.get(stage_id, [42] * 10)

    episodes = []
    scores = []

    for i, seed in enumerate(seeds):
        ep = _run_episode(model, scaler, features, stage_samples, seed,
                          frames_per_pipe=frames_per_pipe, gap_size=gap_size,
                          max_pipes=max_pipes)
        ep["episode_index"] = i
        ep["passed"] = (
            STAGES[stage_id]["pass_avg"] is not None
            and ep["final_score"] >= STAGES[stage_id]["pass_avg"]
        )
        episodes.append(ep)
        scores.append(ep["final_score"])

    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    max_score = max(scores) if scores else 0
    passed = (
        STAGES[stage_id]["pass_avg"] is not None
        and avg_score >= STAGES[stage_id]["pass_avg"]
    )

    return {
        "stage_id": stage_id,
        "team_id": team_id,
        "team_name": artifact.get("team_name", ""),
        "mode": mode,
        "physics": {
            "feature_ratio": round(feature_ratio, 2),
            "features_used": n_used,
            "features_available": n_available,
            "frames_per_pipe": frames_per_pipe,
            "gap_size": gap_size,
        },
        "episodes": episodes,
        "summary": {
            "avg_score": avg_score,
            "max_score": max_score,
            "scores": scores,
            "passed": passed,
        },
    }
