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
MAX_PIPES = 50
PIPE_SPACING_X = 250

# Stage definitions (classification difficulty, not physics)
STAGES = {
    1: {"label": "First Flight",            "confidence_pool": "top_40", "pass_avg": 8},
    2: {"label": "Getting Steady",          "confidence_pool": "top_60", "pass_avg": 6},
    3: {"label": "Tighter Gaps",            "confidence_pool": "top_80", "pass_avg": 5},
    4: {"label": "Under Pressure",          "confidence_pool": "all",    "pass_avg": 4},
    5: {"label": "Survival of the Fittest", "confidence_pool": "bottom_50", "pass_avg": None},
}

STAGE_SEEDS = {
    1: [1031, 1049, 1063, 1091, 1103, 1129, 1151, 1181, 1213, 1237],
    2: [2039, 2053, 2069, 2081, 2099, 2111, 2129, 2141, 2153, 2161],
    3: [3037, 3049, 3061, 3079, 3089, 3109, 3119, 3137, 3163, 3181],
    4: [4007, 4019, 4027, 4049, 4057, 4073, 4091, 4099, 4111, 4127],
    5: [5003, 5009, 5021, 5039, 5051, 5059, 5077, 5081, 5099, 5107],
}

GAP_SIZE = 150


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
) -> dict:
    """Run one game episode. Returns score, frames, classification events."""
    rng = random.Random(seed)
    sample_indices = list(samples.index)
    rng.shuffle(sample_indices)

    frames = []
    classification_events = []
    bird_y = BIRD_START_Y
    bird_vel = 0.0
    score = 0
    alive = True
    total_correct = 0
    n_seen = 0

    for pipe_idx in range(min(MAX_PIPES, len(sample_indices))):
        if not alive:
            break

        # Generate pipe position
        gap_y = rng.randint(80, PLAYABLE_HEIGHT - 80)

        # Classify sample
        sample_row = samples.loc[sample_indices[pipe_idx]]
        X_sample = sample_row[features].values.reshape(1, -1)
        if scaler is not None:
            X_sample = scaler.transform(X_sample)

        prediction = int(model.predict(X_sample)[0])
        true_label = int(sample_row["label"])
        correct = prediction == true_label
        n_seen += 1

        classification_events.append({
            "pipe_index": pipe_idx,
            "true_label": true_label,
            "predicted": prediction,
            "correct": correct,
        })

        if correct:
            total_correct += 1
            # Safe trajectory: bird moves toward gap center
            target_y = gap_y
            n_frames = 30  # frames to reach next pipe
            for f in range(n_frames):
                # Simple proportional control toward target
                diff = target_y - bird_y
                if diff > 5:
                    bird_vel = max(bird_vel + GRAVITY, FLAP_VELOCITY)
                    action = 0  # don't flap (fall toward gap below)
                    if bird_y > target_y:
                        bird_vel = FLAP_VELOCITY
                        action = 1
                elif diff < -5:
                    bird_vel = FLAP_VELOCITY
                    action = 1  # flap up toward gap above
                else:
                    bird_vel += GRAVITY * 0.3
                    action = 0

                bird_vel += GRAVITY
                bird_vel = max(min(bird_vel, 10), -10)
                bird_y += bird_vel
                bird_y = max(0, min(bird_y, PLAYABLE_HEIGHT))

                pipe_x = PIPE_SPACING_X - (f * PIPE_SPACING_X / n_frames)
                frames.append({
                    "t": len(frames),
                    "bird_y": round(bird_y, 1),
                    "alive": True,
                    "score": score,
                    "action": action,
                    "pipes": [{"x": round(pipe_x, 1), "gap_y": gap_y, "gap_size": GAP_SIZE}],
                })

            score += 1
        else:
            # Fail trajectory: bird drifts and crashes
            for f in range(15):
                bird_vel += GRAVITY
                bird_y += bird_vel
                bird_y = max(0, min(bird_y, PLAYABLE_HEIGHT))
                pipe_x = PIPE_SPACING_X - (f * PIPE_SPACING_X / 15)
                is_alive = f < 12
                frames.append({
                    "t": len(frames),
                    "bird_y": round(bird_y, 1),
                    "alive": is_alive,
                    "score": score,
                    "action": -1 if not is_alive else 0,
                    "pipes": [{"x": round(pipe_x, 1), "gap_y": gap_y, "gap_size": GAP_SIZE}],
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

    stage_samples = _select_stage_samples(model, scaler, features, test_df, stage_id)
    if len(stage_samples) < 10:
        raise ValueError(
            f"NO_ELIGIBLE_SAMPLES|Not enough test samples for "
            f"Stage {stage_id} (found {len(stage_samples)})."
        )

    seeds = STAGE_SEEDS.get(stage_id, [42] * 10)
    episodes = []
    scores = []

    for i, seed in enumerate(seeds):
        ep = _run_episode(model, scaler, features, stage_samples, seed)
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
        "episodes": episodes,
        "summary": {
            "avg_score": avg_score,
            "max_score": max_score,
            "scores": scores,
            "passed": passed,
        },
    }
