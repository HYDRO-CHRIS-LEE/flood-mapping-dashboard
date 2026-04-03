# RF Classifier → Flappy Bird Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DQN-based Flappy Bird with RF classifier-based gameplay where each pipe = a real flood test sample, the student's RF model classifies it, correct = pass, wrong = crash.

**Architecture:** Modify `train_rf()` to return the model object alongside metrics. On classifier submit, persist the model artifact to disk via `joblib`. New `rf_game_engine.py` service runs game episodes using the persisted RF model against real test samples. Stage difficulty uses confidence-based sample pools. Frontend removes DQN UI and adds "Deploy My Classifier" flow.

**Tech Stack:** FastAPI, scikit-learn, joblib, existing FlappyBirdEnv, Plotly.js, Canvas replay

---

### Task 1: Modify `train_rf()` to return model + scaler objects

**Files:**
- Modify: `backend/services/rf_engine.py`

Currently `train_rf()` returns `(metrics, importance)`. It needs to also return the trained RF model and scaler for later persistence.

- [ ] **Step 1: Change return type of `train_rf()`**

Change the return from `tuple[dict | None, dict | None]` to `tuple[dict | None, dict | None, object | None, object | None]` — `(metrics, importance, model, scaler)`.

At the end of `train_rf()`, after `y_pred = clf.predict(X_te)`, before returning, the function already has `clf` and `scaler`. Change the return statement from:

```python
return metrics, importance
```

to:

```python
return metrics, importance, clf, scaler
```

And the failure return from `return None, None` to `return None, None, None, None`.

- [ ] **Step 2: Update classifier router to receive new return values**

In `backend/routers/classifier.py`, the call `metrics, importance = train_rf(...)` must become `metrics, importance, model, scaler = train_rf(...)`. Store `model`, `scaler`, and the request params in a module-level temp dict keyed by a request-scoped identifier (to be saved on submit).

Add to `classifier.py`:
```python
import uuid

# Temporary in-memory store for trained models (not yet submitted)
# Cleared on submit or overwritten on next train.
_trained_models: dict[str, dict] = {}
```

After successful training:
```python
train_id = str(uuid.uuid4())
_trained_models[body_team_id_or_train_id] = {
    "model": model,
    "scaler": scaler,
    "features": body.features,
    "hyperparameters": { ... },
    "metrics": metrics,
}
```

Wait — the train endpoint doesn't receive team_id currently. We need to add it to the TrainRequest. Add `team_id: str` field.

After training, store in `_trained_models[team_id]` (overwriting previous). Return `train_id` in response so the frontend can reference it.

- [ ] **Step 3: Run tests, commit**

```bash
python -m pytest tests/test_classifier_api.py -v
git commit -m "feat: return model+scaler from train_rf, store in memory after training"
```

---

### Task 2: Persist RF model artifact on classifier submit

**Files:**
- Modify: `backend/routers/classifier.py`
- Add: `joblib` to `requirements-web.txt`

- [ ] **Step 1: Add joblib to requirements**

Add `joblib>=1.3.0` to `requirements-web.txt`.

- [ ] **Step 2: Modify submit endpoint to persist model artifact**

After writing to the SQLite leaderboard, also persist the model artifact to disk:

```python
import joblib
from datetime import datetime, timezone

MODELS_DIR = os.path.join(DATA_ROOT, "rf_models")

# In submit_to_leaderboard(), after successful DB write:
trained = _trained_models.get(body.team_id)
if trained:
    team_dir = os.path.join(MODELS_DIR, body.team_id)
    os.makedirs(team_dir, exist_ok=True)
    artifact = {
        "model": trained["model"],
        "features": trained["features"],
        "scaler": trained["scaler"],
        "hyperparameters": trained["hyperparameters"],
        "metrics": {
            "f1": body.f1,
            "accuracy": body.accuracy,
            "precision": body.precision_val,
            "recall": body.recall,
        },
        "held_out_events": HELD_OUT_EVENTS,
        "class_labels": {0: "non-flood", 1: "flood"},
        "artifact_version": "rf_artifact_v1",
        "team_id": body.team_id,
        "team_name": body.team_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, os.path.join(team_dir, "model_artifact.pkl"))
    del _trained_models[body.team_id]  # cleanup memory
```

- [ ] **Step 3: Run tests, commit**

```bash
pip install joblib
python -m pytest tests/test_classifier_api.py -v
git commit -m "feat: persist RF model artifact to disk on classifier submit"
```

---

### Task 3: Create RF game engine service

**Files:**
- Create: `backend/services/rf_game_engine.py`
- Create: `tests/test_rf_game_engine.py`

This is the core new logic: run Flappy Bird episodes where each pipe = RF classification of a real sample.

- [ ] **Step 1: Create `backend/services/rf_game_engine.py`**

```python
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
    HELD_OUT_EVENTS, ALL_FEATURES,
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
        correct = (prediction == true_label)
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
        raise ValueError("MODEL_NOT_FOUND|No classifier submitted yet. Train and submit on the Classifier page first.")

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
        raise ValueError(f"ARTIFACT_INCOMPATIBLE|Features missing from test data: {missing}")

    stage_samples = _select_stage_samples(model, scaler, features, test_df, stage_id)
    if len(stage_samples) < 10:
        raise ValueError(f"NO_ELIGIBLE_SAMPLES|Not enough test samples for Stage {stage_id} (found {len(stage_samples)}).")

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
```

- [ ] **Step 2: Write tests**

```python
"""tests/test_rf_game_engine.py"""
from backend.services.rf_game_engine import (
    STAGES, STAGE_SEEDS, get_model_status, load_model_artifact,
)


def test_stages_defined():
    assert len(STAGES) == 5
    assert STAGES[1]["pass_avg"] == 8
    assert STAGES[5]["pass_avg"] is None


def test_stage_seeds_defined():
    assert len(STAGE_SEEDS) == 5
    assert len(STAGE_SEEDS[1]) == 10


def test_model_status_no_model():
    status = get_model_status("nonexistent_team_xyz")
    assert status["has_model"] is False


def test_load_artifact_missing():
    result = load_model_artifact("nonexistent_team_xyz")
    assert result is None
```

- [ ] **Step 3: Run tests, commit**

```bash
python -m pytest tests/test_rf_game_engine.py -v
git commit -m "feat: add RF-based Flappy Bird game engine"
```

---

### Task 4: Rewrite Flappy API router for RF-based gameplay

**Files:**
- Rewrite: `backend/routers/flappy.py`
- Modify: `backend/main.py` (if needed)
- Create: `tests/test_flappy_rf_api.py`

- [ ] **Step 1: Rewrite `backend/routers/flappy.py`**

Remove all DQN/upload code. New endpoints:

```python
"""Flappy Bird API — RF classifier-based gameplay."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.rf_game_engine import (
    STAGES, get_model_status, run_rf_game,
)
from backend.services.flappy_engine import (
    get_unlocked_stages, execute_race, FLAPPY_LB_PATH,
)
from backend.db import get_connection
from utils.flappy_leaderboard import add_entry as lb_add

router = APIRouter(prefix="/api/flappy", tags=["flappy"])


@router.get("/stages")
def stages():
    stage_data = {}
    for sid, s in STAGES.items():
        stage_data[sid] = {"label": s["label"], "pass_avg": s["pass_avg"]}
    return {"ok": True, "data": {"stages": stage_data}}


@router.get("/model-status/{team_id}")
def model_status(team_id: str):
    status = get_model_status(team_id)
    return {"ok": True, "data": status}


@router.get("/unlocked/{team_id}")
def unlocked(team_id: str):
    # Use team_id as team_name for unlock lookup
    return {"ok": True, "data": get_unlocked_stages(team_id)}


class PlayRequest(BaseModel):
    team_id: str
    stage_id: int
    mode: str = "practice"  # "practice" or "leaderboard"


@router.post("/play")
def play(body: PlayRequest):
    try:
        result = run_rf_game(body.team_id, body.stage_id, body.mode)
    except ValueError as e:
        parts = str(e).split("|", 1)
        code = parts[0] if len(parts) == 2 else "GAME_ERROR"
        msg = parts[1] if len(parts) == 2 else str(e)
        status = 404 if code == "MODEL_NOT_FOUND" else 400
        raise HTTPException(status_code=status, detail={
            "ok": False, "error": code, "message": msg,
        })
    return {"ok": True, "data": result}


class SaveResultRequest(BaseModel):
    team_id: str
    team_name: str
    stage_id: int
    avg_score: float
    max_score: int
    episode_scores: list[float]
    passed: bool


@router.post("/save-result")
def save_result(body: SaveResultRequest):
    try:
        lb_add(
            FLAPPY_LB_PATH,
            team_name=body.team_name,
            stage_id=body.stage_id,
            avg_score=body.avg_score,
            max_score=body.max_score,
            survival_steps_avg=0,
            episode_scores=body.episode_scores,
            passed=body.passed,
            race_id="",
            submission_timestamp="",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={
            "ok": False, "error": "LEADERBOARD_SAVE_FAILED", "message": str(exc),
        })
    return {"ok": True, "data": {"message": "Result saved to leaderboard"}}


class RaceBody(BaseModel):
    stage_id: int
    admin_password: str


@router.post("/race")
def race(body: RaceBody):
    # For RF-based race, we'd need all teams' models.
    # For now delegate to the existing race infrastructure if submissions exist,
    # otherwise return error.
    result, err = execute_race(body.stage_id, body.admin_password)
    if err:
        if "password" in err.lower():
            raise HTTPException(status_code=403, detail={
                "ok": False, "error": "INVALID_PASSWORD", "message": err,
            })
        raise HTTPException(status_code=400, detail={
            "ok": False, "error": "RACE_FAILED", "message": err,
        })
    return {"ok": True, "data": {
        "race_id": result["race_id"],
        "results": result["results"],
        "replay": result["replay"],
    }}
```

- [ ] **Step 2: Write tests — `tests/test_flappy_rf_api.py`**

```python
"""tests/test_flappy_rf_api.py"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_get_stages():
    resp = client.get("/api/flappy/stages")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "stages" in body["data"]
    assert "1" in body["data"]["stages"] or 1 in body["data"]["stages"]


def test_model_status_no_model():
    resp = client.get("/api/flappy/model-status/nonexistent_xyz")
    assert resp.status_code == 200
    assert resp.json()["data"]["has_model"] is False


def test_play_no_model():
    resp = client.post("/api/flappy/play", json={
        "team_id": "nonexistent_xyz", "stage_id": 1,
    })
    assert resp.status_code == 404
    assert "MODEL_NOT_FOUND" in resp.json().get("error", "")


def test_unlocked():
    resp = client.get("/api/flappy/unlocked/test_team")
    assert resp.status_code == 200
    assert 1 in resp.json()["data"]
```

- [ ] **Step 3: Remove old test file, run tests, commit**

```bash
rm -f tests/test_flappy_api.py
python -m pytest tests/test_flappy_rf_api.py -v
git commit -m "feat: rewrite flappy router for RF-based gameplay"
```

---

### Task 5: Rewrite Flappy frontend page

**Files:**
- Rewrite: `frontend/js/flappy.js`
- Rewrite: `frontend/pages/flappy.html`

- [ ] **Step 1: Rewrite `frontend/js/flappy.js`**

New `init_flappy()`:
1. Check model status via `GET /api/flappy/model-status/{team_id}`
2. If no model → show message directing to Classifier page
3. If model exists → show model info card (features, F1, accuracy)
4. Load stages via `GET /api/flappy/stages`
5. Load unlocked stages via `GET /api/flappy/unlocked/{team_id}`
6. Render stage badges
7. "Deploy My Classifier" button + stage selector
8. On deploy → `POST /api/flappy/play` → render Canvas replay
9. After replay → "Save to Leaderboard" button (leaderboard mode only)
10. Leaderboard polls every 5s

Keep the existing Canvas replay drawing code (bird, pipes, ground, colors, animation loop).

- [ ] **Step 2: Rewrite `frontend/pages/flappy.html`**

Remove: architecture selector, hyperparameters, file upload, DQN Submit
Add: Model status card, Deploy button, mode toggle (Practice/Leaderboard)
Keep: Canvas, stage badges, leaderboard, footer stats, admin race section

- [ ] **Step 3: Commit**

```bash
git add frontend/js/flappy.js frontend/pages/flappy.html
git commit -m "feat: rewrite flappy frontend for RF classifier deployment"
```

---

### Task 6: Integration test + verify

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 2: End-to-end manual test**

1. Open `#classifier` → train a model → submit to leaderboard
2. Open `#flappy` → verify model status shows (features, F1)
3. Select Stage 1 → click "Deploy My Classifier"
4. Verify: game runs, Canvas replay shows bird navigating pipes
5. Verify: score reflects how many samples the model correctly classified
6. Verify: leaderboard shows result after saving

- [ ] **Step 3: Final commit**

```bash
git add -A && git commit -m "fix: integration fixes for RF-Flappy flow"
```
