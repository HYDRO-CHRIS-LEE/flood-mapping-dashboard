"""
AI Flood Classifier API router — Random Forest training and submission.
"""

import json
import os
from datetime import datetime, timezone

import joblib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import ALL_EVENTS, DATA_ROOT
from backend.db import get_connection
from backend.services.rf_engine import (
    ALL_FEATURES,
    FEATURE_INFO,
    HELD_OUT_EVENTS,
    generate_hints,
    load_training_data,
    train_rf,
)

router = APIRouter(prefix="/api/classifier", tags=["classifier"])

# Temporary in-memory store for trained models (persisted on submit)
_trained_models: dict[str, dict] = {}
MODELS_DIR = os.path.join(DATA_ROOT, "rf_models")


# ── Feature metadata ──────────────────────────────────────────────

@router.get("/features")
def get_features():
    """Return available features, their metadata, and held-out events."""
    features = []
    for key, (icon, short, desc) in FEATURE_INFO.items():
        features.append({
            "key": key,
            "icon": icon,
            "short": short,
            "description": desc,
        })

    held_out = []
    for ev in HELD_OUT_EVENTS:
        meta = ALL_EVENTS.get(ev, {})
        held_out.append({
            "key": ev,
            "label": meta.get("label", ev),
            "year": meta.get("year"),
        })

    return {
        "ok": True,
        "data": {
            "features": features,
            "all_feature_keys": ALL_FEATURES,
            "held_out_events": held_out,
        },
    }


# ── Train ─────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    team_id: str = ""  # optional, used for model storage
    features: list[str]
    n_trees: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    class_weight: bool = False
    scaling: str = "none"
    balance: str = "none"
    sample_pct: int = 100
    outlier: str = "none"


@router.post("/train")
def train_classifier(body: TrainRequest):
    """Train a Random Forest classifier and return metrics."""
    if not body.features:
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": "NO_FEATURES",
                    "message": "At least one feature must be selected."},
        )

    # Validate feature names
    invalid = [f for f in body.features if f not in ALL_FEATURES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": "INVALID_FEATURES",
                    "message": f"Unknown features: {invalid}"},
        )

    # Discover events that have data
    available_events = []
    for key in ALL_EVENTS:
        d = os.path.join(DATA_ROOT, key)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "RF_training_samples.csv")):
            available_events.append(key)

    raw_df = load_training_data(available_events)
    if raw_df is None or len(raw_df) == 0:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "NO_TRAINING_DATA",
                    "message": "No RF_training_samples.csv files found."},
        )

    # Determine which held-out events actually have data
    available_held_out = [e for e in HELD_OUT_EVENTS if e in raw_df["event"].unique()]
    if not available_held_out:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "NO_TEST_DATA",
                    "message": "No held-out test events have training data."},
        )

    metrics, importance, model_obj, scaler_obj = train_rf(
        raw_df,
        body.features,
        body.n_trees,
        body.max_depth,
        available_held_out,
        min_samples_leaf=body.min_samples_leaf,
        max_features_str=body.max_features,
        use_class_weight=body.class_weight,
        bootstrap=body.bootstrap,
        scaling=body.scaling,
        balance=body.balance,
        sample_pct=body.sample_pct,
        outlier_method=body.outlier,
    )

    if metrics is None:
        raise HTTPException(
            status_code=422,
            detail={"ok": False, "error": "TRAIN_FAILED",
                    "message": "Training failed — not enough data after preprocessing."},
        )

    if body.team_id:
        _trained_models[body.team_id] = {
            "model": model_obj,
            "scaler": scaler_obj,
            "features": body.features,
            "hyperparameters": {
                "n_trees": body.n_trees,
                "max_depth": body.max_depth,
                "min_samples_leaf": body.min_samples_leaf,
                "max_features": body.max_features,
                "bootstrap": body.bootstrap,
                "class_weight": body.class_weight,
                "scaling": body.scaling,
                "balance": body.balance,
                "sample_pct": body.sample_pct,
                "outlier": body.outlier,
            },
        }

    hints = generate_hints(metrics, body.features, body.n_trees)

    return {
        "ok": True,
        "data": {
            "metrics": metrics,
            "importance": importance,
            "hints": hints,
        },
    }


# ── Submit to leaderboard ─────────────────────────────────────────

class SubmitRequest(BaseModel):
    team_id: str
    team_name: str
    f1: float
    accuracy: float
    precision_val: float
    recall: float
    features: list[str]
    n_trees: int
    max_depth: int


@router.post("/submit")
def submit_to_leaderboard(body: SubmitRequest):
    """Write a classifier result to the SQLite leaderboard."""
    conn = get_connection()
    try:
        # Check if team already has a better score
        row = conn.execute(
            "SELECT id, f1 FROM classifier_leaderboard WHERE team_id = ? ORDER BY f1 DESC LIMIT 1",
            (body.team_id,),
        ).fetchone()

        if row and row["f1"] is not None and row["f1"] >= body.f1:
            return {
                "ok": True,
                "data": {
                    "status": "kept_existing",
                    "message": f"Existing score ({row['f1']:.4f}) is >= submitted ({body.f1:.4f}).",
                },
            }

        # Delete old entry if exists, insert new
        if row:
            conn.execute("DELETE FROM classifier_leaderboard WHERE team_id = ?", (body.team_id,))

        conn.execute(
            """INSERT INTO classifier_leaderboard
               (team_id, team_name, f1, accuracy, precision_val, recall,
                features, n_trees, max_depth, test_events, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                body.team_id,
                body.team_name,
                body.f1,
                body.accuracy,
                body.precision_val,
                body.recall,
                json.dumps(body.features),
                body.n_trees,
                body.max_depth,
                json.dumps(HELD_OUT_EVENTS),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

        # Persist model artifact to disk
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
            _trained_models.pop(body.team_id, None)
    finally:
        conn.close()

    return {
        "ok": True,
        "data": {"status": "submitted", "f1": body.f1},
    }
