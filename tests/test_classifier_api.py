"""
Tests for the AI Classifier and Leaderboard API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.db import init_db
from backend.main import app

# Ensure DB tables exist (lifespan may not fire in TestClient without context manager)
init_db()

client = TestClient(app)


def test_get_features():
    """GET /api/classifier/features returns feature list and held-out events."""
    resp = client.get("/api/classifier/features")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    data = body["data"]

    assert "features" in data
    assert "all_feature_keys" in data
    assert "held_out_events" in data

    assert len(data["features"]) > 0
    assert len(data["all_feature_keys"]) > 0
    assert len(data["held_out_events"]) > 0

    # Each feature should have key, icon, short, description
    feat = data["features"][0]
    assert "key" in feat
    assert "icon" in feat
    assert "short" in feat
    assert "description" in feat

    # held_out entries should have key, label, year
    ho = data["held_out_events"][0]
    assert "key" in ho
    assert "label" in ho


def test_train_classifier():
    """POST /api/classifier/train with 2 features — verify metrics or 404 if no data."""
    resp = client.post("/api/classifier/train", json={
        "features": ["NDWI", "elevation"],
        "n_trees": 10,
        "max_depth": 3,
    })
    if resp.status_code == 404:
        body = resp.json()
        assert body["ok"] is False
        pytest.skip("No RF training data available on disk")

    if resp.status_code == 422:
        body = resp.json()
        assert body["ok"] is False
        pytest.skip("Training failed — not enough data after preprocessing")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    data = body["data"]

    assert "metrics" in data
    assert "importance" in data
    assert "hints" in data

    metrics = data["metrics"]
    assert "f1" in metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "n_train" in metrics
    assert "n_test" in metrics
    assert "cm" in metrics

    # Importance should have one entry per feature
    assert len(data["importance"]) == 2


def test_train_no_features():
    """POST /api/classifier/train with empty features returns 400."""
    resp = client.post("/api/classifier/train", json={
        "features": [],
        "n_trees": 10,
        "max_depth": 3,
    })
    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "NO_FEATURES"


def test_classifier_leaderboard():
    """GET /api/leaderboard/classifier returns a list."""
    resp = client.get("/api/leaderboard/classifier")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["data"], list)


def test_flappy_leaderboard():
    """GET /api/leaderboard/flappy/1 returns a list."""
    resp = client.get("/api/leaderboard/flappy/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["data"], list)
