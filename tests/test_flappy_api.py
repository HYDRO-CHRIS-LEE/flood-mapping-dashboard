"""
Tests for the Flappy Bird API endpoints.
"""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_get_stages():
    resp = client.get("/api/flappy/stages")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "stages" in body["data"]
    assert "architectures" in body["data"]
    # Check all 5 stages present (keys are strings in JSON)
    stages = body["data"]["stages"]
    assert len(stages) >= 5
    # Architectures is a sorted list
    archs = body["data"]["architectures"]
    assert isinstance(archs, list)
    assert "model1" in archs


def test_get_unlocked():
    resp = client.get("/api/flappy/unlocked/TestTeam")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    # Stage 1 is always unlocked
    assert 1 in body["data"]


def test_race_wrong_password():
    resp = client.post(
        "/api/flappy/race",
        json={"stage_id": 1, "admin_password": "wrong"},
    )
    assert resp.status_code == 403 or (
        resp.status_code == 200 and resp.json()["ok"] is False
    )
