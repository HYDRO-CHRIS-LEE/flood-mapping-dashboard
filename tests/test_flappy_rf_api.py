"""Tests for the RF-based Flappy Bird API endpoints."""

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
    detail = resp.json()
    assert detail.get("error") == "MODEL_NOT_FOUND" or "MODEL_NOT_FOUND" in str(detail)


def test_unlocked():
    resp = client.get("/api/flappy/unlocked/test_team")
    assert resp.status_code == 200
    assert 1 in resp.json()["data"]
