"""
Tests for the Events API endpoints.
"""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_get_events_returns_list():
    """GET /api/events returns ok=true and data is a non-empty list."""
    resp = client.get("/api/events")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


def test_get_events_item_shape():
    """Each item in the events list has key, label, year, region, color."""
    resp = client.get("/api/events")
    body = resp.json()
    for item in body["data"]:
        assert "key" in item
        assert "label" in item
        assert "year" in item
        assert "region" in item
        assert "color" in item


def test_get_event_by_key():
    """GET /api/events/harvey returns the correct event data."""
    resp = client.get("/api/events/harvey")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["data"]["key"] == "harvey"
    assert body["data"]["label"] == "Hurricane Harvey"
    assert body["data"]["year"] == 2017


def test_get_event_not_found():
    """GET /api/events/nonexistent returns 404 with ok=false."""
    resp = client.get("/api/events/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "EVENT_NOT_FOUND"
