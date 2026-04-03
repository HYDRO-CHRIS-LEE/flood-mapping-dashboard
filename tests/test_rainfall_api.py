"""
Tests for the Rainfall API endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_rainfall_returns_data():
    """GET /api/rainfall/harvey returns ok=true with expected shape (or 404 if data missing)."""
    resp = client.get("/api/rainfall/harvey")
    if resp.status_code == 404:
        body = resp.json()
        assert body["ok"] is False
        pytest.skip("Harvey rainfall data not present on disk")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    data = body["data"]
    assert "dates" in data
    assert "precip" in data
    assert "flood_window" in data
    assert "fact" in data
    assert "peak_val" in data
    assert "peak_date" in data
    assert "total_mm" in data
    assert "period_days" in data
    assert "event" in data


def test_rainfall_missing_event():
    """GET /api/rainfall/nonexistent returns 404."""
    resp = client.get("/api/rainfall/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "EVENT_NOT_FOUND"


def test_rainfall_data_shape():
    """Verify dates and precip are lists of the same length."""
    resp = client.get("/api/rainfall/harvey")
    if resp.status_code == 404:
        pytest.skip("Harvey rainfall data not present on disk")
    body = resp.json()
    data = body["data"]
    assert isinstance(data["dates"], list)
    assert isinstance(data["precip"], list)
    assert len(data["dates"]) == len(data["precip"])
    assert len(data["dates"]) > 0
    assert data["period_days"] == len(data["dates"])
