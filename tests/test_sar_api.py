"""
Tests for the SAR flood detection API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_sar_compute():
    """POST /api/sar/compute with harvey — verify response shape."""
    resp = client.post("/api/sar/compute", json={
        "event": "harvey",
        "threshold": None,
        "remove_permanent": True,
    })
    if resp.status_code == 404:
        body = resp.json()
        assert body["ok"] is False
        pytest.skip("Harvey SAR data not present on disk")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    data = body["data"]
    assert "otsu" in data
    assert "threshold" in data
    assert "metrics" in data
    assert "histogram" in data
    assert "tile_url" in data
    assert "bounds" in data
    assert "center" in data
    assert "zoom" in data

    # Metrics shape
    metrics = data["metrics"]
    assert "flood_px" in metrics
    assert "flood_pct" in metrics
    assert "flood_km2" in metrics
    assert "total_valid" in metrics

    # Histogram shape
    histogram = data["histogram"]
    assert "centers" in histogram
    assert "counts" in histogram
    assert len(histogram["centers"]) == len(histogram["counts"])


def test_sar_compute_missing_event():
    """POST /api/sar/compute with nonexistent event returns 404."""
    resp = client.post("/api/sar/compute", json={
        "event": "nonexistent_event",
    })
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "EVENT_NOT_FOUND"


def test_sar_tile_returns_png():
    """GET /api/sar/tile/harvey/-16.0/1 returns PNG image bytes."""
    resp = client.get("/api/sar/tile/harvey/-16.0/1")
    if resp.status_code == 404:
        body = resp.json()
        assert body["ok"] is False
        pytest.skip("Harvey SAR data not present on disk")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # PNG magic bytes
    assert resp.content[:4] == b"\x89PNG"
