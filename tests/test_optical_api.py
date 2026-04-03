"""
Tests for the Optical imagery API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_optical_bounds():
    """GET /api/tif/harvey/bounds returns ok=true with bounds, center, zoom."""
    resp = client.get("/api/tif/harvey/bounds")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    data = body["data"]
    assert "bounds" in data
    assert "center" in data
    assert "zoom" in data
    # bounds should be [[south,west],[north,east]]
    assert len(data["bounds"]) == 2
    assert len(data["bounds"][0]) == 2
    assert len(data["bounds"][1]) == 2
    assert isinstance(data["zoom"], int)


def test_optical_bounds_missing_event():
    """GET /api/tif/nonexistent/bounds returns 404."""
    resp = client.get("/api/tif/nonexistent/bounds")
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "EVENT_NOT_FOUND"


def test_optical_tile_returns_png():
    """GET /api/tif/harvey/NDWI/after returns image/png (or 404 if data missing)."""
    resp = client.get("/api/tif/harvey/NDWI/after")
    if resp.status_code == 404:
        body = resp.json()
        assert body["ok"] is False
        pytest.skip("Harvey NDWI after data not present on disk")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # PNG magic bytes
    assert resp.content[:4] == b"\x89PNG"


def test_optical_tile_invalid_layer():
    """GET /api/tif/harvey/BADLAYER/after returns 400."""
    resp = client.get("/api/tif/harvey/BADLAYER/after")
    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "INVALID_LAYER"
