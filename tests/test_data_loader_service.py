"""Tests for backend.services.data_loader (ported from utils/data_loader.py)."""

import numpy as np
import pytest

from backend.services.data_loader import (
    EVENT_BOUNDS,
    EVENT_CENTERS,
    EVENT_ZOOM,
    load_csv,
    load_tif,
    norm_band,
    rgb_tif_to_rgba,
    rgba_to_png_bytes,
    tif_to_rgba,
)


# ── Dict sanity checks ─────────────────────────────────────────

def test_event_bounds_has_entries():
    assert len(EVENT_BOUNDS) >= 16
    assert "harvey" in EVENT_BOUNDS
    assert len(EVENT_BOUNDS["harvey"]) == 4


def test_event_centers_has_entries():
    assert len(EVENT_CENTERS) >= 16
    assert "harvey" in EVENT_CENTERS
    assert len(EVENT_CENTERS["harvey"]) == 2


def test_event_zoom_has_entries():
    assert len(EVENT_ZOOM) >= 16
    assert EVENT_ZOOM["harvey"] == 9


# ── norm_band ───────────────────────────────────────────────────

def test_norm_band_basic():
    arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    result = norm_band(arr)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_norm_band_nan_handling():
    arr = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    result = norm_band(arr)
    assert not np.any(np.isnan(result))


def test_norm_band_empty():
    arr = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    result = norm_band(arr)
    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))


# ── tif_to_rgba / rgb_tif_to_rgba ──────────────────────────────

def test_tif_to_rgba_shape():
    band = np.random.rand(10, 10).astype(np.float32)
    rgba = tif_to_rgba(band)
    assert rgba.shape == (10, 10, 4)
    assert rgba.dtype == np.uint8


def test_rgb_tif_to_rgba_shape():
    r = np.random.rand(10, 10).astype(np.float32)
    g = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(10, 10).astype(np.float32)
    rgba = rgb_tif_to_rgba(r, g, b)
    assert rgba.shape == (10, 10, 4)
    assert rgba.dtype == np.uint8


# ── rgba_to_png_bytes ──────────────────────────────────────────

def test_rgba_to_png_bytes():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    result = rgba_to_png_bytes(rgba)
    assert isinstance(result, bytes)
    # PNG magic bytes: \x89PNG\r\n\x1a\n
    assert result[:8] == b"\x89PNG\r\n\x1a\n"


# ── Missing-file fallbacks ─────────────────────────────────────

def test_load_csv_missing_returns_none():
    assert load_csv("__nonexistent_event__", "nothing") is None


def test_load_tif_missing_returns_none():
    data, meta = load_tif("__nonexistent_event__", "nothing")
    assert data is None
    assert meta == {}
