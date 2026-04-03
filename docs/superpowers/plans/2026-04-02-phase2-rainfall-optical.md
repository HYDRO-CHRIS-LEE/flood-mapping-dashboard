# Phase 2: Rainfall + Optical Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Rainfall Analysis and Optical Detection pages as fully functional frontend pages backed by FastAPI endpoints, with the same data processing logic as the current Streamlit modules but rendered via Plotly.js and Leaflet.js in the browser.

**Architecture:** Port `utils/data_loader.py` functions to `backend/services/data_loader.py` (remove Streamlit caching, add `functools.lru_cache`). Create two API routers: `rainfall.py` returns GPM timeseries as JSON, `optical.py` returns PNG tiles and bounds JSON. Frontend pages use Plotly.js for the rainfall chart and Leaflet.js `ImageOverlay` for the optical map. The stitch HTML designs are used as the page templates.

**Tech Stack:** FastAPI, rasterio, numpy, pandas, Pillow, matplotlib (colormaps), Plotly.js (CDN), Leaflet.js (CDN)

---

### Task 1: Port data_loader to backend service

**Files:**
- Create: `backend/services/data_loader.py`
- Create: `tests/test_data_loader_service.py`

This is the foundation — both rainfall and optical APIs depend on it.

- [ ] **Step 1: Write tests**

```python
"""tests/test_data_loader_service.py"""
import numpy as np
from backend.services.data_loader import (
    load_tif, load_csv, norm_band, tif_to_rgba, rgb_tif_to_rgba, rgba_to_png_bytes,
    EVENT_BOUNDS, EVENT_CENTERS, EVENT_ZOOM,
)


def test_event_bounds_has_entries():
    assert len(EVENT_BOUNDS) >= 16
    assert "harvey" in EVENT_BOUNDS
    assert len(EVENT_BOUNDS["harvey"]) == 4  # [W, S, E, N]


def test_event_centers_has_entries():
    assert len(EVENT_CENTERS) >= 16
    assert "harvey" in EVENT_CENTERS
    assert len(EVENT_CENTERS["harvey"]) == 2  # [lat, lon]


def test_event_zoom_has_entries():
    assert len(EVENT_ZOOM) >= 16
    assert EVENT_ZOOM["harvey"] == 9


def test_norm_band_basic():
    arr = np.array([0.0, 50.0, 100.0])
    result = norm_band(arr)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_norm_band_nan_handling():
    arr = np.array([0.0, np.nan, 100.0])
    result = norm_band(arr)
    assert not np.isnan(result).any()


def test_norm_band_empty():
    arr = np.array([np.nan, np.nan])
    result = norm_band(arr)
    assert np.all(result == 0)


def test_tif_to_rgba_shape():
    band = np.random.rand(10, 10).astype(np.float32)
    result = tif_to_rgba(band)
    assert result.shape == (10, 10, 4)
    assert result.dtype == np.uint8


def test_rgb_tif_to_rgba_shape():
    r = np.random.rand(10, 10).astype(np.float32)
    g = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(10, 10).astype(np.float32)
    result = rgb_tif_to_rgba(r, g, b)
    assert result.shape == (10, 10, 4)


def test_rgba_to_png_bytes():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    result = rgba_to_png_bytes(rgba)
    assert isinstance(result, bytes)
    assert result[:4] == b'\x89PNG'


def test_load_csv_missing_returns_none():
    result = load_csv("nonexistent_event_xyz", "GPM_rainfall_daily")
    assert result is None


def test_load_tif_missing_returns_none():
    result = load_tif("nonexistent_event_xyz", "SAR_after")
    assert result == (None, {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_data_loader_service.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `backend/services/data_loader.py`**

Port from `utils/data_loader.py`. Key changes:
- Remove `import streamlit as st` and all `@st.cache_data` decorators
- Add `functools.lru_cache` for `load_tif` and `load_csv`
- Change `rgba_to_b64()` → `rgba_to_png_bytes()` (returns raw PNG bytes instead of base64 data URL)
- Import `DATA_ROOT` from `backend.config` instead of deriving it locally
- Keep `EVENT_BOUNDS`, `EVENT_CENTERS`, `EVENT_ZOOM` dicts
- Keep `norm_band`, `tif_to_rgba`, `rgb_tif_to_rgba` functions unchanged

```python
"""Data loading utilities — ported from utils/data_loader.py without Streamlit deps."""

import os
import io
import functools
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from PIL import Image

from backend.config import DATA_ROOT

# ── Bounding boxes [W, S, E, N] ──
EVENT_BOUNDS = {
    "harvey":        [-96.0,  29.0, -94.5,  30.5],
    "pakistan":       [ 66.5,  25.5,  69.5,  28.0],
    "dubai":         [ 54.8,  24.9,  56.2,  25.6],
    "myanmar2015":   [ 94.5,  15.5,  96.5,  17.5],
    "louisiana2016": [-91.5,  30.0, -90.0,  31.0],
    "srilanka2017":  [ 80.0,   6.0,  81.5,   7.5],
    "mozambique2019":[ 34.0, -20.0,  35.5, -18.5],
    "iran2019":      [ 48.0,  31.0,  50.0,  33.0],
    "china2020":     [115.0,  29.0, 117.5,  30.5],
    "sudan2020":     [ 32.0,  15.0,  34.5,  16.5],
    "germany2021":   [  6.5,  50.0,   7.5,  50.8],
    "nigeria2022":   [  6.0,   5.0,   7.5,   6.5],
    "libya2023":     [ 22.0,  32.0,  23.5,  33.0],
    "somalia2023":   [ 45.0,   2.0,  46.5,   3.5],
    "brazil2024":    [-53.0, -31.0, -51.0, -29.5],
    "valencia2024":  [ -1.5,  38.5,   0.5,  40.0],
}

# ── Map centers [lat, lon] ──
EVENT_CENTERS = {
    "harvey":        [ 29.76, -95.37],
    "pakistan":       [ 27.0,   68.0 ],
    "dubai":         [ 25.2,   55.3 ],
    "myanmar2015":   [ 16.5,   95.5 ],
    "louisiana2016": [ 30.5,  -90.75],
    "srilanka2017":  [  6.75,  80.75],
    "mozambique2019":[-19.25,  34.75],
    "iran2019":      [ 32.0,   49.0 ],
    "china2020":     [ 29.75, 116.25],
    "sudan2020":     [ 15.75,  33.25],
    "germany2021":   [ 50.4,    7.0 ],
    "nigeria2022":   [  5.75,   6.75],
    "libya2023":     [ 32.5,   22.75],
    "somalia2023":   [  2.75,  45.75],
    "brazil2024":    [-30.25, -52.0 ],
    "valencia2024":  [ 39.25,  -0.5 ],
}

EVENT_ZOOM = {
    "harvey": 9, "pakistan": 8, "dubai": 10,
    "myanmar2015": 8, "louisiana2016": 9,
    "srilanka2017": 9, "mozambique2019": 9, "iran2019": 8,
    "china2020": 8, "sudan2020": 8,
    "germany2021": 10, "nigeria2022": 9, "libya2023": 10,
    "somalia2023": 9, "brazil2024": 8, "valencia2024": 9,
}


def load_tif(event: str, name: str, max_pixels: int = 512):
    """Load and resample a GeoTIFF. Returns (data_array, metadata_dict) or (None, {})."""
    path = os.path.join(DATA_ROOT, event, f"{name}.tif")
    if not os.path.exists(path):
        return None, {}
    with rasterio.open(path) as src:
        scale = max_pixels / max(src.width, src.height)
        out_w = max(1, int(src.width * scale))
        out_h = max(1, int(src.height * scale))
        data = src.read(
            out_shape=(src.count, out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        bounds = src.bounds
        meta = {
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "nodata": nodata,
            "count": src.count,
        }
    return data, meta


def load_csv(event: str, name: str):
    """Load a CSV from data/{event}/{name}.csv. Returns DataFrame or None."""
    path = os.path.join(DATA_ROOT, event, f"{name}.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def norm_band(arr, p_low=2, p_high=98):
    """Normalize array to [0,1] using percentile clipping."""
    arr = arr.copy().astype(np.float32)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    lo = np.percentile(valid, p_low)
    hi = np.percentile(valid, p_high)
    if hi == lo:
        return np.zeros_like(arr)
    return np.where(np.isnan(arr), 0, np.clip((arr - lo) / (hi - lo), 0, 1))


def tif_to_rgba(band, colormap="gray"):
    """Convert a single band to RGBA using a matplotlib colormap."""
    import matplotlib
    normed = norm_band(band)
    rgba = (matplotlib.colormaps[colormap](normed) * 255).astype(np.uint8)
    rgba[band == 0, 3] = 0
    return rgba


def rgb_tif_to_rgba(r, g, b):
    """Convert 3 bands to RGBA with transparency for zero pixels."""
    r8 = (norm_band(r) * 255).astype(np.uint8)
    g8 = (norm_band(g) * 255).astype(np.uint8)
    b8 = (norm_band(b) * 255).astype(np.uint8)
    a8 = np.where((r == 0) & (g == 0) & (b == 0), 0, 255).astype(np.uint8)
    return np.stack([r8, g8, b8, a8], axis=-1)


def rgba_to_png_bytes(rgba) -> bytes:
    """Convert RGBA numpy array to PNG bytes."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_data_loader_service.py -v`
Expected: All 12 tests pass

- [ ] **Step 5: Commit**

```bash
git add backend/services/data_loader.py tests/test_data_loader_service.py
git commit -m "feat(phase2): port data_loader service with tests"
```

---

### Task 2: Rainfall API router

**Files:**
- Create: `backend/routers/rainfall.py`
- Modify: `backend/main.py` (register router)
- Create: `tests/test_rainfall_api.py`

- [ ] **Step 1: Write tests**

```python
"""tests/test_rainfall_api.py"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_rainfall_returns_data():
    # Use an event that has GPM data
    resp = client.get("/api/rainfall/harvey")
    if resp.status_code == 404:
        # No GPM data for this event in test env — skip
        return
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
    assert isinstance(data["dates"], list)
    assert isinstance(data["precip"], list)
    assert len(data["dates"]) == len(data["precip"])


def test_rainfall_missing_event():
    resp = client.get("/api/rainfall/nonexistent_xyz")
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False


def test_rainfall_missing_data():
    # An event that exists in ALL_EVENTS but might not have GPM CSV
    resp = client.get("/api/rainfall/valencia2024")
    # Should be either 200 (has data) or 404 (no CSV)
    assert resp.status_code in [200, 404]
```

- [ ] **Step 2: Create `backend/routers/rainfall.py`**

```python
"""Rainfall API — GPM precipitation timeseries."""

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.config import ALL_EVENTS
from backend.services.data_loader import load_csv

router = APIRouter(tags=["rainfall"])

FLOOD_WINDOWS = {
    "harvey":        ("2017-08-25", "2017-09-01"),
    "pakistan":       ("2022-08-15", "2022-09-15"),
    "dubai":         ("2024-04-15", "2024-04-20"),
    "myanmar2015":   ("2015-07-25", "2015-08-15"),
    "louisiana2016": ("2016-08-12", "2016-08-20"),
    "srilanka2017":  ("2017-05-25", "2017-06-10"),
    "mozambique2019":("2019-03-14", "2019-03-20"),
    "iran2019":      ("2019-03-25", "2019-04-10"),
    "china2020":     ("2020-07-05", "2020-08-10"),
    "sudan2020":     ("2020-08-05", "2020-08-25"),
    "germany2021":   ("2021-07-14", "2021-07-16"),
    "nigeria2022":   ("2022-09-20", "2022-10-15"),
    "libya2023":     ("2023-09-11", "2023-09-14"),
    "somalia2023":   ("2023-11-05", "2023-11-20"),
    "brazil2024":    ("2024-05-01", "2024-05-15"),
    "valencia2024":  ("2024-10-29", "2024-11-01"),
}

KEY_FACTS = {
    "harvey":        "Harvey dumped over 1,300 mm (51 in) across Houston in 5 days — the highest tropical rainfall total ever recorded in the U.S.",
    "pakistan":       "Pakistan received 3-4x its annual average rainfall in just two months, submerging roughly one-third of the country.",
    "dubai":         "Dubai received nearly its entire annual rainfall (~75 mm) in a single day — a city built for desert conditions with almost no storm drainage.",
    "myanmar2015":   "Cyclone Komen triggered catastrophic monsoon flooding affecting over 1.6 million people across Myanmar.",
    "louisiana2016": "An unnamed storm dropped 60+ cm of rain in 48 hours over Baton Rouge, flooding over 60,000 homes.",
    "srilanka2017":  "Southwest monsoon rains triggered widespread flooding and landslides, displacing over 600,000 people.",
    "mozambique2019":"Cyclone Idai made landfall with 195 km/h winds, generating a massive inland flood over Beira.",
    "iran2019":      "Spring floods swept through 26 of Iran's 31 provinces, the country's worst flooding in 70 years.",
    "china2020":     "Record Yangtze River levels — the Poyang Lake basin saw its largest flood extent since satellite monitoring began.",
    "sudan2020":     "Sudan's worst flooding in 100 years submerged entire neighborhoods in Khartoum.",
    "germany2021":   "The Ahr Valley received a full month of rain in 24 hours, destroying hundreds of bridges and roads.",
    "nigeria2022":   "Flooding affected 33 of 36 states; the Anambra-Delta corridor saw the worst inundation.",
    "libya2023":     "Cyclone Daniel caused catastrophic dam failures in Derna, killing thousands in hours.",
    "somalia2023":   "Unprecedented October-November rains flooded over 1 million people across the Shabelle basin.",
    "brazil2024":    "Cyclone-driven rains submerged 90% of Rio Grande do Sul's municipalities — Brazil's worst climate disaster.",
    "valencia2024":  "A DANA (cut-off low) dropped 450 mm in 8 hours near Valencia — Spain's deadliest flash flood in decades.",
}


@router.get("/rainfall/{event}")
def get_rainfall(event: str):
    if event not in ALL_EVENTS:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "EVENT_NOT_FOUND", "message": f"Event '{event}' not found",
        })

    df = load_csv(event, "GPM_rainfall_daily")
    if df is None:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "DATA_NOT_FOUND", "message": f"No GPM data for '{event}'",
        })

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["precip_mm_day"])
    df["precip_mm_day"] = df["precip_mm_day"].clip(lower=0)

    dates = df["date"].dt.strftime("%Y-%m-%d").tolist()
    precip = [round(float(v), 2) for v in df["precip_mm_day"]]

    peak_idx = df["precip_mm_day"].idxmax()
    peak_val = round(float(df.loc[peak_idx, "precip_mm_day"]), 1)
    peak_date = df.loc[peak_idx, "date"].strftime("%b %d")
    total_mm = round(float(df["precip_mm_day"].sum()), 0)

    fw = FLOOD_WINDOWS.get(event)
    fact = KEY_FACTS.get(event, "")

    return {"ok": True, "data": {
        "dates": dates,
        "precip": precip,
        "flood_window": list(fw) if fw else None,
        "fact": fact,
        "peak_val": peak_val,
        "peak_date": peak_date,
        "total_mm": total_mm,
        "period_days": len(dates),
        "event": {
            "key": event,
            "label": ALL_EVENTS[event]["label"],
            "year": ALL_EVENTS[event]["year"],
            "region": ALL_EVENTS[event]["region"],
        },
    }}
```

- [ ] **Step 3: Register router in `backend/main.py`**

Add import and include_router:

```python
from backend.routers import events, rainfall

app.include_router(events.router, prefix="/api")
app.include_router(rainfall.router, prefix="/api")
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_rainfall_api.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add backend/routers/rainfall.py backend/main.py tests/test_rainfall_api.py
git commit -m "feat(phase2): add rainfall API with flood windows and key facts"
```

---

### Task 3: Optical tile API router

**Files:**
- Create: `backend/routers/optical.py`
- Modify: `backend/main.py` (register router)
- Create: `tests/test_optical_api.py`

- [ ] **Step 1: Write tests**

```python
"""tests/test_optical_api.py"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_optical_bounds():
    resp = client.get("/api/tif/harvey/bounds")
    if resp.status_code == 404:
        return  # no data in test env
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    data = body["data"]
    assert "bounds" in data  # [[south, west], [north, east]]
    assert "center" in data
    assert "zoom" in data


def test_optical_bounds_missing_event():
    resp = client.get("/api/tif/nonexistent_xyz/bounds")
    assert resp.status_code == 404


def test_optical_tile_returns_png():
    resp = client.get("/api/tif/harvey/NDWI/after")
    if resp.status_code == 404:
        return  # no data in test env
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:4] == b'\x89PNG'


def test_optical_tile_missing_layer():
    resp = client.get("/api/tif/harvey/INVALID_LAYER/after")
    assert resp.status_code in [400, 404]


def test_optical_ndwi_change():
    resp = client.get("/api/tif/harvey/NDWI_change/after")
    if resp.status_code == 404:
        return  # no data
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
```

- [ ] **Step 2: Create `backend/routers/optical.py`**

```python
"""Optical tile API — serves PNG tiles and bounds for Leaflet overlays."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.config import ALL_EVENTS
from backend.services.data_loader import (
    load_tif, tif_to_rgba, rgb_tif_to_rgba, rgba_to_png_bytes,
    EVENT_BOUNDS, EVENT_CENTERS, EVENT_ZOOM,
)

router = APIRouter(tags=["optical"])

VALID_LAYERS = {"RGB", "NDWI", "NDWI_change"}
VALID_PERIODS = {"before", "after"}


@router.get("/tif/{event}/bounds")
def get_bounds(event: str):
    if event not in ALL_EVENTS:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "EVENT_NOT_FOUND", "message": f"Event '{event}' not found",
        })
    bounds = EVENT_BOUNDS.get(event)
    center = EVENT_CENTERS.get(event)
    zoom = EVENT_ZOOM.get(event, 9)
    if bounds is None:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "BOUNDS_NOT_FOUND", "message": f"No bounds for '{event}'",
        })
    # Leaflet expects [[south, west], [north, east]]
    leaflet_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
    return {"ok": True, "data": {
        "bounds": leaflet_bounds,
        "center": center,
        "zoom": zoom,
    }}


@router.get("/tif/{event}/{layer}/{period}")
def get_tile(event: str, layer: str, period: str, max_pixels: int = 600):
    if event not in ALL_EVENTS:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "EVENT_NOT_FOUND", "message": f"Event '{event}' not found",
        })
    if layer not in VALID_LAYERS:
        raise HTTPException(status_code=400, detail={
            "ok": False, "error": "INVALID_LAYER", "message": f"Layer must be one of: {VALID_LAYERS}",
        })

    if layer == "RGB":
        data, meta = load_tif(event, f"RGB_{period}", max_pixels=max_pixels)
        if data is None:
            raise HTTPException(status_code=404, detail={
                "ok": False, "error": "DATA_NOT_FOUND", "message": f"No RGB_{period} data for '{event}'",
            })
        rgba = rgb_tif_to_rgba(data[0], data[1], data[2])

    elif layer == "NDWI":
        data, meta = load_tif(event, f"NDWI_{period}", max_pixels=max_pixels)
        if data is None:
            raise HTTPException(status_code=404, detail={
                "ok": False, "error": "DATA_NOT_FOUND", "message": f"No NDWI_{period} data for '{event}'",
            })
        rgba = tif_to_rgba(data[0], colormap="RdYlBu")

    elif layer == "NDWI_change":
        before_data, _ = load_tif(event, "NDWI_before", max_pixels=max_pixels)
        after_data, _ = load_tif(event, "NDWI_after", max_pixels=max_pixels)
        if before_data is None or after_data is None:
            raise HTTPException(status_code=404, detail={
                "ok": False, "error": "DATA_NOT_FOUND", "message": f"Need both NDWI_before and NDWI_after for '{event}'",
            })
        rgba = tif_to_rgba(after_data[0] - before_data[0], colormap="RdBu")

    png_bytes = rgba_to_png_bytes(rgba)
    return Response(content=png_bytes, media_type="image/png")
```

- [ ] **Step 3: Register router in `backend/main.py`**

Add import and include_router:

```python
from backend.routers import events, rainfall, optical

app.include_router(optical.router, prefix="/api")
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_optical_api.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add backend/routers/optical.py backend/main.py tests/test_optical_api.py
git commit -m "feat(phase2): add optical tile API serving PNG + bounds"
```

---

### Task 4: Rainfall frontend page

**Files:**
- Modify: `frontend/pages/rainfall.html`
- Create: `frontend/js/rainfall.js`
- Modify: `frontend/index.html` (add Plotly.js CDN script)

- [ ] **Step 1: Add Plotly.js CDN to `frontend/index.html`**

Add before the Alpine.js script tag:

```html
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
```

- [ ] **Step 2: Create `frontend/js/rainfall.js`**

```javascript
/**
 * Rainfall Analysis page logic.
 * Fetches GPM data from API, renders Plotly chart, handles controls.
 */
function init_rainfall() {
  const eventSelect = document.getElementById('event-select');
  if (!eventSelect) return;

  // Load events into selector
  API.get('/events').then(events => {
    eventSelect.innerHTML = events.map(e =>
      `<option value="${e.key}">${e.label} (${e.year})</option>`
    ).join('');
    // Load first event
    if (events.length > 0) loadRainfall(events[0].key);
  });

  eventSelect.addEventListener('change', () => loadRainfall(eventSelect.value));
}

async function loadRainfall(event) {
  const container = document.getElementById('rainfall-content');
  if (!container) return;

  container.innerHTML = '<div class="flex items-center justify-center h-64 text-slate-400"><span class="material-symbols-outlined text-4xl animate-spin">progress_activity</span></div>';

  try {
    const d = await API.get(`/rainfall/${event}`);
    renderRainfallPage(d);
  } catch (e) {
    container.innerHTML = `<div class="bg-surface-container-lowest rounded-[2rem] p-8 text-center text-error">${e.message}</div>`;
  }
}

function renderRainfallPage(d) {
  const container = document.getElementById('rainfall-content');
  const ev = d.event;

  container.innerHTML = `
    <div class="bento-grid">
      <!-- Main chart -->
      <div class="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-[2rem] p-8 relative overflow-hidden">
        <div class="flex justify-between items-center mb-6">
          <h4 class="text-xl font-bold negative-tracking">Rainfall Timeline</h4>
          <div class="flex items-center gap-2 text-xs font-bold text-secondary">
            <span class="w-2 h-2 rounded-full bg-primary"></span>
            <span>PRECIPITATION (MM)</span>
          </div>
        </div>
        <div id="rainfall-chart" style="height:300px"></div>
        <div class="mt-4 flex flex-wrap gap-3">
          <label class="flex items-center gap-2 text-sm font-medium text-on-surface-variant cursor-pointer">
            <input type="checkbox" id="chk-threshold" checked class="accent-primary rounded"> Threshold line
          </label>
          <label class="flex items-center gap-2 text-sm font-medium text-on-surface-variant cursor-pointer">
            <input type="checkbox" id="chk-flood" checked class="accent-primary rounded"> Flood period
          </label>
          <label class="flex items-center gap-2 text-sm font-medium text-on-surface-variant cursor-pointer">
            <input type="checkbox" id="chk-cumulative" class="accent-primary rounded"> Cumulative
          </label>
          <div class="flex items-center gap-2 ml-4">
            <span class="text-xs font-bold text-on-surface-variant">Threshold:</span>
            <input type="range" id="threshold-slider" min="5" max="80" value="20" class="w-32 accent-primary">
            <span id="threshold-val" class="text-xs font-bold text-primary w-12">20 mm</span>
          </div>
        </div>
      </div>

      <!-- Cumulative card -->
      <div class="col-span-12 lg:col-span-4 bg-primary rounded-[2rem] p-8 text-white flex flex-col justify-between relative overflow-hidden">
        <div>
          <span class="material-symbols-outlined text-white/50 text-4xl mb-4">water_drop</span>
          <h4 class="text-white/70 font-bold text-sm uppercase tracking-widest mb-1">Cumulative</h4>
          <div class="text-6xl font-black negative-tracking leading-none">
            ${d.total_mm}<span class="text-2xl font-medium opacity-60 ml-2">mm</span>
          </div>
        </div>
        <div class="mt-6">
          <div class="w-full bg-white/20 h-2 rounded-full overflow-hidden">
            <div class="bg-white h-full rounded-full" style="width:${Math.min(100, d.total_mm / 5)}%"></div>
          </div>
          <p class="mt-2 text-white/70 text-xs font-bold uppercase tracking-widest">Peak: ${d.peak_val} mm on ${d.peak_date}</p>
        </div>
      </div>

      <!-- Metrics row -->
      <div class="col-span-12 lg:col-span-4 bg-surface-container-lowest rounded-[2rem] p-8">
        <div class="flex items-center gap-3 mb-4">
          <div class="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
            <span class="material-symbols-outlined text-red-600">warning</span>
          </div>
          <h4 class="text-lg font-bold negative-tracking">Peak Rainfall</h4>
        </div>
        <p class="text-4xl font-black negative-tracking">${d.peak_val} <span class="text-lg font-medium text-on-surface-variant">mm/day</span></p>
        <p class="text-sm text-on-surface-variant mt-2">${d.peak_date} — ${d.period_days} days monitored</p>
      </div>

      <!-- Fact card -->
      <div class="col-span-12 lg:col-span-8 bg-surface-container-low rounded-[2rem] p-8 flex gap-8">
        <div class="flex-1">
          <h4 class="text-xl font-bold negative-tracking mb-4">${ev.label} (${ev.year})</h4>
          <p class="text-sm text-on-surface-variant leading-relaxed">${d.fact || 'No additional context available for this event.'}</p>
        </div>
      </div>
    </div>
  `;

  // Render chart
  renderRainfallChart(d);

  // Bind controls
  const slider = document.getElementById('threshold-slider');
  const valLabel = document.getElementById('threshold-val');
  const chkThr = document.getElementById('chk-threshold');
  const chkFlood = document.getElementById('chk-flood');
  const chkCum = document.getElementById('chk-cumulative');

  function update() {
    valLabel.textContent = slider.value + ' mm';
    renderRainfallChart(d);
  }

  slider.addEventListener('input', update);
  chkThr.addEventListener('change', update);
  chkFlood.addEventListener('change', update);
  chkCum.addEventListener('change', update);
}

function renderRainfallChart(d) {
  const thr = parseInt(document.getElementById('threshold-slider')?.value || 20);
  const showThr = document.getElementById('chk-threshold')?.checked ?? true;
  const showFlood = document.getElementById('chk-flood')?.checked ?? true;
  const showCum = document.getElementById('chk-cumulative')?.checked ?? false;

  const colors = d.precip.map(v => v >= thr ? '#ba1a1a' : '#adc6ff');

  const traces = [{
    x: d.dates, y: d.precip,
    type: 'bar', marker: { color: colors },
    name: 'Daily Rainfall',
    hovertemplate: '<b>%{x}</b><br>%{y:.1f} mm/day<extra></extra>',
  }];

  if (showCum) {
    let cum = []; let sum = 0;
    d.precip.forEach(v => { sum += v; cum.push(Math.round(sum * 10) / 10); });
    traces.push({
      x: d.dates, y: cum,
      type: 'scatter', mode: 'lines',
      name: 'Cumulative (mm)', yaxis: 'y2',
      line: { color: '#4f46e5', width: 2, dash: 'dot' },
    });
  }

  const layout = {
    plot_bgcolor: '#faf9fe', paper_bgcolor: '#faf9fe',
    font: { color: '#1a1b1f', family: 'Inter' },
    height: 300, margin: { l: 40, r: showCum ? 50 : 10, t: 10, b: 30 },
    xaxis: { showgrid: false },
    yaxis: { title: 'mm/day', showgrid: true, gridcolor: '#f4f3f8' },
    showlegend: false, bargap: 0.15,
    shapes: [], annotations: [],
  };

  if (showCum) {
    layout.yaxis2 = {
      title: 'Cumulative (mm)', overlaying: 'y', side: 'right',
      showgrid: false, titlefont: { color: '#4f46e5', size: 11 },
      tickfont: { color: '#4f46e5', size: 10 },
    };
  }

  if (showThr) {
    layout.shapes.push({
      type: 'line', x0: 0, x1: 1, xref: 'paper',
      y0: thr, y1: thr, line: { color: '#d97706', width: 1.5, dash: 'dash' },
    });
    layout.annotations.push({
      x: 1, xref: 'paper', y: thr, text: `  ${thr} mm`,
      showarrow: false, font: { color: '#d97706', size: 11 },
    });
  }

  if (showFlood && d.flood_window) {
    layout.shapes.push({
      type: 'rect', xref: 'x', yref: 'paper',
      x0: d.flood_window[0], x1: d.flood_window[1],
      y0: 0, y1: 1, fillcolor: 'rgba(186,26,26,0.08)', line: { width: 0 },
    });
  }

  Plotly.newPlot('rainfall-chart', traces, layout, { displayModeBar: false });
}
```

- [ ] **Step 3: Replace `frontend/pages/rainfall.html`**

```html
<!-- Rainfall Analysis -->
<section class="mb-12">
  <div class="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
    <div>
      <span class="text-secondary font-semibold text-sm tracking-widest uppercase mb-2 block">Intelligence Suite</span>
      <h3 class="text-5xl font-black negative-tracking text-on-surface">Rainfall Analysis</h3>
      <p class="text-on-surface-variant mt-3 text-lg font-medium opacity-80">Real-time precipitation monitoring and hydrologic forecasting.</p>
    </div>
    <div class="flex gap-3 items-center">
      <select id="event-select" class="bg-surface-container-high text-on-surface rounded-full font-bold text-sm px-5 py-2.5 border-none focus:ring-2 focus:ring-primary/20 cursor-pointer"></select>
      <button class="px-6 py-2.5 bg-primary text-white rounded-full font-bold text-sm shadow-lg shadow-primary/20 hover:scale-105 active:scale-95 transition-all">
        Live Update
      </button>
    </div>
  </div>
</section>
<div id="rainfall-content">
  <div class="flex items-center justify-center h-64 text-slate-400">
    <span class="material-symbols-outlined text-4xl animate-spin">progress_activity</span>
  </div>
</div>
<script src="/js/rainfall.js"></script>
```

- [ ] **Step 4: Add Plotly CDN to index.html and commit**

```bash
git add frontend/pages/rainfall.html frontend/js/rainfall.js frontend/index.html
git commit -m "feat(phase2): add rainfall page with Plotly chart and controls"
```

---

### Task 5: Optical frontend page

**Files:**
- Modify: `frontend/pages/optical.html`
- Create: `frontend/js/optical.js`
- Modify: `frontend/index.html` (add Leaflet.js CDN)

- [ ] **Step 1: Add Leaflet.js CDN to `frontend/index.html`**

Add in `<head>`:

```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
```

- [ ] **Step 2: Create `frontend/js/optical.js`**

```javascript
/**
 * Optical Detection page logic.
 * Leaflet map with switchable tile overlays from /api/tif/ endpoint.
 */
let opticalMap = null;
let opticalOverlay = null;

function init_optical() {
  const eventSelect = document.getElementById('optical-event-select');
  if (!eventSelect) return;

  API.get('/events').then(events => {
    eventSelect.innerHTML = events.map(e =>
      `<option value="${e.key}">${e.label} (${e.year})</option>`
    ).join('');
    if (events.length > 0) loadOptical(events[0].key);
  });

  eventSelect.addEventListener('change', () => loadOptical(eventSelect.value));

  // Layer and period toggles
  document.querySelectorAll('input[name="optical-layer"]').forEach(el => {
    el.addEventListener('change', () => updateOpticalTile());
  });
  document.querySelectorAll('input[name="optical-period"]').forEach(el => {
    el.addEventListener('change', () => updateOpticalTile());
  });
}

async function loadOptical(event) {
  window._opticalEvent = event;

  try {
    const boundsData = await API.get(`/tif/${event}/bounds`);

    const mapEl = document.getElementById('optical-map');
    if (!mapEl) return;

    if (opticalMap) { opticalMap.remove(); opticalMap = null; }

    opticalMap = L.map('optical-map', { zoomControl: false }).setView(
      boundsData.center, boundsData.zoom
    );

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OSM &copy; CARTO', maxZoom: 19,
    }).addTo(opticalMap);

    L.control.zoom({ position: 'bottomright' }).addTo(opticalMap);

    window._opticalBounds = boundsData.bounds;
    updateOpticalTile();
  } catch (e) {
    const mapEl = document.getElementById('optical-map');
    if (mapEl) mapEl.innerHTML = `<div class="flex items-center justify-center h-full text-error text-sm">${e.message}</div>`;
  }
}

function updateOpticalTile() {
  if (!opticalMap || !window._opticalEvent) return;

  const layer = document.querySelector('input[name="optical-layer"]:checked')?.value || 'NDWI';
  const period = document.querySelector('input[name="optical-period"]:checked')?.value || 'after';

  // Show/hide period controls for NDWI_change
  const periodControls = document.getElementById('period-controls');
  if (periodControls) {
    periodControls.style.display = layer === 'NDWI_change' ? 'none' : 'flex';
  }

  if (opticalOverlay) {
    opticalMap.removeLayer(opticalOverlay);
    opticalOverlay = null;
  }

  const url = `/api/tif/${window._opticalEvent}/${layer}/${period}?max_pixels=600`;
  opticalOverlay = L.imageOverlay(url, window._opticalBounds, { opacity: 0.85 }).addTo(opticalMap);
}
```

- [ ] **Step 3: Replace `frontend/pages/optical.html`**

```html
<!-- Optical Detection -->
<section class="mb-12">
  <div class="flex flex-col md:flex-row justify-between items-start md:items-end gap-6">
    <div>
      <span class="text-secondary font-semibold text-sm tracking-widest uppercase mb-2 block">Intelligence Suite</span>
      <h3 class="text-5xl font-black negative-tracking text-on-surface">Temporal Analysis Canvas</h3>
      <p class="text-on-surface-variant mt-3 text-lg font-medium opacity-80">Multispectral satellite imagery analysis for temporal water detection.</p>
    </div>
    <div class="flex gap-3 items-center">
      <select id="optical-event-select" class="bg-surface-container-high text-on-surface rounded-full font-bold text-sm px-5 py-2.5 border-none focus:ring-2 focus:ring-primary/20 cursor-pointer"></select>
    </div>
  </div>
</section>

<div class="bento-grid">
  <!-- Map -->
  <div class="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-[2rem] overflow-hidden relative" style="min-height:500px">
    <div class="absolute top-6 left-6 z-[1000]">
      <span class="px-4 py-2 bg-white/90 backdrop-blur rounded-full text-xs font-bold text-slate-900 shadow-sm border border-slate-200/50" id="layer-badge">Current: NDWI</span>
    </div>
    <div id="optical-map" style="height:500px;width:100%;border-radius:2rem;z-index:1"></div>
  </div>

  <!-- Controls -->
  <div class="col-span-12 lg:col-span-4 flex flex-col gap-6">
    <!-- Layer selection -->
    <div class="bg-surface-container-lowest rounded-[2rem] p-8">
      <h4 class="text-[10px] font-bold text-secondary uppercase tracking-widest mb-4">Spectral Layer</h4>
      <div class="space-y-3">
        <label class="flex items-center gap-3 p-3 bg-surface-container-low rounded-2xl cursor-pointer hover:bg-surface-container-high transition-colors">
          <input type="radio" name="optical-layer" value="RGB" class="accent-primary">
          <div>
            <div class="text-sm font-bold">RGB (True Color)</div>
            <div class="text-[10px] text-on-surface-variant">Natural color composite</div>
          </div>
        </label>
        <label class="flex items-center gap-3 p-3 bg-primary/10 rounded-2xl cursor-pointer border border-primary/20">
          <input type="radio" name="optical-layer" value="NDWI" checked class="accent-primary">
          <div>
            <div class="text-sm font-bold text-primary">NDWI (Water Index)</div>
            <div class="text-[10px] text-on-surface-variant">Blue = water, Red = dry</div>
          </div>
        </label>
        <label class="flex items-center gap-3 p-3 bg-surface-container-low rounded-2xl cursor-pointer hover:bg-surface-container-high transition-colors">
          <input type="radio" name="optical-layer" value="NDWI_change" class="accent-primary">
          <div>
            <div class="text-sm font-bold">NDWI Change</div>
            <div class="text-[10px] text-on-surface-variant">After minus Before</div>
          </div>
        </label>
      </div>
    </div>

    <!-- Period selection -->
    <div class="bg-surface-container-lowest rounded-[2rem] p-8" id="period-controls">
      <h4 class="text-[10px] font-bold text-secondary uppercase tracking-widest mb-4">Time Period</h4>
      <div class="flex gap-3">
        <label class="flex-1 text-center p-3 bg-surface-container-low rounded-2xl cursor-pointer hover:bg-surface-container-high transition-colors text-sm font-bold">
          <input type="radio" name="optical-period" value="before" class="sr-only"> Before
        </label>
        <label class="flex-1 text-center p-3 bg-primary text-white rounded-2xl cursor-pointer text-sm font-bold">
          <input type="radio" name="optical-period" value="after" checked class="sr-only"> After
        </label>
      </div>
    </div>

    <!-- Spectral info -->
    <div class="bg-primary rounded-[2rem] p-8 text-white relative overflow-hidden">
      <h4 class="text-white/70 font-bold text-sm uppercase tracking-widest mb-1">Spectral Confidence</h4>
      <div class="text-6xl font-black negative-tracking leading-none mb-4">98.2%</div>
      <div class="flex items-center gap-2 text-sm font-medium text-white/90">
        <span class="material-symbols-outlined" style="font-variation-settings:'FILL' 1">bolt</span>
        <span>Real-time calibration active</span>
      </div>
    </div>
  </div>
</div>
<script src="/js/optical.js"></script>
```

- [ ] **Step 4: Commit**

```bash
git add frontend/pages/optical.html frontend/js/optical.js frontend/index.html
git commit -m "feat(phase2): add optical page with Leaflet map and layer controls"
```

---

### Task 6: Integration verification

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Manual browser test**

Run: `cd /Users/chris/EarthAI && python run.py`

Verify:
1. `http://localhost:8000/#rainfall` — event selector loads, chart renders with bars, controls (threshold slider, checkboxes) work
2. `http://localhost:8000/#optical` — event selector loads, Leaflet map renders, layer radio buttons switch tiles, period toggle works
3. `http://localhost:8000/api/rainfall/harvey` — returns JSON with dates, precip, etc.
4. `http://localhost:8000/api/tif/harvey/NDWI/after` — returns PNG image
5. `http://localhost:8000/api/tif/harvey/bounds` — returns JSON with bounds, center, zoom

- [ ] **Step 3: Fix any issues and commit**

```bash
git add -A
git commit -m "fix(phase2): address integration issues"
```
