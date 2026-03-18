"""
Data loading utilities for the flood mapping dashboard.
Supports 15 flood events (SAR-derived labels, SAR_VH excluded from training features).
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import streamlit as st

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ── 15 flood events ──────────────────────────────────────────────
ALL_EVENTS = {
    "harvey":       {"label": "Hurricane Harvey",          "year": 2017, "region": "Houston, TX, USA",           "color": "#ef4444"},
    "pakistan":     {"label": "Pakistan Mega Flood",        "year": 2022, "region": "Sindh Province, Pakistan",   "color": "#06b6d4"},
    "myanmar2015":  {"label": "Myanmar Cyclone Komen",      "year": 2015, "region": "Irrawaddy Delta, Myanmar",   "color": "#8b5cf6"},
    "louisiana2016":{"label": "Louisiana Flood",            "year": 2016, "region": "Baton Rouge, LA, USA",       "color": "#f97316"},
    "srilanka2017": {"label": "Sri Lanka Flood",            "year": 2017, "region": "Southern Sri Lanka",         "color": "#14b8a6"},
    "mozambique2019":{"label":"Cyclone Idai",               "year": 2019, "region": "Beira, Mozambique",          "color": "#0ea5e9"},
    "iran2019":     {"label": "Iran Flood",                 "year": 2019, "region": "Khuzestan, Iran",            "color": "#d946ef"},
    "china2020":    {"label": "Yangtze River Flood",        "year": 2020, "region": "Hubei / Poyang Lake, China", "color": "#f43f5e"},
    "sudan2020":    {"label": "Sudan Flash Flood",          "year": 2020, "region": "Khartoum, Sudan",            "color": "#fb923c"},
    "germany2021":  {"label": "Ahr Valley Flood",           "year": 2021, "region": "Rhineland-Palatinate, Germany","color": "#a3e635"},
    "nigeria2022":  {"label": "Nigeria Flood",              "year": 2022, "region": "Anambra / Delta State, Nigeria","color": "#fb7185"},
    "libya2023":    {"label": "Libya Flood (Derna)",        "year": 2023, "region": "Derna, Libya",               "color": "#c084fc"},
    "somalia2023":  {"label": "Somalia Flood",              "year": 2023, "region": "Hirshabelle, Somalia",       "color": "#fbbf24"},
    "brazil2024":   {"label": "Brazil Rio Grande Flood",    "year": 2024, "region": "Rio Grande do Sul, Brazil",  "color": "#4ade80"},
    "valencia2024": {"label": "Spain Valencia Flood",       "year": 2024, "region": "Valencia, Spain",            "color": "#60a5fa"},
}

# Bounding boxes [W, S, E, N]
EVENT_BOUNDS = {
    "harvey":        [-96.0,  29.0, -94.5,  30.5],
    "pakistan":      [ 66.5,  25.5,  69.5,  28.0],
    "myanmar2015":   [ 94.5,  15.5,  96.5,  17.5],
    "louisiana2016": [-91.5,  30.0, -90.0,  31.0],
    "srilanka2017":  [ 80.0,   6.0,  81.5,   7.5],
    "mozambique2019":[ 34.0,  -20.0, 35.5, -18.5],
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

# Map centers [lat, lon]
EVENT_CENTERS = {
    "harvey":        [ 29.76, -95.37],
    "pakistan":      [ 27.0,   68.0 ],
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
    "harvey": 9, "pakistan": 8,
    "myanmar2015": 8, "louisiana2016": 9,
    "srilanka2017": 9, "mozambique2019": 9, "iran2019": 8,
    "china2020": 8, "sudan2020": 8,
    "germany2021": 10, "nigeria2022": 9, "libya2023": 10,
    "somalia2023": 9, "brazil2024": 8, "valencia2024": 9,
}

# SAR date configs for GEE export (reference for Colab notebook)
EVENT_DATES = {
    "harvey":        {"before": ("2017-08-01","2017-08-22"), "flood": ("2017-08-26","2017-09-10")},
    "pakistan":      {"before": ("2022-06-01","2022-07-20"), "flood": ("2022-08-15","2022-09-25")},
    "myanmar2015":   {"before": ("2015-06-01","2015-07-20"), "flood": ("2015-07-25","2015-08-20")},
    "louisiana2016": {"before": ("2016-07-01","2016-08-10"), "flood": ("2016-08-12","2016-08-25")},
    "srilanka2017":  {"before": ("2017-04-01","2017-05-15"), "flood": ("2017-05-25","2017-06-15")},
    "mozambique2019":{"before": ("2019-02-01","2019-03-10"), "flood": ("2019-03-14","2019-04-01")},
    "iran2019":      {"before": ("2019-02-01","2019-03-20"), "flood": ("2019-03-25","2019-04-20")},
    "china2020":     {"before": ("2020-06-01","2020-06-30"), "flood": ("2020-07-05","2020-08-15")},
    "sudan2020":     {"before": ("2020-07-01","2020-07-31"), "flood": ("2020-08-05","2020-09-01")},
    "germany2021":   {"before": ("2021-06-01","2021-07-12"), "flood": ("2021-07-14","2021-07-25")},
    "nigeria2022":   {"before": ("2022-08-01","2022-09-15"), "flood": ("2022-09-20","2022-10-20")},
    "libya2023":     {"before": ("2023-08-01","2023-09-09"), "flood": ("2023-09-11","2023-09-25")},
    "somalia2023":   {"before": ("2023-10-01","2023-11-01"), "flood": ("2023-11-05","2023-11-30")},
    "brazil2024":    {"before": ("2024-04-01","2024-04-30"), "flood": ("2024-05-01","2024-05-25")},
    "valencia2024":  {"before": ("2024-10-01","2024-10-28"), "flood": ("2024-10-29","2024-11-10")},
}


def get_available_events() -> list[str]:
    """Return event keys that have at least one data file in data/."""
    available = []
    for key in ALL_EVENTS:
        d = os.path.join(DATA_ROOT, key)
        if os.path.isdir(d) and len(os.listdir(d)) > 0:
            available.append(key)
    return available


@st.cache_data(ttl=3600, show_spinner=False)
def load_tif(event: str, name: str, max_pixels: int = 512):
    path = os.path.join(DATA_ROOT, event, f"{name}.tif")
    if not os.path.exists(path):
        return None, {}
    with rasterio.open(path) as src:
        scale = max_pixels / max(src.width, src.height)
        out_w = max(1, int(src.width  * scale))
        out_h = max(1, int(src.height * scale))
        data  = src.read(out_shape=(src.count, out_h, out_w),
                         resampling=Resampling.bilinear).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        bounds = src.bounds
        meta   = {"bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
                  "nodata": nodata, "count": src.count}
    return data, meta


@st.cache_data(ttl=3600, show_spinner=False)
def load_csv(event: str, name: str):
    path = os.path.join(DATA_ROOT, event, f"{name}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_all_rf_samples(available_events: list[str]) -> pd.DataFrame | None:
    """
    Load RF training CSVs from ALL available events.
    Adds an 'event' column for group-based train/test splitting.
    """
    frames = []
    for ev in available_events:
        df = load_csv(ev, "RF_training_samples")
        if df is not None:
            df["event"] = ev
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined


def norm_band(arr, p_low=2, p_high=98):
    arr   = arr.copy().astype(np.float32)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    lo = np.percentile(valid, p_low)
    hi = np.percentile(valid, p_high)
    if hi == lo:
        return np.zeros_like(arr)
    return np.where(np.isnan(arr), 0, np.clip((arr - lo) / (hi - lo), 0, 1))


def tif_to_rgba(band, colormap="gray"):
    import matplotlib
    normed = norm_band(band)
    rgba   = (matplotlib.colormaps[colormap](normed) * 255).astype(np.uint8)
    rgba[band == 0, 3] = 0
    return rgba


def rgb_tif_to_rgba(r, g, b):
    r8 = (norm_band(r) * 255).astype(np.uint8)
    g8 = (norm_band(g) * 255).astype(np.uint8)
    b8 = (norm_band(b) * 255).astype(np.uint8)
    a8 = np.where((r == 0) & (g == 0) & (b == 0), 0, 255).astype(np.uint8)
    return np.stack([r8, g8, b8, a8], axis=-1)


def rgba_to_b64(rgba):
    import io, base64
    from PIL import Image
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
