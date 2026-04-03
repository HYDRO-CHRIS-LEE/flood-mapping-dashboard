"""
Data loading utilities for the EarthAI backend.
Ported from utils/data_loader.py — no Streamlit dependency.
"""

import io
import os

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from backend.config import DATA_ROOT

# ── Bounding boxes [W, S, E, N] ────────────────────────────────
EVENT_BOUNDS = {
    "harvey":        [-96.0,  29.0, -94.5,  30.5],
    "pakistan":      [ 66.5,  25.5,  69.5,  28.0],
    "dubai":         [ 54.8,  24.9,  56.2,  25.6],
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

# ── Map centers [lat, lon] ──────────────────────────────────────
EVENT_CENTERS = {
    "harvey":        [ 29.76, -95.37],
    "pakistan":      [ 27.0,   68.0 ],
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

# ── Default zoom levels ─────────────────────────────────────────
EVENT_ZOOM = {
    "harvey": 9, "pakistan": 8, "dubai": 10,
    "myanmar2015": 8, "louisiana2016": 9,
    "srilanka2017": 9, "mozambique2019": 9, "iran2019": 8,
    "china2020": 8, "sudan2020": 8,
    "germany2021": 10, "nigeria2022": 9, "libya2023": 10,
    "somalia2023": 9, "brazil2024": 8, "valencia2024": 9,
}


# ── File loaders ────────────────────────────────────────────────

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


def load_csv(event: str, name: str):
    path = os.path.join(DATA_ROOT, event, f"{name}.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


# ── Image helpers ───────────────────────────────────────────────

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


def rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    """Convert an RGBA numpy array to raw PNG bytes."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
