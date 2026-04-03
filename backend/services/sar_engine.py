"""
SAR flood detection engine.
Ported from modules/module1_sar.py — pure functions, no Streamlit dependency.
"""

import io

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu


def compute_otsu(sar_arr: np.ndarray) -> float:
    """Compute Otsu threshold on valid SAR pixels in (-35, 5) dB range."""
    valid = sar_arr[~np.isnan(sar_arr)].flatten()
    valid = valid[(valid > -35) & (valid < 5)]
    if len(valid) < 100:
        return -16.0
    return float(threshold_otsu(valid))


def apply_threshold(
    sar: np.ndarray, thr: float, perm: np.ndarray | None = None
) -> np.ndarray:
    """Binary flood mask: pixels below threshold. Optionally remove permanent water."""
    flood = (sar < thr).astype(np.uint8)
    if perm is not None:
        if perm.shape != flood.shape:
            from skimage.transform import resize
            perm = resize(perm, flood.shape, order=0, preserve_range=True)
        flood = flood & (~(perm > 0.5)).astype(np.uint8)
    return flood


def compute_metrics(sar_arr: np.ndarray, flood_mask: np.ndarray) -> dict:
    """Compute flood statistics from SAR array and binary flood mask."""
    total_valid = int(np.sum(~np.isnan(sar_arr)))
    flood_px = int(np.sum(flood_mask))
    flood_pct = flood_px / total_valid * 100 if total_valid > 0 else 0.0
    flood_km2 = flood_px * 0.0004
    return {
        "flood_px": flood_px,
        "flood_pct": round(flood_pct, 2),
        "flood_km2": round(flood_km2, 2),
        "total_valid": total_valid,
    }


def compute_histogram(sar_arr: np.ndarray, bins: int = 80) -> dict:
    """Compute histogram of valid SAR pixels. Returns centers and counts."""
    valid = sar_arr[~np.isnan(sar_arr)].flatten()
    valid = valid[(valid > -35) & (valid < 5)]
    counts, edges = np.histogram(valid, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {"centers": centers, "counts": counts.tolist()}


def render_flood_tile(sar_arr: np.ndarray, flood_mask: np.ndarray) -> bytes:
    """
    Create RGBA image: gray SAR background with blue (37,99,235) flood pixels.
    Returns PNG bytes.
    """
    sar_norm = np.clip((sar_arr - (-25)) / 25, 0, 1)
    gray = (sar_norm * 200).astype(np.uint8)
    r = gray.copy()
    g = gray.copy()
    b = gray.copy()
    r[flood_mask == 1] = 37
    g[flood_mask == 1] = 99
    b[flood_mask == 1] = 235
    a = np.full_like(r, 220)
    rgba = np.stack([r, g, b, a], axis=-1)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
