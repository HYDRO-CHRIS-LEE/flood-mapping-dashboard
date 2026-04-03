"""
Optical imagery API router.
Serves RGB, NDWI, and NDWI-change tiles as PNG images,
plus bounding-box metadata for map display.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.config import ALL_EVENTS
from backend.services.data_loader import (
    EVENT_BOUNDS,
    EVENT_CENTERS,
    EVENT_ZOOM,
    load_tif,
    rgb_tif_to_rgba,
    rgba_to_png_bytes,
    tif_to_rgba,
)

router = APIRouter(prefix="/api", tags=["optical"])

VALID_LAYERS = {"RGB", "NDWI", "NDWI_change"}
VALID_PERIODS = {"before", "after"}


@router.get("/tif/{event}/bounds")
def get_bounds(event: str):
    """Return geographic bounds, center, and zoom for the event."""
    if event not in ALL_EVENTS:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "EVENT_NOT_FOUND",
                    "message": f"No event with key '{event}'"},
        )

    raw = EVENT_BOUNDS.get(event)
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "BOUNDS_NOT_FOUND",
                    "message": f"No bounds data for event '{event}'"},
        )

    # raw is [W, S, E, N] -> convert to [[south,west],[north,east]]
    w, s, e, n = raw
    bounds = [[s, w], [n, e]]
    center = EVENT_CENTERS.get(event, [(s + n) / 2, (w + e) / 2])
    zoom = EVENT_ZOOM.get(event, 9)

    return {"ok": True, "data": {"bounds": bounds, "center": center, "zoom": zoom}}


@router.get("/tif/{event}/{layer}/{period}")
def get_tile(event: str, layer: str, period: str):
    """Return a PNG tile for the requested event/layer/period."""
    if event not in ALL_EVENTS:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "EVENT_NOT_FOUND",
                    "message": f"No event with key '{event}'"},
        )

    if layer not in VALID_LAYERS:
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": "INVALID_LAYER",
                    "message": f"Layer must be one of {sorted(VALID_LAYERS)}"},
        )

    if period not in VALID_PERIODS:
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": "INVALID_PERIOD",
                    "message": f"Period must be one of {sorted(VALID_PERIODS)}"},
        )

    if layer == "RGB":
        data, meta = load_tif(event, f"RGB_{period}")
        if data is None:
            raise HTTPException(
                status_code=404,
                detail={"ok": False, "error": "DATA_NOT_FOUND",
                        "message": f"RGB_{period}.tif not found for '{event}'"},
            )
        rgba = rgb_tif_to_rgba(data[0], data[1], data[2])

    elif layer == "NDWI":
        data, meta = load_tif(event, f"NDWI_{period}")
        if data is None:
            raise HTTPException(
                status_code=404,
                detail={"ok": False, "error": "DATA_NOT_FOUND",
                        "message": f"NDWI_{period}.tif not found for '{event}'"},
            )
        rgba = tif_to_rgba(data[0], colormap="RdYlBu")

    elif layer == "NDWI_change":
        before_data, _ = load_tif(event, "NDWI_before")
        after_data, _ = load_tif(event, "NDWI_after")
        if before_data is None or after_data is None:
            raise HTTPException(
                status_code=404,
                detail={"ok": False, "error": "DATA_NOT_FOUND",
                        "message": f"NDWI before/after .tif not found for '{event}'"},
            )
        rgba = tif_to_rgba(after_data[0] - before_data[0], colormap="RdBu")

    png = rgba_to_png_bytes(rgba)
    return Response(content=png, media_type="image/png")
