"""
SAR flood detection API router.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from backend.config import ALL_EVENTS
from backend.services.data_loader import (
    EVENT_BOUNDS,
    EVENT_CENTERS,
    EVENT_ZOOM,
    load_tif,
)
from backend.services.sar_engine import (
    apply_threshold,
    compute_histogram,
    compute_metrics,
    compute_otsu,
    render_flood_tile,
)

router = APIRouter(prefix="/api/sar", tags=["sar"])


class SARComputeRequest(BaseModel):
    event: str
    threshold: float | None = None
    remove_permanent: bool = True


@router.post("/compute")
def sar_compute(body: SARComputeRequest):
    """Compute SAR flood detection for an event."""
    event = body.event

    if event not in ALL_EVENTS:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "EVENT_NOT_FOUND",
                    "message": f"No event with key '{event}'"},
        )

    sar_data, _ = load_tif(event, "SAR_after", max_pixels=512)
    if sar_data is None:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "DATA_NOT_FOUND",
                    "message": f"SAR_after.tif not found for '{event}'"},
        )

    sar_arr = sar_data[0]

    # Otsu auto-threshold
    otsu_val = compute_otsu(sar_arr)
    thr = body.threshold if body.threshold is not None else round(otsu_val, 1)

    # Permanent water mask
    perm_arr = None
    if body.remove_permanent:
        perm_data, _ = load_tif(event, "JRC_permanent_water", max_pixels=512)
        if perm_data is not None:
            perm_arr = perm_data[0]

    flood_mask = apply_threshold(sar_arr, thr, perm_arr)
    metrics = compute_metrics(sar_arr, flood_mask)
    histogram = compute_histogram(sar_arr)

    # Build geo metadata
    raw_bounds = EVENT_BOUNDS.get(event)
    if raw_bounds:
        w, s, e, n = raw_bounds
        bounds = [[s, w], [n, e]]
    else:
        bounds = None
    center = EVENT_CENTERS.get(event)
    zoom = EVENT_ZOOM.get(event, 9)

    # Tile URL for the frontend to fetch the PNG
    remove_flag = 1 if body.remove_permanent else 0
    tile_url = f"/api/sar/tile/{event}/{thr}/{remove_flag}"

    return {
        "ok": True,
        "data": {
            "otsu": round(otsu_val, 1),
            "threshold": thr,
            "metrics": metrics,
            "histogram": histogram,
            "tile_url": tile_url,
            "bounds": bounds,
            "center": center,
            "zoom": zoom,
        },
    }


@router.get("/tile/{event}/{threshold}/{remove_perm}")
def sar_tile(event: str, threshold: float, remove_perm: int):
    """Return a PNG flood overlay tile."""
    if event not in ALL_EVENTS:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "EVENT_NOT_FOUND",
                    "message": f"No event with key '{event}'"},
        )

    sar_data, _ = load_tif(event, "SAR_after", max_pixels=512)
    if sar_data is None:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "DATA_NOT_FOUND",
                    "message": f"SAR_after.tif not found for '{event}'"},
        )

    sar_arr = sar_data[0]

    perm_arr = None
    if remove_perm:
        perm_data, _ = load_tif(event, "JRC_permanent_water", max_pixels=512)
        if perm_data is not None:
            perm_arr = perm_data[0]

    flood_mask = apply_threshold(sar_arr, threshold, perm_arr)
    png = render_flood_tile(sar_arr, flood_mask)
    return Response(content=png, media_type="image/png")
