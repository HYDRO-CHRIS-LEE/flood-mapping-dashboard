"""
Rainfall API router.
Serves GPM daily precipitation time-series for each event.
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.config import ALL_EVENTS
from backend.services.data_loader import load_csv

router = APIRouter(prefix="/api", tags=["rainfall"])

# ── Flood windows (start, end) for each event ──────────────────
FLOOD_WINDOWS = {
    "harvey":        ("2017-08-25", "2017-09-01"),
    "pakistan":      ("2022-08-15", "2022-09-15"),
    "dubai":         ("2024-04-15", "2024-04-20"),
    "la2025":        ("2025-01-17", "2025-02-10"),
    "myanmar2015":   ("2015-07-25", "2015-08-15"),
    "chennai2015":   ("2015-11-15", "2015-12-05"),
    "louisiana2016": ("2016-08-12", "2016-08-20"),
    "srilanka2017":  ("2017-05-25", "2017-06-10"),
    "bangladesh2017":("2017-08-01", "2017-08-25"),
    "kerala2018":    ("2018-08-10", "2018-08-20"),
    "japan2018":     ("2018-07-05", "2018-07-12"),
    "mozambique2019":("2019-03-14", "2019-03-20"),
    "iran2019":      ("2019-03-25", "2019-04-10"),
    "china2020":     ("2020-07-05", "2020-08-10"),
    "sudan2020":     ("2020-08-05", "2020-08-25"),
    "germany2021":   ("2021-07-14", "2021-07-16"),
    "kalimantan2021":("2021-01-12", "2021-01-25"),
    "nigeria2022":   ("2022-09-20", "2022-10-15"),
    "libya2023":     ("2023-09-11", "2023-09-14"),
    "somalia2023":   ("2023-11-05", "2023-11-20"),
    "brazil2024":    ("2024-05-01", "2024-05-15"),
    "valencia2024":  ("2024-10-29", "2024-11-01"),
    "afghanistan2024":("2024-05-10","2024-05-14"),
    "tennessee2021": ("2021-08-21", "2021-08-22"),
}

# ── Key facts per event ────────────────────────────────────────
KEY_FACTS = {
    "harvey":        "Harvey dumped over 1,300 mm (51 in) across Houston in 5 days — the highest tropical rainfall total ever recorded in the U.S.",
    "pakistan":      "Pakistan received 3-4x its annual average rainfall in just two months, submerging roughly one-third of the country.",
    "dubai":         "Dubai received nearly its entire annual rainfall (~75 mm) in a single day — a city built for desert conditions with almost no storm drainage.",
    "la2025":        "A series of atmospheric rivers made landfall in rapid succession, delivering extreme precipitation across Southern California.",
    "myanmar2015":   "Cyclone Komen triggered catastrophic monsoon flooding affecting over 1.6 million people across Myanmar.",
    "chennai2015":   "Chennai recorded 344 mm in a single day — the heaviest rainfall in over 100 years, flooding the entire metro area.",
    "louisiana2016": "An unnamed storm dropped 60+ cm of rain in 48 hours over Baton Rouge, flooding over 60,000 homes.",
    "srilanka2017":  "Southwest monsoon rains triggered widespread flooding and landslides, displacing over 600,000 people.",
    "bangladesh2017":"One-third of Bangladesh was submerged as monsoon floods affected 8 million people in August 2017.",
    "kerala2018":    "Kerala's worst flooding in nearly a century — all 14 districts affected, 1.5 million displaced.",
    "japan2018":     "The 'Heisei 30 July Rains' caused record-breaking rainfall across western Japan, triggering floods and landslides.",
    "mozambique2019":"Cyclone Idai made landfall with 195 km/h winds, generating a massive inland flood over Beira.",
    "iran2019":      "Spring floods swept through 26 of Iran's 31 provinces, the country's worst flooding in 70 years.",
    "china2020":     "Record Yangtze River levels — the Poyang Lake basin saw its largest flood extent since satellite monitoring began.",
    "sudan2020":     "Sudan's worst flooding in 100 years submerged entire neighborhoods in Khartoum.",
    "germany2021":   "The Ahr Valley received a full month of rain in 24 hours, destroying hundreds of bridges and roads.",
    "kalimantan2021":"South Kalimantan saw its worst flooding in 50 years, displacing over 60,000 residents.",
    "nigeria2022":   "Flooding affected 33 of 36 states; the Anambra-Delta corridor saw the worst inundation.",
    "libya2023":     "Cyclone Daniel caused catastrophic dam failures in Derna, killing thousands in hours.",
    "somalia2023":   "Unprecedented October-November rains flooded over 1 million people across the Shabelle basin.",
    "brazil2024":    "Cyclone-driven rains submerged 90% of Rio Grande do Sul's municipalities — Brazil's worst climate disaster.",
    "valencia2024":  "A DANA (cut-off low) dropped 450 mm in 8 hours near Valencia — Spain's deadliest flash flood in decades.",
    "afghanistan2024":"Flash floods in Baghlan Province killed hundreds in minutes as normally dry riverbeds overflowed.",
    "tennessee2021": "Humphreys County received 43 cm (17 in) in 24 hours — a 1-in-1000-year event for the region.",
}


@router.get("/rainfall/{event}")
def get_rainfall(event: str):
    """Return GPM daily precipitation time-series for the given event."""
    if event not in ALL_EVENTS:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "EVENT_NOT_FOUND",
                    "message": f"No event with key '{event}'"},
        )

    df = load_csv(event, "GPM_rainfall_daily")
    if df is None:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": "DATA_NOT_FOUND",
                    "message": f"GPM rainfall data not available for '{event}'"},
        )

    # Parse and clean
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["precip_mm_day"])
    df["precip_mm_day"] = df["precip_mm_day"].clip(lower=0)

    # Summary stats
    peak_val = float(df["precip_mm_day"].max())
    peak_date = df.loc[df["precip_mm_day"].idxmax(), "date"].strftime("%b %d")
    total_mm = float(df["precip_mm_day"].sum())
    period_days = len(df)

    dates = [d.strftime("%Y-%m-%d") for d in df["date"]]
    precip = [round(float(v), 2) for v in df["precip_mm_day"]]

    fw = FLOOD_WINDOWS.get(event)
    flood_window = list(fw) if fw else None

    fact = KEY_FACTS.get(event, "")

    ev_meta = ALL_EVENTS[event]
    event_info = {
        "key": event,
        "label": ev_meta["label"],
        "year": ev_meta["year"],
        "region": ev_meta["region"],
    }

    return {
        "ok": True,
        "data": {
            "dates": dates,
            "precip": precip,
            "flood_window": flood_window,
            "fact": fact,
            "peak_val": round(peak_val, 1),
            "peak_date": peak_date,
            "total_mm": round(total_mm, 0),
            "period_days": period_days,
            "event": event_info,
        },
    }
