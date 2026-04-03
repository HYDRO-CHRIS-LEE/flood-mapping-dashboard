"""
EarthAI backend configuration.
Central place for paths, event metadata, and settings.
"""

import os

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_ROOT, "earthai.db")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# ── Auth ─────────────────────────────────────────────────────────
ADMIN_PASSWORD = os.environ.get("EARTHAI_ADMIN_PASSWORD", "earthai2026")

# ── 16 flood events ─────────────────────────────────────────────
ALL_EVENTS = {
    "harvey":       {"label": "Hurricane Harvey",          "year": 2017, "region": "Houston, TX, USA",           "color": "#ef4444"},
    "pakistan":     {"label": "Pakistan Mega Flood",        "year": 2022, "region": "Sindh Province, Pakistan",   "color": "#06b6d4"},
    "dubai":        {"label": "Dubai Flash Flood",          "year": 2024, "region": "UAE",                        "color": "#f59e0b"},
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
