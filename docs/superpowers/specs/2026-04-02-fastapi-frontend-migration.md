# EarthAI: Streamlit → FastAPI + Vanilla Frontend Migration

## Overview

Migrate the EarthAI flood mapping dashboard from Streamlit to a FastAPI backend + vanilla HTML/CSS/JS frontend, using the stitch HTML mockup designs pixel-for-pixel. The app serves ~20 concurrent high school students in a classroom setting.

## Motivation

Streamlit constrains UI customization — its widget system, layout model, and rendering pipeline prevent faithful reproduction of the stitch HTML designs. Moving to FastAPI + vanilla frontend gives complete UI control while reusing all existing Python data processing logic.

## Architecture

```
Browser (HTML/CSS/JS)              FastAPI (Python)
──────────────────────             ──────────────────────
Stitch HTML designs (as-is)        /api/events
Tailwind CSS (CDN)                 /api/rainfall/{event}
Alpine.js (reactivity)             /api/tif/{event}/{layer}/{period}
Leaflet.js (maps)                  /api/sar/compute
Plotly.js (charts)                 /api/classifier/train
sessionStorage (team state)        /api/flappy/*
Hash-based SPA routing             /api/leaderboard/*
                                   SQLite (leaderboard data)
```

**Request flow:** Browser → `fetch()` → FastAPI endpoint → Python service (reused logic) → JSON/PNG response → JS renders into DOM.

## Project Structure

```
EarthAI/
├── backend/
│   ├── main.py                  # FastAPI app, CORS, static mount, router registration
│   ├── config.py                # DATA_ROOT, DB path, STAGES, ALL_EVENTS, constants
│   ├── db.py                    # SQLite init, connection helper, migration
│   ├── routers/
│   │   ├── events.py            # GET /api/events, GET /api/events/{key}
│   │   ├── rainfall.py          # GET /api/rainfall/{event}
│   │   ├── optical.py           # GET /api/tif/{event}/{layer}/{period} → PNG
│   │   ├── sar.py               # POST /api/sar/compute → PNG + metrics JSON
│   │   ├── classifier.py        # POST /api/classifier/train → metrics JSON
│   │   ├── flappy.py            # POST /api/flappy/train, /submit, /race
│   │   └── leaderboard.py       # GET/POST /api/leaderboard/{type}
│   └── services/
│       ├── data_loader.py       # From utils/data_loader.py — GeoTIFF/CSV loading
│       ├── normalization.py     # From utils/normalization.py — z-score normalization
│       ├── sar_engine.py        # From module1 — threshold, Otsu, flood mask, map tile
│       ├── optical_engine.py    # From module2 — NDWI/RGB tile generation
│       ├── rf_engine.py         # From module4 — RF training, metrics, importance
│       └── flappy_engine.py     # From module5 — DQN training, race runner, replay
├── frontend/
│   ├── index.html               # SPA shell: sidebar + topbar + <main id="page-content">
│   ├── pages/
│   │   ├── rainfall.html        # Page template (innerHTML replacement)
│   │   ├── optical.html
│   │   ├── sar.html
│   │   ├── classifier.html
│   │   └── flappy.html
│   ├── css/
│   │   └── style.css            # Consolidated from stitch designs + Tailwind overrides
│   └── js/
│       ├── app.js               # SPA router, page loader, shared utilities
│       ├── api.js               # Centralized fetch wrapper (base URL, error handling)
│       ├── rainfall.js          # Page-specific logic: fetch data → Plotly chart
│       ├── optical.js           # Leaflet map, layer toggle, tile fetching
│       ├── sar.js               # Leaflet map, threshold slider → API → map update
│       ├── classifier.js        # Form submit → API → render metrics/charts
│       └── flappy.js            # Training, upload, submission, replay viewer
├── data/                        # Existing GeoTIFF/CSV data (unchanged)
│   ├── harvey/
│   ├── pakistan/
│   ├── ... (all events)
│   └── earthai.db               # SQLite database (created on first run)
├── requirements.txt             # fastapi, uvicorn, rasterio, scikit-learn, etc.
└── run.py                       # Entry point: uvicorn backend.main:app
```

## Frontend Design

### SPA Shell (`index.html`)

Single HTML file containing:
- **Sidebar** (fixed left, 256px): Copied from stitch designs. Dark glass sidebar with EarthAI branding, navigation links, "Export Report" button. Navigation uses hash routing (`#rainfall`, `#optical`, `#sar`, `#classifier`, `#flappy`).
- **Top App Bar** (sticky top): Search bar, notification icon, settings icon, user profile avatar with dropdown for display name. Team name stored in `sessionStorage`.
- **Main Content Area** (`<main id="page-content">`): Dynamically replaced based on active route.
- **Profile Dropdown**: Alpine.js-driven dropdown with display name input, save button. Name persists in `sessionStorage` and is sent as `X-Team-Name` header on API calls.

The sidebar and topbar are shared across all pages and come directly from the stitch HTML. Only the `<main>` content changes per page.

### Page Templates

Each `pages/*.html` contains only the `<main>` inner content (no `<html>`, `<head>`, `<body>`). Loaded via `fetch()` and inserted into `#page-content`. After insertion, the page-specific JS module initializes (binds events, fetches data, renders charts/maps).

### CSS Strategy

- **Tailwind CSS** via CDN (`https://cdn.tailwindcss.com`) with the same `tailwind.config` from stitch designs (MD3 color tokens).
- **`style.css`**: Custom styles not covered by Tailwind — `.bento-grid`, `.negative-tracking`, `.material-symbols-outlined`, glassmorphism effects, animations.
- **No build step**: All CSS is either Tailwind utility classes (inline in HTML) or the single `style.css` file.

### JavaScript Architecture

- **`app.js`**: Hash router that listens to `hashchange`. On route change: loads page HTML, inserts into DOM, calls page init function. Also handles sidebar active state, profile dropdown.
- **`api.js`**: Wrapper around `fetch()`. Sets `Content-Type`, `X-Team-Name` header from `sessionStorage`. Handles errors uniformly. Base URL configurable for dev vs prod.
- **Page modules** (`rainfall.js`, etc.): Each exports an `init()` function called after page HTML is inserted. Binds event listeners, fetches initial data, renders visualizations.

### Library Versions (CDN)

- Tailwind CSS: `https://cdn.tailwindcss.com?plugins=forms,container-queries`
- Alpine.js: `https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js`
- Leaflet.js: `https://unpkg.com/leaflet@1.9.4/dist/leaflet.js` + CSS
- Plotly.js: `https://cdn.plot.ly/plotly-2.35.0.min.js`
- Material Symbols: Google Fonts CDN
- Inter font: Google Fonts CDN

## Backend Design

### FastAPI Application (`main.py`)

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EarthAI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Routers
app.include_router(events_router, prefix="/api")
app.include_router(rainfall_router, prefix="/api")
app.include_router(optical_router, prefix="/api")
app.include_router(sar_router, prefix="/api")
app.include_router(classifier_router, prefix="/api")
app.include_router(flappy_router, prefix="/api")
app.include_router(leaderboard_router, prefix="/api")

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

### API Endpoints

#### Events

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/events` | — | `[{key, label, year, region, color}, ...]` |
| GET | `/api/events/{key}` | — | `{key, label, year, region, color, has_sar, has_optical, has_gpm}` |

#### Rainfall

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/rainfall/{event}` | — | `{dates: [...], precip: [...], flood_window: [start, end], fact: "..."}` |

#### Optical

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/tif/{event}/{layer}/{period}` | query: `max_pixels=600` | PNG image (`image/png`) with `bounds` in response header |
| GET | `/api/tif/{event}/bounds` | — | `{bounds: [west, south, east, north], center: [lat, lng], zoom: N}` |

`layer`: `RGB`, `NDWI`, `NDWI_change`. `period`: `before`, `after`.

The endpoint renders the GeoTIFF to a PNG with transparency (same logic as current `tif_to_rgba` + `rgba_to_b64`, but returns raw PNG bytes instead of base64). Returns `image/png` content type so Leaflet can use the URL directly as `ImageOverlay` source.

#### SAR

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/api/sar/compute` | `{event, threshold, remove_permanent}` | `{flood_px, flood_pct, flood_km2, otsu_val, histogram: {centers, counts}, tile_url}` |
| GET | `/api/sar/tile/{event}/{threshold}/{remove_perm}` | — | PNG image of flood map overlay |

Two-step flow: POST compute returns metrics + a tile URL. Frontend fetches the tile URL for the Leaflet overlay. This avoids embedding large base64 images in JSON.

#### Classifier

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/api/classifier/train` | `{features, n_trees, max_depth, min_leaf, max_feat, bootstrap, class_weight, scaling, balance, sample_pct, outlier}` | `{accuracy, precision, recall, f1, n_train, n_test, cm, importance, train_events, test_events, elapsed}` |
| POST | `/api/classifier/submit` | `{team, f1, accuracy, precision, recall, features, n_trees, max_depth}` | `{ok: true}` |

Training happens synchronously (RF training is fast, typically <3s). The response contains all data needed to render the results section.

#### Flappy Bird

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/api/flappy/train` | `{arch, lr, batch_size, dropout, gamma, optimizer, episodes}` | `{best_score, model_id}` (model stored server-side in temp) |
| POST | `/api/flappy/upload` | multipart: `state_dict` (.pt), `metadata` (.json) | `{model_id, validation: "ok"}` |
| POST | `/api/flappy/submit` | `{team, model_id, stage_id}` | `{ok: true}` |
| POST | `/api/flappy/race` | `{stage_id, admin_password}` | `{race_id, results: [...], replay: {...}}` |
| GET | `/api/flappy/stages` | — | `[{id, label, pass_avg, ...}]` |
| GET | `/api/flappy/unlocked/{team}` | — | `[1, 2, 3]` (unlocked stage IDs) |

DQN training runs synchronously (demo-scale, <30s). Model stored server-side keyed by `model_id` (UUID). The replay data is returned as JSON for the existing Canvas replay viewer.

#### Leaderboard

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/leaderboard/classifier` | — | `[{team, f1, accuracy, ...}, ...]` sorted by F1 |
| GET | `/api/leaderboard/flappy/{stage_id}` | — | `[{team, avg_score, max_score, passed, ...}, ...]` |

Read from SQLite. Auto-refresh on frontend via polling every 5 seconds (same as current Streamlit `@st.fragment(run_every=5)`).

### SQLite Schema (`db.py`)

```sql
CREATE TABLE IF NOT EXISTS classifier_leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    f1 REAL NOT NULL,
    accuracy REAL,
    precision_val REAL,
    recall REAL,
    features TEXT,          -- JSON array
    n_trees INTEGER,
    max_depth INTEGER,
    test_events TEXT,       -- comma-separated
    submitted_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS flappy_leaderboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    stage_id INTEGER NOT NULL,
    avg_score REAL NOT NULL,
    max_score INTEGER,
    survival_steps_avg REAL,
    episode_scores TEXT,    -- JSON array
    passed INTEGER DEFAULT 0,
    race_id TEXT,
    submitted_at TEXT DEFAULT (datetime('now'))
);
```

On startup, `db.py` creates tables if they don't exist. Optionally migrates data from existing `leaderboard.json` / `flappy_leaderboard.json` files.

### Services Layer

Each service module is a direct port of the existing Streamlit module logic, with Streamlit-specific code removed:

| Service | Source | What changes |
|---------|--------|-------------|
| `data_loader.py` | `utils/data_loader.py` | Remove `@st.cache_data`, return raw data |
| `normalization.py` | `utils/normalization.py` | No changes needed |
| `sar_engine.py` | `modules/module1_sar.py` | Extract `apply_threshold`, `make_map` → return numpy arrays, no Folium |
| `optical_engine.py` | `modules/module2_optical.py` | Extract tile generation → return PNG bytes |
| `rf_engine.py` | `modules/module4_rf.py` | Extract `train_rf`, `generate_hints`, preprocessing → return dicts |
| `flappy_engine.py` | `modules/module5_flappy.py` | Extract training loop, race runner → return results |

Key principle: services are pure Python functions that take parameters and return data. No framework dependencies (no Streamlit, no FastAPI imports). Routers call services and format responses.

## Page-by-Page Frontend Spec

### Rainfall Analysis (`#rainfall`)

**Source design:** `stitch/rainfall_analysis_with_profile_settings/code.html`

**Layout:**
- Hero header: Event overline, "Rainfall Analysis" title, subtitle
- Bento grid (12 cols):
  - **Col 8**: Rainfall timeline — Plotly.js bar chart. Bars colored by threshold. Hover shows date + mm. X-axis: dates. Flood period highlighted as rectangle.
  - **Col 4**: Cumulative precipitation card — large number, progress bar, "+X% above seasonal average"
  - **Col 4**: Return period / flood window indicator
  - **Col 8**: Basin intel section with key facts about the event
- Footer metadata bar

**Interactivity:**
- Threshold slider (HTML `<input type="range">`) → re-colors bars above/below threshold
- Flood period highlight toggle
- Cumulative overlay toggle
- All client-side (data fetched once, filtering in JS)

### Optical Detection (`#optical`)

**Source design:** `stitch/optical_detection_with_profile_settings/code.html`

**Layout:**
- Hero header: "Temporal Analysis Canvas"
- Bento grid:
  - **Col 8**: Leaflet map with ImageOverlay. Layer badge ("Current: Multispectral View"). Zoom controls.
  - **Col 4**: NDVI Variance chart (CSS bar chart)
  - **Col 4**: Spectral Confidence card (primary blue, large %)
  - **Col 4**: Spectral Bands (NIR/SWIR/RED progress bars)
  - **Col 8**: Detection Timeline (3-card grid with timestamps)
- Footer

**Interactivity:**
- Layer selector (RGB / NDWI / NDWI Change) → fetches new PNG tile from `/api/tif/...`
- Period toggle (Before / After) → fetches new tile
- Map zoom/pan via Leaflet

### SAR Detection (`#sar`)

**Source design:** `stitch/sar_detection_with_profile_settings/code.html`

**Layout:**
- Hero header: "SAR Detection Analytics"
- Bento grid:
  - **Col 8**: Leaflet map with SAR overlay. "LIVE SAR STREAM" badge. "SENTINEL-1B" badge.
  - **Col 4**: Flood Mask Metrics — Inundated Area, Critical Assets, Impacted Population, Resource allocation bar
  - **Col 8**: Two cards — Radar Threshold Controls (backscatter slider, polarization ratio slider) + Active Filters (chip toggles)
  - **Col 4**: Recent Detections timeline
- Footer

**Interactivity:**
- Threshold slider → `POST /api/sar/compute` → update metrics + map tile
- Otsu auto-threshold button → compute → set slider value
- Permanent water removal toggle → recompute

### AI Flood Classifier (`#classifier`)

**Source design:** `stitch/ai_flood_classifier_with_profile_settings/code.html`

**Layout:**
- Hero header: "AI Flood Classifier", "Classifier Engine v4.2"
- Bento grid:
  - **Col 4**: Configuration panel — architecture selector (visual only), confidence threshold slider, overlay toggles
  - **Col 2+2**: Quick stats — Mean IoU, Latency (mapped to F1, training time)
  - **Col 8**: Confusion Matrix — 3x3 grid of colored cells (mapped to 2x2 flood/non-flood)
  - **Col 8**: Feature Importance (SHAP) — horizontal progress bars
  - **Full width**: Active Model Leaderboard table
- Footer

**Interactivity:**
- Feature checkboxes, hyperparameter controls (actual Streamlit controls replaced with HTML form elements)
- "Run Inference" / "Train" button → `POST /api/classifier/train` → render results
- Submit to leaderboard button → `POST /api/classifier/submit`
- Leaderboard polls every 5s

### Flappy Bird Competition (`#flappy`)

**Source design:** `stitch/flappy_bird_competition_with_profile_settings/code.html`

**Layout:**
- Hero header: "Agent_Earth_v4.2", "Agent Registry v4.2"
- Bento grid:
  - **Col 8**: Replay viewport — Canvas element for race replay (existing `flappy_race.html` logic embedded). Score + Success Rate overlays.
  - **Col 4**: Live Leaderboard — ranked list with avatars, scores
  - **Col 4**: Reward Weights — distance/pipe clearance/smoothness sliders
  - **Col 8**: Dark panel split — Environment Physics (gravity, jump force, pipe gap, speed) + Telemetry (LR, batch size, gamma, runtime)
- Footer stats: Active Models, Sims/Hour, Prize Pool, Season End
- Footer metadata bar

**Interactivity:**
- Architecture selector, hyperparameter controls → HTML form
- Train button → `POST /api/flappy/train` → show spinner → results
- Upload .pt + metadata.json → `POST /api/flappy/upload`
- Submit to stage → `POST /api/flappy/submit`
- Race (admin) → `POST /api/flappy/race` → render Canvas replay
- Leaderboard polls every 5s

## Deployment

**Local development:**
```bash
pip install -r requirements.txt
python run.py
# → http://localhost:8000
```

**Server deployment:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Single process serves both API and static files. No separate frontend server needed. SQLite file lives in `data/earthai.db`. Data directory (`data/`) must contain the GeoTIFF/CSV files from the Colab export.

## Sub-Project Decomposition

This migration is too large for a single implementation plan. It decomposes into 4 sub-projects, each independently deployable and testable:

### Sub-Project 1: Core Infrastructure
- FastAPI app scaffold (`main.py`, `config.py`, `db.py`)
- SPA shell (`index.html` with sidebar, topbar, router)
- `app.js` (hash router, page loading)
- `api.js` (fetch wrapper)
- `style.css` (consolidated from stitch designs)
- Events API (`/api/events`)
- SQLite initialization
- **Testable output:** App loads, sidebar navigates between empty pages, events API returns data

### Sub-Project 2: Rainfall + Optical Pages
- `services/data_loader.py` (port from utils)
- Rainfall API + `rainfall.js` (Plotly chart, metrics, controls)
- Optical API + `optical.js` (Leaflet map, tile loading, layer toggle)
- **Testable output:** Two pages fully functional with real data

### Sub-Project 3: SAR + Classifier Pages
- `services/sar_engine.py`, `services/rf_engine.py`, `services/normalization.py`
- SAR API + `sar.js` (Leaflet map, threshold control, Otsu)
- Classifier API + `classifier.js` (form, training, results, confusion matrix)
- Classifier leaderboard API
- **Testable output:** Two more pages functional, leaderboard works

### Sub-Project 4: Flappy Bird Page
- `services/flappy_engine.py`
- Flappy API (train, upload, submit, race)
- `flappy.js` (form, Canvas replay, leaderboard)
- Flappy leaderboard API
- **Testable output:** Full Flappy Bird competition page with race replays

Each sub-project follows its own spec → plan → implementation cycle. Sub-Project 1 must be completed first. Sub-Projects 2, 3, 4 can proceed in any order after that.
