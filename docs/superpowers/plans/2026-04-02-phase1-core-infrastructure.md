# Phase 1: Core Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the FastAPI backend scaffold, SPA frontend shell (sidebar + topbar + hash router), events API, SQLite initialization, and shared response/error conventions — so that the app loads in a browser, navigates between 5 placeholder pages, and serves event data from the API.

**Architecture:** FastAPI serves both the `/api/*` JSON endpoints and the static `frontend/` directory. The frontend is a single `index.html` SPA shell with hash-based routing. `app.js` listens for `hashchange` events, loads page HTML fragments from `pages/*.html`, and calls page-specific `init()` functions. `api.js` centralizes all `fetch()` calls with team_id headers and standardized error handling.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, SQLite3, Tailwind CSS (CDN), Alpine.js (CDN), Material Symbols (Google Fonts), Inter font (Google Fonts)

---

### Task 1: Project scaffold — directories, requirements, entry point

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/main.py`
- Create: `backend/config.py`
- Create: `backend/db.py`
- Create: `backend/routers/__init__.py`
- Create: `backend/services/__init__.py`
- Create: `frontend/css/style.css` (empty placeholder)
- Create: `frontend/js/app.js` (empty placeholder)
- Create: `frontend/js/api.js` (empty placeholder)
- Create: `frontend/pages/.gitkeep`
- Create: `run.py`
- Create: `requirements-web.txt`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p backend/routers backend/services frontend/css frontend/js frontend/pages
touch backend/__init__.py backend/routers/__init__.py backend/services/__init__.py
touch frontend/css/style.css frontend/js/app.js frontend/js/api.js frontend/pages/.gitkeep
```

- [ ] **Step 2: Create `requirements-web.txt`**

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
rasterio>=1.3.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
scikit-image>=0.22.0
matplotlib>=3.8.0
Pillow>=10.0.0
```

- [ ] **Step 3: Create `backend/config.py`**

```python
"""Paths, constants, and event metadata for EarthAI backend."""

import os

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_ROOT, "earthai.db")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# ── All flood events (ported from utils/data_loader.py) ──
ALL_EVENTS = {
    "harvey":        {"label": "Hurricane Harvey",       "year": 2017, "region": "Houston, TX, USA",              "color": "#ef4444"},
    "pakistan":       {"label": "Pakistan Mega Flood",    "year": 2022, "region": "Sindh Province, Pakistan",      "color": "#06b6d4"},
    "dubai":         {"label": "Dubai Flash Flood",      "year": 2024, "region": "UAE",                           "color": "#f59e0b"},
    "myanmar2015":   {"label": "Myanmar Cyclone Komen",  "year": 2015, "region": "Irrawaddy Delta, Myanmar",      "color": "#8b5cf6"},
    "louisiana2016": {"label": "Louisiana Flood",        "year": 2016, "region": "Baton Rouge, LA, USA",          "color": "#f97316"},
    "srilanka2017":  {"label": "Sri Lanka Flood",        "year": 2017, "region": "Southern Sri Lanka",            "color": "#14b8a6"},
    "mozambique2019":{"label": "Cyclone Idai",           "year": 2019, "region": "Beira, Mozambique",             "color": "#0ea5e9"},
    "iran2019":      {"label": "Iran Flood",             "year": 2019, "region": "Khuzestan, Iran",               "color": "#d946ef"},
    "china2020":     {"label": "Yangtze River Flood",    "year": 2020, "region": "Hubei / Poyang Lake, China",    "color": "#f43f5e"},
    "sudan2020":     {"label": "Sudan Flash Flood",      "year": 2020, "region": "Khartoum, Sudan",               "color": "#fb923c"},
    "germany2021":   {"label": "Ahr Valley Flood",       "year": 2021, "region": "Rhineland-Palatinate, Germany", "color": "#a3e635"},
    "nigeria2022":   {"label": "Nigeria Flood",          "year": 2022, "region": "Anambra / Delta State, Nigeria", "color": "#fb7185"},
    "libya2023":     {"label": "Libya Flood (Derna)",    "year": 2023, "region": "Derna, Libya",                  "color": "#c084fc"},
    "somalia2023":   {"label": "Somalia Flood",          "year": 2023, "region": "Hirshabelle, Somalia",          "color": "#fbbf24"},
    "brazil2024":    {"label": "Brazil Rio Grande Flood", "year": 2024, "region": "Rio Grande do Sul, Brazil",    "color": "#4ade80"},
    "valencia2024":  {"label": "Spain Valencia Flood",   "year": 2024, "region": "Valencia, Spain",               "color": "#60a5fa"},
}

# ── Flappy admin ──
ADMIN_PASSWORD = os.getenv("FLAPPY_ADMIN_PASSWORD", "earthai2026")
```

- [ ] **Step 4: Create `backend/db.py`**

```python
"""SQLite database initialization and helpers."""

import sqlite3
import json
import os
from backend.config import DB_PATH, DATA_ROOT


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist, migrate legacy JSON if available."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS classifier_leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            team_name TEXT NOT NULL,
            f1 REAL NOT NULL,
            accuracy REAL,
            precision_val REAL,
            recall REAL,
            features TEXT,
            n_trees INTEGER,
            max_depth INTEGER,
            test_events TEXT,
            submitted_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS flappy_leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id TEXT NOT NULL,
            team_name TEXT NOT NULL,
            stage_id INTEGER NOT NULL,
            avg_score REAL NOT NULL,
            max_score INTEGER,
            survival_steps_avg REAL,
            episode_scores TEXT,
            passed INTEGER DEFAULT 0,
            race_id TEXT,
            submitted_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()

    _migrate_legacy_json(conn)
    conn.close()


def _migrate_legacy_json(conn: sqlite3.Connection):
    """One-time migration from legacy JSON leaderboard files."""
    # Classifier leaderboard
    clf_path = os.path.join(DATA_ROOT, "leaderboard.json")
    if os.path.exists(clf_path):
        cursor = conn.execute("SELECT COUNT(*) FROM classifier_leaderboard")
        if cursor.fetchone()[0] == 0:
            with open(clf_path) as f:
                entries = json.load(f)
            for e in entries:
                conn.execute(
                    """INSERT INTO classifier_leaderboard
                       (team_id, team_name, f1, accuracy, precision_val, recall,
                        features, n_trees, max_depth, test_events)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        e.get("team", "unknown"),
                        e.get("team", "unknown"),
                        e.get("f1", 0),
                        e.get("accuracy", 0),
                        e.get("precision", 0),
                        e.get("recall", 0),
                        json.dumps(e.get("features", [])),
                        e.get("n_trees", 0),
                        e.get("max_depth", 0),
                        e.get("test_events", ""),
                    ),
                )
            conn.commit()

    # Flappy leaderboard
    flappy_path = os.path.join(DATA_ROOT, "flappy_leaderboard.json")
    if os.path.exists(flappy_path):
        cursor = conn.execute("SELECT COUNT(*) FROM flappy_leaderboard")
        if cursor.fetchone()[0] == 0:
            with open(flappy_path) as f:
                entries = json.load(f)
            for e in entries:
                conn.execute(
                    """INSERT INTO flappy_leaderboard
                       (team_id, team_name, stage_id, avg_score, max_score,
                        survival_steps_avg, episode_scores, passed, race_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        e.get("team_name", "unknown"),
                        e.get("team_name", "unknown"),
                        e.get("stage_id", 1),
                        e.get("avg_score", 0),
                        e.get("max_score", 0),
                        e.get("survival_steps_avg", 0),
                        json.dumps(e.get("episode_scores", [])),
                        1 if e.get("passed") else 0,
                        e.get("race_id", ""),
                    ),
                )
            conn.commit()
```

- [ ] **Step 5: Create `backend/main.py`**

```python
"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import FRONTEND_DIR
from backend.db import init_db
from backend.routers import events

app = FastAPI(title="EarthAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(events.router, prefix="/api")

# Initialize database
init_db()

# Serve frontend (must be last — catch-all)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
```

- [ ] **Step 6: Create `run.py`**

```python
"""Entry point — run with: python run.py"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
```

- [ ] **Step 7: Verify backend starts**

Run: `cd /Users/chris/EarthAI && python -c "from backend.main import app; print('OK')"`
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add backend/ frontend/ run.py requirements-web.txt
git commit -m "feat(phase1): scaffold FastAPI backend and frontend directories"
```

---

### Task 2: Events API router

**Files:**
- Create: `backend/routers/events.py`
- Create: `tests/test_events_api.py`

- [ ] **Step 1: Write the test**

```python
"""tests/test_events_api.py"""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_get_events_returns_list():
    resp = client.get("/api/events")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0


def test_get_events_item_shape():
    resp = client.get("/api/events")
    item = resp.json()["data"][0]
    assert "key" in item
    assert "label" in item
    assert "year" in item
    assert "region" in item
    assert "color" in item


def test_get_event_by_key():
    resp = client.get("/api/events/harvey")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["data"]["key"] == "harvey"
    assert body["data"]["label"] == "Hurricane Harvey"


def test_get_event_not_found():
    resp = client.get("/api/events/nonexistent")
    assert resp.status_code == 404
    body = resp.json()
    assert body["ok"] is False
    assert body["error"] == "EVENT_NOT_FOUND"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_events_api.py -v`
Expected: FAIL (events router not yet created)

- [ ] **Step 3: Create `backend/routers/events.py`**

```python
"""Events API — list available flood events and their metadata."""

import os
from fastapi import APIRouter, HTTPException

from backend.config import ALL_EVENTS, DATA_ROOT

router = APIRouter(tags=["events"])


def _available_keys() -> list[str]:
    """Return event keys that have at least one data file in data/."""
    available = []
    for key in ALL_EVENTS:
        d = os.path.join(DATA_ROOT, key)
        if os.path.isdir(d) and len(os.listdir(d)) > 0:
            available.append(key)
    return available


@router.get("/events")
def list_events():
    available = _available_keys()
    data = []
    for key in available:
        ev = ALL_EVENTS[key]
        data.append({
            "key": key,
            "label": ev["label"],
            "year": ev["year"],
            "region": ev["region"],
            "color": ev["color"],
        })
    return {"ok": True, "data": data}


@router.get("/events/{key}")
def get_event(key: str):
    if key not in ALL_EVENTS:
        raise HTTPException(status_code=404, detail={
            "ok": False,
            "error": "EVENT_NOT_FOUND",
            "message": f"Event '{key}' not found",
        })
    ev = ALL_EVENTS[key]
    has_dir = os.path.isdir(os.path.join(DATA_ROOT, key))
    return {"ok": True, "data": {
        "key": key,
        "label": ev["label"],
        "year": ev["year"],
        "region": ev["region"],
        "color": ev["color"],
        "has_data": has_dir,
    }}
```

- [ ] **Step 4: Update `backend/main.py` error handler for consistent JSON errors**

Add at the bottom of `main.py`, before the static mount:

```python
from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": "HTTP_ERROR", "message": str(detail)},
    )
```

Add this import at top of `main.py`:

```python
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_events_api.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add backend/routers/events.py backend/main.py tests/test_events_api.py
git commit -m "feat(phase1): add events API with tests"
```

---

### Task 3: Frontend SPA shell — `index.html`

**Files:**
- Create: `frontend/index.html`

This is the most important frontend file. It contains the sidebar, topbar, profile dropdown, and `<main>` container — taken directly from the stitch HTML designs.

- [ ] **Step 1: Create `frontend/index.html`**

```html
<!DOCTYPE html>
<html class="light" lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>EarthAI — Flood Intelligence</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<script>
tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "surface-container-highest":"#e3e2e7","surface-container-high":"#e9e7ed",
        "surface":"#faf9fe","on-tertiary":"#ffffff","surface-bright":"#faf9fe",
        "on-primary-container":"#fefcff","surface-variant":"#e3e2e7",
        "primary":"#0058bc","on-background":"#1a1b1f","tertiary-container":"#c64f00",
        "inverse-primary":"#adc6ff","on-secondary":"#ffffff",
        "surface-container-low":"#f4f3f8","outline-variant":"#c1c6d7",
        "on-error":"#ffffff","on-surface-variant":"#414755",
        "inverse-surface":"#2f3034","error":"#ba1a1a","surface-dim":"#dad9df",
        "surface-container":"#eeedf3","tertiary":"#9e3d00",
        "surface-container-lowest":"#ffffff","secondary":"#405e96",
        "on-secondary-container":"#2d4c83","primary-container":"#0070eb",
        "on-surface":"#1a1b1f","outline":"#717786","secondary-container":"#a1befd",
        "on-primary":"#ffffff","surface-tint":"#005bc1","error-container":"#ffdad6"
      },
      fontFamily: {"headline":["Inter"],"body":["Inter"],"label":["Inter"]},
      borderRadius: {"DEFAULT":"0.25rem","lg":"0.5rem","xl":"0.75rem","full":"9999px"},
    },
  },
};
</script>
<style>
body{font-family:'Inter',sans-serif;background-color:#faf9fe}
.negative-tracking{letter-spacing:-0.04em}
.material-symbols-outlined{font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24}
.bento-grid{display:grid;grid-template-columns:repeat(12,1fr);gap:1.5rem}
#profile-modal[hidden]{display:none}
#profile-modal:not([hidden]){display:flex}
</style>
<link rel="stylesheet" href="/css/style.css"/>
</head>
<body class="flex min-h-screen text-on-surface" x-data="{
  userName: localStorage.getItem('earthai_name') || 'Student',
  teamId: localStorage.getItem('earthai_team_id') || '',
  showProfile: false,
  tempName: '',
  initTeam() {
    if (!this.teamId) {
      this.teamId = crypto.randomUUID();
      localStorage.setItem('earthai_team_id', this.teamId);
    }
    this.tempName = this.userName;
  },
  saveName() {
    this.userName = this.tempName.trim() || 'Student';
    localStorage.setItem('earthai_name', this.userName);
    this.showProfile = false;
  }
}" x-init="initTeam()">

<!-- ══════ Sidebar ══════ -->
<aside class="fixed left-0 top-0 h-screen w-64 border-r border-white/5 bg-slate-950 backdrop-blur-2xl flex flex-col p-4 z-50">
  <div class="mb-8 px-3">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
        <span class="material-symbols-outlined text-white">water_drop</span>
      </div>
      <div>
        <h1 class="text-white font-black tracking-[-0.04em] text-lg leading-tight">EarthAI</h1>
        <p class="text-slate-400 text-[10px] uppercase tracking-widest font-bold">Flood Intelligence</p>
      </div>
    </div>
  </div>

  <nav class="flex-1 space-y-1" id="sidebar-nav">
    <a href="#rainfall" data-page="rainfall" class="nav-link text-slate-400 flex items-center gap-3 p-3 tracking-[-0.02em] text-sm font-medium hover:bg-white/5 hover:text-white transition-all rounded-2xl">
      <span class="material-symbols-outlined">rainy</span><span>Rainfall Analysis</span>
    </a>
    <a href="#optical" data-page="optical" class="nav-link text-slate-400 flex items-center gap-3 p-3 tracking-[-0.02em] text-sm font-medium hover:bg-white/5 hover:text-white transition-all rounded-2xl">
      <span class="material-symbols-outlined">visibility</span><span>Optical Detection</span>
    </a>
    <a href="#sar" data-page="sar" class="nav-link text-slate-400 flex items-center gap-3 p-3 tracking-[-0.02em] text-sm font-medium hover:bg-white/5 hover:text-white transition-all rounded-2xl">
      <span class="material-symbols-outlined">radar</span><span>SAR Detection</span>
    </a>
    <a href="#classifier" data-page="classifier" class="nav-link text-slate-400 flex items-center gap-3 p-3 tracking-[-0.02em] text-sm font-medium hover:bg-white/5 hover:text-white transition-all rounded-2xl">
      <span class="material-symbols-outlined">psychology</span><span>AI Flood Classifier</span>
    </a>
    <a href="#flappy" data-page="flappy" class="nav-link text-slate-400 flex items-center gap-3 p-3 tracking-[-0.02em] text-sm font-medium hover:bg-white/5 hover:text-white transition-all rounded-2xl">
      <span class="material-symbols-outlined">sports_esports</span><span>Flappy Bird Competition</span>
    </a>
  </nav>

  <div class="pt-4 border-t border-white/10 space-y-1">
    <a class="text-slate-400 flex items-center gap-3 p-3 tracking-[-0.02em] text-sm font-medium hover:bg-white/5 hover:text-white transition-all" href="#">
      <span class="material-symbols-outlined">help</span><span>Support</span>
    </a>
    <a class="text-slate-400 flex items-center gap-3 p-3 tracking-[-0.02em] text-sm font-medium hover:bg-white/5 hover:text-white transition-all" href="#">
      <span class="material-symbols-outlined">check_circle</span><span>System Status</span>
    </a>
    <button class="w-full mt-4 bg-primary text-white py-3 rounded-xl font-bold text-sm tracking-tight hover:brightness-110 transition-all active:scale-95">
      Export Report
    </button>
  </div>
</aside>

<!-- ══════ Main area ══════ -->
<div class="flex-1 ml-64 min-h-screen bg-surface flex flex-col">

  <!-- Top App Bar -->
  <header class="w-full sticky top-0 z-40 bg-white/80 backdrop-blur-xl flex justify-between items-center px-8 h-16 shadow-sm shadow-blue-900/5">
    <div class="flex items-center gap-6">
      <h2 class="font-bold text-slate-900 text-xl tracking-[-0.04em]" id="topbar-title">EarthAI</h2>
    </div>
    <div class="flex items-center gap-4">
      <div class="relative">
        <span class="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 text-sm">search</span>
        <input class="bg-surface-container-low border-none rounded-full pl-9 pr-4 py-1.5 text-sm w-64 focus:ring-2 focus:ring-primary/20 transition-all" placeholder="Search..." type="text"/>
      </div>
      <button class="p-2 text-slate-500 hover:bg-slate-100/50 rounded-full transition-colors">
        <span class="material-symbols-outlined">notifications</span>
      </button>
      <button class="p-2 text-slate-500 hover:bg-slate-100/50 rounded-full transition-colors">
        <span class="material-symbols-outlined">settings</span>
      </button>
      <div class="flex items-center gap-3 pl-2 border-l border-slate-200 relative">
        <span class="text-sm font-bold text-slate-700" x-text="userName"></span>
        <button @click="showProfile = !showProfile; tempName = userName" class="relative group outline-none">
          <div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-white text-xs font-bold ring-2 ring-white hover:ring-primary/30 transition-all" x-text="userName.charAt(0).toUpperCase()"></div>
        </button>
        <!-- Profile dropdown -->
        <div x-show="showProfile" @click.away="showProfile = false" x-transition
             class="absolute right-0 top-12 w-72 bg-white rounded-2xl shadow-[0_20px_50px_rgba(0,0,0,0.1)] border border-outline-variant/15 overflow-hidden z-50 p-6">
          <h4 class="text-sm font-black uppercase tracking-widest text-slate-400 mb-4">Account Settings</h4>
          <div class="space-y-4">
            <div>
              <label class="block text-[11px] font-bold text-on-surface-variant uppercase mb-1.5">Display Name</label>
              <input x-model="tempName" @keydown.enter="saveName()"
                     class="w-full bg-surface-container-low border border-outline-variant/20 rounded-xl px-4 py-2.5 text-sm font-medium focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
                     placeholder="Enter your name" type="text"/>
            </div>
            <button @click="saveName()" class="w-full bg-slate-900 text-white py-2.5 rounded-xl text-sm font-bold hover:bg-slate-800 transition-colors">Save Changes</button>
          </div>
        </div>
      </div>
    </div>
  </header>

  <!-- Page content area -->
  <main id="page-content" class="flex-1 p-10 max-w-7xl mx-auto w-full">
    <div class="flex items-center justify-center h-64 text-slate-400">
      <span class="material-symbols-outlined text-6xl">hourglass_empty</span>
    </div>
  </main>

  <!-- Footer -->
  <footer class="px-10 pb-6 max-w-7xl mx-auto w-full flex justify-between items-center text-slate-400 text-[10px] font-bold uppercase tracking-widest">
    <div class="flex gap-8">
      <span>Last Synced: 2 mins ago</span>
      <span>Source: EarthAI Cloud</span>
    </div>
    <div class="flex items-center gap-2">
      <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
      <span>Live Satellite Feed Active</span>
    </div>
  </footer>

</div>

<!-- Scripts -->
<script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
<script src="/js/api.js"></script>
<script src="/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify the file serves from FastAPI**

Run: `cd /Users/chris/EarthAI && python -c "
from fastapi.testclient import TestClient
from backend.main import app
c = TestClient(app)
r = c.get('/')
print('status:', r.status_code)
print('has EarthAI:', 'EarthAI' in r.text)
"`
Expected: `status: 200` and `has EarthAI: True`

- [ ] **Step 3: Commit**

```bash
git add frontend/index.html
git commit -m "feat(phase1): add SPA shell with sidebar, topbar, profile dropdown"
```

---

### Task 4: `api.js` — fetch wrapper

**Files:**
- Create: `frontend/js/api.js`

- [ ] **Step 1: Write `frontend/js/api.js`**

```javascript
/**
 * EarthAI API client.
 * Centralizes all fetch() calls with team_id header and error handling.
 */
const API = (() => {
  const BASE = '/api';

  function teamId() {
    return localStorage.getItem('earthai_team_id') || '';
  }

  function teamName() {
    return localStorage.getItem('earthai_name') || 'Student';
  }

  async function request(method, path, body = null) {
    const opts = {
      method,
      headers: {
        'X-Team-Id': teamId(),
        'X-Team-Name': teamName(),
      },
    };
    if (body !== null) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(`${BASE}${path}`, opts);
    const json = await resp.json();
    if (!json.ok) {
      throw new Error(json.message || json.error || 'API error');
    }
    return json.data;
  }

  return {
    get:  (path)       => request('GET', path),
    post: (path, body) => request('POST', path, body),
  };
})();
```

- [ ] **Step 2: Commit**

```bash
git add frontend/js/api.js
git commit -m "feat(phase1): add API fetch wrapper with team headers"
```

---

### Task 5: `app.js` — SPA hash router and page loader

**Files:**
- Create: `frontend/js/app.js`
- Create: `frontend/pages/rainfall.html`
- Create: `frontend/pages/optical.html`
- Create: `frontend/pages/sar.html`
- Create: `frontend/pages/classifier.html`
- Create: `frontend/pages/flappy.html`

- [ ] **Step 1: Write `frontend/js/app.js`**

```javascript
/**
 * EarthAI SPA router.
 * Listens for hashchange, loads page HTML, calls page init().
 */
const App = (() => {
  const PAGES = {
    rainfall:   { title: 'Rainfall Analysis',      icon: 'rainy' },
    optical:    { title: 'Optical Detection',       icon: 'visibility' },
    sar:        { title: 'SAR Detection',           icon: 'radar' },
    classifier: { title: 'AI Flood Classifier',     icon: 'psychology' },
    flappy:     { title: 'Flappy Bird Competition', icon: 'sports_esports' },
  };

  const DEFAULT_PAGE = 'rainfall';
  const pageCache = {};

  function currentPage() {
    const hash = window.location.hash.slice(1);
    return PAGES[hash] ? hash : DEFAULT_PAGE;
  }

  function updateSidebar(page) {
    document.querySelectorAll('#sidebar-nav .nav-link').forEach(link => {
      const p = link.dataset.page;
      if (p === page) {
        link.classList.remove('text-slate-400');
        link.classList.add('bg-white/10', 'text-white');
      } else {
        link.classList.remove('bg-white/10', 'text-white');
        link.classList.add('text-slate-400');
      }
    });
  }

  function updateTopbar(page) {
    const info = PAGES[page];
    document.getElementById('topbar-title').textContent = info ? info.title : 'EarthAI';
  }

  async function loadPage(page) {
    const container = document.getElementById('page-content');

    // Load HTML fragment
    if (!pageCache[page]) {
      try {
        const resp = await fetch(`/pages/${page}.html`);
        if (!resp.ok) throw new Error(`Page not found: ${page}`);
        pageCache[page] = await resp.text();
      } catch (e) {
        container.innerHTML = `
          <div class="flex flex-col items-center justify-center h-64 text-slate-400 gap-4">
            <span class="material-symbols-outlined text-5xl">error</span>
            <p class="text-sm">${e.message}</p>
          </div>`;
        return;
      }
    }

    container.innerHTML = pageCache[page];

    // Call page-specific init if it exists
    const initFn = window[`init_${page}`];
    if (typeof initFn === 'function') {
      initFn();
    }
  }

  async function navigate() {
    const page = currentPage();
    updateSidebar(page);
    updateTopbar(page);
    await loadPage(page);
  }

  // Init
  function init() {
    window.addEventListener('hashchange', navigate);

    // Set initial hash if none
    if (!window.location.hash) {
      window.location.hash = '#' + DEFAULT_PAGE;
    } else {
      navigate();
    }
  }

  // Auto-init when script loads
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { navigate, PAGES };
})();
```

- [ ] **Step 2: Create 5 placeholder page HTML files**

`frontend/pages/rainfall.html`:
```html
<section class="mb-12">
  <div class="flex justify-between items-end">
    <div>
      <span class="text-secondary font-semibold text-sm tracking-widest uppercase mb-2 block">Intelligence Suite</span>
      <h3 class="text-5xl font-black negative-tracking text-on-surface">Rainfall Analysis</h3>
      <p class="text-on-surface-variant mt-3 text-lg font-medium opacity-80">Real-time precipitation monitoring and hydrologic forecasting.</p>
    </div>
  </div>
</section>
<div class="bg-surface-container-lowest rounded-[2rem] p-8 flex items-center justify-center h-64 text-slate-400">
  <p class="text-sm">Rainfall page — Phase 2</p>
</div>
```

`frontend/pages/optical.html`:
```html
<section class="mb-12">
  <div class="flex justify-between items-end">
    <div>
      <span class="text-secondary font-semibold text-sm tracking-widest uppercase mb-2 block">Intelligence Suite</span>
      <h3 class="text-5xl font-black negative-tracking text-on-surface">Temporal Analysis Canvas</h3>
      <p class="text-on-surface-variant mt-3 text-lg font-medium opacity-80">Multispectral satellite imagery analysis for temporal water detection.</p>
    </div>
  </div>
</section>
<div class="bg-surface-container-lowest rounded-[2rem] p-8 flex items-center justify-center h-64 text-slate-400">
  <p class="text-sm">Optical page — Phase 2</p>
</div>
```

`frontend/pages/sar.html`:
```html
<section class="mb-12">
  <div class="flex justify-between items-end">
    <div>
      <span class="text-secondary font-semibold text-sm tracking-widest uppercase mb-2 block">Intelligence Suite</span>
      <h3 class="text-5xl font-black negative-tracking text-on-surface">SAR Detection Analytics</h3>
      <p class="text-on-surface-variant mt-3 text-lg font-medium opacity-80">All-weather radar monitoring of flood progression.</p>
    </div>
  </div>
</section>
<div class="bg-surface-container-lowest rounded-[2rem] p-8 flex items-center justify-center h-64 text-slate-400">
  <p class="text-sm">SAR page — Phase 3</p>
</div>
```

`frontend/pages/classifier.html`:
```html
<section class="mb-12">
  <div class="flex justify-between items-end">
    <div>
      <span class="text-secondary font-semibold text-sm tracking-widest uppercase mb-2 block">Classifier Engine</span>
      <h3 class="text-5xl font-black negative-tracking text-on-surface">AI Flood Classifier</h3>
      <p class="text-on-surface-variant mt-3 text-lg font-medium opacity-80">Multimodal spatial-temporal segmentation workbench.</p>
    </div>
  </div>
</section>
<div class="bg-surface-container-lowest rounded-[2rem] p-8 flex items-center justify-center h-64 text-slate-400">
  <p class="text-sm">Classifier page — Phase 3</p>
</div>
```

`frontend/pages/flappy.html`:
```html
<section class="mb-12">
  <div class="flex justify-between items-end">
    <div>
      <span class="text-secondary font-semibold text-sm tracking-widest uppercase mb-2 block">Agent Registry v4.2</span>
      <h3 class="text-5xl font-black negative-tracking text-on-surface">Flappy Bird Competition</h3>
      <p class="text-on-surface-variant mt-3 text-lg font-medium opacity-80">Reinforcement learning simulation environment.</p>
    </div>
  </div>
</section>
<div class="bg-surface-container-lowest rounded-[2rem] p-8 flex items-center justify-center h-64 text-slate-400">
  <p class="text-sm">Flappy Bird page — Phase 4</p>
</div>
```

- [ ] **Step 3: Commit**

```bash
git add frontend/js/app.js frontend/pages/
git commit -m "feat(phase1): add SPA router and placeholder pages"
```

---

### Task 6: Integration test — full app start + navigation

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""tests/test_integration.py — verify app serves frontend and API together."""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_index_html_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "EarthAI" in resp.text
    assert "sidebar-nav" in resp.text


def test_page_html_served():
    resp = client.get("/pages/rainfall.html")
    assert resp.status_code == 200
    assert "Rainfall Analysis" in resp.text


def test_api_and_frontend_coexist():
    api_resp = client.get("/api/events")
    assert api_resp.status_code == 200
    assert api_resp.json()["ok"] is True

    html_resp = client.get("/")
    assert html_resp.status_code == 200
    assert "<!DOCTYPE html>" in html_resp.text


def test_js_files_served():
    resp = client.get("/js/app.js")
    assert resp.status_code == 200
    assert "App" in resp.text

    resp = client.get("/js/api.js")
    assert resp.status_code == 200
    assert "API" in resp.text
```

- [ ] **Step 2: Run all tests**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_events_api.py tests/test_integration.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(phase1): add integration tests for frontend + API coexistence"
```

---

### Task 7: Manual smoke test — run the app

**Files:** None (verification only)

- [ ] **Step 1: Start the server**

Run: `cd /Users/chris/EarthAI && python run.py`

- [ ] **Step 2: Open http://localhost:8000 in a browser and verify:**

1. Sidebar shows with EarthAI branding and 5 nav links
2. Top bar shows with search, notifications, settings, profile avatar
3. Clicking profile avatar shows dropdown with display name input
4. Saving a name persists it (refresh page, name stays)
5. Clicking each sidebar link changes the URL hash and loads the placeholder page
6. Active sidebar link is highlighted (bg-white/10 text-white)
7. Top bar title updates when switching pages
8. Footer shows "Last Synced" + green dot

- [ ] **Step 3: Verify API works**

Open: `http://localhost:8000/api/events`
Expected: JSON with `{"ok": true, "data": [...]}`

Open: `http://localhost:8000/api/events/harvey`
Expected: JSON with `{"ok": true, "data": {"key": "harvey", ...}}`

Open: `http://localhost:8000/api/events/nonexistent`
Expected: JSON with `{"ok": false, "error": "EVENT_NOT_FOUND", ...}`

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix(phase1): smoke test fixes"
```
