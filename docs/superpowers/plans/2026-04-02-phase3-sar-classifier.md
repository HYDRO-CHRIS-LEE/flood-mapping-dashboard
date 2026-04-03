# Phase 3: SAR + Classifier Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SAR Detection and AI Flood Classifier pages — SAR with threshold controls and Leaflet flood map overlay, Classifier with RF training form, results dashboard, and SQLite-backed leaderboard.

**Architecture:** Port SAR threshold/Otsu logic and RF training pipeline to `backend/services/`. SAR API returns flood metrics + PNG tile. Classifier API accepts hyperparameters, trains RF synchronously, returns metrics. Leaderboard API reads/writes SQLite. Frontend pages render results using Plotly.js charts and Leaflet.js maps, with all form controls in native HTML.

**Tech Stack:** FastAPI, scikit-learn, scikit-image (Otsu), numpy, SQLite3, Plotly.js, Leaflet.js

---

### Task 1: SAR engine service + API

**Files:**
- Create: `backend/services/sar_engine.py`
- Create: `backend/routers/sar.py`
- Modify: `backend/main.py`
- Create: `tests/test_sar_api.py`

- [ ] **Step 1: Create `backend/services/sar_engine.py`**

Port from `modules/module1_sar.py`. Pure functions, no Streamlit:

```python
"""SAR flood detection engine — threshold, Otsu, flood mask, tile generation."""

import numpy as np
from skimage.filters import threshold_otsu
from backend.services.data_loader import load_tif, rgba_to_png_bytes


def compute_otsu(sar_arr: np.ndarray) -> float:
    valid = sar_arr[~np.isnan(sar_arr)].flatten()
    valid = valid[(valid > -35) & (valid < 5)]
    if len(valid) < 100:
        return -16.0
    return float(threshold_otsu(valid))


def apply_threshold(sar: np.ndarray, thr: float, perm: np.ndarray = None) -> np.ndarray:
    flood = (sar < thr).astype(np.uint8)
    if perm is not None:
        if perm.shape != flood.shape:
            from skimage.transform import resize
            perm = resize(perm, flood.shape, order=0, preserve_range=True)
        flood = flood & (~(perm > 0.5)).astype(np.uint8)
    return flood


def compute_metrics(sar_arr: np.ndarray, flood_mask: np.ndarray) -> dict:
    total_valid = int(np.sum(~np.isnan(sar_arr)))
    flood_px = int(np.sum(flood_mask))
    flood_pct = flood_px / total_valid * 100 if total_valid > 0 else 0
    flood_km2 = flood_px * 0.0004
    return {
        "flood_px": flood_px,
        "flood_pct": round(flood_pct, 2),
        "flood_km2": round(flood_km2, 1),
        "total_valid": total_valid,
    }


def compute_histogram(sar_arr: np.ndarray, bins: int = 80) -> dict:
    valid = sar_arr[~np.isnan(sar_arr)].flatten()
    valid = valid[(valid > -35) & (valid < 5)]
    counts, edges = np.histogram(valid, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {"centers": [round(c, 2) for c in centers], "counts": counts.tolist()}


def render_flood_tile(sar_arr: np.ndarray, flood_mask: np.ndarray) -> bytes:
    sar_norm = np.clip((sar_arr - (-25)) / 25, 0, 1)
    gray = (sar_norm * 200).astype(np.uint8)
    r = gray.copy(); g = gray.copy(); b = gray.copy()
    r[flood_mask == 1] = 37
    g[flood_mask == 1] = 99
    b[flood_mask == 1] = 235
    a = np.full_like(r, 220)
    rgba = np.stack([r, g, b, a], axis=-1)
    return rgba_to_png_bytes(rgba)
```

- [ ] **Step 2: Create `backend/routers/sar.py`**

```python
"""SAR Detection API — compute flood mask, metrics, and map tiles."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from backend.config import ALL_EVENTS
from backend.services.data_loader import load_tif, EVENT_BOUNDS, EVENT_CENTERS, EVENT_ZOOM
from backend.services.sar_engine import (
    compute_otsu, apply_threshold, compute_metrics, compute_histogram, render_flood_tile,
)

router = APIRouter(tags=["sar"])


class SarComputeRequest(BaseModel):
    event: str
    threshold: float = -16.0
    remove_permanent: bool = True


@router.post("/sar/compute")
def sar_compute(req: SarComputeRequest):
    if req.event not in ALL_EVENTS:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "EVENT_NOT_FOUND", "message": f"Event '{req.event}' not found",
        })

    sar_data, _ = load_tif(req.event, "SAR_after", max_pixels=512)
    if sar_data is None:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "DATA_NOT_FOUND", "message": f"No SAR data for '{req.event}'",
        })

    sar_arr = sar_data[0]
    perm_arr = None
    if req.remove_permanent:
        perm_data, _ = load_tif(req.event, "JRC_permanent_water", max_pixels=512)
        if perm_data is not None:
            perm_arr = perm_data[0]

    otsu_val = compute_otsu(sar_arr)
    flood_mask = apply_threshold(sar_arr, req.threshold, perm_arr)
    metrics = compute_metrics(sar_arr, flood_mask)
    histogram = compute_histogram(sar_arr)

    bounds = EVENT_BOUNDS.get(req.event)
    center = EVENT_CENTERS.get(req.event)
    zoom = EVENT_ZOOM.get(req.event, 9)

    return {"ok": True, "data": {
        **metrics,
        "otsu_val": round(otsu_val, 1),
        "threshold": req.threshold,
        "histogram": histogram,
        "bounds": [[bounds[1], bounds[0]], [bounds[3], bounds[2]]] if bounds else None,
        "center": center,
        "zoom": zoom,
        "tile_url": f"/api/sar/tile/{req.event}/{req.threshold}/{1 if req.remove_permanent else 0}",
    }}


@router.get("/sar/tile/{event}/{threshold}/{remove_perm}")
def sar_tile(event: str, threshold: float, remove_perm: int):
    sar_data, _ = load_tif(event, "SAR_after", max_pixels=512)
    if sar_data is None:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "DATA_NOT_FOUND", "message": "No SAR data",
        })

    sar_arr = sar_data[0]
    perm_arr = None
    if remove_perm:
        perm_data, _ = load_tif(event, "JRC_permanent_water", max_pixels=512)
        if perm_data is not None:
            perm_arr = perm_data[0]

    flood_mask = apply_threshold(sar_arr, threshold, perm_arr)
    png_bytes = render_flood_tile(sar_arr, flood_mask)
    return Response(content=png_bytes, media_type="image/png")
```

- [ ] **Step 3: Register router in `backend/main.py`**

Add `from backend.routers.sar import router as sar_router` and `app.include_router(sar_router)`.

- [ ] **Step 4: Write tests — `tests/test_sar_api.py`**

```python
"""tests/test_sar_api.py"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_sar_compute():
    resp = client.post("/api/sar/compute", json={"event": "harvey", "threshold": -16.0, "remove_permanent": True})
    if resp.status_code == 404:
        return
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    d = body["data"]
    assert "flood_px" in d
    assert "flood_pct" in d
    assert "flood_km2" in d
    assert "otsu_val" in d
    assert "histogram" in d
    assert "tile_url" in d
    assert "bounds" in d


def test_sar_compute_missing_event():
    resp = client.post("/api/sar/compute", json={"event": "nonexistent", "threshold": -16.0})
    assert resp.status_code == 404


def test_sar_tile_returns_png():
    resp = client.get("/api/sar/tile/harvey/-16.0/1")
    if resp.status_code == 404:
        return
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:4] == b'\x89PNG'
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_sar_api.py -v`

```bash
git add backend/services/sar_engine.py backend/routers/sar.py backend/main.py tests/test_sar_api.py
git commit -m "feat(phase3): add SAR engine service and API with tests"
```

---

### Task 2: RF engine service + normalization service

**Files:**
- Create: `backend/services/normalization.py`
- Create: `backend/services/rf_engine.py`

- [ ] **Step 1: Create `backend/services/normalization.py`**

Port from `utils/normalization.py`:

```python
"""Event-wise z-score normalization — ported from utils/normalization.py."""

import pandas as pd
import numpy as np

EXCLUDE_FROM_NORM = {"label", "event", "system:index", ".geo", "permanent_water"}


def validate_events(df: pd.DataFrame, min_per_class: int = 30):
    excluded = []
    valid_events = []
    for event in df["event"].unique():
        sub = df[df["event"] == event]
        n_flood = int((sub["label"] == 1).sum())
        n_nonflood = int((sub["label"] == 0).sum())
        if n_flood < min_per_class:
            excluded.append((event, f"flood={n_flood} < {min_per_class}"))
        elif n_nonflood < min_per_class:
            excluded.append((event, f"nonflood={n_nonflood} < {min_per_class}"))
        else:
            valid_events.append(event)
    valid_df = df[df["event"].isin(valid_events)].copy()
    return valid_df, excluded


def normalize_by_event(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    num_cols = [c for c in result.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_FROM_NORM]
    for event in result["event"].unique():
        mask = result["event"] == event
        for col in num_cols:
            vals = result.loc[mask, col]
            mean = vals.mean()
            std = vals.std()
            if std == 0 or pd.isna(std):
                result.loc[mask, col] = 0.0
            else:
                result.loc[mask, col] = (vals - mean) / std
    return result
```

- [ ] **Step 2: Create `backend/services/rf_engine.py`**

Port from `modules/module4_rf.py`. Pure functions:

```python
"""Random Forest training engine — ported from modules/module4_rf.py."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from backend.services.data_loader import load_csv
from backend.services.normalization import validate_events, normalize_by_event

HELD_OUT_EVENTS = ["dubai", "germany2021", "libya2023", "china2020"]

ALL_FEATURES = ["NDWI", "MNDWI", "elevation", "slope", "permanent_water"]

FEATURE_INFO = {
    "NDWI":            {"icon": "water_drop",  "short": "NDWI",       "desc": "Optical water index"},
    "MNDWI":           {"icon": "location_city","short": "MNDWI",     "desc": "Modified water index — better in urban areas"},
    "elevation":       {"icon": "landscape",   "short": "Elevation",  "desc": "Height above sea level (m)"},
    "slope":           {"icon": "signal_cellular_alt","short": "Slope","desc": "Terrain steepness"},
    "permanent_water": {"icon": "waves",       "short": "Perm. Water","desc": "JRC permanent water flag"},
}


def load_training_data(available_events: list[str]) -> pd.DataFrame | None:
    frames = []
    for ev in available_events:
        df = load_csv(ev, "RF_training_samples")
        if df is not None:
            df["event"] = ev
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def event_based_split(df: pd.DataFrame, held_out_events: list[str]):
    test_mask = df["event"].isin(held_out_events)
    return df[~test_mask].copy(), df[test_mask].copy()


def apply_preprocessing(df, features, sample_pct, outlier_method, seed=42):
    result = df.copy()
    if sample_pct < 100:
        frac = sample_pct / 100
        result = result.groupby("label", group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=seed)
        ).reset_index(drop=True)
    if outlier_method == "IQR":
        for f in features:
            q1, q3 = result[f].quantile(0.25), result[f].quantile(0.75)
            iqr = q3 - q1
            result = result[(result[f] >= q1 - 1.5 * iqr) & (result[f] <= q3 + 1.5 * iqr)]
    elif outlier_method == "zscore":
        for f in features:
            z = (result[f] - result[f].mean()) / (result[f].std() + 1e-8)
            result = result[np.abs(z) <= 3]
    return result.reset_index(drop=True)


def apply_class_balance(df, method, seed=42):
    if method == "none":
        return df
    flood = df[df["label"] == 1]
    nonflood = df[df["label"] == 0]
    if method == "oversample":
        if len(flood) < len(nonflood) and len(flood) > 0:
            flood = flood.sample(n=len(nonflood), replace=True, random_state=seed)
        elif len(nonflood) < len(flood) and len(nonflood) > 0:
            nonflood = nonflood.sample(n=len(flood), replace=True, random_state=seed)
    elif method == "undersample":
        if len(flood) < len(nonflood):
            nonflood = nonflood.sample(n=len(flood), random_state=seed)
        elif len(nonflood) < len(flood):
            flood = flood.sample(n=len(nonflood), random_state=seed)
    return pd.concat([flood, nonflood]).reset_index(drop=True)


def train_rf(df, features, n_trees, max_depth, held_out_events,
             min_samples_leaf=1, max_features_str="sqrt",
             use_class_weight=False, bootstrap=True,
             scaling="none", balance="none", seed=42):
    train_df, test_df = event_based_split(df, held_out_events)
    if len(train_df) == 0 or len(test_df) == 0:
        return None, None

    train_df = apply_class_balance(train_df, balance, seed)
    X_tr = train_df[features].values
    y_tr = train_df["label"].values
    X_te = test_df[features].values
    y_te = test_df["label"].values

    scaler = None
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    max_feat = None if max_features_str == "all" else max_features_str

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_leaf=min_samples_leaf,
        max_features=max_feat,
        class_weight="balanced" if use_class_weight else None,
        bootstrap=bootstrap,
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    cm = confusion_matrix(y_te, y_pred)
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "train_events": [e for e in df["event"].unique() if e not in held_out_events],
        "test_events": held_out_events,
        "cm": cm.tolist(),
    }
    importance = {f: round(float(v), 4) for f, v in zip(features, clf.feature_importances_)}
    return metrics, importance


def generate_hints(metrics, features, n_trees):
    rules = [
        (len(features) == 1,
         "You're using only one feature. Try combining different types (e.g. radar + terrain)."),
        (metrics["recall"] < 0.6,
         "Recall is low — the model is missing flood areas. Try adding NDWI or MNDWI."),
        (metrics["precision"] < 0.6,
         "Precision is low — false positives. Try adding elevation or slope."),
        (n_trees < 30 and metrics["f1"] < 0.7,
         "Number of trees is low. Try increasing to 50-100."),
        (metrics["f1"] > 0.85,
         "Great job! Try reducing features — same performance with less = better model."),
    ]
    return [msg for cond, msg in rules if cond][:2]
```

- [ ] **Step 3: Commit**

```bash
git add backend/services/normalization.py backend/services/rf_engine.py
git commit -m "feat(phase3): add normalization and RF engine services"
```

---

### Task 3: Classifier + Leaderboard API routers

**Files:**
- Create: `backend/routers/classifier.py`
- Create: `backend/routers/leaderboard.py`
- Modify: `backend/main.py`
- Create: `tests/test_classifier_api.py`

- [ ] **Step 1: Create `backend/routers/classifier.py`**

```python
"""Classifier API — train Random Forest and submit to leaderboard."""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import ALL_EVENTS, DATA_ROOT
from backend.services.rf_engine import (
    load_training_data, apply_preprocessing, train_rf, generate_hints,
    HELD_OUT_EVENTS, ALL_FEATURES, FEATURE_INFO,
    validate_events, normalize_by_event,
)

router = APIRouter(tags=["classifier"])


class TrainRequest(BaseModel):
    features: list[str]
    n_trees: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    bootstrap: bool = True
    class_weight: bool = False
    scaling: str = "none"
    balance: str = "none"
    sample_pct: int = 100
    outlier: str = "none"


class SubmitRequest(BaseModel):
    team_id: str
    team_name: str
    f1: float
    accuracy: float
    precision_val: float
    recall: float
    features: list[str]
    n_trees: int
    max_depth: int
    test_events: str = ""


@router.get("/classifier/features")
def get_features():
    return {"ok": True, "data": {
        "features": ALL_FEATURES,
        "info": FEATURE_INFO,
        "held_out_events": HELD_OUT_EVENTS,
    }}


@router.post("/classifier/train")
def train_classifier(req: TrainRequest):
    if not req.features:
        raise HTTPException(status_code=400, detail={
            "ok": False, "error": "NO_FEATURES", "message": "Select at least one feature",
        })
    for f in req.features:
        if f not in ALL_FEATURES:
            raise HTTPException(status_code=400, detail={
                "ok": False, "error": "INVALID_FEATURE", "message": f"Unknown feature: {f}",
            })

    import os
    available = [k for k in ALL_EVENTS if os.path.isdir(os.path.join(DATA_ROOT, k))]

    raw_df = load_training_data(available)
    if raw_df is None or len(raw_df) == 0:
        raise HTTPException(status_code=404, detail={
            "ok": False, "error": "NO_TRAINING_DATA", "message": "No RF training samples found",
        })

    drop_cols = [c for c in raw_df.columns if c not in ALL_FEATURES + ["label", "event"]]
    raw_df = raw_df.drop(columns=drop_cols, errors="ignore").dropna()

    valid_df, excluded = validate_events(raw_df)
    available_held_out = [e for e in HELD_OUT_EVENTS if e in valid_df["event"].unique()]
    if not available_held_out:
        raise HTTPException(status_code=400, detail={
            "ok": False, "error": "NO_TEST_EVENTS", "message": "No held-out test events have data",
        })

    df = normalize_by_event(valid_df)

    proc_df = apply_preprocessing(df, req.features, req.sample_pct, req.outlier)
    if len(proc_df) < 10:
        raise HTTPException(status_code=400, detail={
            "ok": False, "error": "TOO_FEW_SAMPLES", "message": "Too few samples after preprocessing",
        })

    t0 = time.time()
    metrics, importance = train_rf(
        proc_df, req.features, req.n_trees, req.max_depth,
        available_held_out,
        min_samples_leaf=req.min_samples_leaf,
        max_features_str=req.max_features,
        use_class_weight=req.class_weight,
        bootstrap=req.bootstrap,
        scaling=req.scaling,
        balance=req.balance,
    )
    elapsed = round(time.time() - t0, 2)

    if metrics is None:
        raise HTTPException(status_code=500, detail={
            "ok": False, "error": "TRAINING_FAILED", "message": "Training failed — check data",
        })

    hints = generate_hints(metrics, req.features, req.n_trees)

    return {"ok": True, "data": {
        **metrics,
        "importance": importance,
        "hints": hints,
        "elapsed": elapsed,
        "excluded_events": [{"event": e, "reason": r} for e, r in excluded],
    }}


@router.post("/classifier/submit")
def submit_classifier(req: SubmitRequest):
    import json
    from backend.db import get_connection

    conn = get_connection()
    conn.execute(
        """INSERT INTO classifier_leaderboard
           (team_id, team_name, f1, accuracy, precision_val, recall, features, n_trees, max_depth, test_events)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (req.team_id, req.team_name, req.f1, req.accuracy, req.precision_val,
         req.recall, json.dumps(req.features), req.n_trees, req.max_depth, req.test_events),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "data": {"message": "Submitted successfully"}}
```

- [ ] **Step 2: Create `backend/routers/leaderboard.py`**

```python
"""Leaderboard API — read from SQLite."""

import json
from fastapi import APIRouter
from backend.db import get_connection

router = APIRouter(tags=["leaderboard"])


@router.get("/leaderboard/classifier")
def get_classifier_leaderboard():
    conn = get_connection()
    rows = conn.execute(
        """SELECT team_name, f1, accuracy, precision_val, recall, features, n_trees, max_depth, submitted_at
           FROM classifier_leaderboard
           ORDER BY f1 DESC LIMIT 20"""
    ).fetchall()
    conn.close()

    entries = []
    for r in rows:
        entries.append({
            "team": r["team_name"],
            "f1": r["f1"],
            "accuracy": r["accuracy"],
            "precision": r["precision_val"],
            "recall": r["recall"],
            "features": json.loads(r["features"]) if r["features"] else [],
            "n_trees": r["n_trees"],
            "max_depth": r["max_depth"],
            "submitted_at": r["submitted_at"],
        })

    return {"ok": True, "data": entries}


@router.get("/leaderboard/flappy/{stage_id}")
def get_flappy_leaderboard(stage_id: int):
    conn = get_connection()
    rows = conn.execute(
        """SELECT team_name, avg_score, max_score, survival_steps_avg, passed, race_id, submitted_at
           FROM flappy_leaderboard
           WHERE stage_id = ?
           ORDER BY avg_score DESC LIMIT 20""",
        (stage_id,),
    ).fetchall()
    conn.close()

    entries = []
    for r in rows:
        entries.append({
            "team": r["team_name"],
            "avg_score": r["avg_score"],
            "max_score": r["max_score"],
            "survival_steps_avg": r["survival_steps_avg"],
            "passed": bool(r["passed"]),
            "race_id": r["race_id"],
            "submitted_at": r["submitted_at"],
        })

    return {"ok": True, "data": entries}
```

- [ ] **Step 3: Register both routers in `backend/main.py`**

- [ ] **Step 4: Write tests — `tests/test_classifier_api.py`**

```python
"""tests/test_classifier_api.py"""
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_get_features():
    resp = client.get("/api/classifier/features")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "features" in body["data"]
    assert "NDWI" in body["data"]["features"]


def test_train_classifier():
    resp = client.post("/api/classifier/train", json={
        "features": ["NDWI", "elevation"],
        "n_trees": 10, "max_depth": 3, "sample_pct": 30,
    })
    if resp.status_code == 404:
        return  # no training data
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    d = body["data"]
    assert "f1" in d
    assert "accuracy" in d
    assert "cm" in d
    assert "importance" in d
    assert "hints" in d


def test_train_no_features():
    resp = client.post("/api/classifier/train", json={"features": []})
    assert resp.status_code == 400


def test_classifier_leaderboard():
    resp = client.get("/api/leaderboard/classifier")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert isinstance(resp.json()["data"], list)


def test_flappy_leaderboard():
    resp = client.get("/api/leaderboard/flappy/1")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
```

- [ ] **Step 5: Run tests and commit**

```bash
git add backend/routers/classifier.py backend/routers/leaderboard.py backend/main.py tests/test_classifier_api.py
git commit -m "feat(phase3): add classifier, leaderboard APIs with tests"
```

---

### Task 4: SAR frontend page

**Files:**
- Modify: `frontend/pages/sar.html`
- Create: `frontend/js/sar.js`

- [ ] **Step 1: Create `frontend/js/sar.js`**

Handles: event selection, threshold slider, Otsu button, permanent water toggle, metrics display, Leaflet map with flood overlay, histogram chart.

- [ ] **Step 2: Replace `frontend/pages/sar.html`**

Bento grid layout matching stitch design:
- Col 8: Leaflet map with "LIVE SAR STREAM" badge
- Col 4: Flood Mask Metrics (inundated area, flood %, est. km², Otsu value)
- Col 8: Radar Threshold Controls (slider + Otsu button + remove permanent checkbox) + Plotly histogram
- Col 4: Event selector

- [ ] **Step 3: Commit**

```bash
git add frontend/pages/sar.html frontend/js/sar.js
git commit -m "feat(phase3): add SAR page with Leaflet map and threshold controls"
```

---

### Task 5: Classifier frontend page

**Files:**
- Modify: `frontend/pages/classifier.html`
- Create: `frontend/js/classifier.js`

- [ ] **Step 1: Create `frontend/js/classifier.js`**

Handles: feature checkboxes, hyperparameter form, train button → POST /api/classifier/train → render metrics, confusion matrix (Plotly heatmap), feature importance (Plotly horizontal bars), hints, submit to leaderboard, leaderboard polling every 5s.

- [ ] **Step 2: Replace `frontend/pages/classifier.html`**

Bento grid layout matching stitch design:
- Col 4: Configuration panel (feature checkboxes, preprocessing selects, model param sliders, "Train" button)
- Col 8: Results area (metrics cards, confusion matrix, feature importance bars)
- Full width: Leaderboard table

- [ ] **Step 3: Commit**

```bash
git add frontend/pages/classifier.html frontend/js/classifier.js
git commit -m "feat(phase3): add classifier page with form, results, and leaderboard"
```

---

### Task 6: Run all tests + verify

- [ ] **Step 1: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 2: Manual browser verify**

- `#sar` — event selector, threshold slider, Otsu button, flood map, histogram, metrics
- `#classifier` — feature checkboxes, train button → results + confusion matrix + feature importance + leaderboard

- [ ] **Step 3: Final commit**

```bash
git add -A && git commit -m "fix(phase3): smoke test fixes"
```
