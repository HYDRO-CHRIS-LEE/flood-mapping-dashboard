# AI Flood Classifier Improvement — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix data leakage via event-wise z-score normalization and improve high school student UX with guided feedback, run history comparison, and a fair F1-based JSON leaderboard.

**Architecture:** Two new utility modules (`utils/normalization.py`, `utils/leaderboard.py`) handle data normalization and leaderboard persistence. The existing `modules/module4_rf.py` is updated to use these utilities, replace the test-event selector with a fixed held-out event, add hint generation, and render run history. `utils/data_loader.py` stays as a raw data loader with no validation logic.

**Tech Stack:** Python 3.11+, pandas, scikit-learn, streamlit, plotly, JSON file I/O

**Spec:** `docs/superpowers/specs/2026-03-17-flood-classifier-improvement-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `utils/normalization.py` | `validate_events()` — exclude events with <30 flood or non-flood samples. `normalize_by_event()` — event-wise z-score on all numeric columns except excluded set. |
| `utils/leaderboard.py` | `load()` / `save()` / `add_entry()` / `get_sorted()` — JSON-file-based leaderboard with atomic writes (`os.replace`), team name normalization (`strip().lower()`), per-team best F1 dedup. |
| `tests/test_normalization.py` | Unit tests for normalization and validation. |
| `tests/test_leaderboard.py` | Unit tests for leaderboard CRUD, atomicity, team dedup. |

### Modified Files

| File | Changes |
|------|---------|
| `modules/module4_rf.py` | Remove `get_leaderboard()` / `render_leaderboard()`. Replace test-event multiselect with fixed `HELD_OUT_EVENT`. Wire in normalization pipeline. Add `generate_hints()`. Add run history to session state + render history table. Update leaderboard submit to use `utils/leaderboard.py`. Update `render_leaderboard()` to use F1, show from JSON. |

### Unchanged Files

`app.py`, `utils/data_loader.py`, `utils/styles.py`, `modules/module1_sar.py`, `modules/module2_optical.py`, `modules/module6_gpm.py`

---

## Task 0: Test Infrastructure Setup

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create tests directory and conftest.py**

The project has no test infrastructure. Create a `conftest.py` that adds the project root to `sys.path` so imports like `from utils.normalization import ...` work.

```python
# tests/conftest.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
```

- [ ] **Step 2: Verify pytest discovers the tests directory**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/ --collect-only 2>&1 | head -3`
Expected: `no tests ran` (no test files yet, but no errors)

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "chore: add test infrastructure with conftest.py"
```

---

## Task 1: Event Validation (`utils/normalization.py` — part 1)

**Files:**
- Create: `utils/normalization.py`
- Create: `tests/test_normalization.py`

- [ ] **Step 1: Write failing test for validate_events**

```python
# tests/test_normalization.py
import pandas as pd
import pytest
from utils.normalization import validate_events


def _make_df(events_config: dict) -> pd.DataFrame:
    """Helper: events_config = {"harvey": (n_flood, n_nonflood), ...}"""
    rows = []
    for ev, (nf, nnf) in events_config.items():
        for _ in range(nf):
            rows.append({"event": ev, "label": 1, "SAR_VH": -15.0,
                         "NDWI": 0.1, "elevation": 10.0, "slope": 1.0,
                         "MNDWI": 0.05, "permanent_water": 0})
        for _ in range(nnf):
            rows.append({"event": ev, "label": 0, "SAR_VH": -10.0,
                         "NDWI": -0.2, "elevation": 20.0, "slope": 2.0,
                         "MNDWI": -0.1, "permanent_water": 0})
    return pd.DataFrame(rows)


def test_validate_excludes_low_sample_events():
    df = _make_df({"harvey": (50, 50), "la2025": (3, 327)})
    valid_df, excluded = validate_events(df, min_per_class=30)
    assert "la2025" not in valid_df["event"].values
    assert "harvey" in valid_df["event"].values
    assert len(excluded) == 1
    assert excluded[0][0] == "la2025"


def test_validate_keeps_all_when_above_threshold():
    df = _make_df({"harvey": (50, 50), "pakistan": (40, 40)})
    valid_df, excluded = validate_events(df, min_per_class=30)
    assert set(valid_df["event"].unique()) == {"harvey", "pakistan"}
    assert len(excluded) == 0


def test_validate_returns_reason_string():
    df = _make_df({"harvey": (50, 50), "bad": (5, 100)})
    _, excluded = validate_events(df, min_per_class=30)
    event_name, reason = excluded[0]
    assert event_name == "bad"
    assert "flood" in reason.lower() or "5" in reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_normalization.py -v`
Expected: FAIL (ImportError — module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# utils/normalization.py
"""Event-wise normalization and validation for RF training data."""

import pandas as pd

EXCLUDE_FROM_NORM = {"label", "event", "system:index", ".geo", "permanent_water"}


def validate_events(
    df: pd.DataFrame, min_per_class: int = 30
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Exclude events where flood or non-flood sample count < min_per_class.
    Returns (valid_df, list of (event_name, reason) tuples).
    """
    excluded: list[tuple[str, str]] = []
    keep_events: list[str] = []

    for ev, grp in df.groupby("event"):
        n_flood = int((grp["label"] == 1).sum())
        n_nonflood = int((grp["label"] == 0).sum())
        if n_flood < min_per_class:
            excluded.append((ev, f"Only {n_flood} flood samples (need {min_per_class})"))
        elif n_nonflood < min_per_class:
            excluded.append((ev, f"Only {n_nonflood} non-flood samples (need {min_per_class})"))
        else:
            keep_events.append(ev)

    valid_df = df[df["event"].isin(keep_events)].copy()
    return valid_df, excluded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_normalization.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add utils/normalization.py tests/test_normalization.py
git commit -m "feat: add validate_events for min-sample filtering"
```

---

## Task 2: Event-Wise Z-Score Normalization (`utils/normalization.py` — part 2)

**Files:**
- Modify: `utils/normalization.py`
- Modify: `tests/test_normalization.py`

- [ ] **Step 1: Write failing tests for normalize_by_event**

Add to `tests/test_normalization.py`:

```python
import numpy as np
from utils.normalization import normalize_by_event, EXCLUDE_FROM_NORM


def test_normalize_centers_each_event_to_zero_mean():
    df = _make_df({"harvey": (50, 50), "pakistan": (50, 50)})
    # Give each event distinct elevation distributions
    harvey_mask = df["event"] == "harvey"
    df.loc[harvey_mask, "elevation"] = np.random.RandomState(42).normal(18, 5, harvey_mask.sum())
    df.loc[~harvey_mask, "elevation"] = np.random.RandomState(42).normal(535, 100, (~harvey_mask).sum())

    result = normalize_by_event(df)

    for ev in ["harvey", "pakistan"]:
        ev_elev = result.loc[result["event"] == ev, "elevation"]
        assert abs(ev_elev.mean()) < 0.01, f"{ev} mean should be ~0, got {ev_elev.mean()}"
        assert abs(ev_elev.std() - 1.0) < 0.1, f"{ev} std should be ~1, got {ev_elev.std()}"


def test_normalize_excludes_specified_columns():
    df = _make_df({"harvey": (50, 50)})
    result = normalize_by_event(df)
    # permanent_water should remain 0 (not normalized)
    assert (result["permanent_water"] == 0).all()
    # label should remain 0 or 1
    assert set(result["label"].unique()) <= {0, 1}
    # event should remain string
    assert result["event"].dtype == object


def test_normalize_handles_zero_std_column():
    df = _make_df({"harvey": (50, 50)})
    df["slope"] = 0.0  # constant column -> std = 0
    result = normalize_by_event(df)
    assert (result["slope"] == 0.0).all()


def test_normalize_does_not_modify_original():
    df = _make_df({"harvey": (50, 50)})
    original_elev = df["elevation"].copy()
    _ = normalize_by_event(df)
    pd.testing.assert_series_equal(df["elevation"], original_elev)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_normalization.py -v -k "normalize"`
Expected: FAIL (ImportError — normalize_by_event not found)

- [ ] **Step 3: Write implementation**

Add to `utils/normalization.py`:

```python
def normalize_by_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Event-wise z-score: (value - event_mean) / event_std for all numeric
    columns not in EXCLUDE_FROM_NORM.  Returns a new DataFrame (original is
    not modified).
    """
    result = df.copy()
    numeric_cols = [
        c for c in result.select_dtypes(include="number").columns
        if c not in EXCLUDE_FROM_NORM
    ]

    for col in numeric_cols:
        stats = result.groupby("event")[col].agg(["mean", "std"])
        for ev in stats.index:
            mask = result["event"] == ev
            mean = stats.loc[ev, "mean"]
            std = stats.loc[ev, "std"]
            if std == 0 or pd.isna(std):
                result.loc[mask, col] = 0.0
            else:
                result.loc[mask, col] = (result.loc[mask, col] - mean) / std

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_normalization.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add utils/normalization.py tests/test_normalization.py
git commit -m "feat: add event-wise z-score normalization to prevent data leakage"
```

---

## Task 3: JSON Leaderboard (`utils/leaderboard.py`)

**Files:**
- Create: `utils/leaderboard.py`
- Create: `tests/test_leaderboard.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_leaderboard.py
import json
import os
import pytest
from utils.leaderboard import load_leaderboard, save_leaderboard, add_entry, get_sorted


@pytest.fixture
def lb_path(tmp_path):
    return str(tmp_path / "leaderboard.json")


def test_load_returns_empty_when_file_missing(lb_path):
    result = load_leaderboard(lb_path)
    assert result == {"held_out_event": "", "entries": []}


def test_save_and_load_roundtrip(lb_path):
    data = {"held_out_event": "dubai", "entries": [
        {"team": "Alpha", "f1": 0.8, "accuracy": 0.85,
         "precision": 0.9, "recall": 0.72, "features": ["SAR_VH"],
         "n_trees": 100, "max_depth": 5, "timestamp": "2026-03-17T10:00:00"}
    ]}
    save_leaderboard(lb_path, data)
    loaded = load_leaderboard(lb_path)
    assert loaded == data


def test_save_is_atomic(lb_path):
    """After save, no temp files should remain."""
    data = {"held_out_event": "dubai", "entries": []}
    save_leaderboard(lb_path, data)
    parent = os.path.dirname(lb_path)
    files = os.listdir(parent)
    assert len(files) == 1  # only leaderboard.json


def test_add_entry_keeps_best_f1_per_team(lb_path):
    data = {"held_out_event": "dubai", "entries": []}
    save_leaderboard(lb_path, data)

    add_entry(lb_path, "dubai", team="Team A", f1=0.7, accuracy=0.75,
              precision=0.8, recall=0.65, features=["SAR_VH"],
              n_trees=100, max_depth=5)
    add_entry(lb_path, "dubai", team="Team A", f1=0.85, accuracy=0.88,
              precision=0.9, recall=0.81, features=["SAR_VH", "NDWI"],
              n_trees=200, max_depth=10)

    result = load_leaderboard(lb_path)
    team_a_entries = [e for e in result["entries"] if e["team"] == "Team A"]
    assert len(team_a_entries) == 1
    assert team_a_entries[0]["f1"] == 0.85


def test_add_entry_normalizes_team_name(lb_path):
    data = {"held_out_event": "dubai", "entries": []}
    save_leaderboard(lb_path, data)

    add_entry(lb_path, "dubai", team="Team A", f1=0.7, accuracy=0.75,
              precision=0.8, recall=0.65, features=["SAR_VH"],
              n_trees=100, max_depth=5)
    add_entry(lb_path, "dubai", team="  team a  ", f1=0.9, accuracy=0.92,
              precision=0.93, recall=0.87, features=["SAR_VH", "NDWI"],
              n_trees=200, max_depth=10)

    result = load_leaderboard(lb_path)
    # Should be treated as same team — only 1 entry
    assert len(result["entries"]) == 1
    # Display name preserved from first submission
    assert result["entries"][0]["team"] == "Team A"
    assert result["entries"][0]["f1"] == 0.9


def test_get_sorted_returns_descending_f1(lb_path):
    data = {"held_out_event": "dubai", "entries": [
        {"team": "B", "f1": 0.6, "accuracy": 0.65, "precision": 0.7,
         "recall": 0.55, "features": [], "n_trees": 50, "max_depth": 3,
         "timestamp": ""},
        {"team": "A", "f1": 0.9, "accuracy": 0.92, "precision": 0.93,
         "recall": 0.87, "features": [], "n_trees": 100, "max_depth": 5,
         "timestamp": ""},
    ]}
    save_leaderboard(lb_path, data)
    sorted_entries = get_sorted(lb_path)
    assert sorted_entries[0]["team"] == "A"
    assert sorted_entries[1]["team"] == "B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_leaderboard.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write implementation**

```python
# utils/leaderboard.py
"""JSON-file-based leaderboard with atomic writes and team dedup."""

import json
import os
import tempfile
from datetime import datetime


def load_leaderboard(path: str) -> dict:
    """Load leaderboard JSON. Returns empty structure if file missing."""
    if not os.path.exists(path):
        return {"held_out_event": "", "entries": []}
    with open(path, "r") as f:
        return json.load(f)


def save_leaderboard(path: str, data: dict) -> None:
    """Atomic write: write to temp file, then os.replace."""
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _normalize_team(name: str) -> str:
    return name.strip().lower()


def add_entry(
    path: str,
    held_out_event: str,
    *,
    team: str,
    f1: float,
    accuracy: float,
    precision: float,
    recall: float,
    features: list[str],
    n_trees: int,
    max_depth: int,
) -> None:
    """Add or update a team's entry. Keeps only the best F1 per team."""
    data = load_leaderboard(path)
    data["held_out_event"] = held_out_event

    norm_name = _normalize_team(team)

    existing_idx = None
    for i, entry in enumerate(data["entries"]):
        if _normalize_team(entry["team"]) == norm_name:
            existing_idx = i
            break

    new_entry = {
        "team": team if existing_idx is None else data["entries"][existing_idx]["team"],
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "features": features,
        "n_trees": n_trees,
        "max_depth": max_depth,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if existing_idx is not None:
        if f1 > data["entries"][existing_idx]["f1"]:
            data["entries"][existing_idx] = new_entry
    else:
        data["entries"].append(new_entry)

    save_leaderboard(path, data)


def get_sorted(path: str) -> list[dict]:
    """Return entries sorted by F1 descending."""
    data = load_leaderboard(path)
    return sorted(data["entries"], key=lambda e: e["f1"], reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_leaderboard.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add utils/leaderboard.py tests/test_leaderboard.py
git commit -m "feat: add JSON-based leaderboard with atomic writes and team dedup"
```

---

## Task 4: Update module4_rf.py — Data Pipeline + Fixed Test Event

**Files:**
- Modify: `modules/module4_rf.py:1-80` (imports, constants, train_rf)

- [ ] **Step 1: Update imports and constants**

At top of `modules/module4_rf.py`, replace lines 1-20 (docstring through `get_leaderboard()`) with the code below. **IMPORTANT: Preserve `FEATURE_INFO` dict and `ALL_FEATURES` list (current lines 22-31) — they are used throughout the module.**

```python
"""
AI Flood Classifier — Random Forest with event-wise z-score normalization.
Fixed held-out test event for fair leaderboard competition.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from utils.data_loader import load_all_rf_samples, ALL_EVENTS
from utils.normalization import validate_events, normalize_by_event
from utils.leaderboard import add_entry, get_sorted
from utils.styles import COLORS
import os
import time

# ── Configuration ────────────────────────────────────────────────
HELD_OUT_EVENT = "dubai"
LEADERBOARD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "leaderboard.json"
)
```

Remove the old `get_leaderboard()` function (lines 18-20).

- [ ] **Step 2: Update train_rf to accept a single held-out event string**

Replace existing `train_rf` function. The signature changes: `test_events: list[str]` → `held_out_event: str`:

```python
def event_based_split(df: pd.DataFrame, held_out_event: str):
    test_mask = df["event"] == held_out_event
    return df[~test_mask].copy(), df[test_mask].copy()


def train_rf(df: pd.DataFrame, features: list[str],
             n_trees: int, max_depth: int,
             held_out_event: str, seed: int = 42):
    train_df, test_df = event_based_split(df, held_out_event)

    if len(train_df) == 0 or len(test_df) == 0:
        return None, None

    X_tr = train_df[features].values; y_tr = train_df["label"].values
    X_te = test_df[features].values;  y_te = test_df["label"].values

    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth if max_depth > 0 else None,
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    cm = confusion_matrix(y_te, y_pred)
    metrics = {
        "accuracy":     accuracy_score(y_te, y_pred),
        "precision":    precision_score(y_te, y_pred, zero_division=0),
        "recall":       recall_score(y_te, y_pred, zero_division=0),
        "f1":           f1_score(y_te, y_pred, zero_division=0),
        "n_train":      len(X_tr),
        "n_test":       len(X_te),
        "train_events": [e for e in df["event"].unique() if e != held_out_event],
        "test_event":   held_out_event,
        "cm":           cm.tolist(),
    }
    importance = dict(zip(features, clf.feature_importances_))
    return metrics, importance
```

- [ ] **Step 3: Run the app to verify it loads without error**

Run: `cd /Users/chris/EarthAI && timeout 10 python -c "from modules.module4_rf import train_rf, HELD_OUT_EVENT; print('OK', HELD_OUT_EVENT)"`
Expected: `OK dubai`

- [ ] **Step 4: Commit**

```bash
git add modules/module4_rf.py
git commit -m "feat: update module4 imports, fix held-out event, simplify train_rf"
```

---

## Task 5: Guided Feedback Hints

**Files:**
- Modify: `modules/module4_rf.py` (add `generate_hints` function)

- [ ] **Step 1: Write failing test**

```python
# tests/test_hints.py
from modules.module4_rf import generate_hints


def test_single_feature_hint_is_highest_priority():
    hints = generate_hints(
        metrics={"recall": 0.5, "precision": 0.5, "f1": 0.4},
        features=["SAR_VH"],
        n_trees=100,
    )
    assert len(hints) >= 1
    assert "하나만" in hints[0] or "one" in hints[0].lower()


def test_low_recall_hint():
    hints = generate_hints(
        metrics={"recall": 0.4, "precision": 0.9, "f1": 0.55},
        features=["SAR_VH", "NDWI"],
        n_trees=100,
    )
    assert any("Recall" in h for h in hints)


def test_high_f1_congratulation():
    hints = generate_hints(
        metrics={"recall": 0.9, "precision": 0.9, "f1": 0.9},
        features=["SAR_VH", "NDWI", "elevation"],
        n_trees=100,
    )
    assert any("훌륭" in h for h in hints)


def test_max_two_hints():
    hints = generate_hints(
        metrics={"recall": 0.3, "precision": 0.3, "f1": 0.3},
        features=["SAR_VH"],
        n_trees=20,
    )
    assert len(hints) <= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_hints.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write implementation**

Add to `modules/module4_rf.py` after `train_rf`:

```python
def generate_hints(metrics: dict, features: list[str], n_trees: int) -> list[str]:
    """Return up to 2 prioritized hints based on model results."""
    rules = [
        (
            len(features) == 1,
            "피처를 하나만 쓰고 있어요. 서로 다른 종류의 정보(레이더 + 지형 등)를 조합해보세요.",
        ),
        (
            metrics["recall"] < 0.6,
            "Recall이 낮습니다 — 실제 홍수 지역을 많이 놓치고 있어요. "
            "SAR_VH 피처가 빠져 있다면 추가해보세요.",
        ),
        (
            metrics["precision"] < 0.6,
            "Precision이 낮습니다 — 홍수가 아닌 곳을 홍수로 잘못 예측하고 있어요. "
            "elevation이나 slope를 추가하면 도움이 될 수 있어요.",
        ),
        (
            n_trees < 30 and metrics["f1"] < 0.7,
            "트리 수가 적습니다. 50~100으로 늘려보세요.",
        ),
        (
            metrics["f1"] > 0.85,
            "훌륭합니다! 피처 수를 줄여도 비슷한 성능이 나오는지 실험해보세요 — "
            "적은 데이터로 같은 결과를 내는 것이 더 좋은 모델입니다.",
        ),
    ]
    return [msg for cond, msg in rules if cond][:2]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/test_hints.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add modules/module4_rf.py tests/test_hints.py
git commit -m "feat: add priority-based guided feedback hints"
```

---

## Task 6: Rewrite render_module4 — Left Panel (Controls)

**Files:**
- Modify: `modules/module4_rf.py:181-305` (render_module4 left column)

This task replaces the data loading section and left control panel. The test-event multiselect is removed and replaced with a fixed held-out display.

- [ ] **Step 1: Replace the data loading and left panel section**

Replace `render_module4` from the start through the end of `col_l` (approximately lines 181-305) with:

```python
def render_module4(available_events: list[str]):

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🤖</div>
        <div>
            <div class="section-title">AI Flood Classifier</div>
            <div class="section-desc">
                Train a Random Forest model across multiple flood events —
                tune parameters and compete for the best F1 score.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    held_out_label = ALL_EVENTS.get(HELD_OUT_EVENT, {}).get("label", HELD_OUT_EVENT)
    held_out_year = ALL_EVENTS.get(HELD_OUT_EVENT, {}).get("year", "")

    st.markdown(f"""
    <div class="callout">
        <strong>From Observation to Prediction</strong> &nbsp;
        We combine data from multiple flood events to train an AI model.
        Features are <strong>normalized per-event</strong> so the model learns flood patterns,
        not geography. All teams are evaluated on
        <strong>{held_out_label} ({held_out_year})</strong> — a flood the model has never seen.
    </div>
    """, unsafe_allow_html=True)

    # ── Load + validate + normalize ────────────────────────────────
    with st.spinner("Loading training data from all events..."):
        raw_df = load_all_rf_samples(available_events)

    if raw_df is None or len(raw_df) == 0:
        st.markdown("""
        <div class="callout warn">
            <strong>No training data found</strong><br>
            Run the Colab export notebook for at least one event to generate
            <code>RF_training_samples.csv</code> files.
        </div>
        """, unsafe_allow_html=True)
        return

    drop_cols = [c for c in raw_df.columns if c not in ALL_FEATURES + ["label", "event"]]
    raw_df = raw_df.drop(columns=drop_cols, errors="ignore").dropna()

    valid_df, excluded = validate_events(raw_df)
    if excluded:
        excl_str = ", ".join(
            f"{ALL_EVENTS.get(e, {}).get('label', e)} ({reason})"
            for e, reason in excluded
        )
        st.caption(f"⚠️ Excluded events: {excl_str}")

    if HELD_OUT_EVENT not in valid_df["event"].unique():
        st.error(f"Held-out event '{HELD_OUT_EVENT}' has no valid data. Cannot proceed.")
        return

    df = normalize_by_event(valid_df)

    train_events = [e for e in df["event"].unique() if e != HELD_OUT_EVENT]
    if len(train_events) == 0:
        st.error("No training events available after filtering.")
        return
    if len(train_events) == 1:
        st.warning("⚠️ Training on only 1 event — results may not generalize. "
                    "Add more event data for better performance.")

    # ── Data summary ───────────────────────────────────────────────
    n_flood = int((df["label"] == 1).sum())
    n_nonflood = int((df["label"] == 0).sum())
    n_events = len(df["event"].unique())

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-unit">{n_events} events (normalized)</div>
        </div>
        <div class="metric-card red">
            <div class="metric-label">Flood</div>
            <div class="metric-value">{n_flood:,}</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Non-flood</div>
            <div class="metric-value">{n_nonflood:,}</div>
        </div>
        <div class="metric-card indigo">
            <div class="metric-label">Test Event</div>
            <div class="metric-value" style="font-size:1.1rem">{held_out_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameter panel ────────────────────────────────────────────
    col_l, col_r = st.columns([1, 1.5])

    with col_l:
        with st.container(border=True):
            st.markdown('<div class="control-title">Model Parameters</div>',
                        unsafe_allow_html=True)

            st.markdown("**Select features** (min 2)")
            selected_features = []
            fc = st.columns(2)
            for i, feat in enumerate(ALL_FEATURES):
                icon, short, desc = FEATURE_INFO[feat]
                with fc[i % 2]:
                    if st.checkbox(f"{icon} {short}",
                                   value=(feat in ["SAR_VH", "NDWI", "elevation"]),
                                   help=desc, key=f"feat_{feat}"):
                        selected_features.append(feat)

            st.markdown("---")
            n_trees = st.slider("Number of trees", 10, 300, 100, 10)
            max_depth = st.slider("Max tree depth (0 = unlimited)", 0, 20, 5)

            st.markdown("---")
            st.markdown(f"**Test event:** {held_out_label} ({held_out_year})")
            st.caption("Fixed for fair competition — all teams evaluated on the same event.")

        disabled = len(selected_features) < 2
        if disabled:
            st.warning("Select at least 2 features.")

        run = st.button("🚀 Train AI Model!", disabled=disabled,
                        use_container_width=True)
```

- [ ] **Step 2: Verify imports resolve**

Run: `cd /Users/chris/EarthAI && python -c "from modules.module4_rf import render_module4; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add modules/module4_rf.py
git commit -m "feat: rewrite module4 left panel with normalization pipeline and fixed test event"
```

---

## Task 7: Rewrite render_module4 — Right Panel + Leaderboard Renderer

**Files:**
- Modify: `modules/module4_rf.py:307-451` (render_module4 right column through end)

**IMPORTANT:** Preserve `render_feature_importance()` and `render_confusion_matrix()` functions — they are called from this task's code. Also remove old `render_leaderboard(board: list)` in this same task.

- [ ] **Step 1: Add render_leaderboard_from_json function**

Replace old `render_leaderboard(board: list)` function with:

```python
def render_leaderboard_from_json():
    """Render leaderboard from JSON file, sorted by F1."""
    entries = get_sorted(LEADERBOARD_PATH)
    if not entries:
        st.markdown(
            '<div class="lb-wrap">'
            '<div class="lb-head">🏆 Team Leaderboard (F1 Score)</div>'
            '<div style="padding:12px 16px;font-size:13px;color:#6b7280">'
            'No submissions yet — train a model and hit <b>Submit</b>!'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    best = entries[0]["f1"]
    icons = ["🥇", "🥈", "🥉"]
    cls_ = ["gold", "silver", "bronze"]

    rows = ""
    for i, e in enumerate(entries[:10]):
        icon = icons[i] if i < 3 else f"#{i+1}"
        cls = cls_[i] if i < 3 else ""
        bw = int(e["f1"] / best * 100) if best > 0 else 0
        feat_short = ", ".join(e.get("features", []))
        rows += (
            f'<div class="lb-row">'
            f'<div class="lb-rank {cls}">{icon}</div>'
            f'<div class="lb-team">{e["team"]}</div>'
            f'<div class="lb-event">{feat_short}</div>'
            f'<div class="lb-bar-wrap"><div class="lb-bar-bg">'
            f'<div class="lb-bar-fill" style="width:{bw}%"></div>'
            f'</div></div>'
            f'<div class="lb-score">{e["f1"]*100:.1f}%</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="lb-wrap">'
        f'<div class="lb-head">🏆 Team Leaderboard (F1 Score)</div>'
        f'{rows}</div>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Replace the right column and everything below**

Replace from `with col_r:` through end of `render_module4` with:

```python
    with col_r:
        if run and not disabled:
            with st.spinner(f"Training Random Forest on {len(train_events)} events..."):
                t0 = time.time()
                metrics, importance = train_rf(
                    df, selected_features, n_trees, max_depth, HELD_OUT_EVENT
                )
                elapsed = time.time() - t0

            if metrics is None:
                st.error("Training failed — check that test event has data.")
            else:
                result = {
                    "metrics": metrics, "importance": importance,
                    "features": selected_features, "n_trees": n_trees,
                    "max_depth": max_depth, "elapsed": elapsed,
                }
                st.session_state["last_result"] = result

                # ── Append to run history ──────────────────────────
                if "run_history" not in st.session_state:
                    st.session_state["run_history"] = []
                history = st.session_state["run_history"]
                history.append({
                    "run_id": len(history) + 1,
                    "features": selected_features,
                    "n_trees": n_trees,
                    "max_depth": max_depth,
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "timestamp": time.strftime("%H:%M:%S"),
                })
                if len(history) > 10:
                    st.session_state["run_history"] = history[-10:]

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            m = res["metrics"]
            f1_val = m["f1"]

            clr = ("green" if f1_val >= 0.85 else "blue" if f1_val >= 0.70
                   else "yellow" if f1_val >= 0.55 else "red")
            emoj = ("🏆" if f1_val >= 0.85 else "⭐" if f1_val >= 0.70
                    else "📈" if f1_val >= 0.55 else "🔧")

            # ── Score cards ────────────────────────────────────────
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card {clr}">
                    <div class="metric-label">{emoj} F1 Score</div>
                    <div class="metric-value">{f1_val*100:.1f}%</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{m['accuracy']*100:.1f}%</div>
                </div>
                <div class="metric-card green">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{m['precision']*100:.1f}%</div>
                </div>
                <div class="metric-card indigo">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">{m['recall']*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Train/test breakdown ───────────────────────────────
            train_ev_str = ", ".join(
                ALL_EVENTS[e]["label"] for e in m["train_events"] if e in ALL_EVENTS
            )
            test_ev_label = ALL_EVENTS.get(m["test_event"], {}).get("label", m["test_event"])
            st.markdown(f"""
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;
                        padding:8px 12px;background:var(--bg3);border-radius:8px">
                <b>Trained on:</b> {train_ev_str} ({m['n_train']} samples)
                &nbsp;·&nbsp;
                <b>Tested on:</b> {test_ev_label} ({m['n_test']} samples)
                &nbsp;·&nbsp; {res['elapsed']:.1f}s
            </div>
            """, unsafe_allow_html=True)

            # ── Hints ──────────────────────────────────────────────
            hints = generate_hints(m, res["features"], res["n_trees"])
            if hints:
                hint_html = "<br>".join(hints)
                st.markdown(f"""
                <div class="callout">
                    <strong>💡 Tips</strong><br>{hint_html}
                </div>
                """, unsafe_allow_html=True)

            # ── Charts: performance bar + confusion matrix ─────────
            cc1, cc2 = st.columns(2)
            with cc1:
                with st.container(border=True):
                    st.markdown('<div class="card-title">Performance</div>',
                                unsafe_allow_html=True)
                    fig = go.Figure(go.Bar(
                        x=["F1", "Accuracy", "Precision", "Recall"],
                        y=[m["f1"], m["accuracy"], m["precision"], m["recall"]],
                        marker_color=[COLORS["blue"], COLORS["cyan"],
                                      COLORS["green_light"], COLORS["indigo"]],
                        text=[f"{v*100:.1f}%" for v in
                              [m["f1"], m["accuracy"], m["precision"], m["recall"]]],
                        textposition="outside",
                        textfont=dict(size=11, color=COLORS["text_sub"]),
                    ))
                    fig.add_hline(y=0.85, line_dash="dot", line_color=COLORS["yellow"],
                                  annotation_text="  Target 85%",
                                  annotation_font_color=COLORS["yellow"],
                                  annotation_font_size=10)
                    fig.update_layout(
                        plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
                        font=dict(family="Inter"), height=190,
                        margin=dict(l=5, r=5, t=5, b=5),
                        yaxis=dict(range=[0, 1.15], tickformat=".0%",
                                   showgrid=True, gridcolor=COLORS["bg3"],
                                   tickfont=dict(color=COLORS["axis"], size=10)),
                        xaxis=dict(showgrid=False,
                                   tickfont=dict(color=COLORS["text_sub"], size=11)),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    config={"displayModeBar": False})

            with cc2:
                with st.container(border=True):
                    st.markdown('<div class="card-title">Confusion Matrix</div>',
                                unsafe_allow_html=True)
                    render_confusion_matrix(m["cm"])

            # ── Feature importance ─────────────────────────────────
            with st.container(border=True):
                st.markdown(
                    '<div class="card-title">Feature Importance — What the AI relied on most</div>',
                    unsafe_allow_html=True)
                render_feature_importance(res["importance"])

            # ── Submit to leaderboard ──────────────────────────────
            if st.button("📤 Submit to Leaderboard", use_container_width=True):
                team = st.session_state.get("team_name", "Team A")
                add_entry(
                    LEADERBOARD_PATH, HELD_OUT_EVENT,
                    team=team, f1=m["f1"], accuracy=m["accuracy"],
                    precision=m["precision"], recall=m["recall"],
                    features=res["features"], n_trees=res["n_trees"],
                    max_depth=res["max_depth"],
                )
                st.success(f"Submitted! F1: {m['f1']*100:.1f}%")

        else:
            st.markdown("""
            <div style="height:280px;display:flex;align-items:center;justify-content:center;
                        border:0.5px dashed var(--border);border-radius:10px;color:var(--text-muted)">
                <div style="text-align:center">
                    <div style="font-size:2.5rem">🤖</div>
                    <div style="margin-top:8px;font-size:13px">
                        Set parameters on the left and press<br><b>Train AI Model!</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Run History ────────────────────────────────────────────────
    history = st.session_state.get("run_history", [])
    if history:
        st.markdown("---")
        st.markdown("### 📊 Run History")
        rows_html = ""
        for i, run in enumerate(reversed(history)):
            feat_str = ", ".join(
                FEATURE_INFO[f][1] if f in FEATURE_INFO else f
                for f in run["features"]
            )
            f1_pct = f"{run['f1']*100:.1f}%"
            acc_pct = f"{run['accuracy']*100:.1f}%"

            if i < len(history) - 1:
                prev = list(reversed(history))[i + 1]
                delta = run["f1"] - prev["f1"]
                delta_str = f"{delta*100:+.1f}%"
                delta_color = COLORS["green"] if delta > 0 else COLORS["red"] if delta < 0 else COLORS["text_muted"]
                vs_prev = f'<span style="color:{delta_color};font-weight:600">{delta_str}</span>'
            else:
                vs_prev = "—"

            rows_html += (
                f"<tr>"
                f"<td style='text-align:center;font-weight:600'>#{run['run_id']}</td>"
                f"<td>{feat_str}</td>"
                f"<td style='text-align:center'>{run['n_trees']}</td>"
                f"<td style='text-align:center'>{run['max_depth']}</td>"
                f"<td style='text-align:center;font-weight:600'>{f1_pct}</td>"
                f"<td style='text-align:center'>{acc_pct}</td>"
                f"<td style='text-align:center'>{vs_prev}</td>"
                f"</tr>"
            )

        st.markdown(f"""
        <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead>
                <tr style="border-bottom:2px solid var(--border);color:var(--text-muted)">
                    <th style="padding:8px 6px;text-align:center">#</th>
                    <th style="padding:8px 6px">Features</th>
                    <th style="padding:8px 6px;text-align:center">Trees</th>
                    <th style="padding:8px 6px;text-align:center">Depth</th>
                    <th style="padding:8px 6px;text-align:center">F1</th>
                    <th style="padding:8px 6px;text-align:center">Acc</th>
                    <th style="padding:8px 6px;text-align:center">vs Prev</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Leaderboard ────────────────────────────────────────────────
    st.markdown("---")
    render_leaderboard_from_json()

    # ── Tips ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="callout">
        <strong>Think About It</strong> &nbsp;
        What does low Recall mean for flood prediction?
        In a real disaster, would you rather have high Precision or high Recall — and why?
    </div>
    """, unsafe_allow_html=True)
```

- [ ] **Step 2: Commit**

```bash
git add modules/module4_rf.py
git commit -m "feat: rewrite module4 right panel with hints, history, and JSON leaderboard"
```

---

## Task 8: Run All Tests + Smoke Test

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/chris/EarthAI && python -m pytest tests/ -v`
Expected: All tests pass (test_normalization: 7, test_leaderboard: 6, test_hints: 4 = 17 total)

- [ ] **Step 2: Smoke test — import all modified modules**

Run:
```bash
cd /Users/chris/EarthAI && python -c "
from utils.normalization import validate_events, normalize_by_event
from utils.leaderboard import load_leaderboard, save_leaderboard, add_entry, get_sorted
from modules.module4_rf import train_rf, generate_hints, HELD_OUT_EVENT, render_module4
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A && git commit -m "fix: address test/import issues from integration"
```
(Skip if no fixes needed)
