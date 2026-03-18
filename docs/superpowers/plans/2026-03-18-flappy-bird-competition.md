# Flappy Bird AI Competition — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Flappy Bird DQN competition module (module5) to the EarthAI dashboard where students train agents, submit checkpoints, and compete across 5 difficulty stages with real-time multi-bird race replays.

**Architecture:** New Streamlit module (`modules/module5_flappy.py`) with utility modules for submission validation (`utils/flappy_submission.py`), deterministic evaluation (`utils/flappy_eval.py`), replay serialization (`utils/flappy_replay.py`), and leaderboard (`utils/flappy_leaderboard.py`). HTML5 Canvas game engine (`static/flappy_race.html`) renders precomputed race replays. Reuses existing DQN code from `flappy_bird/model_dqn/` and gym environment from `flappy_bird/env_flappybird/`.

**Tech Stack:** Python 3.12, Streamlit, PyTorch, HTML5 Canvas, Vanilla JS

**Spec:** `docs/superpowers/specs/2026-03-18-flappy-bird-competition-design.md` (v2)

---

## File Structure

| File | Responsibility | New/Modify |
|------|----------------|------------|
| `modules/module5_flappy.py` | Main UI: controls, training, submission, admin race trigger | Create |
| `utils/flappy_leaderboard.py` | JSON leaderboard: load/save/sort per stage | Create |
| `utils/flappy_submission.py` | Validate uploads, save/load submissions per team/stage | Create |
| `utils/flappy_eval.py` | Deterministic evaluation engine with fixed seeds | Create |
| `utils/flappy_replay.py` | Serialize evaluation episodes to replay JSON | Create |
| `static/flappy_race.html` | HTML5 Canvas multi-bird race renderer | Create |
| `app.py` | Add nav item + route for module5 | Modify |
| `utils/styles.py` | Add stage badge CSS classes | Modify |
| `data/flappy_leaderboard.json` | Leaderboard data (auto-created) | Auto |
| `data/flappy_submissions/` | Team submission storage (auto-created) | Auto |
| `data/flappy_races/` | Race manifests + results + replays (auto-created) | Auto |

---

## Milestone 1: Minimal Competition Backbone

### Task 1: Create flappy_leaderboard.py

**Files:**
- Create: `utils/flappy_leaderboard.py`

- [ ] **Step 1: Write `utils/flappy_leaderboard.py`**

Follow the pattern from `utils/leaderboard.py` (atomic writes, team dedup). Adapt for stage-based scoring.

```python
"""JSON-file-based Flappy Bird leaderboard with per-stage tracking."""

import json
import os
import tempfile
from datetime import datetime

EMPTY_LEADERBOARD = {
    "competition_version": "flappy_competition_v1",
    "eval_protocol_version": "flappy_eval_v1",
    "entries": [],
}


def load_leaderboard(path: str) -> dict:
    if not os.path.exists(path):
        return {**EMPTY_LEADERBOARD, "entries": []}
    with open(path, "r") as f:
        return json.load(f)


def save_leaderboard(path: str, data: dict) -> None:
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
    *,
    team: str,
    stage_id: int,
    avg_score: float,
    max_score: int,
    survival_steps_avg: float,
    episode_scores: list[float],
    passed: bool,
    race_id: str,
) -> None:
    """Add or update a team's entry for a stage. Keeps best avg_score per team per stage."""
    data = load_leaderboard(path)
    norm = _normalize_team(team)

    existing_idx = None
    for i, e in enumerate(data["entries"]):
        if _normalize_team(e["team"]) == norm and e["stage_id"] == stage_id:
            existing_idx = i
            break

    new_entry = {
        "team": team if existing_idx is None else data["entries"][existing_idx]["team"],
        "stage_id": stage_id,
        "avg_score": avg_score,
        "max_score": max_score,
        "survival_steps_avg": survival_steps_avg,
        "episode_scores": episode_scores,
        "passed": passed,
        "race_id": race_id,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if existing_idx is not None:
        if avg_score > data["entries"][existing_idx]["avg_score"]:
            data["entries"][existing_idx] = new_entry
    else:
        data["entries"].append(new_entry)

    save_leaderboard(path, data)


def get_sorted_by_stage(path: str, stage_id: int) -> list[dict]:
    """Return entries for a stage, sorted by avg_score descending."""
    data = load_leaderboard(path)
    stage_entries = [e for e in data["entries"] if e["stage_id"] == stage_id]
    return sorted(stage_entries, key=lambda e: (
        -e["avg_score"], -e["max_score"], -e["survival_steps_avg"]
    ))


def team_passed_stage(path: str, team: str, stage_id: int) -> bool:
    """Check if a team has passed a specific stage."""
    data = load_leaderboard(path)
    norm = _normalize_team(team)
    for e in data["entries"]:
        if _normalize_team(e["team"]) == norm and e["stage_id"] == stage_id and e["passed"]:
            return True
    return False
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('utils/flappy_leaderboard.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add utils/flappy_leaderboard.py
git commit -m "feat: add flappy bird leaderboard utility with per-stage tracking"
```

---

### Task 2: Create flappy_submission.py

**Files:**
- Create: `utils/flappy_submission.py`

- [ ] **Step 1: Write `utils/flappy_submission.py`**

```python
"""Flappy Bird submission management — save, load, validate team submissions."""

import json
import os
import shutil
from datetime import datetime

import torch
from torch import nn


# Whitelisted architectures: must match model definitions in flappybird_app_lab.py
ALLOWED_ARCHITECTURES = {"model1", "model2", "model3"}
EXPECTED_OBS_DIM = 4
EXPECTED_ACTION_DIM = 2


def _build_model(architecture_id: str, dropout: float = 0.2) -> nn.Sequential:
    """Build a whitelisted architecture by ID. Returns the model structure."""
    if architecture_id == "model1":
        return nn.Sequential(
            nn.Linear(EXPECTED_OBS_DIM, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(8, EXPECTED_ACTION_DIM),
        )
    elif architecture_id == "model2":
        return nn.Sequential(
            nn.Linear(EXPECTED_OBS_DIM, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(8, EXPECTED_ACTION_DIM),
        )
    elif architecture_id == "model3":
        return nn.Sequential(
            nn.Linear(EXPECTED_OBS_DIM, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(16, EXPECTED_ACTION_DIM),
        )
    raise ValueError(f"Unknown architecture: {architecture_id}")


def validate_and_load(state_dict_bytes: bytes, metadata: dict) -> tuple[nn.Module | None, str]:
    """
    Validate a submission. Returns (loaded_model, error_message).
    On success: error_message is empty string.
    On failure: loaded_model is None.
    """
    # 1. Check metadata fields
    arch_id = metadata.get("architecture_id", "")
    if arch_id not in ALLOWED_ARCHITECTURES:
        return None, f"Unknown architecture '{arch_id}'. Allowed: {ALLOWED_ARCHITECTURES}"

    # 2. Build whitelisted model
    try:
        model = _build_model(arch_id)
    except Exception as e:
        return None, f"Failed to build model: {e}"

    # 3. Load state dict (CPU only, weights_only=True for safety)
    try:
        import io
        state_dict = torch.load(io.BytesIO(state_dict_bytes),
                                map_location="cpu", weights_only=True)
    except Exception as e:
        return None, f"Failed to load state_dict: {e}"

    # 4. Check keys match
    model_keys = set(model.state_dict().keys())
    upload_keys = set(state_dict.keys())
    if model_keys != upload_keys:
        missing = model_keys - upload_keys
        extra = upload_keys - model_keys
        return None, f"Key mismatch. Missing: {missing}, Extra: {extra}"

    # 5. Load weights
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        return None, f"Shape mismatch: {e}"

    # 6. Dry-run inference
    try:
        model.eval()
        dummy = torch.randn(1, EXPECTED_OBS_DIM)
        with torch.no_grad():
            out = model(dummy)
        if out.shape != (1, EXPECTED_ACTION_DIM):
            return None, f"Output shape {out.shape}, expected (1, {EXPECTED_ACTION_DIM})"
    except Exception as e:
        return None, f"Inference failed: {e}"

    return model, ""


def get_submissions_dir(data_root: str) -> str:
    return os.path.join(data_root, "flappy_submissions")


def save_submission(
    data_root: str,
    team: str,
    stage_id: int,
    state_dict_bytes: bytes,
    metadata: dict,
) -> str:
    """Save a validated submission. Returns the submission directory path."""
    team_dir = os.path.join(
        get_submissions_dir(data_root), f"stage_{stage_id}", team.strip()
    )
    os.makedirs(team_dir, exist_ok=True)

    # Save state dict
    with open(os.path.join(team_dir, "state_dict.pt"), "wb") as f:
        f.write(state_dict_bytes)

    # Save metadata
    with open(os.path.join(team_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Save submission meta
    sub_meta = {
        "team": team.strip(),
        "stage_id": stage_id,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(team_dir, "submission_meta.json"), "w") as f:
        json.dump(sub_meta, f, indent=2)

    return team_dir


def load_submission(team_dir: str) -> tuple[nn.Module | None, dict, str]:
    """Load a saved submission. Returns (model, metadata, error)."""
    sd_path = os.path.join(team_dir, "state_dict.pt")
    meta_path = os.path.join(team_dir, "metadata.json")

    if not os.path.exists(sd_path) or not os.path.exists(meta_path):
        return None, {}, "Missing state_dict.pt or metadata.json"

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    with open(sd_path, "rb") as f:
        sd_bytes = f.read()

    model, err = validate_and_load(sd_bytes, metadata)
    return model, metadata, err


def list_stage_submissions(data_root: str, stage_id: int) -> list[dict]:
    """List all submissions for a stage."""
    stage_dir = os.path.join(get_submissions_dir(data_root), f"stage_{stage_id}")
    if not os.path.isdir(stage_dir):
        return []

    submissions = []
    for team_name in sorted(os.listdir(stage_dir)):
        team_dir = os.path.join(stage_dir, team_name)
        meta_path = os.path.join(team_dir, "submission_meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["team_dir"] = team_dir
            submissions.append(meta)

    return submissions
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('utils/flappy_submission.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add utils/flappy_submission.py
git commit -m "feat: add flappy bird submission validation and storage utility"
```

---

### Task 3: Create flappy_eval.py

**Files:**
- Create: `utils/flappy_eval.py`

- [ ] **Step 1: Write `utils/flappy_eval.py`**

```python
"""Deterministic Flappy Bird evaluation engine."""

import json
import os
import time
import random as py_random
from datetime import datetime

import numpy as np
import torch

# Ensure flappy_bird package is importable
import sys
_fb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "flappy_bird")
if _fb_root not in sys.path:
    sys.path.insert(0, _fb_root)

import env_flappybird.flappybird_env as f_env

# ── Stage definitions ─────────────────────────────────────────
STAGES = {
    1: {"gap_size": 170, "label": "First Flight",            "pass_avg": 3},
    2: {"gap_size": 155, "label": "Getting Steady",          "pass_avg": 5},
    3: {"gap_size": 140, "label": "Tighter Gaps",            "pass_avg": 10},
    4: {"gap_size": 125, "label": "Under Pressure",          "pass_avg": 15},
    5: {"gap_size": 120, "label": "Survival of the Fittest", "pass_avg": None},
}

# Fixed seed lists per stage (10 seeds each, deterministic)
STAGE_SEEDS = {
    1: [1031, 1049, 1063, 1091, 1103, 1129, 1151, 1181, 1213, 1237],
    2: [2039, 2053, 2069, 2081, 2099, 2111, 2129, 2141, 2153, 2161],
    3: [3037, 3049, 3061, 3079, 3089, 3109, 3119, 3137, 3163, 3181],
    4: [4007, 4019, 4027, 4049, 4057, 4073, 4091, 4099, 4111, 4127],
    5: [5003, 5009, 5021, 5039, 5051, 5059, 5077, 5081, 5099, 5107],
}

EVAL_PROTOCOL_VERSION = "flappy_eval_v1"
ENV_VERSION = "flappy_env_v1"
MAX_STEPS_PER_EPISODE = 100000
NUM_EPISODES = 10


def evaluate_model(
    model: torch.nn.Module,
    stage_id: int,
    record_frames: bool = False,
) -> dict:
    """
    Run deterministic evaluation of a model on a stage.

    Returns dict with:
      - episode_scores, avg_score, max_score, survival_steps_avg, passed
      - frames (list of episode frame lists) if record_frames=True
    """
    stage = STAGES[stage_id]
    gap_size = stage["gap_size"]
    seeds = STAGE_SEEDS[stage_id]
    pass_avg = stage["pass_avg"]

    model.eval()

    env = f_env.FlappyBirdEnv()

    episode_scores = []
    survival_steps = []
    all_frames = [] if record_frames else None

    for seed in seeds:
        # Set seeds for determinism
        py_random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        state = env.reset(gap_size=gap_size, is_random_gap=False)
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        ep_frames = []
        step_count = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            # Greedy action (no exploration)
            with torch.no_grad():
                action = model(state_t).argmax(1).item()

            if record_frames:
                # Record frame data for replay
                pipes = []
                for wall in env.walls:
                    pipes.append({
                        "x": float(wall.pos_x),
                        "gap_y": float(wall.gap_center_y) if hasattr(wall, 'gap_center_y') else 0,
                        "gap_size": gap_size,
                    })
                ep_frames.append({
                    "t": step,
                    "bird_y": float(env.player.pos_y),
                    "alive": True,
                    "score": int(env.score),
                    "action": action,
                    "pipes": pipes,
                })

            obs_next, reward, done, _ = env.step(action, gap_size=gap_size)
            step_count = step + 1

            if done:
                if record_frames:
                    ep_frames.append({
                        "t": step + 1,
                        "bird_y": float(env.player.pos_y),
                        "alive": False,
                        "score": int(env.score),
                        "action": -1,
                        "pipes": pipes,
                    })
                break

            state_t = torch.from_numpy(obs_next).float().unsqueeze(0)

        episode_scores.append(int(env.score))
        survival_steps.append(step_count)

        if record_frames:
            all_frames.append({"seed": seed, "frames": ep_frames})

    env.close()

    avg_score = sum(episode_scores) / len(episode_scores)
    max_score = max(episode_scores)
    survival_avg = sum(survival_steps) / len(survival_steps)
    passed = avg_score >= pass_avg if pass_avg is not None else False

    result = {
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "env_version": ENV_VERSION,
        "stage_id": stage_id,
        "gap_size": gap_size,
        "seed_list": seeds,
        "episode_scores": episode_scores,
        "avg_score": round(avg_score, 2),
        "max_score": max_score,
        "survival_steps_avg": round(survival_avg, 1),
        "passed": passed,
    }

    if record_frames:
        result["episodes"] = all_frames

    return result


def run_race(
    teams: list[dict],
    stage_id: int,
    data_root: str,
) -> dict:
    """
    Run official race for multiple teams.

    teams: list of {"team_name": str, "model": nn.Module}
    Returns race result dict with manifest, per-team results, and replay data.
    """
    race_id = f"race_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_stage{stage_id}"
    race_dir = os.path.join(data_root, "flappy_races", race_id)
    os.makedirs(race_dir, exist_ok=True)

    # Save manifest
    manifest = {
        "race_id": race_id,
        "stage_id": stage_id,
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "teams": [{"team_name": t["team_name"]} for t in teams],
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(race_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Evaluate each team
    team_results = []
    for t in teams:
        try:
            result = evaluate_model(t["model"], stage_id, record_frames=True)
            result["team_name"] = t["team_name"]
            result["status"] = "success"
        except Exception as e:
            result = {
                "team_name": t["team_name"],
                "status": "evaluation_failed",
                "error": str(e),
                "episode_scores": [],
                "avg_score": 0,
                "max_score": 0,
                "survival_steps_avg": 0,
                "passed": False,
            }
        team_results.append(result)

    # Rank by avg_score descending
    ranked = sorted(
        [r for r in team_results if r["status"] == "success"],
        key=lambda r: (-r["avg_score"], -r["max_score"], -r["survival_steps_avg"]),
    )
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    # Save results
    results_data = {"race_id": race_id, "stage_id": stage_id, "results": team_results}
    with open(os.path.join(race_dir, "results.json"), "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    # Build replay JSON (separate file for Canvas)
    replay = build_replay_json(race_id, stage_id, team_results)
    with open(os.path.join(race_dir, "replay.json"), "w") as f:
        json.dump(replay, f, ensure_ascii=False)

    return {
        "race_id": race_id,
        "race_dir": race_dir,
        "manifest": manifest,
        "results": team_results,
        "replay": replay,
    }


def build_replay_json(race_id: str, stage_id: int, team_results: list[dict]) -> dict:
    """Build replay JSON for the Canvas renderer."""
    seeds = STAGE_SEEDS[stage_id]
    episodes = []

    for ep_idx, seed in enumerate(seeds):
        frames = []
        # Find max frame count across teams for this episode
        max_t = 0
        team_episodes = {}
        for tr in team_results:
            if tr["status"] != "success" or "episodes" not in tr:
                continue
            ep_data = tr["episodes"][ep_idx] if ep_idx < len(tr.get("episodes", [])) else None
            if ep_data and ep_data["frames"]:
                team_episodes[tr["team_name"]] = ep_data["frames"]
                max_t = max(max_t, len(ep_data["frames"]))

        # Merge per-frame data from all teams
        for t in range(max_t):
            birds = []
            pipes = []
            for team_name, ep_frames in team_episodes.items():
                if t < len(ep_frames):
                    f = ep_frames[t]
                    birds.append({
                        "team": team_name,
                        "y": f["bird_y"],
                        "alive": f["alive"],
                        "score": f["score"],
                        "action": f.get("action", 0),
                    })
                    if not pipes and f.get("pipes"):
                        pipes = f["pipes"]
                else:
                    # Team already dead
                    last = ep_frames[-1] if ep_frames else {"bird_y": 0, "score": 0}
                    birds.append({
                        "team": team_name,
                        "y": last["bird_y"],
                        "alive": False,
                        "score": last["score"],
                        "action": -1,
                    })

            frames.append({"t": t, "pipes": pipes, "birds": birds})

        episodes.append({"seed": seed, "episode_index": ep_idx, "frames": frames})

    return {
        "race_id": race_id,
        "stage_id": stage_id,
        "gap_size": STAGES[stage_id]["gap_size"],
        "episodes": episodes,
    }
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('utils/flappy_eval.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add utils/flappy_eval.py
git commit -m "feat: add deterministic flappy bird evaluation engine with fixed seeds"
```

---

### Task 4: Add stage badge CSS to styles.py

**Files:**
- Modify: `utils/styles.py`

- [ ] **Step 1: Add stage badge CSS**

Add before the `hr { border-color ... }` line in `utils/styles.py`:

```css
/* ── Flappy Bird stage badges ──────────────────────────── */
.stage-row { display: flex; gap: 8px; margin: 14px 0; flex-wrap: wrap; }
.stage-badge {
    padding: 8px 14px; border-radius: var(--radius);
    font-size: var(--fs-sm) !important; font-weight: 500;
    border: 1px solid var(--border); cursor: default;
    display: flex; align-items: center; gap: 6px;
}
.stage-badge.locked { background: var(--bg3); color: var(--text-muted) !important; opacity: 0.6; }
.stage-badge.available { background: var(--blue-light); color: var(--blue) !important; border-color: var(--blue); }
.stage-badge.passed { background: var(--green-light); color: var(--green) !important; border-color: var(--green); }
.stage-badge.active { background: var(--yellow-light); color: var(--yellow) !important; border-color: var(--yellow); }
```

- [ ] **Step 2: Commit**

```bash
git add utils/styles.py
git commit -m "style: add flappy bird stage badge CSS classes"
```

---

### Task 5: Create module5_flappy.py (page scaffold)

**Files:**
- Create: `modules/module5_flappy.py`

- [ ] **Step 1: Write the module scaffold**

This is the largest file. Create `modules/module5_flappy.py` with the full UI. The module should contain:

1. **Constants**: `STAGES`, `FLAPPY_LB_PATH`, `DATA_ROOT`, `ADMIN_PASSWORD`
2. **`render_stage_badges(team, lb_path)`**: Show 5 stage badges with pass/lock status
3. **`render_controls(col_l)`**: Left column with Step 1-5 controls
4. **`render_replay(col_r, replay_data)`**: Right column embedding Canvas HTML
5. **`render_leaderboard(stage_id)`**: Auto-refreshing leaderboard fragment
6. **`render_module5()`**: Main entry point

Key implementation details:

```python
"""Flappy Bird AI Competition — DQN training, submission, and multi-bird race."""

import streamlit as st
import torch
from torch import nn
import numpy as np
import os
import sys
import json
import time

from utils.flappy_leaderboard import (
    add_entry as lb_add_entry,
    get_sorted_by_stage,
    team_passed_stage,
)
from utils.flappy_submission import (
    validate_and_load,
    save_submission,
    load_submission,
    list_stage_submissions,
    ALLOWED_ARCHITECTURES,
    _build_model,
)
from utils.flappy_eval import STAGES, run_race, evaluate_model
from utils.styles import COLORS

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FLAPPY_LB_PATH = os.path.join(DATA_ROOT, "flappy_leaderboard.json")
ADMIN_PASSWORD = "earthai2026"

# Ensure flappy_bird env is importable
_fb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "flappy_bird")
if _fb_root not in sys.path:
    sys.path.insert(0, _fb_root)


def _get_unlocked_stages(team: str) -> set[int]:
    """Return set of stage IDs the team can access."""
    unlocked = {1}
    for s in range(1, 5):
        if team_passed_stage(FLAPPY_LB_PATH, team, s):
            unlocked.add(s + 1)
    return unlocked


def render_module5():
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🐦</div>
        <div>
            <div class="section-title">Flappy Bird AI Competition</div>
            <div class="section-desc">
                Train a DQN agent to fly through pipes — compete across 5 difficulty stages with your classmates.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    team = st.session_state.get("team_name", "Team A")
    unlocked = _get_unlocked_stages(team)

    # ── Stage badges ──
    badges = ""
    for sid, sdef in STAGES.items():
        if team_passed_stage(FLAPPY_LB_PATH, team, sid):
            cls = "passed"
            icon = "✅"
        elif sid in unlocked:
            cls = "available"
            icon = "🔓"
        else:
            cls = "locked"
            icon = "🔒"
        badges += f'<div class="stage-badge {cls}">{icon} Stage {sid}: {sdef["label"]}</div>'

    st.markdown(f'<div class="stage-row">{badges}</div>', unsafe_allow_html=True)

    # ── Two columns ──
    col_l, col_r = st.columns([1, 1.5])

    with col_l:
        # Step 1: Model Architecture
        with st.container(border=True):
            st.markdown('<div class="control-title">Step 1 · Model Architecture</div>',
                        unsafe_allow_html=True)
            arch = st.selectbox("Network", list(ALLOWED_ARCHITECTURES),
                                help="3 DQN architectures with different layer sizes")

        # Step 2: Hyperparameters
        with st.container(border=True):
            st.markdown('<div class="control-title">Step 2 · Hyperparameters</div>',
                        unsafe_allow_html=True)
            hp_c1, hp_c2 = st.columns(2)
            with hp_c1:
                lr = st.select_slider("Learning Rate",
                                      options=[0.00005, 0.0001, 0.0005, 0.001],
                                      value=0.0001)
                batch_size = st.selectbox("Batch Size", [32, 64, 128], index=1)
                dropout = st.slider("Dropout", 0.1, 0.5, 0.2, 0.05)
            with hp_c2:
                gamma = st.slider("Gamma", 0.90, 0.99, 0.98, 0.01)
                optimizer_id = st.selectbox("Optimizer", ["Adam", "SGD"])
                episodes = st.slider("Episodes", 100, 2000, 300, 100)

        # Step 3: Train or Upload
        with st.container(border=True):
            st.markdown('<div class="control-title">Step 3 · Train or Upload</div>',
                        unsafe_allow_html=True)
            train_tab, upload_tab = st.tabs(["Train (Demo)", "Upload .pt"])

            with train_tab:
                if st.button("🚀 Train DQN (Demo)", use_container_width=True):
                    _run_demo_training(arch, lr, batch_size, dropout, gamma,
                                       optimizer_id, episodes)

            with upload_tab:
                sd_file = st.file_uploader("state_dict.pt", type=["pt"])
                meta_file = st.file_uploader("metadata.json", type=["json"])
                if sd_file and meta_file:
                    meta = json.load(meta_file)
                    model, err = validate_and_load(sd_file.read(), meta)
                    if err:
                        st.error(f"Validation failed: {err}")
                    else:
                        st.success("Model validated successfully!")
                        st.session_state["flappy_model"] = model
                        st.session_state["flappy_model_bytes"] = sd_file.getvalue()
                        st.session_state["flappy_metadata"] = meta

        # Step 4: Submit
        with st.container(border=True):
            st.markdown('<div class="control-title">Step 4 · Submit to Stage</div>',
                        unsafe_allow_html=True)
            available_stages = sorted(unlocked)
            stage_sel = st.selectbox("Stage", available_stages,
                                     format_func=lambda s: f"Stage {s}: {STAGES[s]['label']}")
            has_model = "flappy_model" in st.session_state
            if st.button("📤 Submit to Race!", disabled=not has_model,
                         use_container_width=True):
                sd_bytes = st.session_state.get("flappy_model_bytes", b"")
                meta = st.session_state.get("flappy_metadata", {"architecture_id": arch})
                save_submission(DATA_ROOT, team, stage_sel, sd_bytes, meta)
                st.success(f"Submitted to Stage {stage_sel}!")

        # Step 5: Admin Race
        with st.container(border=True):
            st.markdown('<div class="control-title">Step 5 · Official Race (Admin)</div>',
                        unsafe_allow_html=True)
            admin_pw = st.text_input("Admin password", type="password")
            race_stage = st.selectbox("Race stage", list(STAGES.keys()), key="race_stage",
                                      format_func=lambda s: f"Stage {s}: {STAGES[s]['label']}")
            if st.button("🏁 Start Official Race!", use_container_width=True):
                if admin_pw != ADMIN_PASSWORD:
                    st.error("Wrong admin password.")
                else:
                    _run_official_race(race_stage)

    with col_r:
        # Show replay if available
        if "flappy_replay" in st.session_state:
            _render_replay(st.session_state["flappy_replay"])
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:400px;border:1px dashed var(--border);border-radius:var(--radius);'
                'color:var(--text-muted);font-size:var(--fs-base)">'
                '<div style="text-align:center">'
                '<div style="font-size:2.5rem">🐦</div>'
                '<div style="margin-top:8px">Race replay will appear here</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

    # ── Leaderboard ──
    st.markdown("---")
    _render_leaderboard()


def _run_demo_training(arch, lr, batch_size, dropout, gamma, optimizer_id, episodes):
    """Run short DQN training in-dashboard."""
    import env_flappybird.flappybird_env as f_env
    from model_dqn.replay_memory import ReplayMemory
    from torch import optim
    import random as py_random
    import itertools

    model = _build_model(arch, dropout)
    if optimizer_id == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr)
    else:
        opt = optim.SGD(model.parameters(), lr=lr)

    memory = ReplayMemory(100000)
    env = f_env.FlappyBirdEnv()

    progress = st.progress(0, text="Training...")
    score_text = st.empty()
    best_score = 0

    for ep in range(episodes):
        state = env.reset(gap_size=170, is_random_gap=False)
        state_t = torch.from_numpy(state).float().unsqueeze(0)

        for step in itertools.count(0):
            # ε-greedy
            epsilon = 0.5 * (1 / (ep + 1))
            if epsilon <= np.random.uniform(0, 1):
                model.eval()
                with torch.no_grad():
                    action = model(state_t).argmax(1).view(1, 1)
            else:
                action = torch.LongTensor([[py_random.randrange(2)]])

            obs_next, reward, done, _ = env.step(action.item(), gap_size=170)

            if not done:
                next_t = torch.from_numpy(obs_next).float().unsqueeze(0)
            else:
                next_t = None

            from model_dqn.common import Transition
            memory.push(state_t, action, next_t, torch.FloatTensor([reward]))

            # Experience replay
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                sb = torch.cat(batch.state)
                ab = torch.cat(batch.action)
                rb = torch.cat(batch.reward)
                nf = torch.cat([s for s in batch.next_state if s is not None])
                model.eval()
                sav = model(sb).gather(1, ab)
                nfm = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
                nsv = torch.zeros(batch_size)
                if nf.shape[0] > 0:
                    nsv[nfm] = model(nf).max(1)[0].detach()
                expected = rb + gamma * nsv
                model.train()
                loss = torch.nn.functional.mse_loss(sav, expected.unsqueeze(1))
                opt.zero_grad()
                loss.backward()
                opt.step()

            if done:
                break
            state_t = next_t

        score = int(env.score)
        best_score = max(best_score, score)
        progress.progress((ep + 1) / episodes,
                          text=f"Episode {ep+1}/{episodes} | Score: {score} | Best: {best_score}")
        if (ep + 1) % 50 == 0:
            score_text.caption(f"Episode {ep+1}: score={score}, best={best_score}")

    env.close()
    progress.empty()
    score_text.empty()
    st.success(f"Training complete! Best score: {best_score}")

    # Save to session
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    st.session_state["flappy_model"] = model
    st.session_state["flappy_model_bytes"] = buf.getvalue()
    st.session_state["flappy_metadata"] = {"architecture_id": arch}


def _run_official_race(stage_id: int):
    """Run official race evaluation for all submitted teams."""
    subs = list_stage_submissions(DATA_ROOT, stage_id)
    if not subs:
        st.error("No submissions for this stage.")
        return

    # Check for race lock
    lock_path = os.path.join(DATA_ROOT, "flappy_race.lock")
    if os.path.exists(lock_path):
        st.error("Another race is already running.")
        return

    # Create lock
    with open(lock_path, "w") as f:
        f.write(str(time.time()))

    try:
        teams = []
        for sub in subs:
            model, meta, err = load_submission(sub["team_dir"])
            if err:
                st.warning(f"Skipping {sub['team']}: {err}")
                continue
            teams.append({"team_name": sub["team"], "model": model})

        if not teams:
            st.error("No valid submissions to evaluate.")
            return

        with st.spinner(f"Running official race for Stage {stage_id} ({len(teams)} teams)..."):
            race = run_race(teams, stage_id, DATA_ROOT)

        # Update leaderboard
        for r in race["results"]:
            if r["status"] == "success":
                lb_add_entry(
                    FLAPPY_LB_PATH,
                    team=r["team_name"],
                    stage_id=stage_id,
                    avg_score=r["avg_score"],
                    max_score=r["max_score"],
                    survival_steps_avg=r["survival_steps_avg"],
                    episode_scores=r["episode_scores"],
                    passed=r["passed"],
                    race_id=race["race_id"],
                )

        st.session_state["flappy_replay"] = race["replay"]
        st.success(f"Race complete! {len(teams)} teams evaluated.")
        st.rerun()

    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def _render_replay(replay_data: dict):
    """Render Canvas replay viewer."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "static", "flappy_race.html")
    if not os.path.exists(html_path):
        st.error("Replay viewer not found (static/flappy_race.html)")
        return

    with open(html_path, "r") as f:
        html_template = f.read()

    # Inject replay data as JSON
    replay_json = json.dumps(replay_data)
    html_content = html_template.replace("/*__REPLAY_DATA__*/", replay_json)

    st.components.v1.html(html_content, height=500, scrolling=False)


@st.fragment(run_every=5)
def _render_leaderboard():
    """Auto-refreshing leaderboard."""
    tabs = st.tabs([f"Stage {s}" for s in STAGES])
    for i, (sid, sdef) in enumerate(STAGES.items()):
        with tabs[i]:
            entries = get_sorted_by_stage(FLAPPY_LB_PATH, sid)
            if not entries:
                st.caption("No results yet — submit a model and wait for the official race!")
                continue

            icons = ["🥇", "🥈", "🥉"]
            cls_ = ["gold", "silver", "bronze"]
            best = entries[0]["avg_score"] if entries else 1

            rows = ""
            for j, e in enumerate(entries[:10]):
                icon = icons[j] if j < 3 else f"#{j+1}"
                cls = cls_[j] if j < 3 else ""
                bw = int(e["avg_score"] / best * 100) if best > 0 else 0
                passed = "✅" if e["passed"] else ""
                rows += (
                    f'<div class="lb-row">'
                    f'<div class="lb-rank {cls}">{icon}</div>'
                    f'<div class="lb-team">{e["team_name"]}</div>'
                    f'<div class="lb-bar-wrap"><div class="lb-bar-bg">'
                    f'<div class="lb-bar-fill" style="width:{bw}%"></div>'
                    f'</div></div>'
                    f'<div class="lb-score">{e["avg_score"]:.1f}</div>'
                    f'<div style="width:20px;text-align:center">{passed}</div>'
                    f'</div>'
                )

            st.markdown(
                f'<div class="lb-wrap">'
                f'<div class="lb-head">🏆 Stage {sid}: {sdef["label"]} (Gap {sdef["gap_size"]})</div>'
                f'{rows}</div>',
                unsafe_allow_html=True,
            )
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('modules/module5_flappy.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add modules/module5_flappy.py
git commit -m "feat: add flappy bird competition module with training, submission, and race UI"
```

---

### Task 6: Integrate module5 into app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add import**

Add after the existing module imports (around line 18):

```python
from modules.module5_flappy import render_module5
```

- [ ] **Step 2: Add to PAGES dict**

Add to the PAGES dict (around line 41):

```python
    "flappybird":{"label": "Flappy Bird Competition", "icon": "🐦", "section": 3},
```

- [ ] **Step 3: Add sidebar nav section**

After the AI Flood Classifier sidebar section (around line 122), add:

```python
    # ── 4. Flappy Bird Competition nav ────────────────────────────
    st.markdown('<div class="sidebar-section-label">4. Flappy Bird Competition</div>',
                unsafe_allow_html=True)
    if st.button(f"🐦  Flappy Bird Competition", key="nav_flappy", use_container_width=True):
        st.session_state.active_page = "flappybird"
        st.rerun()
```

- [ ] **Step 4: Add routing**

Add to the page routing section (around line 157):

```python
    elif active == "flappybird":
        render_module5()
```

- [ ] **Step 5: Verify syntax and commit**

Run: `python -c "import ast; ast.parse(open('app.py').read()); print('OK')"`

```bash
git add app.py
git commit -m "feat: integrate flappy bird module into sidebar nav and page routing"
```

---

### Task 7: Create static/flappy_race.html (Canvas renderer)

**Files:**
- Create: `static/flappy_race.html`

- [ ] **Step 1: Create directory and write HTML**

```bash
mkdir -p static
```

Write `static/flappy_race.html` — a self-contained HTML5 Canvas game renderer that:

1. Reads replay data injected via `/*__REPLAY_DATA__*/` placeholder
2. Renders multi-bird race with these visual elements:
   - Dark background (#0e1117)
   - Green pipes (classic Flappy Bird style)
   - Colored birds per team (assign from palette)
   - Team name labels next to birds
   - Score overlay per team
   - Dead bird: red X + fade out
   - Last survivor: crown emoji
   - Episode counter and stage info
3. Animation: plays through episodes sequentially at ~30fps
4. Controls: play/pause button, episode skip

Key structure:

```html
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0e1117; overflow: hidden; }
  canvas { display: block; }
  .controls { position: absolute; bottom: 10px; left: 10px; display: flex; gap: 8px; }
  .controls button {
    background: rgba(255,255,255,0.15); color: #f1f5f9; border: none;
    border-radius: 6px; padding: 6px 14px; font-size: 13px; cursor: pointer;
  }
  .controls button:hover { background: rgba(255,255,255,0.25); }
  .info {
    position: absolute; top: 10px; right: 10px; color: #94a3b8;
    font-family: 'Inter', sans-serif; font-size: 12px; text-align: right;
  }
</style>
</head>
<body>
<canvas id="game"></canvas>
<div class="controls">
  <button id="playPause">⏸ Pause</button>
  <button id="nextEp">⏭ Next Episode</button>
</div>
<div class="info" id="info"></div>

<script>
const REPLAY = /*__REPLAY_DATA__*/ {};
const TEAM_COLORS = ['#2563eb','#ef4444','#16a34a','#d97706','#8b5cf6',
                     '#06b6d4','#f43f5e','#fb923c','#a3e635','#c084fc'];
const PIPE_COLOR = '#22c55e';
const PIPE_WIDTH = 52;
const BIRD_RADIUS = 12;
const CANVAS_W = 800;
const CANVAS_H = 480;
const GROUND_H = 60;
const FPS = 30;

const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
canvas.width = CANVAS_W;
canvas.height = CANVAS_H;

let currentEp = 0;
let currentFrame = 0;
let playing = true;
let animId = null;

// Team color map
const teamColorMap = {};
if (REPLAY.episodes && REPLAY.episodes.length > 0) {
  const firstFrame = REPLAY.episodes[0].frames[0];
  if (firstFrame && firstFrame.birds) {
    firstFrame.birds.forEach((b, i) => {
      teamColorMap[b.team] = TEAM_COLORS[i % TEAM_COLORS.length];
    });
  }
}

function drawBackground() {
  // Sky gradient
  const grad = ctx.createLinearGradient(0, 0, 0, CANVAS_H);
  grad.addColorStop(0, '#1a1f2e');
  grad.addColorStop(1, '#0e1117');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Ground
  ctx.fillStyle = '#1e2533';
  ctx.fillRect(0, CANVAS_H - GROUND_H, CANVAS_W, GROUND_H);
  ctx.strokeStyle = '#2d3748';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, CANVAS_H - GROUND_H);
  ctx.lineTo(CANVAS_W, CANVAS_H - GROUND_H);
  ctx.stroke();
}

function drawPipe(x, gapY, gapSize) {
  const gameH = CANVAS_H - GROUND_H;
  // Scale from env coords to canvas
  const cx = (x / 420) * CANVAS_W;
  const gapTop = gameH - ((gapY + gapSize/2) / 580) * gameH;
  const gapBot = gameH - ((gapY - gapSize/2) / 580) * gameH;

  ctx.fillStyle = PIPE_COLOR;
  // Top pipe
  ctx.fillRect(cx - PIPE_WIDTH/2, 0, PIPE_WIDTH, gapTop);
  ctx.fillStyle = '#15803d';
  ctx.fillRect(cx - PIPE_WIDTH/2 - 3, gapTop - 20, PIPE_WIDTH + 6, 20);

  // Bottom pipe
  ctx.fillStyle = PIPE_COLOR;
  ctx.fillRect(cx - PIPE_WIDTH/2, gapBot, PIPE_WIDTH, gameH - gapBot);
  ctx.fillStyle = '#15803d';
  ctx.fillRect(cx - PIPE_WIDTH/2 - 3, gapBot, PIPE_WIDTH + 6, 20);
}

function drawBird(x, y, team, alive, score, color) {
  const gameH = CANVAS_H - GROUND_H;
  const cx = x;
  const cy = gameH - (y / 580) * gameH;

  if (!alive) {
    // Dead: red X
    ctx.globalAlpha = 0.4;
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy - 8); ctx.lineTo(cx + 8, cy + 8);
    ctx.moveTo(cx + 8, cy - 8); ctx.lineTo(cx - 8, cy + 8);
    ctx.stroke();
    ctx.globalAlpha = 1;
    return;
  }

  // Bird body
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, BIRD_RADIUS, 0, Math.PI * 2);
  ctx.fill();

  // Eye
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(cx + 4, cy - 3, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#000';
  ctx.beginPath();
  ctx.arc(cx + 5, cy - 3, 2, 0, Math.PI * 2);
  ctx.fill();

  // Label
  ctx.fillStyle = color;
  ctx.font = 'bold 11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(team, cx, cy - BIRD_RADIUS - 6);
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.fillText(`${score}`, cx, cy + BIRD_RADIUS + 14);
}

function drawFrame() {
  if (!REPLAY.episodes || REPLAY.episodes.length === 0) return;

  const ep = REPLAY.episodes[currentEp];
  if (!ep || !ep.frames || currentFrame >= ep.frames.length) {
    // Move to next episode
    currentEp++;
    currentFrame = 0;
    if (currentEp >= REPLAY.episodes.length) {
      currentEp = REPLAY.episodes.length - 1;
      playing = false;
      document.getElementById('playPause').textContent = '▶ Play';
    }
    return;
  }

  const frame = ep.frames[currentFrame];
  drawBackground();

  // Draw pipes
  if (frame.pipes) {
    frame.pipes.forEach(p => {
      drawPipe(p.x, p.gap_y, p.gap_size || REPLAY.gap_size || 170);
    });
  }

  // Draw birds at staggered x positions
  const birdXStart = 60;
  const birdXSpacing = 20;
  if (frame.birds) {
    // Sort: alive first
    const sorted = [...frame.birds].sort((a, b) => (b.alive ? 1 : 0) - (a.alive ? 1 : 0));
    sorted.forEach((b, i) => {
      const color = teamColorMap[b.team] || TEAM_COLORS[0];
      const bx = birdXStart + i * birdXSpacing;
      drawBird(bx, b.y, b.team, b.alive, b.score, color);
    });

    // Check if we have a winner (only one alive)
    const alive = frame.birds.filter(b => b.alive);
    if (alive.length === 1 && frame.birds.length > 1) {
      ctx.font = '24px sans-serif';
      ctx.textAlign = 'center';
      const wy = (CANVAS_H - GROUND_H) - (alive[0].y / 580) * (CANVAS_H - GROUND_H);
      ctx.fillText('👑', birdXStart, wy - BIRD_RADIUS - 20);
    }
  }

  // Info overlay
  const infoEl = document.getElementById('info');
  const aliveCount = frame.birds ? frame.birds.filter(b => b.alive).length : 0;
  infoEl.innerHTML = `Stage ${REPLAY.stage_id} · Gap ${REPLAY.gap_size}<br>`
    + `Episode ${currentEp + 1}/${REPLAY.episodes.length}<br>`
    + `Frame ${currentFrame}<br>`
    + `Alive: ${aliveCount}/${frame.birds ? frame.birds.length : 0}`;

  currentFrame++;
}

function gameLoop() {
  if (playing) drawFrame();
  animId = setTimeout(() => requestAnimationFrame(gameLoop), 1000 / FPS);
}

// Controls
document.getElementById('playPause').addEventListener('click', () => {
  playing = !playing;
  document.getElementById('playPause').textContent = playing ? '⏸ Pause' : '▶ Play';
});
document.getElementById('nextEp').addEventListener('click', () => {
  if (currentEp < (REPLAY.episodes ? REPLAY.episodes.length - 1 : 0)) {
    currentEp++;
    currentFrame = 0;
  }
});

// Start
drawBackground();
if (REPLAY.episodes && REPLAY.episodes.length > 0) {
  gameLoop();
} else {
  ctx.fillStyle = '#94a3b8';
  ctx.font = '16px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('No replay data', CANVAS_W / 2, CANVAS_H / 2);
}
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add static/flappy_race.html
git commit -m "feat: add HTML5 Canvas multi-bird race replay viewer"
```

---

### Task 8: Integration test

- [ ] **Step 1: Verify all imports work**

```bash
cd /Users/chris/EarthAI && python -c "
from utils.flappy_leaderboard import load_leaderboard, add_entry, get_sorted_by_stage, team_passed_stage
from utils.flappy_submission import validate_and_load, save_submission, list_stage_submissions, _build_model
from utils.flappy_eval import STAGES, STAGE_SEEDS, evaluate_model
from modules.module5_flappy import render_module5
print('All imports OK')
"
```

- [ ] **Step 2: Verify model architectures build and run inference**

```bash
cd /Users/chris/EarthAI && python -c "
import torch
from utils.flappy_submission import _build_model
for arch in ['model1', 'model2', 'model3']:
    m = _build_model(arch)
    m.eval()
    dummy = torch.randn(1, 4)
    out = m(dummy)
    assert out.shape == (1, 2), f'{arch} output shape {out.shape}'
    print(f'{arch}: OK, output shape {out.shape}')
print('All architectures verified')
"
```

- [ ] **Step 3: Verify env loads**

```bash
cd /Users/chris/EarthAI && python -c "
import sys; sys.path.insert(0, 'flappy_bird')
import env_flappybird.flappybird_env as f_env
env = f_env.FlappyBirdEnv()
state = env.reset(gap_size=170)
print(f'Env state shape: {state.shape}')
obs, reward, done, _ = env.step(0, gap_size=170)
print(f'Step OK: reward={reward}, done={done}')
env.close()
print('Env verified')
"
```

- [ ] **Step 4: Final commit with all fixes**

```bash
git add -A
git commit -m "feat: complete flappy bird competition module (milestone 1-4)"
```

---

## Post-Implementation Notes

### What's included:
- Full page scaffold with 5-stage progression
- DQN training in-dashboard (demo mode, capped episodes)
- Upload validation (state_dict only, 9-point checks)
- Deterministic evaluation engine with fixed seeds per stage
- Multi-bird race execution with manifest/results/replay files
- HTML5 Canvas replay viewer with bird colors, pipes, death effects, crown
- Auto-refreshing leaderboard per stage
- Admin gate with password for official race execution
- Race lock to prevent concurrent execution
- Failure handling (skip failed teams, continue race)

### What may need tuning after first run:
- Canvas pipe/bird coordinate mapping (depends on exact env coordinate system)
- Stage pass thresholds (calibrate with random agent baseline)
- Demo training episode cap (based on server performance)
- Replay frame recording may need adjustment based on env wall/pipe data structure

### Environment dependency note:
The existing `flappy_bird/env_flappybird/flappybird_env.py` uses `pygame`/rendering. For headless evaluation, the env's `render()` method should be a no-op or skipped. The eval engine never calls `env.render()`. If the env requires a display on init, we may need to set `os.environ["SDL_VIDEODRIVER"] = "dummy"` before importing.
