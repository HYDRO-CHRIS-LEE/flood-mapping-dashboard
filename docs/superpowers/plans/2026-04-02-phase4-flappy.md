# Phase 4: Flappy Bird Competition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Flappy Bird Competition page — DQN training, model upload/validation, stage submission, admin-triggered races with Canvas replay, and SQLite-backed leaderboard.

**Architecture:** Port flappy_submission, flappy_eval, flappy_replay utilities to `backend/services/flappy_engine.py`. Create `backend/routers/flappy.py` with endpoints for training, upload, submission, race, stages, and unlock status. Frontend uses the existing Canvas replay viewer (from `static/flappy_race.html`) embedded inline, with form controls for model configuration and stage management.

**Tech Stack:** FastAPI, PyTorch, flappy_bird game environment, Canvas replay viewer, SQLite

---

### Task 1: Flappy engine service

**Files:**
- Create: `backend/services/flappy_engine.py`

Port the core logic from `utils/flappy_submission.py`, `utils/flappy_eval.py`, `utils/flappy_replay.py`. This is a thin wrapper that imports and delegates to those existing modules (they are pure Python, no Streamlit deps). Do NOT rewrite them — import them.

```python
"""Flappy Bird engine — thin wrapper around existing utils."""

import os
import sys
import io
import json
import uuid
from datetime import datetime, timezone

from backend.config import DATA_ROOT, ADMIN_PASSWORD

# Ensure flappy_bird env is importable
_FLAPPY_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flappy_bird")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
if _FLAPPY_ROOT not in sys.path:
    sys.path.insert(0, _FLAPPY_ROOT)

from utils.flappy_submission import (
    validate_and_load,
    save_submission,
    load_submission,
    list_stage_submissions,
    _build_model,
    ALLOWED_ARCHITECTURES,
)
from utils.flappy_eval import STAGES, run_race
from utils.flappy_leaderboard import (
    add_entry as lb_add,
    get_sorted_by_stage,
    team_passed_stage,
)

FLAPPY_LB_PATH = os.path.join(DATA_ROOT, "flappy_leaderboard.json")
_LOCK_PATH = os.path.join(DATA_ROOT, ".flappy_race.lock")


def get_stages():
    return {sid: {**s} for sid, s in STAGES.items()}


def get_unlocked_stages(team_name: str) -> list[int]:
    stage_ids = sorted(STAGES.keys())
    unlocked = [1]
    for sid in stage_ids[1:]:
        if team_passed_stage(FLAPPY_LB_PATH, team_name, sid - 1):
            unlocked.append(sid)
        else:
            break
    return unlocked


def train_demo(arch, lr, batch_size, dropout, gamma, optimizer_id, episodes):
    """Run short DQN training. Returns (best_score, state_dict_bytes, metadata)."""
    import torch
    import torch.nn.functional as F
    import random
    import numpy as np

    from env_flappybird.flappybird_env import FlappyBirdEnv
    from model_dqn.replay_memory import ReplayMemory
    from model_dqn.common import Transition

    model = _build_model(arch, dropout)
    model.train()

    if optimizer_id == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    memory = ReplayMemory(10_000)
    env = FlappyBirdEnv()
    best_score = 0

    for ep in range(episodes):
        obs = env.reset(gap_size=170, is_random_gap=False)
        state = torch.FloatTensor(obs).unsqueeze(0)

        for t in range(2000):
            epsilon = max(0.01, 0.5 * (1 / (ep + 1)))
            if random.random() < epsilon:
                action = random.randrange(2)
            else:
                model.eval()
                with torch.no_grad():
                    q = model(state)
                action = int(q.argmax(dim=1).item())
                model.train()

            obs_next, reward, done, info = env.step(action, gap_size=170, dt=0.05)
            next_state = torch.FloatTensor(obs_next).unsqueeze(0) if not done else None
            memory.push(state, torch.LongTensor([[action]]), next_state, torch.FloatTensor([reward]))
            state = next_state if next_state is not None else state

            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
                non_final_next = torch.cat([s for s in batch.next_state if s is not None]) if any(s is not None for s in batch.next_state) else None

                model.eval()
                q_values = model(state_batch).gather(1, action_batch)
                next_state_values = torch.zeros(batch_size)
                if non_final_next is not None:
                    next_state_values[non_final_mask] = model(non_final_next).max(1)[0].detach()
                model.train()
                expected = reward_batch + gamma * next_state_values
                loss = F.mse_loss(q_values, expected.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        best_score = max(best_score, int(env.score))

    env.close()

    model.eval()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    sd_bytes = buf.getvalue()

    metadata = {
        "team_name": "demo",
        "architecture_id": arch,
        "obs_dim": 4,
        "action_dim": 2,
        "framework": "pytorch",
        "dropout": dropout,
    }
    return best_score, sd_bytes, metadata


def validate_upload(sd_bytes, metadata):
    """Validate uploaded model. Returns (model, metadata, error_str)."""
    return validate_and_load(sd_bytes, metadata)


def submit_model(team_name, stage_id, sd_bytes, metadata):
    """Save submission to disk."""
    return save_submission(DATA_ROOT, stage_id, team_name, sd_bytes, metadata)


def execute_race(stage_id, admin_password):
    """Run official race. Returns (race_result, error_str)."""
    if admin_password != ADMIN_PASSWORD:
        return None, "Invalid admin password"

    if os.path.exists(_LOCK_PATH):
        return None, "A race is already in progress"

    try:
        with open(_LOCK_PATH, "w") as f:
            f.write(str(datetime.now(timezone.utc).isoformat()))

        submissions = list_stage_submissions(DATA_ROOT, stage_id)
        if not submissions:
            return None, f"No submissions for Stage {stage_id}"

        teams = []
        for sub in submissions:
            model, meta, err = load_submission(sub["team_dir"])
            if err is not None:
                continue
            teams.append({
                "team_name": sub["team_name"],
                "model": model,
                "submission_timestamp": sub.get("saved_at", ""),
            })

        if not teams:
            return None, "No valid submissions to race"

        race_result = run_race(teams=teams, stage_id=stage_id, data_root=DATA_ROOT)

        for entry in race_result["results"]["results"]:
            lb_add(
                FLAPPY_LB_PATH,
                team_name=entry["team_name"],
                stage_id=stage_id,
                avg_score=entry["avg_score"],
                max_score=entry["max_score"],
                survival_steps_avg=entry["survival_steps_avg"],
                episode_scores=entry["episode_scores"],
                passed=entry["passed"],
                race_id=race_result["race_id"],
                submission_timestamp=entry.get("submission_timestamp", ""),
                status=entry.get("status", "success"),
            )

        return race_result, None

    except Exception as exc:
        return None, str(exc)
    finally:
        if os.path.exists(_LOCK_PATH):
            os.remove(_LOCK_PATH)
```

- [ ] **Step 1: Create the file above**
- [ ] **Step 2: Verify import**: `python -c "from backend.services.flappy_engine import get_stages, ALLOWED_ARCHITECTURES; print('OK', len(get_stages()))"`
- [ ] **Step 3: Commit**

---

### Task 2: Flappy API router

**Files:**
- Create: `backend/routers/flappy.py`
- Modify: `backend/main.py`
- Create: `tests/test_flappy_api.py`

Endpoints:
- `GET /api/flappy/stages` — return STAGES dict + ALLOWED_ARCHITECTURES
- `GET /api/flappy/unlocked/{team_name}` — return unlocked stage IDs
- `POST /api/flappy/train` — synchronous DQN training, return best_score + model_id
- `POST /api/flappy/upload` — multipart file upload (.pt + .json), validate, return model_id
- `POST /api/flappy/submit` — save submission for a stage
- `POST /api/flappy/race` — admin-only race execution, return results + replay

Model storage: server-side dict keyed by model_id (UUID), stored in app state or module-level dict.

- [ ] **Step 1: Create router, register in main.py, write tests**
- [ ] **Step 2: Run tests, commit**

---

### Task 3: Flappy frontend page

**Files:**
- Modify: `frontend/pages/flappy.html`
- Create: `frontend/js/flappy.js`

The page needs:
1. Hero header ("Agent Registry v4.2")
2. Stage badges row (locked/unlocked/passed)
3. Bento grid:
   - Col 8: Replay viewport (Canvas, embedded from static/flappy_race.html logic)
   - Col 4: Live leaderboard
   - Col 4: Model configuration (architecture, hyperparameters, train/upload/submit)
   - Col 8: Dark panel with environment physics + telemetry
4. Footer stats
5. Admin race section (password-protected)

The Canvas replay viewer code from `static/flappy_race.html` should be embedded directly into the page JS (no iframe needed since we control the full page).

- [ ] **Step 1: Create flappy.js and flappy.html**
- [ ] **Step 2: Commit**

---

### Task 4: Integration test + verify

- [ ] **Step 1: Run all tests**
- [ ] **Step 2: Manual browser verification**
- [ ] **Step 3: Final commit**
