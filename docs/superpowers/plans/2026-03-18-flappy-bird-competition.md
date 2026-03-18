# Flappy Bird AI Competition — Implementation Plan v2.1

**Date:** 2026-03-18
**Goal:** Implement a classroom-ready Flappy Bird DQN competition module for EarthAI with deterministic official evaluation, replay-only browser rendering, per-stage leaderboard tracking, and stable submission/race state management. This plan keeps the modular architecture from the current implementation draft, including `module5_flappy.py`, `flappy_submission.py`, `flappy_eval.py`, `flappy_replay.py`, `flappy_leaderboard.py`, and `static/flappy_race.html`, while correcting contract mismatches and bringing the implementation into line with the v2 design intent.

**Key v2.1 corrections relative to the current plan**
- Use **Python as the sole source of truth** for evaluation, ranking, and leaderboard writes. The browser remains replay-only.
- Fix schema consistency for leaderboard entries and team identifiers.
- Move admin credentials out of code and into secrets/environment config.
- Implement stage difficulty as a real evaluation contract, including **speed**, not just gap size. The design defines stage difficulty using both gap size and speed.
- Enforce `state_dict`-only submission validation with stricter metadata checks.
- Separate **submission state**, **race state**, and **leaderboard state** so Streamlit reruns do not corrupt competition flow. The current plan already introduces separate storage roots for submissions, races, and leaderboard, and v2.1 formalizes that as a hard contract.

---

## 1. Final Architecture Contract

### Files

| File | Responsibility | Status |
|---|---|---|
| `modules/module5_flappy.py` | Main Streamlit page: student controls, upload, submission, admin actions, replay embed | Create |
| `utils/flappy_submission.py` | Validate `state_dict + metadata`, save/load active submissions, list stage submissions | Create |
| `utils/flappy_eval.py` | Deterministic official evaluation engine with fixed seed lists and stage config | Create |
| `utils/flappy_replay.py` | Serialize official evaluation trajectories into replay JSON | Create |
| `utils/flappy_leaderboard.py` | Atomic JSON leaderboard read/write/update/sort | Create |
| `static/flappy_race.html` | HTML5 Canvas renderer for precomputed replay only | Create |
| `app.py` | Add navigation item and route to module 5 | Modify |
| `utils/styles.py` | Add stage badge CSS | Modify |
| `data/flappy_leaderboard.json` | Official leaderboard data | Auto-created |
| `data/flappy_submissions/` | Active team submissions by stage | Auto-created |
| `data/flappy_races/` | Frozen race manifests, results, replay files | Auto-created |

### Source-of-truth rule

Only Python may:
- evaluate models,
- compute official scores,
- decide pass/fail,
- update leaderboard entries,
- generate race results,
- freeze race manifests.

JS may only:
- render replay frames,
- show labels/effects/overlays,
- optionally emit a non-authoritative playback-complete event.

---

## 2. Data Model Contracts

## 2.1 Leaderboard schema

Use a single normalized schema everywhere. Do **not** mix `team` and `team_name`.

```json
{
  "competition_version": "flappy_competition_v1",
  "eval_protocol_version": "flappy_eval_v1",
  "entries": [
    {
      "team_name": "Team Alpha",
      "stage_id": 2,
      "avg_score": 12.1,
      "max_score": 24,
      "survival_steps_avg": 418.6,
      "episode_scores": [10, 12, 9, 14, 13, 15, 8, 11, 17, 12],
      "passed": true,
      "race_id": "race_2026-03-18T18-20-00_stage2",
      "submission_timestamp": "2026-03-18T17:52:11",
      "evaluated_at": "2026-03-18T18:21:02",
      "status": "success"
    }
  ]
}
```

### Policy

* One active official ranking record per `team_name + stage_id` in the main stage view.
* Historical official race results may also be preserved in race artifacts.
* Ranking sort order:

  1. `avg_score`
  2. `max_score`
  3. `survival_steps_avg`
  4. earlier `submission_timestamp`

## 2.2 Submission schema

Each active submission must include:

```json
{
  "team_name": "Team Alpha",
  "architecture_id": "model2",
  "obs_dim": 4,
  "action_dim": 2,
  "framework": "pytorch",
  "training_note": "trained in Colab for 3000 episodes"
}
```

Required fields:

* `team_name`
* `architecture_id`
* `obs_dim`
* `action_dim`
* `framework`

## 2.3 Race manifest schema

```json
{
  "race_id": "race_2026-03-18T18-20-00_stage3",
  "stage_id": 3,
  "eval_protocol_version": "flappy_eval_v1",
  "teams": [
    {
      "team_name": "Team Alpha",
      "submission_path": "data/flappy_submissions/stage_3/Team Alpha"
    }
  ],
  "frozen_at": "2026-03-18T18:20:00"
}
```

Once written, a manifest is immutable.

---

## 3. Stage and Evaluation Protocol

## 3.1 Stage configuration

```python
STAGES = {
    1: {"label": "First Flight",            "gap_size": 170, "speed": "slow",   "pass_avg": 3},
    2: {"label": "Getting Steady",          "gap_size": 155, "speed": "slow",   "pass_avg": 5},
    3: {"label": "Tighter Gaps",            "gap_size": 140, "speed": "normal", "pass_avg": 10},
    4: {"label": "Under Pressure",          "gap_size": 125, "speed": "fast",   "pass_avg": 15},
    5: {"label": "Survival of the Fittest", "gap_size": 120, "speed": "fast",   "pass_avg": None},
}
```

## 3.2 Fixed official seed lists

```python
STAGE_SEEDS = {
    1: [1031, 1049, 1063, 1091, 1103, 1129, 1151, 1181, 1213, 1237],
    2: [2039, 2053, 2069, 2081, 2099, 2111, 2129, 2141, 2153, 2161],
    3: [3037, 3049, 3061, 3079, 3089, 3109, 3119, 3137, 3163, 3181],
    4: [4007, 4019, 4027, 4049, 4057, 4073, 4091, 4099, 4111, 4127],
    5: [5003, 5009, 5021, 5039, 5051, 5059, 5077, 5081, 5099, 5107],
}
```

## 3.3 Official evaluation protocol

For official evaluation:

* `model.eval()`
* greedy action only: `argmax(Q(s))`
* no epsilon exploration
* no dropout sampling
* fixed seed list
* fixed env version
* fixed observation preprocessing
* fixed stage speed
* fixed reward and termination rules

Record this protocol in race results:

```json
{
  "eval_protocol_version": "flappy_eval_v1",
  "env_version": "flappy_env_v1",
  "policy_mode": "greedy_eval",
  "frame_skip": 1,
  "reward_version": "classic_score_reward_v1",
  "termination_version": "pipe_ground_collision_v1"
}
```

---

## 4. Storage Layout

```text
data/
  flappy_leaderboard.json
  flappy_race.lock
  flappy_submissions/
    stage_1/
      Team Alpha/
        state_dict.pt
        metadata.json
        submission_meta.json
  flappy_races/
    race_2026-03-18T18-20-00_stage3/
      manifest.json
      results.json
      replay.json
```

### Separation rules

* `flappy_submissions/` = active submitted models
* `flappy_races/` = frozen official evaluation outputs
* `flappy_leaderboard.json` = official rankings only

---

## 5. Implementation Tasks

## Task 1 — `utils/flappy_leaderboard.py`

### Purpose

Atomic leaderboard read/write/update with stage-based sorting.

### Requirements

* Atomic file writes using temp file + replace
* Schema uses `team_name` everywhere
* `add_entry()` accepts `team_name`, not `team`
* `get_sorted_by_stage()` sorts by the ranking policy
* `team_passed_stage()` returns whether the team has officially passed a stage
* preserve `submission_timestamp`
* preserve `status`

### API

```python
def load_leaderboard(path: str) -> dict: ...
def save_leaderboard(path: str, data: dict) -> None: ...
def add_entry(
    path: str,
    *,
    team_name: str,
    stage_id: int,
    avg_score: float,
    max_score: int,
    survival_steps_avg: float,
    episode_scores: list[float],
    passed: bool,
    race_id: str,
    submission_timestamp: str,
    status: str = "success",
) -> None: ...
def get_sorted_by_stage(path: str, stage_id: int) -> list[dict]: ...
def team_passed_stage(path: str, team_name: str, stage_id: int) -> bool: ...
```

---

## Task 2 — `utils/flappy_submission.py`

### Purpose

Validate, save, and load active submissions.

### Submission rules

* Accept only `state_dict.pt` + `metadata.json`
* Reject full serialized model objects, arbitrary pickled models, missing metadata, mismatched architecture IDs, wrong observation/action dimensions

### Validation checklist

1. file readable
2. metadata readable
3. `architecture_id` in allowed whitelist
4. `team_name` exists
5. `obs_dim == EXPECTED_OBS_DIM`
6. `action_dim == 2`
7. model can be constructed from whitelist
8. state_dict loads on CPU
9. dummy forward pass succeeds on shape `(1, EXPECTED_OBS_DIM)`

### Required APIs

```python
def validate_and_load(
    state_dict_bytes: bytes,
    metadata_bytes: bytes,
    *,
    expected_team_name: str | None = None,
) -> tuple[torch.nn.Module | None, dict | None, str | None]: ...

def save_submission(
    data_root: str,
    stage_id: int,
    team_name: str,
    state_dict_bytes: bytes,
    metadata: dict,
) -> str: ...

def load_submission(team_dir: str) -> tuple[torch.nn.Module | None, dict | None, str | None]: ...

def list_stage_submissions(data_root: str, stage_id: int) -> list[dict]: ...
```

### Policy

* latest submission per team per stage overwrites the active slot

---

## Task 3 — `utils/flappy_eval.py`

### Purpose

Run official deterministic evaluation and write race outputs.

### Requirements

* stage config includes both `gap_size` and `speed`
* use `STAGE_SEEDS[stage_id]`
* set all models to `eval()`
* greedy policy only
* handle evaluation failures per team without crashing the entire race
* write `manifest.json` and `results.json`
* call `utils.flappy_replay.build_replay_json()` to generate replay payload

### Required APIs

```python
def evaluate_model(
    model: torch.nn.Module,
    stage_id: int,
    team_name: str,
    *,
    max_steps: int = 5000,
) -> dict: ...

def run_race(
    teams: list[dict],
    stage_id: int,
    data_root: str,
) -> dict: ...
```

### Failure handling

If one team fails: record `status = "evaluation_failed"`, add `error_message`, continue evaluating others.

### Headless env note

If the env requires a display, set SDL dummy mode before import.

---

## Task 4 — `utils/flappy_replay.py`

### Purpose

Keep replay serialization separate from evaluation logic.

### Required API

```python
def build_replay_json(
    race_id: str,
    stage_id: int,
    eval_runs: list[dict],
    stages: dict,
) -> dict: ...
```

### Replay content

* `race_id`, `stage_id`, `gap_size`, `speed`, `episodes`
* optional world metadata: `world_width`, `world_height`, `ground_height`

### Replay frame structure

```json
{
  "t": 0,
  "pipes": [...],
  "birds": [
    {
      "team_name": "Team Alpha",
      "x": 40,
      "y": 120,
      "alive": true,
      "score": 0,
      "action": 0
    }
  ]
}
```

### Important rule

Do not hardcode JS scaling assumptions if env dimensions can be serialized. Prefer replay metadata over front-end magic constants.

---

## Task 5 — `utils/styles.py`

### Purpose

Add stage badge CSS for locked / available / passed / active states.

### Requirements

* add `.stage-row`, `.stage-badge`, `.locked`, `.available`, `.passed`, `.active`

---

## Task 6 — `modules/module5_flappy.py`

### Purpose

Main UI module for student workflow, admin workflow, replay display, and leaderboard view.

### Constants

Do **not** hardcode admin password in source.

```python
FLAPPY_LB_PATH = os.path.join(DATA_ROOT, "flappy_leaderboard.json")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
ADMIN_PASSWORD = st.secrets.get("FLAPPY_ADMIN_PASSWORD") or os.getenv("FLAPPY_ADMIN_PASSWORD")
```

### Page sections

1. title and description
2. stage badges
3. left controls
4. right replay viewer
5. leaderboard tabs

### Student controls

* team name, architecture selector, hyperparameter controls
* demo train button, upload state_dict.pt + metadata.json, submit button

### Admin controls

* password field, enable admin mode if valid
* stage selector, freeze batch button, start official race button

### Key functions

* `render_stage_badges(team_name, lb_path)` — show stage status, N+1 locked unless N passed
* `_run_demo_training(...)` — lightweight, capped 100-300 episodes, non-authoritative
* `_submit_current_model(...)` — validate + save, do not touch leaderboard
* `_freeze_race_batch(stage_id)` — create immutable manifest
* `_run_official_race(stage_id)` — enforce lock, run race, update leaderboard from Python only
* `_render_replay(replay_data)` — inject JSON into HTML template, browser animates only
* `_render_leaderboard()` — tab per stage, read official entries only

### Race lock

* lock file contains timestamp
* if lock is stale beyond threshold, allow recovery
* always remove lock in `finally`

---

## Task 7 — `static/flappy_race.html`

### Purpose

Canvas replay renderer for official race playback.

### Requirements

* no external JS dependencies
* render from injected replay JSON
* show unique bird colors, team labels, pipes, dead-bird fade, winner crown
* overlay: episode index, current score per team, alive/dead state

### Important rule

* no score computation, no leaderboard update, no pass/fail logic

### Scaling

If replay JSON includes world dimensions, use them instead of hardcoded assumptions.

---

## Task 8 — `app.py`

### Purpose

Expose module 5 in the app navigation.

### Requirements

* add `"Flappy Bird Competition"` nav item
* route to `render_module5()`

---

## 6. Demo Training Policy

### Policy

* recommended dashboard range: `100-300`
* hard cap may exist, but UI should describe demo mode clearly
* official competition performance is expected from external training + upload

### UI note

> "Dashboard training is for quick experimentation only. Use external training for serious competition runs."

---

## 7. Integration and Smoke Tests

## 7.1 Existing checks

* imports load
* model architectures build and infer
* env loads and steps once

## 7.2 New required smoke tests

### Smoke test A — single-model evaluation

```bash
python -c "
import torch
from utils.flappy_submission import _build_model
from utils.flappy_eval import evaluate_model
m = _build_model('model1')
m.eval()
res = evaluate_model(m, 1, 'SmokeTeam', max_steps=200)
print(res['team_name'], res['avg_score'])
"
```

### Smoke test B — minimal race artifact generation

```bash
python -c "
import os, json, tempfile, torch
from utils.flappy_submission import _build_model, save_submission
from utils.flappy_eval import run_race

tmp = tempfile.mkdtemp()
m1 = _build_model('model1')
m2 = _build_model('model2')

buf1 = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
buf2 = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
torch.save(m1.state_dict(), buf1.name)
torch.save(m2.state_dict(), buf2.name)

meta1 = {'team_name':'A','architecture_id':'model1','obs_dim':4,'action_dim':2,'framework':'pytorch'}
meta2 = {'team_name':'B','architecture_id':'model2','obs_dim':4,'action_dim':2,'framework':'pytorch'}

with open(buf1.name,'rb') as f: b1 = f.read()
with open(buf2.name,'rb') as f: b2 = f.read()

save_submission(tmp, 1, 'A', b1, meta1)
save_submission(tmp, 1, 'B', b2, meta2)

teams = []
for team in ['A', 'B']:
    from utils.flappy_submission import load_submission
    model, meta, err = load_submission(os.path.join(tmp, 'flappy_submissions', 'stage_1', team))
    assert err is None, err
    teams.append({'team_name': team, 'model': model, 'submission_timestamp': '2026-03-18T00:00:00'})

race = run_race(teams, 1, tmp)
print(race['race_id'])
print(os.path.exists(os.path.join(tmp, 'flappy_races', race['race_id'], 'results.json')))
print(os.path.exists(os.path.join(tmp, 'flappy_races', race['race_id'], 'replay.json')))
"
```

---

## 8. Commit Plan

| Commit | Message |
|--------|---------|
| 1 | `feat: add flappy leaderboard and submission validation utilities` |
| 2 | `feat: add deterministic flappy evaluation and replay serialization` |
| 3 | `style: add flappy stage badge styles` |
| 4 | `feat: add flappy competition module page` |
| 5 | `feat: add canvas replay viewer and app navigation` |
| 6 | `test: add flappy smoke tests for eval and race artifacts` |

---

## 9. Post-Implementation Tuning Checklist

After first working run, verify:

* stage speed actually changes env behavior
* replay coordinate mapping is correct
* leaderboard keys are consistent everywhere
* stale race lock recovery works
* headless env initialization works on deployment machine
* pass thresholds are not too easy or too hard
* JS never affects official scoring

---

## 10. Final Implementation Standard

The module is considered complete only if all of the following are true:

1. students can upload a valid `state_dict + metadata`
2. active submissions are stored by team and stage
3. admin can freeze a race batch
4. official race runs deterministically in Python
5. `results.json` and `replay.json` are written for each race
6. leaderboard updates only from Python race results
7. replay renders correctly in Canvas
8. stage unlock logic follows official pass status
9. admin secret is not hardcoded
10. smoke tests pass

---

## 11. Summary

This v2.1 plan keeps the same modular implementation direction, including the dedicated submission, evaluation, replay, leaderboard, and UI modules. It fixes the main issues that would otherwise cause instability or spec drift:

* inconsistent leaderboard schema,
* replay serialization ownership ambiguity,
* hardcoded admin password,
* missing stage-speed implementation,
* weak submission metadata contract,
* insufficient end-to-end smoke testing.

That makes it suitable not just for a demo, but for a controlled classroom competition.
