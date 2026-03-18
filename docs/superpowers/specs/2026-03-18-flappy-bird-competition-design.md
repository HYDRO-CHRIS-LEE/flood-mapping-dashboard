# Flappy Bird AI Competition — Design Spec v2

**Date:** 2026-03-18
**Scope:** Add a Flappy Bird DQN competition module to the flood mapping dashboard as a standalone reinforcement learning activity, separate from the flood-mapping pipeline. The original v1 spec already defined the module as an independent DQN competition with Streamlit UI, HTML5 replay, stage progression, and leaderboard support; this v2 preserves that structure while tightening evaluation fairness, submission integrity, and operational stability.

---

## 1. Overview

This module introduces a classroom RL competition where students train Deep Q-Network (DQN) agents to play Flappy Bird and compete across five difficulty stages. The goal is not only to rank performance, but also to help students understand:

- the difference between supervised learning and sequential decision-making,
- the effect of hyperparameter tuning on policy quality,
- the importance of evaluation protocol and reproducibility.

This module is pedagogically complementary to the earlier flood-mapping modules: those focus on classification and spatial inference, while this module provides a compact environment for reinforcement learning, policy learning, and controlled competition.

---

## 2. Design Goals

1. **Fair competition**
   All models must be evaluated under an identical and explicitly versioned protocol.

2. **Deterministic replay**
   Python performs the authoritative evaluation; the browser only visualizes a saved replay.

3. **Operational robustness**
   Student submissions, leaderboard updates, and race rendering must remain stable under Streamlit reruns.

4. **Simple classroom workflow**
   Students can either train in-dashboard for short demo runs or train externally and upload a valid checkpoint.

5. **Clear stage progression**
   Difficulty increases gradually, with stage unlocks based on fixed pass criteria.

---

## 3. High-Level Architecture

### 3.1 Core Components

| Component | File | Role |
|---|---|---|
| Main UI | `modules/module5_flappy.py` | Streamlit controls, submission flow, admin race control |
| Replay Renderer | `static/flappy_race.html` | HTML5 Canvas playback of precomputed race trajectories |
| Leaderboard Utility | `utils/flappy_leaderboard.py` | Read/write leaderboard entries, sorting, filtering |
| Submission Utility | `utils/flappy_submission.py` | Validate uploads, save metadata, manage stage submissions |
| Evaluation Utility | `utils/flappy_eval.py` | Deterministic evaluation under fixed protocol |
| Replay Utility | `utils/flappy_replay.py` | Serialize episode trajectories to JSON |
| DQN Models | `flappy_bird/model_dqn/` | Existing DQN architectures and inference code |
| Game Environment | `flappy_bird/env_flappybird/` | Existing Flappy Bird gym-style environment |
| Sidebar Integration | `app.py` | Add navigation item: `"Flappy Bird Competition"` |

### 3.2 Core Principle

**Python is the source of truth.**
The browser never decides scores, ranks, or pass/fail status.

That means:

- Python loads and evaluates submitted models.
- Python computes all official episode outcomes.
- Python saves the authoritative leaderboard and replay data.
- JS only renders precomputed trajectories for visualization.

---

## 4. Authoritative Data Flow

```text
Student / Instructor
  |
Streamlit UI (module5_flappy.py)
  1. Student selects architecture / hyperparameters or uploads checkpoint
  2. Submission metadata is validated and saved
  3. Instructor selects stage and triggers official evaluation
        |
Python Evaluation Engine (authoritative)
  4. Load all valid submissions for that stage
  5. Run deterministic evaluation on fixed seed list
  6. Record per-episode results, aggregate metrics, and pass/fail
  7. Save:
       - leaderboard entry
       - race manifest
       - replay JSON
        |
Streamlit UI
  8. Load saved replay JSON
  9. Send replay payload to HTML5 Canvas
        |
JS Renderer (visualization only)
  10. Render bird trajectories, pipes, deaths, and winner effects
  11. Optionally notify Streamlit that playback finished
```

### 4.1 What JS Must Not Do

JS must **not**:

* compute official score,
* determine ranking,
* update the leaderboard,
* decide pass/fail,
* alter replay state.

It may only:

* animate saved trajectories,
* show labels and effects,
* emit a non-authoritative `"playback_complete"` event.

---

## 5. Evaluation Protocol

### 5.1 Fairness Definition

Fairness is **not** just "same random seed."
Fairness is **same evaluation protocol**, consisting of:

* fixed environment version,
* fixed stage definition,
* fixed observation preprocessing,
* fixed reward definition,
* fixed termination logic,
* fixed action frequency / frame-skip behavior,
* fixed deterministic inference mode,
* fixed seed list.

Every official evaluation must produce a record such as:

```json
{
  "eval_protocol_version": "flappy_eval_v1",
  "env_version": "flappy_env_v1",
  "stage_id": 3,
  "seed_list": [1031, 1049, 1063, 1091, 1103, 1129, 1151, 1181, 1213, 1237],
  "policy_mode": "greedy_eval",
  "frame_skip": 1,
  "reward_version": "classic_score_reward_v1",
  "termination_version": "pipe_ground_collision_v1"
}
```

### 5.2 Deterministic Inference Rules

For official evaluation:

* model must be set to `eval()` mode,
* dropout must be disabled automatically,
* exploration must be disabled,
* action selection must be greedy: `action = argmax(Q(s))`,
* no epsilon-greedy randomness is allowed.

### 5.3 Seed Handling

Each stage uses a fixed list of **10 evaluation seeds**.

* The list is stored in code or config and version-controlled.
* All teams are evaluated on the same 10 seeds.
* The same seed list is used for all teams in that stage during a protocol version.

### 5.4 Aggregate Metrics

For each submission and stage:

* `episode_scores[10]`
* `avg_score`
* `max_score`
* `survival_steps_avg`
* `passed`
* `rank`

Primary ranking metric:

1. `avg_score`
2. tie-breaker: `max_score`
3. tie-breaker: `survival_steps_avg`
4. final tie-breaker: earlier submission timestamp

---

## 6. Stage System

| Stage | Gap Size | Speed  | Pass Condition                              | Label                   |
| ----- | -------: | ------ | ------------------------------------------- | ----------------------- |
| 1     |      170 | Slow   | `avg_score >= 3` over 10 official episodes  | First Flight            |
| 2     |      155 | Slow   | `avg_score >= 5` over 10 official episodes  | Getting Steady          |
| 3     |      140 | Normal | `avg_score >= 10` over 10 official episodes | Tighter Gaps            |
| 4     |      125 | Fast   | `avg_score >= 15` over 10 official episodes | Under Pressure          |
| 5     |      120 | Fast   | No pass threshold; ranked by score          | Survival of the Fittest |

### 6.1 Unlock Logic

* Stage `N+1` unlocks only if Stage `N` is officially passed.
* Unlocks are based only on authoritative Python evaluation results.
* Informal local training runs do not unlock stages.

### 6.2 Threshold Calibration Note

Before classroom launch, thresholds should be sanity-checked using a random agent, a simple heuristic baseline, and an instructor reference DQN. Thresholds may be adjusted once before release, then frozen for the competition period.

---

## 7. Page Layout

```text
+--------------------------------------------------------------+
| Bird Flappy Bird AI Competition                               |
| Train a DQN agent, submit it, and compete across 5 stages   |
+--------------------------------------------------------------+
| [Stage 1 Y] [Stage 2 Lock] [Stage 3 Lock] [Stage 4 Lock] [Stage 5 Lock] |
+-----------------------+--------------------------------------+
| Left Control Panel    | Right Replay Panel                  |
|                       |                                      |
| 1. Team + Model       | Game Race Viewer (Canvas replay)      |
| 2. Hyperparameters    | Official replay only                |
| 3. Train or Upload    |                                      |
| 4. Submit             |                                      |
| 5. Admin Start Race   |                                      |
+-----------------------+--------------------------------------+
| Trophy Leaderboard / Submission Status / Evaluation Logs         |
+--------------------------------------------------------------+
```

---

## 8. User Roles

### 8.1 Student Actions

Students can:

* select a model architecture,
* set training hyperparameters,
* run short in-dashboard training,
* upload a valid checkpoint,
* submit to the current stage,
* view replay and leaderboard.

Students cannot:

* trigger official race evaluation,
* overwrite leaderboard entries directly,
* change stage protocol settings.

### 8.2 Instructor Actions

Instructor can:

* select stage for official evaluation,
* trigger official evaluation,
* freeze current submission set into a race batch,
* replay the batch,
* reset non-official demo state if needed.

### 8.3 Minimal Admin Gate

* `admin_mode` toggle protected by a secret or password,
* `"Start Official Race"` visible only in admin mode.

---

## 9. Left Control Panel Specification

### 9.1 Step 1 - Team and Model Identity

* `team_name` (required, unique)
* `architecture_id`: model1 / model2 / model3
* One active submission per team per stage

### 9.2 Step 2 - Hyperparameters

* Learning Rate: `0.00005 - 0.001`
* Batch Size: `[32, 64, 128]`
* Dropout: `0.1 - 0.5`
* Gamma: `0.90 - 0.99`
* Optimizer: `Adam` or `SGD`
* Episodes: `100 - 300` recommended for dashboard demo mode

### 9.3 Step 3 - Training or Upload

**A. Demo Training**
* `"Train DQN (Demo)"` button
* Headless, capped for server safety
* Not automatically submitted

**B. External Upload**
* Accepted: `state_dict.pt` + `metadata.json`

Example metadata:

```json
{
  "team_name": "Team Alpha",
  "architecture_id": "model2",
  "obs_dim": 8,
  "action_dim": 2,
  "framework": "pytorch",
  "training_note": "trained in Colab for 3000 episodes"
}
```

### 9.4 Upload Validation

Allowed: `state_dict` only. Not allowed: serialized full model objects, arbitrary pickle.

Validation checks:

1. file readable,
2. metadata present,
3. `architecture_id` is allowed,
4. observation dimension matches expected env interface,
5. action dimension equals `2`,
6. state dict keys match whitelisted architecture,
7. tensor shapes are compatible,
8. load succeeds on CPU,
9. dry-run inference succeeds on one dummy observation.

### 9.5 Step 4 - Submit

* `"Submit to Stage"` button
* Validate artifact, store under team/stage, mark active
* No leaderboard update at this point

### 9.6 Step 5 - Official Evaluation (Admin only)

* Stage selector
* `"Freeze Submission Batch"`
* `"Start Official Race"`

---

## 10. Submission and Race State Management

### 10.1 State Separation

* **submission state**: currently active team submissions
* **race state**: one frozen batch for one official evaluation
* **leaderboard state**: historical official results

### 10.2 File Structure

```text
data/flappy_submissions/
  stage_1/
    TeamAlpha/
      state_dict.pt
      metadata.json
      submission_meta.json
    TeamBeta/
      ...
data/flappy_races/
  race_2026-03-18T18-20-00_stage3/
    manifest.json
    results.json
    replay.json
data/flappy_leaderboard.json
```

### 10.3 Race Manifest

```json
{
  "race_id": "race_2026-03-18T18-20-00_stage3",
  "stage_id": 3,
  "eval_protocol_version": "flappy_eval_v1",
  "teams": [
    {"team_name": "Team Alpha", "submission_path": "..."},
    {"team_name": "Team Beta", "submission_path": "..."}
  ],
  "frozen_at": "2026-03-18T18:20:00"
}
```

The manifest is immutable once created.

---

## 11. Replay Viewer

### 11.1 Renderer Requirements

* HTML5 Canvas, Vanilla JS only
* Embedded via `st.components.v1.html`
* No external dependencies

### 11.2 Visual Elements

* Unique bird color per team + team label
* Classic green pipes
* Dark dashboard-consistent background
* Dead bird fade-out
* Winner crown effect
* Optional subtle trail for active bird motion

### 11.3 Information Overlay

* Current episode number
* Current score per team
* Alive/dead status
* Average score summary
* Collision cause (top pipe / bottom pipe / ground)

Optional advanced overlay:

* Current chosen action (flap / no flap)
* Current estimated Q-values

### 11.4 Replay JSON Content

```json
{
  "race_id": "race_2026-03-18T18-20-00_stage3",
  "stage_id": 3,
  "episodes": [
    {
      "seed": 1031,
      "frames": [
        {
          "t": 0,
          "pipes": [...],
          "birds": [
            {"team": "Team Alpha", "x": 40, "y": 120, "alive": true, "score": 0},
            {"team": "Team Beta", "x": 40, "y": 118, "alive": true, "score": 0}
          ]
        }
      ]
    }
  ]
}
```

---

## 12. Leaderboard

### 12.1 File

`data/flappy_leaderboard.json`

### 12.2 Schema

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
      "rank": 1,
      "race_id": "race_2026-03-18T18-20-00_stage2",
      "submission_timestamp": "2026-03-18T17:52:11",
      "evaluated_at": "2026-03-18T18:21:02"
    }
  ]
}
```

### 12.3 Write Policy

* Updated only after official evaluation completes in Python
* Official ranking view = best official result per team per stage
* History view = all official results

### 12.4 Refresh

`@st.fragment(run_every=5)` auto-refresh. Replay rendering and leaderboard updates remain decoupled.

---

## 13. Race Flow

### 13.1 Student Flow

1. Select architecture and hyperparameters
2. Train briefly in dashboard or train externally
3. Upload valid `state_dict + metadata`
4. Submit to selected unlocked stage
5. Wait for official evaluation
6. View replay and leaderboard

### 13.2 Instructor Flow

1. Enter admin mode
2. Choose stage
3. Freeze current active submissions into a race batch
4. Start official evaluation
5. Review saved results
6. Play replay in the browser

### 13.3 Official Evaluation Flow

1. Load race manifest
2. For each team, load validated checkpoint
3. Set model to `eval()`
4. Run 10 official episodes on fixed seed list
5. Compute authoritative metrics
6. Write `results.json`
7. Update leaderboard
8. Generate replay JSON
9. Display replay

---

## 14. Operational Constraints

### 14.1 Dashboard Training Limits

* Dashboard training capped at 100-300 episodes for demo
* Primary training expected from external uploads (Colab)

### 14.2 CPU/GPU

* Official evaluation runs on CPU for reproducibility
* All uploads must be loadable on CPU
* Device-specific checkpoints normalized during validation

### 14.3 Concurrency

* Only one official race at a time
* File lock: `data/flappy_race.lock`

### 14.4 Failure Handling

* Failed team submission marked as `evaluation_failed`
* Error logged, remaining teams continue
* Race batch does not crash on individual failure

---

## 15. Security and Integrity

1. Accept only `state_dict` uploads
2. Use only whitelisted architecture constructors
3. Never execute arbitrary uploaded Python code
4. Never trust browser-returned scores
5. Keep raw submissions and official outputs separate
6. Log all official race runs with race ID and protocol version

---

## 16. Implementation Milestones

### Milestone 1 - Minimal Competition Backbone
* Page scaffold, stage selector, team submission storage, leaderboard JSON, admin gate

### Milestone 2 - Validation Layer
* `state_dict + metadata` upload, architecture whitelist, dummy inference test, submission archive

### Milestone 3 - Authoritative Evaluation Engine
* Fixed seed lists, eval mode, deterministic scoring, results JSON

### Milestone 4 - Replay Pipeline
* Replay serializer, HTML5 Canvas playback, visual overlays, winner/death effects

### Milestone 5 - Polish
* Leaderboard tabs, submission status cards, error messages, race history

---

## 17. Non-Goals

This module does **not** aim to provide:

* a full RL research platform,
* arbitrary user-defined architectures,
* browser-side policy execution,
* live multiplayer gameplay,
* distributed training orchestration.

It is a controlled educational competition module.
