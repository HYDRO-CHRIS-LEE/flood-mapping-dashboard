# Flappy Bird AI Competition — Design Spec

**Date:** 2026-03-18
**Scope:** Add a Flappy Bird DQN competition module to the flood mapping dashboard

## Overview

A new module (module5) where students train DQN agents to play Flappy Bird and compete across 5 difficulty stages in real-time multi-bird races. Separate from the RF flood classifier — this is a standalone DQN hyperparameter tuning competition.

## Architecture

### Files

| Component | File | Role |
|-----------|------|------|
| Main UI | `modules/module5_flappy.py` | DQN controls, training, race trigger |
| Game engine | `static/flappy_race.html` | HTML5 Canvas multi-bird animation |
| Leaderboard | `utils/flappy_leaderboard.py` | JSON-based per-stage scoring |
| DQN model | `flappy_bird/model_dqn/` | Existing code reuse (agent, brain, replay_memory) |
| Game env | `flappy_bird/env_flappybird/` | Existing gym environment reuse |
| Sidebar | `app.py` | New nav item: "Flappy Bird Competition" |

### Data Flow

```
Python (module5_flappy.py)
  1. Load all submitted team models
  2. Create env with identical seed
  3. Compute step-by-step actions for each model
  4. Serialize actions + pipe coordinates as JSON
        ↓
JS (flappy_race.html via st.components.v1.html)
  5. Receive JSON → reconstruct per-frame state
  6. Render 60fps animation on Canvas
  7. Post results (survival steps/score per team) back via postMessage
        ↓
Python
  8. Save results to flappy_leaderboard.json
```

AI inference runs in Python; JS handles replay animation only.

## Page Layout

```
┌─────────────────────────────────────────────────────┐
│ 🐦 Flappy Bird AI Competition                      │
│ "Train a DQN agent to fly through pipes — compete   │
│  across 5 difficulty stages with your classmates."  │
├─────────────────────────────────────────────────────┤
│ [Stage 1 ✅] [Stage 2 🔒] [Stage 3 🔒] [Stage 4 🔒] [Stage 5 🔒] │
├──────────────────┬──────────────────────────────────┤
│  DQN Controls    │                                  │
│  (col_l)         │   🎮 Race Viewer (Canvas)        │
│                  │   Multiple birds fly together    │
│  Step 1: Model   │                                  │
│  Step 2: HParams │                                  │
│  Step 3: Train   │                                  │
│  Step 4: Compete │                                  │
├──────────────────┴──────────────────────────────────┤
│ 🏆 Stage Leaderboard (auto-refresh 5s)              │
└─────────────────────────────────────────────────────┘
```

## Stage System

5 difficulty stages with increasing challenge:

| Stage | Gap Size | Speed | Pass Condition | Label |
|-------|----------|-------|----------------|-------|
| 1 | 170 | Slow | avg ≥ 3 (10 ep) | "First Flight" |
| 2 | 155 | Slow | avg ≥ 5 (10 ep) | "Getting Steady" |
| 3 | 140 | Normal | avg ≥ 10 (10 ep) | "Tighter Gaps" |
| 4 | 125 | Fast | avg ≥ 15 (10 ep) | "Under Pressure" |
| 5 | 120 | Fast | Final ranking | "Survival of the Fittest" |

Rules:
- Stage N+1 unlocks only after Stage N is passed
- Each stage plays 10 episodes with identical random seed across all teams
- Stage 5 has no pass condition — avg_score determines leaderboard rank
- Dead birds fade out; last survivor gets crown effect

## DQN Controls (Left Column)

### Step 1 · Model Architecture
- Network selector: model1 / model2 / model3 (existing 3 architectures)

### Step 2 · Hyperparameters
- Learning Rate: slider 0.00005–0.001
- Batch Size: select [32, 64, 128]
- Dropout: slider 0.1–0.5
- Gamma: slider 0.9–0.99
- Optimizer: [Adam / SGD]

### Step 3 · Training
- Episodes: slider 100–2000 (cap to limit server load)
- "Train DQN!" button → headless training with progress bar
- OR "Upload .pt model" via file_uploader
- Architecture validation on upload (output dim == 2)

### Step 4 · Compete
- Stage selector dropdown
- "Submit to Race!" button → saves model for race
- "Start Race" button → triggers multi-bird race for all submitted models

## Canvas Race Viewer

- Pure HTML5 Canvas + Vanilla JS, no external dependencies
- Embedded via `st.components.v1.html`
- Visual elements:
  - Each team's bird: unique color + team name label
  - Pipes: classic green
  - Background: dark tone (consistent with dashboard)
  - Dead bird: red X + fade out
  - Last survivor: crown effect
- 60fps animation from pre-computed action sequences

## Leaderboard

JSON file: `data/flappy_leaderboard.json`

```json
{
  "entries": [
    {
      "team": "Team Alpha",
      "stages": {
        "1": {"avg_score": 8.2, "max_score": 15, "passed": true},
        "2": {"avg_score": 12.1, "max_score": 24, "passed": true}
      },
      "timestamp": "2026-03-18T14:30:00"
    }
  ]
}
```

- `@st.fragment(run_every=5)` auto-refresh (same pattern as RF leaderboard)
- Tab per stage, ranked by avg_score
- Reuses `utils/flappy_leaderboard.py` with same add_entry/get_sorted pattern

## Race Flow

1. Students train model (dashboard or Colab) and click "Submit to Race"
2. Professor selects stage and clicks "Start Race"
3. Python loads all submitted models, runs 10 episodes headless with same seed
4. Action sequences + pipe coords serialized to JSON
5. Canvas renders multi-bird race animation
6. Results auto-saved to leaderboard

## Constraints

- Max 2000 training episodes in dashboard to limit server load
- `.pt` upload validated for correct architecture
- Same random seed per race for fairness
- No separate professor UI — same page, button-based control
