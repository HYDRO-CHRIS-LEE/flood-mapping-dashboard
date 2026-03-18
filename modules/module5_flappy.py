"""
Flappy Bird AI Competition — Streamlit UI module.

Teams choose a DQN architecture, tune hyperparameters, train (demo) or upload
a state_dict, submit to one of five progressive stages, and compete in
admin-triggered official races rendered via an HTML Canvas replay.
"""

import os
# Must be set before any flappy_bird env import (pyglet/OpenGL)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import io
import json
import sys
import time
import uuid
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components
import torch

from utils.flappy_leaderboard import (
    add_entry as lb_add,
    get_sorted_by_stage,
    team_passed_stage,
)
from utils.flappy_submission import (
    validate_and_load,
    save_submission,
    load_submission,
    list_stage_submissions,
    _build_model,
    ALLOWED_ARCHITECTURES,
)
from utils.flappy_eval import STAGES, run_race
from utils.styles import COLORS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FLAPPY_LB_PATH = os.path.join(DATA_ROOT, "flappy_leaderboard.json")
ADMIN_PASSWORD = os.getenv("FLAPPY_ADMIN_PASSWORD", "earthai2026")

_STAGE_IDS = sorted(STAGES.keys())
_ARCH_OPTIONS = sorted(ALLOWED_ARCHITECTURES)
_LR_OPTIONS = [0.00005, 0.0001, 0.0005, 0.001]

_FLAPPY_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "flappy_bird"
)
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")

_LOCK_PATH = os.path.join(DATA_ROOT, ".flappy_race.lock")


# ── helpers ───────────────────────────────────────────────────────
def _team() -> str:
    return st.session_state.get("team_name", "Team A")


def _unlocked_stages(team: str) -> list[int]:
    """Return the list of stage IDs that *team* has access to.

    Stage 1 is always unlocked.  Stage N is unlocked if stage N-1 is passed.
    """
    unlocked = [1]
    for sid in _STAGE_IDS[1:]:
        prev = sid - 1
        if team_passed_stage(FLAPPY_LB_PATH, team, prev):
            unlocked.append(sid)
        else:
            break
    return unlocked


def _stage_badge(sid: int, unlocked: list[int]) -> str:
    s = STAGES[sid]
    if sid in unlocked:
        icon = "✅" if team_passed_stage(FLAPPY_LB_PATH, _team(), sid) else "🔓"
    else:
        icon = "🔒"
    return (
        f'<div style="display:inline-block;text-align:center;padding:4px 10px;'
        f'margin-right:6px;border-radius:6px;font-size:12px;'
        f'background:{COLORS["bg3"]};border:1px solid {COLORS["border"]}">'
        f'{icon} <b>Stage {sid}</b><br>'
        f'<span style="font-size:11px;color:{COLORS["text_sub"]}">{s["label"]}</span>'
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════
# 1. Main entry point
# ══════════════════════════════════════════════════════════════════
def render_module5():
    team = _team()
    unlocked = _unlocked_stages(team)

    # ── Section header (follows module4 pattern) ──────────────────
    st.markdown(
        """
    <div class="section-header">
        <div class="section-icon">🐦</div>
        <div>
            <div class="section-title">Flappy Bird AI Competition</div>
            <div class="section-desc">
                Train a DQN agent to navigate five progressively harder stages —
                tighter gaps, faster pipes. Beat each stage's score threshold to unlock the next.
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Stage badges row ──────────────────────────────────────────
    badges_html = "".join(_stage_badge(sid, unlocked) for sid in _STAGE_IDS)
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;margin-bottom:16px">{badges_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Two columns ───────────────────────────────────────────────
    col_l, col_r = st.columns([1, 1.5])

    with col_l:
        _render_left_controls(team, unlocked)

    with col_r:
        _render_right_panel()

    # ── Leaderboard below ─────────────────────────────────────────
    st.markdown("---")
    _render_leaderboard()


# ══════════════════════════════════════════════════════════════════
# 2. Left column controls
# ══════════════════════════════════════════════════════════════════
def _render_left_controls(team: str, unlocked: list[int]):
    # ── Step 1 · Model Architecture ───────────────────────────────
    st.markdown(
        f'<div style="font-weight:600;font-size:14px;margin-bottom:4px;'
        f'color:{COLORS["text"]}">Step 1 · Model Architecture</div>',
        unsafe_allow_html=True,
    )
    arch = st.selectbox(
        "Architecture",
        options=_ARCH_OPTIONS,
        label_visibility="collapsed",
        key="flappy_arch",
    )

    # ── Step 2 · Hyperparameters ──────────────────────────────────
    st.markdown(
        f'<div style="font-weight:600;font-size:14px;margin:12px 0 4px;'
        f'color:{COLORS["text"]}">Step 2 · Hyperparameters</div>',
        unsafe_allow_html=True,
    )
    lr = st.select_slider(
        "Learning Rate",
        options=_LR_OPTIONS,
        value=0.0001,
        key="flappy_lr",
    )
    batch_size = st.selectbox(
        "Batch Size", options=[32, 64, 128], index=1, key="flappy_bs"
    )
    dropout = st.slider(
        "Dropout", min_value=0.1, max_value=0.5, value=0.2, step=0.05,
        key="flappy_dropout",
    )
    gamma = st.slider(
        "Gamma", min_value=0.90, max_value=0.99, value=0.95, step=0.01,
        key="flappy_gamma",
    )
    optimizer_id = st.selectbox(
        "Optimizer", options=["Adam", "SGD"], key="flappy_optim"
    )
    episodes = st.slider(
        "Episodes", min_value=100, max_value=2000, value=500, step=100,
        key="flappy_episodes",
    )

    # ── Step 3 · Train or Upload ──────────────────────────────────
    st.markdown(
        f'<div style="font-weight:600;font-size:14px;margin:12px 0 4px;'
        f'color:{COLORS["text"]}">Step 3 · Train or Upload</div>',
        unsafe_allow_html=True,
    )
    tab_train, tab_upload = st.tabs(["Train (Demo)", "Upload .pt"])

    with tab_train:
        if st.button("🚀 Start Training", key="btn_train", use_container_width=True):
            _run_demo_training(
                arch=arch,
                lr=lr,
                batch_size=batch_size,
                dropout=dropout,
                gamma=gamma,
                optimizer_id=optimizer_id,
                episodes=episodes,
            )

    with tab_upload:
        sd_file = st.file_uploader(
            "state_dict.pt", type=["pt"], key="upload_sd"
        )
        meta_file = st.file_uploader(
            "metadata.json", type=["json"], key="upload_meta"
        )
        if sd_file is not None and meta_file is not None:
            sd_bytes = sd_file.read()
            try:
                metadata = json.loads(meta_file.read().decode("utf-8"))
            except Exception as exc:
                st.error(f"Failed to parse metadata.json: {exc}")
                metadata = None

            if metadata is not None:
                model, meta_out, err = validate_and_load(sd_bytes, metadata)
                if err is not None:
                    st.error(err)
                else:
                    st.session_state["flappy_model"] = model
                    st.session_state["flappy_model_bytes"] = sd_bytes
                    st.session_state["flappy_metadata"] = meta_out
                    st.success("Model validated and loaded successfully!")

    # ── Step 4 · Submit to Stage ──────────────────────────────────
    st.markdown(
        f'<div style="font-weight:600;font-size:14px;margin:12px 0 4px;'
        f'color:{COLORS["text"]}">Step 4 · Submit to Stage</div>',
        unsafe_allow_html=True,
    )
    if unlocked:
        submit_stage = st.selectbox(
            "Stage",
            options=unlocked,
            format_func=lambda s: f"Stage {s} — {STAGES[s]['label']}",
            key="flappy_submit_stage",
        )
    else:
        submit_stage = 1

    if st.button("📤 Submit", key="btn_submit", use_container_width=True):
        model = st.session_state.get("flappy_model")
        sd_bytes = st.session_state.get("flappy_model_bytes")
        metadata = st.session_state.get("flappy_metadata")
        if model is None or sd_bytes is None or metadata is None:
            st.error("No validated model. Train or upload first.")
        else:
            try:
                sub_dir = save_submission(
                    data_root=DATA_ROOT,
                    stage_id=submit_stage,
                    team_name=team,
                    state_dict_bytes=sd_bytes,
                    metadata=metadata,
                )
                st.success(f"Submission saved for Stage {submit_stage}.")
            except Exception as exc:
                st.error(f"Submission failed: {exc}")

    # ── Step 5 · Official Race (Admin) ────────────────────────────
    st.markdown(
        f'<div style="font-weight:600;font-size:14px;margin:12px 0 4px;'
        f'color:{COLORS["text"]}">Step 5 · Official Race (Admin)</div>',
        unsafe_allow_html=True,
    )
    admin_pw = st.text_input(
        "Admin Password", type="password", key="flappy_admin_pw"
    )
    race_stage = st.selectbox(
        "Race Stage",
        options=_STAGE_IDS,
        format_func=lambda s: f"Stage {s} — {STAGES[s]['label']}",
        key="flappy_race_stage",
    )
    if st.button("🏁 Start Official Race!", key="btn_race", use_container_width=True):
        if admin_pw != ADMIN_PASSWORD:
            st.error("Invalid admin password.")
        else:
            _run_official_race(race_stage)


# ══════════════════════════════════════════════════════════════════
# 3. Right column
# ══════════════════════════════════════════════════════════════════
def _render_right_panel():
    replay = st.session_state.get("flappy_replay")
    if replay is not None:
        _render_replay(replay)
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:400px;border:2px dashed '
            + COLORS["border"]
            + ";border-radius:12px;"
            'flex-direction:column;color:'
            + COLORS["text_muted"]
            + '">'
            '<div style="font-size:64px;margin-bottom:8px">🐦</div>'
            '<div style="font-size:14px">Race replay will appear here</div>'
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# 4. Demo training
# ══════════════════════════════════════════════════════════════════
def _run_demo_training(
    arch: str,
    lr: float,
    batch_size: int,
    dropout: float,
    gamma: float,
    optimizer_id: str,
    episodes: int,
):
    """Run a short DQN training loop inside the Streamlit process."""

    # Ensure flappy_bird env is importable
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if _FLAPPY_ROOT not in sys.path:
        sys.path.insert(0, _FLAPPY_ROOT)

    from env_flappybird.flappybird_env import FlappyBirdEnv
    from model_dqn.replay_memory import ReplayMemory
    from model_dqn.common import Transition
    import torch.nn.functional as F
    import random
    import numpy as np

    model = _build_model(arch, dropout)
    model.train()

    if optimizer_id == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    memory = ReplayMemory(10_000)
    env = FlappyBirdEnv()

    progress = st.progress(0, text="Training...")
    score_display = st.empty()
    best_score = 0

    for ep in range(episodes):
        obs = env.reset(gap_size=170, is_random_gap=False)
        state = torch.FloatTensor(obs).unsqueeze(0)
        total_reward = 0

        for t in range(2000):
            # Epsilon-greedy
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
            total_reward += reward

            next_state = torch.FloatTensor(obs_next).unsqueeze(0) if not done else None
            action_t = torch.LongTensor([[action]])
            reward_t = torch.FloatTensor([reward])

            memory.push(state, action_t, next_state, reward_t)
            state = next_state if next_state is not None else state

            # Learn
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                non_final_mask = torch.tensor(
                    [s is not None for s in batch.next_state], dtype=torch.bool
                )
                non_final_next = torch.cat(
                    [s for s in batch.next_state if s is not None]
                ) if any(s is not None for s in batch.next_state) else None

                model.eval()
                q_values = model(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(batch_size)
                if non_final_next is not None:
                    next_state_values[non_final_mask] = (
                        model(non_final_next).max(1)[0].detach()
                    )
                model.train()

                expected = reward_batch + gamma * next_state_values
                loss = F.mse_loss(q_values, expected.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        ep_score = int(env.score)
        best_score = max(best_score, ep_score)

        pct = (ep + 1) / episodes
        progress.progress(pct, text=f"Episode {ep+1}/{episodes} — Score: {ep_score} | Best: {best_score}")

    env.close()
    progress.progress(1.0, text="Training complete!")
    score_display.success(f"Training finished. Best score: {best_score}")

    # Save model to session state
    model.eval()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    sd_bytes = buf.getvalue()

    metadata = {
        "team_name": _team(),
        "architecture_id": arch,
        "obs_dim": 4,
        "action_dim": 2,
        "framework": "pytorch",
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "gamma": gamma,
        "optimizer": optimizer_id,
        "episodes": episodes,
    }

    st.session_state["flappy_model"] = model
    st.session_state["flappy_model_bytes"] = sd_bytes
    st.session_state["flappy_metadata"] = metadata


# ══════════════════════════════════════════════════════════════════
# 5. Official race
# ══════════════════════════════════════════════════════════════════
def _run_official_race(stage_id: int):
    """Run an official race for *stage_id*, updating the leaderboard."""

    # Race lock
    if os.path.exists(_LOCK_PATH):
        st.error("A race is already in progress. Please wait.")
        return

    try:
        os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
        with open(_LOCK_PATH, "w") as f:
            f.write(str(time.time()))

        submissions = list_stage_submissions(DATA_ROOT, stage_id)
        if not submissions:
            st.warning(f"No submissions found for Stage {stage_id}.")
            return

        # Load each submission
        teams: list[dict] = []
        for sub in submissions:
            model, meta, err = load_submission(sub["team_dir"])
            if err is not None:
                st.warning(f"Skipping {sub['team_name']}: {err}")
                continue
            teams.append(
                {
                    "team_name": sub["team_name"],
                    "model": model,
                    "submission_timestamp": sub.get("saved_at", ""),
                }
            )

        if not teams:
            st.warning("No valid submissions to race.")
            return

        with st.spinner(f"Running Stage {stage_id} race with {len(teams)} teams..."):
            race_result = run_race(
                teams=teams,
                stage_id=stage_id,
                data_root=DATA_ROOT,
            )

        # Update leaderboard for each team result
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

        # Save replay to session state
        st.session_state["flappy_replay"] = race_result["replay"]
        st.success(f"Race complete! Race ID: {race_result['race_id']}")

    except Exception as exc:
        st.error(f"Race failed: {exc}")
    finally:
        if os.path.exists(_LOCK_PATH):
            os.remove(_LOCK_PATH)


# ══════════════════════════════════════════════════════════════════
# 6. Replay renderer
# ══════════════════════════════════════════════════════════════════
def _render_replay(replay_data: dict):
    """Embed the Canvas-based replay viewer with race data injected."""
    html_path = os.path.join(_STATIC_DIR, "flappy_race.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        st.error("Replay viewer not found (static/flappy_race.html).")
        return

    replay_json = json.dumps(replay_data, ensure_ascii=False)
    html_content = html_content.replace(
        "/*__REPLAY_DATA__*/ null",
        replay_json,
    )
    components.html(html_content, height=500)


# ══════════════════════════════════════════════════════════════════
# 7. Leaderboard
# ══════════════════════════════════════════════════════════════════
@st.fragment(run_every=5)
def _render_leaderboard():
    """Tabbed leaderboard for each stage, auto-refreshing every 5 s."""
    tabs = st.tabs([f"Stage {sid}" for sid in _STAGE_IDS])

    for tab, sid in zip(tabs, _STAGE_IDS):
        with tab:
            entries = get_sorted_by_stage(FLAPPY_LB_PATH, sid)
            stage_def = STAGES[sid]

            if not entries:
                st.markdown(
                    '<div class="lb-wrap">'
                    f'<div class="lb-head">🏆 Stage {sid} — {stage_def["label"]}</div>'
                    '<div style="padding:12px 16px;font-size:13px;color:#6b7280">'
                    "No submissions yet — train a model and submit!"
                    "</div></div>",
                    unsafe_allow_html=True,
                )
                continue

            best_avg = entries[0]["avg_score"] if entries else 1
            icons = ["🥇", "🥈", "🥉"]
            cls_ = ["gold", "silver", "bronze"]

            rows = ""
            for i, e in enumerate(entries[:10]):
                icon = icons[i] if i < 3 else f"#{i+1}"
                cls = cls_[i] if i < 3 else ""
                bw = int(e["avg_score"] / best_avg * 100) if best_avg > 0 else 0
                passed_icon = "✅" if e.get("passed") else "❌"
                rows += (
                    f'<div class="lb-row">'
                    f'<div class="lb-rank {cls}">{icon}</div>'
                    f'<div class="lb-team">{e["team_name"]}</div>'
                    f'<div class="lb-bar-wrap"><div class="lb-bar-bg">'
                    f'<div class="lb-bar-fill" style="width:{bw}%"></div>'
                    f"</div></div>"
                    f'<div class="lb-score" style="min-width:90px;text-align:right">'
                    f'avg {e["avg_score"]:.1f} &nbsp;|&nbsp; max {e["max_score"]} '
                    f"&nbsp;{passed_icon}</div>"
                    f"</div>"
                )

            pass_label = (
                f'Pass: avg ≥ {stage_def["pass_avg"]}'
                if stage_def["pass_avg"] is not None
                else "Ranking only"
            )
            st.markdown(
                f'<div class="lb-wrap">'
                f'<div class="lb-head">🏆 Stage {sid} — {stage_def["label"]} '
                f'<span style="font-weight:400;font-size:12px;color:{COLORS["text_sub"]}">'
                f"({pass_label})</span></div>"
                f"{rows}</div>",
                unsafe_allow_html=True,
            )
