"""
AI Flood Classifier — Random Forest trained across ALL available flood events.
Train/test split is event-based to prevent spatial data leakage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from utils.data_loader import load_all_rf_samples, ALL_EVENTS
from utils.styles import COLORS
import time


@st.cache_resource
def get_leaderboard() -> list:
    return []


FEATURE_INFO = {
    "SAR_VH":         ("📡", "SAR VH",       "Radar backscatter (dB) — core water signal"),
    "NDWI":           ("💧", "NDWI",          "Optical water index — green / NIR ratio"),
    "MNDWI":          ("🏙️", "MNDWI",         "Modified water index — better in urban areas"),
    "elevation":      ("🏔️", "Elevation",     "Height above sea level (m)"),
    "slope":          ("📐", "Slope",          "Terrain steepness (°) — steep slopes rarely flood"),
    "permanent_water":("🌊", "Perm. Water",   "JRC permanent water flag"),
}
ALL_FEATURES = list(FEATURE_INFO.keys())


def event_based_split(df: pd.DataFrame, test_events: list[str]):
    """
    Split by event to avoid spatial leakage.
    Pixels from the same flood event are spatially correlated —
    a random row-level split would leak spatial information into the test set.
    """
    test_mask  = df["event"].isin(test_events)
    train_df   = df[~test_mask].copy()
    test_df    = df[test_mask].copy()
    return train_df, test_df


def train_rf(df: pd.DataFrame, features: list[str],
             n_trees: int, max_depth: int,
             test_events: list[str], seed: int = 42):
    X_all = df[features].values
    y_all = df["label"].values

    train_df, test_df = event_based_split(df, test_events)

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
        "accuracy":   accuracy_score(y_te, y_pred),
        "precision":  precision_score(y_te, y_pred, zero_division=0),
        "recall":     recall_score(y_te, y_pred, zero_division=0),
        "f1":         f1_score(y_te, y_pred, zero_division=0),
        "n_train":    len(X_tr),
        "n_test":     len(X_te),
        "train_events": [e for e in df["event"].unique() if e not in test_events],
        "test_events":  test_events,
        "cm":           cm.tolist(),
    }
    importance = dict(zip(features, clf.feature_importances_))
    return metrics, importance


# ── Leaderboard ───────────────────────────────────────────────────
def render_leaderboard(board: list):
    if not board:
        st.markdown(
            '<div class="lb-wrap">'
            '<div class="lb-head">🏆 Team Leaderboard</div>'
            '<div style="padding:12px 16px;font-size:13px;color:#6b7280">'
            'No submissions yet — train a model and hit <b>Submit</b>!'
            '</div></div>',
            unsafe_allow_html=True,
        )
        return

    sboard = sorted(board, key=lambda x: x["accuracy"], reverse=True)
    best   = sboard[0]["accuracy"]
    icons  = ["🥇","🥈","🥉"]
    cls_   = ["gold","silver","bronze"]

    rows = ""
    for i, e in enumerate(sboard[:10]):
        icon = icons[i] if i < 3 else f"#{i+1}"
        cls  = cls_[i]  if i < 3 else ""
        bw   = int(e["accuracy"] / best * 100)
        rows += (
            f'<div class="lb-row">'
            f'<div class="lb-rank {cls}">{icon}</div>'
            f'<div class="lb-team">{e["team"]}</div>'
            f'<div class="lb-event">{e.get("test_events","")}</div>'
            f'<div class="lb-bar-wrap"><div class="lb-bar-bg">'
            f'<div class="lb-bar-fill" style="width:{bw}%"></div>'
            f'</div></div>'
            f'<div class="lb-score">{e["accuracy"]*100:.1f}%</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="lb-wrap">'
        f'<div class="lb-head">🏆 Team Leaderboard</div>'
        f'{rows}</div>',
        unsafe_allow_html=True,
    )


# ── Feature importance (Plotly — no HTML parser issues) ──────────
def render_feature_importance(importance: dict):
    total  = sum(importance.values()) or 1
    items  = sorted(importance.items(), key=lambda x: x[1])   # ascending for horizontal bar
    labels = [f"{FEATURE_INFO[f][0]} {FEATURE_INFO[f][1]}" if f in FEATURE_INFO else f
              for f, _ in items]
    values = [v / total * 100 for _, v in items]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0,COLORS["blue_light"]],[0.5,COLORS["blue"]],[1,COLORS["blue_dark"]]],
            showscale=False,
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text_sub"]),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
        font=dict(family="Inter", color=COLORS["text"]),
        height=max(180, len(items) * 38),
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor=COLORS["bg3"], ticksuffix="%",
                   tickfont=dict(color=COLORS["axis"], size=10), range=[0, max(values)*1.2]),
        yaxis=dict(showgrid=False, tickfont=dict(color=COLORS["text_sub"], size=12)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Confusion matrix (Plotly) ─────────────────────────────────────
def render_confusion_matrix(cm: list):
    cm_arr = np.array(cm)
    labels = ["Non-flood", "Flood"]
    fig = go.Figure(go.Heatmap(
        z=cm_arr, x=labels, y=labels,
        colorscale=[[0,"#eff6ff"],[1,COLORS["blue_dark"]]],
        showscale=False,
        text=cm_arr, texttemplate="%{text}",
        textfont=dict(size=16, color=COLORS["bg"]),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
        font=dict(family="Inter", color=COLORS["text"]),
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Predicted", tickfont=dict(size=11)),
        yaxis=dict(title="Actual",    tickfont=dict(size=11), autorange="reversed"),
        title=dict(text="Confusion Matrix", font=dict(size=12, color=COLORS["text_sub"]), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Main render ───────────────────────────────────────────────────
def render_module4(available_events: list[str]):

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🤖</div>
        <div>
            <div class="section-title">AI Flood Classifier</div>
            <div class="section-desc">
                Train a Random Forest model across multiple flood events —
                tune parameters and compete for the best accuracy.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Context callout ───────────────────────────────────────────
    st.markdown("""
    <div class="callout">
        <strong>From Observation to Prediction</strong> &nbsp;
        In the previous tabs you explored individual flood events using satellite data.
        Now we combine data from <em>all available events</em> to train an AI model that generalizes
        across different geographies and flood types.
        The train/test split is <strong>event-based</strong> — the model is trained on some events
        and evaluated on held-out events it has never seen, preventing spatial data leakage.
    </div>
    """, unsafe_allow_html=True)

    # ── Load combined data ────────────────────────────────────────
    with st.spinner("Loading training data from all events..."):
        df = load_all_rf_samples(available_events)

    if df is None or len(df) == 0:
        st.markdown("""
        <div class="callout warn">
            <strong>No training data found</strong><br>
            Run the Colab export notebook for at least one event to generate
            <code>RF_training_samples.csv</code> files.
        </div>
        """, unsafe_allow_html=True)
        return

    drop_cols = [c for c in df.columns if c not in ALL_FEATURES + ["label", "event"]]
    df = df.drop(columns=drop_cols, errors="ignore").dropna()

    ev_counts = df.groupby("event").size().to_dict()
    ev_list   = list(ev_counts.keys())

    # ── Data summary ──────────────────────────────────────────────
    n_flood    = int((df["label"] == 1).sum())
    n_nonflood = int((df["label"] == 0).sum())
    n_events   = len(ev_list)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-unit">across all events</div>
        </div>
        <div class="metric-card red">
            <div class="metric-label">Flood Samples</div>
            <div class="metric-value">{n_flood:,}</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Non-flood Samples</div>
            <div class="metric-value">{n_nonflood:,}</div>
        </div>
        <div class="metric-card indigo">
            <div class="metric-label">Events Loaded</div>
            <div class="metric-value">{n_events}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameter panel + results ─────────────────────────────────
    col_l, col_r = st.columns([1, 1.5])

    with col_l:
        with st.container(border=True):
            st.markdown('<div class="control-title">Model Parameters</div>',
                        unsafe_allow_html=True)

            # Feature selection
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
            n_trees   = st.slider("Number of trees", 10, 300, 100, 10)
            max_depth = st.slider("Max tree depth (0 = unlimited)", 0, 20, 5)

            # Event-based test split selector
            st.markdown("---")
            st.markdown("**Test events** (held-out for evaluation)")
            st.caption("These events are NOT used in training — simulates real-world generalization.")

            ev_labels_map = {k: f"{ALL_EVENTS[k]['label']} ({ALL_EVENTS[k]['year']})"
                             for k in ev_list}
            test_events = st.multiselect(
                "Hold out for testing",
                options=ev_list,
                default=ev_list[:1] if ev_list else [],
                format_func=lambda k: ev_labels_map.get(k, k),
                label_visibility="collapsed",
            )

        disabled = len(selected_features) < 2 or len(test_events) == 0 or \
                   len(test_events) >= len(ev_list)
        if len(selected_features) < 2:
            st.warning("Select at least 2 features.")
        elif len(test_events) == 0:
            st.warning("Select at least 1 test event.")
        elif len(test_events) >= len(ev_list):
            st.warning("Keep at least 1 event for training.")

        run = st.button("🚀 Train AI Model!", disabled=disabled,
                        use_container_width=True)

    with col_r:
        if run and not disabled:
            with st.spinner(f"Training Random Forest on {len(ev_list)-len(test_events)} events..."):
                t0 = time.time()
                metrics, importance = train_rf(
                    df, selected_features, n_trees, max_depth, test_events
                )
                elapsed = time.time() - t0

            if metrics is None:
                st.error("Training failed — check that test events have data.")
            else:
                st.session_state["last_result"] = {
                    "metrics": metrics, "importance": importance,
                    "features": selected_features, "n_trees": n_trees,
                    "max_depth": max_depth, "elapsed": elapsed,
                }

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            m   = res["metrics"]
            acc = m["accuracy"]

            clr  = "green" if acc >= 0.90 else "blue" if acc >= 0.80 else "yellow" if acc >= 0.70 else "red"
            emoj = "🏆" if acc >= 0.90 else "⭐" if acc >= 0.80 else "📈" if acc >= 0.70 else "🔧"

            # Score cards
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card {clr}">
                    <div class="metric-label">{emoj} Accuracy</div>
                    <div class="metric-value">{acc*100:.1f}%</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{m['precision']*100:.1f}%</div>
                </div>
                <div class="metric-card green">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">{m['recall']*100:.1f}%</div>
                </div>
                <div class="metric-card indigo">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">{m['f1']*100:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Train/test event breakdown
            train_ev_str = ", ".join([ALL_EVENTS[e]["label"] for e in m["train_events"]
                                      if e in ALL_EVENTS])
            test_ev_str  = ", ".join([ALL_EVENTS[e]["label"] for e in m["test_events"]
                                      if e in ALL_EVENTS])
            st.markdown(f"""
            <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;
                        padding:8px 12px;background:var(--bg3);border-radius:8px">
                <b>Trained on:</b> {train_ev_str} ({m['n_train']} samples)
                &nbsp;·&nbsp;
                <b>Tested on:</b> {test_ev_str} ({m['n_test']} samples)
                &nbsp;·&nbsp; {res['elapsed']:.1f}s
            </div>
            """, unsafe_allow_html=True)

            # Two-column: bar chart + confusion matrix
            cc1, cc2 = st.columns(2)
            with cc1:
                with st.container(border=True):
                    st.markdown('<div class="card-title">Performance</div>',
                                unsafe_allow_html=True)
                    fig = go.Figure(go.Bar(
                        x=["Accuracy","Precision","Recall","F1"],
                        y=[m["accuracy"],m["precision"],m["recall"],m["f1"]],
                        marker_color=[COLORS["blue"],COLORS["cyan"],COLORS["green_light"],COLORS["indigo"]],
                        text=[f"{v*100:.1f}%" for v in [m["accuracy"],m["precision"],m["recall"],m["f1"]]],
                        textposition="outside", textfont=dict(size=11, color=COLORS["text_sub"]),
                    ))
                    fig.add_hline(y=0.9, line_dash="dot", line_color=COLORS["yellow"],
                                  annotation_text="  Target 90%",
                                  annotation_font_color=COLORS["yellow"], annotation_font_size=10)
                    fig.update_layout(
                        plot_bgcolor=COLORS["bg"], paper_bgcolor=COLORS["bg"],
                        font=dict(family="Inter"), height=190,
                        margin=dict(l=5,r=5,t=5,b=5),
                        yaxis=dict(range=[0,1.15], tickformat=".0%",
                                   showgrid=True, gridcolor=COLORS["bg3"],
                                   tickfont=dict(color=COLORS["axis"], size=10)),
                        xaxis=dict(showgrid=False, tickfont=dict(color=COLORS["text_sub"],size=11)),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

            with cc2:
                with st.container(border=True):
                    st.markdown('<div class="card-title">Confusion Matrix</div>',
                                unsafe_allow_html=True)
                    render_confusion_matrix(m["cm"])

            # Feature importance
            with st.container(border=True):
                st.markdown('<div class="card-title">Feature Importance — What the AI relied on most</div>',
                            unsafe_allow_html=True)
                render_feature_importance(res["importance"])

            # Submit to leaderboard
            board = get_leaderboard()
            if st.button("📤 Submit to Leaderboard", use_container_width=True):
                board.append({
                    "team":        st.session_state.get("team_name", "Team A"),
                    "accuracy":    acc,
                    "test_events": ", ".join(m["test_events"]),
                    "features":    ", ".join(res["features"]),
                    "time":        time.strftime("%H:%M:%S"),
                })
                st.success(f"Submitted! Accuracy: {acc*100:.1f}%")

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

    # ── Leaderboard ───────────────────────────────────────────────
    st.markdown("---")
    render_leaderboard(get_leaderboard())

    # ── Tips ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="callout">
        <strong>Tips to Improve Your Score</strong> &nbsp;
        Try adding more trees. Does combining SAR_VH with elevation help?
        Try testing on a geographically different event from your training set — does performance drop?<br><br>
        <strong>Think About It</strong> &nbsp;
        What does low Recall mean for flood prediction?
        In a real disaster, would you rather have high Precision or high Recall — and why?
        What happens when you test on a continent different from where you trained?
    </div>
    """, unsafe_allow_html=True)
