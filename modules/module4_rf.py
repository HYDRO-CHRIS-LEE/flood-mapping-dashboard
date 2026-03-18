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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.data_loader import load_all_rf_samples, ALL_EVENTS
from utils.normalization import validate_events, normalize_by_event
from utils.leaderboard import add_entry, get_sorted
from utils.styles import COLORS
import os
import time

HELD_OUT_EVENT = "dubai"
LEADERBOARD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "leaderboard.json"
)

FEATURE_INFO = {
    "SAR_VH":         ("📡", "SAR VH",       "Radar backscatter (dB) — core water signal"),
    "NDWI":           ("💧", "NDWI",          "Optical water index — green / NIR ratio"),
    "MNDWI":          ("🏙️", "MNDWI",         "Modified water index — better in urban areas"),
    "elevation":      ("🏔️", "Elevation",     "Height above sea level (m)"),
    "slope":          ("📐", "Slope",          "Terrain steepness (°) — steep slopes rarely flood"),
    "permanent_water":("🌊", "Perm. Water",   "JRC permanent water flag"),
}
ALL_FEATURES = list(FEATURE_INFO.keys())


def event_based_split(df: pd.DataFrame, held_out_event: str):
    test_mask = df["event"] == held_out_event
    return df[~test_mask].copy(), df[test_mask].copy()


def apply_preprocessing(df: pd.DataFrame, features: list[str],
                        sample_pct: int, outlier_method: str,
                        seed: int = 42) -> pd.DataFrame:
    """Apply sample size reduction and outlier removal to the full dataset."""
    result = df.copy()

    # 1. Sample size reduction (stratified by label)
    if sample_pct < 100:
        frac = sample_pct / 100
        result = result.groupby("label", group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=seed)
        ).reset_index(drop=True)

    # 2. Outlier removal
    if outlier_method == "IQR method":
        for f in features:
            q1 = result[f].quantile(0.25)
            q3 = result[f].quantile(0.75)
            iqr = q3 - q1
            mask = (result[f] >= q1 - 1.5 * iqr) & (result[f] <= q3 + 1.5 * iqr)
            result = result[mask]
    elif outlier_method == "Z-score (>3σ)":
        for f in features:
            z = (result[f] - result[f].mean()) / (result[f].std() + 1e-8)
            result = result[np.abs(z) <= 3]

    return result.reset_index(drop=True)


def apply_class_balance(df: pd.DataFrame, method: str,
                        seed: int = 42) -> pd.DataFrame:
    """Balance classes in training data only."""
    if method == "None":
        return df

    flood = df[df["label"] == 1]
    nonflood = df[df["label"] == 0]

    if method == "Oversample minority":
        if len(flood) < len(nonflood) and len(flood) > 0:
            flood = flood.sample(n=len(nonflood), replace=True, random_state=seed)
        elif len(nonflood) < len(flood) and len(nonflood) > 0:
            nonflood = nonflood.sample(n=len(flood), replace=True, random_state=seed)
    elif method == "Undersample majority":
        if len(flood) < len(nonflood):
            nonflood = nonflood.sample(n=len(flood), random_state=seed)
        elif len(nonflood) < len(flood):
            flood = flood.sample(n=len(nonflood), random_state=seed)

    return pd.concat([flood, nonflood]).reset_index(drop=True)


def train_rf(df: pd.DataFrame, features: list[str],
             n_trees: int, max_depth: int,
             held_out_event: str,
             min_samples_leaf: int = 1,
             max_features_str: str = "sqrt",
             use_class_weight: bool = False,
             bootstrap: bool = True,
             scaling: str = "None",
             balance: str = "None",
             seed: int = 42):
    train_df, test_df = event_based_split(df, held_out_event)

    if len(train_df) == 0 or len(test_df) == 0:
        return None, None

    # Class balance (training set only)
    train_df = apply_class_balance(train_df, balance, seed)

    X_tr = train_df[features].values; y_tr = train_df["label"].values
    X_te = test_df[features].values;  y_te = test_df["label"].values

    # Feature scaling (fit on train, transform both)
    scaler = None
    if scaling == "StandardScaler":
        scaler = StandardScaler()
    elif scaling == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    # Map max_features string to sklearn param
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


# ── Main render ───────────────────────────────────────────────────
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
        reversed_history = list(reversed(history))
        rows_html = ""
        for i, run in enumerate(reversed_history):
            feat_str = ", ".join(
                FEATURE_INFO[f][1] if f in FEATURE_INFO else f
                for f in run["features"]
            )
            f1_pct = f"{run['f1']*100:.1f}%"
            acc_pct = f"{run['accuracy']*100:.1f}%"

            if i < len(reversed_history) - 1:
                prev = reversed_history[i + 1]
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
