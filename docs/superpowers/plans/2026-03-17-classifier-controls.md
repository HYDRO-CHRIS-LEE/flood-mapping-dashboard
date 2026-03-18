# Classifier Controls Expansion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 hyperparameters and 4 data preprocessing options to the AI Flood Classifier so students have more knobs to experiment with.

**Architecture:** All changes in `modules/module4_rf.py`. Add preprocessing helper functions, new UI controls in a "Data Preprocessing" container above the existing "Model Parameters" container, additional hyperparameters below the existing sliders, wire everything into the `train_rf` function, and update the results display.

**Tech Stack:** Streamlit, scikit-learn (RandomForestClassifier, StandardScaler, MinMaxScaler), pandas, numpy

---

### Task 1: Add preprocessing helper functions

**Files:**
- Modify: `modules/module4_rf.py` (add functions after `event_based_split`, before `train_rf`)

- [ ] **Step 1: Add sklearn imports**

At the top of `modules/module4_rf.py`, add to the existing sklearn imports:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

- [ ] **Step 2: Add `apply_preprocessing` function**

Add this function after `event_based_split` (line 38), before `train_rf`:

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add modules/module4_rf.py
git commit -m "feat: add preprocessing helper functions for classifier"
```

---

### Task 2: Update `train_rf` to accept new parameters

**Files:**
- Modify: `modules/module4_rf.py` (update `train_rf` function)

- [ ] **Step 1: Update `train_rf` signature and body**

Replace the existing `train_rf` function (lines 41-73) with:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add modules/module4_rf.py
git commit -m "feat: extend train_rf with new hyperparams and preprocessing"
```

---

### Task 3: Add UI controls and wire to training

**Files:**
- Modify: `modules/module4_rf.py` (update `render_module4` function)

- [ ] **Step 1: Add Data Preprocessing container**

In `render_module4`, inside `with col_l:`, BEFORE the existing `with st.container(border=True):` block for Model Parameters (line 303), add a new container:

```python
        # ── Data Preprocessing ────────────────────────────────────
        with st.container(border=True):
            st.markdown('<div class="control-title">Data Preprocessing</div>',
                        unsafe_allow_html=True)
            pp_c1, pp_c2 = st.columns(2)
            with pp_c1:
                scaling = st.selectbox(
                    "Feature Scaling",
                    ["None", "StandardScaler", "MinMaxScaler"],
                    help="Normalize value ranges — RF usually doesn't need this, but experiment!")
                outlier_method = st.selectbox(
                    "Outlier Removal",
                    ["None", "IQR method", "Z-score (>3σ)"],
                    help="Remove extreme values — how does it affect accuracy?")
            with pp_c2:
                balance = st.selectbox(
                    "Class Balance",
                    ["None", "Oversample minority", "Undersample majority"],
                    help="Handle flood/non-flood sample imbalance")
                sample_pct = st.slider(
                    "Sample Size (%)", 10, 100, 100, 10,
                    help="Use partial data — see how data volume affects performance")
```

- [ ] **Step 2: Add new hyperparameters to the existing Model Parameters container**

After the existing `max_depth` slider (line 320) and before `st.markdown("---")` (line 322), add:

```python
            min_leaf = st.slider("Min samples per leaf", 1, 20, 1,
                                 help="Larger = less overfitting")
            max_feat = st.selectbox(
                "Max features per split",
                ["sqrt", "log2", "all"],
                help="Features considered per split — diversity vs accuracy")
            bc1, bc2 = st.columns(2)
            with bc1:
                use_bootstrap = st.checkbox("Bootstrap", value=True,
                                            help="Each tree trains on a random subset")
            with bc2:
                use_class_wt = st.checkbox("Class weight", value=False,
                                           help="Auto-weight when flood samples are rare")
```

- [ ] **Step 3: Apply preprocessing before training**

Right before the `run = st.button(...)` line, add preprocessing application. And update the training call.

After the preprocessing container and model params container, before `disabled = ...`:

```python
        # Apply sample size + outlier removal to full dataset
        proc_df = apply_preprocessing(df, selected_features, sample_pct,
                                      outlier_method)
        if len(proc_df) < 10:
            st.warning("Too few samples after preprocessing.")
```

Then update the `train_rf` call inside `with col_r:` (around line 337-338) to pass all new parameters:

```python
                metrics, importance = train_rf(
                    proc_df, selected_features, n_trees, max_depth,
                    HELD_OUT_EVENT,
                    min_samples_leaf=min_leaf,
                    max_features_str=max_feat,
                    use_class_weight=use_class_wt,
                    bootstrap=use_bootstrap,
                    scaling=scaling,
                    balance=balance,
                )
```

Also update the `result` dict to include new params:

```python
                result = {
                    "metrics": metrics, "importance": importance,
                    "features": selected_features, "n_trees": n_trees,
                    "max_depth": max_depth, "elapsed": elapsed,
                    "min_leaf": min_leaf, "max_feat": max_feat,
                    "bootstrap": use_bootstrap, "class_weight": use_class_wt,
                    "scaling": scaling, "balance": balance,
                    "sample_pct": sample_pct, "outlier": outlier_method,
                }
```

- [ ] **Step 4: Update the results info line to show preprocessing**

After the existing train/test breakdown `st.markdown` (around line 407-415), add a preprocessing summary line:

```python
            # ── Preprocessing summary ─────────────────────────────
            pp_parts = []
            if res.get("scaling", "None") != "None":
                pp_parts.append(res["scaling"])
            if res.get("balance", "None") != "None":
                pp_parts.append(res["balance"])
            if res.get("outlier", "None") != "None":
                pp_parts.append(res["outlier"])
            if res.get("sample_pct", 100) < 100:
                pp_parts.append(f"{res['sample_pct']}% sample")
            if pp_parts:
                pp_str = " · ".join(pp_parts)
                st.markdown(f"""
                <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;
                            padding:6px 12px;background:var(--bg3);border-radius:8px">
                    <b>Preprocessing:</b> {pp_str}
                </div>
                """, unsafe_allow_html=True)
```

- [ ] **Step 5: Update run history to include new params**

Update the history append block (around line 356-366) to include the new parameters:

```python
                history.append({
                    "run_id": len(history) + 1,
                    "features": selected_features,
                    "n_trees": n_trees,
                    "max_depth": max_depth,
                    "min_leaf": min_leaf,
                    "max_feat": max_feat,
                    "f1": metrics["f1"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "scaling": scaling,
                    "balance": balance,
                    "timestamp": time.strftime("%H:%M:%S"),
                })
```

- [ ] **Step 6: Commit**

```bash
git add modules/module4_rf.py
git commit -m "feat: add preprocessing UI and hyperparameter controls to classifier"
```
