"""
Random Forest flood classifier engine.
Ported from modules/module4_rf.py — pure functions, no Streamlit dependency.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from backend.config import DATA_ROOT
from backend.services.normalization import normalize_by_event, validate_events

# ── Constants ──────────────────────────────────────────────────────

HELD_OUT_EVENTS = ["dubai", "germany2021", "libya2023", "china2020"]

FEATURE_INFO = {
    "NDWI":            ("water_drop",      "NDWI",         "Optical water index -- green / NIR ratio"),
    "MNDWI":           ("location_city",   "MNDWI",        "Modified water index -- better in urban areas"),
    "elevation":       ("landscape",       "Elevation",    "Height above sea level (m)"),
    "slope":           ("square_foot",     "Slope",        "Terrain steepness (deg) -- steep slopes rarely flood"),
    "permanent_water": ("waves",           "Perm. Water",  "JRC permanent water flag"),
}

ALL_FEATURES = list(FEATURE_INFO.keys())


# ── Data loading ───────────────────────────────────────────────────

def load_training_data(available_events: list[str]) -> pd.DataFrame | None:
    """Load RF_training_samples CSVs from available events, adding event column."""
    frames = []
    for ev in available_events:
        path = os.path.join(DATA_ROOT, ev, "RF_training_samples.csv")
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        df["event"] = ev
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ── Train/test split ──────────────────────────────────────────────

def event_based_split(
    df: pd.DataFrame, held_out_events: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by event membership."""
    test_mask = df["event"].isin(held_out_events)
    return df[~test_mask].copy(), df[test_mask].copy()


# ── Preprocessing ─────────────────────────────────────────────────

def apply_preprocessing(
    df: pd.DataFrame,
    features: list[str],
    sample_pct: int,
    outlier_method: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply stratified sampling + outlier removal (IQR or zscore)."""
    result = df.copy()

    # 1. Sample size reduction (stratified by label)
    if sample_pct < 100:
        frac = sample_pct / 100
        result = (
            result.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=seed))
            .reset_index(drop=True)
        )

    # 2. Outlier removal
    if outlier_method == "IQR method":
        for f in features:
            q1 = result[f].quantile(0.25)
            q3 = result[f].quantile(0.75)
            iqr = q3 - q1
            mask = (result[f] >= q1 - 1.5 * iqr) & (result[f] <= q3 + 1.5 * iqr)
            result = result[mask]
    elif outlier_method == "Z-score (>3s)":
        for f in features:
            z = (result[f] - result[f].mean()) / (result[f].std() + 1e-8)
            result = result[np.abs(z) <= 3]

    return result.reset_index(drop=True)


# ── Class balancing ───────────────────────────────────────────────

def apply_class_balance(
    df: pd.DataFrame, method: str, seed: int = 42
) -> pd.DataFrame:
    """Balance classes: none / oversample / undersample."""
    if method == "none":
        return df

    flood = df[df["label"] == 1]
    nonflood = df[df["label"] == 0]

    if method == "oversample":
        if len(flood) < len(nonflood) and len(flood) > 0:
            flood = flood.sample(n=len(nonflood), replace=True, random_state=seed)
        elif len(nonflood) < len(flood) and len(nonflood) > 0:
            nonflood = nonflood.sample(n=len(flood), replace=True, random_state=seed)
    elif method == "undersample":
        if len(flood) < len(nonflood):
            nonflood = nonflood.sample(n=len(flood), random_state=seed)
        elif len(nonflood) < len(flood):
            flood = flood.sample(n=len(nonflood), random_state=seed)

    return pd.concat([flood, nonflood]).reset_index(drop=True)


# ── Full training pipeline ────────────────────────────────────────

def train_rf(
    df: pd.DataFrame,
    features: list[str],
    n_trees: int,
    max_depth: int,
    held_out_events: list[str],
    min_samples_leaf: int = 1,
    max_features_str: str = "sqrt",
    use_class_weight: bool = False,
    bootstrap: bool = True,
    scaling: str = "none",
    balance: str = "none",
    sample_pct: int = 100,
    outlier_method: str = "none",
    seed: int = 42,
) -> tuple[dict | None, dict | None, RandomForestClassifier | None, StandardScaler | MinMaxScaler | None]:
    """
    Full RF pipeline: preprocess, split, balance, scale, train, evaluate.
    Returns (metrics_dict, importance_dict, clf, scaler) or (None, None, None, None) on failure.
    """
    # Drop columns not needed and remove NaNs
    keep_cols = [c for c in df.columns if c in features + ["label", "event"]]
    df = df[keep_cols].dropna()

    # Validate and normalize
    valid_df, _ = validate_events(df)
    if len(valid_df) == 0:
        return None, None, None, None
    df = normalize_by_event(valid_df)

    # Preprocessing (sampling + outliers)
    df = apply_preprocessing(df, features, sample_pct, outlier_method, seed)

    # Split
    train_df, test_df = event_based_split(df, held_out_events)
    if len(train_df) == 0 or len(test_df) == 0:
        return None, None, None, None

    # Class balance (training set only)
    train_df = apply_class_balance(train_df, balance, seed)

    X_tr = train_df[features].values
    y_tr = train_df["label"].values
    X_te = test_df[features].values
    y_te = test_df["label"].values

    # Feature scaling (fit on train, transform both)
    scaler = None
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "minmax":
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
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    cm = confusion_matrix(y_te, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "train_events": [e for e in df["event"].unique() if e not in held_out_events],
        "test_events": held_out_events,
        "cm": cm.tolist(),
    }
    importance = {f: float(v) for f, v in zip(features, clf.feature_importances_)}
    return metrics, importance, clf, scaler


# ── Coaching hints ────────────────────────────────────────────────

def generate_hints(
    metrics: dict, features: list[str], n_trees: int
) -> list[str]:
    """Return up to 2 prioritized hints based on model results."""
    rules = [
        (
            len(features) == 1,
            "You're using only one feature. Try combining different types of information "
            "(e.g. radar + terrain) for better results.",
        ),
        (
            metrics["recall"] < 0.6,
            "Recall is low -- the model is missing many actual flood areas. "
            "Try adding more features like NDWI or MNDWI to capture water signals.",
        ),
        (
            metrics["precision"] < 0.6,
            "Precision is low -- the model is predicting flood in non-flood areas. "
            "Try adding elevation or slope to help distinguish terrain.",
        ),
        (
            n_trees < 30 and metrics["f1"] < 0.7,
            "The number of trees is low. Try increasing to 50-100.",
        ),
        (
            metrics["f1"] > 0.85,
            "Great job! Try reducing the number of features -- achieving similar "
            "performance with less data means a better model.",
        ),
    ]
    return [msg for cond, msg in rules if cond][:2]
