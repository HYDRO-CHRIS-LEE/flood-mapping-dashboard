"""Event-wise normalization and validation for RF training data."""

import pandas as pd

EXCLUDE_FROM_NORM = {"label", "event", "system:index", ".geo", "permanent_water"}


def validate_events(
    df: pd.DataFrame, min_per_class: int = 30
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Exclude events where flood or non-flood sample count < min_per_class.
    Returns (valid_df, list of (event_name, reason) tuples).
    """
    excluded: list[tuple[str, str]] = []
    keep_events: list[str] = []

    for ev, grp in df.groupby("event"):
        n_flood = int((grp["label"] == 1).sum())
        n_nonflood = int((grp["label"] == 0).sum())
        if n_flood < min_per_class:
            excluded.append((ev, f"Only {n_flood} flood samples (need {min_per_class})"))
        elif n_nonflood < min_per_class:
            excluded.append((ev, f"Only {n_nonflood} non-flood samples (need {min_per_class})"))
        else:
            keep_events.append(ev)

    valid_df = df[df["event"].isin(keep_events)].copy()
    return valid_df, excluded


def normalize_by_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Event-wise z-score: (value - event_mean) / event_std for all numeric
    columns not in EXCLUDE_FROM_NORM.  Returns a new DataFrame (original is
    not modified).
    """
    result = df.copy()
    numeric_cols = [
        c for c in result.select_dtypes(include="number").columns
        if c not in EXCLUDE_FROM_NORM
    ]

    for col in numeric_cols:
        stats = result.groupby("event")[col].agg(["mean", "std"])
        for ev in stats.index:
            mask = result["event"] == ev
            mean = stats.loc[ev, "mean"]
            std = stats.loc[ev, "std"]
            if std == 0 or pd.isna(std):
                result.loc[mask, col] = 0.0
            else:
                result.loc[mask, col] = (result.loc[mask, col] - mean) / std

    return result
