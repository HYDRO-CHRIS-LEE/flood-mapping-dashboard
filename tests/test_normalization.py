# tests/test_normalization.py
import pandas as pd
import pytest
from utils.normalization import validate_events


def _make_df(events_config: dict) -> pd.DataFrame:
    """Helper: events_config = {"harvey": (n_flood, n_nonflood), ...}"""
    rows = []
    for ev, (nf, nnf) in events_config.items():
        for _ in range(nf):
            rows.append({"event": ev, "label": 1, "SAR_VH": -15.0,
                         "NDWI": 0.1, "elevation": 10.0, "slope": 1.0,
                         "MNDWI": 0.05, "permanent_water": 0})
        for _ in range(nnf):
            rows.append({"event": ev, "label": 0, "SAR_VH": -10.0,
                         "NDWI": -0.2, "elevation": 20.0, "slope": 2.0,
                         "MNDWI": -0.1, "permanent_water": 0})
    return pd.DataFrame(rows)


def test_validate_excludes_low_sample_events():
    df = _make_df({"harvey": (50, 50), "la2025": (3, 327)})
    valid_df, excluded = validate_events(df, min_per_class=30)
    assert "la2025" not in valid_df["event"].values
    assert "harvey" in valid_df["event"].values
    assert len(excluded) == 1
    assert excluded[0][0] == "la2025"


def test_validate_keeps_all_when_above_threshold():
    df = _make_df({"harvey": (50, 50), "pakistan": (40, 40)})
    valid_df, excluded = validate_events(df, min_per_class=30)
    assert set(valid_df["event"].unique()) == {"harvey", "pakistan"}
    assert len(excluded) == 0


def test_validate_returns_reason_string():
    df = _make_df({"harvey": (50, 50), "bad": (5, 100)})
    _, excluded = validate_events(df, min_per_class=30)
    event_name, reason = excluded[0]
    assert event_name == "bad"
    assert "flood" in reason.lower() or "5" in reason


import numpy as np
from utils.normalization import normalize_by_event, EXCLUDE_FROM_NORM


def test_normalize_centers_each_event_to_zero_mean():
    df = _make_df({"harvey": (50, 50), "pakistan": (50, 50)})
    harvey_mask = df["event"] == "harvey"
    df.loc[harvey_mask, "elevation"] = np.random.RandomState(42).normal(18, 5, harvey_mask.sum())
    df.loc[~harvey_mask, "elevation"] = np.random.RandomState(42).normal(535, 100, (~harvey_mask).sum())

    result = normalize_by_event(df)

    for ev in ["harvey", "pakistan"]:
        ev_elev = result.loc[result["event"] == ev, "elevation"]
        assert abs(ev_elev.mean()) < 0.01, f"{ev} mean should be ~0, got {ev_elev.mean()}"
        assert abs(ev_elev.std() - 1.0) < 0.1, f"{ev} std should be ~1, got {ev_elev.std()}"


def test_normalize_excludes_specified_columns():
    df = _make_df({"harvey": (50, 50)})
    result = normalize_by_event(df)
    assert (result["permanent_water"] == 0).all()
    assert set(result["label"].unique()) <= {0, 1}
    assert result["event"].dtype == object


def test_normalize_handles_zero_std_column():
    df = _make_df({"harvey": (50, 50)})
    df["slope"] = 0.0
    result = normalize_by_event(df)
    assert (result["slope"] == 0.0).all()


def test_normalize_does_not_modify_original():
    df = _make_df({"harvey": (50, 50)})
    original_elev = df["elevation"].copy()
    _ = normalize_by_event(df)
    pd.testing.assert_series_equal(df["elevation"], original_elev)
