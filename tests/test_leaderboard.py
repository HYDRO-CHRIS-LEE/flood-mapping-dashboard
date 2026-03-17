# tests/test_leaderboard.py
import json
import os
import pytest
from utils.leaderboard import load_leaderboard, save_leaderboard, add_entry, get_sorted


@pytest.fixture
def lb_path(tmp_path):
    return str(tmp_path / "leaderboard.json")


def test_load_returns_empty_when_file_missing(lb_path):
    result = load_leaderboard(lb_path)
    assert result == {"held_out_event": "", "entries": []}


def test_save_and_load_roundtrip(lb_path):
    data = {"held_out_event": "dubai", "entries": [
        {"team": "Alpha", "f1": 0.8, "accuracy": 0.85,
         "precision": 0.9, "recall": 0.72, "features": ["SAR_VH"],
         "n_trees": 100, "max_depth": 5, "timestamp": "2026-03-17T10:00:00"}
    ]}
    save_leaderboard(lb_path, data)
    loaded = load_leaderboard(lb_path)
    assert loaded == data


def test_save_is_atomic(lb_path):
    """After save, no temp files should remain."""
    data = {"held_out_event": "dubai", "entries": []}
    save_leaderboard(lb_path, data)
    parent = os.path.dirname(lb_path)
    files = os.listdir(parent)
    assert len(files) == 1  # only leaderboard.json


def test_add_entry_keeps_best_f1_per_team(lb_path):
    data = {"held_out_event": "dubai", "entries": []}
    save_leaderboard(lb_path, data)

    add_entry(lb_path, "dubai", team="Team A", f1=0.7, accuracy=0.75,
              precision=0.8, recall=0.65, features=["SAR_VH"],
              n_trees=100, max_depth=5)
    add_entry(lb_path, "dubai", team="Team A", f1=0.85, accuracy=0.88,
              precision=0.9, recall=0.81, features=["SAR_VH", "NDWI"],
              n_trees=200, max_depth=10)

    result = load_leaderboard(lb_path)
    team_a_entries = [e for e in result["entries"] if e["team"] == "Team A"]
    assert len(team_a_entries) == 1
    assert team_a_entries[0]["f1"] == 0.85


def test_add_entry_normalizes_team_name(lb_path):
    data = {"held_out_event": "dubai", "entries": []}
    save_leaderboard(lb_path, data)

    add_entry(lb_path, "dubai", team="Team A", f1=0.7, accuracy=0.75,
              precision=0.8, recall=0.65, features=["SAR_VH"],
              n_trees=100, max_depth=5)
    add_entry(lb_path, "dubai", team="  team a  ", f1=0.9, accuracy=0.92,
              precision=0.93, recall=0.87, features=["SAR_VH", "NDWI"],
              n_trees=200, max_depth=10)

    result = load_leaderboard(lb_path)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["team"] == "Team A"
    assert result["entries"][0]["f1"] == 0.9


def test_get_sorted_returns_descending_f1(lb_path):
    data = {"held_out_event": "dubai", "entries": [
        {"team": "B", "f1": 0.6, "accuracy": 0.65, "precision": 0.7,
         "recall": 0.55, "features": [], "n_trees": 50, "max_depth": 3,
         "timestamp": ""},
        {"team": "A", "f1": 0.9, "accuracy": 0.92, "precision": 0.93,
         "recall": 0.87, "features": [], "n_trees": 100, "max_depth": 5,
         "timestamp": ""},
    ]}
    save_leaderboard(lb_path, data)
    sorted_entries = get_sorted(lb_path)
    assert sorted_entries[0]["team"] == "A"
    assert sorted_entries[1]["team"] == "B"
