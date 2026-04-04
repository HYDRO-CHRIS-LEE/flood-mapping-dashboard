"""Tests for the RF-based Flappy Bird game engine."""

from backend.services.rf_game_engine import (
    STAGES,
    STAGE_SEEDS,
    get_model_status,
    load_model_artifact,
)


def test_stages_defined():
    assert len(STAGES) == 5
    assert STAGES[1]["pass_avg"] == 5
    assert STAGES[5]["pass_avg"] is None


def test_stage_seeds_defined():
    assert len(STAGE_SEEDS) == 5
    assert len(STAGE_SEEDS[1]) == 10


def test_model_status_no_model():
    status = get_model_status("nonexistent_team_xyz")
    assert status["has_model"] is False


def test_load_artifact_missing():
    result = load_model_artifact("nonexistent_team_xyz")
    assert result is None
