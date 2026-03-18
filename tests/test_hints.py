# tests/test_hints.py
from modules.module4_rf import generate_hints


def test_single_feature_hint_is_highest_priority():
    hints = generate_hints(
        metrics={"recall": 0.5, "precision": 0.5, "f1": 0.4},
        features=["SAR_VH"],
        n_trees=100,
    )
    assert len(hints) >= 1
    assert "only one feature" in hints[0]


def test_low_recall_hint():
    hints = generate_hints(
        metrics={"recall": 0.4, "precision": 0.9, "f1": 0.55},
        features=["SAR_VH", "NDWI"],
        n_trees=100,
    )
    assert any("Recall" in h for h in hints)


def test_high_f1_congratulation():
    hints = generate_hints(
        metrics={"recall": 0.9, "precision": 0.9, "f1": 0.9},
        features=["SAR_VH", "NDWI", "elevation"],
        n_trees=100,
    )
    assert any("Great" in h for h in hints)


def test_max_two_hints():
    hints = generate_hints(
        metrics={"recall": 0.3, "precision": 0.3, "f1": 0.3},
        features=["SAR_VH"],
        n_trees=20,
    )
    assert len(hints) <= 2
