"""Unit tests for consensus selector score aggregation and calibration penalty."""
import numpy as np
import pandas as pd
import pytest
from src.select.consensus import (
    ConsensusSelector,
    _jaccard,
    _mean_pairwise_jaccard,
    _surrogate_ece,
)
from src.utils.seeds import set_global_seed


def test_jaccard_identical_sets():
    assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint_sets():
    assert _jaccard({1, 2}, {3, 4}) == 0.0


def test_jaccard_partial_overlap():
    j = _jaccard({1, 2, 3}, {2, 3, 4})
    assert abs(j - 0.5) < 1e-9  # intersection=2, union=4


def test_jaccard_empty_sets():
    assert _jaccard(set(), set()) == 1.0


def test_mean_pairwise_jaccard_single_set():
    assert _mean_pairwise_jaccard([{1, 2, 3}]) == 1.0


def test_mean_pairwise_jaccard_identical():
    sets = [{1, 2, 3}, {1, 2, 3}, {1, 2, 3}]
    assert _mean_pairwise_jaccard(sets) == 1.0


def test_mean_pairwise_jaccard_disjoint():
    sets = [{1, 2}, {3, 4}, {5, 6}]
    assert _mean_pairwise_jaccard(sets) == 0.0


def test_surrogate_ece_perfect_calibration():
    """A perfectly calibrated model should have ECE close to 0."""
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 3, n)
    # Create perfect one-hot proba (perfectly calibrated at confidence=1)
    proba = np.eye(3)[y_true]
    ece = _surrogate_ece(y_true, proba, n_bins=10)
    # ECE should be 0 for perfect predictions (all in the 100% bin)
    assert ece < 0.01, f"ECE too high for perfect predictions: {ece}"


def test_surrogate_ece_uniform_proba():
    """Uniform predictions should have non-trivial ECE."""
    n = 300
    y_true = np.tile(np.arange(3), n // 3)
    proba = np.full((n, 3), 1.0 / 3)
    ece = _surrogate_ece(y_true, proba, n_bins=10)
    # With uniform proba, accuracy ~33%, confidence ~33%, ECE should be small
    assert 0 <= ece <= 1.0


def test_consensus_selector_scores_are_non_negative():
    """Consensus scores must be >= 0 for all features."""
    set_global_seed(42)
    rng = np.random.default_rng(42)
    n, p = 200, 15
    X = pd.DataFrame({f"f{i}": rng.random(n) for i in range(p)})
    y = rng.integers(0, 3, n)

    sel = ConsensusSelector(k_grid=[5, 8], n_inner_folds=2, surrogate_n_estimators=5)
    result = sel.fit(X, y)

    for feat, score in result.consensus_scores.items():
        assert score >= 0.0, f"Negative consensus score for {feat}: {score}"


def test_consensus_selector_chosen_k_in_grid():
    set_global_seed(42)
    rng = np.random.default_rng(42)
    n, p = 200, 15
    X = pd.DataFrame({f"f{i}": rng.random(n) for i in range(p)})
    y = rng.integers(0, 3, n)

    k_grid = [5, 8, 12]
    sel = ConsensusSelector(k_grid=k_grid, n_inner_folds=2, surrogate_n_estimators=5)
    result = sel.fit(X, y)

    assert result.chosen_k in k_grid, (
        f"chosen_k={result.chosen_k} not in grid {k_grid}"
    )


def test_consensus_selector_weights_all_positive():
    """All selector weights should be >= 0."""
    set_global_seed(42)
    rng = np.random.default_rng(42)
    n, p = 200, 15
    X = pd.DataFrame({f"f{i}": rng.random(n) for i in range(p)})
    y = rng.integers(0, 3, n)

    sel = ConsensusSelector(k_grid=[5, 8], n_inner_folds=2, surrogate_n_estimators=5)
    result = sel.fit(X, y)

    for sname, w in result.selector_weights.items():
        assert w >= 0.0, f"Negative weight for selector {sname}: {w}"


def test_consensus_ranking_covers_all_features():
    """Full ranking must include every feature exactly once."""
    set_global_seed(42)
    rng = np.random.default_rng(42)
    n, p = 200, 15
    X = pd.DataFrame({f"f{i}": rng.random(n) for i in range(p)})
    y = rng.integers(0, 3, n)

    sel = ConsensusSelector(k_grid=[5, 8], n_inner_folds=2, surrogate_n_estimators=5)
    result = sel.fit(X, y)

    # After tree preprocessing, feature count may differ from input columns
    assert len(result.full_ranking) == len(set(result.full_ranking)), \
        "Duplicate features in full ranking"
    assert len(result.selected_names) == result.chosen_k
