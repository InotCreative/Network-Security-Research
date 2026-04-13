"""Unit tests for meta-feature construction and entropy functions."""
import numpy as np
import pytest
from src.ensemble.meta_features import (
    build_meta_features,
    entropy,
    top_two_margin,
    vote_entropy,
    meta_feature_names,
)


@pytest.fixture
def fake_probas():
    rng = np.random.default_rng(0)
    n, C, M = 50, 10, 4
    probas = []
    for _ in range(M):
        p = rng.dirichlet(np.ones(C), size=n)
        probas.append(p)
    return probas, n, C, M


def test_entropy_shape(fake_probas):
    probas, n, C, M = fake_probas
    h = entropy(probas[0])
    assert h.shape == (n,)
    assert np.all(h >= 0)


def test_entropy_uniform_max():
    """Uniform distribution maximises entropy."""
    C = 10
    p_uniform = np.full((1, C), 1.0 / C)
    p_peaked = np.eye(1, C, 0)
    assert entropy(p_uniform)[0] > entropy(p_peaked)[0]


def test_top_two_margin_non_negative(fake_probas):
    probas, n, C, M = fake_probas
    m = top_two_margin(probas[0])
    assert np.all(m >= -1e-9)


def test_build_meta_features_shape(fake_probas):
    probas, n, C, M = fake_probas
    meta = build_meta_features(probas)
    assert meta.shape[0] == n
    assert meta.shape[1] > 0
    assert not np.isnan(meta).any()


def test_build_meta_features_with_path_gap(fake_probas):
    probas, n, C, M = fake_probas
    p_w = probas[0]
    p_s = probas[1]
    meta = build_meta_features(probas, p_weighted=p_w, p_stack=p_s)
    meta_no_gap = build_meta_features(probas)
    assert meta.shape[1] > meta_no_gap.shape[1]


def test_meta_feature_names_length(fake_probas):
    probas, n, C, M = fake_probas
    meta = build_meta_features(probas)
    names = meta_feature_names(C, M, include_path_gap=False)
    assert len(names) == meta.shape[1], (
        f"Name count {len(names)} != feature count {meta.shape[1]}"
    )


def test_meta_feature_names_with_gap(fake_probas):
    probas, n, C, M = fake_probas
    meta = build_meta_features(probas, p_weighted=probas[0], p_stack=probas[1])
    names = meta_feature_names(C, M, include_path_gap=True)
    assert len(names) == meta.shape[1]


def test_vote_entropy_range():
    votes = np.array([[10, 0, 0], [5, 5, 0], [4, 3, 3]])
    ve = vote_entropy(votes.astype(float), n_models=10)
    assert np.all(ve >= 0) and np.all(ve <= 1 + 1e-9)


def test_probability_normalization():
    """Rows of build_meta_features input must not affect output row sums.

    When calibrated_probas rows sum to 1, the full-probability sub-block in
    the meta-feature matrix should also sum to 1 per model.
    """
    rng = np.random.default_rng(99)
    n, C, M = 30, 5, 3
    probas = []
    for _ in range(M):
        p = rng.dirichlet(np.ones(C), size=n)
        # Verify input is normalised
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-7)
        probas.append(p)
    meta = build_meta_features(probas)
    # The first M*C columns are the full probability vectors
    for m_idx in range(M):
        block = meta[:, m_idx * C:(m_idx + 1) * C]
        np.testing.assert_allclose(
            block.sum(axis=1), 1.0, atol=1e-7,
            err_msg=f"Probability block for model {m_idx} rows don't sum to 1",
        )


def test_simplified_meta_features_exclude_probabilities(fake_probas):
    """Simplified meta-features should NOT contain full probability vectors."""
    probas, n, C, M = fake_probas
    meta_full = build_meta_features(probas, simplified=False)
    meta_simple = build_meta_features(probas, simplified=True)
    # simplified should have M*C fewer columns (the probability vectors)
    expected_diff = M * C
    assert meta_full.shape[1] - meta_simple.shape[1] == expected_diff, (
        f"Expected {expected_diff} fewer columns in simplified mode, "
        f"got {meta_full.shape[1] - meta_simple.shape[1]}"
    )
