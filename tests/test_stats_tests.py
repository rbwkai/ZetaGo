"""McNemar's exact test and the two-way ANOVA interaction test (task 4.0b),
verified against hand-computable answers and a synthetic planted-effect
dataset, not just "it runs"."""

import numpy as np

from eval.stats_tests import mcnemar_exact, two_way_anova_interaction


def test_mcnemar_identical_predictions_is_perfectly_null():
    a = np.array([True, False, True, True, False, False])
    b = a.copy()
    r = mcnemar_exact(a, b)
    assert r["n_discordant"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_symmetric_disagreement_is_not_significant():
    # 2 items where only A is right, 2 where only B is right: perfectly balanced
    # discordance -> two-sided exact binomial p-value is exactly 1.0.
    a = np.array([True, True, False, False, True, True])
    b = np.array([False, False, True, True, True, True])
    r = mcnemar_exact(a, b)
    assert r["a_right_b_wrong"] == 2
    assert r["a_wrong_b_right"] == 2
    assert r["n_discordant"] == 4
    assert r["p_value"] == 1.0


def test_mcnemar_one_sided_sweep_is_significant_and_hand_computable():
    # A right / B wrong on all 10 discordant items, 0 the other way:
    # two-sided exact binomial p = 2 * 0.5**10 (hand-computable).
    a = np.array([True] * 10)
    b = np.array([False] * 10)
    r = mcnemar_exact(a, b)
    assert r["a_right_b_wrong"] == 10
    assert r["a_wrong_b_right"] == 0
    expected_p = 2 * (0.5**10)
    assert abs(r["p_value"] - expected_p) < 1e-12


def test_mcnemar_rejects_mismatched_shapes():
    import pytest

    with pytest.raises(ValueError):
        mcnemar_exact(np.array([True, False]), np.array([True]))


def _synthetic_cell_means(a: int, b: int, n: int, interaction_strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha = np.arange(a, dtype=np.float64) * 2.0  # main effect A
    beta = np.arange(b, dtype=np.float64) * 3.0  # main effect B
    data = np.zeros((a, b, n))
    for i in range(a):
        for j in range(b):
            cell_mean = 10.0 + alpha[i] + beta[j] + interaction_strength * i * j
            data[i, j, :] = cell_mean + rng.normal(scale=0.1, size=n)
    return data


def test_anova_no_interaction_planted_is_not_significant():
    data = _synthetic_cell_means(a=3, b=2, n=5, interaction_strength=0.0, seed=0)
    r = two_way_anova_interaction(data)
    assert r["p_interaction"] > 0.05  # no planted interaction -> not detected
    assert r["p_a"] < 0.01  # main effects ARE real and large relative to noise
    assert r["p_b"] < 0.01


def test_anova_planted_interaction_is_detected():
    data = _synthetic_cell_means(a=3, b=2, n=5, interaction_strength=5.0, seed=0)
    r = two_way_anova_interaction(data)
    assert r["p_interaction"] < 0.01  # large planted interaction -> detected


def test_anova_rejects_bad_shapes():
    import pytest

    with pytest.raises(ValueError):
        two_way_anova_interaction(np.zeros((3, 3)))  # not 3-D
    with pytest.raises(ValueError):
        two_way_anova_interaction(np.zeros((1, 3, 5)))  # only 1 level of factor A
