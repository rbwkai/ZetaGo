"""Statistical tests promised in `RESEARCH_REDESIGN.md` SS3 and `EXECUTION_PLAN.md`
task 4.2, and never implemented until now (`EXECUTION_Phase4.md` F5): McNemar's
test for paired model comparisons on identical positions, and an explicit
interaction test for Factor A x model.

Both are plain NumPy/SciPy, no new heavy dependency (no statsmodels): the
two-way ANOVA is the standard balanced-design sum-of-squares decomposition,
and McNemar uses the exact binomial form rather than the chi-squared
approximation, which is unreliable when the discordant count is small.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import binomtest
from scipy.stats import f as f_dist


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict:
    """Exact (binomial) McNemar test for two classifiers' paired correctness
    over the *identical* set of items -- the right test for "is model A
    better than model B on the same data," which independent-CI comparisons
    (used elsewhere in this project) are not.

    `correct_a`/`correct_b`: boolean (or 0/1) arrays, same length, same order
    -- i.e. row `i` in both must be the same validation/test position.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError(f"paired arrays must be the same shape, got {correct_a.shape} vs {correct_b.shape}")

    a_right_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_right = int(np.sum(~correct_a & correct_b))
    n_discordant = a_right_b_wrong + a_wrong_b_right

    if n_discordant == 0:
        p_value = 1.0
    else:
        k = min(a_right_b_wrong, a_wrong_b_right)
        p_value = float(binomtest(k, n_discordant, 0.5).pvalue)

    return {
        "n": int(len(correct_a)),
        "both_right": int(np.sum(correct_a & correct_b)),
        "both_wrong": int(np.sum(~correct_a & ~correct_b)),
        "a_right_b_wrong": a_right_b_wrong,
        "a_wrong_b_right": a_wrong_b_right,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def two_way_anova_interaction(data: np.ndarray) -> Dict:
    """Two-way ANOVA, balanced design, `data` shape `[n_A, n_B, n_reps]`
    (e.g. [n_models, n_encodings, n_seeds]). Returns the interaction term's
    F-statistic/p-value plus both main effects, computed from the standard
    sum-of-squares decomposition (no statsmodels dependency)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D [n_A, n_B, n_reps] array, got shape {data.shape}")
    a, b, n = data.shape
    if a < 2 or b < 2 or n < 2:
        raise ValueError(f"need >=2 levels of each factor and >=2 reps per cell, got shape {data.shape}")

    grand = data.mean()
    mean_a = data.mean(axis=(1, 2))
    mean_b = data.mean(axis=(0, 2))
    mean_ab = data.mean(axis=2)

    ss_a = b * n * np.sum((mean_a - grand) ** 2)
    ss_b = a * n * np.sum((mean_b - grand) ** 2)
    ss_ab = n * np.sum((mean_ab - mean_a[:, None] - mean_b[None, :] + grand) ** 2)
    ss_total = np.sum((data - grand) ** 2)
    ss_error = ss_total - ss_a - ss_b - ss_ab

    df_a, df_b, df_ab, df_e = a - 1, b - 1, (a - 1) * (b - 1), a * b * (n - 1)
    ms_e = ss_error / df_e

    def _f_test(ss, df):
        ms = ss / df
        F = float(ms / ms_e) if ms_e > 0 else float("inf")
        p = float(f_dist.sf(F, df, df_e))
        return F, p

    F_a, p_a = _f_test(ss_a, df_a)
    F_b, p_b = _f_test(ss_b, df_b)
    F_ab, p_ab = _f_test(ss_ab, df_ab)

    return {
        "shape": {"n_A": a, "n_B": b, "n_reps": n},
        "ss_a": float(ss_a), "df_a": df_a, "F_a": F_a, "p_a": p_a,
        "ss_b": float(ss_b), "df_b": df_b, "F_b": F_b, "p_b": p_b,
        "ss_interaction": float(ss_ab), "df_interaction": df_ab,
        "F_interaction": F_ab, "p_interaction": p_ab,
        "ss_error": float(ss_error), "df_error": df_e,
        "ss_total": float(ss_total),
    }
