"""Elo ratings (via a Bradley-Terry maximum-likelihood fit) with bootstrap CIs.

Uses Zermelo's iterative MM algorithm rather than sequential online Elo updates,
so ratings don't depend on an arbitrary game order -- the natural fit for a
round-robin's fixed set of results.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .tournament import GameResult


def _pairwise_wins(results: List[GameResult]) -> Dict[Tuple[str, str], float]:
    """wins[(i, j)] = number of times i beat j (ties split 0.5/0.5)."""
    wins: Dict[Tuple[str, str], float] = defaultdict(float)
    for r in results:
        if r.winner == "black":
            wins[(r.black, r.white)] += 1.0
        elif r.winner == "white":
            wins[(r.white, r.black)] += 1.0
        else:
            wins[(r.black, r.white)] += 0.5
            wins[(r.white, r.black)] += 0.5
    return wins


def bradley_terry_strengths(
    results: List[GameResult], names: List[str], n_iter: int = 300, eps: float = 1e-9
) -> Dict[str, float]:
    """Fit Bradley-Terry strengths pi_i > 0 s.t. P(i beats j) = pi_i / (pi_i + pi_j),
    via Zermelo's MM algorithm. Strengths are arbitrary-scale; use `to_elo` to
    convert to an Elo scale anchored at a chosen reference agent."""
    wins = _pairwise_wins(results)
    total_wins = defaultdict(float)
    n_between: Dict[Tuple[str, str], float] = defaultdict(float)
    for (i, j), w in wins.items():
        total_wins[i] += w
        n_between[(i, j)] += w

    strength = {n: 1.0 for n in names}
    for _ in range(n_iter):
        new_strength = {}
        for i in names:
            denom = 0.0
            for j in names:
                if j == i:
                    continue
                nij = n_between.get((i, j), 0.0) + n_between.get((j, i), 0.0)
                if nij == 0.0:
                    continue
                denom += nij / (strength[i] + strength[j])
            new_strength[i] = (total_wins[i] / denom) if denom > 0 else strength[i]
        # Renormalise against the first (arbitrary anchor) to prevent drift.
        norm = max(new_strength[names[0]], eps)
        strength = {n: max(v / norm, eps) for n, v in new_strength.items()}
    return strength


def to_elo(strength: Dict[str, float], anchor: str) -> Dict[str, float]:
    base = strength[anchor]
    return {n: 400.0 * math.log10(v / base) for n, v in strength.items()}


def bootstrap_elo_ci(
    results: List[GameResult],
    names: List[str],
    anchor: str,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """95% (by default) bootstrap CI on each agent's Elo (relative to `anchor`),
    resampling games with replacement and refitting Bradley-Terry each time."""
    rng = np.random.default_rng(seed)
    n = len(results)
    boot_elo: Dict[str, List[float]] = {name: [] for name in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [results[i] for i in idx]
        strength = bradley_terry_strengths(sample, names)
        elo = to_elo(strength, anchor)
        for name in names:
            boot_elo[name].append(elo[name])

    out = {}
    for name in names:
        lo, hi = np.percentile(boot_elo[name], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[name] = (float(lo), float(hi))
    return out
