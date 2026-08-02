from __future__ import annotations

"""Track B data path (EXECUTION_Phase3.md task 3.0a).

Loads `train.h5` via `training.supervised.data.load_split`/`apply_dedup` --
reused, not duplicated -- deduplicates unconditionally (F2: "a Track B result
computed on raw rows is not reportable"), and hands the autoencoder training
loop *only* the raw N=2 stone planes. `_assert_unlabelled` is a real runtime
guard, not documentation: it is the thing that would fail loudly if this
function were ever refactored to pass the full split dict through instead of
a states-only payload (P3-G2).

A reproducible held-out slice of the deduplicated train split is set aside for
reconstruction-loss model selection (SS3.B). This is distinct from `val.h5`,
which the analysis stage (tasks 3.2+) uses for clustering/probes and which the
AE never trains or selects on.
"""

from typing import Dict, Tuple

import numpy as np

from ..supervised.data import apply_dedup, load_split

_FORBIDDEN_KEYS = ("moves", "values", "margins")


def _assert_unlabelled(payload: Dict[str, np.ndarray]) -> None:
    """Fail loudly if a label field is reachable from Track B's data path (P3-G2)."""
    leaked = [k for k in _FORBIDDEN_KEYS if k in payload]
    if leaked:
        raise AssertionError(f"Track B data path must not read label fields, found: {leaked}")


def load_unlabelled_train(
    train_h5: str,
    holdout_frac: float = 0.1,
    seed: int = 0,
    max_rows: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(fit_states, holdout_states)`, both float32 `[N, 2, 7, 7]` (N=2:
    current-player, opponent stone planes -- SS3.A).

    `fit_states` is what the autoencoder trains its reconstruction loss on.
    `holdout_states` is a disjoint slice used only to score reconstruction
    loss for model selection (SS3.B); it is never used for clustering/probe
    analysis, and the two slices never overlap.
    """
    split = load_split(train_h5, max_rows=max_rows)
    split = apply_dedup(split, "unique")

    downstream = {"states": np.asarray(split["states"][:, :2], dtype=np.float32)}
    _assert_unlabelled(downstream)
    states = downstream["states"]

    n = len(states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = int(round(n * holdout_frac))
    hold_idx = np.sort(perm[:n_hold])
    fit_idx = np.sort(perm[n_hold:])
    return states[fit_idx], states[hold_idx]
