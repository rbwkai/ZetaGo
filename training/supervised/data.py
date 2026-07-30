from __future__ import annotations

"""Data loaders and small helpers for supervised experiments.

This module provides `load_split` which reads the HDF5 `train.h5` / `val.h5` /
`test.h5` files produced by `data/build_dataset.py` and returns a small dict
of NumPy arrays used throughout the experiments.

Returned fields:
- `states`: uint8 tensor [N, 6, 7, 7] as produced by the engine replay
- `moves`: int16 policy targets (0..48, 49=pass)
- `values`: int8 final outcome from side-to-move POV (+1,-1,0)
- `margins`: float32 final score margin from side-to-move POV (NaN if unscored)
- `players`, `game_id`, `move_no`: auxiliary fields used by history features

`subsample` deterministically selects a random subset of rows (used by
sklearn-style baselines to cap training size while keeping reproducibility).

`subsample_games` and `apply_dedup` implement Phase 1 tasks 1.4 (Factor B:
nested, by-game data-volume subsampling) and 1.2c (deduplication of repeated
positions) respectively. See EXECUTION_Phase1.md tasks 1.2c/1.4.
"""

import os
from typing import Dict

import numpy as np


def load_split(path: str, max_rows: int = 0) -> Dict[str, np.ndarray]:
    # lazy import so `--help` and other lightweight CLI actions don't require h5py
    import h5py

    if not os.path.exists(path):
        raise FileNotFoundError(f"missing split: {path}")
    with h5py.File(path, "r") as h:
        n = len(h["moves"])
        use_n = min(n, max_rows) if max_rows else n
        # return NumPy arrays with predictable dtypes for downstream code
        return {
            "states": h["states"][:use_n],
            "moves": h["moves"][:use_n].astype(np.int64),
            "values": h["values"][:use_n].astype(np.int64),
            "margins": h["margins"][:use_n].astype(np.float32),
            "players": h["players"][:use_n].astype(np.int8),
            "game_id": h["game_id"][:use_n].astype(np.uint32),
            "move_no": h["move_no"][:use_n].astype(np.int32),
        }


def subsample(split: Dict[str, np.ndarray], max_rows: int, seed: int) -> Dict[str, np.ndarray]:
    """Return a deterministic random subset of `split` rows.

    Used to cap sklearn baselines while preserving reproducible sampling.
    """
    if max_rows <= 0:
        return split
    n = len(next(iter(split.values())))
    if n <= max_rows:
        return split
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    idx.sort()
    return {k: v[idx] for k, v in split.items()}


def subsample_games(split: Dict[str, np.ndarray], n_games: int, seed: int) -> Dict[str, np.ndarray]:
    """Return the rows belonging to a deterministic, seed-fixed sample of games.

    Factor B (EXECUTION_Phase1.md task 1.4): a fixed random permutation of the
    split's unique game ids is drawn once per seed; each data-volume level takes
    a prefix of that permutation. Levels are therefore strictly nested for a
    given seed (more games = superset), while different seeds see different
    game orderings. `n_games <= 0` or `n_games >= total games` returns the
    split unchanged (the "full" level).
    """
    uniq_games = np.unique(split["game_id"])
    if n_games <= 0 or n_games >= len(uniq_games):
        return split
    rng = np.random.default_rng(seed)
    chosen = rng.permutation(uniq_games)[:n_games]
    mask = np.isin(split["game_id"], chosen)
    return {k: v[mask] for k, v in split.items()}


def apply_dedup(split: Dict[str, np.ndarray], mode: str) -> Dict[str, np.ndarray]:
    """Reduce position multiplicity in `split` (training rows only; see task 1.2c).

    `mode`:
      - "none": return `split` unchanged (raw rows, current behaviour).
      - "unique": exactly one row per distinct board position. When a position
        recurs with different expert moves across games, the kept row's move
        is the *mode* (most frequent) move for that position, ties broken by
        first occurrence in row order -- this respects majority expert opinion
        rather than an arbitrary first-seen row.
      - "cap:K": at most K rows per distinct position, keeping the first K
        occurrences in row order (preserves whatever natural move diversity
        the data has, up to K examples).

    Position identity is the exact `states` tensor (bytes-equal). Row order
    within a kept position group is otherwise preserved.
    """
    if mode == "none":
        return split
    n = len(split["moves"])
    if n == 0:
        return split
    states = split["states"]
    keys = np.array([s.tobytes() for s in states], dtype=object)

    if mode == "unique":
        keep_idx = []
        moves = split["moves"]
        # groups: key -> list of row indices, in row order
        groups: Dict[bytes, list] = {}
        for i, k in enumerate(keys):
            groups.setdefault(k, []).append(i)
        for idx_list in groups.values():
            if len(idx_list) == 1:
                keep_idx.append(idx_list[0])
                continue
            group_moves = moves[idx_list]
            vals, counts = np.unique(group_moves, return_counts=True)
            mode_move = vals[np.argmax(counts)]
            # first row in the group whose move equals the mode move
            for i in idx_list:
                if moves[i] == mode_move:
                    keep_idx.append(i)
                    break
        keep_idx = np.sort(np.asarray(keep_idx, dtype=np.int64))
        return {k: v[keep_idx] for k, v in split.items()}

    if mode.startswith("cap:"):
        cap = int(mode.split(":", 1)[1])
        if cap <= 0:
            raise ValueError(f"--dedup cap:K requires K > 0, got {mode!r}")
        seen: Dict[bytes, int] = {}
        keep_idx = []
        for i, k in enumerate(keys):
            c = seen.get(k, 0)
            if c < cap:
                keep_idx.append(i)
                seen[k] = c + 1
        keep_idx = np.asarray(keep_idx, dtype=np.int64)
        return {k: v[keep_idx] for k, v in split.items()}

    raise ValueError(f"unknown --dedup mode: {mode!r}")
