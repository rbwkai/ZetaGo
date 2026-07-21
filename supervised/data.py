from __future__ import annotations

"""Data loaders and small helpers for supervised experiments.

This module provides `load_split` which reads the HDF5 `train.h5` and
`val.h5` files produced by `data/build_dataset.py` and returns a small dict
of NumPy arrays used throughout the experiments.

Returned fields:
- `states`: uint8 tensor [N, 6, 7, 7] as produced by the engine replay
- `moves`: int16 policy targets (0..48, 49=pass)
- `values`: int8 final outcome from side-to-move POV (+1,-1,0)
- `players`, `game_id`, `move_no`: auxiliary fields used by history features

`subsample` deterministically selects a random subset of rows (used by
sklearn-style baselines to cap training size while keeping reproducibility).
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
