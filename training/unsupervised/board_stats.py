from __future__ import annotations

"""Per-position board statistics, used as probe/characterisation targets for
Track B's cluster analysis (EXECUTION_Phase3.md tasks 3.0b, 3.3, 3.4) -- never
as AE training input.

Reuses `training.supervised.features.find_groups` for the connected-component
traversal rather than a third flood-fill implementation (F5): that function
was extracted from `features.py::_liberty_plane` for exactly this purpose.
"""

from typing import Dict

import numpy as np

from ..supervised.features import BOARD, find_groups

_CENTER = (BOARD - 1) / 2.0  # 3.0 for BOARD=7


def group_stats(curr: np.ndarray, opp: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-position group statistics from raw stone planes `curr`/`opp` ([N, BOARD, BOARD]).

    `largest_group_size`: stones in the biggest group on the board (either colour).
    `num_groups`: total number of groups on the board (both colours).
    `num_groups_atari`: groups with exactly one liberty.
    """
    n = curr.shape[0]
    largest_group = np.zeros(n, dtype=np.int32)
    num_groups = np.zeros(n, dtype=np.int32)
    num_groups_atari = np.zeros(n, dtype=np.int32)

    for i in range(n):
        board = np.zeros((BOARD, BOARD), dtype=np.int8)
        board[curr[i] > 0] = 1
        board[opp[i] > 0] = -1

        groups = find_groups(board)
        num_groups[i] = len(groups)
        if groups:
            largest_group[i] = max(len(cells) for _, cells, _ in groups)
        num_groups_atari[i] = sum(1 for _, _, libs in groups if len(libs) == 1)

    return {
        "largest_group_size": largest_group,
        "num_groups": num_groups,
        "num_groups_atari": num_groups_atari,
    }


def edge_vs_centre(curr: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Mean Chebyshev distance from the board centre over occupied points,
    in [0, 3] for BOARD=7 (0 = every stone at the centre point, 3 = every
    stone on an edge/corner). 0.0 for an empty board, by convention."""
    n = curr.shape[0]
    occ = (curr > 0) | (opp > 0)
    rr, cc = np.mgrid[0:BOARD, 0:BOARD]
    dist = np.maximum(np.abs(rr - _CENTER), np.abs(cc - _CENTER)).astype(np.float32)

    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mask = occ[i]
        if mask.any():
            out[i] = float(dist[mask].mean())
    return out


def board_statistics(split: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Per-position scalar statistics for probe/characterisation targets (task 3.0b).

    `split` must contain `states` ([N, >=2, BOARD, BOARD], planes 0/1 =
    current-player/opponent stones) and `move_no`. Reads only those two
    fields -- never `moves`/`values`/`margins`.
    """
    states = split["states"]
    curr = states[:, 0]
    opp = states[:, 1]

    stone_count = (curr > 0).reshape(len(curr), -1).sum(axis=1) + (opp > 0).reshape(len(opp), -1).sum(axis=1)
    gs = group_stats(curr, opp)

    return {
        "stone_count": stone_count.astype(np.int32),
        "move_no": np.asarray(split["move_no"], dtype=np.int32),
        "largest_group_size": gs["largest_group_size"],
        "num_groups": gs["num_groups"],
        "num_groups_atari": gs["num_groups_atari"],
        "edge_vs_centre": edge_vs_centre(curr, opp),
    }
