from __future__ import annotations

"""Feature encodings for Track A experiments.

This module converts engine-style `states` tensors (the `states` dataset in
the HDF5 files) into model inputs for the three encodings used in experiments:

- N=2: current-player stones, opponent stones
- N=4: N=2 + liberties + turn indicator
- N=7: N=4 + history t-1, history t-2, ko/capture history

All functions operate on NumPy arrays and return NumPy arrays.
"""

from typing import Dict, Tuple

import numpy as np

BOARD = 7


def _liberty_plane(curr: np.ndarray, opp: np.ndarray) -> np.ndarray:
    # For each position, compute a single float plane giving normalized
    # liberties for each connected group (clipped to 4 liberties -> value in [0,1]).
    n = curr.shape[0]
    out = np.zeros((n, BOARD, BOARD), dtype=np.float32)

    for i in range(n):
        board = np.zeros((BOARD, BOARD), dtype=np.int8)
        board[curr[i] > 0] = 1
        board[opp[i] > 0] = -1

        plane = np.zeros((BOARD, BOARD), dtype=np.float32)
        visited = np.zeros((BOARD, BOARD), dtype=bool)

        for color in (1, -1):
            for r in range(BOARD):
                for c in range(BOARD):
                    if board[r, c] != color or visited[r, c]:
                        continue

                    stack = [(r, c)]
                    group = []
                    libs = set()
                    visited[r, c] = True
                    while stack:
                        x, y = stack.pop()
                        group.append((x, y))
                        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                            if nx < 0 or nx >= BOARD or ny < 0 or ny >= BOARD:
                                continue
                            if board[nx, ny] == 0:
                                libs.add((nx, ny))
                            elif board[nx, ny] == color and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))

                    v = min(len(libs), 4) / 4.0
                    for gx, gy in group:
                        plane[gx, gy] = v

        out[i] = plane

    return out


def _absolute_bw(states: np.ndarray, players: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Convert engine `states` (which store planes in side-to-move POV) into
    # absolute black / white occupancy arrays aligned to board coordinates.
    curr = states[:, 0]
    opp = states[:, 1]
    black_to_move = (players == 1)[:, None, None]
    black = np.where(black_to_move, curr, opp).astype(np.uint8)
    white = np.where(black_to_move, opp, curr).astype(np.uint8)
    return black, white


def _history_planes(
    states: np.ndarray,
    players: np.ndarray,
    game_id: np.ndarray,
    move_no: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Build history planes (t-1, t-2) and a simple ko/capture indicator plane.
    n = len(players)
    h1 = np.zeros((n, BOARD, BOARD), dtype=np.float32)
    h2 = np.zeros((n, BOARD, BOARD), dtype=np.float32)
    kcap = np.zeros((n, BOARD, BOARD), dtype=np.float32)

    black_abs, white_abs = _absolute_bw(states, players)
    occ_abs = (black_abs | white_abs).astype(np.uint8)

    pos_to_idx = {(int(game_id[i]), int(move_no[i])): i for i in range(n)}

    for i in range(n):
        gid = int(game_id[i])
        mn = int(move_no[i])
        p = int(players[i])

        prev_idx = pos_to_idx.get((gid, mn - 1))
        prev2_idx = pos_to_idx.get((gid, mn - 2))

        if prev_idx is not None:
            h1[i] = black_abs[prev_idx] if p == 1 else white_abs[prev_idx]
            kcap[i] = (occ_abs[i] != occ_abs[prev_idx]).astype(np.float32)

        if prev2_idx is not None:
            h2[i] = black_abs[prev2_idx] if p == 1 else white_abs[prev2_idx]

    return h1, h2, kcap


def make_features(split: Dict[str, np.ndarray], encoding: int) -> np.ndarray:
    # Top-level feature builder. Input: `split` dict from `data.load_split`.
    # Output: NumPy array shaped [N, C, 7, 7] where C depends on `encoding`.
    states = split["states"].astype(np.float32)
    curr = states[:, 0]
    opp = states[:, 1]

    if encoding == 2:
        return np.stack([curr, opp], axis=1)

    if encoding == 4:
        liberties = _liberty_plane(curr, opp)
        turn = states[:, 2]
        return np.stack([curr, opp, liberties, turn], axis=1)

    if encoding == 7:
        liberties = _liberty_plane(curr, opp)
        turn = states[:, 2]
        h1, h2, kcap = _history_planes(
            split["states"],
            split["players"],
            split["game_id"],
            split["move_no"],
        )
        return np.stack([curr, opp, liberties, turn, h1, h2, kcap], axis=1)

    raise ValueError(f"unsupported encoding N={encoding}")
