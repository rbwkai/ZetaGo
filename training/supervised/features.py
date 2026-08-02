from __future__ import annotations

"""Feature encodings for Track A experiments.

This module converts engine-style `states` tensors (the `states` dataset in
the HDF5 files) into model inputs for the three encodings used in experiments:

- N=2: current-player stones, opponent stones
- N=4: N=2 + liberties + turn indicator
- N=7: N=4 + signed history t-1, signed history t-2, occupancy-change plane

Feature definitions are frozen as implemented here (Phase 1 task 1.0a); the plan
text (originally "N=4 = stones+turn+last move", "N=7 = +atari maps, +ko point")
has been corrected to match, not the other way around. The history planes at
t-1/t-2 are signed (+1 mover's stone, -1 opponent's stone, 0 empty) so that both
players' stones are represented in every plane, rather than only the mover's
own stones as in the earlier version of this module. The occupancy-change plane
is a capture/placement diff, not a ko indicator, and is named accordingly.

All functions operate on NumPy arrays and return NumPy arrays.
"""

from typing import Dict, List, Tuple

import numpy as np

BOARD = 7


def find_groups(board: np.ndarray) -> List[Tuple[int, List[Tuple[int, int]], set]]:
    """Connected-component groups on a single (BOARD, BOARD) int8 board (+1/-1/0).

    Returns one `(color, cells, liberties)` tuple per group, black groups
    before white, in row-major discovery order. Shared by `_liberty_plane`
    below and `training/unsupervised/board_stats.py` (EXECUTION_Phase3.md
    task 3.0b) so the flood-fill traversal exists in exactly one place.
    """
    visited = np.zeros((BOARD, BOARD), dtype=bool)
    groups: List[Tuple[int, List[Tuple[int, int]], set]] = []

    for color in (1, -1):
        for r in range(BOARD):
            for c in range(BOARD):
                if board[r, c] != color or visited[r, c]:
                    continue

                stack = [(r, c)]
                cells = []
                libs = set()
                visited[r, c] = True
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                        if nx < 0 or nx >= BOARD or ny < 0 or ny >= BOARD:
                            continue
                        if board[nx, ny] == 0:
                            libs.add((nx, ny))
                        elif board[nx, ny] == color and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))

                groups.append((color, cells, libs))

    return groups


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
        for _color, cells, libs in find_groups(board):
            v = min(len(libs), 4) / 4.0
            for gx, gy in cells:
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
    # Build signed history planes (t-1, t-2), each +1/-1/0 for mover/opponent/empty
    # so both players' stones are visible, plus an occupancy-change (capture/
    # placement diff) plane. Not a ko indicator: it fires on every stone placement
    # and every capture alike, with no way to distinguish a ko recapture from an
    # ordinary move.
    n = len(players)
    h1 = np.zeros((n, BOARD, BOARD), dtype=np.float32)
    h2 = np.zeros((n, BOARD, BOARD), dtype=np.float32)
    occ_change = np.zeros((n, BOARD, BOARD), dtype=np.float32)

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
            mover, opponent = (black_abs, white_abs) if p == 1 else (white_abs, black_abs)
            h1[i] = mover[prev_idx].astype(np.float32) - opponent[prev_idx].astype(np.float32)
            occ_change[i] = (occ_abs[i] != occ_abs[prev_idx]).astype(np.float32)

        if prev2_idx is not None:
            mover, opponent = (black_abs, white_abs) if p == 1 else (white_abs, black_abs)
            h2[i] = mover[prev2_idx].astype(np.float32) - opponent[prev2_idx].astype(np.float32)

    return h1, h2, occ_change


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
        h1, h2, occ_change = _history_planes(
            split["states"],
            split["players"],
            split["game_id"],
            split["move_no"],
        )
        return np.stack([curr, opp, liberties, turn, h1, h2, occ_change], axis=1)

    raise ValueError(f"unsupported encoding N={encoding}")
