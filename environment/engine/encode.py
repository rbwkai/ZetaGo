"""NumPy encodings: board arrays, ML tensors, legal-move masks, move<->index helpers.

This is the only engine module that imports NumPy; the rules core stays NumPy-free.
"""

import numpy as np

from .constants import BLACK, DEFAULT_SIZE


def move_to_index(row, col, n=DEFAULT_SIZE):
    """Board point (row, col) -> policy index 0..N*N-1."""
    return row * n + col


def index_to_move(idx, n=DEFAULT_SIZE):
    """Policy index -> (row, col), or None for the pass index (N*N)."""
    if idx == n * n:
        return None
    return divmod(idx, n)


def _grid(bb, n):
    """49-bit bitboard -> (n, n) uint8 grid (row 0 = top)."""
    g = np.zeros((n, n), dtype=np.uint8)
    i = 0
    while bb:
        if bb & 1:
            g[i // n, i % n] = 1
        bb >>= 1
        i += 1
    return g


def board_to_array(board):
    """(N, N) int8 array: +1 black, -1 white, 0 empty (row 0 = top)."""
    n = board.N
    arr = _grid(board.black, n).astype(np.int8)
    arr -= _grid(board.white, n).astype(np.int8)
    return arr


def board_to_tensor(board):
    """(6, N, N) uint8 planes, encoded from the current player's perspective:

      0: current player's stones        3: empty points
      1: opponent's stones              4: last move (1 hot, empty if last was a pass)
      2: side-to-move (all 1s if Black) 5: legal moves for the current player
    """
    n = board.N
    t = np.zeros((6, n, n), dtype=np.uint8)
    if board.current_player == BLACK:
        me, opp = board.black, board.white
        t[2, :, :] = 1
    else:
        me, opp = board.white, board.black
    t[0] = _grid(me, n)
    t[1] = _grid(opp, n)
    t[3] = _grid(board._geo.FULL & ~(board.black | board.white), n)
    if board.last_move is not None:
        t[4, board.last_move // n, board.last_move % n] = 1
    for (r, c) in board.get_legal_moves():
        t[5, r, c] = 1
    return t


def legal_moves_mask(board):
    """(N*N + 1,) uint8 mask; index N*N is the pass move (always legal)."""
    n = board.N
    mask = np.zeros(n * n + 1, dtype=np.uint8)
    for (r, c) in board.get_legal_moves():
        mask[r * n + c] = 1
    mask[n * n] = 1
    return mask
