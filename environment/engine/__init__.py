"""ZetaGo bitboard Go engine (7x7 by default, size-generic)."""

from .constants import BLACK, WHITE, EMPTY, KOMI, PASS, DEFAULT_SIZE
from .board import GoBoard
from .scoring import tromp_taylor_area
from .encode import (
    move_to_index,
    index_to_move,
    board_to_array,
    board_to_tensor,
    legal_moves_mask,
)

__all__ = [
    "BLACK",
    "WHITE",
    "EMPTY",
    "KOMI",
    "PASS",
    "DEFAULT_SIZE",
    "GoBoard",
    "tromp_taylor_area",
    "move_to_index",
    "index_to_move",
    "board_to_array",
    "board_to_tensor",
    "legal_moves_mask",
]
