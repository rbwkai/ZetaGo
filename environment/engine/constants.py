"""Core constants for the ZetaGo bitboard engine."""

BLACK = 1
WHITE = -1
EMPTY = 0

KOMI = 9.5            # Tromp-Taylor / area komi, awarded to White
DEFAULT_SIZE = 7

# Pass is encoded as the index just past the last board point. For the default
# 7x7 board this is 49 (board points are 0..48). Size-generic boards expose
# ``GoBoard.PASS`` for the same value computed from their own N.
PASS = DEFAULT_SIZE * DEFAULT_SIZE
