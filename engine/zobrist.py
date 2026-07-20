"""Zobrist hashing used for positional-superko detection.

The hash is *board-only* (it does not include the side to move), which matches
KataGo's ``koRule = POSITIONAL``: a move is illegal if it recreates any board
position that has occurred before, regardless of whose turn it was.
"""

import random
from functools import lru_cache


@lru_cache(maxsize=None)
def zobrist_table(n):
    """Return ``stone`` where ``stone[color_idx][point]`` is a 64-bit key.

    ``color_idx``: 0 = Black, 1 = White. Deterministic for a given board size,
    so replaying the same game always yields the same hash sequence.
    """
    rng = random.Random(0x5A7A6000 + n)
    nn = n * n
    stone = (
        [rng.getrandbits(64) for _ in range(nn)],
        [rng.getrandbits(64) for _ in range(nn)],
    )
    return stone
