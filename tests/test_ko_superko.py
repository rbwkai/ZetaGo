"""Ko and positional-superko enforcement (the simple-ko engine had a gap here)."""

from engine import GoBoard, BLACK, WHITE

# White (3,3) is in atari; (3,4) is the ko point.
SIMPLE_KO = """
.......
.......
...XO..
..XO.O.
...XO..
.......
.......
"""

# Two independent kos. ko-A (rows 0-2) has a white stone (1,1) capturable by black;
# ko-B (rows 4-6) has a black stone (5,1) capturable by white.
DOUBLE_KO = """
.XO....
XO.O...
.XO....
.......
.OX....
OX.X...
.OX....
"""


def test_simple_ko_forbids_immediate_recapture():
    b = GoBoard.from_ascii(SIMPLE_KO, to_move=BLACK)
    assert b.play_move(3, 4) is True               # black captures white (3,3)
    assert (b.white & (1 << (3 * 7 + 3))) == 0
    # White's immediate recapture at (3,3) would recreate the previous position.
    assert b.play_move(3, 3) is False
    # After playing elsewhere on both sides, the recapture is legal again.
    assert b.play_move(0, 0) is True               # white elsewhere
    assert b.play_move(6, 6) is True               # black elsewhere
    assert b.play_move(3, 3) is True               # white recaptures (new position)


def test_positional_superko_forbids_longer_cycle():
    b = GoBoard.from_ascii(DOUBLE_KO, to_move=BLACK)
    assert b.play_move(1, 2) is True               # black captures in ko-A  -> S1
    s1 = b.zobrist
    assert b.play_move(5, 2) is True               # white captures in ko-B  -> S2
    s2 = b.zobrist
    assert s1 != s2                                 # the two boards differ...

    # Black retaking ko-B at (5,1) recreates S1, which is two plies back (not the
    # immediately previous board S2). A simple-ko check compares only against the
    # previous board and would wrongly allow it; positional superko forbids it.
    assert b.play_move(5, 1) is False
    assert b.zobrist == s2                          # rejected move left board unchanged
    assert s1 in b.position_history


def test_superko_history_grows_and_is_deterministic():
    b = GoBoard()
    seq = [(0, 0), (6, 6), (0, 1), (6, 5), (1, 0)]
    for r, c in seq:
        assert b.play_move(r, c) is True
    assert len(b.position_history) == len(seq) + 1   # +1 for the empty start position

    # Replaying through a copy reproduces the exact hash sequence (seeded Zobrist).
    b2 = GoBoard()
    for r, c in seq:
        b2.play_move(r, c)
    assert b2.zobrist == b.zobrist
