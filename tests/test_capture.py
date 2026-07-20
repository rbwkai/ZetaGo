"""Capture detection: single/multi stone, edges, suicide-but-captures, no false capture."""

import random

from engine import GoBoard, BLACK, WHITE


def test_single_stone_capture():
    b = GoBoard.from_ascii(
        """
        O......
        X......
        .......
        .......
        .......
        .......
        .......
        """,
        to_move=BLACK,
    )
    # Black (0,1) takes White (0,0)'s last liberty.
    assert b.play_move(0, 1) is True
    assert b.white == 0
    assert b.captured_by_black == 1
    assert b.zobrist == b._hash_of(b.black, b.white)


def test_multi_stone_group_capture():
    b = GoBoard.from_ascii(
        """
        OO.....
        XX.....
        .......
        .......
        .......
        .......
        .......
        """,
        to_move=BLACK,
    )
    # White group {(0,0),(0,1)} has only liberty (0,2); black fills it.
    assert b.play_move(0, 2) is True
    assert b.white == 0
    assert b.captured_by_black == 2


def test_edge_capture_uses_edge_masks():
    b = GoBoard.from_ascii(
        """
        .......
        .......
        ......X
        ......O
        ......X
        .......
        .......
        """,
        to_move=BLACK,
    )
    # White (3,6) sits on the right edge with neighbours (2,6),(4,6),(3,5).
    # Black plays (3,5): the off-board "(3,7)" must not count as a liberty.
    assert b.play_move(3, 5) is True
    assert b.white == 0
    assert b.captured_by_black == 1


def test_capture_that_looks_like_suicide_is_legal():
    b = GoBoard.from_ascii(
        """
        .OX....
        OOX....
        XX.....
        .......
        .......
        .......
        .......
        """,
        to_move=BLACK,
    )
    # Black (0,0) would have no liberties, but it captures the white group
    # {(0,1),(1,0),(1,1)} whose only liberty was (0,0) -> legal.
    assert b.play_move(0, 0) is True
    assert b.white == 0
    assert b.captured_by_black == 3


def test_no_false_capture_when_group_has_liberties():
    b = GoBoard.from_ascii(
        """
        O......
        .......
        .......
        .......
        .......
        .......
        .......
        """,
        to_move=BLACK,
    )
    assert b.play_move(1, 0) is True       # white (0,0) still has liberty (0,1)
    assert b.white == (1 << 0)
    assert b.captured_by_black == 0


def test_zobrist_matches_recompute_over_random_game():
    b = GoBoard()
    rng = random.Random(12345)
    for _ in range(80):
        moves = b.get_legal_moves()
        if not moves:
            b.pass_move()
            continue
        r, c = rng.choice(moves)
        assert b.play_move(r, c) is True
        assert b.zobrist == b._hash_of(b.black, b.white)
