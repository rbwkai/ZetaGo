"""Board-statistics extractor: group size, atari, edge-vs-centre (task 3.0b).

Hand-constructed boards with known answers, plus one cross-check against the
engine's own capture logic where the two overlap (a group's only liberty,
once filled, is exactly what the engine captures).
"""

import numpy as np

from training.supervised.features import BOARD, find_groups
from training.unsupervised.board_stats import board_statistics, edge_vs_centre, group_stats


def _from_ascii(text: str):
    """'.' empty, 'X' current-player ('curr') stone, 'O' opponent stone."""
    curr = np.zeros((BOARD, BOARD), dtype=np.int8)
    opp = np.zeros((BOARD, BOARD), dtype=np.int8)
    for r, line in enumerate(text.strip("\n").splitlines()):
        line = line.strip()
        for c, ch in enumerate(line):
            if ch in ("X", "x"):
                curr[r, c] = 1
            elif ch in ("O", "o"):
                opp[r, c] = 1
    return curr, opp


def _combined(curr: np.ndarray, opp: np.ndarray) -> np.ndarray:
    board = np.zeros((BOARD, BOARD), dtype=np.int8)
    board[curr > 0] = 1
    board[opp > 0] = -1
    return board


def test_single_stone_four_liberties():
    curr, opp = _from_ascii(
        """
        .......
        .......
        .......
        ...X...
        .......
        .......
        .......
        """
    )
    groups = find_groups(_combined(curr, opp))
    assert len(groups) == 1
    color, cells, libs = groups[0]
    assert color == 1
    assert cells == [(3, 3)]
    assert len(libs) == 4

    gs = group_stats(curr[None], opp[None])
    assert gs["largest_group_size"][0] == 1
    assert gs["num_groups"][0] == 1
    assert gs["num_groups_atari"][0] == 0


def test_corner_stone_two_liberties():
    curr, opp = _from_ascii(
        """
        X......
        .......
        .......
        .......
        .......
        .......
        .......
        """
    )
    groups = find_groups(_combined(curr, opp))
    assert len(groups) == 1
    _, cells, libs = groups[0]
    assert cells == [(0, 0)]
    assert libs == {(1, 0), (0, 1)}


def test_filled_corner_group_three_liberties():
    curr, opp = _from_ascii(
        """
        XX.....
        X......
        .......
        .......
        .......
        .......
        .......
        """
    )
    groups = find_groups(_combined(curr, opp))
    assert len(groups) == 1
    _, cells, libs = groups[0]
    assert set(cells) == {(0, 0), (0, 1), (1, 0)}
    assert libs == {(0, 2), (1, 1), (2, 0)}

    gs = group_stats(curr[None], opp[None])
    assert gs["largest_group_size"][0] == 3
    assert gs["num_groups"][0] == 1
    assert gs["num_groups_atari"][0] == 0


def test_atari_group_matches_engine_capture():
    ascii_board = """
    .......
    ..O....
    .OX....
    ..O....
    .......
    .......
    .......
    """
    curr, opp = _from_ascii(ascii_board)
    groups = find_groups(_combined(curr, opp))
    black_groups = [g for g in groups if g[0] == 1]
    assert len(black_groups) == 1
    _, cells, libs = black_groups[0]
    assert cells == [(2, 2)]
    assert libs == {(2, 3)}  # exactly one liberty: in atari

    gs = group_stats(curr[None], opp[None])
    assert gs["num_groups"][0] == 4  # 1 black + 3 separate white stones
    assert gs["num_groups_atari"][0] == 1
    assert gs["largest_group_size"][0] == 1

    # Cross-check against the engine (F5 acceptance: "verified against the
    # engine's own legality/scoring where they overlap"): filling a group's
    # only liberty is exactly what the engine's capture logic removes.
    from engine import GoBoard, WHITE

    b = GoBoard.from_ascii(ascii_board, to_move=WHITE)
    assert b.play_move(2, 3) is True
    assert b.black == 0


def test_two_separate_groups_different_sizes():
    curr, opp = _from_ascii(
        """
        .......
        .......
        .......
        .XXX...
        .......
        .......
        OO.....
        """
    )
    groups = find_groups(_combined(curr, opp))
    assert len(groups) == 2
    gs = group_stats(curr[None], opp[None])
    assert gs["num_groups"][0] == 2
    assert gs["largest_group_size"][0] == 3
    assert gs["num_groups_atari"][0] == 0


def test_empty_board():
    curr = np.zeros((BOARD, BOARD), dtype=np.int8)
    opp = np.zeros((BOARD, BOARD), dtype=np.int8)
    assert find_groups(_combined(curr, opp)) == []
    gs = group_stats(curr[None], opp[None])
    assert gs["num_groups"][0] == 0
    assert gs["largest_group_size"][0] == 0
    assert gs["num_groups_atari"][0] == 0
    assert edge_vs_centre(curr[None], opp[None])[0] == 0.0


def test_edge_vs_centre_extremes():
    centre, _ = _from_ascii(
        """
        .......
        .......
        .......
        ...X...
        .......
        .......
        .......
        """
    )
    corner, _ = _from_ascii(
        """
        X......
        .......
        .......
        .......
        .......
        .......
        .......
        """
    )
    empty_opp = np.zeros((BOARD, BOARD), dtype=np.int8)
    assert edge_vs_centre(centre[None], empty_opp[None])[0] == 0.0
    assert edge_vs_centre(corner[None], empty_opp[None])[0] == 3.0


def test_board_statistics_reads_only_states_and_move_no():
    curr, opp = _from_ascii(
        """
        XX.....
        X......
        .......
        .......
        .......
        .......
        .......
        """
    )
    # 6-channel states tensor, matching data.load_split's documented shape;
    # only planes 0/1 (curr/opp) are used.
    states = np.zeros((1, 6, BOARD, BOARD), dtype=np.uint8)
    states[0, 0] = curr
    states[0, 1] = opp
    split = {"states": states, "move_no": np.array([5], dtype=np.int32)}

    stats = board_statistics(split)
    assert stats["stone_count"][0] == 3
    assert stats["move_no"][0] == 5
    assert stats["largest_group_size"][0] == 3
    assert stats["num_groups"][0] == 1
    assert stats["num_groups_atari"][0] == 0
