"""ML tensor / array encoding and move<->index helpers."""

import numpy as np

from engine import GoBoard, WHITE, move_to_index, index_to_move


def test_board_to_array_has_no_horizontal_wrap():
    b = GoBoard()
    b.play_move(3, 6)                       # black on the right edge
    arr, player = b.get_state()
    assert arr[3, 6] == 1
    assert arr[4, 0] == 0                    # must not wrap to the next row
    assert player == WHITE


def test_tensor_shape_dtype_and_values():
    t = GoBoard().get_tensor()
    assert t.shape == (6, 7, 7)
    assert t.dtype == np.uint8
    assert set(np.unique(t)).issubset({0, 1})


def test_side_to_move_plane():
    b = GoBoard()
    assert b.get_tensor()[2].sum() == 49     # black to move -> all ones
    b.play_move(0, 0)
    assert b.get_tensor()[2].sum() == 0      # white to move -> all zeros


def test_planes_are_current_player_relative():
    b = GoBoard()
    b.play_move(3, 3)                        # black stone; now white to move
    t = b.get_tensor()
    assert t[0].sum() == 0                    # current player (white) has no stones
    assert t[1][3, 3] == 1                    # opponent (black) stone shows in plane 1


def test_last_move_plane_set_and_cleared_by_pass():
    b = GoBoard()
    b.play_move(2, 5)
    t = b.get_tensor()
    assert t[4][2, 5] == 1
    assert t[4].sum() == 1
    b.pass_move()
    assert b.get_tensor()[4].sum() == 0


def test_legal_plane_matches_get_legal_moves():
    b = GoBoard()
    b.play_move(3, 3)
    t = b.get_tensor()
    legal = set(b.get_legal_moves())
    plane = {(r, c) for r in range(7) for c in range(7) if t[5][r, c] == 1}
    assert plane == legal


def test_legal_moves_mask_includes_pass():
    m = GoBoard().legal_moves_mask()
    assert m.shape == (50,)
    assert m[49] == 1                         # pass is always legal
    assert int(m[:49].sum()) == 49            # every board point legal on an empty board


def test_move_index_roundtrip():
    for r in range(7):
        for c in range(7):
            assert index_to_move(move_to_index(r, c)) == (r, c)
    assert index_to_move(49) is None
