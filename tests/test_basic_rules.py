"""Basic move legality, turn order, passing, game end, and suicide."""

from engine import GoBoard, BLACK, WHITE


def test_empty_point_rejected_when_occupied():
    b = GoBoard()
    assert b.play_move(3, 3) is True
    assert b.play_move(3, 3) is False


def test_turn_alternation():
    b = GoBoard()
    assert b.current_player == BLACK
    b.play_move(3, 3)
    assert b.current_player == WHITE
    b.pass_move()
    assert b.current_player == BLACK


def test_all_points_legal_on_empty_board():
    b = GoBoard()
    assert len(b.get_legal_moves()) == 49


def test_pass_then_move_resets_pass_counter():
    b = GoBoard()
    b.pass_move()
    assert b.consecutive_passes == 1
    assert not b.is_game_over()
    b.play_move(0, 0)
    assert b.consecutive_passes == 0


def test_two_consecutive_passes_end_game():
    b = GoBoard()
    b.pass_move()
    b.pass_move()
    assert b.is_game_over()


def test_illegal_move_does_not_mutate_state():
    b = GoBoard()
    b.play_move(3, 3)
    snap = (b.black, b.white, b.zobrist, b.current_player, b.move_number)
    assert b.play_move(3, 3) is False
    assert (b.black, b.white, b.zobrist, b.current_player, b.move_number) == snap


def test_play_index_matches_play_move_and_pass_index():
    a = GoBoard()
    b = GoBoard()
    a.play_move(3, 3)
    b.play_index(3 * 7 + 3)
    assert a.black == b.black
    assert b.play_index(b.PASS) is True
    assert b.consecutive_passes == 1


def test_empty_board_zobrist_is_zero():
    assert GoBoard().zobrist == 0


def test_single_stone_suicide_is_illegal():
    b = GoBoard.from_ascii(
        """
        .......
        ...X...
        ..X.X..
        ...X...
        .......
        .......
        .......
        """,
        to_move=WHITE,
    )
    # (2,3) is surrounded by black; white there captures nothing -> suicide.
    assert b.play_move(2, 3) is False


def test_multi_stone_suicide_is_illegal():
    b = GoBoard.from_ascii(
        """
        .......
        .......
        ...XX..
        ..XO.X.
        ...XX..
        .......
        .......
        """,
        to_move=WHITE,
    )
    # White at (3,3); the empty (3,4) is the group's last liberty. Filling it makes
    # the 2-stone white group liberty-less while capturing nothing -> illegal.
    assert b.play_move(3, 4) is False
