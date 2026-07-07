"""Tromp-Taylor area scoring, including the dead-stone / neutral edge cases."""

from engine import GoBoard, tromp_taylor_area

HALF = "\n".join(["..X.O.."] * 7)        # black wall col 2, white wall col 4 -> area (21, 21)
OFFSET = "\n".join(["..X..O."] * 7)      # black col 2, white col 5 -> area (21, 14)


def test_empty_board_scores_zero_area():
    b = GoBoard()
    assert tromp_taylor_area(b.black, b.white) == (0, 0)
    assert b.get_final_score() == (0.0, 9.5, "White")


def test_full_black_board():
    b = GoBoard.from_ascii("\n".join(["XXXXXXX"] * 7))
    assert tromp_taylor_area(b.black, b.white) == (49, 0)
    assert b.get_final_score() == (49.0, 9.5, "Black")


def test_split_territory_with_komi():
    b = GoBoard.from_ascii(HALF)
    assert tromp_taylor_area(b.black, b.white) == (21, 21)
    assert b.get_final_score() == (21.0, 30.5, "White")


def test_single_dame_counts_for_neither():
    b = GoBoard.from_ascii(
        "\n".join(
            ["XXXXXXX", "XXXXXXX", "XXXOXXX", "XXX.XXX", "XXXXXXX", "XXXXXXX", "XXXXXXX"]
        )
    )
    # (3,3) is empty and borders both white (2,3) and black -> neutral.
    assert tromp_taylor_area(b.black, b.white) == (47, 1)


def test_dead_stone_is_not_removed():
    b = GoBoard.from_ascii(
        "\n".join(
            ["X.OOOOO", "..OOOOO", "OOOOOOO", "OOOOOOO", "OOOOOOO", "OOOOOOO", "OOOOOOO"]
        )
    )
    # The lone black stone (0,0) still counts 1 for Black, and the empty pocket it
    # touches becomes neutral (bordered by both) rather than white territory.
    assert tromp_taylor_area(b.black, b.white) == (1, 45)


def test_komi_flips_the_winner():
    near = GoBoard.from_ascii(OFFSET, komi=6.5)
    assert tromp_taylor_area(near.black, near.white) == (21, 14)
    assert near.get_final_score()[2] == "Black"        # 21 vs 14 + 6.5 = 20.5
    far = GoBoard.from_ascii(OFFSET, komi=9.5)
    assert far.get_final_score()[2] == "White"         # 21 vs 14 + 9.5 = 23.5


def test_tie_is_reachable_only_with_integer_komi():
    b = GoBoard.from_ascii(HALF, komi=0.0)
    assert b.get_final_score() == (21.0, 21.0, "Tie")
