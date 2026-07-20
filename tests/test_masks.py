"""Bit-geometry tests: neighbours, edge-safe shifts (no wraparound), flood-fill."""

from engine.masks import geometry


def test_neighbors_center():
    g = geometry(7)
    i = 3 * 7 + 3
    expected = (
        (1 << (2 * 7 + 3))
        | (1 << (4 * 7 + 3))
        | (1 << (3 * 7 + 2))
        | (1 << (3 * 7 + 4))
    )
    assert g.NEIGHBORS[i] == expected


def test_neighbors_corner():
    g = geometry(7)
    assert g.NEIGHBORS[0] == (1 << 1) | (1 << 7)  # (0,0) -> (0,1),(1,0)


def test_shift_east_no_wrap_off_right_edge():
    g = geometry(7)
    # (3,6) shifted east leaves the board; it must NOT appear at (4,0).
    assert g.shift_e(1 << (3 * 7 + 6)) == 0


def test_shift_west_no_wrap_off_left_edge():
    g = geometry(7)
    assert g.shift_w(1 << (3 * 7 + 0)) == 0


def test_dilate_is_cell_plus_neighbors():
    g = geometry(7)
    i = 3 * 7 + 3
    assert g.dilate(1 << i) == (1 << i) | g.NEIGHBORS[i]


def test_flood_fills_connected_region_only():
    g = geometry(7)
    top_row = (1 << 0) | (1 << 1) | (1 << 2)
    detached = 1 << (5 * 7 + 5)
    region = top_row | detached
    assert g.flood(1 << 0, region) == top_row
