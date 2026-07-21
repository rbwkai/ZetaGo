"""Tromp-Taylor area scoring.

Pure area scoring with no life/death analysis and no dead-stone removal: every
stone on the board counts for its own colour, and an empty region counts for a
player only if every stone bordering it is that player's. Regions bordered by
both colours (dame) or by neither count for nobody.
"""

from .masks import geometry, popcount


def tromp_taylor_area(black, white, n=7):
    """Return ``(black_area, white_area)`` for the given bitboards."""
    geo = geometry(n)
    occupied = black | white
    empty = geo.FULL & ~occupied
    black_area = popcount(black)
    white_area = popcount(white)

    rem = empty
    while rem:
        seed = rem & (-rem)
        region = geo.flood(seed, empty)
        border = geo.dilate(region) & occupied
        touches_black = (border & black) != 0
        touches_white = (border & white) != 0
        if touches_black and not touches_white:
            black_area += popcount(region)
        elif touches_white and not touches_black:
            white_area += popcount(region)
        rem &= ~region

    return black_area, white_area
