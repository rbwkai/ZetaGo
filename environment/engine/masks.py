"""Board geometry: bit layout, neighbour masks, and bitwise flood-fill.

The whole rules core is pure-Python integer-bitmask arithmetic (no NumPy): a board
position is two ints, ``black`` and ``white``. Bit ``i = row * N + col`` is set when
that colour occupies the point. Row 0 is the top of the board.
"""

from functools import lru_cache


def iter_bits(x):
    """Yield the indices of the set bits in integer ``x`` (ascending)."""
    while x:
        lsb = x & (-x)
        yield lsb.bit_length() - 1
        x ^= lsb


def popcount(x):
    """Number of set bits."""
    return x.bit_count()


class Geometry:
    """Precomputed masks and shift / flood operations for an N x N board."""

    def __init__(self, n):
        self.N = n
        self.NN = n * n
        self.FULL = (1 << self.NN) - 1
        # File masks used to stop horizontal shifts from wrapping across rows.
        self.LEFT_FILE = sum(1 << (r * n + 0) for r in range(n))
        self.RIGHT_FILE = sum(1 << (r * n + (n - 1)) for r in range(n))
        self.NOT_LEFT = self.FULL & ~self.LEFT_FILE
        self.NOT_RIGHT = self.FULL & ~self.RIGHT_FILE
        self.NEIGHBORS = [self._neighbors(i) for i in range(self.NN)]
        self.SINGLE = [1 << i for i in range(self.NN)]

    def _neighbors(self, i):
        r, c = divmod(i, self.N)
        m = 0
        if r > 0:
            m |= 1 << (i - self.N)
        if r < self.N - 1:
            m |= 1 << (i + self.N)
        if c > 0:
            m |= 1 << (i - 1)
        if c < self.N - 1:
            m |= 1 << (i + 1)
        return m

    # Whole-board one-step shifts. shift_w/shift_e mask the opposite file so that a
    # stone on an edge column does not wrap onto the adjacent row.
    def shift_n(self, b):
        return b >> self.N

    def shift_s(self, b):
        return (b << self.N) & self.FULL

    def shift_w(self, b):
        return (b >> 1) & self.NOT_RIGHT

    def shift_e(self, b):
        return (b << 1) & self.NOT_LEFT

    def dilate(self, b):
        """``b`` grown by one step in all four orthogonal directions."""
        return (
            b
            | (b >> self.N)
            | ((b << self.N) & self.FULL)
            | ((b >> 1) & self.NOT_RIGHT)
            | ((b << 1) & self.NOT_LEFT)
        ) & self.FULL

    def flood(self, seed, region):
        """All cells 4-connected to ``seed`` while staying inside ``region``."""
        g = seed & region
        if not g:
            return 0
        while True:
            nxt = (g | self.dilate(g)) & region
            if nxt == g:
                return g
            g = nxt


@lru_cache(maxsize=None)
def geometry(n):
    """Return the (cached) Geometry for an N x N board."""
    return Geometry(n)
