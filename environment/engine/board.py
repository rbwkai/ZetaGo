"""GoBoard: an N x N Go engine using integer bitboards + Zobrist positional superko.

Rules (matched exactly by the KataGo data-generation config):
  * suicide is ILLEGAL (single- or multi-stone),
  * positional superko (no board position may ever repeat),
  * Tromp-Taylor area scoring with a fixed komi.

The public API mirrors the original ``go_board.py`` so existing callers and the GUI
keep working: ``play_move``, ``pass_move``, ``get_legal_moves``, ``get_final_score``,
``is_game_over``, ``get_state``, ``copy`` and the BLACK/WHITE/EMPTY constants.
"""

from .constants import BLACK, WHITE, EMPTY, KOMI, DEFAULT_SIZE
from .masks import geometry, iter_bits, popcount
from .zobrist import zobrist_table


class GoBoard:
    # Class-level constants, compatible with the original go_board.GoBoard API.
    BLACK = BLACK
    WHITE = WHITE
    EMPTY = EMPTY
    KOMI = KOMI
    BOARD_SIZE = DEFAULT_SIZE

    def __init__(self, n=DEFAULT_SIZE, komi=KOMI):
        self.N = n
        self.NN = n * n
        self.PASS = n * n
        self.komi = float(komi)
        self.BLACK = BLACK
        self.WHITE = WHITE
        self.EMPTY = EMPTY

        self._geo = geometry(n)
        self._zt = zobrist_table(n)

        self.black = 0
        self.white = 0
        self.current_player = BLACK
        self.zobrist = 0                  # board-only hash; empty board == 0
        self.position_history = {0}       # every board position ever seen (superko)
        self.consecutive_passes = 0
        self.last_move = None             # bit index of last stone, or None for pass/none
        self.move_number = 0
        self.captured_by_black = 0        # white stones captured by black
        self.captured_by_white = 0        # black stones captured by white

    # ---- construction helpers -------------------------------------------------
    @classmethod
    def from_ascii(cls, text, to_move=BLACK, n=DEFAULT_SIZE, komi=KOMI):
        """Build a board from an ASCII diagram (``X`` black, ``O`` white, ``.`` empty).

        Rows may be space-separated ("X O .") or contiguous (".XO...."). Used mainly
        by tests to set up exact positions. Resets superko history to this position.
        """
        board = cls(n=n, komi=komi)
        black = 0
        white = 0
        r = 0
        for line in text.strip("\n").splitlines():
            cells = line.split()
            if not cells:
                continue
            if len(cells) == 1 and len(cells[0]) > 1:
                cells = list(cells[0])
            for c, ch in enumerate(cells):
                i = r * n + c
                if ch in ("X", "x", "#", "B", "b"):
                    black |= 1 << i
                elif ch in ("O", "o", "W", "w", "@"):
                    white |= 1 << i
            r += 1
        board.black = black
        board.white = white
        board.current_player = to_move
        board.zobrist = board._hash_of(black, white)
        board.position_history = {board.zobrist}
        return board

    def copy(self):
        nb = GoBoard(self.N, self.komi)
        nb.black = self.black
        nb.white = self.white
        nb.current_player = self.current_player
        nb.zobrist = self.zobrist
        nb.position_history = set(self.position_history)
        nb.consecutive_passes = self.consecutive_passes
        nb.last_move = self.last_move
        nb.move_number = self.move_number
        nb.captured_by_black = self.captured_by_black
        nb.captured_by_white = self.captured_by_white
        return nb

    # ---- hashing --------------------------------------------------------------
    def _hash_of(self, black, white):
        """Recompute the board-only Zobrist hash from scratch (used for tests/setup)."""
        st = self._zt
        h = 0
        for j in iter_bits(black):
            h ^= st[0][j]
        for j in iter_bits(white):
            h ^= st[1][j]
        return h

    def _result_hash(self, i, captured, my_idx, opp_idx):
        """Incremental hash after placing my stone at ``i`` and removing ``captured``."""
        st = self._zt
        h = self.zobrist ^ st[my_idx][i]
        for j in iter_bits(captured):
            h ^= st[opp_idx][j]
        return h

    # ---- core move logic ------------------------------------------------------
    def _try_play(self, i):
        """Place the current player's stone at bit ``i`` and resolve captures.

        Returns ``(me2, opp2, captured)`` (new mover stones, new opponent stones,
        captured-opponent mask) or ``None`` if the move is suicide. Does not check
        occupancy or superko.
        """
        geo = self._geo
        if self.current_player == BLACK:
            me, opp = self.black, self.white
        else:
            me, opp = self.white, self.black

        me2 = me | (1 << i)
        empty = geo.FULL & ~(me2 | opp)

        captured = 0
        seen = 0
        for nb in iter_bits(geo.NEIGHBORS[i] & opp):
            bit = 1 << nb
            if bit & seen:
                continue
            group = geo.flood(bit, opp)
            seen |= group
            if geo.dilate(group) & empty == 0:
                captured |= group

        opp2 = opp & ~captured
        empty2 = geo.FULL & ~(me2 | opp2)
        my_group = geo.flood(1 << i, me2)
        if geo.dilate(my_group) & empty2 == 0:
            # No liberties after captures. A successful capture always frees a
            # liberty adjacent to ``i``, so this only triggers when nothing was
            # captured, i.e. a true suicide.
            return None
        return me2, opp2, captured

    def is_legal(self, row, col):
        return self._legal_index(row * self.N + col)

    def _legal_index(self, i):
        if (self.black | self.white) & (1 << i):
            return False
        res = self._try_play(i)
        if res is None:
            return False
        _, _, captured = res
        my_idx = 0 if self.current_player == BLACK else 1
        h = self._result_hash(i, captured, my_idx, 1 - my_idx)
        return h not in self.position_history

    def get_legal_moves(self):
        moves = []
        empty = self._geo.FULL & ~(self.black | self.white)
        for i in iter_bits(empty):
            if self._legal_index(i):
                moves.append((i // self.N, i % self.N))
        return moves

    def play_move(self, row, col):
        return self.play_index(row * self.N + col)

    def play_index(self, i):
        """Play board index ``i`` (0..NN-1), or pass if ``i == self.PASS``.

        Returns True if the move was legal and applied, False otherwise (no mutation).
        """
        if i == self.PASS:
            return self.pass_move()
        if (self.black | self.white) & (1 << i):
            return False
        res = self._try_play(i)
        if res is None:
            return False
        me2, opp2, captured = res
        my_idx = 0 if self.current_player == BLACK else 1
        new_hash = self._result_hash(i, captured, my_idx, 1 - my_idx)
        if new_hash in self.position_history:
            return False                      # positional superko

        if self.current_player == BLACK:
            self.black, self.white = me2, opp2
            self.captured_by_black += popcount(captured)
        else:
            self.white, self.black = me2, opp2
            self.captured_by_white += popcount(captured)
        self.zobrist = new_hash
        self.position_history.add(new_hash)
        self.last_move = i
        self.consecutive_passes = 0
        self.current_player = -self.current_player
        self.move_number += 1
        return True

    def pass_move(self):
        self.consecutive_passes += 1
        self.last_move = None
        self.current_player = -self.current_player
        self.move_number += 1
        return True

    def is_game_over(self):
        return self.consecutive_passes >= 2

    @property
    def last_move_rc(self):
        """Last move as (row, col), or None if the last action was a pass / none."""
        if self.last_move is None:
            return None
        return divmod(self.last_move, self.N)

    # ---- scoring --------------------------------------------------------------
    def get_final_score(self):
        from .scoring import tromp_taylor_area

        b, w = tromp_taylor_area(self.black, self.white, self.N)
        black = float(b)
        white = float(w) + self.komi
        if black > white:
            winner = "Black"
        elif white > black:
            winner = "White"
        else:
            winner = "Tie"
        return black, white, winner

    # ---- encoding / interop (NumPy lives in encode.py) ------------------------
    def get_state(self):
        from .encode import board_to_array
        return board_to_array(self), self.current_player

    def get_tensor(self):
        from .encode import board_to_tensor
        return board_to_tensor(self)

    def legal_moves_mask(self):
        from .encode import legal_moves_mask
        return legal_moves_mask(self)

    def __str__(self):
        lines = ["  " + " ".join(str(c) for c in range(self.N))]
        for r in range(self.N):
            row = [str(r)]
            for c in range(self.N):
                bit = 1 << (r * self.N + c)
                row.append("X" if self.black & bit else "O" if self.white & bit else ".")
            lines.append(" ".join(row))
        return "\n".join(lines)
