"""Parse KataGo ``.sgfs`` self-play files into lightweight records for replay.

KataGo's ``match`` writes one SGF game per line in a ``.sgfs`` file. We only need a
few properties, so this is a small, dependency-free scanner rather than a full SGF
parser. SGF point coordinates use a top-left origin, which matches the engine's
``(row, col)`` with row 0 at the top::

    col = ord(x) - ord('a')      row = ord(y) - ord('a')
"""

import glob
import os
import re
from dataclasses import dataclass

_RE_SIZE = re.compile(r"SZ\[([^\]]*)\]")
_RE_KOMI = re.compile(r"KM\[([^\]]*)\]")
_RE_RESULT = re.compile(r"RE\[([^\]]*)\]")
_RE_RULES = re.compile(r"RU\[([^\]]*)\]")
_RE_MOVE = re.compile(r";([BW])\[([a-z]{0,2})\]")


@dataclass
class SgfGame:
    size: int
    komi: float
    result: str               # raw RE value, e.g. "B+1.5", "W+R", "0", ""
    rules: str                # raw RU value
    moves: list               # list of (color, point): color in {1,-1}; point=(row,col) or None
    winner: int               # 1 black, -1 white, 0 unknown / jigo / no-result

    @property
    def is_resign(self):
        return self.result[2:3].upper() == "R" if len(self.result) > 2 else False


def _point(coord, size):
    """SGF coordinate string -> (row, col), or None for a pass / out-of-range vertex."""
    if not coord:
        return None
    col = ord(coord[0]) - 97
    row = ord(coord[1]) - 97
    if 0 <= col < size and 0 <= row < size:
        return (row, col)
    return None  # e.g. legacy "tt" pass on small boards


def parse_record(line, default_size=7):
    """Parse a single SGF game line into an :class:`SgfGame`."""
    m = _RE_SIZE.search(line)
    size = int(m.group(1)) if m else default_size
    m = _RE_KOMI.search(line)
    komi = float(m.group(1)) if m and m.group(1) else 0.0
    m = _RE_RESULT.search(line)
    result = m.group(1) if m else ""
    m = _RE_RULES.search(line)
    rules = m.group(1) if m else ""

    moves = []
    for color, coord in _RE_MOVE.findall(line):
        moves.append((1 if color == "B" else -1, _point(coord, size)))

    head = result[:1].upper()
    winner = 1 if head == "B" else -1 if head == "W" else 0
    return SgfGame(size=size, komi=komi, result=result, rules=rules, moves=moves, winner=winner)


def iter_files(sgf_dir):
    """Yield ``.sgfs`` file paths in a directory, sorted for determinism."""
    return sorted(glob.glob(os.path.join(sgf_dir, "*.sgfs")))


def iter_games(paths):
    """Yield :class:`SgfGame` for every game line across the given ``.sgfs`` files."""
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield parse_record(line)
