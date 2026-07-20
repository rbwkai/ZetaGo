"""Oracle cross-check: replay real KataGo self-play games through the engine.

These tests are the strongest correctness gate: KataGo generated the games under the
exact same rules the engine claims to implement (positional superko, area scoring,
suicide illegal, komi 9.5). Every KataGo move must therefore be legal in the engine,
and scored (non-resign) games must agree on the winner. Skipped automatically until
self-play data exists in ``data/raw/sgf``.
"""

import os

import pytest

from engine import GoBoard

pytestmark = pytest.mark.oracle

SGF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "sgf")


def _load_games(limit):
    from data.sgf_reader import iter_files, iter_games

    files = iter_files(SGF_DIR)
    if not files:
        return None
    out = []
    for game in iter_games(files):
        out.append(game)
        if len(out) >= limit:
            break
    return out


def _replay(game):
    """Replay a game; return the board, or None if any move was illegal."""
    b = GoBoard(n=game.size, komi=game.komi or 9.5)
    for color, point in game.moves:
        if color != b.current_player:
            return None
        if point is None:
            b.pass_move()
        elif not b.play_move(*point):
            return None
    return b


def test_engine_replays_katago_selfplay_legally():
    games = _load_games(300)
    if not games:
        pytest.skip("no KataGo .sgfs data in data/raw/sgf yet")
    checked = 0
    for game in games:
        if game.size != 7:
            continue
        assert _replay(game) is not None, (
            f"KataGo game illegal in engine (rules mismatch!): {game.result}"
        )
        checked += 1
    assert checked >= 100


def test_engine_score_agrees_with_katago_on_counted_games():
    games = _load_games(4000)
    if not games:
        pytest.skip("no KataGo .sgfs data in data/raw/sgf yet")
    compared = 0
    agree = 0
    for game in games:
        # Only games decided by counting are a fair area-score oracle; resignations
        # (and timeouts) end before the position is fully resolved.
        if game.size != 7 or game.winner == 0 or game.is_resign or "+T" in game.result.upper():
            continue
        board = _replay(game)
        if board is None:
            continue
        winner = board.get_final_score()[2]
        engine_winner = 1 if winner == "Black" else -1 if winner == "White" else 0
        compared += 1
        agree += int(engine_winner == game.winner)
    if compared == 0:
        pytest.skip("no counted (non-resign) games available to compare")
    assert agree / compared >= 0.98, f"engine/KataGo winner agreement only {agree}/{compared}"
