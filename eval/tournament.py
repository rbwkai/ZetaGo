"""Round-robin tournament runner: colour-balanced by construction.

**Why colour balance is mandatory, not optional (EXECUTION_Phase1.md SS5):**
White wins 90.6% of games in this corpus (DATASET.md SS8) -- an unbalanced
colour assignment would swamp every model-strength difference. `round_robin`
therefore plays every pairing with each agent taking Black exactly as often as
White.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
    sys.path.insert(0, os.path.join(_root_dir, "environment"))

from engine import GoBoard  # noqa: E402

from .agents import apply_move  # noqa: E402


@dataclass
class GameResult:
    black: str
    white: str
    winner: str  # "black" | "white" | "tie"
    black_score: float
    white_score: float
    plies: int


def play_game(black_name: str, black_agent, white_name: str, white_agent,
              board_size: int = 7, komi: float = 9.5, max_plies: int = 200) -> GameResult:
    # max_plies is a real backstop, not a defensive-only default: a
    # UniformRandomAgent rarely chooses to pass by chance (1 in ~legal-moves+1
    # each turn) and keeps refilling captured territory, so random-vs-random
    # games routinely run past 100 plies and some hit the cap without a
    # natural double-pass. Scored via Tromp-Taylor area on whatever position
    # exists at truncation -- verified in task 1.5's harness self-test.
    board = GoBoard(n=board_size, komi=komi)
    black_agent.reset()
    white_agent.reset()

    while not board.is_game_over() and board.move_number < max_plies:
        agent = black_agent if board.current_player == board.BLACK else white_agent
        move_index = agent.select_move(board)
        legal = apply_move(board, move_index)
        if not legal:
            # An agent proposed an illegal move (shouldn't happen if it respects
            # the legal mask) -- treat as a forced pass rather than crash the
            # tournament, so one bad agent can't take down the whole run.
            board.pass_move()

    black_score, white_score, winner_str = board.get_final_score()
    winner = {"Black": "black", "White": "white", "Tie": "tie"}[winner_str]
    return GameResult(
        black=black_name, white=white_name, winner=winner,
        black_score=black_score, white_score=white_score, plies=board.move_number,
    )


def round_robin(
    agents: Dict[str, object],
    games_per_colour: int,
    seed: int = 0,
    board_size: int = 7,
    komi: float = 9.5,
) -> List[GameResult]:
    """Every unordered pair of distinct agents plays `games_per_colour` games
    with each agent as Black, and `games_per_colour` as White -- colour-balanced
    by construction, independent of `games_per_colour`'s value."""
    names = list(agents.keys())
    results: List[GameResult] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for _ in range(games_per_colour):
                results.append(play_game(a, agents[a], b, agents[b], board_size, komi))
                results.append(play_game(b, agents[b], a, agents[a], board_size, komi))
    return results


def win_rate(results: List[GameResult], name: str) -> float:
    played = [r for r in results if name in (r.black, r.white)]
    if not played:
        return float("nan")
    wins = 0.0
    for r in played:
        if r.winner == "tie":
            wins += 0.5
        elif (r.winner == "black" and r.black == name) or (r.winner == "white" and r.white == name):
            wins += 1.0
    return wins / len(played)
