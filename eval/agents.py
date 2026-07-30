"""Wrap models (and a uniform-random floor) as Go-playing agents.

An agent implements `reset()` and `select_move(board) -> int | None`, where the
returned value is a policy index in `0..N*N-1` for a board point, or `None` for
pass. `apply_move` below applies whichever the agent returns to a `GoBoard`.

**Pass rule (EXECUTION_Phase1.md task 1.5, per finding F3):** `pass` is the
single most frequent training label (9.1%), so an unconstrained argmax agent
passes far too eagerly and forfeits games it could still contest. `GreedyAgent`
uses the plan's simpler stated option -- forbid passing until a minimum ply --
rather than a score-estimate comparison, which would need a second model call
per candidate move. The threshold is a constructor argument (`min_ply_before_pass`,
default 20, chosen because >=10 stones are on the board in 67.4% of this
corpus's positions by roughly that point -- see DATASET.md SS9); it is a
heuristic, not a tuned value, and must be stated in the paper wherever this
agent is used.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np

from training.supervised.features import make_features


class Agent(Protocol):
    def reset(self) -> None: ...
    def select_move(self, board) -> Optional[int]: ...


def apply_move(board, move_index: Optional[int]) -> bool:
    """Apply a policy index (or None for pass) to `board`. Returns legality."""
    if move_index is None or move_index == board.PASS:
        return board.pass_move()
    return board.play_index(move_index)


class UniformRandomAgent:
    """Picks uniformly among all currently legal moves (board points + pass).
    The floor agent, and the harness's own self-test opponent (task 1.5)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def select_move(self, board) -> Optional[int]:
        legal = board.get_legal_moves()  # list of (row, col)
        choices: List[Optional[int]] = [r * board.N + c for r, c in legal]
        choices.append(None)  # pass is always legal
        idx = self.rng.integers(0, len(choices))
        return choices[idx]


class GreedyAgent:
    """Wraps a fitted `SupervisedModel` (SS `training/supervised/models/base.py`)
    as a player: builds features from the live game history each turn (so N=7's
    history planes see this game's actual past positions, not another game's),
    argmaxes the model's move probabilities over legal moves, and applies the
    pass rule described in the module docstring."""

    def __init__(self, model, encoding: int, min_ply_before_pass: int = 20, game_id: int = 0):
        self.model = model
        self.encoding = encoding
        self.min_ply_before_pass = min_ply_before_pass
        self.game_id = game_id
        self._states: List[np.ndarray] = []
        self._players: List[int] = []
        self._move_nos: List[int] = []

    def reset(self) -> None:
        self._states = []
        self._players = []
        self._move_nos = []

    def select_move(self, board) -> Optional[int]:
        self._states.append(board.get_tensor())
        self._players.append(board.current_player)
        self._move_nos.append(board.move_number)

        split = {
            "states": np.stack(self._states),
            "players": np.asarray(self._players, dtype=np.int8),
            "game_id": np.full(len(self._states), self.game_id, dtype=np.uint32),
            "move_no": np.asarray(self._move_nos, dtype=np.int32),
        }
        x = make_features(split, self.encoding)[-1:]
        if getattr(self.model, "expects_flattened", True):
            x = x.reshape(1, -1)
        move_proba, _ = self.model.predict(x)
        proba = move_proba[0].astype(np.float64).copy()

        legal_mask = board.legal_moves_mask().astype(bool)  # (N*N + 1,), pass always True
        proba[~legal_mask] = 0.0

        pass_index = board.N * board.N
        board_legal = legal_mask.copy()
        board_legal[pass_index] = False
        if board.move_number < self.min_ply_before_pass and board_legal.any():
            proba[pass_index] = 0.0

        if proba.sum() <= 0.0:
            # Every board move masked and pass forbidden by the min-ply rule
            # can't happen (pass is only forbidden when a board move exists),
            # but if the model assigned zero probability everywhere, fall back
            # to a uniform choice over whatever remains legal rather than crash.
            fallback = legal_mask.copy()
            if board.move_number < self.min_ply_before_pass and board_legal.any():
                fallback[pass_index] = False
            proba = fallback.astype(np.float64)

        move_index = int(np.argmax(proba))
        return None if move_index == pass_index else move_index
