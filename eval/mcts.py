"""PUCT tree search over one frozen supervised model -- evaluation-time only.

Design and rationale: EXECUTION_Phase2.md SS1. This module never updates the
wrapped model's weights and never records a training example; `run_mcts` is
called once per real move (from `MCTSAgent.select_move` in `eval/agents.py`)
and returns a single move, exactly like `GreedyAgent` does with one forward
pass instead of `n_simulations`. No Dirichlet root noise -- that is a
self-play *training*-time exploration device with no role in a pure
strongest-move search.

`n_simulations`/`c_puct`/`margin_scale` are configuration, not tuned values,
same status as `GreedyAgent`'s `min_ply_before_pass` (which this module also
applies, at every node, not just the root).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from training.supervised.features import make_features


class _Node:
    __slots__ = ("board", "parent", "prior", "visit_count", "value_sum", "children", "expanded")

    def __init__(self, board, parent: Optional["_Node"], prior: float):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[Optional[int], "_Node"] = {}
        self.expanded = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def _puct_score(parent: _Node, child: _Node, c_puct: float) -> float:
    # child.value is backed up from the child's own side-to-move's perspective;
    # the parent's side-to-move is the opponent, hence the negation.
    q = -child.value
    u = c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
    return q + u


def _evaluate_leaf(board, model, encoding, states, players, move_nos, game_id, min_ply_before_pass):
    """One forward pass at `board`'s position. Returns (priors, raw_value) where
    priors maps a legal move index (or None for pass) to a probability, and
    raw_value is the model's predicted score margin (side-to-move's
    perspective -- DATASET.md/build_dataset.py's stored convention, so no sign
    flip is needed here; only the *backup* step negates across levels)."""
    split = {
        "states": np.stack(states),
        "players": np.asarray(players, dtype=np.int8),
        "game_id": np.full(len(states), game_id, dtype=np.uint32),
        "move_no": np.asarray(move_nos, dtype=np.int32),
    }
    x = make_features(split, encoding)[-1:]
    if getattr(model, "expects_flattened", True):
        x = x.reshape(1, -1)
    move_proba, value_pred = model.predict(x)
    proba = move_proba[0].astype(np.float64).copy()
    raw_value = float(value_pred[0])

    legal_mask = board.legal_moves_mask().astype(bool)
    proba[~legal_mask] = 0.0

    pass_index = board.N * board.N
    board_legal = legal_mask.copy()
    board_legal[pass_index] = False
    if board.move_number < min_ply_before_pass and board_legal.any():
        proba[pass_index] = 0.0

    total = proba.sum()
    if total <= 0.0:
        # Same fallback as GreedyAgent: every board move masked and pass
        # forbidden by the min-ply rule can't happen together, but if the
        # model assigned zero probability everywhere, fall back to uniform
        # over whatever remains legal rather than produce a degenerate node.
        proba = board_legal.astype(np.float64) if board_legal.any() else legal_mask.astype(np.float64)
        total = proba.sum()
    proba /= total

    priors: Dict[Optional[int], float] = {}
    for i in range(board.N * board.N):
        if legal_mask[i] and proba[i] > 0.0:
            priors[i] = float(proba[i])
    if legal_mask[pass_index] and proba[pass_index] > 0.0:
        priors[None] = float(proba[pass_index])
    return priors, raw_value


def _terminal_value(board) -> float:
    """+-1 from the perspective of whoever is to move at this (game-over)
    node; 0 for a tie. Uses the true Tromp-Taylor outcome rather than a
    network guess -- strictly more correct once the game has actually ended."""
    _, _, winner_str = board.get_final_score()
    if winner_str == "Tie":
        return 0.0
    side_to_move_is_black = board.current_player == board.BLACK
    black_won = winner_str == "Black"
    return 1.0 if (black_won == side_to_move_is_black) else -1.0


def run_mcts(
    board,
    model,
    encoding: int,
    states: List[np.ndarray],
    players: List[int],
    move_nos: List[int],
    game_id: int,
    n_simulations: int = 100,
    c_puct: float = 1.5,
    min_ply_before_pass: int = 20,
    margin_scale: float = 15.0,
) -> Optional[int]:
    """Run `n_simulations` PUCT simulations from `board`'s position and return
    the move (policy index, or None for pass) with the highest root visit
    count. `states`/`players`/`move_nos` are the *real* game's history so far,
    already including `board`'s own entry (the same convention `GreedyAgent`
    uses) -- every simulated line extends its own copy of this path, so N=7's
    history planes stay correct for wherever a given simulation currently is."""
    root = _Node(board, None, 1.0)

    for _ in range(n_simulations):
        node = root
        path_states = list(states)
        path_players = list(players)
        path_move_nos = list(move_nos)

        # 1) Select down to an unexpanded leaf.
        while node.expanded and node.children:
            node = max(node.children.values(), key=lambda c: _puct_score(node, c, c_puct))
            path_states.append(node.board.get_tensor())
            path_players.append(node.board.current_player)
            path_move_nos.append(node.board.move_number)

        # 2) Expand + evaluate the leaf.
        if node.board.is_game_over():
            value = _terminal_value(node.board)
        else:
            priors, raw_value = _evaluate_leaf(
                node.board, model, encoding, path_states, path_players, path_move_nos,
                game_id, min_ply_before_pass,
            )
            value = float(np.tanh(raw_value / margin_scale))
            for move, p in priors.items():
                child_board = node.board.copy()
                if move is None:
                    child_board.pass_move()
                else:
                    child_board.play_index(move)
                node.children[move] = _Node(child_board, node, p)
            node.expanded = True

        # 3) Backup, negating at each level (zero-sum, alternating perspective).
        v = value
        cur: Optional[_Node] = node
        while cur is not None:
            cur.visit_count += 1
            cur.value_sum += v
            v = -v
            cur = cur.parent

    return max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
