"""Evaluation metrics and helpers.

This module provides small utilities used by the trainer to compute move
top-k accuracy from model probabilities, value regression metrics, and to
serialize results to CSV/JSON for later analysis.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Sequence, Tuple

import numpy as np

from .types import RunMetrics


def softmax_np(x: np.ndarray) -> np.ndarray:
    # Numerically stable row-wise softmax for model logits
    z = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def topk_acc_from_proba(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    k = min(k, proba.shape[1])
    topk = np.argpartition(-proba, kth=k - 1, axis=1)[:, :k]
    hit = (topk == y[:, None]).any(axis=1)
    return float(hit.mean())


def bootstrap_ci_over_games(
    per_row_value: np.ndarray,
    game_id: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """95% (by default) bootstrap CI for the mean of `per_row_value`, resampled
    over *games* rather than positions (EXECUTION_Phase1.md task 1.3): positions
    within a game are correlated, so resampling positions directly understates
    the interval. Resamples game indices with replacement, pools each resampled
    game's rows, and takes percentiles of the resulting means."""
    per_row_value = np.asarray(per_row_value, dtype=np.float64)
    uniq_games, inverse = np.unique(game_id, return_inverse=True)
    n_games = len(uniq_games)
    game_sum = np.bincount(inverse, weights=per_row_value, minlength=n_games)
    game_cnt = np.bincount(inverse, minlength=n_games).astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_games, size=(n_boot, n_games))
    boot_mean = game_sum[idx].sum(axis=1) / game_cnt[idx].sum(axis=1)
    lo, hi = np.percentile(boot_mean, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def value_metrics(pred: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """`pred`/`y` are score margins (task 1.1b), not +-1 win/loss labels. MSE/MAE
    are computed directly on the margin; `acc` is the *derived* win/loss accuracy
    from sign(margin), to compare against the side-to-move baseline -- never
    against a marginal-majority baseline (F2, EXECUTION_Phase1.md SS3)."""
    pred = np.asarray(pred, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    acc = float(np.mean(np.sign(pred) == np.sign(y)))
    return mse, mae, acc


def macro_f1(pred: np.ndarray, y: np.ndarray, n_classes: int = 50) -> float:
    """Unweighted mean per-class F1. A class with zero true and zero predicted
    instances contributes 0 by convention (matches the majority-class baseline:
    only the predicted class has a nonzero F1 term)."""
    pred = np.asarray(pred)
    y = np.asarray(y)
    f1s = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp / denom) if denom else 0.0
    return float(f1s.mean())


def majority_class_baseline(y: np.ndarray, n_classes: int = 50) -> Tuple[float, float]:
    """Policy baseline: always predict the single most frequent training class.
    Returns (top1_accuracy, macro_f1). Mandated by Guidelines v1.1 §2.3.6."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=n_classes)
    majority = int(counts.argmax())
    top1 = float(counts.max() / len(y)) if len(y) else 0.0
    pred = np.full(len(y), majority)
    return top1, macro_f1(pred, y, n_classes)


def uniform_random_legal_baseline(states: np.ndarray) -> float:
    """Policy baseline: expected top-1 accuracy of picking uniformly among legal
    moves. `states` plane index 5 is the engine's legal-board-point mask; pass
    is always legal so every position has (legal board points + 1) choices.
    This is an analytic expectation, not a simulated agent."""
    legal_board = states[:, 5].reshape(len(states), -1).sum(axis=1)
    legal_count = legal_board.astype(np.float64) + 1.0
    return float(np.mean(1.0 / legal_count))


def side_to_move_baseline(players: np.ndarray, values: np.ndarray) -> float:
    """Value baseline: 'predict a win iff White is to move' (F2, DATASET.md §8).
    Not the marginal-majority baseline -- see EXECUTION_Phase1.md §3."""
    players = np.asarray(players)
    values = np.asarray(values)
    pred = np.where(players == -1, 1, -1)
    return float(np.mean(pred == values))


def save_metrics(rows: Sequence[RunMetrics], out_csv: str, out_json: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    with open(out_json, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)


def print_summary(rows: Sequence[RunMetrics]) -> None:
    if not rows:
        return
    print("\n=== Supervised Track A Results ===")
    print(
        f"{'model':<14} {'enc':<3} {'seed':>4} {'games':>8} {'dedup':<8} {'top1':>7} {'top3':>7} "
        f"{'macroF1':>8} {'val_mse':>9} {'val_mae':>9} {'val_acc':>8} {'baseline':>8} {'train_s':>9}"
    )
    for r in rows:
        print(
            f"{r.model:<14} {r.encoding:<3d} {r.seed:>4d} {r.data_volume_games:>8d} {r.dedup:<8} "
            f"{r.move_top1:7.4f} {r.move_top3:7.4f} {r.move_macro_f1:8.4f} "
            f"{r.value_mse:9.4f} {r.value_mae:9.4f} {r.value_acc:8.4f} {r.value_baseline_acc:8.4f} "
            f"{r.train_seconds:9.2f}"
        )
