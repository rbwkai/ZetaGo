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


def value_metrics(pred: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    pred = np.asarray(pred, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    pred_cls = np.clip(np.rint(pred), -1, 1)
    acc = float(np.mean(pred_cls == y))
    return mse, mae, acc


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
        f"{'model':<14} {'enc':<3} {'top1':>7} {'top3':>7} {'val_mse':>9} {'val_mae':>9} "
        f"{'val_acc':>8} {'train_s':>9} {'infer_ms/sample':>16}"
    )
    for r in rows:
        print(
            f"{r.model:<14} {r.encoding:<3d} {r.move_top1:7.4f} {r.move_top3:7.4f} "
            f"{r.value_mse:9.4f} {r.value_mae:9.4f} {r.value_acc:8.4f} "
            f"{r.train_seconds:9.2f} {r.infer_ms_per_sample:16.4f}"
        )
