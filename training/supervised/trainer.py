from __future__ import annotations

"""Trainer orchestration for Track A supervised experiments.

This module wires together the data loader, feature encoders, model factory,
and evaluation code. It keeps experiments reproducible by using the exact
train/val splits produced by the dataset builder and by permitting fixed
random seeds and deterministic subsampling for baseline models.

Flow:
 - load HDF5 splits via `load_split`
 - build features per-encoding with `make_features`
 - instantiate models from `create_model(name, in_channels, args, device)`
 - call `fit` then `predict`, then compute metrics and record results
"""

import argparse
import time
from typing import List

import numpy as np

from .data import load_split, subsample
from .features import make_features
from .metrics import print_summary, save_metrics, topk_acc_from_proba, value_metrics
from .models import create_model
from .types import RunMetrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track A supervised experiments on ZetaGo train/val.")
    ap.add_argument("--train-h5", default="data/processed/train.h5")
    ap.add_argument("--val-h5", default="data/processed/val.h5")
    ap.add_argument("--encodings", nargs="+", type=int, default=[2, 4, 7], choices=[2, 4, 7])
    ap.add_argument("--model", default="all", choices=["all", "logreg", "rf", "knn", "cnn"])

    ap.add_argument("--max-train", type=int, default=0, help="cap rows loaded from train.h5 (0=all)")
    ap.add_argument("--max-val", type=int, default=0, help="cap rows loaded from val.h5 (0=all)")
    ap.add_argument(
        "--baseline-train-cap",
        type=int,
        default=80000,
        help="cap rows used by sklearn-style baselines after feature build (0=all)",
    )

    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--knn-k", type=int, default=11)

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-loss-weight", type=float, default=0.5)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default="results/supervised_track_a_metrics.csv")
    ap.add_argument("--out-json", default="results/supervised_track_a_metrics.json")
    return ap.parse_args()


def _resolve_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    if arg == "cuda":
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("--device cuda requested but PyTorch is not installed") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return "cuda"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _selected_models(model_arg: str) -> List[str]:
    if model_arg == "all":
        return ["logreg", "rf", "knn", "cnn"]
    return [model_arg]


def _prepare_baseline_train(
    x_train: np.ndarray,
    y_move_train: np.ndarray,
    y_value_train: np.ndarray,
    cap: int,
    seed: int,
):
    sample = subsample({"x": x_train, "ym": y_move_train, "yv": y_value_train}, cap, seed)
    xb = sample["x"].reshape(len(sample["x"]), -1)
    return xb, sample["ym"], sample["yv"]


def _evaluate_one(
    name: str,
    enc: int,
    model,
    x_train: np.ndarray,
    y_move_train: np.ndarray,
    y_value_train: np.ndarray,
    x_val: np.ndarray,
    y_move_val: np.ndarray,
    y_value_val: np.ndarray,
    baseline_train_cap: int,
    seed: int,
) -> RunMetrics:
    if model.expects_flattened:
        xb, ymb, yvb = _prepare_baseline_train(x_train, y_move_train, y_value_train, baseline_train_cap, seed)
        xv = x_val.reshape(len(x_val), -1)
    else:
        xb, ymb, yvb = x_train, y_move_train, y_value_train
        xv = x_val

    train_start = time.perf_counter()
    model.fit(xb, ymb, yvb)
    train_seconds = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    move_proba, value_pred = model.predict(xv)
    infer_seconds = time.perf_counter() - infer_start

    move_pred = np.argmax(move_proba, axis=1)
    top1 = float(np.mean(move_pred == y_move_val))
    top3 = topk_acc_from_proba(move_proba, y_move_val, 3)
    mse, mae, vacc = value_metrics(value_pred, y_value_val)

    return RunMetrics(
        model=name,
        encoding=enc,
        move_top1=top1,
        move_top3=top3,
        value_mse=mse,
        value_mae=mae,
        value_acc=vacc,
        train_seconds=train_seconds,
        infer_seconds_total=infer_seconds,
        infer_ms_per_sample=1000.0 * infer_seconds / max(1, len(y_move_val)),
    )


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    print("Loading HDF5 splits...")
    train = load_split(args.train_h5, max_rows=args.max_train)
    val = load_split(args.val_h5, max_rows=args.max_val)
    print(f"train rows: {len(train['moves']):,}")
    print(f"val rows:   {len(val['moves']):,}")

    rows: List[RunMetrics] = []
    names = _selected_models(args.model)

    for enc in args.encodings:
        print(f"\nBuilding features for N={enc}...")
        x_train = make_features(train, enc)
        x_val = make_features(val, enc)

        y_move_train = train["moves"]
        y_move_val = val["moves"]
        y_value_train = train["values"]
        y_value_val = val["values"]

        for name in names:
            print(f"Training {name}...")
            model = create_model(name, in_channels=x_train.shape[1], args=args, device=device)
            metrics = _evaluate_one(
                name,
                enc,
                model,
                x_train,
                y_move_train,
                y_value_train,
                x_val,
                y_move_val,
                y_value_val,
                args.baseline_train_cap,
                args.seed,
            )
            rows.append(metrics)

    print_summary(rows)
    save_metrics(rows, args.out_csv, args.out_json)
    print(f"\nSaved: {args.out_csv}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
