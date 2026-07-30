from __future__ import annotations

"""Trainer orchestration for Track A supervised experiments.

This module wires together the data loader, feature encoders, model factory,
and evaluation code. It keeps experiments reproducible by using the exact
train/val splits produced by the dataset builder and by permitting fixed
random seeds and deterministic subsampling for baseline models.

Flow (per EXECUTION_Phase1.md SS6, tasks 1.1/1.1b/1.2c/1.3/1.4):
 - load HDF5 splits via `load_split` (train/val)
 - compute the three data-derived baselines once (majority-class, uniform-
   random-legal, side-to-move) and record them as fixed reference rows
 - for each seed x Factor B data-volume level x feature encoding x model:
   - subsample train by game to the volume level (nested, deterministic per
     seed), then apply the chosen --dedup rule
   - build features, fit, predict, score against val (unchanged, full size)
   - record one RunMetrics row, with a 95% bootstrap-over-games CI on move
     top-1 accuracy

The value head predicts the score margin (task 1.1b), not win/loss: `values`
(+-1) is no longer used as a training target anywhere in this module. Derived
win/loss accuracy (sign of the predicted margin) is reported via
`value_acc`/`value_baseline_acc`, always beside the side-to-move baseline.
"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np

from .data import apply_dedup, load_split, subsample, subsample_games
from .features import make_features
from .metrics import (
    bootstrap_ci_over_games,
    majority_class_baseline,
    print_summary,
    save_metrics,
    side_to_move_baseline,
    topk_acc_from_proba,
    uniform_random_legal_baseline,
    value_metrics,
)
from .metrics import macro_f1 as macro_f1_fn
from .models import create_model
from .types import RunMetrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track A supervised experiments on ZetaGo train/val.")
    ap.add_argument("--train-h5", default="data/processed/train.h5")
    ap.add_argument("--val-h5", default="data/processed/val.h5")
    ap.add_argument("--encodings", nargs="+", type=int, default=[2, 4, 7], choices=[2, 4, 7])
    ap.add_argument("--model", default="all", choices=["all", "logreg", "rf", "knn", "svm", "cnn"])

    ap.add_argument("--max-train", type=int, default=0, help="debug cap on rows read from train.h5 (0=all)")
    ap.add_argument("--max-val", type=int, default=0, help="debug cap on rows read from val.h5 (0=all)")
    ap.add_argument(
        "--baseline-train-cap",
        type=int,
        default=80000,
        help="cap rows used by sklearn-style baselines after feature build (0=all)",
    )

    ap.add_argument(
        "--volumes",
        nargs="+",
        default=["full"],
        help="Factor B data-volume levels, in games sampled per seed (nested); "
        "e.g. --volumes 1000 5000 20000 full (EXECUTION_Phase1.md SS2)",
    )
    ap.add_argument(
        "--dedup",
        default="none",
        help="position-multiplicity control applied to the training split after "
        "volume subsampling: none | unique | cap:K (EXECUTION_Phase1.md task 1.2c)",
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="one RunMetrics row per seed")

    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--knn-k", type=int, default=11)
    ap.add_argument("--svm-c", type=float, default=1.0)

    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-loss-weight", type=float, default=0.5)
    ap.add_argument("--value-scale", type=float, default=15.0, help="CNN margin-target normaliser (task 1.1b)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

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
        return ["logreg", "rf", "knn", "svm", "cnn"]
    return [model_arg]


def _parse_volumes(tokens: List[str]) -> List[int]:
    """'full' -> 0 (meaning: no subsampling); otherwise an int game count."""
    out = []
    for t in tokens:
        out.append(0 if t == "full" else int(t))
    return out


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
    seed: int,
    dedup: str,
    volume: Tuple[int, int, int],
    model,
    x_train: np.ndarray,
    y_move_train: np.ndarray,
    y_value_train: np.ndarray,
    x_val: np.ndarray,
    y_move_val: np.ndarray,
    y_value_val: np.ndarray,
    val_game_id: np.ndarray,
    value_baseline_acc: float,
    baseline_train_cap: int,
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
    correct = (move_pred == y_move_val).astype(np.float64)
    top1 = float(correct.mean())
    ci_lo, ci_hi = bootstrap_ci_over_games(correct, val_game_id, seed=seed)
    top3 = topk_acc_from_proba(move_proba, y_move_val, 3)
    macro_f1 = macro_f1_fn(move_pred, y_move_val)
    mse, mae, vacc = value_metrics(value_pred, y_value_val)

    games, rows, unique = volume
    return RunMetrics(
        model=name,
        encoding=enc,
        seed=seed,
        dedup=dedup,
        data_volume_games=games,
        data_volume_rows=rows,
        data_volume_unique=unique,
        move_top1=top1,
        move_top1_ci_lo=ci_lo,
        move_top1_ci_hi=ci_hi,
        move_top3=top3,
        move_macro_f1=macro_f1,
        value_mse=mse,
        value_mae=mae,
        value_acc=vacc,
        value_baseline_acc=value_baseline_acc,
        train_seconds=train_seconds,
        infer_seconds_total=infer_seconds,
        infer_ms_per_sample=1000.0 * infer_seconds / max(1, len(y_move_val)),
    )


def _unique_position_count(states: np.ndarray) -> int:
    return len({s.tobytes() for s in states})


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    print("Loading HDF5 splits...")
    train_full = load_split(args.train_h5, max_rows=args.max_train)
    val = load_split(args.val_h5, max_rows=args.max_val)
    print(f"train rows: {len(train_full['moves']):,}  val rows: {len(val['moves']):,}")

    # Data-derived baselines: fixed properties of val, independent of model/seed/
    # volume/encoding. Computed once (SS3, SS1.1).
    maj_top1, maj_macro_f1 = majority_class_baseline(val["moves"])
    uniform_legal = uniform_random_legal_baseline(val["states"])
    value_baseline_acc = side_to_move_baseline(val["players"], val["values"])
    print(
        f"\nBaselines (val): majority-class top1={maj_top1:.4f} macro-F1={maj_macro_f1:.4f} | "
        f"uniform-random-legal top1={uniform_legal:.4f} | side-to-move value acc={value_baseline_acc:.4f}"
    )

    rows: List[RunMetrics] = [
        RunMetrics(
            model="baseline_majority_class", encoding=0, seed=-1, dedup="n/a",
            data_volume_games=-1, data_volume_rows=-1, data_volume_unique=-1,
            move_top1=maj_top1, move_top1_ci_lo=float("nan"), move_top1_ci_hi=float("nan"),
            move_top3=float("nan"), move_macro_f1=maj_macro_f1,
            value_mse=float("nan"), value_mae=float("nan"), value_acc=float("nan"),
            value_baseline_acc=value_baseline_acc, train_seconds=0.0,
            infer_seconds_total=0.0, infer_ms_per_sample=0.0,
        ),
        RunMetrics(
            model="baseline_uniform_random_legal", encoding=0, seed=-1, dedup="n/a",
            data_volume_games=-1, data_volume_rows=-1, data_volume_unique=-1,
            move_top1=uniform_legal, move_top1_ci_lo=float("nan"), move_top1_ci_hi=float("nan"),
            move_top3=float("nan"), move_macro_f1=float("nan"),
            value_mse=float("nan"), value_mae=float("nan"), value_acc=float("nan"),
            value_baseline_acc=value_baseline_acc, train_seconds=0.0,
            infer_seconds_total=0.0, infer_ms_per_sample=0.0,
        ),
        RunMetrics(
            model="baseline_side_to_move", encoding=0, seed=-1, dedup="n/a",
            data_volume_games=-1, data_volume_rows=-1, data_volume_unique=-1,
            move_top1=float("nan"), move_top1_ci_lo=float("nan"), move_top1_ci_hi=float("nan"),
            move_top3=float("nan"), move_macro_f1=float("nan"),
            value_mse=float("nan"), value_mae=float("nan"), value_acc=value_baseline_acc,
            value_baseline_acc=value_baseline_acc, train_seconds=0.0,
            infer_seconds_total=0.0, infer_ms_per_sample=0.0,
        ),
    ]

    names = _selected_models(args.model)
    volume_levels = _parse_volumes(args.volumes)
    val_features_cache: Dict[int, np.ndarray] = {}

    for seed in args.seeds:
        for n_games in volume_levels:
            train_sub = subsample_games(train_full, n_games, seed)
            train_sub = apply_dedup(train_sub, args.dedup)
            games_used = n_games if n_games > 0 else len(np.unique(train_full["game_id"]))
            rows_used = len(train_sub["moves"])
            unique_used = _unique_position_count(train_sub["states"])
            volume = (games_used, rows_used, unique_used)
            print(
                f"\n[seed {seed}] volume={games_used:,} games -> {rows_used:,} rows "
                f"({unique_used:,} unique) after --dedup {args.dedup}"
            )

            for enc in args.encodings:
                print(f"Building features for N={enc}...")
                x_train = make_features(train_sub, enc)
                if enc not in val_features_cache:
                    val_features_cache[enc] = make_features(val, enc)
                x_val = val_features_cache[enc]

                y_move_train = train_sub["moves"]
                y_move_val = val["moves"]
                y_value_train = train_sub["margins"]
                y_value_val = val["margins"]

                for name in names:
                    print(f"Training {name} (seed={seed}, N={enc}, games={games_used:,})...")
                    model = create_model(name, in_channels=x_train.shape[1], args=args, device=device, seed=seed)

                    metrics = _evaluate_one(
                        name, enc, seed, args.dedup, volume, model,
                        x_train, y_move_train, y_value_train,
                        x_val, y_move_val, y_value_val, val["game_id"],
                        value_baseline_acc, args.baseline_train_cap,
                    )
                    rows.append(metrics)

    print_summary(rows)
    save_metrics(rows, args.out_csv, args.out_json)
    print(f"\nSaved: {args.out_csv}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
