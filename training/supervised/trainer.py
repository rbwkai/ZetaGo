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
import hashlib

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
    ap.add_argument("--dedupe-val", action="store_true", default=True, help="drop val positions that exactly match any train position (by flattened hash)")
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
        # Extract targets and run lightweight diagnostics to catch common dataset leaks early.
        y_move_train = train["moves"]
        y_move_val = val["moves"]
        y_value_train = train["values"]
        y_value_val = val["values"]

        # Run lightweight diagnostics to catch common dataset leaks early.
        def _run_diagnostics(enc, xtr, xv, ytr, yv, tr_split, val_split):
            print(f"[diag] running diagnostics for encoding={enc} on {len(ytr):,} train rows and {len(yv):,} val rows")
            # Basic target checks
            u_tr, c_tr = np.unique(ytr, return_counts=True)
            u_val, c_val = np.unique(yv, return_counts=True)
            print(f"[diag] train unique moves: {len(u_tr)} (sample: {u_tr[:5].tolist()})")
            print(f"[diag] val   unique moves: {len(u_val)} (sample: {u_val[:5].tolist()})")

            if len(u_tr) <= 1:
                print("[diag][ERROR] train targets appear constant or degenerate")

            # Check for accidental column-wise leakage: if any flattened feature column
            # equals the target vector for many rows, that's strong evidence of leakage.
            Xf = xtr.reshape(len(xtr), -1)
            ncols = Xf.shape[1]
            # check a few metadata arrays for exact column matches
            for name, arr in (('moves', ytr), ('move_no', tr_split['move_no']), ('game_id', tr_split['game_id'])):
                arr = arr.astype(Xf.dtype)
                # compute fraction of rows where column equals target per column
                # avoid allocating huge intermediate by computing in blocks if needed
                matches = (Xf == arr[:, None])
                col_frac = matches.sum(axis=0) / float(len(arr))
                best_idx = int(col_frac.argmax())
                best_frac = float(col_frac[best_idx])
                if best_frac > 0.9:
                    print(f"[diag][ERROR] column {best_idx} in flattened features matches '{name}' for {best_frac*100:.2f}% of rows (likely leakage)")
                elif best_frac > 0.01:
                    # map flattened column to (plane,row,col) when shape is canonical
                    plane = None
                    row = col = None
                    try:
                        per_plane = xtr.shape[1] * xtr.shape[2] * xtr.shape[3]
                    except Exception:
                        per_plane = None
                    if Xf.shape[1] and xtr.ndim == 4:
                        C = xtr.shape[1]
                        plane_size = xtr.shape[2] * xtr.shape[3]
                        plane = best_idx // plane_size
                        rem = best_idx % plane_size
                        row = rem // xtr.shape[3]
                        col = rem % xtr.shape[3]
                        print(f"[diag][WARN] best match for '{name}' is column {best_idx} (plane={plane}, r={row}, c={col}) with {best_frac*100:.2f}% equality — investigate correlation/leakage")
                    else:
                        print(f"[diag][WARN] best match for '{name}' is column {best_idx} with {best_frac*100:.2f}% equality — investigate correlation/leakage")

            # Check for exact duplicate flattened positions between train and val
            def _hash_rows(A):
                return np.array([hashlib.sha1(r.tobytes()).hexdigest() for r in A])

            try:
                htr = _hash_rows(Xf)
                hv = _hash_rows(xv.reshape(len(xv), -1))
                inter = np.intersect1d(htr, hv)
                if len(inter) > 0:
                    print(f"[diag][WARN] found {len(inter):,} exact duplicate positions between train and val (possible contamination)")
            except Exception:
                # Hashing may be expensive for very large datasets; ignore failures
                pass

        _run_diagnostics(enc, x_train, x_val, y_move_train, y_move_val, train, val)

        # Optionally drop any validation positions that exactly match a train position
        if args.dedupe_val:
            try:
                htr = np.array([hashlib.sha1(r.tobytes()).hexdigest() for r in x_train.reshape(len(x_train), -1)])
                hv = np.array([hashlib.sha1(r.tobytes()).hexdigest() for r in x_val.reshape(len(x_val), -1)])
                keep_mask = ~np.isin(hv, htr)
                removed = int((~keep_mask).sum())
                if removed:
                    print(f"[diag] --dedupe-val: removed {removed:,} val positions that matched train positions")
                x_val = x_val[keep_mask]
                y_move_val = y_move_val[keep_mask]
                y_value_val = y_value_val[keep_mask]
                # also update val metadata dict slices so downstream code (if any) can access them
                for k in ("players", "game_id", "move_no", "margins"):
                    if k in val:
                        val[k] = val[k][keep_mask]
            except Exception:
                print("[diag] --dedupe-val: hashing failed or OOM; skipping dedupe")

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
