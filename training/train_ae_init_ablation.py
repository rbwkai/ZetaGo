#!/usr/bin/env python3
"""AE-init vs random-init at the smallest data volume (EXECUTION_Phase3.md task 3.6).

Copies task 3.1's trained AE encoder body (three conv layers, architecturally
identical to the CNN body -- F3) into a fresh N=2 CNN, trains at the
1,000-game volume with `--dedup none` and 5 seeds -- matching the Track A
sweep's own N=2/1,000-game cells -- against random-init as the control arm.

Both arms are trained locally in this same run rather than compared against
the frozen sweep JSON directly: that sweep was GPU-trained, this one is CPU,
and `main.tex` SS VI documents ~0.003 run-to-run drift from that source alone.
The sweep's own mean for this cell (0.8272, 5 seeds, recomputed from
`kaggle/result/supervised_track_a_metrics.json` -- not the pooled-encoding
0.8192 that appears in the paper's Table V, see EXECUTION_Phase3.md SS4 task
3.6) is reported alongside purely as reference context.

One row, one table (SS2): this is not a second headline result. If AE-init
does not help, that is a publishable null and is reported as one.
"""

import argparse
import json
import os
import sys
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import numpy as np

from training.supervised.data import apply_dedup, load_split, subsample_games
from training.supervised.features import make_features
from training.supervised.losses import WeightedPolicyValueLoss
from training.supervised.metrics import bootstrap_ci_over_games, value_metrics
from training.supervised.models.cnn_model import CNNModel
from training.unsupervised.autoencoder import load_autoencoder


def train_one(x_train, y_move_train, y_value_train, x_val, y_move_val, y_value_val, val_game_id, seed, args, ae_body_state):
    model = CNNModel(
        in_channels=x_train.shape[1],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=seed,
        device="cpu",
        loss_fn=WeightedPolicyValueLoss(args.value_loss_weight),
        value_scale=args.value_scale,
    )
    if ae_body_state is not None:
        # Overwrites only the body's random init (F3: identical architecture,
        # so the state dict keys already match); policy/value heads keep the
        # seed-derived random init from CNNModel.__init__ above.
        model.model.body.load_state_dict(ae_body_state)

    t0 = time.perf_counter()
    model.fit(x_train, y_move_train, y_value_train)
    train_seconds = time.perf_counter() - t0

    move_proba, value_pred = model.predict(x_val)
    move_pred = np.argmax(move_proba, axis=1)
    correct = (move_pred == y_move_val).astype(np.float64)
    top1 = float(correct.mean())
    ci_lo, ci_hi = bootstrap_ci_over_games(correct, val_game_id, seed=seed)
    _mse, _mae, vacc = value_metrics(value_pred, y_value_val)
    return {
        "seed": seed,
        "move_top1": top1,
        "move_top1_ci_lo": ci_lo,
        "move_top1_ci_hi": ci_hi,
        "value_acc": vacc,
        "train_seconds": train_seconds,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track B task 3.6: AE-init vs random-init at 1k games, N=2.")
    ap.add_argument("--train-h5", default="data/processed/train.h5")
    ap.add_argument("--val-h5", default="data/processed/val.h5")
    ap.add_argument("--ae-checkpoint", default="models/unsupervised/ae_latent64_seed42.pt")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--dedup", default="none", help="matches the Track A sweep's N=2/1k cells")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=12, help="matches the Track A sweep default")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--value-loss-weight", type=float, default=0.5)
    ap.add_argument("--value-scale", type=float, default=15.0)
    ap.add_argument("--max-train", type=int, default=0, help="debug cap on rows read from train.h5 (0=all)")
    ap.add_argument("--max-val", type=int, default=0, help="debug cap on rows read from val.h5 (0=all)")
    ap.add_argument("--out-json", default="results/unsupervised_track_b_ae_init_ablation.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading train/val splits...")
    train_full = load_split(args.train_h5, max_rows=args.max_train)
    val = load_split(args.val_h5, max_rows=args.max_val)

    print(f"Loading AE checkpoint: {args.ae_checkpoint}")
    ae = load_autoencoder(args.ae_checkpoint)
    ae_body_state = ae.encoder.body.state_dict()

    x_val = make_features(val, 2)
    y_move_val = val["moves"]
    y_value_val = val["margins"]

    results = {"ae_init": [], "random_init": []}
    for seed in args.seeds:
        train_sub = subsample_games(train_full, args.games, seed)
        train_sub = apply_dedup(train_sub, args.dedup)
        x_train = make_features(train_sub, 2)
        y_move_train = train_sub["moves"]
        y_value_train = train_sub["margins"]

        print(f"[seed={seed}] random-init...")
        r_random = train_one(
            x_train, y_move_train, y_value_train, x_val, y_move_val, y_value_val, val["game_id"],
            seed, args, ae_body_state=None,
        )
        results["random_init"].append(r_random)
        print(f"  top1={r_random['move_top1']:.4f}")

        print(f"[seed={seed}] ae-init...")
        r_ae = train_one(
            x_train, y_move_train, y_value_train, x_val, y_move_val, y_value_val, val["game_id"],
            seed, args, ae_body_state=ae_body_state,
        )
        results["ae_init"].append(r_ae)
        print(f"  top1={r_ae['move_top1']:.4f}")

        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Task 3.6 summary ===")
    for arm in ("random_init", "ae_init"):
        vals = [r["move_top1"] for r in results[arm]]
        mean = sum(vals) / len(vals)
        sd = float(np.std(vals))
        print(f"{arm:>12}: mean={mean:.4f} sd={sd:.4f} n={len(vals)} per-seed={[round(v, 4) for v in vals]}")
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
