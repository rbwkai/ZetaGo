#!/usr/bin/env python3
"""CLI for Track B autoencoder training (EXECUTION_Phase3.md task 3.1).

Loops over --latents x --seeds, training one ConvAutoencoder per cell on the
deduplicated N=2 training positions (training.unsupervised.data.load_unlabelled_train).
Per F6, every run's weights are persisted via --save-model-dir starting from
the first run, not as a follow-up task -- the sweep saves after every cell,
same discipline as training/supervised/trainer.py's --resume/--save-model-dir.
"""

import argparse
import json
import os
import sys
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from training.unsupervised.autoencoder import ConvAutoencoder
from training.unsupervised.data import load_unlabelled_train


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track B: convolutional autoencoder training.")
    ap.add_argument("--train-h5", default="data/processed/train.h5")
    ap.add_argument("--latents", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--holdout-seed", type=int, default=0, help="fixed across all cells so every latent/seed sees the same fit/holdout split")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--save-model-dir", default="models/unsupervised")
    ap.add_argument("--out-json", default="results/unsupervised_track_b_training.json")
    ap.add_argument("--max-train", type=int, default=0, help="debug cap on rows read from train.h5 (0=all)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading + deduplicating train split...")
    fit_states, holdout_states = load_unlabelled_train(
        args.train_h5, holdout_frac=args.holdout_frac, seed=args.holdout_seed, max_rows=args.max_train
    )
    print(f"fit: {len(fit_states):,} positions  holdout: {len(holdout_states):,} positions")

    os.makedirs(args.save_model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    results = []
    for latent in args.latents:
        for seed in args.seeds:
            print(f"\n=== latent={latent} seed={seed} ===")
            model = ConvAutoencoder(
                in_channels=fit_states.shape[1],
                hidden=args.hidden,
                latent=latent,
                board_size=fit_states.shape[-1],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
                device=args.device,
            )
            t0 = time.perf_counter()
            history = model.fit(fit_states, holdout_states)
            train_seconds = time.perf_counter() - t0

            model_path = os.path.join(args.save_model_dir, f"ae_latent{latent}_seed{seed}.pt")
            model.save(model_path)  # F6: persisted now, not as a follow-up

            results.append(
                {
                    "latent": latent,
                    "seed": seed,
                    "epochs": args.epochs,
                    "final_train_loss": history[-1]["train_loss"],
                    "final_holdout_loss": history[-1]["holdout_loss"],
                    "train_seconds": train_seconds,
                    "model_path": model_path,
                    "history": history,
                }
            )
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[latent={latent} seed={seed}] saved {model_path} ({train_seconds:.1f}s)")

    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
