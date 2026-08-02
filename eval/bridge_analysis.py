"""Track A <-> B bridge (EXECUTION_Phase3.md task 3.5): does supervision buy a
representation that encodes board properties better than reconstruction alone?

Runs the identical clustering (task 3.2) and probe/control protocol (task 3.4)
against the champion CNN's 64-d value-head penultimate representation, on the
*same* deduplicated validation positions and the same board statistics used
for the AE analysis (P3-G6). The champion is an N=4 model while the AE is N=2
(EXECUTION_Phase3.md SS3.A) -- the two are not fed identical inputs, and that
is reported as a stated limitation rather than papered over with a re-train
(SS4 task 3.5's own recommendation).

Usage:
  python -m eval.bridge_analysis --model-path models/supervised/cnn_enc4_seed46_vol107969_none.pt --encoding 4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from training.supervised.data import apply_dedup, load_split
from training.supervised.features import make_features
from training.supervised.models.cnn_model import load_cnn
from training.unsupervised.board_stats import board_statistics

from .unsupervised_analysis import (
    BOARD_STATS_KEYS,
    characterise_clusters,
    dbscan_on_pca,
    kmeans_silhouette_sweep,
    pca_curve,
    probe_with_control,
)


def extract_penultimate(model, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """64-d value-head penultimate (post-ReLU: `value_head[4]`), via a forward
    hook (F4) so this reads exactly what the real forward pass produces."""
    torch = model.torch
    captured = {}

    def _hook(_module, _inp, out):
        captured["z"] = out.detach().cpu().numpy()

    handle = model.model.value_head[4].register_forward_hook(_hook)
    model.model.eval()
    outs = []
    try:
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                xb = torch.from_numpy(np.asarray(x[i : i + batch_size], dtype=np.float32)).to(model.device)
                model.model(xb)
                outs.append(captured["z"])
    finally:
        handle.remove()
    return np.concatenate(outs, axis=0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track A <-> B bridge: champion 64-d penultimate vs AE latent.")
    ap.add_argument("--val-h5", default="data/processed/val.h5")
    ap.add_argument("--model-path", default="models/supervised/cnn_enc4_seed46_vol107969_none.pt")
    ap.add_argument("--encoding", type=int, default=4)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--dbscan-min-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-test-size", type=float, default=0.3)
    ap.add_argument("--max-val", type=int, default=0, help="debug cap on rows read from val.h5 (0=all)")
    ap.add_argument("--out-json", default="results/unsupervised_track_b_bridge.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading + deduplicating validation split...")
    split = load_split(args.val_h5, max_rows=args.max_val)
    split = apply_dedup(split, "unique")
    print(f"val: {len(split['moves']):,} unique positions")

    stats = board_statistics(split)
    x = make_features(split, args.encoding)

    print(f"Loading champion {args.model_path} (N={args.encoding})...")
    model = load_cnn(args.model_path)
    z = extract_penultimate(model, x)
    assert z.shape[1] == 64, f"expected a 64-d penultimate, got {z.shape[1]}"

    k_range = range(args.k_min, args.k_max + 1)
    best_k, best_sil, labels, sweep = kmeans_silhouette_sweep(z, k_range, args.seed)
    db = dbscan_on_pca(z, args.seed, min_samples=args.dbscan_min_samples)
    clusters = characterise_clusters(labels, stats)
    probes = {key: probe_with_control(z, stats[key], args.seed, args.probe_test_size) for key in BOARD_STATS_KEYS}

    print(f"kmeans k={best_k} silhouette={best_sil:.3f} | dbscan clusters={db['n_clusters']} noise={db['noise_frac']:.2f}")
    for key in BOARD_STATS_KEYS:
        p = probes[key]
        print(f"  probe[{key}]: r2={p['probe_r2']:.3f} control_r2={p['control_r2']:.3f} beats_control={p['beats_control']}")

    payload = {
        "model_path": args.model_path,
        "encoding": args.encoding,
        "latent_dim": int(z.shape[1]),
        "n_positions": int(len(z)),
        "limitation": (
            f"champion is N={args.encoding}, AE comparison latents are N=2 (SS3.A) -- "
            "inputs are not identical between the two representations compared here"
        ),
        "pca": pca_curve(z),
        "kmeans": {"k": best_k, "silhouette": best_sil, "sweep": sweep},
        "dbscan": db,
        "clusters": clusters,
        "probes": probes,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
