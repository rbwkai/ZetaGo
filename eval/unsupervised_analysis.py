"""Track B analysis: PCA, k-means, DBSCAN, seed stability, cluster
characterisation, and linear probes with randomised controls
(EXECUTION_Phase3.md tasks 3.2/3.3/3.4).

Runs on deduplicated validation positions, encoded through each trained AE
checkpoint (`training/train_autoencoder.py`, task 3.1). Only ever reads
`states` for encoding; board statistics (probe/characterisation targets) come
from `training.unsupervised.board_stats.board_statistics`, which itself never
reads `moves`/`values`/`margins`.

k for k-means is selected per seed by silhouette score (design's stated
protocol), not fixed in advance -- disagreement between seeds about k is
itself part of what "clusters are stable" (P3-G4) means, so it is reported,
not concealed by forcing a common k. DBSCAN runs on a PCA-reduced space with
an eps set by a fixed nearest-neighbour-distance heuristic (R3: "do not tune
eps until it produces something photogenic") -- a degenerate result (one
cluster, or all noise) is a valid, expected outcome and is reported as such.

Usage:
  python -m eval.unsupervised_analysis --model-dir models/unsupervised
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from training.supervised.data import apply_dedup, load_split
from training.unsupervised.autoencoder import load_autoencoder
from training.unsupervised.board_stats import board_statistics

BOARD_STATS_KEYS = [
    "stone_count",
    "move_no",
    "largest_group_size",
    "num_groups",
    "num_groups_atari",
    "edge_vs_centre",
]


def load_val_positions(val_h5: str, max_rows: int = 0):
    """Deduplicated validation positions (F2: unconditional): raw N=2 stone
    planes for encoding, plus board statistics for probe/characterisation
    targets. Never touches `moves`/`values`/`margins`."""
    split = load_split(val_h5, max_rows=max_rows)
    split = apply_dedup(split, "unique")
    states_n2 = np.asarray(split["states"][:, :2], dtype=np.float32)
    stats = board_statistics(split)
    return states_n2, stats


def pca_curve(latents: np.ndarray) -> Dict:
    pca = PCA(n_components=min(latents.shape))
    pca.fit(latents)
    evr = pca.explained_variance_ratio_
    return {"explained_variance_ratio": evr.tolist(), "cumulative": np.cumsum(evr).tolist()}


def kmeans_silhouette_sweep(latents: np.ndarray, k_range: range, seed: int):
    sweep = {}
    best = None
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(latents)
        s = float(silhouette_score(latents, labels))
        sweep[k] = s
        if best is None or s > best[1]:
            best = (k, s, labels)
    best_k, best_score, best_labels = best
    return best_k, best_score, best_labels, sweep


def dbscan_on_pca(latents: np.ndarray, seed: int, variance_target: float = 0.9, min_samples: int = 10) -> Dict:
    pca = PCA(n_components=variance_target, random_state=seed)
    reduced = pca.fit_transform(latents)
    nn = NearestNeighbors(n_neighbors=min_samples).fit(reduced)
    dists, _ = nn.kneighbors(reduced)
    eps = float(np.median(dists[:, -1]))  # fixed heuristic, not tuned per-run (R3)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(reduced)
    n_clusters = len({l for l in labels if l != -1})
    return {
        "n_components": int(reduced.shape[1]),
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "noise_frac": float(np.mean(labels == -1)),
        "degenerate": n_clusters <= 1,
    }


def characterise_clusters(labels: np.ndarray, stats: Dict[str, np.ndarray]) -> Dict:
    out = {}
    for c in sorted(set(labels)):
        mask = labels == c
        entry = {"n": int(mask.sum())}
        for key in BOARD_STATS_KEYS:
            vals = stats[key][mask].astype(np.float64)
            entry[key] = {"mean": float(vals.mean()), "std": float(vals.std())}
        out[str(c)] = entry
    return out


def probe_with_control(latents: np.ndarray, target: np.ndarray, seed: int, test_size: float = 0.3) -> Dict:
    """A probe (latent -> target) paired with a shuffled-target control on the
    identical train/test split (`hewitt2019designing`). Neither number is
    reportable without the other (P3-G5)."""
    target = np.asarray(target, dtype=np.float64)
    idx = np.arange(len(target))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    probe = LinearRegression().fit(latents[train_idx], target[train_idx])
    probe_r2 = float(r2_score(target[test_idx], probe.predict(latents[test_idx])))

    rng = np.random.default_rng(seed)
    shuffled = target.copy()
    rng.shuffle(shuffled)
    control = LinearRegression().fit(latents[train_idx], shuffled[train_idx])
    control_r2 = float(r2_score(shuffled[test_idx], control.predict(latents[test_idx])))

    return {"probe_r2": probe_r2, "control_r2": control_r2, "beats_control": probe_r2 > control_r2}


def analyze_latent_size(
    latent_size: int,
    seeds: List[int],
    model_dir: str,
    val_states: np.ndarray,
    val_stats: Dict[str, np.ndarray],
    k_range: range,
    dbscan_min_samples: int,
    probe_seed: int,
    probe_test_size: float,
) -> Dict:
    per_seed = {}
    labels_by_seed = {}
    for seed in seeds:
        path = os.path.join(model_dir, f"ae_latent{latent_size}_seed{seed}.pt")
        print(f"  [latent={latent_size} seed={seed}] encoding {len(val_states):,} val positions...")
        model = load_autoencoder(path)
        z = model.encode(val_states)

        best_k, best_sil, labels, sweep = kmeans_silhouette_sweep(z, k_range, seed)
        labels_by_seed[seed] = labels

        per_seed[seed] = {
            "pca": pca_curve(z),
            "kmeans": {"k": best_k, "silhouette": best_sil, "sweep": sweep},
            "dbscan": dbscan_on_pca(z, seed, min_samples=dbscan_min_samples),
            "clusters": characterise_clusters(labels, val_stats),
            "probes": {
                key: probe_with_control(z, val_stats[key], probe_seed, probe_test_size)
                for key in BOARD_STATS_KEYS
            },
        }

    ari = {}
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            ari[f"{a}v{b}"] = float(adjusted_rand_score(labels_by_seed[a], labels_by_seed[b]))

    return {"per_seed": per_seed, "seed_stability_ari": ari}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track B: PCA/k-means/DBSCAN/probes analysis (tasks 3.2-3.4).")
    ap.add_argument("--val-h5", default="data/processed/val.h5")
    ap.add_argument("--model-dir", default="models/unsupervised")
    ap.add_argument("--latents", nargs="+", type=int, default=[64, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--dbscan-min-samples", type=int, default=10)
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--probe-test-size", type=float, default=0.3)
    ap.add_argument("--max-val", type=int, default=0, help="debug cap on rows read from val.h5 (0=all)")
    ap.add_argument("--out-json", default="results/unsupervised_track_b_analysis.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading + deduplicating validation split...")
    val_states, val_stats = load_val_positions(args.val_h5, max_rows=args.max_val)
    print(f"val: {len(val_states):,} unique positions")

    k_range = range(args.k_min, args.k_max + 1)
    results = {}
    for latent in args.latents:
        print(f"\n=== latent={latent} ===")
        results[latent] = analyze_latent_size(
            latent,
            args.seeds,
            args.model_dir,
            val_states,
            val_stats,
            k_range,
            args.dbscan_min_samples,
            args.probe_seed,
            args.probe_test_size,
        )
        for seed in args.seeds:
            km = results[latent]["per_seed"][seed]["kmeans"]
            db = results[latent]["per_seed"][seed]["dbscan"]
            print(f"  seed={seed}: kmeans k={km['k']} silhouette={km['silhouette']:.3f} | "
                  f"dbscan clusters={db['n_clusters']} noise={db['noise_frac']:.2f}")
        print(f"  seed-stability ARI: {results[latent]['seed_stability_ari']}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
