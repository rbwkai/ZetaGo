"""Learning curve: mean top-1 vs. mean unique training positions (log scale),
one line per model, mean over 3 encodings and 5 seeds (`main.tex` Fig. 2).

Regenerates a figure that previously had no committed generator
(`EXECUTION_Phase4.md` F4) -- this script is the reproducibility fix, not just
the figure. Emits PDF (vector line plot) rather than PNG.

Run from the repo root:
    venv/bin/python -m data.plot_learning_curve
"""

import argparse
import json
import os

import numpy as np

_MARKERS = {"logreg": "o", "rf": "s", "knn": "^", "svm": "D", "cnn": "*"}


def load_curve(metrics_json: str):
    rows = json.load(open(metrics_json))
    cells = [r for r in rows if r["seed"] != -1]
    models = sorted(set(r["model"] for r in cells))
    volumes = sorted(set(r["data_volume_games"] for r in cells))

    curve = {}
    for m in models:
        xs, ys = [], []
        for v in volumes:
            sel = [r for r in cells if r["model"] == m and r["data_volume_games"] == v]
            xs.append(float(np.mean([r["data_volume_unique"] for r in sel])))
            ys.append(float(np.mean([r["move_top1"] for r in sel])))
        curve[m] = (np.array(xs), np.array(ys))
    return curve


def plot(curve: dict, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m, (xs, ys) in sorted(curve.items()):
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker=_MARKERS.get(m, "o"), label=m)
    ax.set_xscale("log")
    ax.set_xlabel("mean unique training positions (log scale)")
    ax.set_ylabel("mean top-1 accuracy (3 encodings, 5 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics-json", default="results/supervised_track_a_metrics.json")
    ap.add_argument("--out", default="Docs/paper/figures/learning_curve.pdf")
    args = ap.parse_args()
    plot(load_curve(args.metrics_json), args.out)


if __name__ == "__main__":
    main()
