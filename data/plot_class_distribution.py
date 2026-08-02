"""Plot the 50-class move-label distribution (EXECUTION_Phase1.md task 1.9).

Required by the course rubric. Also the evidence for F3's corrected macro-F1
justification: the v2 plan predicted a "severely imbalanced" distribution
dominated by centre/tengen-adjacent points; measured, the single outsized
class is `pass`, and the board-point distribution is comparatively flat.

Run from the repo root:
    venv/bin/python -m data.plot_class_distribution
"""

import argparse
import os

import h5py
import numpy as np

BOARD = 7
PASS_INDEX = BOARD * BOARD


def _labels():
    labels = [f"{r},{c}" for r in range(BOARD) for c in range(BOARD)]
    labels.append("pass")
    return labels


def plot(h5_path: str, out_path: str, split_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(h5_path, "r") as h:
        moves = h["moves"][:]

    n = len(moves)
    counts = np.bincount(moves, minlength=PASS_INDEX + 1)
    shares = 100.0 * counts / n

    order = np.arange(PASS_INDEX + 1)
    colors = ["#4C72B0"] * PASS_INDEX + ["#C44E52"]  # pass bar highlighted

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(order, shares[order], color=colors, width=0.9)
    ax.set_xlabel("Move index (0-48 board points row-major, 49 = pass)")
    ax.set_ylabel("Share of positions (%)")
    ax.set_title(
        f"ZetaGo 7x7 move-label distribution ({split_name}, n={n:,})\n"
        f"mode = pass ({shares[PASS_INDEX]:.2f}%); "
        f"max board point = index {int(np.argmax(shares[:PASS_INDEX]))} "
        f"({shares[:PASS_INDEX].max():.2f}%)"
    )
    ax.axhline(shares[PASS_INDEX], color="#C44E52", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlim(-1, PASS_INDEX + 1)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}  (mode=pass {shares[PASS_INDEX]:.3f}%, "
          f"max board point {shares[:PASS_INDEX].max():.3f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/processed/train.h5")
    ap.add_argument("--out", default="Docs/paper/figures/class_distribution.pdf")
    ap.add_argument("--split-name", default="train")
    args = ap.parse_args()
    plot(args.inp, args.out, args.split_name)


if __name__ == "__main__":
    main()
