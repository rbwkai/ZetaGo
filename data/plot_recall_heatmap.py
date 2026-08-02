"""Per-point top-1 recall heatmap for the champion model, on the val split
(`main.tex` Fig. 3) -- the confusion-matrix artifact for this 50-way task: the
diagonal of a 50x50 confusion matrix, laid out on board geometry rather than
as an unreadable 50x50 grid (`EXECUTION_Phase4.md` SS3.C). `pass` (the 50th
class) does not fit the 7x7 geometry and is reported in the title instead.

Regenerates a figure that previously had no committed generator (F4). Uses
`pcolormesh`, not `imshow`, so the mesh faces stay vector paths in the PDF
rather than a rasterised image.

Run from the repo root:
    venv/bin/python -m data.plot_recall_heatmap
"""

import argparse
import os

import numpy as np

BOARD = 7
PASS_INDEX = BOARD * BOARD


def compute_recall(model_path: str, encoding: int, val_h5: str, max_rows: int = 0):
    from training.supervised.data import load_split
    from training.supervised.features import make_features
    from training.supervised.models.cnn_model import load_cnn

    split = load_split(val_h5, max_rows=max_rows)
    x = make_features(split, encoding)
    model = load_cnn(model_path, device="cpu")
    move_proba, _ = model.predict(x)
    move_pred = np.argmax(move_proba, axis=1)
    y_true = split["moves"]

    counts = np.bincount(y_true, minlength=PASS_INDEX + 1)
    recall = np.full(PASS_INDEX + 1, np.nan)
    for c in range(PASS_INDEX + 1):
        if counts[c] == 0:
            continue
        mask = y_true == c
        recall[c] = float((move_pred[mask] == c).mean())
    return recall, counts


def plot(recall: np.ndarray, counts: np.ndarray, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = recall[:PASS_INDEX].reshape(BOARD, BOARD)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    mesh = ax.pcolormesh(grid, cmap="RdYlGn", vmin=0, vmax=1, edgecolors="white", linewidth=0.5)
    ax.invert_yaxis()  # row 0 = top, matching engine.encode.board_to_array
    ax.set_xticks(np.arange(BOARD) + 0.5)
    ax.set_xticklabels(range(BOARD))
    ax.set_yticks(np.arange(BOARD) + 0.5)
    ax.set_yticklabels(range(BOARD))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    for r in range(BOARD):
        for c in range(BOARD):
            ax.text(
                c + 0.5, r + 0.5, f"{grid[r, c] * 100:.0f}",
                ha="center", va="center", fontsize=7,
                color="black" if grid[r, c] > 0.4 else "white",
            )
    fig.colorbar(mesh, ax=ax, label="top-1 recall")
    pass_recall, pass_n = recall[PASS_INDEX], int(counts[PASS_INDEX])
    ax.set_title(f"Champion per-point top-1 recall (val)\npass recall = {pass_recall * 100:.1f}% (n={pass_n:,})")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")

    centre = 3 * BOARD + 3
    orth = [centre - BOARD, centre + BOARD, centre - 1, centre + 1]
    print(f"centre (index {centre}) recall: {recall[centre] * 100:.1f}%")
    print(f"orthogonal-neighbour recalls: {[round(recall[i] * 100, 1) for i in orth]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="models/supervised/cnn_enc4_seed46_vol107969_none.pt")
    ap.add_argument("--encoding", type=int, default=4)
    ap.add_argument("--val-h5", default="data/processed/val.h5")
    ap.add_argument("--out", default="Docs/paper/figures/recall_heatmap.pdf")
    ap.add_argument("--max-rows", type=int, default=0, help="debug cap (0=all)")
    args = ap.parse_args()
    recall, counts = compute_recall(args.model_path, args.encoding, args.val_h5, args.max_rows)
    plot(recall, counts, args.out)


if __name__ == "__main__":
    main()
