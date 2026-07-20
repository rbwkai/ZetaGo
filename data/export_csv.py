"""Export an HDF5 dataset to a human-readable CSV.

Each CSV row is one position. The board is reconstructed in absolute colours
(``X`` = Black, ``O`` = White, ``.`` = empty) as seven 7-character rows joined by
``/`` so a person can read it directly, e.g. ``......./...XO../..XO.O./...`` .

    venv/bin/python -m data.export_csv                       # train.h5+val.h5 -> .csv, plus sample
    venv/bin/python -m data.export_csv --in data/processed/val.h5 --out data/processed/val.csv
    venv/bin/python -m data.export_csv --in data/processed/train.h5 --out s.csv --limit 500
"""

import argparse
import csv
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOARD = 7
PASS_INDEX = BOARD * BOARD
_GLYPHS = np.frombuffer(b".XO", dtype=np.uint8)   # code 0=empty, 1=black, 2=white
_HEADER = ["game_id", "move_no", "to_move", "board",
           "move", "move_index", "value", "margin", "winner"]


def _move_str(idx):
    return "pass" if idx == PASS_INDEX else f"{idx // BOARD},{idx % BOARD}"


def export(in_path, out_path, limit=0, block=8192):
    with h5py.File(in_path, "r") as h:
        n = len(h["moves"])
        total = min(limit, n) if limit else n
        states = h["states"]
        moves = h["moves"][:total]
        values = h["values"][:total]
        margins = h["margins"][:total]
        players = h["players"][:total]
        game_id = h["game_id"][:total]
        move_no = h["move_no"][:total]

        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(_HEADER)
            for lo in range(0, total, block):
                hi = min(lo + block, total)
                st = states[lo:hi]                      # (b, 6, 7, 7)
                pl = players[lo:hi]
                black_to_move = (pl == 1)[:, None, None]
                black = np.where(black_to_move, st[:, 0], st[:, 1]).astype(np.uint8)
                white = np.where(black_to_move, st[:, 1], st[:, 0]).astype(np.uint8)
                glyphs = _GLYPHS[black + 2 * white]     # (b, 7, 7) ascii codes
                for k in range(hi - lo):
                    i = lo + k
                    rows = glyphs[k].tobytes().decode("ascii")
                    board = "/".join(rows[r * BOARD:(r + 1) * BOARD] for r in range(BOARD))
                    p = int(pl[k])
                    to_move = "B" if p == 1 else "W"
                    v = int(values[i])
                    winner = "-" if v == 0 else (to_move if v == 1 else ("W" if to_move == "B" else "B"))
                    m = float(margins[i])
                    margin = "" if np.isnan(m) else f"{m:.1f}"
                    w.writerow([int(game_id[i]), int(move_no[i]), to_move, board,
                                _move_str(int(moves[i])), int(moves[i]), v, margin, winner])
    return total


def main():
    ap = argparse.ArgumentParser(description="Export an HDF5 dataset to human-readable CSV.")
    ap.add_argument("--in", dest="inp", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0, help="max rows (0 = all)")
    args = ap.parse_args()

    if args.inp:
        out = args.out or os.path.splitext(args.inp)[0] + ".csv"
        n = export(args.inp, out, args.limit)
        print(f"wrote {n:,} rows -> {out}")
        return

    # Default: full train.csv + val.csv plus a small readable sample.
    base = "data/processed"
    jobs = [
        (os.path.join(base, "val.h5"), os.path.join(base, "val.csv"), 0),
        (os.path.join(base, "train.h5"), os.path.join(base, "sample.csv"), 500),
        (os.path.join(base, "train.h5"), os.path.join(base, "train.csv"), 0),
    ]
    for inp, out, lim in jobs:
        if os.path.exists(inp):
            n = export(inp, out, lim)
            size_mb = os.path.getsize(out) / 1e6
            print(f"wrote {n:,} rows -> {out}  ({size_mb:.1f} MB)")
        else:
            print(f"skip (missing): {inp}")


if __name__ == "__main__":
    main()
