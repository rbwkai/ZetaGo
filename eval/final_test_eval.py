"""Final held-out evaluation harness (EXECUTION_Phase4.md task 4.0a).

Loads every persisted checkpoint in `--model-dir` (the naming convention
`training.supervised.trainer` writes: `{model}_enc{N}_seed{S}_vol{G}_{dedup}.{pt,joblib}`),
evaluates each against `--split {val,test}`, and reports every metric the
paper's Results section quotes -- including the three data-derived baselines,
recomputed on that split (never carried over from a different split's
numbers). Output rows are always labelled with the split actually used, so a
val-split output can never be mistaken for a test-split one (F2).

Discipline enforced here, not just documented (SS3.A):
  - developed and exercised against val first (P4-G2's acceptance check);
  - `--split test` must be typed explicitly, there is no default;
  - the output file is written once -- a second run refuses to overwrite an
    existing `--out-json`/`--out-csv` unless `--force` is passed, and prints
    a warning banner when it is;
  - the output file's SHA-256 is printed on completion, for P4-G1's record.

Usage:
  python -m eval.final_test_eval --split val  --model-dir models/final_row
  python -m eval.final_test_eval --split test --model-dir models/final_row --force
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

from training.supervised.data import load_split
from training.supervised.features import make_features
from training.supervised.metrics import (
    bootstrap_ci_over_games,
    macro_f1 as macro_f1_fn,
    majority_class_baseline,
    save_metrics as _save_supervised_metrics,
    side_to_move_baseline,
    topk_acc_from_proba,
    uniform_random_legal_baseline,
    value_metrics,
)

_CKPT_RE = re.compile(r"^(?P<model>[a-z]+)_enc(?P<enc>\d+)_seed(?P<seed>\d+)_vol(?P<vol>\d+)_(?P<dedup>\w+)\.(pt|joblib)$")


@dataclass
class FinalMetrics:
    split: str
    model: str
    encoding: int
    seed: int
    dedup: str
    data_volume_games: int
    checkpoint: str
    move_top1: float
    move_top1_ci_lo: float
    move_top1_ci_hi: float
    move_top3: float
    move_macro_f1: float
    value_mse: float
    value_mae: float
    value_acc: float
    value_baseline_acc: float
    majority_class_baseline_top1: float
    uniform_random_legal_baseline_top1: float


def discover_checkpoints(model_dir: str) -> List[Dict]:
    found = []
    for fn in sorted(os.listdir(model_dir)):
        m = _CKPT_RE.match(fn)
        if not m:
            continue
        found.append(
            {
                "path": os.path.join(model_dir, fn),
                "model": m.group("model"),
                "encoding": int(m.group("enc")),
                "seed": int(m.group("seed")),
                "volume_games": int(m.group("vol")),
                "dedup": m.group("dedup"),
            }
        )
    return found


def _load_model(entry: Dict):
    if entry["model"] == "cnn":
        from training.supervised.models.cnn_model import load_cnn

        return load_cnn(entry["path"], device="cpu")
    import joblib

    return joblib.load(entry["path"])


def evaluate_checkpoint(entry: Dict, split: str, split_data: Dict[str, np.ndarray]) -> FinalMetrics:
    model = _load_model(entry)
    enc = entry["encoding"]

    x = make_features(split_data, enc)
    if model.expects_flattened:
        x = x.reshape(len(x), -1)

    move_proba, value_pred = model.predict(x)
    y_move = split_data["moves"]
    y_value = split_data["margins"]

    move_pred = np.argmax(move_proba, axis=1)
    correct = (move_pred == y_move).astype(np.float64)
    top1 = float(correct.mean())
    ci_lo, ci_hi = bootstrap_ci_over_games(correct, split_data["game_id"], seed=entry["seed"])
    top3 = topk_acc_from_proba(move_proba, y_move, 3)
    macro_f1 = macro_f1_fn(move_pred, y_move)
    mse, mae, vacc = value_metrics(value_pred, y_value)

    maj_top1, _ = majority_class_baseline(y_move)
    uniform_legal = uniform_random_legal_baseline(split_data["states"])
    # side-to-move baseline is defined against the +-1 win/loss label, not the
    # continuous margin `y_value` used above for value_mse/mae/acc -- these are
    # two different fields in the split (DATASET.md SS8).
    value_baseline_acc = side_to_move_baseline(split_data["players"], split_data["values"])

    return FinalMetrics(
        split=split,
        model=entry["model"],
        encoding=enc,
        seed=entry["seed"],
        dedup=entry["dedup"],
        data_volume_games=entry["volume_games"],
        checkpoint=entry["path"],
        move_top1=top1,
        move_top1_ci_lo=ci_lo,
        move_top1_ci_hi=ci_hi,
        move_top3=top3,
        move_macro_f1=macro_f1,
        value_mse=mse,
        value_mae=mae,
        value_acc=vacc,
        value_baseline_acc=value_baseline_acc,
        majority_class_baseline_top1=maj_top1,
        uniform_random_legal_baseline_top1=uniform_legal,
    )


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Final held-out evaluation (task 4.0a).")
    ap.add_argument(
        "--split", required=True, choices=["val", "test"],
        help="No default, deliberately (SS3.A): typing 'test' must be a conscious act.",
    )
    ap.add_argument("--val-h5", default="data/processed/val.h5")
    ap.add_argument("--test-h5", default="data/processed/test.h5")
    ap.add_argument("--model-dir", required=True, help="directory of trainer.py --save-model-dir checkpoints")
    ap.add_argument("--max-rows", type=int, default=0, help="debug cap on rows read from the split (0=all)")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out-json/--out-csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_json = args.out_json or f"results/final_{args.split}_metrics.json"
    out_csv = args.out_csv or f"results/final_{args.split}_metrics.csv"

    for out_path in (out_json, out_csv):
        if os.path.exists(out_path) and not args.force:
            print(
                f"REFUSING TO OVERWRITE: {out_path} already exists. This split is meant to be read "
                f"exactly once (SS3.A). Pass --force if this is a deliberate second write.",
                file=sys.stderr,
            )
            sys.exit(1)
    if args.force and (os.path.exists(out_json) or os.path.exists(out_csv)):
        print(f"*** --force: overwriting an existing {args.split}-split result. This is a disclosed second read. ***")

    checkpoints = discover_checkpoints(args.model_dir)
    if not checkpoints:
        print(f"No checkpoints matching the naming convention found in {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(checkpoints)} checkpoint(s) in {args.model_dir}")

    split_path = args.val_h5 if args.split == "val" else args.test_h5
    print(f"Loading --split {args.split} from {split_path} ...")
    split_data = load_split(split_path, max_rows=args.max_rows)
    print(f"{args.split} rows: {len(split_data['moves']):,}")

    rows: List[FinalMetrics] = []
    for entry in checkpoints:
        print(f"Evaluating {entry['model']} (N={entry['encoding']}, seed={entry['seed']}) on {args.split}...")
        rows.append(evaluate_checkpoint(entry, args.split, split_data))

    _save_supervised_metrics(rows, out_csv, out_json)  # asdict-compatible dataclass rows

    print(f"\n=== Final {args.split}-split results ===")
    print(f"{'model':<8} {'enc':>3} {'seed':>4} {'top1':>7} {'top3':>7} {'macroF1':>8} {'val_mse':>9} {'val_acc':>8}")
    for r in sorted(rows, key=lambda r: (r.model, r.seed)):
        print(
            f"{r.model:<8} {r.encoding:>3d} {r.seed:>4d} {r.move_top1:7.4f} {r.move_top3:7.4f} "
            f"{r.move_macro_f1:8.4f} {r.value_mse:9.4f} {r.value_acc:8.4f}"
        )

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")
    print(f"SHA-256 ({out_json}): {_sha256(out_json)}")
    print(f"SHA-256 ({out_csv}): {_sha256(out_csv)}")


if __name__ == "__main__":
    main()
