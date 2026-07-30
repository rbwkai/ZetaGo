"""Acceptance gates for a freshly built train/val/test dataset.

Ten checks from ``Docs/EXECUTION_Phase0.md`` Sec. 5 that must all pass before any
model is trained on the dataset: they catch a repeat of the low-diversity /
resignation-collapse / komi-filter failure modes that made the corpus this
document replaces unusable. Three more (G11-G13) verify the Phase 1 task 1.2b
fix: test must be carved from the same generation run as train/val, not a
separately-generated run from a different regime (see ``Docs/DATASET.md`` §11).

G10 (val rows whose exact board position also occurs somewhere in train) is
reported, not enforced as pass/fail: this overlap is concentrated almost
entirely in the opening (measured >99% at ply<5, falling to <10% by ply 50+)
because independently-sampled self-play games converge on similar early
positions -- it is the expected structure of an opening book, not train/val
leakage. The only leakage-relevant check is G1 (zero shared *games*).

G2/G3 thresholds were revised after a second Phase 0 investigation: the original
<3x / >200,000 targets were calibrated against a corpus generated with
`policyInitAreaProp` diversity, which was then found to silently corrupt ~94%
of games (KataGo's `match` subcommand does not record policy-init stones in
the SGF -- see the config file's comments and Docs/EXECUTION_Phase0.md). The
safe replacement (chosenMoveTemperature-only diversity, oracle-verified
stable at 98.5% winner-agreement across scale) has a lower diversity ceiling
inherent to 7x7 Go's converging opening tree. G2/G3 here are calibrated as an
order-of-magnitude regression safety net against a repeat of that catastrophic
collapse (2,300 unique / 230x duplication), not a scientific target -- they
will not, and are not expected to, approach the original numbers.

This script does NOT replace `pytest -m oracle`: G1-G9/G10 check the built
train/val HDF5 for internal consistency, but the KataGo winner-agreement
oracle check is the check that actually caught the corruption above. Always
run `venv/bin/python -m pytest -q -m oracle` (or the full suite) after any
regeneration, before trusting a build_dataset run's output.

Run after ``python -m data.build_dataset``:
    venv/bin/python -m data.check_dataset
    venv/bin/python -m data.check_dataset --train other/train.h5 --val other/val.h5 --test other/test.h5
"""

import argparse
import sys

import h5py
import numpy as np


def _position_keys(h):
    return np.array([s.tobytes() for s in h["states"][:]], dtype=object)


def _median_game_length(h):
    game_id = h["game_id"][:]
    if len(game_id) == 0:
        return 0.0
    _, counts = np.unique(game_id, return_counts=True)
    return float(np.median(counts))


def _margin_std(h):
    m = h["margins"][:]
    m = m[~np.isnan(m)]
    return float(np.std(m)) if len(m) else float("nan")


def _side_to_move_baseline(h):
    # "predict a win iff White is to move": players +1 Black, -1 White; values
    # +1 win / -1 loss / 0 jigo from the side-to-move's perspective (see F2,
    # Docs/DATASET.md §8/§9).
    players = h["players"][:]
    values = h["values"][:]
    if len(values) == 0:
        return float("nan")
    pred = np.where(players == -1, 1, -1)
    return float(np.mean(pred == values))


def run_gates(train_path: str, val_path: str, test_path: str | None = None) -> bool:
    tr = h5py.File(train_path, "r")
    va = h5py.File(val_path, "r")
    te = h5py.File(test_path, "r") if test_path else None

    tk, vk = _position_keys(tr), _position_keys(va)
    ts, vs = set(tk.tolist()), set(vk.tolist())
    train_games = set(tr["game_id"][:].tolist())
    val_games = set(va["game_id"][:].tolist())
    game_overlap = len(train_games & val_games)
    dup_factor = len(tk) / len(ts) if ts else float("inf")
    mean_ply = len(tk) / len(train_games) if train_games else 0.0
    n = tr["states"].shape[-1]           # board size (7); pass index is BOARD*BOARD (49)
    pass_index = n * n
    pass_moves = int((tr["moves"][:] == pass_index).sum())
    margin_nan_pct = 100.0 * np.isnan(tr["margins"][:]).mean() if len(tr["margins"]) else 100.0
    n_dropped = int(tr.attrs.get("n_dropped", -1))
    n_skipped = int(tr.attrs.get("n_skipped_meta", -1))
    counts = np.bincount(tr["moves"][:], minlength=pass_index + 1)
    distinct_moves = int((counts > 0).sum())
    majority_pct = 100.0 * counts.max() / len(tr["moves"]) if len(tr["moves"]) else 0.0
    overlap_mask = np.isin(vk, list(ts)) if len(vk) else np.zeros(0, dtype=bool)
    val_seen_pct = 100.0 * overlap_mask.mean() if len(vk) else 0.0

    checks = [
        ("G1", "game overlap train/val", game_overlap, "== 0", game_overlap == 0),
        ("G2", "duplication factor", f"{dup_factor:.2f}x", "< 100x (regression net)", dup_factor < 100.0),
        ("G3", "unique train positions", f"{len(ts):,}", "> 20,000 (regression net)", len(ts) > 20_000),
        ("G4", "mean game length (ply)", f"{mean_ply:.1f}", "> 25", mean_ply > 25.0),
        ("G5", "pass moves present", pass_moves, "> 0", pass_moves > 0),
        ("G6", "margin NaN fraction", f"{margin_nan_pct:.1f}%", "< 20%", margin_nan_pct < 20.0),
        ("G7", "n_dropped (rules tripwire)", n_dropped, "== 0", n_dropped == 0),
        ("G8", "n_skipped_meta (komi filter)", n_skipped, "== 0", n_skipped == 0),
        ("G9", "distinct move labels", f"{distinct_moves}/{pass_index + 1}", "50/50", distinct_moves == pass_index + 1),
    ]

    print(f"train: {len(tk):,} rows, {len(train_games):,} games   |   "
          f"val: {len(vk):,} rows, {len(val_games):,} games\n")
    print(f"{'gate':<5}{'check':<32}{'value':<18}{'requirement':<28}{'status'}")
    all_pass = True
    for gid, name, value, req, ok in checks:
        status = "PASS" if ok else "FAIL"
        all_pass &= ok
        print(f"{gid:<5}{name:<32}{str(value):<18}{req:<28}{status}")

    print(f"{'G10':<5}{'val rows w/ position in train':<32}{f'{val_seen_pct:.1f}%':<18}"
          f"{'report only, not blocking':<28}INFO")

    if len(vk):
        vmn = va["move_no"][:]
        print("\nG10 detail — overlap rate by ply (expected to concentrate in the opening):")
        for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 10_000)):
            m = (vmn >= lo) & (vmn < hi)
            if m.sum():
                print(f"    ply {lo:>3}-{hi if hi < 10_000 else '+':<4}: "
                      f"{m.sum():>6,} val rows, overlap {100 * overlap_mask[m].mean():5.1f}%")

    if te is not None:
        test_games = set(te["game_id"][:].tolist())
        train_test_overlap = len(train_games & test_games)
        val_test_overlap = len(val_games & test_games)

        val_med_ply, test_med_ply = _median_game_length(va), _median_game_length(te)
        val_mstd, test_mstd = _margin_std(va), _margin_std(te)
        val_stm, test_stm = _side_to_move_baseline(va), _side_to_move_baseline(te)

        # 15% relative tolerance around val: generous enough for sampling noise at
        # test's smaller size, tight enough to catch a regime mismatch like the one
        # this gate was written to catch (test was 20 vs val's 27 median ply).
        def _close(a, b, tol=0.15):
            return abs(a - b) <= tol * max(abs(b), 1e-9)

        test_checks = [
            ("G11", "game overlap train/test", train_test_overlap, "== 0", train_test_overlap == 0),
            ("G12", "game overlap val/test", val_test_overlap, "== 0", val_test_overlap == 0),
            ("G13a", "test median game length vs val", f"{test_med_ply:.1f} (val {val_med_ply:.1f})",
             "within 15%", _close(test_med_ply, val_med_ply)),
            ("G13b", "test margin std vs val", f"{test_mstd:.2f} (val {val_mstd:.2f})",
             "within 15%", _close(test_mstd, val_mstd)),
            ("G13c", "test side-to-move baseline vs val", f"{test_stm:.4f} (val {val_stm:.4f})",
             "within 15%", _close(test_stm, val_stm)),
        ]
        print(f"\ntest: {len(te['moves']):,} rows, {len(test_games):,} games\n")
        for gid, name, value, req, ok in test_checks:
            status = "PASS" if ok else "FAIL"
            all_pass &= ok
            print(f"{gid:<5}{name:<32}{str(value):<24}{req:<28}{status}")

    print()
    print("ALL BLOCKING GATES PASSED — safe to proceed to training." if all_pass
          else "GATES FAILED — do not train on this dataset. See Docs/EXECUTION_Phase0.md.")
    tr.close()
    va.close()
    if te is not None:
        te.close()
    return all_pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="data/processed/train.h5")
    ap.add_argument("--val", default="data/processed/val.h5")
    ap.add_argument("--test", default="data/processed/test.h5")
    args = ap.parse_args()
    ok = run_gates(args.train, args.val, args.test)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
