"""Build the supervised dataset from KataGo self-play SGFs.

Every game is replayed through the ZetaGo engine (the single source of truth), so the
dataset's notion of board / legality / encoding is exactly what a model sees at
inference. Replaying also asserts that every KataGo move is legal under our rules: a
non-zero ``dropped`` count is the tripwire that the engine and KataGo rules diverged.

Layout (append-safe & resumable):
  * one shard ``data/processed/shards/<file>.h5`` per source ``.sgfs`` file. A new
    self-play run writes new files = new shards; existing shards are untouched and
    skipped on re-run unless ``--force``.
  * ``game_id`` and the train/val/test split are derived from a stable hash of
    (file, line), so adding data later never renumbers or re-splits existing games.
  * a final merge concatenates shards into ``train.h5`` / ``val.h5`` / ``test.h5``.

All three splits are carved from the same generation run (see EXECUTION_Phase1.md
task 1.2b): an earlier version sealed the test set from a separately-generated
5,000-game run, which turned out to come from a different regime (shorter games,
lower margin variance) and was not comparable to train/val. Splitting one run
three ways instead removes that mismatch by construction.

Run from the repo root:
    venv/bin/python "data/dataset generation/build_dataset.py"
"""

import argparse
import glob
import hashlib
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial

import h5py
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "environment"))

from engine import GoBoard                       # noqa: E402
from data.sgf_reader import parse_record         # noqa: E402

BOARD_SIZE = 7
KOMI = 9.5
PASS_INDEX = BOARD_SIZE * BOARD_SIZE              # 49
N_PLANES = 6
SPLIT_MOD = 20                                    # key % SPLIT_MOD: 0 -> val, 1 -> test, else train
SPLIT_NAMES = ("train", "val", "test")

SGF_DIR = "data/raw/sgf"
OUT_DIR = "data/processed"
SHARD_DIR = os.path.join(OUT_DIR, "shards")
CONFIG_PATH = "environment/katago/configs/selfplay7x7_match.cfg"
KATAGO_VERSION = "v1.16.5 (Eigen/CPU, AVX2)"
SCHEMA_VERSION = 1

_FIELDS_1D = {
    "moves": np.int16,
    "values": np.int8,
    "margins": np.float32,
    "players": np.int8,
    "game_id": np.uint32,
    "move_no": np.int16,
}


def _stable_key(basename, line_no):
    """Deterministic per-game key from (file, line) -> game_id and split."""
    return zlib.crc32(f"{basename}:{line_no}".encode()) & 0xFFFFFFFF


def _parse_margin(result):
    """'B+1.5' -> +1.5 (black ahead); 'W+0.5' -> -0.5; resign/timeout/jigo -> None."""
    if "+" not in result:
        return None
    side, _, val = result.partition("+")
    try:
        x = float(val)
    except ValueError:
        return None
    return x if side.upper() == "B" else -x


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------
def process_file(path, out_dir=SHARD_DIR, force=False):
    basename = os.path.basename(path)
    shard_path = os.path.join(out_dir, basename + ".h5")
    if os.path.exists(shard_path) and not force:
        with h5py.File(shard_path, "r") as h:
            return {"file": basename, "reused": True, "n_pos": int(h.attrs["n_pos"]),
                    "n_games": int(h.attrs["n_games"]), "n_dropped": int(h.attrs["n_dropped"]),
                    "n_skip_meta": int(h.attrs["n_skip_meta"]),
                    "n_val_pos": int(h.attrs["n_val_pos"]), "n_test_pos": int(h.attrs["n_test_pos"]),
                    "rules": h.attrs.get("rules", "")}

    states, moves, values, margins = [], [], [], []
    players, game_ids, move_nos, split = [], [], [], []
    n_games = n_dropped = n_skip_meta = 0
    rules_seen = ""

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            game = parse_record(line)
            if game.size != BOARD_SIZE or abs(game.komi - KOMI) > 1e-6:
                n_skip_meta += 1
                continue
            if not rules_seen and game.rules:
                rules_seen = game.rules

            key = _stable_key(basename, line_no)
            mod = key % SPLIT_MOD
            split_flag = 1 if mod == 0 else (2 if mod == 1 else 0)  # 0 train, 1 val, 2 test
            margin_black = _parse_margin(game.result)

            board = GoBoard(n=BOARD_SIZE, komi=KOMI)
            rows = []
            ok = True
            for color, point in game.moves:
                if color != board.current_player:
                    ok = False
                    break
                p = board.current_player
                state = board.get_tensor()
                if point is None:
                    move_idx = PASS_INDEX
                    legal = board.pass_move()
                else:
                    move_idx = point[0] * BOARD_SIZE + point[1]
                    legal = board.play_move(*point)
                if not legal:
                    ok = False
                    break
                if game.winner == 0:
                    value = 0
                else:
                    value = 1 if p == game.winner else -1
                margin_pov = (margin_black * p) if margin_black is not None else np.nan
                rows.append((state, move_idx, value, margin_pov, p, key, board.move_number - 1, split_flag))

            if not ok:
                n_dropped += 1
                continue
            n_games += 1
            for st, mv, vl, mg, pl, gid, mn, sf in rows:
                states.append(st); moves.append(mv); values.append(vl); margins.append(mg)
                players.append(pl); game_ids.append(gid); move_nos.append(mn); split.append(sf)

    n_pos = len(states)
    states_arr = (np.stack(states).astype(np.uint8) if n_pos
                  else np.zeros((0, N_PLANES, BOARD_SIZE, BOARD_SIZE), np.uint8))
    split_arr = np.asarray(split, np.uint8)
    n_val_pos = int(np.sum(split_arr == 1)) if n_pos else 0
    n_test_pos = int(np.sum(split_arr == 2)) if n_pos else 0

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(shard_path, "w") as h5:
        chunks = (min(1024, n_pos), N_PLANES, BOARD_SIZE, BOARD_SIZE) if n_pos else None
        h5.create_dataset("states", data=states_arr, chunks=chunks,
                          compression="gzip" if n_pos else None,
                          compression_opts=4 if n_pos else None)
        h5.create_dataset("moves", data=np.asarray(moves, np.int16))
        h5.create_dataset("values", data=np.asarray(values, np.int8))
        h5.create_dataset("margins", data=np.asarray(margins, np.float32))
        h5.create_dataset("players", data=np.asarray(players, np.int8))
        h5.create_dataset("game_id", data=np.asarray(game_ids, np.uint32))
        h5.create_dataset("move_no", data=np.asarray(move_nos, np.int16))
        h5.create_dataset("split", data=split_arr)
        h5.attrs["n_pos"] = n_pos
        h5.attrs["n_games"] = n_games
        h5.attrs["n_dropped"] = n_dropped
        h5.attrs["n_skip_meta"] = n_skip_meta
        h5.attrs["n_val_pos"] = n_val_pos
        h5.attrs["n_test_pos"] = n_test_pos
        h5.attrs["rules"] = rules_seen

    return {"file": basename, "reused": False, "n_pos": n_pos, "n_games": n_games,
            "n_dropped": n_dropped, "n_skip_meta": n_skip_meta, "n_val_pos": n_val_pos,
            "n_test_pos": n_test_pos, "rules": rules_seen}


# ---------------------------------------------------------------------------
# Merge shards -> train.h5 / val.h5
# ---------------------------------------------------------------------------
def _make_output(path, n):
    h = h5py.File(path, "w")
    chunks = (min(1024, n), N_PLANES, BOARD_SIZE, BOARD_SIZE) if n else None
    h.create_dataset("states", shape=(n, N_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8,
                     chunks=chunks, compression="gzip" if n else None,
                     compression_opts=4 if n else None)
    for name, dt in _FIELDS_1D.items():
        h.create_dataset(name, shape=(n,), dtype=dt)
    return h


def merge_shards(out_dir, meta, sgf_dir):
    """Concatenate every shard into train.h5 / val.h5 / test.h5.

    No position-level deduplication: the split is already correct at the game
    level (crc32(file:line) % SPLIT_MOD), and position overlap across distinct
    self-play games is normal opening-book structure, not leakage. Filtering it out
    (as an earlier revision did, matching a validation position's SHA1 against every
    train position) silently destroyed the validation set when the corpus was
    low-diversity, and gives each feature encoding a *different* val subset once
    re-applied per experiment -- breaking the identical-test-set requirement the
    Track A factorial depends on. Overlap is instead measured and reported (see
    EXECUTION_Phase0.md acceptance gates) rather than deleted.

    All three splits are carved from the same shard set (see module docstring and
    EXECUTION_Phase1.md task 1.2b): an earlier version sealed test from a
    separately-generated run that turned out to be from a different regime.

    Only merges shards whose source .sgfs file is still present in ``sgf_dir``: an
    orphan shard (source deleted/replaced) would otherwise silently blend two
    different generation runs into one dataset.
    """
    shard_paths = sorted(glob.glob(os.path.join(out_dir, "shards", "*.h5")))
    live = {os.path.basename(p) + ".h5" for p in glob.glob(os.path.join(sgf_dir, "*.sgfs"))}
    orphans = [p for p in shard_paths if os.path.basename(p) not in live]
    if orphans:
        raise SystemExit(
            f"{len(orphans)} orphan shard(s) have no matching .sgfs in {sgf_dir} and "
            "would silently blend two different corpora into one dataset:\n"
            + "\n".join("  " + os.path.basename(p) for p in orphans[:10])
            + "\nMove or delete them from data/processed/shards/ before merging "
              "(see EXECUTION_Phase0.md)."
        )

    n_val = sum(int(h5py.File(p).attrs["n_val_pos"]) for p in shard_paths)
    n_test = sum(int(h5py.File(p).attrs["n_test_pos"]) for p in shard_paths)
    n_train = sum(int(h5py.File(p).attrs["n_pos"]) for p in shard_paths) - n_val - n_test

    outputs = {
        "train": _make_output(os.path.join(out_dir, "train.h5"), n_train),
        "val": _make_output(os.path.join(out_dir, "val.h5"), n_val),
        "test": _make_output(os.path.join(out_dir, "test.h5"), n_test),
    }
    counts = {"train": n_train, "val": n_val, "test": n_test}
    offsets = {"train": 0, "val": 0, "test": 0}
    for sp in shard_paths:
        with h5py.File(sp, "r") as h:
            if int(h.attrs["n_pos"]) == 0:
                continue
            split_flag = h["split"][:]
            masks = {"train": split_flag == 0, "val": split_flag == 1, "test": split_flag == 2}
            for name in ("states", *_FIELDS_1D):
                data = h[name][:]
                for split_name, mask in masks.items():
                    n = int(mask.sum())
                    if n:
                        off = offsets[split_name]
                        outputs[split_name][name][off:off + n] = data[mask]
            for split_name, mask in masks.items():
                offsets[split_name] += int(mask.sum())

    for split_name, h in outputs.items():
        for k, v in meta.items():
            h.attrs[k] = v
        h.attrs["n_positions"] = counts[split_name]
        h.close()
    return n_train, n_val, n_test


# ---------------------------------------------------------------------------
# Metadata / dataset card
# ---------------------------------------------------------------------------
def _read_config(path):
    cfg = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    return cfg


def _sha256(path):
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(totals, rules):
    cfg = _read_config(CONFIG_PATH)
    net = cfg.get("nnModelFile", "")
    return {
        "schema_version": SCHEMA_VERSION,
        "board_size": BOARD_SIZE,
        "komi": KOMI,
        "rules": rules or "koPOSITIONALscoreAREAtaxNONEsui0",
        "ko_rule": "POSITIONAL",
        "scoring": "AREA",
        "suicide_legal": False,
        "max_visits": int(cfg.get("maxVisits", "0") or 0),
        "katago_version": KATAGO_VERSION,
        "net_name": os.path.basename(net),
        "net_sha256": _sha256(net),
        "n_games": totals["n_games"],
        "n_dropped": totals["n_dropped"],
        "n_skipped_meta": totals["n_skip_meta"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_mod": SPLIT_MOD,
    }


def write_dataset_card(out_dir, meta, n_train, n_val, n_test):
    n_pos = n_train + n_val + n_test
    card = f"""# ZetaGo 7x7 Self-Play Dataset

Supervised dataset of `(board_state_tensor, move_played, game_outcome)` triples for
move-prediction pre-training and unsupervised position analysis.

## Source
- **Engine:** KataGo {meta['katago_version']}, self-play via the `match` subcommand
- **Network:** `{meta['net_name']}`
  - sha256: `{meta['net_sha256']}`
- **Search:** maxVisits = {meta['max_visits']}, numSearchThreads = 1
- **Generated config:** `{CONFIG_PATH}`

## Rules (identical in the engine and in generation)
- Board size: **{meta['board_size']}x{meta['board_size']}**
- Komi: **{meta['komi']}** (fixed)
- Ko: **positional superko** (`{meta['ko_rule']}`)
- Scoring: **area / Tromp-Taylor** (`{meta['scoring']}`)
- Suicide: **illegal** (`multiStoneSuicideLegal = false`)
- KataGo rules string (from SGFs): `{meta['rules']}`

> Note: this is Tromp-Taylor area scoring **except** multi-stone suicide is forbidden,
> matched on both the engine and KataGo so the labels are self-consistent.

## Size
- Games: **{meta['n_games']:,}**
- Positions: **{n_pos:,}**  (train **{n_train:,}**, val **{n_val:,}**, test **{n_test:,}**)
- Games dropped for rule mismatch (tripwire, should be 0): **{meta['n_dropped']}**
- Games skipped (wrong size/komi): **{meta['n_skipped_meta']}**
- Created: {meta['created_utc']}

## Files
- `train.h5`, `val.h5`, `test.h5` — split by game via `crc32(file:line) % {meta['split_mod']}`
  (`== 0` val, `== 1` test, else train). All three splits are carved from the same
  generation run, so they share one regime; see `Docs/results/DATASET.md` §11.
- `shards/*.h5` — one shard per source `.sgfs` file (append-safe, resumable)
- `train.csv`, `val.csv`, `test.csv`, `sample.csv` — human-readable export (board as X/O/.
    rows joined by `/`, move as `row,col`); regenerate with `venv/bin/python "data/dataset generation/export_csv.py"`

## HDF5 schema
| dataset | shape | dtype | meaning |
|---|---|---|---|
| `states`  | [N, 6, 7, 7] | uint8   | planes: 0 current-player stones, 1 opponent stones, 2 side-to-move, 3 empty, 4 last move, 5 legal moves |
| `moves`   | [N]          | int16   | policy target: 0..48 board point (row*7+col), 49 = pass |
| `values`  | [N]          | int8    | game outcome from the side-to-move's perspective: +1 win, -1 loss, 0 jigo |
| `margins` | [N]          | float32 | final score margin from side-to-move's perspective; NaN if game ended by resignation |
| `players` | [N]          | int8    | side to move: +1 Black, -1 White |
| `game_id` | [N]          | uint32  | stable per-game id (crc32 of file:line) |
| `move_no` | [N]          | int16   | ply index within the game |

Root attrs on `train.h5`/`val.h5` record every field above plus `schema_version`.

## Reproduce
```bash
katago/bin/katago match -config {CONFIG_PATH} \\
    -sgf-output-dir data/raw/sgf -log-file data/raw/match.log
venv/bin/python "data/dataset generation/build_dataset.py"
```
"""
    with open(os.path.join(out_dir, "DATASET_CARD.md"), "w") as fh:
        fh.write(card)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the ZetaGo dataset from KataGo SGFs.")
    ap.add_argument("--sgf-dir", default=SGF_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--force", action="store_true", help="rebuild shards even if they exist")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files (debug)")
    args = ap.parse_args()

    shard_dir = os.path.join(args.out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.sgf_dir, "*.sgfs")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No .sgfs files in {args.sgf_dir}")
        return

    print(f"Processing {len(files)} file(s) with {args.workers} worker(s)...")
    worker = partial(process_file, out_dir=shard_dir, force=args.force)
    totals = {"n_pos": 0, "n_games": 0, "n_dropped": 0, "n_skip_meta": 0, "n_val_pos": 0, "n_test_pos": 0}
    rules = ""
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(worker, files), 1):
            for k in totals:
                totals[k] += res.get(k, 0)
            rules = rules or res.get("rules", "")
            tag = "reused" if res.get("reused") else "built"
            print(f"  [{i}/{len(files)}] {res['file']}: {res['n_pos']:>7,} pos, "
                  f"{res['n_games']:>5,} games, dropped={res['n_dropped']} ({tag})")

    print("Merging shards -> train.h5 / val.h5 / test.h5 ...")
    meta = build_metadata(totals, rules)
    n_train, n_val, n_test = merge_shards(args.out_dir, meta, args.sgf_dir)
    write_dataset_card(args.out_dir, meta, n_train, n_val, n_test)

    print("\n=== DONE ===")
    print(f"games:      {totals['n_games']:,}")
    print(f"positions:  {n_train + n_val + n_test:,}  "
          f"(train {n_train:,} / val {n_val:,} / test {n_test:,})")
    print(f"dropped (rule mismatch tripwire): {totals['n_dropped']}")
    print(f"skipped (wrong size/komi):        {totals['n_skip_meta']}")
    print(f"card: {os.path.join(args.out_dir, 'DATASET_CARD.md')}")


if __name__ == "__main__":
    main()
