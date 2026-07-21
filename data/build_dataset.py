"""Build the supervised dataset from KataGo self-play SGFs.

Every game is replayed through the ZetaGo engine (the single source of truth), so the
dataset's notion of board / legality / encoding is exactly what a model sees at
inference. Replaying also asserts that every KataGo move is legal under our rules: a
non-zero ``dropped`` count is the tripwire that the engine and KataGo rules diverged.

Layout (append-safe & resumable):
  * one shard ``data/processed/shards/<file>.h5`` per source ``.sgfs`` file. A new
    self-play run writes new files = new shards; existing shards are untouched and
    skipped on re-run unless ``--force``.
  * ``game_id`` and the train/val split are derived from a stable hash of (file, line),
    so adding data later never renumbers or re-splits existing games.
  * a final merge concatenates shards into ``train.h5`` / ``val.h5``.

Run from the repo root:
    venv/bin/python -m data.build_dataset
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
VAL_MOD = 20                                      # ~5% of games go to validation

SGF_DIR = "data/raw/sgf"
OUT_DIR = "data/processed"
SHARD_DIR = os.path.join(OUT_DIR, "shards")
CONFIG_PATH = "katago/configs/selfplay7x7_match.cfg"
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
                    "n_val_pos": int(h.attrs["n_val_pos"]), "rules": h.attrs.get("rules", "")}

    states, moves, values, margins = [], [], [], []
    players, game_ids, move_nos, is_val = [], [], [], []
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
            val_flag = 1 if (key % VAL_MOD == 0) else 0
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
                rows.append((state, move_idx, value, margin_pov, p, key, board.move_number - 1, val_flag))

            if not ok:
                n_dropped += 1
                continue
            n_games += 1
            for st, mv, vl, mg, pl, gid, mn, vf in rows:
                states.append(st); moves.append(mv); values.append(vl); margins.append(mg)
                players.append(pl); game_ids.append(gid); move_nos.append(mn); is_val.append(vf)

    n_pos = len(states)
    states_arr = (np.stack(states).astype(np.uint8) if n_pos
                  else np.zeros((0, N_PLANES, BOARD_SIZE, BOARD_SIZE), np.uint8))
    n_val_pos = int(np.sum(is_val)) if n_pos else 0

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
        h5.create_dataset("is_val", data=np.asarray(is_val, np.uint8))
        h5.attrs["n_pos"] = n_pos
        h5.attrs["n_games"] = n_games
        h5.attrs["n_dropped"] = n_dropped
        h5.attrs["n_skip_meta"] = n_skip_meta
        h5.attrs["n_val_pos"] = n_val_pos
        h5.attrs["rules"] = rules_seen

    return {"file": basename, "reused": False, "n_pos": n_pos, "n_games": n_games,
            "n_dropped": n_dropped, "n_skip_meta": n_skip_meta, "n_val_pos": n_val_pos,
            "rules": rules_seen}


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


def merge_shards(out_dir, meta):
    shard_paths = sorted(glob.glob(os.path.join(out_dir, "shards", "*.h5")))
    n_val = sum(int(h5py.File(p).attrs["n_val_pos"]) for p in shard_paths)
    n_pos = sum(int(h5py.File(p).attrs["n_pos"]) for p in shard_paths)
    n_train = n_pos - n_val

    train = _make_output(os.path.join(out_dir, "train.h5"), n_train)
    val = _make_output(os.path.join(out_dir, "val.h5"), n_val)
    ti = vi = 0
    for sp in shard_paths:
        with h5py.File(sp, "r") as h:
            if int(h.attrs["n_pos"]) == 0:
                continue
            isv = h["is_val"][:].astype(bool)
            ntr, nva = int((~isv).sum()), int(isv.sum())
            for name in ("states", *_FIELDS_1D):
                data = h[name][:]
                if ntr:
                    train[name][ti:ti + ntr] = data[~isv]
                if nva:
                    val[name][vi:vi + nva] = data[isv]
            ti += ntr
            vi += nva

    for h, n in ((train, n_train), (val, n_val)):
        for k, v in meta.items():
            h.attrs[k] = v
        h.attrs["n_positions"] = n
        h.close()
    return n_train, n_val


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
        "val_split_mod": VAL_MOD,
    }


def write_dataset_card(out_dir, meta, n_train, n_val):
    n_pos = n_train + n_val
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
- Positions: **{n_pos:,}**  (train **{n_train:,}**, val **{n_val:,}**)
- Games dropped for rule mismatch (tripwire, should be 0): **{meta['n_dropped']}**
- Games skipped (wrong size/komi): **{meta['n_skipped_meta']}**
- Created: {meta['created_utc']}

## Files
- `train.h5`, `val.h5` (split by game via `crc32(file:line) % {meta['val_split_mod']}`)
- `shards/*.h5` — one shard per source `.sgfs` file (append-safe, resumable)
- `train.csv`, `val.csv`, `sample.csv` — human-readable export (board as X/O/. rows joined
  by `/`, move as `row,col`); regenerate with `venv/bin/python -m data.export_csv`

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
venv/bin/python -m data.build_dataset
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
    totals = {"n_pos": 0, "n_games": 0, "n_dropped": 0, "n_skip_meta": 0, "n_val_pos": 0}
    rules = ""
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(worker, files), 1):
            for k in totals:
                totals[k] += res.get(k, 0)
            rules = rules or res.get("rules", "")
            tag = "reused" if res.get("reused") else "built"
            print(f"  [{i}/{len(files)}] {res['file']}: {res['n_pos']:>7,} pos, "
                  f"{res['n_games']:>5,} games, dropped={res['n_dropped']} ({tag})")

    print("Merging shards -> train.h5 / val.h5 ...")
    meta = build_metadata(totals, rules)
    n_train, n_val = merge_shards(args.out_dir, meta)
    write_dataset_card(args.out_dir, meta, n_train, n_val)

    print("\n=== DONE ===")
    print(f"games:      {totals['n_games']:,}")
    print(f"positions:  {n_train + n_val:,}  (train {n_train:,} / val {n_val:,})")
    print(f"dropped (rule mismatch tripwire): {totals['n_dropped']}")
    print(f"skipped (wrong size/komi):        {totals['n_skip_meta']}")
    print(f"card: {os.path.join(args.out_dir, 'DATASET_CARD.md')}")


if __name__ == "__main__":
    main()
