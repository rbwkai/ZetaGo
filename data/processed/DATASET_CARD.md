# ZetaGo 7x7 Self-Play Dataset

Supervised dataset of `(board_state_tensor, move_played, game_outcome)` triples for
move-prediction pre-training and unsupervised position analysis.

## Source
- **Engine:** KataGo v1.16.5 (Eigen/CPU, AVX2), self-play via the `match` subcommand
- **Network:** `b18c384nbt-optimisticv13-s5971M.bin.gz`
  - sha256: `4714cddd467d29fba4cb52964f5d324fc208c0528953694a7bbe21a75b332d5d`
- **Search:** maxVisits = 16, numSearchThreads = 1
- **Generated config:** `katago/configs/selfplay7x7_match.cfg`

## Rules (identical in the engine and in generation)
- Board size: **7x7**
- Komi: **9.5** (fixed)
- Ko: **positional superko** (`POSITIONAL`)
- Scoring: **area / Tromp-Taylor** (`AREA`)
- Suicide: **illegal** (`multiStoneSuicideLegal = false`)
- KataGo rules string (from SGFs): `koPOSITIONALscoreAREAtaxNONEsui0`

> Note: this is Tromp-Taylor area scoring **except** multi-stone suicide is forbidden,
> matched on both the engine and KataGo so the labels are self-consistent.

## Size
- Games: **50,000**
- Positions: **557,248**  (train **530,192**, val **27,056**)
- Games dropped for rule mismatch (tripwire, should be 0): **0**
- Games skipped (wrong size/komi): **0**
- Created: 2026-07-21T13:03:54Z

## Files
- `train.h5`, `val.h5` (split by game via `crc32(file:line) % 20`)
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
katago/bin/katago match -config katago/configs/selfplay7x7_match.cfg \
    -sgf-output-dir data/raw/sgf -log-file data/raw/match.log
venv/bin/python -m data.build_dataset
```
