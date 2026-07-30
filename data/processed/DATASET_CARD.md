# ZetaGo 7x7 Self-Play Dataset

Supervised dataset of `(board_state_tensor, move_played, game_outcome)` triples for
move-prediction pre-training and unsupervised position analysis.

## Source
- **Engine:** KataGo v1.16.5 (Eigen/CPU, AVX2), self-play via the `match` subcommand
- **Network:** `g170e-b10c128-s1141046784-d204142634.txt.gz`
  - sha256: `3d8a24697ba25fe4da39af4c2b6bd405907b0ad8295322f5a550fa2d8fe4a2f4`
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
- Games: **120,000**
- Positions: **3,677,092**  (train **3,492,258**, val **184,834**)
- Games dropped for rule mismatch (tripwire, should be 0): **0**
- Games skipped (wrong size/komi): **0**
- Created: 2026-07-30T12:16:58Z

## Diversity, correctness verification & position overlap

Measured via `venv/bin/python -m data.check_dataset` (all 9 blocking acceptance gates in
`Docs/EXECUTION_Phase0.md` pass) and confirmed by the KataGo oracle cross-check
(`pytest -m oracle`, both tests pass — every replayed move is legal and >=98% of counted
games agree with the engine's own Tromp-Taylor scoring on the winner).

**This config is the second iteration of the diversity fix.** An earlier revision used
`policyInitAreaProp = 0.35` to break the "every game opens identically" problem
(originally: only ~2,300 unique positions across the whole corpus). That mechanism was
found to be unsafe: KataGo's `match` subcommand does not record policy-initialised stones
in the output SGF at all (no setup properties, no synthetic moves — only a `startTurnIdx=N`
note in a free-text comment), so replaying "from empty board" silently reconstructs the
wrong position for any game where that count is nonzero. Measured on a since-discarded
50,000-game run at `policyInitAreaProp = 0.35`: **94.3% of games had `startTurnIdx > 0`**
(mean 16, median 12 hidden plies) — this corrupted the extracted `(state, move)` label for
nearly every position, not just the ~4.5% of games where it happened to flip the recorded
winner (which is what first surfaced the bug, via the oracle test). See
`katago/configs/selfplay7x7_match.cfg` and `Docs/EXECUTION_Phase0.md` for the full
investigation.

The current config instead raises `chosenMoveTemperatureEarly`/`chosenMoveTemperatureHalflife`
with `policyInitAreaProp = 0` (every move is genuinely searched and appears in the SGF).
This has a lower diversity ceiling than the unsafe mechanism appeared to offer — 7x7 Go's
opening tree converges hard even under added sampling temperature — so the corpus was
grown to 120,000 games (still ~68 min locally) to compensate with more absolute unique
positions rather than a higher per-game yield.

- Unique train positions: **58,007** (train has 3,492,258 rows -> duplication factor
  **60.20x**). For comparison, the discarded `policyInitAreaProp` run's *real* diversity
  (measured only on its `startTurnIdx == 0` games, the ones not affected by the bug) was
  ~2,300 unique positions — this is a genuine, oracle-verified ~25x improvement.
- Mean game length: **30.6 ply**; pass moves present: **317,709**; margin NaN: **0.0%**
  (no resignations — every game reaches a natural double-pass or move-limit ending).
- Game overlap between train/val: **0** (split is by game, not by position — see Files below).
- Val rows whose exact board position also occurs in train: **90.7%** — this is *not*
  train/val leakage (zero shared games); it concentrates in the opening and stays high
  even into the midgame at this corpus size, because 120,000 independently-sampled games
  are drawn from the same finite, converging early-position space:

  | ply     | 0–5   | 5–10  | 10–15 | 15–20 | 20–30 | 30–50 | 50+   |
  |---------|-------|-------|-------|-------|-------|-------|-------|
  | overlap | 100.0%| 100.0%| 99.9% | 99.8% | 90.2% | 59.5% | 50.5% |

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
