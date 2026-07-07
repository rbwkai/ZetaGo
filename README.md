# ZetaGo — 7×7 Go: engine, GUI, and KataGo dataset

ZetaGo is a 7×7 Go project with three parts:

1. **A verified bitboard game engine** (`engine/`) — integer-bitboard board representation,
   Zobrist **positional superko**, capture/suicide rules, and **Tromp-Taylor area scoring**,
   developed test-first and cross-checked against KataGo.
2. **A Pygame GUI** (`gui/`, `play_gui.py`) — play against another human, a random bot, or
   **KataGo**, with a wood board, stone graphics, sounds, and live score.
3. **A KataGo self-play dataset pipeline** (`data/`) — generate tens of thousands of 7×7
   self-play games and extract `(board_state_tensor, move, outcome)` triples to HDF5 (+ CSV).

The engine is the single source of truth: the GUI plays through it and the dataset is built
by replaying KataGo's games through it, so rules and encoding are consistent everywhere.

## Rules (identical in the engine, the GUI's KataGo opponent, and data generation)

| Rule | Value |
|---|---|
| Board / komi | 7×7, komi **9.5** (fixed) |
| Ko | **positional superko** (no board position may repeat) |
| Scoring | **area / Tromp-Taylor** (no dead-stone removal) |
| Suicide | **illegal** (single- and multi-stone) |

> This is Tromp-Taylor area scoring **except** multi-stone suicide is forbidden — pinned the
> same way on both the engine and KataGo so training labels stay self-consistent.

## Project structure

```
engine/            Bitboard Go engine (single source of truth)
  board.py           GoBoard: bitboards, captures, suicide, superko, play/score
  masks.py           bit geometry: neighbours, edge-safe shifts, flood-fill
  zobrist.py         Zobrist hashing (positional superko)
  scoring.py         Tromp-Taylor area scoring
  encode.py          NumPy tensors / arrays / move<->index helpers
tests/             pytest suite (rules, captures, ko/superko, scoring, encoding, oracle)
gui/               Pygame interface (theme, assets, board_view, widgets, app)
play_gui.py        GUI entry point
play_terminal.py   ASCII terminal play (Human vs Random / KataGo)
katago_gtp.py      KataGo GTP subprocess client (used by GUI + terminal)
data/              Dataset generation + extraction
  build_dataset.py   replay SGFs -> HDF5 shards + train/val + DATASET_CARD.md
  export_csv.py      HDF5 -> human-readable CSV
  sgf_reader.py      KataGo .sgfs parser
katago/
  bin/               KataGo binary + bundled configs   (gitignored)
  models/            neural nets                        (gitignored)
  configs/           selfplay7x7_match.cfg, gui_gtp.cfg (tracked)
requirements.txt   numpy, h5py, sgfmill, tqdm, pytest, pygame
```

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

All commands below use `venv/bin/python` so they run inside that environment.

## Play (GUI)

```bash
venv/bin/python play_gui.py
```

Pick an opponent (Human / Random / **KataGo**) and your colour, then **Start**. Click an
intersection to play; use **Pass**, **Undo**, **Resign**, or **Menu**. A hover ghost stone
shows where you'll play, the last move is ringed, and the side panel shows turn, captures,
and a live area-score estimate. The KataGo opponent runs in a background thread so the UI
never freezes.

Headless smoke test (no display needed):

```bash
venv/bin/python play_gui.py --selftest
```

## Run the tests

```bash
venv/bin/python -m pytest -q                 # full suite
venv/bin/python -m pytest -q -m "not oracle" # rules/scoring/encoding only
venv/bin/python -m pytest -q -m oracle       # cross-check vs KataGo self-play data
```

The **oracle** tests replay real KataGo self-play games through the engine and assert every
move is legal and that counted games agree on the winner — the strongest correctness gate.

## Generate the dataset

KataGo (CPU/Eigen build) and a small net are expected under `katago/`. Then:

```bash
# 1. Self-play game generation (writes .sgfs files)
katago/bin/katago match -config katago/configs/selfplay7x7_match.cfg \
    -sgf-output-dir data/raw/sgf -log-file data/raw/match.log

# 2. Extract (state, move, outcome) triples to HDF5 (+ shards, train/val, dataset card)
venv/bin/python -m data.build_dataset

# 3. Optional: human-readable CSV export
venv/bin/python -m data.export_csv
```

Outputs land in `data/processed/` (`train.h5`, `val.h5`, `shards/`, `train.csv`, `val.csv`,
`sample.csv`) and are documented in `data/processed/DATASET_CARD.md`. The extractor is
**append-safe and resumable**: re-running after generating more games only processes the new
files. A non-zero "dropped" count is a tripwire that the engine and KataGo rules have diverged.

### HDF5 schema

| dataset | shape | dtype | meaning |
|---|---|---|---|
| `states`  | [N, 6, 7, 7] | uint8   | planes: current-player stones, opponent stones, side-to-move, empty, last move, legal moves |
| `moves`   | [N] | int16   | policy target 0..48 (row*7+col), 49 = pass |
| `values`  | [N] | int8    | outcome from side-to-move POV: +1 win / -1 loss / 0 jigo |
| `margins` | [N] | float32 | score margin from side-to-move POV; NaN if resigned |
| `players` | [N] | int8    | side to move (+1 Black / -1 White) |
| `game_id` | [N] | uint32  | stable per-game id |
| `move_no` | [N] | int16   | ply within the game |

## KataGo setup

A CPU (Eigen) KataGo build and a neural net go under `katago/bin/` and `katago/models/`
(both gitignored). A small net such as `g170e-b10c128` keeps self-play and GUI responses fast
on CPU; download it from the [KataGo releases](https://github.com/lightvector/KataGo/releases)
or [katagotraining.org](https://katagotraining.org/). See `KATAGO_INTEGRATION.md` for the GTP
client details. The committed configs in `katago/configs/` pin the rules above.

## Engine API (quick reference)

```python
from engine import GoBoard, tromp_taylor_area, move_to_index

b = GoBoard()              # 7x7, Black to move (GoBoard(n) for other sizes)
b.play_move(3, 3)          # -> True/False (False = illegal, no mutation)
b.pass_move()
b.get_legal_moves()        # [(row, col), ...]
b.get_final_score()        # (black, white, "Black"/"White"/"Tie")
b.get_tensor()             # (6, 7, 7) uint8 ML tensor
b.is_game_over()
```

## License

MIT — see `LICENSE`.

## References

- [KataGo](https://github.com/lightvector/KataGo) and [its rules](https://lightvector.github.io/KataGo/rules.html)
- [Tromp-Taylor rules](https://tromp.github.io/go.html)
