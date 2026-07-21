# KataGo Integration Guide

This project can run with either the built-in random bot or KataGo through the GTP
(Go Text Protocol) interface. The integration is split across two Python files:

- [play_terminal.py](play_terminal.py) handles the terminal UI, setup prompts, and
  the main game loop.
- [katago_gtp.py](katago_gtp.py) starts KataGo as a subprocess and translates
  between the local board representation and GTP commands.

## What Gets Bundled

The repo includes a KataGo layout under `katago/`:

- `katago/bin/katago` is the engine executable.
- `katago/bin/default_gtp.cfg` is the default GTP config.
- `katago/models/*.bin.gz` contains the neural net model file.

When you choose KataGo in the terminal game, the launcher tries to discover these
files automatically.

## High-Level Flow

```mermaid
flowchart TD
    A[Start play_terminal.py] --> B[Choose random bot or KataGo]
    B -->|KataGo| C[discover_katago_defaults()]
    C --> D[setup_katago()]
    D --> E[KataGoGTP.start()]
    E --> F[Spawn KataGo subprocess]
    F --> G[Send boardsize, komi, clear_board]
    G --> H[Game loop]
    H --> I[Human move or bot move]
    I --> J[Sync move with KataGo via play]
    J --> K[Ask KataGo for genmove]
    K --> H
```

## How The Integration Works

### 1. The terminal game starts the session

`play_terminal.py` is the entry point. It asks whether you want a random bot or
KataGo. If KataGo is selected, it calls `setup_katago()`.

### 2. The code discovers bundled paths

`discover_katago_defaults()` looks for the shipped files in these locations:

- `katago/bin/katago`
- `katago/bin/default_gtp.cfg`
- `katago/models/*.bin.gz`

If it finds a matching file, that path is used as the default prompt value. You
can still override any of them manually.

### 3. The KataGo subprocess is created

`KataGoGTP.start()` builds a command like this:

```bash
./katago/bin/katago gtp -model ./katago/models/<MODEL>.bin.gz -config ./katago/bin/default_gtp.cfg
```

The executable is launched with `subprocess.Popen`, and the code keeps its
stdin/stdout/stderr open so it can talk to the engine while the game runs.

### 4. The engine is initialized

After the process starts, the wrapper sends these GTP setup commands:

- `boardsize 7`
- `komi 9.5`
- `clear_board`

This puts KataGo in the same board state as the local `GoBoard` object.

### 5. Moves stay synchronized in both directions

The game loop keeps the local board and KataGo aligned:

- When the human plays, `play_terminal.py` updates `GoBoard` first.
- Then it calls `katago_client.play(...)` so KataGo learns about that move.
- When it is KataGo's turn, `katago_client.genmove(...)` asks KataGo for a move.
- The returned GTP coordinate is converted back into local row/col values and
  applied to `GoBoard`.

## Coordinate Conversion

The local board uses zero-based row/column coordinates with the origin at the top
left.

KataGo uses GTP coordinates such as `D4`.

`katago_gtp.py` translates between the two formats:

- `to_gtp_vertex(row, col, size)` converts local coordinates into a GTP move.
- `from_gtp_vertex(vertex, size)` converts a KataGo move back into row/col.

This is why the 7x7 board still works correctly even though the engine speaks in
GTP notation.

## Error Handling

The wrapper raises `KataGoError` when something goes wrong, such as:

- the executable path is invalid,
- the model or config file cannot be read,
- KataGo returns a malformed response,
- or the engine times out / exits unexpectedly.

The terminal game catches that error and falls back to the random bot if KataGo
cannot be started.

## Shutdown

When the game ends, `play_terminal.py` closes the KataGo client in a `finally`
block. `KataGoGTP.close()` sends `quit` to the process, waits briefly, and kills
it if needed.

## In Short

The integration is intentionally small:

1. `play_terminal.py` manages the user experience and game loop.
2. `katago_gtp.py` manages KataGo as an external process.
3. GTP commands keep the engine and local board synchronized.
4. The bundled `katago/bin` and `katago/models` files provide the defaults.
