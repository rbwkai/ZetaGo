# ZetaGo: 7x7 Go Game Engine

A clean, production-ready Go Game engine built with Python and NumPy. Designed for Machine Learning applications and neural network integration using PyTorch.

## Features

- **Pure Python/NumPy Implementation:** Clean, readable, object-oriented code without bitboards or C++ extensions
- **Complete Go Rules:**
  - Liberty calculation and stone capture mechanics (flood-fill algorithm)
  - Suicide rule (preventing immediate self-capture)
  - Ko rule (preventing immediate board state repetition)
  - Pass moves and consecutive pass game termination
  - Tromp-Taylor scoring with komi (9.5 points for White)
- **Terminal Game Play:** Simple ASCII-based interface to play against a random bot
- **ML-Ready:** Designed as a foundation for neural network agents and reinforcement learning

## Tech Stack

- **Python:** 3.10+
- **NumPy:** For efficient board representation and state management
- **PyTorch Integration:** Ready for future neural network agents

## Project Structure

```
ZetaGo/
├── go_board.py       # Core game logic and rules
├── play_terminal.py  # Terminal-based interactive gameplay
├── README.md         # This file
└── .gitignore        # Git ignore patterns
```

## Installation

1. **Clone or download the project:**
   ```bash
   cd ZetaGo
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy
   ```

4. **Optional - Install PyTorch for ML extensions:**
   ```bash
   pip install torch
   ```

## Usage

### Playing in the Terminal

Run the interactive game:
```bash
python play_terminal.py
```

**Game Instructions:**
- Enter coordinates as `row,col` (e.g., `3,4` for row 3, column 4)
- Rows and columns are indexed 0–6
- Enter `pass` to pass a turn
- The game ends when both players pass consecutively
- Final score is displayed using Tromp-Taylor scoring with 9.5 komi

**Example Session:**
```
  0 1 2 3 4 5 6
0 . . . . . . .
1 . . . . . . .
2 . . . . . . .
3 . . . . . . .
4 . . . . . . .
5 . . . . . . .
6 . . . . . . .

Black to move. Enter move (e.g., '3,3' or 'pass'): 3,3

  0 1 2 3 4 5 6
0 . . . . . . .
1 . . . . . . .
2 . . . . . . .
3 . . . X . . .
4 . . . . . . .
5 . . . . . . .
6 . . . . . . .

White to move. Enter move (e.g., '3,3' or 'pass'): pass
```

## Core Classes & Methods

### `GoBoard` (go_board.py)

**Initialization:**
```python
board = GoBoard()  # Creates a 7x7 empty board, Black starts
```

**Key Methods:**
- `place_stone(row: int, col: int) -> bool` - Place a stone; returns True if valid
- `pass_move() -> bool` - Pass the current turn
- `get_legal_moves() -> list[tuple]` - Return all legal coordinates on the board
- `is_game_over() -> bool` - Check if both players have passed
- `calculate_score() -> tuple[float, float]` - Returns (Black score, White score)
- `print_board()` - ASCII print of the current board state
- `get_board_state() -> np.ndarray` - Return the raw board array

**Board Representation:**
- `1` = Black stone
- `-1` = White stone
- `0` = Empty intersection

**State:**
- `self.current_player` - Track whose turn it is (`1` or `-1`)
- `self.passed_last_turn` - Track if the last move was a pass
- `self.move_history` - Full history of board states for Ko rule

### `GoEnv` Alias
`GoEnv` is an alias for `GoBoard`, supporting both naming conventions.

## Algorithm Details

### Liberty & Capture (BFS Flood-Fill)
When a stone is placed, the algorithm:
1. Identifies all connected stones of the same color (BFS)
2. Counts adjacent empty spaces (liberties)
3. Removes opponent groups with 0 liberties
4. Checks if the placed stone itself has liberties (suicide rule)

### Ko Rule
Prevents a move that would recreate the board state from exactly one turn prior. Simple but effective for 7x7 boards.

### Tromp-Taylor Scoring
- Count stones on board
- Count empty regions controlled by one color (BFS from empty spaces)
- White receives +9.5 komi (compensation for moving second)
- Higher score wins

## Design Decisions

1. **NumPy Arrays:** Efficient for ML/neural network integration
2. **Object-Oriented:** Clean encapsulation of game state and rules
3. **No Bitboards:** Prioritizes readability over C-level performance; sufficient for 7x7
4. **Flood-Fill Algorithm:** Standard, efficient approach for liberty calculation
5. **Full Move History:** Enables Ko rule detection and future ML rollout/tree search
6. **Type Hints:** Production-ready code with full type annotations

## Future Enhancements

- Alphanumeric coordinate system (A-G for columns)
- Configurable board sizes (9x9, 19x19)
- Minimax and Monte Carlo Tree Search (MCTS) bots
- PyTorch neural network player integration
- Game state serialization/deserialization
- Elo rating system for bot battles

## Contributing

This is a student project for learning ML and Go game engines. Contributions are welcome!

## License

Open source for educational purposes.

## References

- [American Go Association Rules](https://www.usgo.org/what-go)
- [Tromp-Taylor Scoring](https://en.wikipedia.org/wiki/Scoring_in_Go#Tromp-Taylor_Scoring)
- [Go on Wikipedia](https://en.wikipedia.org/wiki/Go_(game))
