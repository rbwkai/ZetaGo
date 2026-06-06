"""
Pure Python/NumPy 7x7 Go Game Engine

This module implements a clean, object-oriented Go board with complete rule enforcement:
- Liberty calculation via flood-fill
- Capture detection and removal
- Suicide rule (with capture exception)
- Ko rule (prevent immediate board repetition)
- Tromp-Taylor scoring with komi

Author: ML Go Engine Team
Date: June 2026
"""

from typing import Tuple, List, Set, Optional
from collections import deque
import numpy as np
import copy


class GoBoard:
    """
    A 7x7 Go game board with complete rule enforcement.
    
    Board representation:
    - 1 = Black stone
    - -1 = White stone
    - 0 = Empty intersection
    
    Player encoding:
    - 1 = Black (moves first)
    - -1 = White
    """
    
    # Board constants
    BOARD_SIZE: int = 7
    BLACK: int = 1
    WHITE: int = -1
    EMPTY: int = 0
    KOMI: float = 9.5  # Tromp-Taylor komi (advantage to White)
    
    def __init__(self) -> None:
        """Initialize a new 7x7 Go board with Black to move."""
        self.board: np.ndarray = np.zeros(
            (self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8
        )
        self.current_player: int = self.BLACK
        self.move_history: List[np.ndarray] = []  # Track board states for Ko rule
        self.consecutive_passes: int = 0  # Track consecutive passes for game end
        self.last_move: Optional[Tuple[int, int]] = None
        
    def copy(self) -> 'GoBoard':
        """Return a deep copy of the board state."""
        new_board = GoBoard()
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.move_history = [b.copy() for b in self.move_history]
        new_board.consecutive_passes = self.consecutive_passes
        new_board.last_move = self.last_move
        return new_board
    
    # ==================== LIBERTIES & GROUPS ====================
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (up, down, left, right)."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE:
                neighbors.append((nr, nc))
        return neighbors
    
    def _find_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """
        Find all stones connected to the stone at (row, col) using BFS.
        
        Args:
            row, col: Position of a stone
            
        Returns:
            Set of (row, col) tuples representing the connected group
        """
        if self.board[row, col] == self.EMPTY:
            return set()
        
        stone_color = self.board[row, col]
        group = set()
        queue = deque([(row, col)])
        visited = {(row, col)}
        
        while queue:
            r, c = queue.popleft()
            group.add((r, c))
            
            for nr, nc in self._get_neighbors(r, c):
                if (nr, nc) not in visited and self.board[nr, nc] == stone_color:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return group
    
    def _get_liberties(self, group: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """
        Calculate liberties (empty adjacent spaces) for a group of stones.
        
        Args:
            group: Set of (row, col) tuples forming a connected group
            
        Returns:
            Set of empty (row, col) tuples adjacent to the group
        """
        liberties = set()
        for row, col in group:
            for nr, nc in self._get_neighbors(row, col):
                if self.board[nr, nc] == self.EMPTY:
                    liberties.add((nr, nc))
        return liberties
    
    def _get_group_liberties_count(self, row: int, col: int) -> int:
        """
        Get the number of liberties for the group containing stone at (row, col).
        
        Args:
            row, col: Position of a stone
            
        Returns:
            Number of liberties for the group
        """
        group = self._find_group(row, col)
        if not group:
            return 0
        liberties = self._get_liberties(group)
        return len(liberties)
    
    # ==================== MOVE VALIDATION & EXECUTION ====================
    
    def _capture_opponent_groups(self, row: int, col: int) -> bool:
        """
        Capture any opponent groups with zero liberties adjacent to (row, col).
        
        Args:
            row, col: Position where a stone was just placed
            
        Returns:
            True if any captures were made, False otherwise
        """
        opponent = -self.current_player
        captured = False
        
        for nr, nc in self._get_neighbors(row, col):
            if self.board[nr, nc] == opponent:
                opponent_group = self._find_group(nr, nc)
                opponent_liberties = self._get_liberties(opponent_group)
                
                if len(opponent_liberties) == 0:
                    # Capture this group
                    for gr, gc in opponent_group:
                        self.board[gr, gc] = self.EMPTY
                    captured = True
        
        return captured
    
    def _is_valid_move(self, row: int, col: int) -> bool:
        """
        Check if a move at (row, col) is valid.
        
        Validation rules:
        1. Intersection must be empty
        2. Move must not violate Ko rule
        3. Move must not be suicide (place stone with 0 liberties), UNLESS it captures
        
        Args:
            row, col: Position to check
            
        Returns:
            True if move is legal, False otherwise
        """
        # Rule 1: Must be empty
        if self.board[row, col] != self.EMPTY:
            return False
        
        # Simulate the move to check remaining rules
        board_backup = self.board.copy()
        self.board[row, col] = self.current_player
        
        # Check if this move captures anything
        captured = self._capture_opponent_groups(row, col)
        
        # Rule 3: Suicide rule - if no captures, must have liberties
        my_group = self._find_group(row, col)
        my_liberties = self._get_liberties(my_group)
        
        if len(my_liberties) == 0 and not captured:
            # Invalid: suicide move (no captures, no liberties)
            self.board = board_backup
            return False
        
        # Rule 2: Ko rule - board state must not match immediate previous state
        if len(self.move_history) > 0:
            previous_board = self.move_history[-1]
            if np.array_equal(self.board, previous_board):
                # Invalid: Ko rule violation
                self.board = board_backup
                return False
        
        # Valid move; restore board for actual execution
        self.board = board_backup
        return True
    
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Get list of all legal move positions on the board.
        
        Returns:
            List of (row, col) tuples for all empty intersections that satisfy rules
        """
        legal_moves = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row, col] == self.EMPTY and self._is_valid_move(row, col):
                    legal_moves.append((row, col))
        return legal_moves
    
    def play_move(self, row: int, col: int) -> bool:
        """
        Execute a move at (row, col).
        
        Args:
            row, col: Board position
            
        Returns:
            True if move was legal and executed, False if move was illegal
        """
        if not self._is_valid_move(row, col):
            return False
        
        # Save current board state for Ko rule
        self.move_history.append(self.board.copy())
        
        # Place the stone
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        
        # Capture opponent groups
        self._capture_opponent_groups(row, col)
        
        # Reset consecutive passes and switch player
        self.consecutive_passes = 0
        self.current_player = -self.current_player
        
        return True
    
    def pass_move(self) -> bool:
        """
        Execute a pass move.
        
        Returns:
            True if move was executed, False if game is already over
        """
        self.consecutive_passes += 1
        self.current_player = -self.current_player
        return True
    
    def is_game_over(self) -> bool:
        """
        Check if the game has ended (two consecutive passes).
        
        Returns:
            True if game is over, False otherwise
        """
        return self.consecutive_passes >= 2
    
    # ==================== SCORING ====================
    
    def _get_territory(self) -> Tuple[float, float]:
        """
        Calculate territory using Tromp-Taylor scoring.
        
        Tromp-Taylor: Score = (Black stones + Black territory) - (White stones + White territory)
        where territory is empty points that are surrounded by only one color.
        
        Returns:
            Tuple of (black_score, white_score) before komi
        """
        black_score = 0
        white_score = 0
        visited = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=bool)
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.board[row, col] != self.EMPTY or visited[row, col]:
                    continue
                
                # BFS to find connected empty region
                territory = set()
                boundary_colors = set()
                queue = deque([(row, col)])
                visited[row, col] = True
                
                while queue:
                    r, c = queue.popleft()
                    territory.add((r, c))
                    
                    for nr, nc in self._get_neighbors(r, c):
                        if visited[nr, nc]:
                            continue
                        
                        if self.board[nr, nc] == self.EMPTY:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                        else:
                            # Found boundary stone
                            boundary_colors.add(self.board[nr, nc])
                
                # Award territory based on boundary
                if len(boundary_colors) == 1:
                    # Territory is surrounded by only one color
                    color = list(boundary_colors)[0]
                    territory_size = len(territory)
                    if color == self.BLACK:
                        black_score += territory_size
                    else:
                        white_score += territory_size
        
        # Add stones to score
        black_stones = int(np.sum(self.board == self.BLACK))
        white_stones = int(np.sum(self.board == self.WHITE))
        
        black_score += black_stones
        white_score += white_stones
        
        return float(black_score), float(white_score)
    
    def get_final_score(self) -> Tuple[float, float, str]:
        """
        Calculate final Tromp-Taylor score with komi.
        
        Returns:
            Tuple of (black_score, white_score, winner_string)
            where winner_string is "Black", "White", or "Tie"
        """
        black_score, white_score = self._get_territory()
        
        # Apply komi to White
        white_score += self.KOMI
        
        if black_score > white_score:
            winner = "Black"
        elif white_score > black_score:
            winner = "White"
        else:
            winner = "Tie"
        
        return black_score, white_score, winner
    
    # ==================== BOARD DISPLAY ====================
    
    def __str__(self) -> str:
        """Return ASCII representation of the board."""
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(self.BOARD_SIZE)))
        
        for row in range(self.BOARD_SIZE):
            line = f"{row} "
            for col in range(self.BOARD_SIZE):
                cell = self.board[row, col]
                if cell == self.BLACK:
                    line += "X "
                elif cell == self.WHITE:
                    line += "O "
                else:
                    line += ". "
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_state(self) -> Tuple[np.ndarray, int]:
        """
        Get current board state and current player.
        
        Returns:
            Tuple of (board_array, current_player)
        """
        return self.board.copy(), self.current_player
