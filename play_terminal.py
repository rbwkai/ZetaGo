"""
Terminal-based Go Game: Human vs Random Bot

This script allows a human player to play a 7x7 Go game against a bot that
plays random legal moves. The game displays the board in ASCII format and
tracks the score.

Usage:
    python play_terminal.py

Controls:
    Enter coordinates as "row,col" (e.g., "3,4") or "pass" to pass.
    
Author: ML Go Engine Team
Date: June 2026
"""

import random
from go_board import GoBoard


def clear_screen() -> None:
    """Clear terminal screen (cross-platform)."""
    import os
    os.system("cls" if os.name == "nt" else "clear")


def print_game_state(board: GoBoard, human_player: int) -> None:
    """
    Print the current game state with instructions.
    
    Args:
        board: Current GoBoard instance
        human_player: 1 for Black, -1 for White
    """
    clear_screen()
    print("=" * 40)
    print("    7x7 GO GAME: Human vs Random Bot")
    print("=" * 40)
    print()
    print(board)
    print()
    
    player_symbol = "X (Black)" if human_player == GoBoard.BLACK else "O (White)"
    current_symbol = "X (Black)" if board.current_player == GoBoard.BLACK else "O (White)"
    
    print(f"You are playing as: {player_symbol}")
    print(f"Current player:     {current_symbol}")
    print()


def get_human_move(board: GoBoard) -> bool:
    """
    Prompt the human player for a move.
    
    Args:
        board: Current GoBoard instance
        
    Returns:
        True if move was successful, False if invalid
    """
    while True:
        user_input = input("Your move (row,col or 'pass'): ").strip().lower()
        
        if user_input == "pass":
            board.pass_move()
            print("You passed.")
            return True
        
        try:
            row, col = map(int, user_input.split(","))
            
            # Validate range
            if not (0 <= row < GoBoard.BOARD_SIZE and 0 <= col < GoBoard.BOARD_SIZE):
                print(f"Invalid coordinates! Board is 0-{GoBoard.BOARD_SIZE-1}.")
                continue
            
            # Attempt move
            if board.play_move(row, col):
                print(f"Played at ({row}, {col}).")
                return True
            else:
                print("Illegal move! Try again.")
                legal_moves = board.get_legal_moves()
                if legal_moves:
                    print(f"Some legal moves: {legal_moves[:5]}")
                continue
                
        except (ValueError, IndexError):
            print("Invalid format! Use 'row,col' (e.g., '3,4') or 'pass'.")
            continue


def get_bot_move(board: GoBoard) -> None:
    """
    Execute a random legal move for the bot.
    
    Args:
        board: Current GoBoard instance
    """
    legal_moves = board.get_legal_moves()
    
    if legal_moves:
        # Random move strategy: 30% pass, 70% play random legal move
        if random.random() < 0.30 and len(legal_moves) > 3:
            board.pass_move()
            print("Bot passed.")
        else:
            move = random.choice(legal_moves)
            row, col = move
            board.play_move(row, col)
            print(f"Bot played at ({row}, {col}).")
    else:
        # No legal moves available; must pass
        board.pass_move()
        print("Bot passed (no legal moves).")


def play_game() -> None:
    """
    Main game loop: Human vs Random Bot.
    """
    print()
    print("=" * 40)
    print("    7x7 GO GAME: Human vs Random Bot")
    print("=" * 40)
    print()
    print("Do you want to play as Black (goes first)? (y/n)")
    
    choice = input("Your choice: ").strip().lower()
    human_player = GoBoard.BLACK if choice == 'y' else GoBoard.WHITE
    bot_player = -human_player
    
    print()
    board = GoBoard()
    
    print(f"You are: {'Black (X)' if human_player == GoBoard.BLACK else 'White (O)'}")
    print(f"Bot is:  {'White (O)' if human_player == GoBoard.BLACK else 'Black (X)'}")
    print()
    input("Press Enter to start...")
    
    # ==================== GAME LOOP ====================
    
    while not board.is_game_over():
        print_game_state(board, human_player)
        
        if board.current_player == human_player:
            # Human's turn
            if not get_human_move(board):
                continue
        else:
            # Bot's turn
            input("Press Enter for bot's move...")
            get_bot_move(board)
        
        print()
        input("Press Enter to continue...")
    
    # ==================== GAME OVER ====================
    
    print_game_state(board, human_player)
    
    black_score, white_score, winner = board.get_final_score()
    
    print("=" * 40)
    print("              GAME OVER!")
    print("=" * 40)
    print()
    print(f"Black (X) Score: {black_score:.1f}")
    print(f"White (O) Score: {white_score:.1f}")
    print(f"Komi applied:    {GoBoard.KOMI}")
    print()
    print(f"WINNER: {winner}")
    
    if winner != "Tie":
        human_status = "You WIN!" if (winner == "Black" and human_player == GoBoard.BLACK) or (winner == "White" and human_player == GoBoard.WHITE) else "You LOSE!"
        print(human_status)
    
    print()


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
