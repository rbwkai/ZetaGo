"""
Terminal-based Go Game: Human vs Bot (Random or KataGo)

This script allows a human player to play a 7x7 Go game against either a
random bot or KataGo via GTP. The game displays the board in ASCII format
and tracks the score.

Usage:
    python play_terminal.py

Controls:
    Enter coordinates as "row,col" (e.g., "3,4") or "pass" to pass.
    
Author: ML Go Engine Team
Date: June 2026
"""

import random
from pathlib import Path
from go_board import GoBoard
from katago_gtp import KataGoConfig, KataGoError, KataGoGTP


PROJECT_ROOT = Path(__file__).resolve().parent


def _first_existing_path(candidates: list[Path]) -> str:
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def _resolve_project_path(path_text: str) -> str:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _discover_model_path() -> str:
    model_dirs = [PROJECT_ROOT / "models", PROJECT_ROOT / "katago" / "models"]
    for model_dir in model_dirs:
        if model_dir.exists():
            models = sorted(model_dir.glob("*.bin.gz"))
            if models:
                return str(models[0])
    return ""


def discover_katago_defaults() -> tuple[str, str, str]:
    executable = _first_existing_path([
        PROJECT_ROOT / "katago" / "bin" / "katago",
        PROJECT_ROOT / "katago" / "katago",
        PROJECT_ROOT / "bin" / "katago",
        PROJECT_ROOT / "katago",
        PROJECT_ROOT / "katago.exe",
    ])
    config_path = _first_existing_path([
        PROJECT_ROOT / "katago" / "bin" / "default_gtp.cfg",
        PROJECT_ROOT / "katago" / "bin" / "gtp_human5k_example.cfg",
        PROJECT_ROOT / "katago" / "bin" / "gtp_human9d_search_example.cfg",
        PROJECT_ROOT / "katago" / "bin" / "match_example.cfg",
        PROJECT_ROOT / "katago" / "bin" / "analysis_example.cfg",
    ])
    model_path = _discover_model_path()
    return executable, config_path, model_path


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
    print("      7x7 GO GAME: Human vs Bot")
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


def get_katago_move(board: GoBoard, katago: KataGoGTP) -> bool:
    """
    Ask KataGo for a move and apply it to the board.

    Args:
        board: Current GoBoard instance
        katago: Running KataGo GTP client

    Returns:
        True if move was applied, False if KataGo resigned
    """
    row, col, raw = katago.genmove(board.current_player)

    if raw == "resign":
        print("KataGo resigned.")
        return False

    if raw == "pass":
        board.pass_move()
        print("KataGo passed.")
        return True

    if row is None or col is None:
        raise RuntimeError(f"Unexpected KataGo move: {raw}")

    if board.play_move(row, col):
        print(f"KataGo played at ({row}, {col}) [{raw.upper()}].")
        return True

    raise RuntimeError(
        f"KataGo suggested illegal move for local rules: ({row}, {col}) [{raw.upper()}]"
    )


def setup_katago() -> KataGoGTP:
    """Prompt user for KataGo command settings and start engine."""
    default_executable, default_config, default_model = discover_katago_defaults()

    print()
    print("KataGo setup: using autodetected project paths")
    print()

    exe_prompt = default_executable or "./katago"
    config_prompt = default_config or "./katago/bin/default_gtp.cfg"
    model_prompt = default_model or ""

    exe_input = input(f"KataGo executable path [{exe_prompt}]: ").strip() or exe_prompt
    model_input = input(f"Neural net model path (.bin.gz) [{model_prompt or 'required'}]: ").strip() or model_prompt
    if not model_input:
        raise KataGoError("Model path is required")

    cfg_input = input(f"Config path (.cfg) [{config_prompt}]: ").strip()
    exe = _resolve_project_path(exe_input)
    model = _resolve_project_path(model_input)
    cfg_path = _resolve_project_path(cfg_input or config_prompt) if (cfg_input or config_prompt) else None

    client = KataGoGTP(
        KataGoConfig(
            executable=exe,
            model_path=model,
            config_path=cfg_path,
            board_size=GoBoard.BOARD_SIZE,
            komi=GoBoard.KOMI,
        )
    )
    client.start()
    return client


def play_game() -> None:
    """
    Main game loop: Human vs Random Bot.
    """
    print()
    print("=" * 40)
    print("    7x7 GO GAME")
    print("=" * 40)
    print()
    print("Choose opponent:")
    print("  1) Random bot")
    print("  2) KataGo engine (GTP)")
    opponent_choice = input("Your choice [1/2]: ").strip()
    use_katago = opponent_choice == "2"

    katago_client = None
    if use_katago:
        try:
            katago_client = setup_katago()
            print("KataGo engine started successfully.")
        except KataGoError as e:
            print(f"Could not start KataGo: {e}")
            print("Falling back to random bot.")
            use_katago = False

    print()
    print("Do you want to play as Black (goes first)? (y/n)")
    
    choice = input("Your choice: ").strip().lower()
    human_player = GoBoard.BLACK if choice == 'y' else GoBoard.WHITE
    
    print()
    board = GoBoard()
    
    print(f"You are: {'Black (X)' if human_player == GoBoard.BLACK else 'White (O)'}")
    bot_label = "KataGo" if use_katago else "Random Bot"
    print(f"Opponent: {bot_label}")
    print(f"Bot is:   {'White (O)' if human_player == GoBoard.BLACK else 'Black (X)'}")
    print()
    input("Press Enter to start...")
    
    try:
        # ==================== GAME LOOP ====================
        while not board.is_game_over():
            print_game_state(board, human_player)

            if board.current_player == human_player:
                # Human's turn
                if not get_human_move(board):
                    continue
                if use_katago and katago_client is not None:
                    if board.last_move is None:
                        katago_client.play(human_player, None, None)
                    else:
                        row, col = board.last_move
                        katago_client.play(human_player, row, col)
            else:
                # Bot's turn
                input("Press Enter for bot's move...")
                if use_katago and katago_client is not None:
                    move_applied = get_katago_move(board, katago_client)
                    if not move_applied:
                        break
                else:
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
    finally:
        if katago_client is not None:
            katago_client.close()


if __name__ == "__main__":
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
