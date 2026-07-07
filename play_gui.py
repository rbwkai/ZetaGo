"""Launch the ZetaGo GUI.

    venv/bin/python play_gui.py              # play (Human / Random / KataGo)
    venv/bin/python play_gui.py --selftest   # headless smoke test, then exit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="ZetaGo GUI")
    ap.add_argument("--size", type=int, default=7, help="board size (default 7)")
    ap.add_argument("--selftest", action="store_true", help="run a headless smoke test and exit")
    args = ap.parse_args()

    if args.selftest:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from gui.app import GoGUI

    gui = GoGUI(board_size=args.size)
    if args.selftest:
        mode, moves = gui.selftest()
        print(f"selftest OK: ended in mode={mode!r} after {moves} moves")
        return
    gui.run()


if __name__ == "__main__":
    main()
