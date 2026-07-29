#!/usr/bin/env python3
"""Entry point for modular OOP supervised training."""

import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from training.supervised.trainer import main


if __name__ == "__main__":
    main()
