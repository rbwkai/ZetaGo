"""Make the repository root importable so `import engine` / `import data` work
regardless of how pytest is invoked."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
