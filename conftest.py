"""Make the repository root importable so `import engine` / `import data` work
regardless of how pytest is invoked."""

import os
import sys

root_dir = os.path.dirname(__file__)
sys.path.insert(0, root_dir)
# Add environment directory so `engine` and `gui` are importable
sys.path.insert(0, os.path.join(root_dir, "environment"))
