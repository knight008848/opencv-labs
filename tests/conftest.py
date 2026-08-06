"""Make the project root importable so ``from src...`` works from any cwd.

Without this, a bare ``pytest`` run outside the repo root fails with
``ModuleNotFoundError: No module named 'src'`` (only ``python -m pytest``
from the root happened to work by putting the cwd on sys.path).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
