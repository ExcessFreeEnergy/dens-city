#!/usr/bin/env python3
"""
dens-city: 3D Interactive Raylib Molecular Visualizer.

Usage:
    uv run python scripts/run_interactive.py --materials argon
    uv run python scripts/run_interactive.py --materials water benzene 5cb
    uv run python scripts/run_interactive.py --materials all
"""

import sys
from pathlib import Path

# Ensure src/ is on PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dens_city.ui.cli import main

if __name__ == "__main__":
    # If --interactive not explicitly supplied, ensure interactive mode is activated
    args = sys.argv[1:]
    if "--interactive" not in args and "-i" not in args:
        args = ["--interactive"] + args
    main(args)
