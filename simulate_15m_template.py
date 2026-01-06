"""Backwards-compatible entrypoint for the 15m template."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    runpy.run_path(str(repo_root / "scripts" / "simulate_15m_template.py"), run_name="__main__")
