"""Convenience wrapper for running the sample Spotify data collection CLI."""

from __future__ import annotations

from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from app.sample_data_cli import main


if __name__ == "__main__":
    main()
