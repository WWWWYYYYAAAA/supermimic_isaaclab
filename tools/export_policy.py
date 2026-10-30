"""Compatibility CLI for the standalone policy exporter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.export_policy import main

if __name__ == "__main__":
    raise SystemExit(main())
