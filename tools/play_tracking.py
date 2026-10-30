"""Compatibility CLI for the standalone SuperMimic live tracking viewer."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.play import main, preview_model, set_preview_state, ROOT

__all__ = ["main", "preview_model", "set_preview_state", "ROOT"]

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
