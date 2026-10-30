"""SuperMimic world-model learning with an Isaac Lab manager-based environment.

Prefer this checkout's pinned Isaac Lab source over editable installations of
another checkout. Third-party binary dependencies still come from the environment.
"""

from pathlib import Path
import sys

_source = Path(__file__).resolve().parents[1] / "IsaacLab/source"
if _source.is_dir():
    for _package in sorted(_source.iterdir(), reverse=True):
        if _package.is_dir() and (_package / _package.name).is_dir():
            sys.path.insert(0, str(_package))
