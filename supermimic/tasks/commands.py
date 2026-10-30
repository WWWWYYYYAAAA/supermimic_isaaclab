"""Compatibility import; implementation lives in tasks.tracking.mdp.commands."""

import sys
from .tracking.mdp import commands as _implementation

sys.modules[__name__] = _implementation
