"""Compatibility import; implementation lives in tasks.tracking.mdp.actions."""

import sys
from .tracking.mdp import actions as _implementation

sys.modules[__name__] = _implementation
