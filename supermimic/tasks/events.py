"""Compatibility import; implementation lives in tasks.tracking.mdp.events."""

import sys
from .tracking.mdp import events as _implementation

sys.modules[__name__] = _implementation
