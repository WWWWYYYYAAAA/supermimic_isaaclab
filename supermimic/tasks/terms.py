"""Compatibility import; implementation lives in tasks.tracking.mdp.terms."""

import sys
from .tracking.mdp import terms as _implementation

sys.modules[__name__] = _implementation
