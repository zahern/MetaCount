"""
main_hpc.py  (root-level re-export shim)
----------------------------------------
The canonical implementation lives in ``metacountregressor/main_hpc.py``.
This shim makes the flat ``import main_hpc`` return that exact module object, so
the package-form and flat form resolve to a single implementation and cannot
drift apart again.
"""
import sys as _sys

from metacountregressor import main_hpc as _mod

_sys.modules[__name__] = _mod
