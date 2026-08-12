"""
GA_CMF_AADT_JAX.py  (root-level re-export shim)
-----------------------------------------------
The canonical implementation lives in ``metacountregressor/GA_CMF_AADT_JAX.py``.
This shim makes the flat ``import GA_CMF_AADT_JAX`` return that exact module
object, so the package-form and flat form resolve to a single implementation and
cannot drift apart again.
"""
import sys as _sys

from metacountregressor import GA_CMF_AADT_JAX as _mod

_sys.modules[__name__] = _mod
