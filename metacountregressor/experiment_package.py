"""
experiment_package.py  (root-level re-export shim)
--------------------------------------------------
The canonical implementation lives in ``metacountregressor/experiment_package.py``.
This shim makes the flat ``import experiment_package`` return that exact module
object, so the package-form (``from metacountregressor.experiment_package import
ExperimentBuilder``) and the flat form resolve to a single implementation and can
never drift apart again.
"""
import sys as _sys

from metacountregressor import experiment_package as _mod

# Replace this shim module with the canonical package module so that every
# name (public or private) is served from one place.
_sys.modules[__name__] = _mod
