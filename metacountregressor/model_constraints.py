"""
model_constraints.py  (root-level re-export shim)
--------------------------------------------------
The actual implementation lives in metacountregressor/model_constraints.py
so that ``from metacountregressor import ModelConstraints`` works reliably
in all environments (installed package, editable install, source tree).

This shim allows ``import model_constraints`` to also work when the
project root is in sys.path (e.g. during development).
"""
try:
    from metacountregressor.model_constraints import ModelConstraints
except ImportError:
    # Fallback: direct relative import when metacountregressor package
    # is not yet on sys.path (rare edge case in bare source trees).
    import os as _os, importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "model_constraints_impl",
        _os.path.join(_os.path.dirname(__file__), "metacountregressor", "model_constraints.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ModelConstraints = _mod.ModelConstraints

__all__ = ["ModelConstraints"]
