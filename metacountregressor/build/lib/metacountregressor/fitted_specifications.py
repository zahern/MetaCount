"""Compatibility shim forwarding every attribute to the canonical implementation.

The canonical code for this module lives in ``metacountregressor/fitted_specifications.py``
(the top-level installed package).  This shim keeps the historical
``metacountregressor.metacountregressor.fitted_specifications`` import path working and
guarantees both paths resolve to one implementation, in every layout
(wheel install, editable install, source checkout).
"""
import importlib as _importlib
import os as _os

_THIS_FILE = _os.path.abspath(__file__)
_FOUND = []


def _canonical_module():
    if _FOUND:
        return _FOUND[0]
    try:
        module = _importlib.import_module("metacountregressor.fitted_specifications")
    except ImportError:
        module = None
    if module is not None and _os.path.abspath(
        getattr(module, "__file__", "")
    ) != _THIS_FILE:
        _FOUND.append(module)
        return module
    # Source-tree layout: the package directory itself sits on sys.path, so
    # the canonical implementation is importable as a flat top-level module.
    module = _importlib.import_module("fitted_specifications")
    _FOUND.append(module)
    return module


def __getattr__(name):
    return getattr(_canonical_module(), name)


def __dir__():
    return dir(_canonical_module())
