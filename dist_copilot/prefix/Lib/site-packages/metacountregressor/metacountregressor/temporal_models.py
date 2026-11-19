"""Compatibility shim for the canonical temporal model implementation."""

import importlib as _importlib
import os as _os

_THIS_FILE = _os.path.abspath(__file__)


def _canonical_module():
    module = _importlib.import_module("temporal_models")
    if _os.path.abspath(getattr(module, "__file__", "")) == _THIS_FILE:
        raise ImportError("The canonical temporal_models module could not be located")
    return module


def __getattr__(name):
    return getattr(_canonical_module(), name)


def __dir__():
    return dir(_canonical_module())