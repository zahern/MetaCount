"""
help_guide.py  (root-level re-export shim)
------------------------------------------
The actual implementation lives in metacountregressor/help_guide.py.
This shim allows ``import help_guide`` from the project root.
"""
try:
    from metacountregressor.help_guide import get_help, get_templates
except ImportError:
    import os as _os, importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "help_guide_impl",
        _os.path.join(_os.path.dirname(__file__), "metacountregressor", "help_guide.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    get_help = _mod.get_help
    get_templates = _mod.get_templates

__all__ = ["get_help", "get_templates"]
