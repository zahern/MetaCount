"""
fitted_specifications.py  (root-level re-export shim)
------------------------------------------------------
The actual implementation lives in metacountregressor/fitted_specifications.py.
This shim allows ``import fitted_specifications`` from the project root.
"""
try:
    from metacountregressor.fitted_specifications import (
        describe_book_cmf_spec,
        describe_book_latent_class_spec,
        describe_book_nb_baseline_spec,
        list_book_specifications,
        load_book_cmf_spec,
        load_book_latent_class_spec,
        load_book_nb_baseline_spec,
    )
except ImportError:
    import os as _os, importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "fitted_specifications_impl",
        _os.path.join(_os.path.dirname(__file__), "metacountregressor", "fitted_specifications.py"),
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    describe_book_cmf_spec          = _mod.describe_book_cmf_spec
    describe_book_latent_class_spec = _mod.describe_book_latent_class_spec
    describe_book_nb_baseline_spec  = _mod.describe_book_nb_baseline_spec
    list_book_specifications        = _mod.list_book_specifications
    load_book_cmf_spec              = _mod.load_book_cmf_spec
    load_book_latent_class_spec     = _mod.load_book_latent_class_spec
    load_book_nb_baseline_spec      = _mod.load_book_nb_baseline_spec

__all__ = [
    "describe_book_cmf_spec",
    "describe_book_latent_class_spec",
    "describe_book_nb_baseline_spec",
    "list_book_specifications",
    "load_book_cmf_spec",
    "load_book_latent_class_spec",
    "load_book_nb_baseline_spec",
]
