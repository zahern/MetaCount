"""Version marker for the nested compatibility package.

Resolves the version from the canonical installed package first, then from a
flat sibling ``_version.py`` (source-tree layout), and finally falls back to a
literal so the import never breaks.
"""
try:
    from metacountregressor._version import __version__
except ImportError:
    try:
        from _version import __version__
    except ImportError:
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
