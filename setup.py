"""Legacy shim — all metadata now lives in pyproject.toml.

Kept only for tooling that still expects a setup.py to exist.
"""
from setuptools import setup

setup()
