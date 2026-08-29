"""Compatibility re-export for the package's Bayesian compiler."""

from ..bayesian_model import (
    BayesianModel,
    BayesianModelError,
    build_bayesian_model,
    resolve_search_spec,
)

__all__ = [
    "BayesianModel",
    "BayesianModelError",
    "build_bayesian_model",
    "resolve_search_spec",
]
