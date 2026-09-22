"""Numba fixed-effects count evaluator for local structural-search smoke runs.

This backend deliberately supports only single-class fixed-effects Poisson/NB2
models. Random, grouped, heterogeneity, zero-inflated, and latent-class terms
remain on the JAX evaluator so selecting Numba cannot silently change their
meaning.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - exercised only without optional extra
    raise ImportError(
        "engine='numba' requires the optional dependency: "
        "pip install 'metacountregressor[numba]'"
    ) from exc


@njit(cache=True)
def _softplus(value):
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


@njit(cache=True)
def _logaddexp(left, right):
    if left >= right:
        return left + math.log1p(math.exp(right - left))
    return right + math.log1p(math.exp(left - right))


@njit(cache=True)
def _poisson_nll(params, design, response):
    total = 0.0
    n_rows, n_cols = design.shape
    for row in range(n_rows):
        eta = 0.0
        for col in range(n_cols):
            eta += design[row, col] * params[col]
        eta = min(25.0, max(-25.0, eta))
        mu = math.exp(eta)
        y = response[row]
        total += mu - y * eta + math.lgamma(y + 1.0)
    return total


@njit(cache=True)
def _nb2_nll(params, design, response):
    total = 0.0
    n_rows, n_cols = design.shape
    alpha = _softplus(params[n_cols])
    inv_alpha = 1.0 / alpha
    log_inv_alpha = math.log(inv_alpha)
    for row in range(n_rows):
        eta = 0.0
        for col in range(n_cols):
            eta += design[row, col] * params[col]
        eta = min(25.0, max(-25.0, eta))
        log_denom = _logaddexp(log_inv_alpha, eta)
        y = response[row]
        log_likelihood = (
            math.lgamma(y + inv_alpha)
            - math.lgamma(inv_alpha)
            - math.lgamma(y + 1.0)
            + inv_alpha * (log_inv_alpha - log_denom)
            + y * (eta - log_denom)
        )
        total -= log_likelihood
    return total


class NumbaFixedCountEvaluator:
    """Search evaluator for single-class fixed-effects count models."""

    def __init__(
        self,
        df,
        id_col,
        y_col,
        all_variables,
        allowed_roles,
        allowed_distributions,
        mode="single",
        group_id_col=None,
        offset_col=None,
        R=1,
        max_latent_classes=1,
        **_kwargs,
    ):
        if mode != "single":
            raise ValueError("engine='numba' supports mode='single' only")
        if int(max_latent_classes) != 1:
            raise ValueError("engine='numba' supports one latent class only")
        if group_id_col is not None:
            raise ValueError("engine='numba' does not support grouped effects")
        self.id_col = id_col
        self.y_col = y_col
        self.vars = list(all_variables)
        self.mode = mode
        self.group_id_col = group_id_col
        self.offset_col = offset_col
        self.R = 1
        self.max_latent_classes = 1
        self.allowed_distributions = allowed_distributions
        self.allowed_roles = {
            var: [role for role in allowed_roles.get(var, [0, 1]) if role in (0, 1)]
            or [0]
            for var in self.vars
        }
        self.mutual_exclusion = []
        self.cache = {}
        self.df_train = df.reset_index(drop=True)
        self._response = pd.to_numeric(
            self.df_train[y_col], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float64)
        if np.any(self._response < 0):
            raise ValueError("Numba count evaluator requires non-negative responses")
        self._design_cache = {}
        self._offset = None
        if offset_col is not None:
            self._offset = pd.to_numeric(
                self.df_train[offset_col], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=np.float64)

    def build_spec(self, decision):
        decision = np.asarray(decision, dtype=int).reshape(-1)
        n_vars = len(self.vars)
        if len(decision) < 2 * n_vars + 1:
            return None
        roles = decision[:n_vars]
        fixed = []
        for index, var in enumerate(self.vars):
            role = int(roles[index])
            if role not in self.allowed_roles.get(var, [0]):
                return None
            if role == 1:
                fixed.append(var)
        return {
            "fixed_terms": fixed,
            "rdm_terms": [],
            "rdm_cor_terms": [],
            "grouped_terms": [],
            "hetro_in_means": [],
            "hetro_in_variances": [],
            "zi_terms": [],
            "membership_terms": [],
            "class_membership": None,
            "dispersion": int(decision[2 * n_vars]) % 2,
            "latent_classes": 1,
            "group_id_col": None,
        }

    def _design(self, fixed_terms):
        key = tuple(fixed_terms)
        if key not in self._design_cache:
            columns = [np.ones(len(self.df_train), dtype=np.float64)]
            for column in fixed_terms:
                values = pd.to_numeric(
                    self.df_train[column], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=np.float64)
                values = np.where(np.isfinite(values), values, 0.0)
                columns.append(values)
            design = np.column_stack(columns)
            if self._offset is not None:
                design = design.copy()
                design[:, 0] += self._offset
            self._design_cache[key] = np.ascontiguousarray(design)
        return self._design_cache[key]

    def fitness(self, decision):
        decision = np.asarray(decision, dtype=int).reshape(-1)
        key = tuple(decision.tolist())
        if key in self.cache:
            return self.cache[key]
        spec = self.build_spec(decision)
        if spec is None:
            self.cache[key] = 1e12
            return self.cache[key]
        design = self._design(spec["fixed_terms"])
        n_rows = len(self._response)
        n_params = design.shape[1] + int(spec["dispersion"] == 1)
        initial = np.zeros(n_params, dtype=np.float64)
        mean_response = max(float(np.mean(self._response)), 1e-6)
        initial[0] = math.log(mean_response)
        if spec["dispersion"]:
            initial[-1] = 0.541324854612918
            objective = lambda params: float(_nb2_nll(params, design, self._response))
        else:
            objective = lambda params: float(_poisson_nll(params, design, self._response))
        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            options={"maxiter": 250, "ftol": 1e-9},
        )
        value = float(result.fun) if np.isfinite(result.fun) else 1e12
        if not result.success and value >= 1e12:
            value = 1e12
        bic = 2.0 * value + n_params * math.log(max(n_rows, 1))
        self.cache[key] = float(bic)
        return self.cache[key]
