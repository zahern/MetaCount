from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
except ImportError:  # pragma: no cover
    jax = None
    jnp = None
    jit = None
    value_and_grad = None


def _require_jax() -> None:
    if jax is None or jnp is None:
        raise ImportError("JAX is required for metacountregressor.survival_models")


def _normal_logpdf(z):
    return -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * z * z


def _normal_logsurv(z):
    return jnp.log(0.5 * jax.scipy.special.erfc(z / jnp.sqrt(2.0)) + 1e-12)


class JAXRandomEffectsAFTFitter:
    """
    JAX-native AFT fitter with optional correlated random parameter simulation.

    Model fitting is JAX autodiff + L-BFGS-B. Random-effects capability is
    represented by post-fit coefficient draws using the inverse-Hessian
    approximation with optional correlated blocks.
    """

    family = "lognormal"

    def __init__(
        self,
        random_terms: Optional[List[str]] = None,
        correlated_random_terms: Optional[List[str]] = None,
        n_draws: int = 200,
        seed: int = 42,
        maxiter: int = 2000,
        ftol: float = 1e-12,
        gtol: float = 1e-7,
        n_restarts: int = 3,
    ):
        self.random_terms = list(random_terms or [])
        self.correlated_random_terms = list(correlated_random_terms or [])
        self.n_draws = int(n_draws)
        self.seed = int(seed)
        self.maxiter = int(maxiter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.n_restarts = int(n_restarts)

        self.feature_cols_: List[str] = []
        self.duration_col_: str = ""
        self.event_col_: Optional[str] = None

        self.params_ = pd.Series(dtype=float)
        self.summary_ = pd.DataFrame()
        self.variance_matrix_ = pd.DataFrame()

        self.log_likelihood_ = float("nan")
        self.converged_ = False
        self.result_ = None

        self.random_mean_ = pd.Series(dtype=float)
        self.random_cov_ = pd.DataFrame()

    def _prepare_design(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: Optional[str],
        feature_cols: List[str],
    ):
        cols = [duration_col] + feature_cols
        if event_col is not None:
            cols.append(event_col)
        data = df[cols].dropna().copy()

        X = data[feature_cols].to_numpy(dtype=float)
        X = np.column_stack([np.ones((X.shape[0], 1)), X])
        y = data[duration_col].to_numpy(dtype=float)
        event = np.ones_like(y) if event_col is None else data[event_col].to_numpy(dtype=float)
        return data, X, y, event

    @staticmethod
    def _softplus(x):
        return jax.nn.softplus(x) + 1e-8

    def _unpack(self, params):
        beta = params[:-1]
        sigma = self._softplus(params[-1])
        return beta, sigma

    def _negloglik_lognormal(self, params, X, y, event):
        beta, sigma = self._unpack(params)
        mu = X @ beta
        logy = jnp.log(jnp.clip(y, 1e-12, None))
        z = (logy - mu) / sigma

        ll_event = _normal_logpdf(z) - jnp.log(jnp.clip(y, 1e-12, None)) - jnp.log(sigma)
        ll_cens = _normal_logsurv(z)
        ll = event * ll_event + (1.0 - event) * ll_cens
        return -jnp.sum(ll)

    def _negloglik_weibull(self, params, X, y, event):
        beta, sigma = self._unpack(params)
        mu = X @ beta
        logy = jnp.log(jnp.clip(y, 1e-12, None))
        z = (logy - mu) / sigma

        ll_event = z - jnp.log(sigma) - jnp.exp(z)
        ll_cens = -jnp.exp(z)
        ll = event * ll_event + (1.0 - event) * ll_cens
        return -jnp.sum(ll)

    def _negloglik_loglogistic(self, params, X, y, event):
        beta, sigma = self._unpack(params)
        mu = X @ beta
        logy = jnp.log(jnp.clip(y, 1e-12, None))
        z = (logy - mu) / sigma
        ez = jnp.exp(z)

        ll_event = z - jnp.log(sigma) - 2.0 * jnp.log1p(ez)
        ll_cens = -jnp.log1p(ez)
        ll = event * ll_event + (1.0 - event) * ll_cens
        return -jnp.sum(ll)

    def _negloglik(self, params, X, y, event):
        if self.family == "lognormal":
            return self._negloglik_lognormal(params, X, y, event)
        if self.family == "weibull":
            return self._negloglik_weibull(params, X, y, event)
        if self.family == "loglogistic":
            return self._negloglik_loglogistic(params, X, y, event)
        raise ValueError(f"Unsupported family '{self.family}'")

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        _require_jax()

        feature_cols = list(feature_cols or [
            col for col in df.columns if col not in {duration_col, event_col}
        ])
        data, X_np, y_np, event_np = self._prepare_design(df, duration_col, event_col, feature_cols)

        X = jnp.asarray(X_np)
        y = jnp.asarray(y_np)
        event = jnp.asarray(event_np)

        # Smarter initialisation: intercept = log(mean(y)), log_scale starts at
        # softplus_inverse(1.0) ≈ 0.54 so sigma starts near 1.0.
        _log_mean_y = float(np.log(np.clip(float(np.mean(y_np)), 1e-8, None)))
        init = np.zeros(X_np.shape[1] + 1, dtype=float)
        init[0] = _log_mean_y          # intercept
        init[-1] = 0.541               # log_scale reparam → sigma ≈ 1

        obj = jit(lambda p: self._negloglik(p, X, y, event))
        obj_vg = jit(value_and_grad(obj))

        def _fun(params_np):
            val, grad = obj_vg(jnp.asarray(params_np))
            return float(val), np.asarray(grad, dtype=float)

        _opts = {"maxiter": self.maxiter, "ftol": self.ftol, "gtol": self.gtol, "disp": False}

        def _run_opt(x0):
            return minimize(
                fun=lambda p: _fun(p)[0],
                x0=x0,
                jac=lambda p: _fun(p)[1],
                method="L-BFGS-B",
                options=_opts,
            )

        result = _run_opt(init)

        # Multi-restart if first attempt did not converge
        if not result.success and self.n_restarts > 1:
            rng_r = np.random.default_rng(self.seed + 1)
            for _attempt in range(self.n_restarts - 1):
                perturbed = result.x + rng_r.normal(scale=0.1, size=result.x.shape)
                candidate = _run_opt(perturbed)
                if candidate.fun < result.fun:
                    result = candidate
                if result.success:
                    break

        self.result_ = result
        self.converged_ = bool(result.success)
        if not self.converged_:
            import warnings as _warnings
            _warnings.warn(
                f"JAXRandomEffectsAFTFitter ({self.family}): optimizer did not converge. "
                f"Message: {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        beta = np.asarray(result.x[:-1], dtype=float)
        sigma = float(np.log1p(np.exp(result.x[-1])) + 1e-8)

        coef_names = ["intercept", *feature_cols, "log_scale"]
        coef_values = np.concatenate([beta, np.array([sigma], dtype=float)])
        self.params_ = pd.Series(coef_values, index=coef_names, dtype=float)

        self.log_likelihood_ = float(-result.fun)
        k = len(coef_values)
        n = max(len(data), 1)
        aic = 2 * k - 2 * self.log_likelihood_
        bic = k * np.log(n) - 2 * self.log_likelihood_

        hess_inv = getattr(result, "hess_inv", None)
        if hess_inv is not None and hasattr(hess_inv, "todense"):
            cov = np.asarray(hess_inv.todense(), dtype=float)
        elif hess_inv is not None:
            cov = np.asarray(hess_inv, dtype=float)
        else:
            cov = np.eye(k, dtype=float) * np.nan

        if np.isfinite(cov[-1, -1]):
            dsoftplus = 1.0 / (1.0 + np.exp(-result.x[-1]))
            cov[-1, :] *= dsoftplus
            cov[:, -1] *= dsoftplus

        stderr = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        zvals = coef_values / np.where(stderr > 0, stderr, np.nan)
        pvals = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(zvals) / np.sqrt(2.0))))

        self.variance_matrix_ = pd.DataFrame(cov, index=coef_names, columns=coef_names)
        self.summary_ = pd.DataFrame(
            {
                "coef": coef_values,
                "stderr": stderr,
                "z": zvals,
                "pvalue": pvals,
                "aic": aic,
                "bic": bic,
            },
            index=coef_names,
        )

        self.feature_cols_ = list(feature_cols)
        self.duration_col_ = duration_col
        self.event_col_ = event_col
        self._prepare_random_effects()
        return self

    def _prepare_random_effects(self):
        requested = [name for name in self.random_terms if name in self.feature_cols_]
        correlated = [name for name in self.correlated_random_terms if name in requested]

        if not requested:
            self.random_mean_ = pd.Series(dtype=float)
            self.random_cov_ = pd.DataFrame()
            return

        idx = [name for name in requested if name in self.params_.index]
        if not idx:
            self.random_mean_ = pd.Series(dtype=float)
            self.random_cov_ = pd.DataFrame()
            return

        mean = self.params_.loc[idx].copy()
        cov = self.variance_matrix_.loc[idx, idx].copy()

        for r in cov.index:
            for c in cov.columns:
                if r != c and (r not in correlated or c not in correlated):
                    cov.loc[r, c] = 0.0

        self.random_mean_ = mean
        self.random_cov_ = cov

    def summary_frame(self) -> pd.DataFrame:
        return self.summary_.copy()

    def simulate_coefficients(self, n_draws: Optional[int] = None) -> pd.DataFrame:
        if self.random_mean_.empty:
            return pd.DataFrame()

        draw_count = int(n_draws or self.n_draws)
        rng = np.random.default_rng(self.seed)

        cov = self.random_cov_.to_numpy(dtype=float)
        cov = cov + np.eye(cov.shape[0]) * 1e-9
        draws = rng.multivariate_normal(
            mean=self.random_mean_.to_numpy(dtype=float),
            cov=cov,
            size=draw_count,
        )
        return pd.DataFrame(draws, columns=list(self.random_mean_.index))

    def predict_expectation(self, df: pd.DataFrame, n_draws: Optional[int] = None) -> pd.DataFrame:
        if self.params_.empty:
            raise RuntimeError("Model must be fit before prediction")

        X = df[self.feature_cols_].to_numpy(dtype=float)
        X = np.column_stack([np.ones((X.shape[0], 1)), X])
        beta = self.params_.loc[["intercept", *self.feature_cols_]].to_numpy(dtype=float)
        sigma = float(self.params_.loc["log_scale"])
        eta = X @ beta

        if self.family == "lognormal":
            base = np.exp(eta + 0.5 * sigma * sigma)
        elif self.family == "weibull":
            base = np.exp(eta) * np.vectorize(math.gamma)(1.0 + sigma)
        else:
            base = np.exp(eta) * (np.pi * sigma / np.sin(np.pi * sigma + 1e-8))

        out = pd.DataFrame({"mean": base}, index=df.index)
        draws = self.simulate_coefficients(n_draws=n_draws)
        if draws.empty:
            out["p10"] = out["mean"]
            out["p50"] = out["mean"]
            out["p90"] = out["mean"]
            return out

        design = df.loc[:, draws.columns].fillna(0.0).to_numpy(dtype=float)
        mean_vec = self.random_mean_.loc[draws.columns].to_numpy(dtype=float)
        delta = draws.to_numpy(dtype=float).T - mean_vec[:, None]
        adjustment = np.exp(design @ delta)
        sims = base[:, None] * adjustment

        out["p10"] = np.quantile(sims, 0.10, axis=1)
        out["p50"] = np.quantile(sims, 0.50, axis=1)
        out["p90"] = np.quantile(sims, 0.90, axis=1)
        out["draw_mean"] = sims.mean(axis=1)
        return out


class RandomEffectsAFTFitter(JAXRandomEffectsAFTFitter):
    """Backward-compatible alias for previous class name."""


class LogNormalRandomEffectsAFTFitter(JAXRandomEffectsAFTFitter):
    family = "lognormal"


class WeibullRandomEffectsAFTFitter(JAXRandomEffectsAFTFitter):
    family = "weibull"


class LogLogisticRandomEffectsAFTFitter(JAXRandomEffectsAFTFitter):
    family = "loglogistic"


@dataclass
class SurvivalSearchProblem:
    df: pd.DataFrame
    duration_col: str
    event_col: Optional[str] = None
    variables: Optional[List[str]] = None
    family: str = "lognormal"
    random_terms: Optional[List[str]] = None
    correlated_random_terms: Optional[List[str]] = None
    n_draws: int = 200

    def run(self) -> Dict[str, Any]:
        vars_used = list(self.variables or [])
        fitter_map = {
            "lognormal": LogNormalRandomEffectsAFTFitter,
            "weibull": WeibullRandomEffectsAFTFitter,
            "loglogistic": LogLogisticRandomEffectsAFTFitter,
        }
        fitter = fitter_map[self.family](
            random_terms=self.random_terms,
            correlated_random_terms=self.correlated_random_terms,
            n_draws=self.n_draws,
        )
        fitter.fit(
            self.df,
            duration_col=self.duration_col,
            event_col=self.event_col,
            feature_cols=vars_used,
        )
        n = len(self.df[[self.duration_col] + vars_used].dropna())
        k = len(fitter.params_)
        ll = float(fitter.log_likelihood_)
        bic = k * np.log(max(n, 1)) - 2 * ll
        return {
            "family": self.family,
            "result": fitter,
            "loglik": ll,
            "bic": bic,
            "n_params": k,
            "n_obs": n,
            "params_df": fitter.summary_frame(),
        }


__all__ = [
    "JAXRandomEffectsAFTFitter",
    "RandomEffectsAFTFitter",
    "LogNormalRandomEffectsAFTFitter",
    "WeibullRandomEffectsAFTFitter",
    "LogLogisticRandomEffectsAFTFitter",
    "SurvivalSearchProblem",
]
