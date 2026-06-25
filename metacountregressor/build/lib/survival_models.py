"""
survival_models.py
==================
JAX-native AFT (Accelerated Failure Time) models with full simulation-based
random parameters.  Borrows build_eta, build_jax_data, ModelSpec, Halton draws
and the Cholesky-correlated random-effect machinery directly from main_hpc so
that every feature available to count models is also available here:

  • Fixed parameters
  • Independent normal random parameters    (Kr_ind)
  • Correlated normal random parameters     (Kr_cor, Cholesky factorisation)
  • Heterogeneity in means                  (Kh)
  • Latent-class survival (via LC wrapper)  – C > 1 supported

Distributional families
-----------------------
  lognormal  : log T = η + σε,  ε ~ N(0,1)
  weibull    : log T = η + σε,  ε ~ Gumbel(0,1)  [extreme-value min]
  loglogistic: log T = η + σε,  ε ~ Logistic(0,1)

The scale σ is always estimated (parametrised via softplus so σ > 0).
Right-censoring is handled via the survival function for censored observations.

Public API
----------
  AFTFitter(family, random_terms, correlated_random_terms, ...)
      .fit(df, duration_col, event_col, feature_cols)
      .summary_frame()       -> pd.DataFrame
      .predict_median(df)    -> pd.Series  (median survival time)
      .predict_mean(df)      -> pd.Series  (mean survival time, lognormal only)
      .bic() / .aic()

  # Backward-compatible aliases
  LogNormalRandomEffectsAFTFitter
  WeibullRandomEffectsAFTFitter
  LogLogisticRandomEffectsAFTFitter

  SurvivalSearchProblem   – fits all three families, returns BIC-ranked table
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxopt import LBFGS

jax.config.update("jax_enable_x64", True)

# ── Import infrastructure from main_hpc ──────────────────────────────────────
try:
    from .main_hpc import (
        build_base_index,
        build_param_index,
        build_eta,
        build_jax_data,
        generate_halton_normal,
        ensure_3d,
        unpack_params,
        compute_standard_errors,
        balance_panel_dataframe,
        extract_offset,
    )
    # Extended ModelSpec (adds membership_names, K_membership) lives in the patch
    from .main_hpc_lc_patch import (
        ModelSpec,
        build_param_index as lc_build_param_index,
    )
except ImportError:
    from main_hpc import (
        build_base_index,
        build_param_index,
        build_eta,
        build_jax_data,
        generate_halton_normal,
        ensure_3d,
        unpack_params,
        compute_standard_errors,
        balance_panel_dataframe,
        extract_offset,
    )
    from main_hpc_lc_patch import (
        ModelSpec,
        build_param_index as lc_build_param_index,
    )

_SURVIVAL_FAMILIES = {"lognormal", "weibull", "loglogistic"}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Per-family AFT log-likelihoods
#
#     All functions receive arrays of shape (N, P, R):
#       y         – observed duration  (always > 0)
#       event     – 1 = observed, 0 = right-censored
#       eta       – linear predictor  (same shape as y after broadcast)
#       sigma_raw – raw scale param (scalar); transformed via softplus
#
#     Return:  ll of shape (N, P, R)
# ─────────────────────────────────────────────────────────────────────────────

def _aft_lognormal_ll(y, event, eta, sigma_raw):
    """
    log T = η + σε,  ε ~ N(0,1)
    f(t)  = φ(z)/(tσ)       log f = logpdf(z) − log t − log σ
    S(t)  = Φ(−z)           log S = log_ndtr(−z)
    where z = (log t − η)/σ
    """
    sigma   = jax.nn.softplus(sigma_raw)
    log_y   = jnp.log(jnp.clip(y, 1e-12, None))
    z       = (log_y - eta) / sigma
    ll_obs  = (
        -0.5 * jnp.log(2.0 * jnp.pi)
        - 0.5 * z ** 2
        - log_y
        - jnp.log(sigma)
    )
    ll_cens = jsp.special.log_ndtr(-z)
    return jnp.where(event > 0, ll_obs, ll_cens)


def _aft_weibull_ll(y, event, eta, sigma_raw):
    """
    log T = η + σε,  ε ~ Gumbel min (extreme value type I for minima)
    f(t)  : gumbel pdf   log f = z − exp(z) − log t − log σ
    S(t)  : exp(−exp(z)) log S = −exp(z)
    where z = (log t − η)/σ
    """
    sigma   = jax.nn.softplus(sigma_raw)
    log_y   = jnp.log(jnp.clip(y, 1e-12, None))
    z       = (log_y - eta) / sigma
    ll_obs  = z - jnp.exp(z) - log_y - jnp.log(sigma)
    ll_cens = -jnp.exp(z)
    return jnp.where(event > 0, ll_obs, ll_cens)


def _aft_loglogistic_ll(y, event, eta, sigma_raw):
    """
    log T = η + σε,  ε ~ Logistic(0,1)
    f(t)  : logistic pdf  log f = z − log t − log σ − 2 log(1+exp(z))
    S(t)  : 1/(1+exp(z))  log S = −softplus(z)
    where z = (log t − η)/σ
    """
    sigma   = jax.nn.softplus(sigma_raw)
    log_y   = jnp.log(jnp.clip(y, 1e-12, None))
    z       = (log_y - eta) / sigma
    ll_obs  = z - log_y - jnp.log(sigma) - 2.0 * jnp.log1p(jnp.exp(z))
    ll_cens = -jax.nn.softplus(z)
    return jnp.where(event > 0, ll_obs, ll_cens)


def aft_loglik(y, event, eta, sigma_raw, family: str):
    """Dispatch to the correct AFT log-likelihood by family name."""
    if family == "lognormal":
        return _aft_lognormal_ll(y, event, eta, sigma_raw)
    if family == "weibull":
        return _aft_weibull_ll(y, event, eta, sigma_raw)
    if family == "loglogistic":
        return _aft_loglogistic_ll(y, event, eta, sigma_raw)
    raise ValueError(f"Unknown AFT family '{family}'. Choose: lognormal, weibull, loglogistic")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  survival_mixed_model_loglik
#
#     Drop-in analogue of main_hpc.mixed_model_loglik for survival data.
#     Uses build_eta (which handles fixed + independent + correlated + hetero
#     random params via Halton draws) so all mixing structures are automatic.
#
#     Latent-class (C > 1) is supported via the same LC wrapper pattern as
#     main_hpc_lc_patch: per-class theta_c slices are passed recursively with
#     latent_classes=1.
# ─────────────────────────────────────────────────────────────────────────────

def survival_mixed_model_loglik(
    params,
    data: dict,
    spec: ModelSpec,
    family: str,
    indivi: bool = False,
):
    """
    Negative log-likelihood (or individual LLs) for an AFT survival model.

    Parameters
    ----------
    params   : 1-D JAX/numpy array of model parameters
    data     : dict produced by build_jax_survival_data — must contain
               "y" (duration), "event" (0/1), plus all standard keys
               (Xf, Xr_ind, draws_ind, etc.)
    spec     : ModelSpec with model in {"lognormal","weibull","loglogistic"}
    family   : AFT distribution family (may differ from spec.model if spec.model
               is used purely for parameter-layout purposes)
    indivi   : if True, return per-individual log-likelihoods (N,)

    Returns
    -------
    scalar  NLL  when indivi=False
    (N,)   ll_n  when indivi=True   (note: positive = higher likelihood)
    """
    # ── Latent-class wrapper ─────────────────────────────────────────────────
    if spec.latent_classes > 1:
        C         = spec.latent_classes
        base_spec = replace(spec, latent_classes=1)
        base_idx  = build_base_index(base_spec)
        K_base    = base_idx["total_params"]

        theta_all = params[:C * K_base].reshape(C, K_base)
        logits    = params[C * K_base:]
        logits_f  = jnp.concatenate([jnp.array([0.0]), logits])
        log_pi    = jax.nn.log_softmax(logits_f)           # (C,)

        ll_classes = []
        for c in range(C):
            ll_c = survival_mixed_model_loglik(
                theta_all[c], data, base_spec, family, indivi=True
            )
            ll_classes.append(ll_c + log_pi[c])

        ll_stack = jnp.stack(ll_classes, axis=1)           # (N, C)
        ll_ind   = jsp.special.logsumexp(ll_stack, axis=1) # (N,)

        if indivi:
            return ll_ind
        return -jnp.sum(ll_ind)

    # ── Single-class branch ──────────────────────────────────────────────────
    blocks  = unpack_params(params, spec)
    sigma   = blocks["sigma"]                              # raw; softplus inside aft_loglik

    # build_eta produces (N, P, R): fixed + random (Halton-integrated) predictor
    eta     = build_eta(params, data, spec)
    eta     = jnp.clip(eta, -500.0, 500.0)

    if eta.ndim == 2:
        eta = eta[..., None]

    y       = ensure_3d(data["y"])                         # (N, P, 1)
    event   = ensure_3d(data["event"])                     # (N, P, 1)
    mask    = ensure_3d(data["mask"])                      # (N, P, 1)
    R       = eta.shape[-1]

    ll      = aft_loglik(y, event, eta, sigma, family)    # (N, P, R)
    ll      = ll * mask
    ll_panel = jnp.sum(ll, axis=1)                         # (N, R)

    if R > 1:
        ll_ind = jsp.special.logsumexp(ll_panel, axis=-1) - jnp.log(R)
    else:
        ll_ind = ll_panel.squeeze(-1)

    if indivi:
        return ll_ind
    return -jnp.sum(ll_ind)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  build_jax_survival_data
#
#     Extends build_jax_data with an "event" key in the data dict.
# ─────────────────────────────────────────────────────────────────────────────

def build_jax_survival_data(
    df: pd.DataFrame,
    id_col: str,
    duration_col: str,
    event_col: Optional[str],
    fixed_cols: List[str],
    random_ind_cols: Optional[List[str]] = None,
    random_cor_cols: Optional[List[str]] = None,
    hetro_cols: Optional[List[str]] = None,
    random_ind_dists: Optional[List[str]] = None,
    random_cor_dists: Optional[List[str]] = None,
    R: int = 200,
    family: str = "lognormal",
) -> Tuple[dict, ModelSpec]:
    """
    Prepare the JAX data dict and ModelSpec for a survival model.

    The event indicator is attached as data["event"] (shape N, P, 1).
    The duration is stored in data["y"].

    One row per individual is assumed (panel P=1); repeat-measure data works
    automatically if multiple rows share the same id_col.
    """
    random_ind_cols  = list(random_ind_cols  or [])
    random_cor_cols  = list(random_cor_cols  or [])
    hetro_cols       = list(hetro_cols       or [])
    random_ind_dists = list(random_ind_dists or ["normal"] * len(random_ind_cols))
    random_cor_dists = list(random_cor_dists or ["normal"] * len(random_cor_cols))

    # Balance the panel (fills missing T-slots with the mask=0)
    df_bal = balance_panel_dataframe(df, id_col, [duration_col] + fixed_cols
                                     + random_ind_cols + random_cor_cols + hetro_cols
                                     + ([event_col] if event_col else []))

    # Extract event column (before calling build_jax_data which ignores it)
    ids_ordered = df_bal[id_col].unique()
    N = len(ids_ordered)
    id_to_idx = {iid: i for i, iid in enumerate(ids_ordered)}
    T_vals = sorted(df_bal.groupby(id_col).size().unique())
    P = max(T_vals)

    # Build event matrix (N, P, 1)
    event_matrix = np.zeros((N, P, 1), dtype=float)
    if event_col is not None:
        for _, row_g in df_bal.groupby(id_col):
            n_idx = id_to_idx[row_g[id_col].iloc[0]]
            for t, (_, row) in enumerate(row_g.iterrows()):
                if t < P:
                    event_matrix[n_idx, t, 0] = float(row[event_col])
    else:
        event_matrix[:] = 1.0   # all observed if no event column

    # Use build_jax_data with duration as y
    draws_ind = None
    draws_cor = None
    if random_ind_cols:
        draws_ind = generate_halton_normal(N, len(random_ind_cols), R, seed=42)
    if random_cor_cols:
        draws_cor = generate_halton_normal(N, len(random_cor_cols), R, seed=43)

    intercept_name = "__INTERCEPT__"
    data, spec = build_jax_data(
        df_bal,
        id_col         = id_col,
        y_col          = duration_col,
        fixed_cols     = [intercept_name] + fixed_cols,
        random_ind_cols= random_ind_cols,
        random_cor_cols= random_cor_cols,
        grouped_cols   = [],
        hetro_cols     = hetro_cols,
        zi_cols        = [],
        random_ind_dists = random_ind_dists,
        random_cor_dists = random_cor_dists,
        grouped_dists  = [],
        draws_ind      = draws_ind,
        draws_cor      = draws_cor,
        R              = R,
    )
    # Override model name so build_base_index knows to add sigma
    spec = replace(spec, model=family)

    data["event"] = jnp.array(event_matrix)
    return data, spec


def _build_survival_data_simple(
    df: pd.DataFrame,
    id_col: str,
    duration_col: str,
    event_col: Optional[str],
    fixed_cols: List[str],
    random_ind_cols: List[str],
    random_cor_cols: List[str],
    hetro_cols: List[str],
    R: int,
    family: str,
) -> Tuple[dict, ModelSpec]:
    """
    Lightweight builder for cross-sectional survival data.
    Treats each individual as a single-period panel (P=1).
    """
    rng = np.random.default_rng(42)

    cols_needed = [id_col, duration_col] + fixed_cols + random_ind_cols + random_cor_cols
    if event_col:
        cols_needed.append(event_col)
    df2 = df[list(dict.fromkeys(cols_needed))].dropna().reset_index(drop=True)

    N  = len(df2)
    Kf = len(fixed_cols) + 1    # +1 for intercept

    # Build fixed design matrix (N, 1, Kf)
    Xf = np.column_stack([np.ones(N)] + [df2[c].to_numpy(float) for c in fixed_cols])
    Xf = Xf[:, None, :]         # (N, 1, Kf)

    # Duration (N, 1, 1)
    y_vals  = df2[duration_col].to_numpy(float)[:, None, None]

    # Event (N, 1, 1)
    if event_col:
        ev_vals = df2[event_col].to_numpy(float)[:, None, None]
    else:
        ev_vals = np.ones((N, 1, 1), dtype=float)

    mask = np.ones((N, 1, 1), dtype=float)
    offset = np.zeros((N, 1, 1), dtype=float)

    # Halton draws for independent random effects
    Kr_ind = len(random_ind_cols)
    Kr_cor = len(random_cor_cols)

    if Kr_ind > 0:
        draws_ind = generate_halton_normal(N, Kr_ind, R, seed=42)  # (N, Kr_ind, R)
        Xr_ind    = np.column_stack([df2[c].to_numpy(float) for c in random_ind_cols])
        Xr_ind    = Xr_ind[:, None, :]
    else:
        draws_ind = np.zeros((N, 0, R))
        Xr_ind    = np.zeros((N, 1, 0))

    if Kr_cor > 0:
        draws_cor = generate_halton_normal(N, Kr_cor, R, seed=43)
        Xr_cor    = np.column_stack([df2[c].to_numpy(float) for c in random_cor_cols])
        Xr_cor    = Xr_cor[:, None, :]
    else:
        draws_cor = np.zeros((N, 0, R))
        Xr_cor    = np.zeros((N, 1, 0))

    Xh = np.zeros((N, 1, 0))

    Kh = len(hetro_cols)
    if Kh > 0:
        Xh = np.column_stack([df2[c].to_numpy(float) for c in hetro_cols])
        Xh = Xh[:, None, :]

    data = {
        "Xf":        jnp.array(Xf),
        "Xr_ind":    jnp.array(Xr_ind),
        "Xr_cor":    jnp.array(Xr_cor),
        "Xg":        jnp.zeros((N, 1, 0)),
        "Xh":        jnp.array(Xh),
        "Xzi":       jnp.zeros((N, 1, 0)),
        "Xmem":      jnp.zeros((N, 1, 0)),
        "y":         jnp.array(y_vals),
        "mask":      jnp.array(mask),
        "offset":    jnp.array(offset),
        "event":     jnp.array(ev_vals),
        "draws_ind": jnp.array(draws_ind),
        "draws_cor": jnp.array(draws_cor),
        "draws_g":   jnp.zeros((N, 0, R)),
        "group_ids": jnp.zeros(N, dtype=int),
    }

    spec = ModelSpec(
        Kf               = Kf,
        Kr_ind           = Kr_ind,
        Kr_cor           = Kr_cor,
        Kg               = 0,
        Kh               = Kh if Kh > 0 else 0,
        Kzi              = 0,
        model            = family,
        zero_inflated    = False,
        fixed_names      = tuple(["__INTERCEPT__"] + fixed_cols),
        zi_names         = (),
        random_ind_names = tuple(random_ind_cols),
        random_cor_names = tuple(random_cor_cols),
        grouped_names    = (),
        hetro_names      = tuple(hetro_cols),
        random_ind_dists = tuple(["normal"] * Kr_ind),
        random_cor_dists = tuple(["normal"] * Kr_cor),
        grouped_dists    = (),
        latent_classes   = 1,
        membership_names = (),
        K_membership     = 0,
    )
    return data, spec


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Initialisation helper — OLS on log(duration) for uncensored observations
# ─────────────────────────────────────────────────────────────────────────────

def _survival_ols_init(data: dict, K_base: int) -> np.ndarray:
    """
    Compute OLS starting values from the observed (uncensored) events.

    Returns a parameter vector of length K_base where:
      params[:Kf]      = OLS betas on log(duration)
      params[K_base-1] = softplus_inverse(sigma_hat)
    """
    y_raw    = np.array(data["y"])       # (N, 1, 1) duration
    Xf       = np.array(data["Xf"])      # (N, 1, Kf)
    ev_raw   = np.array(data["event"])   # (N, 1, 1) 0/1

    y_flat   = np.log(np.clip(y_raw[:, 0, 0], 1e-12, None))  # log(T), (N,)
    X_flat   = Xf[:, 0, :]                                     # (N, Kf)
    ev_flat  = ev_raw[:, 0, 0].astype(bool)

    # Use uncensored obs; fall back to all if too few
    if ev_flat.sum() >= X_flat.shape[1] + 2:
        y_fit = y_flat[ev_flat]
        X_fit = X_flat[ev_flat]
    else:
        y_fit = y_flat
        X_fit = X_flat

    try:
        beta_ols = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
    except Exception:
        beta_ols = np.zeros(X_flat.shape[1])

    resid     = y_fit - X_fit @ beta_ols
    sigma_hat = max(float(resid.std()), 0.1)
    sigma_raw = float(np.log(np.exp(sigma_hat) - 1.0 + 1e-8))

    params = np.zeros(K_base)
    params[:len(beta_ols)] = beta_ols
    params[K_base - 1]     = sigma_raw
    return params


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SurvivalModel — fit / predict / diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class SurvivalModel:
    """
    Low-level model object.  Analogous to CountModel in main_hpc.

    Fits survival_mixed_model_loglik via jaxopt LBFGS with OLS warm start.
    """

    def __init__(self, spec: ModelSpec, data: dict, family: str,
                 maxiter: int = 1500, n_restarts: int = 2):
        self.spec     = spec
        self.data     = data
        self.family   = family
        self.maxiter  = maxiter
        self.n_restarts = n_restarts
        self.params   = None
        self._result  = None

    def _objective(self, p):
        return survival_mixed_model_loglik(p, self.data, self.spec, self.family)

    def fit(self) -> "SurvivalModel":
        K = build_base_index(self.spec)["total_params"]
        p0_np  = _survival_ols_init(self.data, K)
        p0     = jnp.array(p0_np)

        solver = LBFGS(fun=self._objective, maxiter=self.maxiter)
        result = solver.run(p0)
        best   = result

        # Restarts with small perturbations
        rng = np.random.default_rng(0)
        for _ in range(self.n_restarts - 1):
            p_try = jnp.array(np.array(result.params) + rng.normal(0, 0.05, K))
            r_try = solver.run(p_try)
            if float(r_try.state.value) < float(best.state.value):
                best = r_try

        self.params  = np.array(best.params)
        self._result = best
        return self

    def loglik(self) -> float:
        return float(-self._objective(jnp.array(self.params)))

    def bic(self) -> float:
        k = len(self.params)
        n = int(self.data["y"].shape[0])
        return k * np.log(n) - 2.0 * self.loglik()

    def aic(self) -> float:
        return 2.0 * len(self.params) - 2.0 * self.loglik()

    def predict_median(self) -> np.ndarray:
        """
        Median survival time E[T_0.5] via the linear predictor.
        For all AFT families: median(T) = exp(η).
        Averaged over Halton draws.
        """
        eta = build_eta(self.params, self.data, self.spec)  # (N, P, R)
        return np.array(jnp.exp(eta).mean(axis=(-2, -1)))   # (N,)

    def predict_mean(self) -> np.ndarray:
        """
        Mean survival time (closed-form where available):
          lognormal   : E[T] = exp(η + σ²/2)
          weibull     : E[T] = exp(η) · Γ(1 + σ)
          loglogistic : E[T] = exp(η) · πσ / sin(πσ)   [only for σ < 1]
        Averaged over Halton draws.
        """
        blocks = unpack_params(self.params, self.spec)
        sigma  = float(jax.nn.softplus(blocks["sigma"]))
        eta    = np.array(build_eta(self.params, self.data, self.spec))  # (N, P, R)
        exp_eta = np.exp(eta)

        if self.family == "lognormal":
            mean_draws = exp_eta * np.exp(0.5 * sigma ** 2)
        elif self.family == "weibull":
            mean_draws = exp_eta * math.gamma(1.0 + sigma)
        else:  # loglogistic
            if sigma < 1.0:
                mean_draws = exp_eta * (math.pi * sigma / math.sin(math.pi * sigma))
            else:
                mean_draws = exp_eta   # undefined for σ≥1, fall back to median

        return mean_draws.mean(axis=(-2, -1))   # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  AFTFitter — high-level interface
# ─────────────────────────────────────────────────────────────────────────────

class AFTFitter:
    """
    High-level AFT fitter with full simulation-based random parameters.

    Replaces JAXRandomEffectsAFTFitter.  Random effects are now proper
    simulation-integrated mixed parameters (not post-fit Hessian draws).

    Parameters
    ----------
    family : "lognormal" | "weibull" | "loglogistic"
    random_terms : variables with independently-distributed normal random coef.
    correlated_random_terms : variables with jointly-normal random coef.
        (Cholesky-correlated).  Must be a subset of random_terms.
    hetro_in_means : variables that enter heterogeneity-in-means of random coef.
    n_draws : number of Halton draws for simulation integration.
    maxiter / n_restarts : LBFGS options.
    """

    def __init__(
        self,
        family: str = "lognormal",
        random_terms: Optional[List[str]] = None,
        correlated_random_terms: Optional[List[str]] = None,
        hetro_in_means: Optional[List[str]] = None,
        n_draws: int = 200,
        maxiter: int = 1500,
        n_restarts: int = 2,
    ):
        if family not in _SURVIVAL_FAMILIES:
            raise ValueError(f"family must be one of {_SURVIVAL_FAMILIES}")
        self.family                  = family
        self.random_terms            = list(random_terms or [])
        self.correlated_random_terms = list(correlated_random_terms or [])
        self.hetro_in_means          = list(hetro_in_means or [])
        self.n_draws                 = int(n_draws)
        self.maxiter                 = int(maxiter)
        self.n_restarts              = int(n_restarts)

        # filled after fit()
        self._model: Optional[SurvivalModel] = None
        self._spec:  Optional[ModelSpec]     = None
        self._data:  Optional[dict]          = None
        self._feature_cols:  List[str]       = []
        self._duration_col:  str             = ""
        self._event_col:     Optional[str]   = None
        self.summary_:       pd.DataFrame    = pd.DataFrame()
        self.params_:        pd.Series       = pd.Series(dtype=float)
        self.log_likelihood_: float          = float("nan")

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> "AFTFitter":
        reserved = {duration_col}
        if event_col:
            reserved.add(event_col)
        feature_cols = list(feature_cols or [
            c for c in df.columns if c not in reserved
        ])

        self._duration_col = duration_col
        self._event_col    = event_col
        self._feature_cols = feature_cols

        # Separate independent vs correlated random terms
        rdm_ind = [v for v in self.random_terms if v not in self.correlated_random_terms
                   and v in feature_cols]
        rdm_cor = [v for v in self.correlated_random_terms if v in feature_cols]
        hetro   = [v for v in self.hetro_in_means if v in feature_cols]

        df2 = df.copy()
        df2["__id__"] = np.arange(len(df2))
        data, spec = _build_survival_data_simple(
            df              = df2,
            id_col          = "__id__",
            duration_col    = duration_col,
            event_col       = event_col,
            fixed_cols      = feature_cols,
            random_ind_cols = rdm_ind,
            random_cor_cols = rdm_cor,
            hetro_cols      = hetro,
            R               = self.n_draws,
            family          = self.family,
        )

        self._data = data
        self._spec = spec

        model = SurvivalModel(
            spec       = spec,
            data       = data,
            family     = self.family,
            maxiter    = self.maxiter,
            n_restarts = self.n_restarts,
        )
        model.fit()
        self._model = model

        self.log_likelihood_ = model.loglik()
        self._build_summary()
        return self

    def _build_summary(self):
        """Extract parameters, standard errors, z-stats."""
        params_np = self._model.params
        K         = len(params_np)
        spec      = self._spec
        family    = self.family

        # Parameter names
        names = ["__INTERCEPT__"] + self._feature_cols

        if spec.Kr_ind > 0:
            for nm in spec.random_ind_names:
                names.append(f"mean({nm})")
            for nm in spec.random_ind_names:
                names.append(f"sd({nm})")

        if spec.Kr_cor > 0:
            for nm in spec.random_cor_names:
                names.append(f"cor_mean({nm})")
            Kchol = spec.Kr_cor * (spec.Kr_cor + 1) // 2
            i = 0
            for r in range(spec.Kr_cor):
                for c in range(r + 1):
                    names.append(f"chol({spec.random_cor_names[r]},{spec.random_cor_names[c]})")
                    i += 1

        if spec.Kh > 0 and spec.K_random_total > 0:
            for rnd in list(spec.random_cor_names) + list(spec.random_ind_names):
                for z in spec.hetro_names:
                    names.append(f"hetro({rnd}|{z})")

        names.append("sigma")

        if len(names) != K:
            names = [f"param_{i}" for i in range(K)]

        # Standard errors via numerical Hessian
        try:
            objective = partial(
                survival_mixed_model_loglik,
                data=self._data, spec=self._spec, family=self.family
            )
            se_np = np.array(compute_standard_errors(params_np, objective))
        except Exception:
            se_np = np.full(K, np.nan)

        zvals = params_np / np.where(se_np > 1e-10, se_np, np.nan)
        pvals = 2.0 * (1.0 - 0.5 * (
            1.0 + np.vectorize(math.erf)(np.abs(zvals) / np.sqrt(2.0))
        ))

        n   = int(self._data["y"].shape[0])
        k   = K
        ll  = self.log_likelihood_
        aic = 2 * k - 2 * ll
        bic = k * np.log(n) - 2 * ll

        self.params_ = pd.Series(params_np, index=names, dtype=float)
        self.summary_ = pd.DataFrame(
            {
                "coef":   params_np,
                "stderr": se_np,
                "z":      zvals,
                "pvalue": pvals,
            },
            index=names,
        )
        self._bic = bic
        self._aic = aic

    # ── public accessors ─────────────────────────────────────────────────────

    def summary_frame(self) -> pd.DataFrame:
        return self.summary_.copy()

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  AFT Model  —  family: {self.family}")
        print(f"  LL = {self.log_likelihood_:.4f}   "
              f"AIC = {self._aic:.2f}   BIC = {self._bic:.2f}")
        print(f"{'='*60}")
        print(self.summary_.to_string(float_format=lambda x: f"{x:+.4f}"))
        print(f"{'='*60}\n")

    def bic(self) -> float:
        return self._bic

    def aic(self) -> float:
        return self._aic

    def predict_median(self, df: pd.DataFrame) -> pd.Series:
        """Median survival time for new data."""
        if self._model is None:
            raise RuntimeError("Call .fit() first.")
        Xf_new = df[self._feature_cols].to_numpy(float)
        Xf_new = np.column_stack([np.ones(len(Xf_new)), Xf_new])
        beta_f = np.array(self._model.params[:Xf_new.shape[1]])
        eta    = Xf_new @ beta_f
        return pd.Series(np.exp(eta), index=df.index, name="median_survival")

    def predict_mean(self, df: pd.DataFrame) -> pd.Series:
        """Mean survival time for new data (fixed-effects only)."""
        if self._model is None:
            raise RuntimeError("Call .fit() first.")
        Xf_new = df[self._feature_cols].to_numpy(float)
        Xf_new = np.column_stack([np.ones(len(Xf_new)), Xf_new])
        beta_f = np.array(self._model.params[:Xf_new.shape[1]])
        eta    = Xf_new @ beta_f
        spec   = self._spec
        blocks = unpack_params(self._model.params, spec)
        sigma  = float(jax.nn.softplus(blocks["sigma"]))

        if self.family == "lognormal":
            mean = np.exp(eta + 0.5 * sigma ** 2)
        elif self.family == "weibull":
            mean = np.exp(eta) * math.gamma(1.0 + sigma)
        else:
            if sigma < 1.0:
                mean = np.exp(eta) * (math.pi * sigma / math.sin(math.pi * sigma))
            else:
                mean = np.exp(eta)

        return pd.Series(mean, index=df.index, name="mean_survival")

    def simulate_draw(self, n_draws: int = 500, seed: int = 0) -> pd.DataFrame:
        """
        Draw from the posterior of the random parameters (if any).
        Returns a DataFrame of shape (n_draws, n_random_params).
        """
        spec   = self._spec
        params = self._model.params
        idx    = build_base_index(spec)

        if spec.Kr_ind == 0 and spec.Kr_cor == 0:
            return pd.DataFrame()

        rng = np.random.default_rng(seed)
        rows = {}

        if spec.Kr_ind > 0:
            s0, s1 = idx["ind_mean"]
            means  = params[s0:s1]
            sd_s0, sd_s1 = idx["ind_sd"]
            sds    = np.abs(params[sd_s0:sd_s1])
            for nm, mu, sd in zip(spec.random_ind_names, means, sds):
                rows[nm] = rng.normal(mu, sd, n_draws)

        if spec.Kr_cor > 0:
            s0, s1 = idx["cor_mean"]
            means  = params[s0:s1]
            c0, c1 = idx["chol"]
            chol_v = params[c0:c1]
            K      = spec.Kr_cor
            L      = np.zeros((K, K))
            i = 0
            for r in range(K):
                for c in range(r + 1):
                    L[r, c] = chol_v[i]; i += 1
            L[np.diag_indices(K)] = np.abs(np.diag(L))
            Sigma  = L @ L.T
            draws  = rng.multivariate_normal(means, Sigma, n_draws)
            for j, nm in enumerate(spec.random_cor_names):
                rows[nm] = draws[:, j]

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Backward-compatible aliases and search problem
# ─────────────────────────────────────────────────────────────────────────────

class _FamilyFitter(AFTFitter):
    """Subclass that fixes the family at class definition time."""
    _fixed_family: str = "lognormal"

    def __init__(self, random_terms=None, correlated_random_terms=None,
                 hetro_in_means=None, n_draws=200, maxiter=1500, n_restarts=2,
                 **_ignored):
        super().__init__(
            family                  = self._fixed_family,
            random_terms            = random_terms,
            correlated_random_terms = correlated_random_terms,
            hetro_in_means          = hetro_in_means,
            n_draws                 = n_draws,
            maxiter                 = maxiter,
            n_restarts              = n_restarts,
        )


class RandomEffectsAFTFitter(_FamilyFitter):
    """Backward-compatible: defaults to lognormal."""
    _fixed_family = "lognormal"


class LogNormalRandomEffectsAFTFitter(_FamilyFitter):
    _fixed_family = "lognormal"


class WeibullRandomEffectsAFTFitter(_FamilyFitter):
    _fixed_family = "weibull"


class LogLogisticRandomEffectsAFTFitter(_FamilyFitter):
    _fixed_family = "loglogistic"


class SurvivalSearchProblem:
    """
    Fits all three AFT families (and optionally multiple random-effect specs),
    returns a BIC-ranked comparison.

    Parameters
    ----------
    df, duration_col, event_col  : data
    variables                    : outcome covariates
    random_terms                 : independent random-parameter variables
    correlated_random_terms      : correlated random-parameter variables
    families                     : list to search over; default all three
    n_draws                      : Halton draws
    """

    def __init__(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: Optional[str] = None,
        variables: Optional[List[str]] = None,
        random_terms: Optional[List[str]] = None,
        correlated_random_terms: Optional[List[str]] = None,
        families: Optional[List[str]] = None,
        n_draws: int = 200,
    ):
        self.df                      = df
        self.duration_col            = duration_col
        self.event_col               = event_col
        self.variables               = list(variables or [])
        self.random_terms            = list(random_terms or [])
        self.correlated_random_terms = list(correlated_random_terms or [])
        self.families                = list(families or list(_SURVIVAL_FAMILIES))
        self.n_draws                 = int(n_draws)

    def run(self) -> pd.DataFrame:
        """
        Fit all requested families.  Returns a DataFrame ranked by BIC with
        columns: family, LL, k, n, AIC, BIC, converged, fitter.
        """
        rows = []
        for fam in self.families:
            fitter = AFTFitter(
                family                  = fam,
                random_terms            = self.random_terms,
                correlated_random_terms = self.correlated_random_terms,
                n_draws                 = self.n_draws,
            )
            try:
                fitter.fit(
                    df           = self.df,
                    duration_col = self.duration_col,
                    event_col    = self.event_col,
                    feature_cols = self.variables or None,
                )
                n  = int(fitter._data["y"].shape[0])
                k  = len(fitter.params_)
                ll = fitter.log_likelihood_
                rows.append({
                    "family":    fam,
                    "LL":        ll,
                    "k":         k,
                    "n":         n,
                    "AIC":       fitter.aic(),
                    "BIC":       fitter.bic(),
                    "converged": True,
                    "fitter":    fitter,
                })
            except Exception as exc:
                warnings.warn(f"SurvivalSearchProblem: {fam} failed — {exc}", RuntimeWarning)
                rows.append({
                    "family": fam, "LL": float("nan"), "k": None, "n": None,
                    "AIC": float("nan"), "BIC": float("nan"),
                    "converged": False, "fitter": None,
                })

        cmp = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
        cmp["dBIC"] = cmp["BIC"] - cmp["BIC"].min()
        return cmp


__all__ = [
    # Core engine
    "AFTFitter",
    "SurvivalModel",
    "survival_mixed_model_loglik",
    "aft_loglik",
    # Backward-compatible aliases
    "RandomEffectsAFTFitter",
    "LogNormalRandomEffectsAFTFitter",
    "WeibullRandomEffectsAFTFitter",
    "LogLogisticRandomEffectsAFTFitter",
    # Search
    "SurvivalSearchProblem",
]
