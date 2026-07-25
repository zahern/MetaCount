"""
multivariate_count_regressor.py
================================
Multivariate count-data regression for jointly predicting M activity types
that a person undertakes in a day (e.g. work trips, shopping trips, recreation,
eating-out, etc.).

Model structure
---------------
For person i and activity type m = 1, …, M:

    Y_{i,m} | x_i  ~  NegBin(mu_{i,m},  alpha_m)

    log mu_{i,m}  =  x_i @ beta_m  +  offset_{i,m}

The M marginal distributions are coupled through a **Gaussian copula** with
correlation matrix Sigma (an M×M symmetric positive-definite matrix).  This
captures the fact that a person's activities in a day are jointly determined —
someone who makes many work trips tends to make fewer leisure trips, etc.

Estimation is two-stage (Joe, 1997; Ahmad et al., 2023):
  1. **Margins**: fit M independent univariate NB regressions via L-BFGS-B to
     obtain beta_m and alpha_m for each activity type.
  2. **Joint copula**: hold alpha_m fixed and optimise the joint Gaussian-copula
     log-likelihood over {beta_m} ∪ {L_ij} where L is the lower-triangular
     Cholesky factor of Sigma.

Public API
----------
>>> from metacountregressor.multivariate_count_regressor import (
...     MultivariateCountRegressor,
... )
>>> model = MultivariateCountRegressor(activity_names=["work", "shop", "rec"])
>>> model.fit(X, Y, offsets)      # X: (n,k), Y: (n,M), offsets: (n,M)
>>> print(model.summary())
>>> preds = model.predict(X_new, offsets_new)  # (n_new, M)

Notes
-----
* JAX + scipy are required.
* For Poisson marginals set ``marginal='poisson'``.
* Shared covariates (same X for all outcomes) or outcome-specific covariate
  matrices ``X_list`` are both accepted.
* The Gaussian copula is the default.  A vine-copula decomposition into
  sequential bivariate Frank copulas is also available via
  ``copula='vine-frank'``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ── JAX ─────────────────────────────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import jax.scipy.special as jsp_special
    jax.config.update("jax_enable_x64", True)
    _JAX_OK = True
except ImportError:
    _JAX_OK = False

try:
    from scipy.optimize import minimize as _scipy_minimize
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ── Local imports (graceful fallback) ────────────────────────────────────
try:
    from .bivariate_copula import (
        nb_log_pmf,
        nb_log_cdf,
        fit_univariate_nb,
        _hessian_se,
        _z_p,
        _sig_stars,
    )
except ImportError:
    from bivariate_copula import (
        nb_log_pmf,
        nb_log_cdf,
        fit_univariate_nb,
        _hessian_se,
        _z_p,
        _sig_stars,
    )

__all__ = [
    "MultivariateCountRegressor",
    "MultivariateCountFit",
    "gaussian_copula_loglik",
    "vine_frank_copula_loglik",
]

# ═══════════════════════════════════════════════════════════════════════
# 1.  Utility: Cholesky ↔ correlation-matrix conversions
# ═══════════════════════════════════════════════════════════════════════

def _chol_to_corr(L: jnp.ndarray) -> jnp.ndarray:
    """
    Given lower-triangular Cholesky factor L, return the correlation matrix
    Sigma = L @ L.T with diagonal normalised to 1.

    Parameters
    ----------
    L : (M, M) lower-triangular JAX array

    Returns
    -------
    Sigma : (M, M) correlation matrix
    """
    Sigma = L @ L.T
    d = jnp.sqrt(jnp.clip(jnp.diag(Sigma), 1e-12, None))
    Sigma = Sigma / jnp.outer(d, d)
    return Sigma


def _pack_chol(L_lower: np.ndarray) -> np.ndarray:
    """Flatten the lower-triangular part of L into a 1-D parameter vector."""
    M = L_lower.shape[0]
    idx = np.tril_indices(M)
    return L_lower[idx]


def _unpack_chol(chol_vec: Union[np.ndarray, jnp.ndarray], M: int):
    """
    Reconstruct a lower-triangular Cholesky matrix from a flat vector.

    The diagonal is exponentiated (so it stays positive) and off-diagonal
    entries are free.  This gives an unconstrained parameterisation for a
    positive-definite Sigma.
    """
    n_elem = M * (M + 1) // 2
    chol_vec = chol_vec[:n_elem]
    L = jnp.zeros((M, M))
    row, col = jnp.tril_indices(M)
    L = L.at[row, col].set(chol_vec)
    # Exponentiate diagonal to enforce positive definiteness
    diag_idx = jnp.arange(M)
    L = L.at[diag_idx, diag_idx].set(jnp.exp(jnp.diag(L)))
    return L


def _init_chol(M: int, empirical_corr: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return a flat initial vector for the Cholesky factor.

    If *empirical_corr* is provided (shape M×M), use its Cholesky factor
    (clipped for numerical stability); otherwise start at the identity.
    """
    if empirical_corr is not None:
        try:
            Sigma = np.clip(empirical_corr, -0.95, 1.0)
            np.fill_diagonal(Sigma, 1.0)
            L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(M))
            return _pack_chol(np.log(np.diag(L))[:, None] * np.eye(M) +
                              np.tril(L, -1) + np.diag(np.log(np.diag(L))))
        except np.linalg.LinAlgError:
            pass
    # Identity start: log(1)=0 on diagonal, 0 off-diagonal
    v = np.zeros(M * (M + 1) // 2)
    # diagonal entries of L=I satisfy log(1)=0, already zero
    return v


# ═══════════════════════════════════════════════════════════════════════
# 2.  Poisson log-PMF (for mixed marginal option)
# ═══════════════════════════════════════════════════════════════════════

def poisson_log_pmf(k, mu):
    """Log P(Y=k) for Poisson(mu)."""
    return k * jnp.log(jnp.clip(mu, 1e-300, None)) - mu - jsp_special.gammaln(k + 1.0)


def poisson_log_cdf(k, mu):
    """Log P(Y <= k) for Poisson(mu) via regularised lower incomplete gamma."""
    k_safe = jnp.maximum(k, 0.0)
    # regularised lower incomplete gamma = 1 - betainc(k+1, 0, mu) ... use:
    # P(Y<=k) = Q(k+1, mu) where Q is upper regularised Gamma
    # P(Y<=k) = 1 - P(Y > k) = gammaincupper(k+1, mu)... tricky in JAX
    # Use: P(Y<=k) = regularized lower incomplete gamma(k+1, mu)
    # jax: gammainc(a, x) = regularized lower incomplete gamma P(a,x)
    val = jsp_special.gammainc(k_safe + 1.0, jnp.clip(mu, 1e-300, None))
    return jnp.log(jnp.clip(val, 1e-300, 1.0))


# ═══════════════════════════════════════════════════════════════════════
# 3.  Gaussian copula log-likelihood for M count outcomes
# ═══════════════════════════════════════════════════════════════════════

def gaussian_copula_loglik(
    Y: jnp.ndarray,           # (n, M) integer counts
    mus: jnp.ndarray,         # (n, M) predicted means
    alphas: jnp.ndarray,      # (M,)   overdispersion (one per marginal)
    chol_vec: jnp.ndarray,    # (M*(M+1)//2,) Cholesky factor params
    M: int,
    marginal: str = "nb",
) -> jnp.ndarray:
    """
    Log-likelihood for M jointly distributed count outcomes coupled through
    a Gaussian copula.

    For discrete Y the bivariate rectangle probability (generalised to M dims)
    is:

        P(Y_1=y_1, …, Y_M=y_M)
          = sum_{s in {0,1}^M}  (-1)^{|s|}  C_Gauss(F_1(y_1-s_1), …, F_M(y_M-s_M))

    where C_Gauss is the M-dimensional Gaussian copula CDF evaluated at the
    M marginal CDFs.

    Because a full M-dimensional copula CDF requires expensive numerical
    integration, we follow Ahmad et al. (2023) and work on the **log-density**
    (continuous-approximation) side:

        log L_i ≈  sum_m log p_m(y_{i,m})
                  + log c_Gauss(u_{i,1}, …, u_{i,M})

    where u_{i,m} = F_m(y_{i,m}) (mid-CDF for discrete),
    and   c_Gauss is the M-variate Gaussian copula *density*.

    The Gaussian copula density is:

        log c(u_1,…,u_M; Sigma) =
            -0.5 * log det(Sigma)
            - 0.5 * z^T (Sigma^{-1} - I) z

    with z_m = Phi^{-1}(u_m).

    This is a common practical approximation for discrete copulas used in
    the transportation/activity-travel literature.

    Parameters
    ----------
    Y       : (n, M) observed counts
    mus     : (n, M) predicted means mu_{i,m} = exp(x_i @ beta_m + offset_{i,m})
    alphas  : (M,)  overdispersion; use 0 for Poisson, > 0 for NB
    chol_vec: flat lower-triangular Cholesky params (length M*(M+1)//2)
    M       : number of outcomes
    marginal: 'nb' | 'poisson'

    Returns
    -------
    scalar total log-likelihood
    """
    L = _unpack_chol(chol_vec, M)
    Sigma = _chol_to_corr(L)

    eps = 1e-9

    # ── Marginal log-PMFs and CDFs ─────────────────────────────────────
    if marginal == "nb":
        # (n, M)
        log_pmf = jnp.stack(
            [nb_log_pmf(Y[:, m], mus[:, m], alphas[m]) for m in range(M)], axis=1
        )
        # Smooth mid-CDF: use average of CDF at y and y-1
        log_cdf = jnp.stack(
            [nb_log_cdf(Y[:, m], mus[:, m], alphas[m]) for m in range(M)], axis=1
        )
        log_cdf_m1 = jnp.stack(
            [nb_log_cdf(Y[:, m] - 1, mus[:, m], alphas[m]) for m in range(M)], axis=1
        )
    else:  # poisson
        log_pmf = jnp.stack(
            [poisson_log_pmf(Y[:, m], mus[:, m]) for m in range(M)], axis=1
        )
        log_cdf = jnp.stack(
            [poisson_log_cdf(Y[:, m], mus[:, m]) for m in range(M)], axis=1
        )
        log_cdf_m1 = jnp.stack(
            [poisson_log_cdf(Y[:, m] - 1, mus[:, m]) for m in range(M)], axis=1
        )

    # Mid-point CDF u_{i,m} = 0.5*(F(y) + F(y-1)), clipped to (eps, 1-eps)
    cdf_hi = jnp.exp(log_cdf)
    cdf_lo = jnp.exp(log_cdf_m1)
    u = jnp.clip(0.5 * (cdf_hi + cdf_lo), eps, 1.0 - eps)  # (n, M)

    # ── Probit transform: z_{i,m} = Phi^{-1}(u_{i,m}) ────────────────
    z = jsp_special.ndtri(u)  # (n, M)

    # ── Gaussian copula log-density ────────────────────────────────────
    # log det(Sigma)
    sign, log_det = jnp.linalg.slogdet(Sigma)
    log_det = jnp.where(sign > 0, log_det, jnp.zeros_like(log_det))

    # Sigma^{-1}
    Sigma_inv = jnp.linalg.solve(Sigma, jnp.eye(M))

    # z^T (Sigma^{-1} - I) z  for each obs  (n,)
    zI = z @ (Sigma_inv - jnp.eye(M))   # (n, M)
    quad = jnp.sum(zI * z, axis=1)      # (n,)

    log_copula_density = -0.5 * log_det - 0.5 * quad  # (n,)

    # ── Total log-likelihood ───────────────────────────────────────────
    ll_marginals = jnp.sum(log_pmf, axis=1)  # (n,)
    ll = jnp.sum(ll_marginals + log_copula_density)
    return ll


# ═══════════════════════════════════════════════════════════════════════
# 4.  Vine copula (pair-copula construction using Frank bivariate copulas)
# ═══════════════════════════════════════════════════════════════════════

def _frank_copula_cdf(u: jnp.ndarray, v: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    """Frank copula CDF: C(u,v;rho) = -(1/rho) * log(inner/eta)."""
    eps = 1e-12
    rho_safe = jnp.where(jnp.abs(rho) < eps, eps * jnp.sign(rho + eps), rho)
    eta = 1.0 - jnp.exp(-rho_safe)
    eta_safe = jnp.where(jnp.abs(eta) < eps, eps, eta)
    inner = eta_safe - (1.0 - jnp.exp(-rho_safe * u)) * (1.0 - jnp.exp(-rho_safe * v))
    inner_safe = jnp.clip(inner / eta_safe, eps, 1.0 - eps)
    C = -jnp.log(inner_safe) / rho_safe
    return jnp.clip(C, eps, 1.0 - eps)


def vine_frank_copula_loglik(
    Y: jnp.ndarray,         # (n, M)
    mus: jnp.ndarray,       # (n, M)
    alphas: jnp.ndarray,    # (M,)
    rho_params: jnp.ndarray,  # (M*(M-1)//2,)  one rho per pair (i<j)
    M: int,
    marginal: str = "nb",
) -> jnp.ndarray:
    """
    D-vine (pair-copula construction) log-likelihood using Frank bivariate
    copulas.  The vine factorises the M-dimensional density as a product of
    M*(M-1)/2 bivariate copula densities, one for each adjacent pair in
    each tree level.

    This is equivalent to the R-vine representation with a path graph
    structure (1-2, 2-3, …, (M-1)-M in tree 1; conditional pairs in tree 2…).

    For practical tractability (no numerical integration of conditional CDFs
    beyond tree 1), this implementation uses the **C-vine** approximation
    where all pairs are conditioned on the first variable as the central node.

    Parameters
    ----------
    rho_params : (M*(M-1)//2,) — one Frank rho per unique pair (i,j) with i<j,
                  ordered as (0,1),(0,2),…,(0,M-1),(1,2),…
    """
    eps = 1e-9

    # ── Marginal CDFs ─────────────────────────────────────────────────
    if marginal == "nb":
        log_pmf = jnp.stack(
            [nb_log_pmf(Y[:, m], mus[:, m], alphas[m]) for m in range(M)], axis=1
        )
        log_cdf = jnp.stack(
            [nb_log_cdf(Y[:, m], mus[:, m], alphas[m]) for m in range(M)], axis=1
        )
        log_cdf_m1 = jnp.stack(
            [nb_log_cdf(Y[:, m] - 1, mus[:, m], alphas[m]) for m in range(M)], axis=1
        )
    else:
        log_pmf = jnp.stack(
            [poisson_log_pmf(Y[:, m], mus[:, m]) for m in range(M)], axis=1
        )
        log_cdf = jnp.stack(
            [poisson_log_cdf(Y[:, m], mus[:, m]) for m in range(M)], axis=1
        )
        log_cdf_m1 = jnp.stack(
            [poisson_log_cdf(Y[:, m] - 1, mus[:, m]) for m in range(M)], axis=1
        )

    cdf_hi = jnp.exp(log_cdf)
    cdf_lo = jnp.exp(log_cdf_m1)
    u = jnp.clip(0.5 * (cdf_hi + cdf_lo), eps, 1.0 - eps)  # (n, M)

    ll_marginals = jnp.sum(log_pmf, axis=1)  # (n,)

    # ── C-vine: all pairs (0,m) and (i,j) with i<j (tree 1 only) ─────
    # For simplicity and JAX-differentiability, use the C-vine where the
    # first variable (m=0) is the hub and all M-1 pairs involve variable 0.
    # Additional tree pairs (j,k | 0) use conditional CDFs.
    # Here we implement tree-1 only (the dominant term) for tractability.
    rho_idx = 0
    log_vine = jnp.zeros(Y.shape[0])

    for i in range(M):
        for j in range(i + 1, M):
            if rho_idx >= len(rho_params):
                break
            rho = rho_params[rho_idx]
            rho_idx += 1

            ui = u[:, i]
            uj = u[:, j]

            # Frank copula CDF at (ui, uj) and boundary terms
            # Compute cell mass: C(F_i, F_j) - C(F_im1, F_j) - C(F_i, F_jm1) + C(F_im1, F_jm1)
            cdf_hi_i  = cdf_hi[:, i];   cdf_lo_i = cdf_lo[:, i]
            cdf_hi_j  = cdf_hi[:, j];   cdf_lo_j = cdf_lo[:, j]

            C_pp = _frank_copula_cdf(cdf_hi_i, cdf_hi_j, rho)
            C_pm = _frank_copula_cdf(cdf_hi_i, cdf_lo_j, rho)
            C_mp = _frank_copula_cdf(cdf_lo_i, cdf_hi_j, rho)
            C_mm = _frank_copula_cdf(cdf_lo_i, cdf_lo_j, rho)

            mass_ij = jnp.clip(C_pp - C_pm - C_mp + C_mm, eps, None)
            f_i = jnp.clip(cdf_hi_i - cdf_lo_i, eps, None)
            f_j = jnp.clip(cdf_hi_j - cdf_lo_j, eps, None)
            # copula density correction: log c(u,v) ≈ log(mass/(f_i*f_j))
            log_vine = log_vine + jnp.log(mass_ij) - jnp.log(f_i) - jnp.log(f_j)

    return jnp.sum(ll_marginals) + jnp.sum(log_vine)


# ═══════════════════════════════════════════════════════════════════════
# 5.  Joint parameter packing / unpacking
# ═══════════════════════════════════════════════════════════════════════

def _pack_joint_params(
    betas_list: List[np.ndarray],   # [beta_1, …, beta_M]
    copula_params: np.ndarray,       # Cholesky vec or rho vec
) -> np.ndarray:
    return np.concatenate([b.ravel() for b in betas_list] + [copula_params])


def _unpack_joint_params(
    params: Union[np.ndarray, jnp.ndarray],
    ks: List[int],   # [k_1, …, k_M]  columns per outcome
    n_copula: int,
):
    """Return betas_list, copula_params."""
    splits = np.cumsum([0] + ks)
    betas_list = [params[splits[m]: splits[m + 1]] for m in range(len(ks))]
    copula_params = params[splits[-1]: splits[-1] + n_copula]
    return betas_list, copula_params


# ═══════════════════════════════════════════════════════════════════════
# 6.  Result dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MultivariateCountFit:
    """Fitted multivariate count model."""
    activity_names: List[str]
    M: int
    n: int
    k_total: int

    # Per-outcome estimates
    coef: List[np.ndarray]        # [beta_1, …, beta_M]
    alphas: np.ndarray            # (M,)  overdispersion
    se: List[np.ndarray]          # [se_1, …, se_M]
    feature_names: List[List[str]]

    # Copula
    copula: str
    correlation: np.ndarray       # (M, M) implied correlation matrix
    copula_params_raw: np.ndarray # raw optimised copula parameters

    # Fit quality
    loglik: float
    aic: float
    bic: float
    converged: bool
    marginal: str

    # Stage-1 marginal results (for reference)
    marginal_fits: List[dict] = field(default_factory=list)

    # Standard errors for copula params
    se_copula: np.ndarray = field(default_factory=lambda: np.array([]))

    def predict(
        self,
        X_list: List[np.ndarray],
        offsets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict expected activity counts for new observations.

        Parameters
        ----------
        X_list : list of M arrays, each (n_new, k_m)
        offsets : (n_new, M) or None

        Returns
        -------
        preds : (n_new, M)  predicted means
        """
        n_new = X_list[0].shape[0]
        if offsets is None:
            offsets = np.zeros((n_new, self.M))
        preds = np.zeros((n_new, self.M))
        for m, (beta, X) in enumerate(zip(self.coef, X_list)):
            preds[:, m] = np.exp(X @ beta + offsets[:, m])
        return preds

    def summary(self) -> str:
        """Print a formatted model summary."""
        lines = [
            "=" * 70,
            f"  Multivariate Count Regressor  ({self.copula} copula)",
            f"  Activities: {', '.join(self.activity_names)}",
            f"  Observations: {self.n}   Parameters: {self.k_total}",
            f"  Log-lik: {self.loglik:.4f}   AIC: {self.aic:.2f}   BIC: {self.bic:.2f}",
            f"  Converged: {self.converged}",
            "=" * 70,
        ]
        for m in range(self.M):
            lines.append(f"\n  Activity {m+1}: {self.activity_names[m]}"
                         f"  (alpha = {self.alphas[m]:.4f})")
            lines.append(f"  {'Variable':<30s}  {'Coef':>10s}  {'SE':>8s}  "
                         f"{'z':>7s}  {'p':>8s}  Sig")
            lines.append("  " + "-" * 65)
            bm = self.coef[m]
            se_m = self.se[m] if len(self.se) > m else np.full(len(bm), np.nan)
            zs, ps = _z_p(bm, se_m)
            for i, nm in enumerate(self.feature_names[m]):
                z_s = f"{zs[i]:+7.3f}" if np.isfinite(zs[i]) else "     nan"
                p_s = f"{ps[i]:8.4f}" if np.isfinite(ps[i]) else "     nan"
                lines.append(
                    f"  {nm:<30s}  {bm[i]:+10.4f}  {se_m[i]:8.4f}  {z_s}  {p_s}  "
                    f"{_sig_stars(ps[i]) if np.isfinite(ps[i]) else ''}"
                )

        lines.append("\n  Copula Correlation Matrix (Sigma):")
        corr = self.correlation
        header = "  " + " " * 20 + "".join(f"{nm:>12s}" for nm in self.activity_names)
        lines.append(header)
        for i, nm in enumerate(self.activity_names):
            row = "  " + f"{nm:<20s}" + "".join(f"{corr[i, j]:+12.4f}" for j in range(self.M))
            lines.append(row)
        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════
# 7.  MultivariateCountRegressor — main class
# ═══════════════════════════════════════════════════════════════════════

class MultivariateCountRegressor:
    """
    Jointly estimate M count-data regression models (Poisson or NB) for
    multiple activity types using a copula to capture cross-activity dependence.

    Parameters
    ----------
    activity_names : list of str
        Names of the M activity types (e.g. ['work', 'shop', 'recreation']).
        Length determines M.
    copula : str
        Copula family for joint dependence:
        - ``'gaussian'``   M-variate Gaussian copula (default, recommended).
        - ``'vine-frank'`` C-vine with Frank bivariate copulas.
    marginal : str
        Marginal distribution: ``'nb'`` (negative binomial, default) or
        ``'poisson'``.
    maxiter : int
        Maximum L-BFGS-B iterations for the joint estimation step.
    stage1_maxiter : int
        Maximum iterations for stage-1 univariate fits.
    verbose : bool
        Print progress messages.

    Examples
    --------
    Predict work, shopping, and recreation trips jointly:

    >>> import numpy as np
    >>> from metacountregressor.multivariate_count_regressor import (
    ...     MultivariateCountRegressor,
    ... )
    >>> np.random.seed(42)
    >>> n, k = 500, 4
    >>> X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
    >>> true_betas = [np.array([0.3, 0.5, -0.2, 0.1]),
    ...               np.array([0.1, -0.3, 0.4, 0.0]),
    ...               np.array([0.6, 0.1, 0.0, -0.4])]
    >>> Y = np.column_stack([
    ...     np.random.poisson(np.exp(X @ b)) for b in true_betas
    ... ])
    >>> model = MultivariateCountRegressor(
    ...     activity_names=['work', 'shop', 'recreation'], marginal='poisson'
    ... )
    >>> fit = model.fit(X, Y)
    >>> print(fit.summary())
    >>> preds = fit.predict([X, X, X])  # (n, 3) predicted means
    """

    def __init__(
        self,
        activity_names: Optional[List[str]] = None,
        copula: str = "gaussian",
        marginal: str = "nb",
        maxiter: int = 1000,
        stage1_maxiter: int = 500,
        verbose: bool = True,
    ):
        if not _JAX_OK:
            raise ImportError("JAX is required for MultivariateCountRegressor. "
                              "Install with: pip install jax jaxlib")
        if not _SCIPY_OK:
            raise ImportError("SciPy is required. Install with: pip install scipy")

        self.activity_names = activity_names or []
        self.copula = copula.lower()
        self.marginal = marginal.lower()
        self.maxiter = maxiter
        self.stage1_maxiter = stage1_maxiter
        self.verbose = verbose
        self._fit: Optional[MultivariateCountFit] = None

    # ── fit ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: Union[np.ndarray, List[np.ndarray]],
        Y: np.ndarray,
        offsets: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        feature_names: Optional[List[List[str]]] = None,
        alpha_init: Optional[np.ndarray] = None,
        fix_alphas: bool = True,
    ) -> "MultivariateCountFit":
        """
        Fit the multivariate count model.

        Parameters
        ----------
        X : (n, k) array  OR  list of M arrays (n, k_m)
            Covariate matrix / matrices.  If a single array is supplied it is
            used for all M outcomes (shared covariates).
        Y : (n, M) array
            Observed activity counts.  Each column is one activity type.
        offsets : (n, M) array or list of (n,) arrays, optional
            Log-offsets (e.g. log-exposure).  Zeros if not supplied.
        feature_names : list of M lists of str, optional
            Names for regression coefficients.
        alpha_init : (M,) array, optional
            Starting values for overdispersion.  If None, fit stage-1 NB.
        fix_alphas : bool
            If True (default), fix alpha_m at stage-1 values during joint
            estimation (standard two-stage approach).

        Returns
        -------
        MultivariateCountFit
        """
        Y = np.asarray(Y, dtype=np.float64)
        n, M = Y.shape

        # ── Activity names ──────────────────────────────────────────────
        if not self.activity_names:
            self.activity_names = [f"activity_{m+1}" for m in range(M)]
        assert len(self.activity_names) == M, (
            f"activity_names has {len(self.activity_names)} entries but Y has {M} columns"
        )

        # ── Covariate matrices ──────────────────────────────────────────
        if isinstance(X, np.ndarray):
            X_list = [X.copy() for _ in range(M)]
        else:
            X_list = [np.asarray(x, dtype=np.float64) for x in X]
        assert len(X_list) == M

        ks = [x.shape[1] for x in X_list]

        # ── Offsets ─────────────────────────────────────────────────────
        if offsets is None:
            off_arr = np.zeros((n, M))
        elif isinstance(offsets, np.ndarray):
            off_arr = offsets.astype(np.float64)
        else:
            off_arr = np.column_stack([np.asarray(o, dtype=np.float64) for o in offsets])

        # ── Feature names ────────────────────────────────────────────────
        if feature_names is None:
            feature_names = [
                [f"x{j}" for j in range(ks[m])] for m in range(M)
            ]

        # ── Stage 1: Univariate NB fits ──────────────────────────────────
        if self.verbose:
            print(f"[MultivariateCountRegressor] Stage 1: fitting {M} univariate margins …")

        marginal_fits: List[dict] = []
        alphas = np.zeros(M)
        beta_init_list: List[np.ndarray] = []

        for m in range(M):
            if self.verbose:
                print(f"  Marginal {m+1}/{M}: {self.activity_names[m]} … ", end="", flush=True)

            if self.marginal == "nb":
                fit_m = fit_univariate_nb(
                    y=Y[:, m],
                    x=X_list[m],
                    offset=off_arr[:, m],
                    feature_names=feature_names[m],
                    maxiter=self.stage1_maxiter,
                )
                alphas[m] = fit_m["alpha"]
                beta_init_list.append(fit_m["coef"].copy())
            else:
                # Poisson: simple log-linear fit (alpha not needed)
                beta_init_list.append(np.zeros(ks[m]))
                alphas[m] = 1e-8  # near-zero overdispersion → Poisson
                fit_m = {"converged": True, "aic": np.nan, "bic": np.nan,
                         "loglik": np.nan, "alpha": 0.0, "coef": beta_init_list[m]}

            marginal_fits.append(fit_m)
            if self.verbose:
                conv = "OK" if fit_m.get("converged", False) else "WARN"
                print(f"{conv}")

        if alpha_init is not None:
            alphas = np.asarray(alpha_init, dtype=np.float64)

        # ── Stage 2: Joint copula estimation ───────────────────────────
        if self.verbose:
            print(f"[MultivariateCountRegressor] Stage 2: joint {self.copula} copula fit …")

        # Empirical correlation of residuals (for warm-starting copula)
        resid_arr = np.zeros((n, M))
        for m in range(M):
            mu_m = np.exp(X_list[m] @ beta_init_list[m] + off_arr[:, m])
            resid_arr[:, m] = Y[:, m] - mu_m
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                emp_corr = np.corrcoef(resid_arr.T)
                emp_corr = np.clip(emp_corr, -0.95, 0.95)
                np.fill_diagonal(emp_corr, 1.0)
            except Exception:
                emp_corr = np.eye(M)

        # Build copula parameter vector
        if self.copula == "gaussian":
            n_copula = M * (M + 1) // 2
            chol_init = np.zeros(n_copula)
            # Warm start from empirical correlation
            try:
                L_emp = np.linalg.cholesky(emp_corr + 1e-6 * np.eye(M))
                idx = np.tril_indices(M)
                chol_flat = L_emp[idx].copy()
                # Log-transform diagonal (to unconstrained parameterisation)
                diag_pos = [i * (i + 1) // 2 + i for i in range(M)]
                chol_flat[diag_pos] = np.log(np.clip(chol_flat[diag_pos], 1e-6, None))
                chol_init = chol_flat
            except np.linalg.LinAlgError:
                pass
        else:  # vine-frank
            n_copula = M * (M - 1) // 2
            # Warm-start rho from empirical correlation
            rho_init = []
            for i in range(M):
                for j in range(i + 1, M):
                    r = float(emp_corr[i, j])
                    rho_init.append(np.clip(r * 4.0, -8.0, 8.0))
            chol_init = np.array(rho_init)

        init_params = _pack_joint_params(beta_init_list, chol_init)
        a_jax = jnp.asarray(alphas)
        Y_jax = jnp.asarray(Y)

        # Build per-outcome JAX arrays
        X_jax = [jnp.asarray(x) for x in X_list]
        off_jax = jnp.asarray(off_arr)

        # ── Negative log-likelihood ────────────────────────────────────
        @jax.jit
        def neg_ll(params):
            betas_list_j, copula_p = _unpack_joint_params_jax(params, ks, n_copula)
            mus = jnp.stack(
                [jnp.exp(X_jax[m] @ betas_list_j[m] + off_jax[:, m])
                 for m in range(M)],
                axis=1,
            )
            if self.copula == "gaussian":
                ll = gaussian_copula_loglik(
                    Y_jax, mus, a_jax, copula_p, M, self.marginal
                )
            else:
                ll = vine_frank_copula_loglik(
                    Y_jax, mus, a_jax, copula_p, M, self.marginal
                )
            return -ll

        grad_fn = jax.jit(jax.grad(neg_ll))

        def obj(p):
            p_jax = jnp.asarray(p)
            return float(neg_ll(p_jax)), np.asarray(grad_fn(p_jax))

        # Bounds: copula params bounded for stability
        n_beta = sum(ks)
        if self.copula == "gaussian":
            # Off-diagonal Cholesky entries unbounded; diagonal > -inf (exp'd)
            bounds = [(None, None)] * n_beta + [(None, None)] * n_copula
        else:
            bounds = [(None, None)] * n_beta + [(-8.0, 8.0)] * n_copula

        res = _scipy_minimize(
            obj,
            init_params,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.maxiter, "ftol": 1e-10, "gtol": 1e-6},
        )

        params_opt = np.asarray(res.x)
        converged = bool(res.success)
        if self.verbose:
            status = "converged" if converged else f"stopped ({res.message})"
            print(f"  Joint optimisation {status}.")

        # ── Unpack results ─────────────────────────────────────────────
        betas_list_opt, copula_params_opt = _unpack_joint_params(params_opt, ks, n_copula)

        if self.copula == "gaussian":
            L_opt = _unpack_chol(jnp.asarray(copula_params_opt), M)
            corr_matrix = np.asarray(_chol_to_corr(L_opt))
        else:
            # Vine: build implied pairwise corr matrix
            corr_matrix = np.eye(M)
            idx = 0
            for i in range(M):
                for j in range(i + 1, M):
                    rho = float(copula_params_opt[idx])
                    idx += 1
                    # Frank rho → Kendall tau → Pearson (approx)
                    try:
                        tau = 1.0 - 4.0 / rho * (1.0 - _frank_debye(rho))
                    except Exception:
                        tau = 0.0
                    r = np.sin(np.pi / 2.0 * np.clip(tau, -0.99, 0.99))
                    corr_matrix[i, j] = r
                    corr_matrix[j, i] = r

        # ── Standard errors via Hessian ─────────────────────────────────
        try:
            se_full = _hessian_se(neg_ll, params_opt)
            splits = np.cumsum([0] + ks)
            se_list = [se_full[splits[m]: splits[m + 1]] for m in range(M)]
            se_copula = se_full[splits[-1]: splits[-1] + n_copula]
        except Exception:
            se_list = [np.full(ks[m], np.nan) for m in range(M)]
            se_copula = np.full(n_copula, np.nan)

        # ── Information criteria ────────────────────────────────────────
        ll_val = -float(neg_ll(jnp.asarray(params_opt)))
        k_total = len(params_opt)
        aic = 2.0 * k_total - 2.0 * ll_val
        bic = k_total * np.log(n) - 2.0 * ll_val

        if self.verbose:
            print(f"  Log-lik = {ll_val:.4f}  AIC = {aic:.2f}  BIC = {bic:.2f}")

        self._fit = MultivariateCountFit(
            activity_names=list(self.activity_names),
            M=M,
            n=n,
            k_total=k_total,
            coef=[np.asarray(b) for b in betas_list_opt],
            alphas=alphas.copy(),
            se=se_list,
            feature_names=feature_names,
            copula=self.copula,
            correlation=corr_matrix,
            copula_params_raw=copula_params_opt,
            loglik=ll_val,
            aic=aic,
            bic=bic,
            converged=converged,
            marginal=self.marginal,
            marginal_fits=marginal_fits,
            se_copula=se_copula,
        )
        return self._fit

    def predict(
        self,
        X: Union[np.ndarray, List[np.ndarray]],
        offsets: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict expected activity counts using the fitted model.

        Parameters
        ----------
        X : (n, k) or list of M arrays (n, k_m)
        offsets : (n, M) or None

        Returns
        -------
        preds : (n, M)
        """
        if self._fit is None:
            raise RuntimeError("Call fit() first.")
        if isinstance(X, np.ndarray):
            X_list = [X] * self._fit.M
        else:
            X_list = list(X)
        return self._fit.predict(X_list, offsets)

    def summary(self) -> str:
        if self._fit is None:
            return "Model not yet fitted. Call fit() first."
        return self._fit.summary()


# ═══════════════════════════════════════════════════════════════════════
# 8.  JAX-compatible unpack (used inside jit)
# ═══════════════════════════════════════════════════════════════════════

def _unpack_joint_params_jax(
    params: jnp.ndarray,
    ks: List[int],
    n_copula: int,
):
    """JAX-traceable version of _unpack_joint_params."""
    splits = [0] + list(np.cumsum(ks))
    betas_list = [params[splits[m]: splits[m + 1]] for m in range(len(ks))]
    copula_params = params[splits[-1]: splits[-1] + n_copula]
    return betas_list, copula_params


# ═══════════════════════════════════════════════════════════════════════
# 9.  Utility: Frank Debye function (for vine→Pearson conversion)
# ═══════════════════════════════════════════════════════════════════════

def _frank_debye(rho: float, n_terms: int = 50) -> float:
    """Debye function D_1(rho) = (1/rho) ∫_0^rho t/(e^t - 1) dt."""
    if abs(rho) < 1e-8:
        return 1.0
    t_vals = np.linspace(1e-8, abs(rho), n_terms)
    integrand = t_vals / (np.exp(t_vals) - 1.0)
    return float(np.trapz(integrand, t_vals) / abs(rho))


# ═══════════════════════════════════════════════════════════════════════
# 10.  Convenience wrapper: fit from a single DataFrame
# ═══════════════════════════════════════════════════════════════════════

def fit_multivariate_activity_model(
    df: "pd.DataFrame",
    activity_cols: List[str],
    covariate_cols: Union[List[str], Dict[str, List[str]]],
    offset_col: Optional[Union[str, Dict[str, str]]] = None,
    copula: str = "gaussian",
    marginal: str = "nb",
    add_intercept: bool = True,
    maxiter: int = 1000,
    verbose: bool = True,
) -> MultivariateCountFit:
    """
    Convenience function to fit a multivariate activity count model directly
    from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with one row per person.
    activity_cols : list of str
        Column names for each activity count (Y_1, …, Y_M).
    covariate_cols : list of str  OR  dict {activity_name: [col, …]}
        If a list: the same covariates are used for all activities.
        If a dict: activity-specific covariate lists.
    offset_col : str or dict, optional
        Column name for log-offset (e.g. 'log_exposure').
        If dict: activity-specific offset columns.
    copula : str
        'gaussian' (default) or 'vine-frank'.
    marginal : str
        'nb' (default) or 'poisson'.
    add_intercept : bool
        Prepend a constant column (default True).
    maxiter : int
        Optimisation iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    MultivariateCountFit
    """
    import pandas as pd

    M = len(activity_cols)
    n = len(df)
    Y = df[activity_cols].values.astype(float)

    # ── Build X_list ────────────────────────────────────────────────────
    X_list = []
    feat_names = []
    for m, act in enumerate(activity_cols):
        if isinstance(covariate_cols, dict):
            cols = covariate_cols.get(act, covariate_cols.get(activity_cols[m], []))
        else:
            cols = list(covariate_cols)

        X_m = df[cols].values.astype(float)
        if add_intercept:
            X_m = np.column_stack([np.ones(n), X_m])
            names_m = ["const"] + list(cols)
        else:
            names_m = list(cols)
        X_list.append(X_m)
        feat_names.append(names_m)

    # ── Build offsets ────────────────────────────────────────────────────
    if offset_col is None:
        off_arr = np.zeros((n, M))
    elif isinstance(offset_col, str):
        off_col = df[offset_col].values.astype(float)
        off_arr = np.column_stack([off_col] * M)
    else:
        off_arr = np.column_stack([
            df[offset_col.get(act, offset_col.get(activity_cols[m], None))
               if isinstance(offset_col, dict) else offset_col].values.astype(float)
            for m, act in enumerate(activity_cols)
        ])

    model = MultivariateCountRegressor(
        activity_names=activity_cols,
        copula=copula,
        marginal=marginal,
        maxiter=maxiter,
        verbose=verbose,
    )
    return model.fit(X_list, Y, off_arr, feature_names=feat_names)


# ═══════════════════════════════════════════════════════════════════════
# 11.  Quick self-test
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """Synthetic demonstration: 3 activities, 500 persons, shared covariates."""
    import numpy as np

    np.random.seed(2024)
    n, k = 500, 4
    M = 3
    activity_names = ["work_trips", "shop_trips", "recreation_trips"]

    X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
    true_betas = [
        np.array([0.30,  0.50, -0.20,  0.10]),
        np.array([0.10, -0.30,  0.40,  0.00]),
        np.array([0.60,  0.10,  0.00, -0.40]),
    ]
    true_alpha = [0.5, 0.3, 0.8]

    # Generate correlated counts through a Gaussian copula
    from scipy.stats import multivariate_normal, nbinom
    Sigma_true = np.array([[1.0, 0.4, -0.2],
                           [0.4, 1.0,  0.3],
                           [-0.2, 0.3, 1.0]])
    Z = multivariate_normal.rvs(mean=np.zeros(M), cov=Sigma_true, size=n)
    from scipy.stats import norm
    U = norm.cdf(Z)  # (n, M) uniform marginals

    Y = np.zeros((n, M), dtype=float)
    for m in range(M):
        mu_m = np.exp(X @ true_betas[m])
        r_m = 1.0 / true_alpha[m]
        p_m = r_m / (r_m + mu_m)
        Y[:, m] = nbinom.ppf(U[:, m], n=r_m, p=p_m)

    print(f"\n{'='*60}")
    print("  MULTIVARIATE ACTIVITY COUNT MODEL  –  DEMO")
    print(f"{'='*60}")
    print(f"  {n} persons, {M} activity types, {k} covariates\n")
    print("  Activity summary:")
    for m, nm in enumerate(activity_names):
        print(f"    {nm:25s}  mean={Y[:, m].mean():.2f}  max={int(Y[:, m].max())}")

    model = MultivariateCountRegressor(
        activity_names=activity_names,
        copula="gaussian",
        marginal="nb",
        maxiter=500,
        verbose=True,
    )
    fit = model.fit(X, Y, feature_names=[["const", "x1", "x2", "x3"]] * M)
    print(fit.summary())

    preds = fit.predict([X] * M)
    for m, nm in enumerate(activity_names):
        rmse = np.sqrt(np.mean((preds[:, m] - Y[:, m]) ** 2))
        print(f"  RMSE {nm:25s}: {rmse:.4f}")
    return fit


if __name__ == "__main__":
    _demo()
