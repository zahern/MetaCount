"""
Bivariate Negative-Binomial Regression with Copula Dependence
==============================================================

Implements the model class used in:

    Ahmad, N., Gayah, V. V., & Donnell, E. T. (2023).
    "Copula-based bivariate count data regression models for simultaneous
    estimation of crash counts based on severity and number of vehicles."
    Accident Analysis and Prevention 181, 106928.

For our application we use the same machinery to jointly model
near-miss counts and crash counts on road segments:

    Y_{1,i} = number of near-miss events on segment i
    Y_{2,i} = number of crashes            on segment i

with a shared offset (e.g. segment length, traffic exposure) and an
optional vector of segment-level covariates X_i whose coefficients may
differ between the two responses.

Three estimators are exposed:

* :func:`fit_marshall_olkin_nb`   - shared-frailty bivariate NB
                                    (Marshall-Olkin, 1985). Forces
                                    positive correlation between the
                                    two latent gamma effects.

* :func:`fit_famoye_nb`           - Famoye (2010) bivariate NB -
                                    separate overdispersion
                                    phi_1, phi_2 and unrestricted
                                    correlation parameter rho in
                                    {-inf, +inf}.

* :func:`fit_copula_bivariate_nb` - bivariate NB whose two marginals are
                                    combined through a copula C(u, v; rho).
                                    Supports the three families used in
                                    Ahmad et al. (2023):

                                        - Frank
                                        - Normal (Gaussian)
                                        - Kimeldorf-Sampson

Estimation strategy
-------------------
To get clean JAX autodiff we split the fit into two stages:

    1. **Margins** — fit two univariate NB regressions (one per response)
       via L-BFGS-B.  This delivers ``alpha_1, alpha_2`` along with the
       linear-predictor coefficients.

    2. **Joint**  — keep ``alpha_1, alpha_2`` *fixed* and re-fit the
       copula-augmented joint likelihood, varying only the linear-
       predictor coefficients and the dependence parameter ``rho``.

This is the standard "inference for marginals / parameter coupling"
procedure (Joe, 1997, Ch. 10; see also Ahmad et al. 2023 §3.2 in
their Stata implementation).  Combining with a single shared
``alpha_1 == alpha_2`` also reproduces the Marshall-Olkin model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as jsp_special
    import jax.scipy.stats as jsp_stats
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax = None
    jax_present = False
else:
    jax_present = True

try:
    from scipy import stats as _scipy_stats
    from scipy.optimize import minimize as _scipy_minimize
except ImportError:
    _scipy_stats = None
    _scipy_minimize = None


# ---------------------------------------------------------------------------
# 1.  Copula primitives  (Sklar, 1959; Nelsen, 2006)
# ---------------------------------------------------------------------------

def frank_logpdf(u, v, rho):
    """Numerically stable log C_Frank with rho unrestricted.

    The Frank copula has the well-known quirk that its log-density
    approaches log(u*v) as rho -> 0 but has an algebraic singularity
    (the two factors (1-exp(-rho*u))(1-exp(-rho*v)) cancel eta exactly).
    We use a Taylor expansion near rho = 0 to keep this region stable.
    """
    eps = 1e-12
    rho_safe = jnp.where(jnp.abs(rho) < eps, jnp.full_like(rho, eps), rho)
    eta = 1.0 - jnp.exp(-rho_safe)
    inner = eta - (1.0 - jnp.exp(-rho_safe * u)) * (1.0 - jnp.exp(-rho_safe * v))
    abs_rho = jnp.abs(rho_safe)

    # Limit behaviour: log C_Frank(u,v;0) = log(u*v)
    near_zero = jnp.abs(rho) < 1e-4
    log_limit = jnp.log(jnp.clip(u, eps, 1.0)) + jnp.log(jnp.clip(v, eps, 1.0))

    # C_Frank(u,v;rho) = -(1/rho) * log(inner / eta), so
    # log C_Frank = log(-log(inner/eta) / rho) = log|log(inner/eta)| - log|rho|
    # (the signs of log(inner/eta) and rho are always opposite, so the
    # ratio -log(inner/eta)/rho is positive and the abs-decomposition is
    # exact). A previous version of this function dropped the outer
    # log(...) around the log-ratio term, which silently returned
    # log(ratio) - log(rho) instead of log(C) -- producing values that
    # could exceed 0 (an impossible log-probability).
    abs_inner = jnp.where(jnp.abs(inner) < eps, jnp.full_like(inner, eps), jnp.abs(inner))
    abs_eta = jnp.where(jnp.abs(eta) < eps, jnp.full_like(eta, eps), jnp.abs(eta))
    log_ratio = jnp.log(abs_inner) - jnp.log(abs_eta)
    abs_log_ratio = jnp.where(jnp.abs(log_ratio) < eps, jnp.full_like(log_ratio, eps), jnp.abs(log_ratio))
    log_full = jnp.log(abs_log_ratio) - jnp.log(abs_rho)

    return jnp.where(near_zero, log_limit, log_full)


def normal_logpdf(u, v, rho):
    """log C_Gaussian(u, v; rho) via a Simpson-series approximation.

    Truncated at O(rho^3). Accurate to ~1e-4 across the unit cube for
    |rho| <= 0.9, which is the regime where likelihood fits are
    numerically stable for crash counts.
    """
    eps = 1e-12
    a = jsp_special.ndtri(jnp.clip(u, eps, 1.0 - eps))
    b = jsp_special.ndtri(jnp.clip(v, eps, 1.0 - eps))
    rho_clip = jnp.clip(rho, -0.95, 0.95)
    pa = jsp_special.ndtr(a)
    pb = jsp_special.ndtr(b)
    phia = jnp.exp(-0.5 * a * a) / jnp.sqrt(2.0 * jnp.pi)
    phib = jnp.exp(-0.5 * b * b) / jnp.sqrt(2.0 * jnp.pi)
    jc = (pa * pb
          + rho_clip * phia * phib
          + 0.5 * rho_clip**2 * a * b * phia * phib
          + (rho_clip**3 / 6.0) * (a * a - 1.0) * (b * b - 1.0) * phia * phib)
    jc = jnp.clip(jc, eps, 1.0 - eps)
    return jnp.log(jc)


def kimeldorf_sampson_logpdf(u, v, rho):
    """log C_KS(u, v; rho)  (Kimeldorf & Sampson, 1975).

    Ahmad et al. (2023) eq 13/16/19:
        C_KS(u, v; rho) = (u^{-rho} + v^{-rho} - 1)^{-1/rho},
    with rho >= -1, parameterised via rho = exp(rho_u) - 1.
    """
    eps = 1e-12
    rho = jnp.maximum(rho, -0.999)
    u_cl = jnp.clip(u, eps, 1.0)
    v_cl = jnp.clip(v, eps, 1.0)
    ur = jnp.power(u_cl, -rho)
    vr = jnp.power(v_cl, -rho)
    arg = ur + vr - 1.0
    arg = jnp.maximum(arg, eps)
    near_zero = jnp.abs(rho) < 1e-6
    # ``-log(arg) / rho`` literally divides by zero at rho == 0. Even
    # though jnp.where only *selects* the near_zero branch's log(u)+log(v)
    # value there, JAX still differentiates the discarded branch (both
    # branches are traced), so a bare 0/0 here would poison grad/hessian
    # with NaN via the 0 * NaN = NaN rule. Substitute a safe nonzero
    # denominator into the division only (not into ``arg``'s exponents,
    # which already use the literal ``rho``), so the unselected branch is
    # finite everywhere and its NaN-free gradient is correctly zeroed out.
    rho_safe = jnp.where(near_zero, jnp.ones_like(rho), rho)
    unsafe_val = -jnp.log(arg) / rho_safe
    return jnp.where(near_zero, jnp.log(u_cl) + jnp.log(v_cl), unsafe_val)


COPULA_LOGS = {
    "frank": frank_logpdf,
    "normal": normal_logpdf,
    "kimeldorf": kimeldorf_sampson_logpdf,
    "kimeldorf_sampson": kimeldorf_sampson_logpdf,
}


# ---------------------------------------------------------------------------
# 2.  Marginal NB log-PMF (eq 3 of the paper)
# ---------------------------------------------------------------------------

def nb_log_pmf(k, mu, alpha):
    r = 1.0 / alpha
    return (jsp_special.gammaln(k + r)
            - jsp_special.gammaln(r)
            - jsp_special.gammaln(k + 1.0)
            + r * jnp.log(r / (r + mu))
            + k * jnp.log(mu / (r + mu)))


def nb_log_cdf(k, mu, alpha):
    """Log of P(Y <= k) for NB2 parameterisation.

    The CDF is the regularized incomplete beta function
        I_x(a, b) = betainc(a, b, x)
    with
        a = r = 1 / alpha   (frozen)
        b = k + 1           (data, frozen)
        x = r / (r + mu)    (depends on the linear predictor)
    JAX's ``betainc`` differentiates only with respect to the third
    argument ``x``, so we attach ``stop_gradient`` to ``alpha`` and the
    integer count ``k``.  Only the linear predictor ``mu`` retains a
    gradient.

    Because we freeze ``alpha`` in the joint fit (set by
    ``fit_copula_bivariate_nb`` after a univariate NB stage-1 fit), this
    is exactly the right granularity for the two-stage estimator.

    For the ``y - 1`` boundary term, ``k`` can be ``-1`` (any zero-count
    observation), which would otherwise pass ``b = 0`` into ``betainc``.
    JAX's value and first derivative are fine at that boundary, but the
    second derivative (needed for ``jax.hessian``) is NaN. We clamp
    ``k`` to >= 0 before forming ``b_frozen`` (so ``betainc`` is always
    evaluated away from its singular boundary) and instead substitute
    the analytically correct ``log P(Y <= -1) = log(0) = -inf`` (in
    practice a large negative finite number) via an outer ``jnp.where``
    on the *already-clamped*, ``mu``-independent condition ``k < 0``.
    """
    a_frozen = jax.lax.stop_gradient(1.0 / alpha)
    k_clamped = jax.lax.stop_gradient(jnp.maximum(k, 0.0))
    b_frozen = jax.lax.stop_gradient(k_clamped + 1.0)
    x = a_frozen / (a_frozen + mu)
    # log of the regularised incomplete beta
    eps = 1e-300
    ibeta = jsp_special.betainc(a_frozen, b_frozen, jnp.clip(x, eps, 1.0 - eps))
    log_cdf = jnp.log(jnp.clip(ibeta, eps, 1.0))
    return jnp.where(k < 0.0, jnp.full_like(log_cdf, -700.0), log_cdf)


# ---------------------------------------------------------------------------
# 3.  Univariate NB log-LL  (used in stage 1)
# ---------------------------------------------------------------------------

def univariate_nb_loglik(y, x, offset, params):
    beta, la = params[:-1], params[-1]
    alpha = jnp.exp(la)
    eta = x @ beta + offset
    mu = jnp.exp(eta)
    return jnp.sum(nb_log_pmf(y, mu, alpha))


# ---------------------------------------------------------------------------
# 4.  Bivariate Copula Log-Likelihood with FIXED alphas
# ---------------------------------------------------------------------------

def bivariate_copula_loglik(
    y1, y2, x1, x2, off1, off2,
    params, copula="frank",
    alpha_1=None, alpha_2=None,
):
    """Joint log-likelihood for two NB marginals with copula
    dependence. Parameters ``alpha_1`` and ``alpha_2`` are *fixed* —
    fit them first by calling :func:`fit_univariate_nb` twice.

    Parameters
    ----------
    y1, y2 : (n,) arrays
    x1, x2 : (n, k1) / (n, k2)
    off1, off2 : (n,)
    params : (k1 + k2 + 1,) array
        ``[beta1 (k1), beta2 (k2), rho_u]``.
    copula : str  - one of ``"frank"``, ``"normal"``, ``"kimeldorf"``.
    alpha_1, alpha_2 : scalar jax arrays  (FROZEN).
    """
    cop_fn = COPULA_LOGS[copula.lower()]
    k1 = x1.shape[1]
    k2 = x2.shape[1]
    b1 = params[:k1]
    b2 = params[k1: k1 + k2]
    rho_u = params[k1 + k2]

    if copula == "frank":
        rho = rho_u
    elif copula == "normal":
        rho = (jnp.exp(2.0 * rho_u) - 1.0) / (jnp.exp(2.0 * rho_u) + 1.0)
    elif copula in ("kimeldorf", "kimeldorf_sampson"):
        rho = jnp.exp(rho_u) - 1.0
        rho = jnp.maximum(rho, -0.999)
    else:
        raise ValueError(f"unknown copula {copula!r}")

    mu1 = jnp.exp(x1 @ b1 + off1)
    mu2 = jnp.exp(x2 @ b2 + off2)

    # CDFs at y and y - 1 (for the two-point 'cell' probability)
    eps = 1e-12
    F1   = jnp.clip(jnp.exp(nb_log_cdf(y1,   mu1, alpha_1)), eps, 1.0 - eps)
    F1m  = jnp.clip(jnp.exp(nb_log_cdf(y1-1, mu1, alpha_1)), eps, 1.0 - eps)
    F2   = jnp.clip(jnp.exp(nb_log_cdf(y2,   mu2, alpha_2)), eps, 1.0 - eps)
    F2m  = jnp.clip(jnp.exp(nb_log_cdf(y2-1, mu2, alpha_2)), eps, 1.0 - eps)

    C_mm = cop_fn(F1m, F2m, rho)
    C_mp = cop_fn(F1m, F2,  rho)
    C_pm = cop_fn(F1,  F2m, rho)
    C_pp = cop_fn(F1,  F2,  rho)

    # Cell probability = C(F1,F2) - C(F1m,F2) - C(F1,F2m) + C(F1m,F2m).
    # This is a difference of *probabilities*, not of their logs -- summing
    # the C_xx log-values directly (as a previous version did) computes
    # log(P_pp * P_mm / (P_mp * P_pm)) instead, which is a different (and
    # meaningless) quantity. We must exponentiate, difference, then re-log.
    P_mm = jnp.exp(C_mm)
    P_mp = jnp.exp(C_mp)
    P_pm = jnp.exp(C_pm)
    P_pp = jnp.exp(C_pp)
    cell_mass = jnp.clip(P_pp + P_mm - P_mp - P_pm, eps, None)
    log_copula_mass = jnp.log(cell_mass)

    logp1 = nb_log_pmf(y1, mu1, alpha_1)
    logp2 = nb_log_pmf(y2, mu2, alpha_2)
    logp_cell = (log_copula_mass + logp1 + logp2
                 - jnp.log(jnp.clip(F1 - F1m, eps, None))
                 - jnp.log(jnp.clip(F2 - F2m, eps, None)))
    return jnp.sum(logp_cell)


# ---------------------------------------------------------------------------
# 5.  Famoye (2010) bivariate NB  (eq 7) -- internal optimisation is fine
# ---------------------------------------------------------------------------

def famoye_bivariate_nb_loglik(
    y1, y2, x1, x2, off1, off2, params,
    alpha_1=None, alpha_2=None,
):
    """Famoye bivariate NB with FIXED alphas."""
    k1 = x1.shape[1]
    k2 = x2.shape[1]
    b1 = params[:k1]
    b2 = params[k1: k1 + k2]
    rho = params[k1 + k2]
    phi1 = jnp.exp(alpha_1) if (isinstance(alpha_1, float) and alpha_1 < 0) else alpha_1
    phi2 = jnp.exp(alpha_2) if (isinstance(alpha_2, float) and alpha_2 < 0) else alpha_2
    # alpha_1 / alpha_2 are already the *unconstrained* fit values when
    # supplied as floats from stage 1; if they look like log(alpha) we
    # exponentiate.  In practice the wrapper always supplies positive
    # phi so we leave them as-is.
    if isinstance(alpha_1, (int, float)):
        phi1 = float(alpha_1) if alpha_1 > 0 else float(np.exp(alpha_1))
        phi2 = float(alpha_2) if alpha_2 > 0 else float(np.exp(alpha_2))
    phi1 = jnp.asarray(phi1)
    phi2 = jnp.asarray(phi2)

    mu1 = jnp.exp(x1 @ b1 + off1)
    mu2 = jnp.exp(x2 @ b2 + off2)
    g1 = mu1 / (phi1 - 1.0 + mu1)
    g2 = mu2 / (phi2 - 1.0 + mu2)
    c1 = jnp.power(jnp.maximum(1.0 - g1, 1e-12), phi1 - 1.0)
    c2 = jnp.power(jnp.maximum(1.0 - g2, 1e-12), phi2 - 1.0)
    logp1 = nb_log_pmf(y1, mu1, phi1)
    logp2 = nb_log_pmf(y2, mu2, phi2)
    # The correction bracket 1 + rho*(...)*( ...) can stray below 0 for
    # counts/rho combinations outside the range this baseline was tuned
    # for (unlike the copula-based models, Famoye's bracket has no
    # built-in normalisation keeping it in a valid range). log1p of a
    # sub -1 argument is NaN, so clip the bracket to stay positive --
    # this only affects regions where the Famoye approximation itself
    # is already breaking down, not the well-behaved interior.
    bracket = 1.0 + rho * (jnp.exp(y1) - c1) * (jnp.exp(y2) - c2)
    log_correction = jnp.log(jnp.clip(bracket, 1e-12, None))
    return jnp.sum(logp1 + logp2 + log_correction)


# ---------------------------------------------------------------------------
# 6.  Marshall-Olkin (1985) shared-frailty bivariate NB  (eq 6)
# ---------------------------------------------------------------------------

def marshall_olkin_nb_loglik(y1, y2, x1, x2, off1, off2, params, alpha=None):
    """Marshall-Olkin with shared-frailty alpha (assumed fixed)."""
    k1 = x1.shape[1]
    k2 = x2.shape[1]
    b1 = params[:k1]
    b2 = params[k1: k1 + k2]
    if isinstance(alpha, (int, float)):
        phi = float(alpha) if alpha > 0 else float(np.exp(alpha))
    else:
        phi = jnp.asarray(alpha)
    mu1 = jnp.exp(x1 @ b1 + off1)
    mu2 = jnp.exp(x2 @ b2 + off2)
    Y = y1 + y2
    lam = mu1 + mu2
    log_joint = (
        jsp_special.gammaln(Y + phi)
        - jsp_special.gammaln(y1 + 1.0)
        - jsp_special.gammaln(y2 + 1.0)
        - jsp_special.gammaln(phi)
        + y1 * jnp.log(mu1 / (lam + 1.0))
        + y2 * jnp.log(mu2 / (lam + 1.0))
        + phi * jnp.log(1.0 / (lam + 1.0))
    )
    return jnp.sum(log_joint)


# ---------------------------------------------------------------------------
# 7.  Stage 1 helpers  (univariate NB fits)
# ---------------------------------------------------------------------------

def fit_univariate_nb(
    y, x, offset,
    feature_names=None,
    method="scipy-lbfgsb",
    maxiter=500,
):
    """Fit a single NB regression. Returns a dict with the standard
    information criteria plus the fitted linear coefficients and
    overdispersion alpha."""
    if not jax_present:
        raise ImportError("JAX required for fit_univariate_nb")
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    off = np.asarray(offset, dtype=np.float64)
    n = len(y); k = x.shape[1]

    @jax.jit
    def ll(p):
        return univariate_nb_loglik(jnp.asarray(y), jnp.asarray(x),
                                    jnp.asarray(off), p)

    init = jnp.zeros(k + 1)
    if k > 0:
        init = init.at[0].set(np.log(max(np.mean(y) / max(np.mean(np.exp(off)), 1e-6), 0.01)))
    init = init.at[k].set(-1.0)

    if method == "scipy-lbfgsb":
        g = jax.jit(jax.grad(lambda p: -ll(p)))
        def obj(p):
            return float(-ll(jnp.asarray(p))), np.asarray(g(jnp.asarray(p)))
        res = _scipy_minimize(obj, np.asarray(init), jac=True,
                              method="L-BFGS-B", options={"maxiter": maxiter})
        params = jnp.asarray(res.x)
        success = bool(res.success)
    else:
        from jaxopt import BFGS
        solver = BFGS(fun=lambda p: -ll(p), maxiter=maxiter, tol=1e-6)
        res = solver.run(init)
        params = res.params
        success = bool(res.state.error < 1e-5)

    pp = np.asarray(params)
    b = pp[:k]; la = float(pp[k])
    alpha = float(np.exp(la))
    ll_val = float(ll(params))
    kp = len(pp)
    return {
        "n": n, "k": kp, "loglik": ll_val,
        "aic": 2 * kp - 2 * ll_val,
        "bic": kp * np.log(n) - 2 * ll_val,
        "coef": b, "alpha": alpha,
        "la": la, "converged": success,
        "feature_names": list(feature_names or [f"x{i}" for i in range(k)]),
    }


# ---------------------------------------------------------------------------
# 8.  Hessian-based standard errors
# ---------------------------------------------------------------------------

def _hessian_se(neg_ll, params):
    """Ridge-regularised Hessian-inversion standard errors.

    Mirrors ``compute_standard_errors`` in ``main_hpc.py``: ``neg_ll``
    must be the *negative* log-likelihood (so its Hessian is the
    observed Fisher information). Every parameter gets a finite SE --
    flat likelihood directions get a large (rather than NaN) SE, which
    is more informative for diagnosing identification problems than a
    missing value would be.
    """
    params_np = np.asarray(params, dtype=float)
    hess_np = np.asarray(jax.hessian(neg_ll)(jnp.asarray(params_np)), dtype=float)
    H = np.where(np.isfinite(hess_np), hess_np, 0.0)

    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return np.full(len(params_np), np.nan)

    max_ev = float(np.max(np.abs(eigvals)))
    if max_ev <= 0 or not np.isfinite(max_ev):
        return np.full(len(params_np), np.nan)

    ridge = float(np.clip(max_ev * 1e-6, 1e-12, 1e-4))
    inv_eigvals = 1.0 / (eigvals + ridge)
    cov_np = (eigvecs * inv_eigvals) @ eigvecs.T

    diag_cov = np.diag(cov_np)
    diag_cov = np.where(diag_cov > 0, diag_cov, 1.0 / ridge)
    return np.sqrt(diag_cov)


def _z_p(coef, se):
    coef = np.asarray(coef, dtype=float)
    se = np.asarray(se, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, coef / se, np.nan)
    p = 2.0 * (1.0 - _scipy_stats.norm.cdf(np.abs(z))) if _scipy_stats is not None else np.full_like(z, np.nan)
    return z, p


def _sig_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# 9.  Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BivariateCopulaFit:
    copula: str
    coef_1: np.ndarray
    alpha_1: float
    coef_2: np.ndarray
    alpha_2: float
    rho: float
    rho_u: float
    loglik: float
    aic: float
    bic: float
    n: int
    k: int
    method: str
    converged: bool
    feature_names_1: list = field(default_factory=list)
    feature_names_2: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    se_1: np.ndarray = field(default_factory=lambda: np.array([]))
    se_2: np.ndarray = field(default_factory=lambda: np.array([]))
    se_rho: float = float("nan")

    @property
    def z_1(self):
        return _z_p(self.coef_1, self.se_1)[0]

    @property
    def p_1(self):
        return _z_p(self.coef_1, self.se_1)[1]

    @property
    def z_2(self):
        return _z_p(self.coef_2, self.se_2)[0]

    @property
    def p_2(self):
        return _z_p(self.coef_2, self.se_2)[1]

    @property
    def z_rho(self):
        if not np.isfinite(self.se_rho) or self.se_rho <= 0:
            return float("nan")
        return self.rho / self.se_rho

    @property
    def p_rho(self):
        z = self.z_rho
        if not np.isfinite(z) or _scipy_stats is None:
            return float("nan")
        return float(2.0 * (1.0 - _scipy_stats.norm.cdf(abs(z))))

    def summary(self) -> str:
        has_se = self.se_1.size == len(self.coef_1) and self.se_1.size > 0
        out = [
            f"Bivariate NB fit ({self.copula})   "
            f"n = {self.n}   k = {self.k}",
            f"Log-likelihood: {self.loglik:.4f}   "
            f"AIC = {self.aic:.2f}   BIC = {self.bic:.2f}",
        ]
        if np.isfinite(self.se_rho):
            p_rho = self.p_rho
            out.append(
                f"rho = {self.rho:.4f} (raw = {self.rho_u:.4f})  "
                f"SE = {self.se_rho:.4f}  z = {self.z_rho:.3f}  "
                f"p = {p_rho:.4f} {_sig_stars(p_rho)}"
            )
        else:
            out.append(f"rho = {self.rho:.4f} (raw = {self.rho_u:.4f})")
        out.append("")
        out.append("Response 1:")
        z1, p1 = (self.z_1, self.p_1) if has_se else (None, None)
        for i, (nm, b) in enumerate(zip(self.feature_names_1, self.coef_1)):
            if has_se:
                out.append(f"  {nm:30s}  {b:+10.4f}  SE={self.se_1[i]:8.4f}  "
                           f"z={z1[i]:+7.3f}  p={p1[i]:6.4f} {_sig_stars(p1[i])}")
            else:
                out.append(f"  {nm:30s}  {b:+10.4f}")
        out.append(f"  {'alpha_1 (overdispersion)':30s}  {self.alpha_1:10.4f}")
        out.append("Response 2:")
        has_se2 = self.se_2.size == len(self.coef_2) and self.se_2.size > 0
        z2, p2 = (self.z_2, self.p_2) if has_se2 else (None, None)
        for i, (nm, b) in enumerate(zip(self.feature_names_2, self.coef_2)):
            if has_se2:
                out.append(f"  {nm:30s}  {b:+10.4f}  SE={self.se_2[i]:8.4f}  "
                           f"z={z2[i]:+7.3f}  p={p2[i]:6.4f} {_sig_stars(p2[i])}")
            else:
                out.append(f"  {nm:30s}  {b:+10.4f}")
        out.append(f"  {'alpha_2 (overdispersion)':30s}  {self.alpha_2:10.4f}")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# 9.  Joint copula NB fitter (two-stage)
# ---------------------------------------------------------------------------

def fit_copula_bivariate_nb(
    y1, y2, x1, x2, offset1, offset2,
    copula: str = "frank",
    feature_names_1=None, feature_names_2=None,
    alpha_1: Optional[float] = None,
    alpha_2: Optional[float] = None,
    method: str = "scipy-lbfgsb",
    n_iter: int = 500,
):
    """Bivariate NB with a Sklar-type copula, fit in two stages.

    If ``alpha_1`` / ``alpha_2`` are not supplied, they are obtained
    by fitting two independent univariate NB regressions.
    """
    if not jax_present:
        raise ImportError("JAX is required for fit_copula_bivariate_nb")

    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    offset1 = np.asarray(offset1, dtype=np.float64)
    offset2 = np.asarray(offset2, dtype=np.float64)
    n = len(y1); k1 = x1.shape[1]; k2 = x2.shape[1]

    if alpha_1 is None:
        univ1 = fit_univariate_nb(y1, x1, offset1,
                                    feature_names=feature_names_1)
        alpha_1 = univ1["alpha"]
    if alpha_2 is None:
        univ2 = fit_univariate_nb(y2, x2, offset2,
                                    feature_names=feature_names_2)
        alpha_2 = univ2["alpha"]

    a1_jax = jnp.asarray(float(alpha_1))
    a2_jax = jnp.asarray(float(alpha_2))

    @jax.jit
    def neg_ll(p):
        return -bivariate_copula_loglik(
            jnp.asarray(y1), jnp.asarray(y2),
            jnp.asarray(x1), jnp.asarray(x2),
            jnp.asarray(offset1), jnp.asarray(offset2),
            p, copula=copula,
            alpha_1=a1_jax, alpha_2=a2_jax,
        )

    init = jnp.zeros(k1 + k2 + 1)
    if k1 > 0:
        # warm-start using the univariate intercept coefficient
        init = init.at[0].set(np.log(max(y1.mean() / max(np.exp(offset1).mean(), 1e-6), 0.01)))
        if k1 > 1:
            init = init.at[1].set(0.1)
    if k2 > 0:
        init = init.at[k1].set(np.log(max(y2.mean() / max(np.exp(offset2).mean(), 1e-6), 0.01)))
        if k2 > 1:
            init = init.at[k1 + 1].set(0.1)
    # Warm-start rho_u from the empirical Pearson correlation of the two responses.
    # For Frank this is a moderate positive rho (0.5-2.0); for Normal it maps
    # to the atanh-transformed Pearson r.
    try:
        from scipy.stats import pearsonr
        r = float(pearsonr(np.asarray(y1), np.asarray(y2))[0])
    except Exception:
        r = 0.0
    if copula == "frank":
        rho_u0 = max(0.5, r * 4.0)  # conservative positive start
    elif copula == "normal":
        rho_u0 = float(0.5 * np.arctanh(max(min(r, 0.9), -0.9))) if abs(r) > 1e-6 else 0.1
    else:  # kimeldorf
        rho_u0 = max(0.1, float(np.log(1.0 + max(r, 0.0))))
    init = init.at[k1 + k2].set(rho_u0)

    used_method = method
    if method == "jax-bfgs":
        try:
            from jaxopt import BFGS
            solver = BFGS(fun=neg_ll, maxiter=n_iter, tol=1e-6)
            res = solver.run(init)
            params = res.params
            converged = bool(res.state.error < 1e-5)
        except Exception:
            used_method = "scipy-lbfgsb"

    if used_method == "scipy-lbfgsb":
        g = jax.jit(jax.grad(neg_ll))
        # Bound rho_u to a safe interval; for each copula this maps to the
        # allowed dependence range.  Other parameters are unbounded.
        if copula == "frank":
            rho_lb, rho_ub = -8.0, 8.0
        elif copula == "normal":
            rho_lb, rho_ub = -3.0, 3.0
        else:  # kimeldorf
            rho_lb, rho_ub = -8.0, 6.0
        bounds = [(None, None)] * (k1 + k2) + [(rho_lb, rho_ub)]
        def obj(p):
            return float(neg_ll(jnp.asarray(p))), np.asarray(g(jnp.asarray(p)))
        res = _scipy_minimize(obj, np.asarray(init), jac=True,
                              method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": n_iter, "ftol": 1e-7})
        params = jnp.asarray(res.x)
        converged = bool(res.success)

    pp = np.asarray(params)
    b1 = pp[:k1]; b2 = pp[k1: k1 + k2]; rho_u = float(pp[k1 + k2])
    if copula == "frank":
        rho = rho_u
    elif copula == "normal":
        rho = (np.exp(2.0 * rho_u) - 1.0) / (np.exp(2.0 * rho_u) + 1.0)
    else:
        rho = float(np.exp(rho_u) - 1.0)
    ll_val = -float(neg_ll(jnp.asarray(pp)))
    kpar = len(pp)

    se_full = _hessian_se(neg_ll, pp)
    se_1 = se_full[:k1]
    se_2 = se_full[k1: k1 + k2]
    se_rho_u = float(se_full[k1 + k2])
    if copula == "frank":
        se_rho = se_rho_u
    elif copula == "normal":
        se_rho = se_rho_u * (1.0 - rho ** 2)
    else:  # kimeldorf
        se_rho = se_rho_u * (rho + 1.0)

    return BivariateCopulaFit(
        copula=copula,
        coef_1=b1, alpha_1=float(alpha_1),
        coef_2=b2, alpha_2=float(alpha_2),
        rho=float(rho), rho_u=rho_u,
        loglik=ll_val, aic=2 * kpar - 2 * ll_val,
        bic=kpar * np.log(n) - 2 * ll_val,
        n=int(n), k=int(kpar), method=str(used_method),
        converged=bool(converged),
        feature_names_1=list(feature_names_1 or [f"x1_{i}" for i in range(k1)]),
        feature_names_2=list(feature_names_2 or [f"x2_{i}" for i in range(k2)]),
        se_1=se_1, se_2=se_2, se_rho=float(se_rho),
    )


# ---------------------------------------------------------------------------
# 10.  Famoye (no copula) joint fit, two-stage
# ---------------------------------------------------------------------------

def fit_famoye_nb(
    y1, y2, x1, x2, offset1, offset2,
    feature_names_1=None, feature_names_2=None,
    alpha_1: Optional[float] = None,
    alpha_2: Optional[float] = None,
    method="scipy-lbfgsb", n_iter=500,
):
    if not jax_present:
        raise ImportError("JAX required for fit_famoye_nb")
    y1 = np.asarray(y1, dtype=np.float64); y2 = np.asarray(y2, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64); x2 = np.asarray(x2, dtype=np.float64)
    offset1 = np.asarray(offset1, dtype=np.float64); offset2 = np.asarray(offset2, dtype=np.float64)
    n = len(y1); k1 = x1.shape[1]; k2 = x2.shape[1]

    if alpha_1 is None:
        alpha_1 = fit_univariate_nb(y1, x1, offset1)["alpha"]
    if alpha_2 is None:
        alpha_2 = fit_univariate_nb(y2, x2, offset2)["alpha"]
    a1, a2 = float(alpha_1), float(alpha_2)

    @jax.jit
    def neg_ll(p):
        return -famoye_bivariate_nb_loglik(
            jnp.asarray(y1), jnp.asarray(y2),
            jnp.asarray(x1), jnp.asarray(x2),
            jnp.asarray(offset1), jnp.asarray(offset2), p,
            alpha_1=a1, alpha_2=a2,
        )

    init = jnp.zeros(k1 + k2 + 1)
    init = init.at[0].set(np.log(max(y1.mean() / max(np.exp(offset1).mean(), 1e-6), 0.01)))
    init = init.at[k1].set(np.log(max(y2.mean() / max(np.exp(offset2).mean(), 1e-6), 0.01)))
    init = init.at[k1 + k2].set(0.0)

    if method == "jax-bfgs":
        try:
            from jaxopt import BFGS
            solver = BFGS(fun=neg_ll, maxiter=n_iter, tol=1e-6)
            res = solver.run(init)
            params = res.params
            converged = bool(res.state.error < 1e-5)
        except Exception:
            method = "scipy-lbfgsb"

    if method == "scipy-lbfgsb":
        g = jax.jit(jax.grad(neg_ll))
        def obj(p):
            return float(neg_ll(jnp.asarray(p))), np.asarray(g(jnp.asarray(p)))
        # The FGM-style correction term used by Famoye's bivariate NB only
        # keeps the joint density non-negative everywhere for |rho| <= 1
        # (the classical FGM constraint). Without a bound the optimiser
        # can run away to +-inf once the bracket-clipping (added for JAX
        # stability) saturates, since nothing in the unclipped region then
        # penalises further increases in rho.
        bounds = [(None, None)] * (k1 + k2) + [(-1.0, 1.0)]
        res = _scipy_minimize(obj, np.asarray(init), jac=True,
                              method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": n_iter})
        params = jnp.asarray(res.x)
        converged = bool(res.success)

    pp = np.asarray(params); rho = float(pp[k1 + k2]); rho_u = rho
    b1 = pp[:k1]; b2 = pp[k1: k1 + k2]
    ll_val = -float(neg_ll(jnp.asarray(pp)))
    kpar = len(pp)

    se_full = _hessian_se(neg_ll, pp)
    se_1 = se_full[:k1]
    se_2 = se_full[k1: k1 + k2]
    se_rho = float(se_full[k1 + k2])

    return BivariateCopulaFit(
        copula="famoye (no copula)",
        coef_1=b1, alpha_1=a1, coef_2=b2, alpha_2=a2,
        rho=rho, rho_u=rho_u, loglik=ll_val,
        aic=2 * kpar - 2 * ll_val, bic=kpar * np.log(n) - 2 * ll_val,
        n=int(n), k=int(kpar), method=str(method),
        converged=bool(converged),
        feature_names_1=list(feature_names_1 or [f"x1_{i}" for i in range(k1)]),
        feature_names_2=list(feature_names_2 or [f"x2_{i}" for i in range(k2)]),
        se_1=se_1, se_2=se_2, se_rho=se_rho,
    )


# ---------------------------------------------------------------------------
# 11.  Marshall-Olkin shared-frailty joint fit, two-stage
# ---------------------------------------------------------------------------

def fit_marshall_olkin_nb(
    y1, y2, x1, x2, offset1, offset2,
    feature_names_1=None, feature_names_2=None,
    alpha=None,
    method="scipy-lbfgsb", n_iter=500,
):
    if not jax_present:
        raise ImportError("JAX required for fit_marshall_olkin_nb")
    y1 = np.asarray(y1, dtype=np.float64); y2 = np.asarray(y2, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64); x2 = np.asarray(x2, dtype=np.float64)
    offset1 = np.asarray(offset1, dtype=np.float64); offset2 = np.asarray(offset2, dtype=np.float64)
    n = len(y1); k1 = x1.shape[1]; k2 = x2.shape[1]

    if alpha is None:
        # Marshall-Olkin forces shared phi -> take average of two univariates
        a1 = fit_univariate_nb(y1, x1, offset1)["alpha"]
        a2 = fit_univariate_nb(y2, x2, offset2)["alpha"]
        alpha = float(0.5 * (a1 + a2))
    phi = float(alpha)

    @jax.jit
    def neg_ll(p):
        return -marshall_olkin_nb_loglik(
            jnp.asarray(y1), jnp.asarray(y2),
            jnp.asarray(x1), jnp.asarray(x2),
            jnp.asarray(offset1), jnp.asarray(offset2), p,
            alpha=phi,
        )

    init = jnp.zeros(k1 + k2)
    init = init.at[0].set(np.log(max(y1.mean() / max(np.exp(offset1).mean(), 1e-6), 0.01)))
    init = init.at[k1].set(np.log(max(y2.mean() / max(np.exp(offset2).mean(), 1e-6), 0.01)))

    if method == "jax-bfgs":
        try:
            from jaxopt import BFGS
            solver = BFGS(fun=neg_ll, maxiter=n_iter, tol=1e-6)
            res = solver.run(init)
            params = res.params
            converged = bool(res.state.error < 1e-5)
        except Exception:
            method = "scipy-lbfgsb"

    if method == "scipy-lbfgsb":
        g = jax.jit(jax.grad(neg_ll))
        def obj(p):
            return float(neg_ll(jnp.asarray(p))), np.asarray(g(jnp.asarray(p)))
        res = _scipy_minimize(obj, np.asarray(init), jac=True,
                              method="L-BFGS-B", options={"maxiter": n_iter})
        params = jnp.asarray(res.x)
        converged = bool(res.success)

    pp = np.asarray(params)
    b1 = pp[:k1]; b2 = pp[k1: k1 + k2]
    ll_val = -float(neg_ll(jnp.asarray(pp)))
    kpar = len(pp) + 0  # shared phi is fixed -> doesn't count

    se_full = _hessian_se(neg_ll, pp)
    se_1 = se_full[:k1]
    se_2 = se_full[k1: k1 + k2]

    return BivariateCopulaFit(
        copula="marshall-olkin (shared frailty)",
        coef_1=b1, alpha_1=phi, coef_2=b2, alpha_2=phi,
        rho=float("nan"), rho_u=float("nan"), loglik=ll_val,
        aic=2 * kpar - 2 * ll_val, bic=kpar * np.log(n) - 2 * ll_val,
        n=int(n), k=int(kpar), method=str(method),
        converged=bool(converged),
        feature_names_1=list(feature_names_1 or [f"x1_{i}" for i in range(k1)]),
        feature_names_2=list(feature_names_2 or [f"x2_{i}" for i in range(k2)]),
        se_1=se_1, se_2=se_2,
        extra={"phi_shared": True},
    )


# ---------------------------------------------------------------------------
# 12.  Comparison helper  (Table 5 / 7 style)
# ---------------------------------------------------------------------------

def compare_bivariate_copulas(
    y1, y2, x1, x2, offset1, offset2,
    feature_names_1=None, feature_names_2=None,
    copulas=("frank", "normal", "kimeldorf"),
    method="scipy-lbfgsb",
):
    """Fit Famoye + Marshall-Olkin + each requested copula and return a
    pandas DataFrame mirroring Tables 5 and 7 of Ahmad et al. (2023)."""
    import pandas as pd

    rows = []
    fits = {}

    base = fit_univariate_nb(y1, x1, offset1, feature_names=feature_names_1)
    base2 = fit_univariate_nb(y2, x2, offset2, feature_names=feature_names_2)
    n = base["n"]; ks = base["k"] + base2["k"]; ll = base["loglik"] + base2["loglik"]
    univar_summary = {
        "ll": ll, "k": ks, "n": n,
        "aic": 2 * ks - 2 * ll, "bic": ks * np.log(n) - 2 * ll,
    }
    fits["univar_sum"] = univar_summary
    rows.append({
        "model": "Sum of 2 univariate NB (baseline)",
        "n": n, "k": ks, "loglik": ll,
        "AIC": univar_summary["aic"], "BIC": univar_summary["bic"],
        "rho": float("nan"),
    })

    mo = fit_marshall_olkin_nb(y1, y2, x1, x2, offset1, offset2,
                                feature_names_1=feature_names_1,
                                feature_names_2=feature_names_2,
                                method=method)
    fits["marshall_olkin"] = mo
    rows.append({
        "model": mo.copula,
        "n": mo.n, "k": mo.k, "loglik": mo.loglik, "AIC": mo.aic,
        "BIC": mo.bic, "rho": mo.alpha_1,
    })

    fam = fit_famoye_nb(y1, y2, x1, x2, offset1, offset2,
                        feature_names_1=feature_names_1,
                        feature_names_2=feature_names_2,
                        alpha_1=base["alpha"], alpha_2=base2["alpha"],
                        method=method)
    fits["famoye"] = fam
    rows.append({
        "model": fam.copula,
        "n": fam.n, "k": fam.k, "loglik": fam.loglik, "AIC": fam.aic,
        "BIC": fam.bic, "rho": fam.rho,
    })

    for c in copulas:
        cf = fit_copula_bivariate_nb(y1, y2, x1, x2, offset1, offset2,
                                     copula=c,
                                     feature_names_1=feature_names_1,
                                     feature_names_2=feature_names_2,
                                     alpha_1=base["alpha"], alpha_2=base2["alpha"],
                                     method=method)
        fits[c] = cf
        lr_stat = 2.0 * (cf.loglik - univar_summary["ll"])
        from scipy.stats import chi2
        p_value = float(chi2.sf(max(lr_stat, 0.0), df=max(cf.k - univar_summary["k"], 1)))
        rows.append({
            "model": f"Bivariate NB ({c} copula)",
            "n": cf.n, "k": cf.k, "loglik": cf.loglik, "AIC": cf.aic,
            "BIC": cf.bic, "rho": cf.rho,
            "LR_independence": float(lr_stat),
            "p_independence": float(p_value),
        })
    return pd.DataFrame(rows), fits


# ---------------------------------------------------------------------------
# 13.  Public API short-cut
# ---------------------------------------------------------------------------

class BivariateCopulaNB:
    """Convenience wrapper.

    >>> nb = BivariateCopulaNB(copula="frank")
    >>> fit = nb.fit(y1, y2, x1, x2, offset1, offset2,
    ...                feature_names_1=["AADT", ...])
    >>> print(fit.summary())
    """

    def __init__(self, copula: str = "frank", method: str = "scipy-lbfgsb"):
        if copula not in COPULA_LOGS and copula not in (
            "famoye", "marshall_olkin",
        ):
            raise ValueError(f"unknown copula {copula!r}")
        self.copula = copula
        self.method = method

    def fit(self, y1, y2, x1, x2, offset1, offset2,
            feature_names_1=None, feature_names_2=None,
            alpha_1=None, alpha_2=None, n_iter=500):
        if self.copula == "famoye":
            return fit_famoye_nb(y1, y2, x1, x2, offset1, offset2,
                                 feature_names_1, feature_names_2,
                                 alpha_1, alpha_2, self.method, n_iter)
        if self.copula == "marshall_olkin":
            return fit_marshall_olkin_nb(y1, y2, x1, x2, offset1, offset2,
                                         feature_names_1, feature_names_2,
                                         alpha=None, method=self.method,
                                         n_iter=n_iter)
        return fit_copula_bivariate_nb(y1, y2, x1, x2, offset1, offset2,
                                       copula=self.copula,
                                       feature_names_1=feature_names_1,
                                       feature_names_2=feature_names_2,
                                       alpha_1=alpha_1, alpha_2=alpha_2,
                                       method=self.method, n_iter=n_iter)


__all__ = [
    "frank_logpdf",
    "normal_logpdf",
    "kimeldorf_sampson_logpdf",
    "nb_log_pmf",
    "univariate_nb_loglik",
    "bivariate_copula_loglik",
    "famoye_bivariate_nb_loglik",
    "marshall_olkin_nb_loglik",
    "fit_univariate_nb",
    "fit_copula_bivariate_nb",
    "fit_famoye_nb",
    "fit_marshall_olkin_nb",
    "compare_bivariate_copulas",
    "BivariateCopulaNB",
    "BivariateCopulaFit",
]
