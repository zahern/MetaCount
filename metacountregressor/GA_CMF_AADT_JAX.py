# ============================================================
# Hierarchical SPF  â€”  GA Variable + Random Parameter Selection
# JAX backend  |  Single AADT  |  Per-Variable Random Effects
# ============================================================
#
# Model structure:
#
#   log mu_i = [alpha0  +  sum_k (alpha_k + sigma_alpha_k * u_ik) * X_ki]
#            + [beta0   +  sum_k (beta_k  + sigma_beta_k  * v_ik) * X_ki]
#              * log(AADT_i)
#
#   Each variable independently may be:
#     - excluded entirely            (GA inclusion gene = 0)
#     - included as FIXED effect     (inclusion = 1, random gene = 0)
#     - included as RANDOM parameter (inclusion = 1, random gene = 1)
#
#   Zero random flags => pure fixed-effects, closed-form likelihood (fast).
#
# Why JAX?
#   1. JIT compilation via jax.jit  â€” likelihood evaluated ~10-50x faster
#   2. Exact analytic gradients via jax.value_and_grad  â€” no finite differences
#   3. Exact Hessian at solution via jax.hessian  â€” precise standard errors
#   4. GPU/TPU support at zero extra cost
#
# Optimizer:
#   scipy.optimize.minimize (L-BFGS-B) receives the JAX-computed gradient
#   directly via jac=True.
#
# Search gene layout  (total = 2*(k_base + k_loc) + 2):
#   [0 .. k_base-1]              inclusion flags  â€” baseline vars
#   [k_base .. k_base+k_loc-1]   inclusion flags  â€” local vars
#   [k_base+k_loc]               use_halton  (0=random, 1=Halton)
#   [k_base+k_loc+1]             model       (0=Poisson, 1=NB)
#   [k_base+k_loc+2 .. +k_base+1] random flags â€” baseline vars
#   [.. +k_loc]                  random flags â€” local vars
#
# CMF reporting (traditional road-safety format):
#   Component A  ->  CMF = exp(alpha_k)        direct multiplier on crash rate
#   Component B  ->  CMF = AADT_mean ^ beta_k  evaluated at sample mean AADT
# ============================================================

import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import qmc
try:
    from .Solvers_METAJAX import MultiStartSA
except ImportError:
    from Solvers_METAJAX import MultiStartSA

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
from jaxopt import ScipyMinimize as JaxoptMinimize

# Enable 64-bit floats â€” essential for numerical optimisation
try:
    from ._jax_config import configure_jax
except ImportError:  # flat import (script run from inside the package dir)
    from _jax_config import configure_jax
configure_jax()
try:
    from .nb_parameterization import nb2_logpmf
except ImportError:
    from nb_parameterization import nb2_logpmf


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HALTON / RANDOM DRAWS
# (generated in NumPy once, then converted to JAX arrays)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _ghq_draws(n, n_rand, points=7):
    """Gauss-Hermite quadrature nodes for the standard-normal random effects.

    This is the deterministic integration rule used by SAS PROC NLMIXED
    (adaptive Gauss-Hermite quadrature).  Here the rule is non-adaptive: the
    nodes are fixed, but the likelihood is evaluated exactly (up to quadrature
    error) rather than by Monte Carlo simulation, so there is no simulation
    noise and no dependence on a seed.

    Returns
    -------
    draws : jnp.ndarray, shape (R, n, n_rand)
        Node coordinates, broadcast across observations.
    log_weights : jnp.ndarray, shape (R,)
        Log quadrature weights (already normalised for N(0, 1)).
    """
    from itertools import product

    x, w = np.polynomial.hermite.hermgauss(int(points))
    # E[f(U)] = pi^{-1/2} * sum_i w_i f(sqrt(2) x_i) for U ~ N(0,1)
    log_w1 = np.log(w) - 0.5 * np.log(np.pi)
    nodes = np.sqrt(2.0) * x

    grid = np.array(list(product(nodes, repeat=n_rand)), dtype=np.float64)  # (R, q)
    logw = np.array([sum(log_w1[i] for i in idx)
                     for idx in product(range(len(nodes)), repeat=n_rand)],
                    dtype=np.float64)

    R = grid.shape[0]
    draws = jnp.broadcast_to(jnp.array(grid)[:, None, :], (R, n, n_rand))
    return draws, jnp.array(logw)


def _aghq_components(params, y, AADT, baseline_mat, locals_mat,
                     rand_baseline, rand_local, model, offset):
    """Unpack a hierarchical model into the pieces AGHQ needs.

    Returns
    -------
    eta0 : (N,)      linear predictor at u = 0
    a    : (N, q)    coefficient multiplying each random effect in eta
    theta: scalar    NB2 dispersion (exp of the raw parameter) or None
    """
    k_base = baseline_mat.shape[1]
    k_loc  = locals_mat.shape[1]
    N      = len(y)

    idx    = 0
    alpha0 = params[idx]; idx += 1
    alpha, sig_a = [], []
    for k in range(k_base):
        alpha.append(params[idx]); idx += 1
        if rand_baseline[k]:
            sig_a.append(params[idx]); idx += 1
        else:
            sig_a.append(0.0)

    beta0 = params[idx]; idx += 1
    beta, sig_b = [], []
    for k in range(k_loc):
        beta.append(params[idx]); idx += 1
        if rand_local[k]:
            sig_b.append(params[idx]); idx += 1
        else:
            sig_b.append(0.0)

    log_aadt = jnp.log(AADT)

    eta0 = alpha0 + (baseline_mat @ jnp.stack(alpha) if k_base > 0 else 0.0)
    B0   = beta0 + (locals_mat @ jnp.stack(beta) if k_loc > 0 else 0.0)
    eta0 = eta0 + B0 * log_aadt
    if offset is not None:
        eta0 = eta0 + offset

    cols = []
    for k in range(k_base):
        if rand_baseline[k]:
            cols.append(sig_a[k] * baseline_mat[:, k])
    for k in range(k_loc):
        if rand_local[k]:
            cols.append(sig_b[k] * locals_mat[:, k] * log_aadt)
    a = jnp.stack(cols, axis=1) if cols else jnp.zeros((N, 0))

    alpha = jnp.exp(params[idx]) if model == 'nb' else None
    return eta0, a, alpha

def _aghq_conditional_modes(params, y, AADT, baseline_mat, locals_mat,
                            rand_baseline, rand_local, model, offset,
                            n_newton=25, max_step=1.0):
    """Posterior (empirical-Bayes) mode of the random effects per observation.

    Damped Newton with a step-size cap; an undamped step diverges for some
    high-count observations when several random effects are present.

    Returns (u_hat, mu_hat) where u_hat is (N, q) and mu_hat is (N,).
    """
    eta0, a, alpha = _aghq_components(params, y, AADT, baseline_mat, locals_mat,
                                      rand_baseline, rand_local, model, offset)
    N, q = a.shape
    if q == 0:
        return jnp.zeros((N, 0)), jnp.exp(eta0)

    def _curv(mu):
        if model == 'nb':
            # -d2 log p / d eta^2 for NB2 with Var = mu + alpha*mu^2
            return mu * (1.0 + alpha * y) / (1.0 + alpha * mu) ** 2
        return mu

    def body(_, u):
        eta = eta0 + jnp.sum(a * u, axis=1)
        mu  = jnp.exp(jnp.clip(eta, -30.0, 30.0))
        c   = _curv(mu)
        H = jnp.eye(q)[None, :, :] + c[:, None, None] * (a[:, :, None] * a[:, None, :])
        if model == 'nb':
            # For NB2, d log p(y | eta) / d eta is
            # (y - mu) / (1 + alpha * mu), not the Poisson score.
            score = (y - mu) / (1.0 + alpha * mu)
        else:
            score = y - mu
        g = score[:, None] * a - u
        step = jnp.linalg.solve(H, g[:, :, None])[:, :, 0]
        nrm = jnp.sqrt(jnp.sum(step * step, axis=1))
        scale = jnp.minimum(1.0, max_step / (nrm + 1e-12))
        u = u + scale[:, None] * step
        return jnp.clip(u, -8.0, 8.0)

    u = jnp.zeros((N, q))
    u = jax.lax.fori_loop(0, int(n_newton), body, u)
    eta = eta0 + jnp.sum(a * u, axis=1)
    mu = jnp.exp(jnp.clip(eta, -30.0, 30.0))
    return u, mu


def _aghq_loglik(params, y, AADT, baseline_mat, locals_mat,
                 rand_baseline, rand_local, model, offset, points=7):
    """Adaptive Gauss-Hermite quadrature (SAS PROC NLMIXED default).

    The nodes are centred at each observation's conditional mode and scaled by
    the inverse curvature of its posterior, so the rule tracks the integrand
    instead of using a fixed grid.
    """
    eta0, a, alpha = _aghq_components(params, y, AADT, baseline_mat, locals_mat,
                                      rand_baseline, rand_local, model, offset)
    N, q = a.shape
    if q == 0:
        mu = jnp.exp(jnp.clip(eta0, -30.0, 30.0))
        ll = y * eta0 - mu - gammaln(y + 1.0)
        return -jnp.sum(ll)

    u_hat, mu_hat = _aghq_conditional_modes(
        params, y, AADT, baseline_mat, locals_mat, rand_baseline, rand_local,
        model, offset)

    # The nodes depend on the parameters only through the conditional mode and
    # curvature.  Differentiating through the inner Newton solve is numerically
    # unstable (NaN gradients), and its contribution to the score is negligible,
    # so the adaptive pieces are treated as fixed at each evaluation.
    u_hat = jax.lax.stop_gradient(u_hat)
    mu_hat = jax.lax.stop_gradient(mu_hat)

    # Posterior precision H = I + c a a'  (Gauss-Newton, exact for Poisson/NB2)
    if model == 'nb':
        c = mu_hat * (1.0 + alpha * y) / (1.0 + alpha * mu_hat) ** 2
    else:
        c = mu_hat
    H = jnp.eye(q)[None, :, :] + c[:, None, None] * (a[:, :, None] * a[:, None, :])

    # Stable: factor H = L L' and use S = L^{-T} (S S' = H^{-1}); this avoids
    # inverting H, which overflows for extreme trial values during optimisation.
    L = jnp.linalg.cholesky(H + 1e-10 * jnp.eye(q)[None, :, :])
    Linv = jnp.linalg.solve(L, jnp.broadcast_to(jnp.eye(q), (N, q, q)))
    S = jnp.swapaxes(Linv, 1, 2)
    S = jax.lax.stop_gradient(S)
    logdetH = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)), axis=1)

    x, w = np.polynomial.hermite.hermgauss(int(points))
    log_w1 = np.log(w) - 0.5 * np.log(np.pi)
    z1 = np.sqrt(2.0) * x

    from itertools import product
    z = np.array(list(product(z1, repeat=q)), dtype=np.float64)          # (R, q)
    logw = np.array([sum(log_w1[i] for i in idx)
                     for idx in product(range(len(z1)), repeat=q)])       # (R,)
    z = jnp.array(z)
    logw = jnp.array(logw)
    R = z.shape[0]

    # u_k = u_hat + S z_k
    u = u_hat[None, :, :] + jnp.einsum('nij,rj->rni', S, z)              # (R, N, q)
    eta = eta0[None, :] + jnp.einsum('rni,ni->rn', u, a)                 # (R, N)
    eta = jnp.clip(eta, -30.0, 30.0)

    if model == 'nb':
        ll = nb2_logpmf(y[None, :], eta, jnp.log(alpha))
    else:
        mu = jnp.exp(eta)
        ll = y[None, :] * eta - mu - gammaln(y + 1.0)[None, :]

    # importance correction: phi(u)/phi(z) and the Jacobian |S|
    quad = -0.5 * (jnp.sum(u * u, axis=2) - jnp.sum(z * z, axis=1)[:, None])
    loglik_i = logsumexp(ll + quad + logw[:, None], axis=0) - 0.5 * logdetH
    # Guard against non-finite values probed during optimisation.
    loglik_i = jnp.nan_to_num(loglik_i, nan=-1e6, posinf=-1e6, neginf=-1e6)
    return -jnp.sum(loglik_i)


def generate_draws(n, R, n_rand, use_halton=True, seed=42):
    """
    Returns JAX array of standard-normal draws shaped (R, n, n_rand).
    Returns None when n_rand == 0 (no simulation needed).
    """
    if n_rand == 0:
        return None

    np.random.seed(seed)

    if use_halton:
        sampler = qmc.Halton(d=n_rand, scramble=True, seed=seed)
        sampler.fast_forward(100)
        # Give each observation a CONTIGUOUS block of R sequence points.
        # Drawing R blocks of n and filling draws[r, i] instead would hand
        # observation i a strided subsequence (indices i, n+i, 2n+i, ...),
        # which is a poor low-discrepancy set and badly biases the simulated
        # likelihood for non-linear integrands (e.g. high-count observations).
        uniform_draws = sampler.random(R * n)
        clipped = np.clip(uniform_draws, 1e-12, 1 - 1e-12)
        draws_np = norm.ppf(clipped).reshape(n, R, n_rand).transpose(1, 0, 2)
    else:
        draws_np = norm.ppf(np.random.rand(R, n, n_rand))

    return jnp.array(draws_np)   # hand off to JAX


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PARAMETER VECTOR LAYOUT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
#   alpha0
#   [alpha_k, (sigma_alpha_k if rand)]  for each active baseline var
#   beta0
#   [beta_k,  (sigma_beta_k  if rand)]  for each active local var
#   (log_theta if NB)

def count_params(rand_baseline, rand_local, model):
    n  = 1 + len(rand_baseline) + sum(rand_baseline)   # alpha block
    n += 1 + len(rand_local)   + sum(rand_local)        # beta  block
    if model == 'nb':
        n += 1
    return n


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# JAX LOG-LIKELIHOODS  (JIT-compiled)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# rand_baseline / rand_local are Python tuples of bools â€” they control
# static code structure so they are declared as static_argnums.
# Data arrays (y, AADT, matrices, draws) are regular traced arguments.
#
# Two entry-points:
#   _fixed_ll  â€” pure fixed effects, no draws needed
#   _mixed_ll  â€” simulation-based, draws required


@partial(jax.jit, static_argnames=('rand_baseline', 'rand_local', 'model'))
def _fixed_ll(params, y, AADT, baseline_mat, locals_mat,
              rand_baseline, rand_local, model, offset=None):
    """
    Closed-form log-likelihood — called when all random flags are False.
    JIT-compiled: ~2-5x faster than NumPy on CPU for moderate N.

    offset: optional (N,) array of log(exposure) e.g. log(segment length).
    """
    k_base = baseline_mat.shape[1]
    k_loc  = locals_mat.shape[1]

    idx = 0
    alpha0 = params[idx]; idx += 1

    alphas = []
    for k in range(k_base):
        alphas.append(params[idx]); idx += 1
    alpha = jnp.stack(alphas) if k_base > 0 else jnp.zeros(0)

    beta0 = params[idx]; idx += 1

    betas = []
    for k in range(k_loc):
        betas.append(params[idx]); idx += 1
    beta = jnp.stack(betas) if k_loc > 0 else jnp.zeros(0)

    log_A  = alpha0 + (baseline_mat @ alpha if k_base > 0 else 0.0)
    log_A  = jnp.clip(log_A, -20.0, 20.0)
    B      = beta0  + (locals_mat  @ beta  if k_loc  > 0 else 0.0)
    log_mu = log_A + B * jnp.log(AADT)
    if offset is not None:
        log_mu = log_mu + offset
    log_mu = jnp.clip(log_mu, -30.0, 30.0)
    mu     = jnp.exp(log_mu)

    if model == 'poisson':
        ll = y * log_mu - mu - gammaln(y + 1.0)
    else:
        log_theta = params[idx]
        ll = nb2_logpmf(y, log_mu, log_theta)

    return -jnp.sum(ll)


@partial(jax.jit, static_argnames=('rand_baseline', 'rand_local', 'model'))
def _mixed_ll(params, y, AADT, baseline_mat, locals_mat,
              rand_baseline, rand_local, draws, model, offset=None,
              log_weights=None):
    """
    Mixed log-likelihood with per-variable random effects.

    Two integration rules are supported:
      * simulation (default): ``draws`` are Halton/normal draws and the
        likelihood is the average over draws (maximum simulated likelihood);
      * Gauss-Hermite quadrature: ``draws`` are the quadrature nodes and
        ``log_weights`` carries the log weights.  This is the deterministic
        rule used by SAS PROC NLMIXED.

    draws : (R, N, n_rand)
    offset: optional (N,) array of log(exposure)
    log_weights : optional (R,) quadrature log-weights
    """
    R      = draws.shape[0]
    N      = len(y)
    k_base = baseline_mat.shape[1]
    k_loc  = locals_mat.shape[1]

    idx       = 0
    alpha0    = params[idx]; idx += 1

    alpha       = []
    sigma_alpha = []
    for k in range(k_base):
        alpha.append(params[idx]); idx += 1
        if rand_baseline[k]:
            sigma_alpha.append(params[idx]); idx += 1
        else:
            sigma_alpha.append(0.0)

    beta0     = params[idx]; idx += 1

    beta       = []
    sigma_beta = []
    for k in range(k_loc):
        beta.append(params[idx]); idx += 1
        if rand_local[k]:
            sigma_beta.append(params[idx]); idx += 1
        else:
            sigma_beta.append(0.0)

    log_AADT = jnp.log(AADT)   # (N,)

    # — Component A — shape (R, N) —
    log_A    = jnp.full((R, N), alpha0)
    draw_col = 0
    for k in range(k_base):
        xk = baseline_mat[:, k][None, :]    # (1, N)
        if rand_baseline[k]:
            u  = draws[:, :, draw_col]      # (R, N)
            log_A = log_A + (alpha[k] + sigma_alpha[k] * u) * xk
            draw_col += 1
        else:
            log_A = log_A + alpha[k] * xk
    log_A = jnp.clip(log_A, -20.0, 20.0)

    # — Component B — shape (R, N) —
    B = jnp.full((R, N), beta0)
    for k in range(k_loc):
        xk = locals_mat[:, k][None, :]
        if rand_local[k]:
            v = draws[:, :, draw_col]
            B = B + (beta[k] + sigma_beta[k] * v) * xk
            draw_col += 1
        else:
            B = B + beta[k] * xk

    log_mu = log_A + B * log_AADT[None, :]
    if offset is not None:
        log_mu = log_mu + offset[None, :]
    log_mu = jnp.clip(log_mu, -30.0, 30.0)
    mu     = jnp.exp(log_mu)

    if model == 'poisson':
        sim_ll = y[None, :] * log_mu - mu - gammaln(y + 1.0)[None, :]
    else:
        sim_ll = nb2_logpmf(y[None, :], log_mu, params[idx])

    if log_weights is None:
        loglik_i = logsumexp(sim_ll, axis=0) - jnp.log(R)
    else:
        loglik_i = logsumexp(sim_ll + log_weights[:, None], axis=0)
    return -jnp.sum(loglik_i)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# EXPONENTIAL-ELASTICITY FORM  (paper Eq 37-39)
#   Np = A(z1) * AADT^{B(z2)}
#   log A(z1) = alpha0 + sum_k alpha_k * X_ki
#   B(z2)     = beta0 * exp( sum_j beta_j * z_ji )        [beta0 = exp(raw_beta0) > 0]
# so   log_mu_i = log A(z1) + B(z2) * log(AADT_i)
#
# Hierarchical extension keeps per-variable random effects on the
# Component A covariates and on the Component B exponent covariates
# (simulation-based MSL via _mixed_ll_exp).
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def count_params_exp(rand_baseline, rand_local, model):
    n = 1 + len(rand_baseline) + sum(rand_baseline)   # alpha block
    n += 1 + len(rand_local) + sum(rand_local)        # raw_beta0 + beta/exponent block
    if model == 'nb':
        n += 1
    return n


def _param_labels_exp(selected_baseline, selected_local,
                      rand_baseline, rand_local, model):
    labels = ['alpha0']
    for k, v in enumerate(selected_baseline):
        labels.append(f'alpha[{v}]')
        if rand_baseline[k]:
            labels.append(f'sigma_alpha[{v}]')
    labels.append('log_beta0')
    for k, v in enumerate(selected_local):
        labels.append(f'beta[{v}]')
        if rand_local[k]:
            labels.append(f'sigma_beta[{v}]')
    if model == 'nb':
        labels.append('log_theta')
    return labels


@partial(jax.jit, static_argnames=('rand_baseline', 'rand_local', 'model'))
def _fixed_ll_exp(params, y, AADT, baseline_mat, locals_mat,
                  rand_baseline, rand_local, model, offset=None):
    """Closed-form log-likelihood for the exp-elasticity form (all fixed)."""
    k_base = baseline_mat.shape[1]
    k_loc  = locals_mat.shape[1]

    idx     = 0
    alpha0  = params[idx]; idx += 1
    alphas  = []
    for k in range(k_base):
        alphas.append(params[idx]); idx += 1
    alpha   = jnp.stack(alphas) if k_base > 0 else jnp.zeros(0)

    raw_beta0 = params[idx]; idx += 1
    beta0   = jnp.exp(raw_beta0)                      # base elasticity > 0
    betas   = []
    for k in range(k_loc):
        betas.append(params[idx]); idx += 1
    beta    = jnp.stack(betas) if k_loc > 0 else jnp.zeros(0)

    log_A = alpha0 + (baseline_mat @ alpha if k_base > 0 else 0.0)
    log_A = jnp.clip(log_A, -20.0, 20.0)
    expo  = locals_mat @ beta if k_loc > 0 else 0.0
    B     = beta0 * jnp.exp(expo)
    log_mu = log_A + B * jnp.log(AADT)
    if offset is not None:
        log_mu = log_mu + offset
    log_mu = jnp.clip(log_mu, -30.0, 30.0)
    mu     = jnp.exp(log_mu)

    if model == 'poisson':
        ll = y * log_mu - mu - gammaln(y + 1.0)
    else:
        log_theta = params[idx]
        ll = nb2_logpmf(y, log_mu, log_theta)

    return -jnp.sum(ll)


@partial(jax.jit, static_argnames=('rand_baseline', 'rand_local', 'model'))
def _mixed_ll_exp(params, y, AADT, baseline_mat, locals_mat,
                  rand_baseline, rand_local, draws, model, offset=None):
    """
    Simulation-based log-likelihood for the exp-elasticity form with
    per-variable random effects (hierarchical, MSL).
    draws : (R, N, n_rand) standard normal, pre-generated.
    offset: optional (N,) array of log(exposure) e.g. log(segment length).
    """
    R      = draws.shape[0]
    N      = len(y)
    k_base = baseline_mat.shape[1]
    k_loc  = locals_mat.shape[1]

    idx       = 0
    alpha0    = params[idx]; idx += 1
    alpha       = []
    sigma_alpha = []
    for k in range(k_base):
        alpha.append(params[idx]); idx += 1
        if rand_baseline[k]:
            sigma_alpha.append(params[idx]); idx += 1
        else:
            sigma_alpha.append(0.0)

    raw_beta0 = params[idx]; idx += 1
    beta0    = jnp.exp(raw_beta0)
    beta       = []
    sigma_beta = []
    for k in range(k_loc):
        beta.append(params[idx]); idx += 1
        if rand_local[k]:
            sigma_beta.append(params[idx]); idx += 1
        else:
            sigma_beta.append(0.0)

    log_AADT = jnp.log(AADT)            # (N,)

    # — Component A — shape (R, N) —
    log_A    = jnp.full((R, N), alpha0)
    draw_col = 0
    for k in range(k_base):
        xk = baseline_mat[:, k][None, :]    # (1, N)
        if rand_baseline[k]:
            u = draws[:, :, draw_col]       # (R, N)
            log_A = log_A + (alpha[k] + sigma_alpha[k] * u) * xk
            draw_col += 1
        else:
            log_A = log_A + alpha[k] * xk
    log_A = jnp.clip(log_A, -20.0, 20.0)

    # — Component B — exponential elasticity, shape (R, N) —
    expo    = jnp.zeros((R, N))
    for k in range(k_loc):
        xk = locals_mat[:, k][None, :]
        if rand_local[k]:
            v = draws[:, :, draw_col]
            expo = expo + (beta[k] + sigma_beta[k] * v) * xk
            draw_col += 1
        else:
            expo = expo + beta[k] * xk
    B = beta0 * jnp.exp(expo)

    log_mu = log_A + B * log_AADT[None, :]
    if offset is not None:
        log_mu = log_mu + offset[None, :]
    log_mu = jnp.clip(log_mu, -30.0, 30.0)
    mu     = jnp.exp(log_mu)

    if model == 'poisson':
        sim_ll = y[None, :] * log_mu - mu - gammaln(y + 1.0)[None, :]
    else:
        sim_ll = nb2_logpmf(y[None, :], log_mu, params[idx])

    loglik_i = logsumexp(sim_ll, axis=0) - jnp.log(R)
    return -jnp.sum(loglik_i)


def make_objective_exp(y_jax, AADT_jax, baseline_jax, locals_jax,
                       rand_baseline, rand_local, draws, R, model,
                       offset=None):
    """Return the JAX objective for the exp-elasticity form."""
    rand_baseline_t = tuple(rand_baseline)
    rand_local_t    = tuple(rand_local)
    simulate        = any(rand_baseline_t) or any(rand_local_t)
    if not simulate:
        return lambda p: _fixed_ll_exp(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, model, offset)
    return lambda p: _mixed_ll_exp(
        p, y_jax, AADT_jax, baseline_jax, locals_jax,
        rand_baseline_t, rand_local_t, draws, model, offset)


def spf_loglike(params, y, AADT, baseline_mat, locals_mat,
                rand_baseline, rand_local, draws=None, R=200, model='poisson',
                offset=None):
    """
    Dispatcher: routes to fixed or mixed JIT kernel.
    rand_baseline / rand_local must be tuples (hashable for JIT caching).

    offset: optional (N,) array of log(exposure) e.g. log(segment length).
    """
    rand_baseline = tuple(rand_baseline)
    rand_local    = tuple(rand_local)

    simulate = any(rand_baseline) or any(rand_local)

    if not simulate:
        return _fixed_ll(params, y, AADT, baseline_mat, locals_mat,
                         rand_baseline, rand_local, model, offset)
    else:
        return _mixed_ll(params, y, AADT, baseline_mat, locals_mat,
                         rand_baseline, rand_local, draws, model, offset)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SCIPY OPTIMIZER BRIDGE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# jax.value_and_grad returns (loss, gradient) in a single forward+backward
# pass.  Wrapping it for scipy.minimize(jac=True) gives exact gradients
# with zero extra cost over evaluating the loss alone.

def make_objective(y_jax, AADT_jax, baseline_jax, locals_jax,
                   rand_baseline, rand_local, draws, R, model,
                   offset=None, log_weights=None,
                   integration='halton', aghq_points=7):
    """
    Returns the raw JAX objective function for use with jaxopt.

    integration:
      'halton' -- maximum simulated likelihood (default)
      'ghq'    -- fixed-node Gauss-Hermite quadrature
      'aghq'   -- adaptive Gauss-Hermite quadrature (NLMIXED default):
                  nodes centred/scaled at each observation's conditional mode.
    """
    rand_baseline_t = tuple(rand_baseline)
    rand_local_t    = tuple(rand_local)
    simulate        = any(rand_baseline_t) or any(rand_local_t)

    if integration == 'aghq' and simulate:
        return lambda p: _aghq_loglik(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, model, offset, aghq_points)

    if not simulate:
        raw_fn = lambda p: _fixed_ll(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, model, offset)
    else:
        raw_fn = lambda p: _mixed_ll(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, draws, model, offset, log_weights)

    return raw_fn


def run_optimizer(objective, p0, method="SLSQP", maxiter=2000, disp=False):
    """
    Optimize using jaxopt.ScipyMinimize so autodiff is handled by JAXopt.
    objective must be a raw JAX function (scalar output); gradients are
    computed via JAX autodiff internally — no manual jac=True needed.
    """
    solver = JaxoptMinimize(
        fun=objective,
        method=method,
        maxiter=maxiter,
        tol=1e-8,
    )
    return solver.run(jnp.array(p0, dtype=jnp.float64))


def run_optimizer_bounded(objective, p0, bounds, method="SLSQP", maxiter=2000):
    """Bounded variant, used by AGHQ to keep the random-effect SDs finite."""
    from jaxopt import ScipyBoundedMinimize
    solver = ScipyBoundedMinimize(
        fun=objective, method=method, maxiter=maxiter, tol=1e-8)
    lb = jnp.array([b[0] for b in bounds], dtype=jnp.float64)
    ub = jnp.array([b[1] for b in bounds], dtype=jnp.float64)
    return solver.run(jnp.array(p0, dtype=jnp.float64), (lb, ub))


def _sigma_bounds(selected_baseline, selected_local, rand_baseline, rand_local,
                  model, limit=20.0, big=50.0):
    """Bounds vector used by the bounded optimiser.

    sigma  -> +/-limit
    log_alpha -> a plausible overdispersion range; an unbounded dispersion is
                 the main cause of the NB2 ridge where mu and alpha diverge
                 together.
    other  -> +/-big (finite, because jaxopt rejects infinite bounds)
    """
    labels = _param_labels(selected_baseline, selected_local,
                           rand_baseline, rand_local, model)
    out = []
    for lbl in labels:
        if lbl.startswith('sigma_'):
            out.append((-limit, limit))
        elif lbl in ('log_alpha', 'log_theta'):
            out.append((-6.0, 1.0))
        else:
            out.append((-big, big))
    return out


def _initial_params(labels, sigma0=0.5):
    """Start vector with sigma seeded away from zero.

    The likelihood is exactly even in each sigma (u and -u give the same
    likelihood), so sigma = 0 is a stationary point: a gradient optimiser
    started there never moves.  Seeding sigma at sigma0 breaks the symmetry.
    """
    p = np.zeros(len(labels), dtype=np.float64)
    for i, lbl in enumerate(labels):
        if lbl.startswith('sigma_'):
            p[i] = sigma0
    return p


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STANDARD ERRORS  (exact Hessian via JAX autodiff)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_se(result, y_jax, AADT_jax, baseline_jax, locals_jax,
               rand_baseline, rand_local, draws, R, model, offset=None,
               integration='halton', ghq_points=7):
    """
    Compute standard errors from the observed information matrix.

    Uses jax.hessian (exact second derivatives) at the MLE, which is
    far more accurate than the L-BFGS-B inverse-Hessian approximation.

    offset: optional (N,) array of log(exposure) e.g. log(segment length).
    integration: must match the integration used for the fit ('halton' or
        'ghq'); for 'ghq' the quadrature weights are rebuilt here.

    Returns (se, cov) as numpy arrays.
    """
    rand_baseline_t = tuple(rand_baseline)
    rand_local_t    = tuple(rand_local)
    simulate        = any(rand_baseline_t) or any(rand_local_t)

    n_rand = sum(rand_baseline_t) + sum(rand_local_t)
    log_w = None
    if simulate and integration == 'ghq' and n_rand > 0:
        draws, log_w = _ghq_draws(len(y_jax), n_rand, ghq_points)

    if integration == 'aghq' and simulate:
        raw_fn = lambda p: _aghq_loglik(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, model, offset, ghq_points)
    elif not simulate:
        raw_fn = lambda p: _fixed_ll(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, model, offset)
    else:
        raw_fn = lambda p: _mixed_ll(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, draws, model, offset, log_w)

    hess_fn = jax.jit(jax.hessian(raw_fn))
    # support both jaxopt result (.params) and scipy result (.x)
    p_star  = jnp.array(result.params if hasattr(result, "params") else result.x)

    try:
        H = np.array(hess_fn(p_star))
        H = np.where(np.isfinite(H), H, 0.0)
        # â”€â”€ PARTE-regularized SEs (Alghamdi et al. 2026) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Uses data-adaptive shrinkage per eigen-direction instead of a
        # uniform ridge penalty that collapses to 1/Î» when the Hessian is
        # near-singular (producing identical SEs for all parameters).
        try:
            from regularization import apply_parte_to_fixed_effects
            pr = apply_parte_to_fixed_effects(
                np.asarray(p_star, dtype=float), H, variant="k3d3"
            )
            se = pr.se_parte.copy()
        except Exception:
            # Fallback: plain ridge-regularized Hessian inversion
            eigvals, eigvecs = np.linalg.eigh(H)
            max_ev = float(np.max(np.abs(eigvals)))
            if max_ev > 0 and np.isfinite(max_ev):
                ridge = float(np.clip(max_ev * 1e-6, 1e-12, 1e-4))
                inv_eigvals = 1.0 / (eigvals + ridge)
                cov = (eigvecs * inv_eigvals) @ eigvecs.T
                diag_cov = np.diag(cov)
                diag_cov = np.where(diag_cov > 0, diag_cov, 1.0 / ridge)
                se = np.sqrt(diag_cov)
            else:
                n = len(p_star)
                cov = np.full((n, n), np.nan)
                se  = np.full(n, np.nan)
        n = len(p_star)
        cov = np.full((n, n), np.nan)  # covariance not computed with PARTE
    except np.linalg.LinAlgError:
        print("  WARNING: Hessian singular - returning NaN SEs.")
        n = len(p_star)
        cov = np.full((n, n), np.nan)
        se  = np.full(n, np.nan)

    return se, cov


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MODEL EVALUATOR  (called inside GA fitness)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def evaluate_model(solution,
                   data,
                   baseline_vars,
                   local_vars,
                   use_halton=True,
                   model='poisson',
                   rand_baseline_all=None,
                   rand_local_all=None,
                   R=200,
                   integration='halton',
                   ghq_points=7):

    k_base = len(baseline_vars)
    k_loc  = len(local_vars)

    baseline_mask = solution[:k_base].astype(bool)
    local_mask    = solution[k_base:k_base+k_loc].astype(bool)

    active_baseline = [v for v, m in zip(baseline_vars, baseline_mask) if m]
    active_local    = [v for v, m in zip(local_vars,    local_mask)    if m]

    if rand_baseline_all is None:
        rand_baseline_all = [False] * k_base
    if rand_local_all is None:
        rand_local_all = [False] * k_loc
    if len(rand_baseline_all) != k_base or len(rand_local_all) != k_loc:
        raise ValueError("Random-parameter flags must match the candidate variable lists.")

    rand_baseline = tuple(r for r, m in zip(rand_baseline_all, baseline_mask) if m)
    rand_local    = tuple(r for r, m in zip(rand_local_all,    local_mask)    if m)

    k_base_act = len(active_baseline)
    k_loc_act  = len(active_local)

    y_np    = np.asarray(data["FREQ"],  dtype=np.float64)
    AADT_np = np.asarray(data["AADT"],  dtype=np.float64)
    # Exposure offset: log(segment length) — matches paper's offset formulation
    if "LENGTH" in data.columns:
        offset_np = np.log(np.asarray(data["LENGTH"], dtype=np.float64))
    else:
        offset_np = None
    N       = len(y_np)

    # Standardise continuous predictors (non-binary) to mean=0, sd=1.
    # Critical for stability when candidates span very different scales
    # (e.g. MCV 0-2000 vs 0/1 indicators).  The *reported* model is refitted
    # in original space by fit_final_model, so this only affects search speed.
    bmat_np, bmat_means, bmat_stds, bmat_is_binary = _standardise_matrix(
        data, active_baseline, rand_baseline)
    lmat_np, lmat_means, lmat_stds, lmat_is_binary = _standardise_matrix(
        data, active_local, rand_local)

    # Convert to JAX arrays once — reused across optimizer iterations
    y_jax       = jnp.array(y_np)
    AADT_jax    = jnp.array(AADT_np)
    baseline_jax = jnp.array(bmat_np)
    locals_jax   = jnp.array(lmat_np)
    offset_jax   = jnp.array(offset_np) if offset_np is not None else None

    n_rand = sum(rand_baseline) + sum(rand_local)
    if integration == 'aghq':
        draws, log_w = None, None
    elif integration == 'ghq' and n_rand > 0:
        draws, log_w = _ghq_draws(N, n_rand, ghq_points)
    else:
        draws, log_w = generate_draws(N, R, n_rand, use_halton), None

    n_params = count_params(rand_baseline, rand_local, model)
    p0       = np.zeros(n_params)

    objective = make_objective(y_jax, AADT_jax, baseline_jax, locals_jax,
                               rand_baseline, rand_local, draws, R, model,
                               offset_jax, log_w, integration, ghq_points)

    # Return standardization info for unstandardizing coefficients later
    std_info = {
        "baseline_means": np.array(bmat_means) if k_base_act > 0 else np.array([]),
        "baseline_stds": np.array(bmat_stds) if k_base_act > 0 else np.array([]),
        "baseline_is_binary": np.array(bmat_is_binary) if k_base_act > 0 else np.array([]),
        "local_means": np.array(lmat_means) if k_loc_act > 0 else np.array([]),
        "local_stds": np.array(lmat_stds) if k_loc_act > 0 else np.array([]),
        "local_is_binary": np.array(lmat_is_binary) if k_loc_act > 0 else np.array([]),
        "fitted_scale": "standardised",
    }

    res = run_optimizer(objective, p0, method="SLSQP", maxiter=1000)

    params    = jnp.array(res.params)
    ll        = -float(objective(params))
    bic       = -2.0 * ll + n_params * np.log(N)

    # Significance penalty â€” diagonal Hessian via forward-over-reverse autodiff (O(p))
    grad_fn   = jax.grad(objective)
    n_p       = len(params)
    diag_hess = jnp.array([
        jax.jvp(grad_fn, (params,), (jnp.eye(n_p)[i],))[1][i]
        for i in range(n_p)
    ])
    se_approx = np.array(jnp.sqrt(jnp.maximum(1.0 / (jnp.abs(diag_hess) + 1e-30), 0.0)))

    params    = np.array(params)
    z         = np.where(se_approx > 1e-12, params / se_approx, 0.0)
    p_vals    = 2 * (1 - norm.cdf(np.abs(z)))
    # exclude sigma and log_theta from significance test
    sig_mask  = ~np.array([
        lbl.startswith('sigma_') or lbl == 'log_theta'
        for lbl in _param_labels(active_baseline, active_local, rand_baseline, rand_local, model)
    ])
    n_insig   = (p_vals[sig_mask] > 0.05).sum()
    penalty   = 5.0 * n_insig

    fmt_b = [f"{v}({'R' if r else 'F'})" for v, r in zip(active_baseline, rand_baseline)]
    fmt_l = [f"{v}({'R' if r else 'F'})" for v, r in zip(active_local,    rand_local)]
    print(f"  BIC={bic:.2f}  pen={penalty:.1f}  fit={bic+penalty:.2f}  "
          f"{model.upper()}  "
          f"A=[{', '.join(fmt_b) if fmt_b else '-'}]  "
          f"B=[{', '.join(fmt_l) if fmt_l else '-'}]")

    return bic + penalty, std_info


def _param_labels(selected_baseline, selected_local, rand_baseline, rand_local, model):
    """Return list of string labels matching the parameter vector order."""
    labels = ['alpha0']
    for k, v in enumerate(selected_baseline):
        labels.append(f'alpha[{v}]')
        if rand_baseline[k]:
            labels.append(f'sigma_alpha[{v}]')
    labels.append('beta0')
    for k, v in enumerate(selected_local):
        labels.append(f'beta[{v}]')
        if rand_local[k]:
            labels.append(f'sigma_beta[{v}]')
    if model == 'nb':
        labels.append('log_theta')
    return labels


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SOLVER-BASED SEARCH (MultiStartSA)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_ga(data, baseline_vars, local_vars, R=200, corr_threshold=0.8,
           min_baseline=2, min_local=1,
           max_baseline=None, max_local=None):
    """Search for a two-tier CMF specification.

    Dimension bounds (the programme's constraint C3, and the hierarchical
    integrity requirement ``K_L^min >= 1``) are enforced as hard rejections:
    a candidate that violates them is never scored, so it can neither enter
    the incumbent nor be reported.

    ``min_local >= 1`` guarantees that the selected model always retains at
    least one AADT-elasticity modifier, i.e. that the site-specific elasticity
    ``eta_i = beta + sum_k gamma_k z_k`` remains context-dependent.  Without it
    the search will happily return an empty lower tier, which silently discards
    the two-tier interpretation entirely.
    """
    k_base = len(baseline_vars)
    k_loc = len(local_vars)
    D = k_base + k_loc
    offset_rf = D + 2

    if max_baseline is None:
        max_baseline = k_base
    if max_local is None:
        max_local = k_loc

    min_baseline = max(0, int(min_baseline))
    min_local = max(0, int(min_local))
    max_baseline = min(int(max_baseline), k_base)
    max_local = min(int(max_local), k_loc)
    if min_local > k_loc:
        raise ValueError(
            f"min_local={min_local} exceeds the number of lower-tier "
            f"candidates ({k_loc}); the lower tier cannot be populated."
        )
    if min_baseline > max_baseline:
        raise ValueError("min_baseline exceeds max_baseline.")

    # ── Correlation constraint (Eq. 291): |corr(Z_k, Z_k')| <= c for baseline (upper-level) pairs
    # Compute pairwise correlations among baseline candidates from data
    _baseline_corr_matrix = None
    _baseline_corr_valid = set()
    if baseline_vars:
        _avail_baseline = [v for v in baseline_vars if v in data.columns]
        if len(_avail_baseline) >= 2:
            _corr_df = data[_avail_baseline].corr()
            _baseline_corr_matrix = _corr_df.abs()
            _baseline_corr_valid = set(_avail_baseline)

    def _check_corr_constraint(baseline_vars_selected: list[str]) -> bool:
        """Check if a set of baseline variables satisfies the correlation constraint.
        Returns True if all pairwise absolute correlations <= corr_threshold.
        """
        if _baseline_corr_matrix is None or len(baseline_vars_selected) < 2:
            return True
        for i, v1 in enumerate(baseline_vars_selected):
            if v1 not in _baseline_corr_valid:
                continue
            for v2 in baseline_vars_selected[i+1:]:
                if v2 not in _baseline_corr_valid:
                    continue
                try:
                    corr_val = _baseline_corr_matrix.loc[v1, v2]
                    if np.isfinite(corr_val) and corr_val > corr_threshold:
                        return False
                except KeyError:
                    continue
        return True

    class _CMFSearchEvaluator:
        def __init__(self):
            self.vars = [*baseline_vars, *local_vars]
            self.allowed_roles = {v: [0, 1] for v in self.vars}
            self.allowed_distributions = {v: ["normal"] for v in self.vars}

        @staticmethod
        def _to_binary(x):
            return np.asarray(x, dtype=int) % 2

        def decode(self, decision):
            # Solver dimension is 2*D+1: [include(D), random(D), model_bit]
            dec = self._to_binary(decision)
            include = dec[:D]
            random_flags = dec[D:2 * D]
            model_bit = dec[2 * D]

            full = np.zeros(2 * D + 2, dtype=int)
            full[:D] = include
            full[D] = 1  # keep Halton enabled in solver path
            full[D + 1] = model_bit
            full[offset_rf:offset_rf + D] = random_flags
            return full

        def build_spec(self, decision):
            full = self.decode(decision)
            model = "poisson" if int(full[D + 1]) == 0 else "nb"

            baseline_terms = []
            local_terms = []

            for idx, var in enumerate(baseline_vars):
                if int(full[idx]) == 1:
                    role = "R" if int(full[offset_rf + idx]) == 1 else "F"
                    baseline_terms.append((var, role))

            for idx, var in enumerate(local_vars):
                full_idx = k_base + idx
                if int(full[full_idx]) == 1:
                    role = "R" if int(full[offset_rf + full_idx]) == 1 else "F"
                    local_terms.append((var, role))

            # Check correlation constraint on baseline (upper-level) terms
            selected_baseline = [var for var, _ in baseline_terms]
            if not _check_corr_constraint(selected_baseline):
                return None  # Invalid spec - correlation constraint violated

            # Check dimension bounds (C3) and hierarchical integrity (K_L^min >= 1)
            n_base, n_loc = len(baseline_terms), len(local_terms)
            if not (min_baseline <= n_base <= max_baseline):
                return None  # Invalid spec - upper-tier size out of bounds
            if not (min_local <= n_loc <= max_local):
                return None  # Invalid spec - lower-tier size out of bounds

            return {
                "baseline_terms": tuple(sorted(baseline_terms)),
                "local_terms": tuple(sorted(local_terms)),
                "model": model,
            }

        @staticmethod
        def structural_signature(spec_dict):
            if spec_dict is None:
                return None
            return (
                spec_dict["baseline_terms"],
                spec_dict["local_terms"],
                spec_dict["model"],
            )

        def fitness(self, decision):
            full = self.decode(decision)
            model = "poisson" if int(full[D + 1]) == 0 else "nb"

            # Check correlation constraint on baseline terms before evaluating
            baseline_mask = full[:k_base].astype(bool)
            selected_baseline = [v for v, m in zip(baseline_vars, baseline_mask) if m]
            if not _check_corr_constraint(selected_baseline):
                return 1e12  # Large penalty for correlation constraint violation

            # Hard dimension bounds (C3) and hierarchical integrity (K_L^min >= 1).
            # Enforced here, not only in build_spec, because the optimiser
            # minimises this fitness function directly: without the check it
            # would happily return an empty lower tier, which silently drops
            # the two-tier interpretation.  Rejected candidates are never
            # scored, so no elastic miss-specification can become the incumbent.
            local_mask = full[k_base:k_base + k_loc].astype(bool)
            if not (min_baseline <= len(selected_baseline) <= max_baseline):
                return 1e12  # upper-tier size out of bounds
            n_loc = int(local_mask.sum())
            if not (min_local <= n_loc <= max_local):
                return 1e12  # lower-tier size out of bounds (e.g. K_L^min = 1)

            rand_baseline_all = [bool(full[offset_rf + i]) for i in range(k_base)]
            rand_local_all = [bool(full[offset_rf + k_base + i]) for i in range(k_loc)]
            fitness_val, _ = evaluate_model(
                full,
                data,
                baseline_vars,
                local_vars,
                use_halton=True,
                model=model,
                rand_baseline_all=rand_baseline_all,
                rand_local_all=rand_local_all,
                R=R,
            )
            return fitness_val

    evaluator = _CMFSearchEvaluator()
    dim = 2 * D + 1
    solver = MultiStartSA(
        evaluator=evaluator,
        dimension=dim,
        n_starts=1,
        max_iter=600,
        mutation_rate=0.3,
        step_size=1,
        alpha=0.995,
        min_changes=1,
        max_changes=3,
    )
    solutions, scores = solver.optimize()
    if len(scores) == 0:
        raise RuntimeError("CMF solver returned no candidate solutions.")

    best_idx = int(np.argmin(scores))
    best_decision = solutions[best_idx]
    fitness = float(scores[best_idx])

    solution = evaluator.decode(best_decision)
    baseline_mask = solution[:k_base].astype(bool)
    local_mask = solution[k_base:k_base + k_loc].astype(bool)
    use_halton = True
    model = "poisson" if int(solution[k_base + k_loc + 1]) == 0 else "nb"
    rand_baseline_all = [bool(solution[offset_rf + i]) for i in range(k_base)]
    rand_local_all = [bool(solution[offset_rf + k_base + i]) for i in range(k_loc)]

    selected_baseline = [v for v, m in zip(baseline_vars, baseline_mask) if m]
    selected_local    = [v for v, m in zip(local_vars,    local_mask)    if m]
    rand_baseline     = tuple(r for r, m in zip(rand_baseline_all, baseline_mask) if m)
    rand_local        = tuple(r for r, m in zip(rand_local_all,    local_mask)    if m)

    return (selected_baseline, selected_local,
            rand_baseline, rand_local,
            use_halton, model, fitness)


# ─────────────────────────────────────────────────────────────────────────
# FINAL ESTIMATION
# ─────────────────────────────────────────────────────────────────────────

def _standardise_column(values):
    """Return (standardised_values, mean, sd, is_binary).

    Binary 0/1 columns are left untouched (mean 0, sd 1 in the return) so a
    unit change stays a one-unit change.  Continuous columns are z-scored.
    """
    vals = np.asarray(values, dtype=np.float64)
    uniq = np.unique(vals[~np.isnan(vals)])
    if len(uniq) == 2 and set(uniq).issubset({0, 1}):
        return vals, 0.0, 1.0, True
    mean = np.nanmean(vals)
    sd   = np.nanstd(vals)
    if sd > 0:
        return (vals - mean) / sd, mean, sd, False
    return vals, mean, 1.0, False


def _standardise_matrix(data, cols, rand_flags=None):
    """Build a standardised design matrix and return its column metadata.

    ``rand_flags`` (optional, same length as ``cols``) marks random-parameter
    columns.  Those are deliberately left **raw**: rescaling a random slope
    ``(alpha + sigma u) x`` by ``1/sd`` also produces a term ``-sigma u mean/sd``,
    i.e. an induced random intercept perfectly correlated with the slope.  That
    is a different model, so random covariates must not be standardised if the
    fitted specification is to remain the intended one.
    """
    if not cols:
        return (np.zeros((len(data), 0)), np.array([]), np.array([]),
                np.array([], dtype=bool))
    mats, means, stds, bins = [], [], [], []
    for k, col in enumerate(cols):
        is_rand = bool(rand_flags[k]) if rand_flags is not None else False
        if is_rand:
            vals = np.asarray(data[col], dtype=np.float64)
            uniq = np.unique(vals[~np.isnan(vals)])
            is_bin = len(uniq) == 2 and set(uniq).issubset({0, 1})
            mats.append(vals); means.append(0.0); stds.append(1.0); bins.append(is_bin)
            continue
        m, mean, sd, is_bin = _standardise_column(data[col])
        mats.append(m); means.append(mean); stds.append(sd); bins.append(is_bin)
    return (np.column_stack(mats), np.array(means), np.array(stds),
            np.array(bins, dtype=bool))


def _unstandardise_params(params, selected_baseline, selected_local,
                          rand_baseline, rand_local, model,
                          b_means, b_stds, l_means, l_stds):
    """Map a parameter vector from standardised space to original space.

    For a continuous covariate, ``alpha_std * (x - mean) / sd`` equals
    ``(alpha_std / sd) * x - alpha_std * mean / sd``, so the slope is divided
    by ``sd`` and the intercept (alpha0 or beta0) absorbs the mean term.
    Binary covariates are unaffected because their mean/sd are 0/1.
    """
    p = np.asarray(params, dtype=np.float64).copy()
    labels = _param_labels(selected_baseline, selected_local,
                           rand_baseline, rand_local, model)

    # Alpha block
    i = labels.index('alpha0') + 1
    shift = 0.0
    for k, _ in enumerate(selected_baseline):
        p[i] = p[i] / b_stds[k]
        shift += p[i] * b_means[k]
        i += 1
        if rand_baseline[k]:
            p[i] = p[i] / b_stds[k]
            i += 1
    p[labels.index('alpha0')] -= shift

    # Beta block
    i = labels.index('beta0') + 1
    shift = 0.0
    for k, _ in enumerate(selected_local):
        p[i] = p[i] / l_stds[k]
        shift += p[i] * l_means[k]
        i += 1
        if rand_local[k]:
            p[i] = p[i] / l_stds[k]
            i += 1
    p[labels.index('beta0')] -= shift

    return p


def fit_final_model(data,
                    selected_baseline, selected_local,
                    rand_baseline, rand_local,
                    model='poisson', use_halton=True, R=500,
                    integration='halton', ghq_points=7, seed=42):
    """
    Re-estimates the GA-selected model at higher R for final inference.

    integration: 'halton' (maximum simulated likelihood, default),
        'ghq' (fixed-node Gauss-Hermite) or 'aghq' (adaptive Gauss-Hermite,
        the rule used by SAS PROC NLMIXED; nodes are centred/scaled at each
        observation's conditional mode).  For quadrature the node count is
        ghq_points ** n_random.

    Returns (result, y_jax, AADT_jax, baseline_jax, locals_jax, draws,
             std_info, offset_jax).
    """
    y_np    = np.asarray(data["FREQ"],  dtype=np.float64)
    AADT_np = np.asarray(data["AADT"],  dtype=np.float64)
    N       = len(y_np)

    k_base = len(selected_baseline)
    k_loc  = len(selected_local)

    # ── Stage 1: fit on the standardised design ───────────────────────────
    # The search fits in standardised space for numerical stability, so the
    # final refit starts from the same representation to reuse the good start.
    bmat_s, b_means, b_stds, b_bin = _standardise_matrix(
        data, selected_baseline, rand_baseline)
    lmat_s, l_means, l_stds, l_bin = _standardise_matrix(
        data, selected_local, rand_local)

    # Original (raw) design, used for the reported estimates and SEs
    bmat_raw = (data[selected_baseline].values.astype(np.float64)
                if k_base > 0 else np.zeros((N, 0)))
    lmat_raw = (data[selected_local].values.astype(np.float64)
                if k_loc  > 0 else np.zeros((N, 0)))

    # Exposure offset
    if "LENGTH" in data.columns:
        offset_np = np.log(np.asarray(data["LENGTH"], dtype=np.float64))
    else:
        offset_np = None

    y_jax        = jnp.array(y_np)
    AADT_jax     = jnp.array(AADT_np)
    offset_jax   = jnp.array(offset_np) if offset_np is not None else None

    n_rand = sum(rand_baseline) + sum(rand_local)
    if integration == 'aghq':
        draws, log_w = None, None
    elif integration == 'ghq' and n_rand > 0:
        draws, log_w = _ghq_draws(N, n_rand, ghq_points)
    else:
        draws, log_w = generate_draws(N, R, n_rand, use_halton, seed=seed), None

    n_params = count_params(rand_baseline, rand_local, model)
    _labels  = _param_labels(selected_baseline, selected_local,
                             rand_baseline, rand_local, model)
    p_start  = _initial_params(_labels)

    print("  Compiling JAX kernel (first call) …")
    obj_std = make_objective(y_jax, AADT_jax, jnp.array(bmat_s), jnp.array(lmat_s),
                             rand_baseline, rand_local, draws, R, model, offset_jax,
                             log_w, integration, ghq_points)
    if integration == 'aghq' or model == 'nb':
        _bnds = _sigma_bounds(selected_baseline, selected_local,
                              rand_baseline, rand_local, model)
        res_std = run_optimizer_bounded(obj_std, p_start, _bnds,
                                        method="SLSQP", maxiter=2000)
    else:
        res_std = run_optimizer(obj_std, np.zeros(n_params), method="SLSQP", maxiter=2000)

    # ── Stage 2: refit in ORIGINAL predictor space ────────────────────────
    # Reporting and inference are on the interpretable scale (CMF per unit,
    # elasticity per original covariate).  The standardised solution is only
    # used to initialise, so the optimum is unchanged but the parameterisation
    # no longer distorts continuous-variable coefficients.
    p0_orig = _unstandardise_params(
        np.asarray(res_std.params), selected_baseline, selected_local,
        rand_baseline, rand_local, model, b_means, b_stds, l_means, l_stds)

    baseline_jax = jnp.array(bmat_raw)
    locals_jax   = jnp.array(lmat_raw)
    objective = make_objective(y_jax, AADT_jax, baseline_jax, locals_jax,
                               rand_baseline, rand_local, draws, R, model,
                               offset_jax, log_w, integration, ghq_points)

    # Refit from the warm start; also try a zero start and keep the better
    # optimum, since the warm start came from a slightly different
    # (reparameterised) surface and can occasionally settle in a poor basin.
    if integration == 'aghq' or model == 'nb':
        _bnds = _sigma_bounds(selected_baseline, selected_local,
                              rand_baseline, rand_local, model)
        res = run_optimizer_bounded(objective, p0_orig, _bnds,
                                    method="SLSQP", maxiter=2000)
        res0 = run_optimizer_bounded(objective, p_start, _bnds,
                                     method="SLSQP", maxiter=2000)
    else:
        res = run_optimizer(objective, p0_orig, method="SLSQP", maxiter=2000)
        res0 = run_optimizer(objective, np.zeros(n_params), method="SLSQP", maxiter=2000)
    if float(objective(jnp.array(res0.params))) < float(objective(jnp.array(res.params))):
        res = res0

    std_info = {
        "baseline_means": b_means,
        "baseline_stds": b_stds,
        "baseline_is_binary": b_bin,
        "local_means": l_means,
        "local_stds": l_stds,
        "local_is_binary": l_bin,
        "fitted_scale": "original",
    }

    return res, y_jax, AADT_jax, baseline_jax, locals_jax, draws, std_info, offset_jax


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SUMMARY TABLE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_summary_table(result,
                        selected_baseline, selected_local,
                        rand_baseline, rand_local,
                        model,
                        se):
    """
    se : numpy array of standard errors (from exact Hessian or fallback).
    """
    params = np.array(result.params if hasattr(result, "params") else result.x)
    labels = _param_labels(selected_baseline, selected_local, rand_baseline, rand_local, model)

    # Display sigma parameters as absolute values (|sigma| is the identified quantity)
    display_params = params.copy()
    for i, lbl in enumerate(labels):
        if lbl.startswith('sigma_'):
            display_params[i] = abs(params[i])

    z      = np.where(se > 1e-12, display_params / se, 0.0)
    p      = 2 * (1 - norm.cdf(np.abs(z)))

    rows = list(zip(labels, display_params, se, z, p))
    return pd.DataFrame(rows, columns=["Parameter", "Estimate", "Std.Err", "z-value", "p-value"])


def print_summary_table(df):
    def stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    def _pval_fmt(p):
        if not np.isfinite(p): return "    n/a"
        if p < 0.001: return "  <0.001"
        return f"  {p:.4f}"

    def _fmt(v, w, sign=False):
        if not np.isfinite(v): return "n/a".rjust(w)
        fspec = f"{{:{'+' if sign else ''}{w}.4f}}" if abs(v) < 1e4 else f"{{:{'+' if sign else ''}{w}.2f}}"
        return fspec.format(v)

    W = 78
    print("\n" + "=" * W)
    print("  MODEL SUMMARY")
    print("=" * W)
    print(f"  {'Parameter':<30}  {'Estimate':>12}  {'Std.Err':>11}  {'z-value':>10}  {'p-value':>8}  Signif")
    print("  " + "-" * 29 + "  " + "-" * 12 + "  " + "-" * 11 + "  " + "-" * 10 + "  " + "-" * 8 + "  " + "-" * 6)

    for _, row in df.iterrows():
        param = str(row["Parameter"])[:30]
        est   = _fmt(row["Estimate"], 12, True)
        se    = _fmt(row["Std.Err"],   11, False)
        z     = _fmt(row["z-value"],   10, True)
        pv    = _pval_fmt(row["p-value"])
        st    = stars(row["p-value"])
        print(f"  {param:<30} {est} {se} {z} {pv}{st}")

    print("")
    print("  Signif: *** p<0.01   ** p<0.05   * p<0.10")
    print("  sigma_* rows: standard deviation of the random parameter distribution")
    print("  SE computed from exact observed information matrix (JAX Hessian + ridge)")
    print("=" * W + "\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CMF REPORTING  â€”  traditional road-safety format
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def print_cmf_results(result,
                      selected_baseline, selected_local,
                      rand_baseline, rand_local,
                      AADT_mean,
                      model='poisson'):
    """
    Prints CMFs in traditional road-safety format with full disclosure of
    fixed vs random status and heterogeneity ranges for random parameters.

    Component A  ->  CMF = exp(alpha_k)
      Fixed  : single multiplier applies identically at every site
      Random : CMF ~ LogNormal; 95% range = [exp(alpha +/- 1.96*sigma)]

    Component B  ->  CMF = AADT_mean ^ beta_k
      Fixed  : constant elasticity shift
      Random : elasticity varies; 95% range given
    """
    params = np.array(result.params if hasattr(result, "params") else result.x)
    idx    = 0
    W      = 72
    col_w  = 20

    print("\n" + "=" * W)
    print("   CRASH MODIFICATION FACTORS  â€”  Traditional Reporting Format")
    print("=" * W)
    print(f"   AADT reference (sample mean) : {AADT_mean:,.0f} veh/day")
    print(f"   Model                        : {model.upper()}")
    print()
    print("   [R] = RANDOM parameter  â€” effect varies by site")
    print("   [F] = FIXED  parameter  â€” effect constant across all sites")

    # â”€â”€ Component A â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    alpha0 = params[idx]; idx += 1

    print(f"\n{'=' * W}")
    print("  COMPONENT A  â€”  Site Characteristics  [CMF = exp(alpha_k)]")
    print(f"{'=' * W}")
    print(f"\n  Intercept  alpha0 = {alpha0:+.4f}  ->  exp(alpha0) = {np.exp(alpha0):.4f}")

    hdr = (f"\n  {'Variable':<{col_w}}  {'':>3}  {'alpha':>8}  "
           f"{'CMF':>8}  {'Delta Crashes':>14}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 3))

    for k, v in enumerate(selected_baseline):
        a   = params[idx]; idx += 1
        cmf = np.exp(a)
        pct = (cmf - 1) * 100
        tag = "[R]" if rand_baseline[k] else "[F]"
        print(f"  {v:<{col_w}}  {tag:>3}  {a:>+8.4f}  {cmf:>8.4f}  {pct:>+13.2f}%")
        if rand_baseline[k]:
            sig = params[idx]; idx += 1
            sig_abs = abs(sig)
            lo  = np.exp(a - 1.96 * sig_abs)
            hi  = np.exp(a + 1.96 * sig_abs)
            print(f"  {'':>{col_w}}       sigma = {sig_abs:.4f}  ->  95% CMF range: [{lo:.4f}, {hi:.4f}]")
            print(f"  {'':>{col_w}}       Mean CMF = {cmf:.4f}  "
                  f"(sites range {(lo-1)*100:+.1f}% to {(hi-1)*100:+.1f}%)")

    if any(rand_baseline):
        print()
        print("  NOTE: Mean CMF is the average effect; sigma captures site heterogeneity.")

    # â”€â”€ Component B â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    beta0 = params[idx]; idx += 1

    print(f"\n{'=' * W}")
    print("  COMPONENT B  â€”  AADT Elasticity  [CMF = AADT_mean ^ beta_k]")
    print(f"  Evaluated at mean AADT = {AADT_mean:,.0f} veh/day")
    print(f"{'=' * W}")
    print(f"\n  Base elasticity  beta0 = {beta0:+.4f}")
    print(f"  -> 1% increase in AADT changes crashes by {beta0:.4f}%  (reference site)")

    if selected_local:
        hdr2 = (f"\n  {'Variable':<{col_w}}  {'':>3}  {'beta':>8}  "
                f"{'Eff. elast.':>12}  {'CMF @ mean':>11}  {'Delta Crashes':>14}")
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 3))

        for k, v in enumerate(selected_local):
            b          = params[idx]; idx += 1
            elasticity = beta0 + b
            cmf        = float(AADT_mean) ** b
            pct        = (cmf - 1) * 100
            tag        = "[R]" if rand_local[k] else "[F]"
            print(f"  {v:<{col_w}}  {tag:>3}  {b:>+8.4f}  "
                  f"{elasticity:>12.4f}  {cmf:>11.4f}  {pct:>+13.2f}%")
            if rand_local[k]:
                sig    = params[idx]; idx += 1
                sig_abs = abs(sig)
                e_lo   = elasticity - 1.96 * sig_abs
                e_hi   = elasticity + 1.96 * sig_abs
                cmf_lo = float(AADT_mean) ** (b - 1.96 * sig_abs)
                cmf_hi = float(AADT_mean) ** (b + 1.96 * sig_abs)
                print(f"  {'':>{col_w}}       sigma = {sig_abs:.4f}  ->  "
                      f"95% elast. range: [{e_lo:.4f}, {e_hi:.4f}]")
                print(f"  {'':>{col_w}}       95% CMF @ mean AADT:  [{cmf_lo:.4f}, {cmf_hi:.4f}]")

        if any(rand_local):
            print()
            print("  NOTE: 'Eff. elast.' = beta0 + beta_k (mean elasticity when variable=1).")
            print("  Sigma captures site-level variation in the AADT-crash relationship.")

    # â”€â”€ NB dispersion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if model == 'nb':
        theta = np.exp(params[idx])
        print(f"\n{'=' * W}")
        print("  NEGATIVE BINOMIAL DISPERSION")
        print(f"{'=' * W}")
        print(f"  theta = {theta:.4f}  ->  Var[Y] = mu + mu^2/theta")
        print(f"  Smaller theta = greater overdispersion.")

    # â”€â”€ Consolidated random-parameter summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    any_random = any(rand_baseline) or any(rand_local)
    if any_random:
        print(f"\n{'=' * W}")
        print("  RANDOM PARAMETER SUMMARY  (all heterogeneous variables)")
        print(f"{'=' * W}")
        print(f"\n  {'Variable':<{col_w}}  {'Comp':>5}  {'Mean coef':>10}  "
              f"{'sigma':>8}  {'95% range'}")
        print("  " + "-" * 60)

        p2 = 0; p2 += 1  # skip alpha0
        for k, v in enumerate(selected_baseline):
            a = params[p2]; p2 += 1
            if rand_baseline[k]:
                sig = params[p2]; p2 += 1
                sig_abs = abs(sig)
                lo, hi = np.exp(a - 1.96*sig_abs), np.exp(a + 1.96*sig_abs)
                print(f"  {v:<{col_w}}  {'A':>5}  {a:>+10.4f}  {sig_abs:>8.4f}  "
                      f"CMF in [{lo:.4f}, {hi:.4f}]")

        p2 += 1  # skip beta0
        for k, v in enumerate(selected_local):
            b = params[p2]; p2 += 1
            if rand_local[k]:
                sig  = params[p2]; p2 += 1
                sig_abs = abs(sig)
                eff  = beta0 + b
                e_lo = eff - 1.96*sig_abs
                e_hi = eff + 1.96*sig_abs
                print(f"  {v:<{col_w}}  {'B':>5}  {b:>+10.4f}  {sig_abs:>8.4f}  "
                      f"elast. in [{e_lo:.4f}, {e_hi:.4f}]")

    print("\n" + "=" * W + "\n")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":

    data = pd.read_csv('Ex-16-3.csv')

    baseline_vars = ["URB", "HISNOW", "SLOPE"]
    local_vars    = ["GBRPM", "EXPOSE", "INTPM", "SPEED", "INCLANES"]

    print("=" * 72)
    print("  GA: joint variable + model + random-parameter selection  (JAX backend)")
    print("  Each variable independently: excluded / fixed / random")
    print("=" * 72)

    (selected_baseline, selected_local,
     rand_baseline, rand_local,
     use_halton, model, fitness) = run_ga(data, baseline_vars, local_vars, R=200)

    print("\n" + "-" * 72)
    print("  GA BEST SOLUTION")
    print("-" * 72)
    print(f"  {'Variable':<14}  {'Component':>10}  {'Status':>8}")
    print("  " + "-" * 38)
    for v, r in zip(selected_baseline, rand_baseline):
        print(f"  {v:<14}  {'A':>10}  {'RANDOM' if r else 'fixed':>8}")
    for v, r in zip(selected_local, rand_local):
        print(f"  {v:<14}  {'B':>10}  {'RANDOM' if r else 'fixed':>8}")
    n_rnd = sum(rand_baseline) + sum(rand_local)
    print(f"\n  Model         : {model.upper()}")
    print(f"  Halton draws  : {use_halton}")
    print(f"  Random params : {n_rnd}  "
          f"({sum(rand_baseline)} in A,  {sum(rand_local)} in B)")
    print("-" * 72)

    # â”€â”€ Final estimation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nFitting final model (R=500, JAX) â€¦")
    (result,
     y_jax, AADT_jax, baseline_jax, locals_jax,
     draws) = fit_final_model(
        data, selected_baseline, selected_local,
        rand_baseline, rand_local,
        model=model, use_halton=use_halton, R=500
    )

    # â”€â”€ Exact standard errors from JAX Hessian â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\nComputing exact standard errors (JAX Hessian) â€¦")
    se, cov = compute_se(
        result, y_jax, AADT_jax, baseline_jax, locals_jax,
        rand_baseline, rand_local, draws, R=500, model=model
    )

    # â”€â”€ Coefficient table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df = build_summary_table(result, selected_baseline, selected_local,
                             rand_baseline, rand_local, model, se)
    print_summary_table(df)

    # â”€â”€ CMF table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    AADT_mean = data["AADT"].mean()
    print_cmf_results(result, selected_baseline, selected_local,
                      rand_baseline, rand_local,
                      AADT_mean, model=model)
