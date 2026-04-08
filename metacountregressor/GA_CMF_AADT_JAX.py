# ============================================================
# Hierarchical SPF  —  GA Variable + Random Parameter Selection
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
#   1. JIT compilation via jax.jit  — likelihood evaluated ~10-50x faster
#   2. Exact analytic gradients via jax.value_and_grad  — no finite differences
#   3. Exact Hessian at solution via jax.hessian  — precise standard errors
#   4. GPU/TPU support at zero extra cost
#
# Optimizer:
#   scipy.optimize.minimize (L-BFGS-B) receives the JAX-computed gradient
#   directly via jac=True.  No jaxopt dependency required.
#
# GA gene layout  (total = 2*(k_base + k_loc) + 2):
#   [0 .. k_base-1]              inclusion flags  — baseline vars
#   [k_base .. k_base+k_loc-1]   inclusion flags  — local vars
#   [k_base+k_loc]               use_halton  (0=random, 1=Halton)
#   [k_base+k_loc+1]             model       (0=Poisson, 1=NB)
#   [k_base+k_loc+2 .. +k_base+1] random flags — baseline vars
#   [.. +k_loc]                  random flags — local vars
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
import pygad

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

# Enable 64-bit floats — essential for numerical optimisation
jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────
# HALTON / RANDOM DRAWS
# (generated in NumPy once, then converted to JAX arrays)
# ─────────────────────────────────────────────────────────────

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
        draws_np = np.empty((R, n, n_rand))
        for draw_idx in range(R):
            uniform_draws = sampler.random(n)
            clipped = np.clip(uniform_draws, 1e-12, 1 - 1e-12)
            draws_np[draw_idx] = norm.ppf(clipped)
    else:
        draws_np = norm.ppf(np.random.rand(R, n, n_rand))

    return jnp.array(draws_np)   # hand off to JAX


# ─────────────────────────────────────────────────────────────
# PARAMETER VECTOR LAYOUT
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# JAX LOG-LIKELIHOODS  (JIT-compiled)
# ─────────────────────────────────────────────────────────────
#
# rand_baseline / rand_local are Python tuples of bools — they control
# static code structure so they are declared as static_argnums.
# Data arrays (y, AADT, matrices, draws) are regular traced arguments.
#
# Two entry-points:
#   _fixed_ll  — pure fixed effects, no draws needed
#   _mixed_ll  — simulation-based, draws required


@partial(jax.jit, static_argnames=('rand_baseline', 'rand_local', 'model'))
def _fixed_ll(params, y, AADT, baseline_mat, locals_mat,
              rand_baseline, rand_local, model):
    """
    Closed-form log-likelihood — called when all random flags are False.
    JIT-compiled: ~2-5x faster than NumPy on CPU for moderate N.
    """
    k_base = baseline_mat.shape[1]
    k_loc  = locals_mat.shape[1]

    idx = 0
    alpha0 = params[idx]; idx += 1

    # Unpack alpha (all fixed here, no sigma entries)
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
    log_mu = jnp.clip(log_A + B * jnp.log(AADT), -30.0, 30.0)
    mu     = jnp.exp(log_mu)

    if model == 'poisson':
        ll = y * log_mu - mu - gammaln(y + 1.0)
    else:
        log_theta = params[idx]
        theta = jnp.exp(log_theta)
        ll = (  gammaln(y + 1.0/theta)
              - gammaln(1.0/theta)
              - gammaln(y + 1.0)
              + y * (log_mu + jnp.log(theta))
              - (y + 1.0/theta) * jnp.log(1.0 + theta * mu))

    return -jnp.sum(ll)


@partial(jax.jit, static_argnames=('rand_baseline', 'rand_local', 'model'))
def _mixed_ll(params, y, AADT, baseline_mat, locals_mat,
              rand_baseline, rand_local, draws, model):
    """
    Simulation-based log-likelihood with per-variable random effects.
    JIT-compiled: the loop over variables unrolls at trace time.

    draws : (R, N, n_rand)  standard-normal, pre-generated
    """
    R      = draws.shape[0]
    N      = len(y)
    k_base = baseline_mat.shape[1]
    k_loc  = locals_mat.shape[1]

    # ── Unpack parameters ─────────────────────────────────────
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

    if model == 'nb':
        theta = jnp.exp(params[idx])

    log_AADT = jnp.log(AADT)   # (N,)

    # ── Component A  —  shape (R, N) ─────────────────────────
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

    # ── Component B  —  shape (R, N) ─────────────────────────
    B = jnp.full((R, N), beta0)
    for k in range(k_loc):
        xk = locals_mat[:, k][None, :]
        if rand_local[k]:
            v = draws[:, :, draw_col]
            B = B + (beta[k] + sigma_beta[k] * v) * xk
            draw_col += 1
        else:
            B = B + beta[k] * xk

    log_mu = jnp.clip(log_A + B * log_AADT[None, :], -30.0, 30.0)
    mu     = jnp.exp(log_mu)

    if model == 'poisson':
        sim_ll = y[None, :] * log_mu - mu - gammaln(y + 1.0)[None, :]
    else:
        sim_ll = (  gammaln(y[None, :] + 1.0/theta)
                  - gammaln(1.0/theta)
                  - gammaln(y[None, :] + 1.0)
                  + y[None, :] * (log_mu + jnp.log(theta))
                  - (y[None, :] + 1.0/theta) * jnp.log(1.0 + theta * mu))

    loglik_i = logsumexp(sim_ll, axis=0) - jnp.log(R)
    return -jnp.sum(loglik_i)


def spf_loglike(params, y, AADT, baseline_mat, locals_mat,
                rand_baseline, rand_local, draws=None, R=200, model='poisson'):
    """
    Dispatcher: routes to fixed or mixed JIT kernel.
    rand_baseline / rand_local must be tuples (hashable for JIT caching).
    """
    rand_baseline = tuple(rand_baseline)
    rand_local    = tuple(rand_local)

    simulate = any(rand_baseline) or any(rand_local)

    if not simulate:
        return _fixed_ll(params, y, AADT, baseline_mat, locals_mat,
                         rand_baseline, rand_local, model)
    else:
        return _mixed_ll(params, y, AADT, baseline_mat, locals_mat,
                         rand_baseline, rand_local, draws, model)


# ─────────────────────────────────────────────────────────────
# SCIPY OPTIMIZER BRIDGE
# ─────────────────────────────────────────────────────────────
#
# jax.value_and_grad returns (loss, gradient) in a single forward+backward
# pass.  Wrapping it for scipy.minimize(jac=True) gives exact gradients
# with zero extra cost over evaluating the loss alone.

def make_objective(y_jax, AADT_jax, baseline_jax, locals_jax,
                   rand_baseline, rand_local, draws, R, model):
    """
    Returns a (value, gradient) callable compatible with scipy L-BFGS-B.
    Gradients are exact (JAX autodiff), not finite-difference approximations.
    """
    rand_baseline_t = tuple(rand_baseline)
    rand_local_t    = tuple(rand_local)
    simulate        = any(rand_baseline_t) or any(rand_local_t)

    if not simulate:
        raw_fn = lambda p: _fixed_ll(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, model)
    else:
        raw_fn = lambda p: _mixed_ll(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, draws, model)

    val_and_grad = jax.value_and_grad(raw_fn)

    def objective(p_np):
        p   = jnp.array(p_np)
        v, g = val_and_grad(p)
        return float(v), np.array(g, dtype=np.float64)

    return objective


def run_optimizer(objective, p0, method="L-BFGS-B", maxiter=2000, disp=False):
    """
    Thin wrapper around scipy.optimize.minimize.
    objective must return (value, gradient) — i.e. jac=True.
    """
    return minimize(
        objective,
        p0,
        method=method,
        jac=True,
        options={'maxiter': maxiter, 'disp': disp}
    )


# ─────────────────────────────────────────────────────────────
# STANDARD ERRORS  (exact Hessian via JAX autodiff)
# ─────────────────────────────────────────────────────────────

def compute_se(result, y_jax, AADT_jax, baseline_jax, locals_jax,
               rand_baseline, rand_local, draws, R, model):
    """
    Compute standard errors from the observed information matrix.

    Uses jax.hessian (exact second derivatives) at the MLE, which is
    far more accurate than the L-BFGS-B inverse-Hessian approximation.

    Returns (se, cov) as numpy arrays.
    """
    rand_baseline_t = tuple(rand_baseline)
    rand_local_t    = tuple(rand_local)
    simulate        = any(rand_baseline_t) or any(rand_local_t)

    if not simulate:
        raw_fn = lambda p: _fixed_ll(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, model)
    else:
        raw_fn = lambda p: _mixed_ll(
            p, y_jax, AADT_jax, baseline_jax, locals_jax,
            rand_baseline_t, rand_local_t, draws, model)

    hess_fn = jax.jit(jax.hessian(raw_fn))
    p_star  = jnp.array(result.x)

    try:
        H   = np.array(hess_fn(p_star))
        cov = np.linalg.inv(H)
        se  = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        print("  WARNING: Hessian singular — falling back to L-BFGS-B approx.")
        try:
            cov = result.hess_inv.todense()
        except AttributeError:
            cov = np.array(result.hess_inv)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    return se, cov


# ─────────────────────────────────────────────────────────────
# MODEL EVALUATOR  (called inside GA fitness)
# ─────────────────────────────────────────────────────────────

def evaluate_model(solution,
                   data,
                   baseline_vars,
                   local_vars,
                   use_halton=True,
                   model='poisson',
                   rand_baseline_all=None,
                   rand_local_all=None,
                   R=200):

    k_base = len(baseline_vars)
    k_loc  = len(local_vars)

    baseline_mask = solution[:k_base].astype(bool)
    local_mask    = solution[k_base:k_base+k_loc].astype(bool)

    active_baseline = [v for v, m in zip(baseline_vars, baseline_mask) if m]
    active_local    = [v for v, m in zip(local_vars,    local_mask)    if m]

    rand_baseline = tuple(r for r, m in zip(rand_baseline_all, baseline_mask) if m)
    rand_local    = tuple(r for r, m in zip(rand_local_all,    local_mask)    if m)

    k_base_act = len(active_baseline)
    k_loc_act  = len(active_local)

    y_np    = np.asarray(data["FREQ"],  dtype=np.float64)
    AADT_np = np.asarray(data["AADT"],  dtype=np.float64)
    N       = len(y_np)

    bmat_np = data[active_baseline].values.astype(np.float64) if k_base_act > 0 else np.zeros((N, 0))
    lmat_np = data[active_local].values.astype(np.float64)    if k_loc_act  > 0 else np.zeros((N, 0))

    # Convert to JAX arrays once — reused across optimizer iterations
    y_jax       = jnp.array(y_np)
    AADT_jax    = jnp.array(AADT_np)
    baseline_jax = jnp.array(bmat_np)
    locals_jax   = jnp.array(lmat_np)

    n_rand = sum(rand_baseline) + sum(rand_local)
    draws  = generate_draws(N, R, n_rand, use_halton)

    n_params = count_params(rand_baseline, rand_local, model)
    p0       = np.zeros(n_params)

    objective = make_objective(y_jax, AADT_jax, baseline_jax, locals_jax,
                               rand_baseline, rand_local, draws, R, model)

    res = run_optimizer(objective, p0, method="L-BFGS-B", maxiter=1000)

    ll  = -res.fun
    bic = -2.0 * ll + n_params * np.log(N)

    # Significance penalty — use fast fixed-SE from L-BFGS-B inverse here
    try:
        cov_approx = res.hess_inv.todense()
    except AttributeError:
        cov_approx = np.atleast_2d(res.hess_inv)
    se_approx = np.sqrt(np.maximum(np.diag(cov_approx), 0.0))

    params    = res.x
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

    return bic + penalty


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


# ─────────────────────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────────────────────

def run_ga(data, baseline_vars, local_vars, R=200):
    k_base    = len(baseline_vars)
    k_loc     = len(local_vars)
    offset_rf = k_base + k_loc + 2

    def fitness_func(ga_instance, solution, sol_idx):
        use_halton = bool(solution[k_base + k_loc])
        model      = "poisson" if int(solution[k_base + k_loc + 1]) == 0 else "nb"

        rand_baseline_all = [bool(solution[offset_rf + i])           for i in range(k_base)]
        rand_local_all    = [bool(solution[offset_rf + k_base + i])  for i in range(k_loc)]

        fitness = evaluate_model(
            solution, data, baseline_vars, local_vars,
            use_halton=use_halton, model=model,
            rand_baseline_all=rand_baseline_all,
            rand_local_all=rand_local_all,
            R=R
        )
        return -fitness   # PyGAD maximises; we minimise BIC+penalty

    n_genes    = 2 * (k_base + k_loc) + 2
    gene_space = [[0, 1]] * n_genes

    ga = pygad.GA(
        num_generations        = 50,
        sol_per_pop            = 10,
        num_parents_mating     = 2,
        num_genes              = n_genes,
        gene_space             = gene_space,
        gene_type              = int,
        fitness_func           = fitness_func,
        mutation_percent_genes = 20,
        crossover_type         = "uniform"
    )

    ga.run()
    solution, fitness, _ = ga.best_solution()

    baseline_mask     = solution[:k_base].astype(bool)
    local_mask        = solution[k_base:k_base+k_loc].astype(bool)
    use_halton        = bool(solution[k_base + k_loc])
    model             = "poisson" if int(solution[k_base + k_loc + 1]) == 0 else "nb"
    rand_baseline_all = [bool(solution[offset_rf + i])           for i in range(k_base)]
    rand_local_all    = [bool(solution[offset_rf + k_base + i])  for i in range(k_loc)]

    selected_baseline = [v for v, m in zip(baseline_vars, baseline_mask) if m]
    selected_local    = [v for v, m in zip(local_vars,    local_mask)    if m]
    rand_baseline     = tuple(r for r, m in zip(rand_baseline_all, baseline_mask) if m)
    rand_local        = tuple(r for r, m in zip(rand_local_all,    local_mask)    if m)

    return (selected_baseline, selected_local,
            rand_baseline, rand_local,
            use_halton, model, fitness)


# ─────────────────────────────────────────────────────────────
# FINAL ESTIMATION
# ─────────────────────────────────────────────────────────────

def fit_final_model(data,
                    selected_baseline, selected_local,
                    rand_baseline, rand_local,
                    model='poisson', use_halton=True, R=500):
    """
    Re-estimates the GA-selected model at higher R for final inference.
    Returns (result, y_jax, AADT_jax, baseline_jax, locals_jax, draws)
    so the caller can compute exact Hessian SE without re-building arrays.
    """
    y_np    = np.asarray(data["FREQ"],  dtype=np.float64)
    AADT_np = np.asarray(data["AADT"],  dtype=np.float64)
    N       = len(y_np)

    k_base = len(selected_baseline)
    k_loc  = len(selected_local)

    bmat_np = data[selected_baseline].values.astype(np.float64) if k_base > 0 else np.zeros((N, 0))
    lmat_np = data[selected_local].values.astype(np.float64)    if k_loc  > 0 else np.zeros((N, 0))

    y_jax        = jnp.array(y_np)
    AADT_jax     = jnp.array(AADT_np)
    baseline_jax = jnp.array(bmat_np)
    locals_jax   = jnp.array(lmat_np)

    n_rand = sum(rand_baseline) + sum(rand_local)
    draws  = generate_draws(N, R, n_rand, use_halton)

    p0 = np.zeros(count_params(rand_baseline, rand_local, model))

    objective = make_objective(y_jax, AADT_jax, baseline_jax, locals_jax,
                               rand_baseline, rand_local, draws, R, model)

    print("  Compiling JAX kernel (first call) …")
    res = run_optimizer(objective, p0, method="L-BFGS-B", maxiter=2000, disp=True)

    return res, y_jax, AADT_jax, baseline_jax, locals_jax, draws


# ─────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def build_summary_table(result,
                        selected_baseline, selected_local,
                        rand_baseline, rand_local,
                        model,
                        se):
    """
    se : numpy array of standard errors (from exact Hessian or fallback).
    """
    params = result.x
    z      = np.where(se > 1e-12, params / se, 0.0)
    p      = 2 * (1 - norm.cdf(np.abs(z)))
    labels = _param_labels(selected_baseline, selected_local, rand_baseline, rand_local, model)

    rows = list(zip(labels, params, se, z, p))
    return pd.DataFrame(rows, columns=["Parameter", "Estimate", "Std.Err", "z", "p-value"])


def print_summary_table(df):
    def stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    df = df.copy()
    df["Signif"] = df["p-value"].apply(stars)
    print("\n================ MODEL SUMMARY ================\n")
    print(df.to_string(index=False, float_format="%.4f"))
    print("\nSignificance: *** p<0.01,  ** p<0.05,  * p<0.10")
    print("sigma_* rows: standard deviation of the random parameter distribution")
    print("SE computed from exact observed information matrix (JAX Hessian)\n")


# ─────────────────────────────────────────────────────────────
# CMF REPORTING  —  traditional road-safety format
# ─────────────────────────────────────────────────────────────

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
    params = result.x
    idx    = 0
    W      = 72
    col_w  = 20

    print("\n" + "=" * W)
    print("   CRASH MODIFICATION FACTORS  —  Traditional Reporting Format")
    print("=" * W)
    print(f"   AADT reference (sample mean) : {AADT_mean:,.0f} veh/day")
    print(f"   Model                        : {model.upper()}")
    print()
    print("   [R] = RANDOM parameter  — effect varies by site")
    print("   [F] = FIXED  parameter  — effect constant across all sites")

    # ── Component A ───────────────────────────────────────────
    alpha0 = params[idx]; idx += 1

    print(f"\n{'=' * W}")
    print("  COMPONENT A  —  Site Characteristics  [CMF = exp(alpha_k)]")
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
            lo  = np.exp(a - 1.96 * sig)
            hi  = np.exp(a + 1.96 * sig)
            print(f"  {'':>{col_w}}       sigma = {sig:.4f}  ->  95% CMF range: [{lo:.4f}, {hi:.4f}]")
            print(f"  {'':>{col_w}}       Mean CMF = {cmf:.4f}  "
                  f"(sites range {(lo-1)*100:+.1f}% to {(hi-1)*100:+.1f}%)")

    if any(rand_baseline):
        print()
        print("  NOTE: Mean CMF is the average effect; sigma captures site heterogeneity.")

    # ── Component B ───────────────────────────────────────────
    beta0 = params[idx]; idx += 1

    print(f"\n{'=' * W}")
    print("  COMPONENT B  —  AADT Elasticity  [CMF = AADT_mean ^ beta_k]")
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
                e_lo   = elasticity - 1.96 * sig
                e_hi   = elasticity + 1.96 * sig
                cmf_lo = float(AADT_mean) ** (b - 1.96 * sig)
                cmf_hi = float(AADT_mean) ** (b + 1.96 * sig)
                print(f"  {'':>{col_w}}       sigma = {sig:.4f}  ->  "
                      f"95% elast. range: [{e_lo:.4f}, {e_hi:.4f}]")
                print(f"  {'':>{col_w}}       95% CMF @ mean AADT:  [{cmf_lo:.4f}, {cmf_hi:.4f}]")

        if any(rand_local):
            print()
            print("  NOTE: 'Eff. elast.' = beta0 + beta_k (mean elasticity when variable=1).")
            print("  Sigma captures site-level variation in the AADT-crash relationship.")

    # ── NB dispersion ─────────────────────────────────────────
    if model == 'nb':
        theta = np.exp(params[idx])
        print(f"\n{'=' * W}")
        print("  NEGATIVE BINOMIAL DISPERSION")
        print(f"{'=' * W}")
        print(f"  theta = {theta:.4f}  ->  Var[Y] = mu + mu^2/theta")
        print(f"  Smaller theta = greater overdispersion.")

    # ── Consolidated random-parameter summary ─────────────────
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
                lo, hi = np.exp(a - 1.96*sig), np.exp(a + 1.96*sig)
                print(f"  {v:<{col_w}}  {'A':>5}  {a:>+10.4f}  {sig:>8.4f}  "
                      f"CMF in [{lo:.4f}, {hi:.4f}]")

        p2 += 1  # skip beta0
        for k, v in enumerate(selected_local):
            b = params[p2]; p2 += 1
            if rand_local[k]:
                sig  = params[p2]; p2 += 1
                eff  = beta0 + b
                e_lo = eff - 1.96*sig
                e_hi = eff + 1.96*sig
                print(f"  {v:<{col_w}}  {'B':>5}  {b:>+10.4f}  {sig:>8.4f}  "
                      f"elast. in [{e_lo:.4f}, {e_hi:.4f}]")

    print("\n" + "=" * W + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

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

    # ── Final estimation ─────────────────────────────────────
    print("\nFitting final model (R=500, JAX) …")
    (result,
     y_jax, AADT_jax, baseline_jax, locals_jax,
     draws) = fit_final_model(
        data, selected_baseline, selected_local,
        rand_baseline, rand_local,
        model=model, use_halton=use_halton, R=500
    )

    # ── Exact standard errors from JAX Hessian ───────────────
    print("\nComputing exact standard errors (JAX Hessian) …")
    se, cov = compute_se(
        result, y_jax, AADT_jax, baseline_jax, locals_jax,
        rand_baseline, rand_local, draws, R=500, model=model
    )

    # ── Coefficient table ─────────────────────────────────────
    df = build_summary_table(result, selected_baseline, selected_local,
                             rand_baseline, rand_local, model, se)
    print_summary_table(df)

    # ── CMF table ─────────────────────────────────────────────
    AADT_mean = data["AADT"].mean()
    print_cmf_results(result, selected_baseline, selected_local,
                      rand_baseline, rand_local,
                      AADT_mean, model=model)
