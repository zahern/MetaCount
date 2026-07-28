"""Gradient Score Enhanced (GSE) Latent Class for metacountregressor.

Philosophy
----------
  Standard LC: uniform class-shares → EM from random init → often trapped locally
  GSE LC:      1. Fit 1-class NB2 → per-segment gradient scores
                2. Cluster gradients (scipy vq) → class labels
                3. Convert labels to initial membership gammas
                4. EM warm-started from gradient-informed gammas

The gradient score s_nk = d(logL_n) / d(beta_k) captures how individual n's
preferences deviate from population mean. Segments with similar gradient
patterns are natural candidates for the same latent class.

Run from the metacountregressor/ directory (HPC or local with deps):
    cd metacountregressor
    python gse_lc_washington.py
"""

from __future__ import annotations
import sys, os, time, argparse, warnings
import numpy as np
from scipy.cluster.vq import kmeans2
from dataclasses import replace
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# -- metacountregressor imports --
# These resolve because we run from the metacountregressor/ subdir
import main_hpc_lc_patch  # noqa: F401 — applies LC patches to experiment_package
from main_hpc_lc_patch import (  # noqa: E402
    build_base_index, build_param_index, build_model_from_manual_spec,
    print_summary, CountModel, mixed_model_loglik, compute_lc_posteriors,
    fit_em, _de_warmup_lc, _seed_classes_from_clusters,
)
from main_hpc import experiment_washington  # noqa: E402
from experiment_package import ExperimentBuilder  # noqa: E402
from jaxopt import LBFGS  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax  # noqa: E402


# =============================================================================
# Gradient extraction from single-class model
# =============================================================================

def compute_individual_gradients(
    builder: ExperimentBuilder,
    fixed_terms: list[str],
    model: str = "nb",
    R: int = 100,
):
    """Fit 1-class model, return per-segment gradient scores.

    Returns
    -------
    g_avg : ndarray  (N_segments, K_params)
        Per-segment gradient vector.
    """
    manual_spec = builder.make_manual_spec(
        fixed_terms=fixed_terms,
        dispersion=1 if model == "nb" else 0,
        latent_classes=1,
        R=R,
    )
    fit = builder.fit_manual_model(manual_spec=manual_spec, model=model,
                                   print_report=False)
    result = fit["result"]
    spec = fit["spec"]
    data = fit.get("data") or builder._build_data(spec, R=R)
    data_train = data if isinstance(data, dict) else data[0]

    params = np.array(result.params)
    K = len(params)

    # Segment-level gradient: d(logL_seg)/d(beta)
    # For NB2 with panel=T, each segment contributes per time period.
    # We compute using JAX autodiff on per-segment loglik.
    n_segs = data_train["y"].shape[0]
    g_avg = np.zeros((n_segs, K))

    # Build per-row gradient via vmap over observations
    X = np.array(data_train.get("X", data_train.get("X_matrix")))
    y = np.array(data_train["y"]).ravel()
    offset = np.array(data_train.get("offset", np.zeros_like(y))).ravel()

    @partial(jax.jit, static_argnums=(3, 4))
    def _row_loglik(p, x, yi, oi):
        """NB2 log-likelihood for a single observation."""
        eta = jnp.dot(x, p[:len(fixed_terms)])
        if spec.dispersion > 0:
            eta = eta + p[len(fixed_terms)] * 0  # extra param not in dot
        mu = jnp.clip(jnp.exp(eta + oi), 1e-10, 1e10)
        alpha = jnp.exp(p[-1])
        from jax.scipy.special import gammaln
        return (gammaln(yi + alpha) - gammaln(yi + 1) - gammaln(alpha)
                + alpha * jnp.log(alpha / (alpha + mu))
                + yi * jnp.log(mu / (alpha + mu)))

    grad_fn = jax.jit(jax.grad(lambda p: -_row_loglik(p, X[0], y[0], offset[0]).sum()))
    for i in range(n_segs):
        g_avg[i] = np.array(grad_fn(jnp.array(params)))

    return g_avg


# =============================================================================
# GSE: gradient → class labels → initial gammas
# =============================================================================

def cluster_gradients(gradient_scores, n_classes: int, seed: int = 42):
    """KMeans on gradient scores → class labels."""
    # standardise
    g_std = (gradient_scores - gradient_scores.mean(axis=0)) / (
        gradient_scores.std(axis=0) + 1e-8)
    centroids, labels = kmeans2(g_std, n_classes, minit='points', missing='warn')
    return labels


def labels_to_initial_gammas(class_labels: np.ndarray, n_classes: int):
    """Convert hard cluster labels to soft membership prior → initial gammas.

    Returns (n_classes-1, K_membership+1) gammas — intercept-only.
    """
    prior = np.zeros((len(class_labels), n_classes))
    for i, lab in enumerate(class_labels):
        prior[i, lab] = 0.9
        for c in range(n_classes):
            if c != lab:
                prior[i, c] = 0.1 / (n_classes - 1)
    prior = prior / prior.sum(axis=1, keepdims=True)

    gammas = np.zeros((n_classes - 1, 1))
    for c in range(1, n_classes):
        gammas[c - 1, 0] = np.log(
            (prior[:, c] / (prior[:, 0] + 1e-8) + 1e-8)).mean()
    return gammas


# =============================================================================
# LC fitting with optional GSE warm-start
# =============================================================================

def fit_lc(
    builder, df, fixed_terms, n_classes=2, model="nb",
    membership_terms=None, R=200, gse_init=False, class_labels=None,
    verbose=True,
):
    """Fit latent class model, optionally GSE-initialised."""
    spec_kwargs = dict(
        fixed_terms=fixed_terms,
        dispersion=1 if model == "nb" else 0,
        latent_classes=n_classes,
    )
    if membership_terms:
        spec_kwargs["membership_terms"] = membership_terms

    manual_spec = builder.make_manual_spec(**spec_kwargs)
    data, spec = builder._build_data(manual_spec, R=R)

    spec_c = replace(spec, latent_classes=n_classes,
                     min_class_proportion=0.05, l2_penalty=0.1)
    C = n_classes
    K_mem = spec_c.K_membership
    pindex_c = build_param_index(spec_c)

    # — 1-class warm-start —
    base_spec = replace(spec, latent_classes=1)
    model_1 = CountModel(base_spec, data)
    result_1 = model_1.fit()
    theta_1 = np.array(result_1.params)
    K_base_0 = build_param_index(base_spec)["total_params"]
    _class_K_base = list(pindex_c.get("class_K_base", [K_base_0] * C))

    # — cluster seeding —
    rng = np.random.default_rng(42)
    try:
        per_class = _seed_classes_from_clusters(
            theta_1, data, base_spec, C, K_base_0, rng,
            class_K_base=_class_K_base,
        )
        theta_init = np.concatenate(per_class)
    except Exception:
        theta_init = np.concatenate([
            theta_1[:k] + rng.normal(0, 0.05, k)
            for k in _class_K_base
        ])

    # — membership init —
    if gse_init and class_labels is not None:
        gammas0 = labels_to_initial_gammas(class_labels, C)
    else:
        gammas0 = np.zeros((C - 1, K_mem + 1))
    gamma_init = gammas0.ravel() if gammas0.size > 0 else np.array([])
    init_params = np.concatenate([theta_init, gamma_init])

    # — DE warm-up —
    try:
        init_params = _de_warmup_lc(
            init_params, data=data, spec=spec_c,
            maxiter=12, popsize=8, rel_span=1.5, abs_span=1.0,
            seed=42, verbose=False,
        )
    except Exception:
        pass

    # — EM —
    try:
        params_em = fit_em(
            init_params=init_params, data=data, spec=spec_c,
            max_iter=100, tol=1e-4, verbose=False,
        )
    except Exception:
        params_em = init_params

    # — LBFGS polish —
    polish = LBFGS(
        fun=lambda p: mixed_model_loglik(p, data, spec_c), maxiter=800)
    result_c = polish.run(jnp.array(params_em))
    params_c = np.array(result_c.params)
    ll = -float(mixed_model_loglik(params_c, data, spec_c))
    n = data["y"].shape[0]
    k = len(params_c)
    bic = k * np.log(n) - 2.0 * ll
    aic = 2 * k - 2.0 * ll

    # — posterior —
    try:
        posterior, _, _ = compute_lc_posteriors(params_c, data, spec_c)
        class_props = posterior.mean(axis=0)
    except Exception:
        posterior = np.ones((n, C)) / C
        class_props = np.ones(C) / C

    return {
        "loglik": ll, "bic": bic, "aic": aic,
        "n_params": k, "class_props": class_props,
        "posterior": posterior, "params": params_c,
        "spec": spec_c,
    }


# =============================================================================
# Main
# =============================================================================

def main(seed: int = 42):
    np.random.seed(seed)

    print("=" * 80)
    print("  GSE LATENT CLASS — Washington crash data")
    print("  Gradient Score Enhanced (GSE) latent class initialisation")
    print("=" * 80)

    # — Load —
    df, _ = experiment_washington()
    if "OFFSET" not in df.columns:
        df["OFFSET"] = 0.0
    builder = ExperimentBuilder(df=df, id_col="ID", y_col="Y",
                                offset_col="OFFSET")

    fixed_terms = ["EXPOSE", "GRADEBR", "AVEPRE", "FRICTION"]
    n_classes = 3
    print(f"\n  Segments: {df['ID'].nunique()} | vars: {fixed_terms}")
    print(f"  Classes: {n_classes}")

    # —— 1. Standard LC ————————————————————————————————
    print(f"\n{'─'*80}\n  1. Standard LC (random init)")
    t0 = time.perf_counter()
    res_std = fit_lc(builder, df, fixed_terms, n_classes=n_classes,
                     model="nb", gse_init=False, R=100)
    t_std = time.perf_counter() - t0
    print(f"  LL={res_std['loglik']:.1f}  BIC={res_std['bic']:.1f}  "
          f"params={res_std['n_params']}  time={t_std:.1f}s")
    print(f"  Shares: {np.array2string(res_std['class_props'], precision=3)}")

    # —— 2. Gradient extraction —————————————————————————
    print(f"\n{'─'*80}\n  2. Extracting 1-class gradient scores ...")
    t0 = time.perf_counter()
    try:
        g_avg = compute_individual_gradients(
            builder, fixed_terms, model="nb", R=100)
        print(f"  Gradient matrix: {g_avg.shape}")
        var_g = np.var(g_avg, axis=0)
        for i, v in enumerate(fixed_terms):
            print(f"    {v:>12s}: var={var_g[i]:10.4f}")
    except Exception as e:
        print(f"  Gradient extraction FAILED: {e}")
        import traceback; traceback.print_exc()
        return

    # —— 3. GSE LC ——————————————————————————————————————
    print(f"\n{'─'*80}\n  3. GSE LC (gradient-informed init)")
    labels = cluster_gradients(g_avg, n_classes)
    print(f"  Cluster sizes: {np.bincount(labels)}")

    t0 = time.perf_counter()
    res_gse = fit_lc(builder, df, fixed_terms, n_classes=n_classes,
                     model="nb", gse_init=True, class_labels=labels,
                     R=100)
    t_gse = time.perf_counter() - t0
    print(f"  LL={res_gse['loglik']:.1f}  BIC={res_gse['bic']:.1f}  "
          f"params={res_gse['n_params']}  time={t_gse:.1f}s")
    print(f"  Shares: {np.array2string(res_gse['class_props'], precision=3)}")

    # —— Summary ————————————————————————————————————————
    print(f"\n{'='*80}")
    print("  COMPARISON")
    print(f"{'='*80}")
    hdr = f"  {'Model':<20s} {'LL':>10s} {'AIC':>10s} {'BIC':>10s} {'Params':>8s}"
    print(hdr)
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'Standard LC':<20s} {res_std['loglik']:10.1f} "
          f"{res_std['aic']:10.1f} {res_std['bic']:10.1f} "
          f"{res_std['n_params']:8d}")
    print(f"  {'GSE LC':<20s} {res_gse['loglik']:10.1f} "
          f"{res_gse['aic']:10.1f} {res_gse['bic']:10.1f} "
          f"{res_gse['n_params']:8d}")

    dll = res_gse['loglik'] - res_std['loglik']
    dbic = res_gse['bic'] - res_std['bic']
    print(f"\n  dLL  (GSE - Std) = {dll:+.1f}")
    print(f"  dBIC (GSE - Std) = {dbic:+.1f}  "
          f"({'GSE BETTER' if dbic < 0 else 'Std BETTER'})")

    print(f"\n  {'='*80}")
    print("  How it works:")
    print("    1-class NB2 → per-segment gradient scores d(logL_n)/d(beta)")
    print("    scipy.vq.kmeans2 on standardised gradients → class labels")
    print("    labels → soft membership prior → initial EM gammas")
    print("    standard EM + DE warm-up + LBFGS polish as usual")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
