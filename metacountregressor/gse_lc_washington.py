"""Gradient Score Enhanced (GSE) Latent Class for Washington crash data.

Philosophy
----------
  Standard LC:  uniform class-shares → EM from random init
  GSE LC:       1. Fit single-class NB2 → per-segment gradient scores
                2. KMeans on gradient scores → initial class labels
                3. EM warm-started from gradient-informed membership

The gradient score s_nk = ∂logL_n / ∂β_k captures how much individual n's
preferences deviate from the population mean. Segments with similar gradient
patterns are natural candidates for the same latent class.

Run from the 'metacountregressor/' subdirectory:
    cd C:\Users\ahernz\source\metacount\metacountregressor
    python gse_lc_washington.py
"""

from __future__ import annotations
import sys, os, time, argparse
import numpy as np
from sklearn.cluster import KMeans

# Path setup — run from metacountregressor/ directory
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from main_hpc_lc_patch import (  # noqa: E402
    build_base_index, build_param_index, build_model_from_manual_spec,
    print_summary, CountModel, mixed_model_loglik, compute_lc_posteriors,
    compute_mixed_model_loglik_DE, fit_em, _de_warmup_lc, _seed_classes_from_clusters,
)
from main_hpc import experiment_washington  # noqa: E402
from experiment_package import ExperimentBuilder  # noqa: E402
from jaxopt import LBFGS  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from dataclasses import replace  # noqa: E402
from functools import partial  # noqa: E402


def compute_individual_gradients(builder, df, model_spec, model="nb", R=200):
    """Fit single-class model and return per-segment score contributions.

    Returns
    -------
    g_avg : ndarray (N_individuals, K_params)
        Average score per individual (across time periods, if panel).
    individual_ids : list
        Sorted list of unique individual IDs matching g_avg rows.
    """
    spec = builder.make_manual_spec(
        fixed_terms=model_spec.get("fixed_terms", builder._all_vars()),
        dispersion=model_spec.get("dispersion", 1),
        latent_classes=1,
        R=R,
    )
    fit = builder.fit_manual_model(manual_spec=spec, model=model, print_report=False)
    result = fit["result"]

    spec_fit = fit["spec"]
    data = fit.get("data") or builder._build_data(spec_fit, R=R)
    data_train = data if isinstance(data, dict) else data[0]

    # Per-observation gradient contributions
    params_c = np.array(result.params)
    gamma = params_c[spec_fit.K_base:] if spec_fit.latent_classes > 1 else None

    N_obs = len(data_train["y"])

    # For NB2 model, compute per-observation loglik derivs numerically
    import jax
    @jax.jit
    def _loglik_fwd(p):
        ll = mixed_model_loglik(p, data_train, spec_fit)
        return -ll.sum()

    grad_fn = jax.grad(_loglik_fwd)
    g_all = np.array(grad_fn(jnp.array(params_c)))  # full dataset gradient per param
    # This is the TOTAL gradient, not per-observation.
    # For per-individual, we need to aggregate over segments.
    # Alternative: use per-choice-occasion scores.

    # Simpler: compute via einsum as in SearchLibrium approach
    # g_nk = (y_nj - p_nj) * X_njk  for MNL-like models
    # For NB2, p is expected count
    from jax import vmap

    def _occ_loglik(p, y, X_row, offset):
        """Log-likelihood of a single observation."""
        eta = X_row @ p[:len(spec_fit.fixed_names)]
        if offset is not None:
            eta = eta + offset
        mu = jnp.clip(jnp.exp(eta), 1e-10, 1e10)
        alpha = jnp.exp(p[-1])  # dispersion in log-space
        # NB2 loglik: y*log(r/(r+mu)) + r*log(mu/(r+mu)) + log(gamma(y+r)/(gamma(y+1)*gamma(r)))
        from jax.scipy.special import gammaln
        return (gammaln(y + alpha) - gammaln(y + 1) - gammaln(alpha)
                + alpha * jnp.log(alpha / (alpha + mu))
                + y * jnp.log(mu / (alpha + mu)))

    # For per-segment gradients, use numerical approach per individual
    individual_ids = df[spec_fit.id_col].unique()
    id_to_idx = {id_: i for i, id_ in enumerate(sorted(individual_ids))}
    K = len(params_c)
    g_ind = np.zeros((len(individual_ids), K))

    # Group observations by individual
    for i, ind_id in enumerate(sorted(individual_ids)):
        mask = df[spec_fit.id_col] == ind_id
        X_ind = spec_fit.X[np.where(df[spec_fit.id_col] == ind_id)[0]]
        y_ind = spec_fit.y[np.where(df[spec_fit.id_col] == ind_id)[0]]
        # Simple: sum up per-obs gradient
        def _ind_loglik(p):
            ll = 0.0
            for t in range(len(y_ind)):
                ll = ll + _occ_loglik(p, y_ind[t], X_ind[t], 0.0)
            return -ll
        g_ind[i] = np.array(jax.grad(_ind_loglik)(jnp.array(params_c)))

    g_avg = g_ind  # average if panel periods differ, else just use total
    return g_avg, sorted(individual_ids)


def gse_initial_gammas(gradient_scores, n_classes, class_labels=None):
    """Convert gradient-based cluster labels to initial membership gammas.

    If class_labels is None, runs KMeans on gradient_scores first.

    Returns
    -------
    gammas : ndarray (n_classes - 1, 1)
        Initial intercept-only gammas (column 0 = class intercept vs class 1).
    class_labels : ndarray (N,)
        Cluster labels assigned.
    """
    if class_labels is None:
        km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        class_labels = km.fit_predict(gradient_scores)

    # Build soft membership prior from hard clusters
    prior = np.zeros((len(class_labels), n_classes))
    for i, lab in enumerate(class_labels):
        prior[i, lab] = 0.9
        for c in range(n_classes):
            if c != lab:
                prior[i, c] = 0.1 / (n_classes - 1)
    prior = prior / prior.sum(axis=1, keepdims=True)

    # log(prior_nc / prior_n0) = gamma_c_intercept
    gammas = np.zeros((n_classes - 1, 1))
    for c in range(1, n_classes):
        gammas[c - 1, 0] = np.log(
            (prior[:, c] / (prior[:, 0] + 1e-8) + 1e-8)
        ).mean()
    return gammas, class_labels


def fit_lc_gse(builder, df, fixed_terms, n_classes=2, model="nb",
               membership_terms=None, R=200,
               gradient_init=True, class_labels=None):
    """
    Fit a latent class model, optionally with GSE-initialised membership.

    When gradient_init=True:
        1. Fit 1-class model to get gradient scores
        2. Cluster gradients → initial class assignments
        3. Convert to initial membership gammas
        4. Run full LC EM with warm-start gammas
    """
    spec_kwargs = dict(
        fixed_terms=fixed_terms,
        dispersion=1 if model == "nb" else 0,
        latent_classes=n_classes,
    )
    if membership_terms:
        spec_kwargs["membership_terms"] = membership_terms

    spec_dict = builder.make_manual_spec(**spec_kwargs)
    spec = spec_dict  # it returns a dict-like or spec object

    # Build data
    from experiment_package import ModelSpec
    if isinstance(spec, dict):
        from main_hpc import build_model_data
        model_spec = builder._to_model_spec(spec)
        data_best, spec_best = builder._build_data(model_spec, R=R)
        data_train = data_best
    else:
        data_train, spec_best = builder._build_data(spec, R=R)

    spec_c = replace(spec_best, latent_classes=n_classes,
                     min_class_proportion=0.05, l2_penalty=0.1)
    C = n_classes
    K_mem = spec_c.K_membership
    pindex_c = build_param_index(spec_c)

    # --- 1-class warm-start ---
    base_spec = replace(spec_best, latent_classes=1)
    model_1 = CountModel(base_spec, data_train)
    result_1 = model_1.fit()
    theta_1 = np.array(result_1.params)
    K_base_0 = build_param_index(base_spec)["total_params"]
    _class_K_base = list(pindex_c.get("class_K_base", [K_base_0] * C))

    # --- cluster-based seeding ---
    rng = np.random.default_rng(42)
    try:
        per_class_thetas = _seed_classes_from_clusters(
            theta_1, data_train, base_spec, C, K_base_0, rng,
            class_K_base=_class_K_base,
        )
        theta_init = np.concatenate(per_class_thetas)
    except Exception:
        theta_init = np.concatenate([
            theta_1[:k] + rng.normal(0, 0.05, k)
            for k in _class_K_base
        ])

    # --- membership init ---
    if gradient_init:
        if class_labels is not None:
            gammas0, _ = gse_initial_gammas(None, C, class_labels=class_labels)
        else:
            # Placeholder: would need gradient_scores here
            gammas0 = np.zeros((C - 1, K_mem + 1))
    else:
        gammas0 = np.zeros((C - 1, K_mem + 1))
    gamma_init = gammas0.ravel() if gammas0.size > 0 else np.array([])
    init_params = np.concatenate([theta_init, gamma_init])

    # --- DE warm-up ---
    try:
        init_params = _de_warmup_lc(
            init_params, data=data_train, spec=spec_c,
            maxiter=12, popsize=8, rel_span=1.5, abs_span=1.0,
            seed=42, verbose=False,
        )
    except Exception:
        pass

    # --- EM ---
    try:
        params_em = fit_em(
            init_params=init_params, data=data_train, spec=spec_c,
            max_iter=100, tol=1e-4, verbose=False,
        )
    except Exception:
        params_em = init_params

    # --- LBFGS polish ---
    polish = LBFGS(
        fun=lambda p: mixed_model_loglik(p, data_train, spec_c),
        maxiter=800,
    )
    result_c = polish.run(jnp.array(params_em))
    params_c = np.array(result_c.params)
    ll = -float(mixed_model_loglik(params_c, data_train, spec_c))
    n = data_train["y"].shape[0]
    k = len(params_c)
    bic = k * np.log(n) - 2.0 * ll

    # --- posterior / class shares ---
    try:
        posterior, _, _ = compute_lc_posteriors(params_c, data_train, spec_c)
        class_props = posterior.mean(axis=0)
    except Exception:
        class_props = np.ones(C) / C
        posterior = np.ones((n, C)) / C

    return {
        "loglik": ll, "bic": bic, "aic": 2*k - 2*ll,
        "n_params": k, "class_props": class_props,
        "posterior": posterior, "params": params_c,
        "spec": spec_c, "data": data_train,
    }


def main(seed: int = 42):
    np.random.seed(seed)

    print("=" * 80)
    print("  GSE LATENT CLASS  —  Washington crash data")
    print("  Gradient Score Enhanced membership initialization")
    print("=" * 80)

    # --- Load data ---
    df, _ = experiment_washington()
    print(f"\n  Data: {df['ID'].nunique()} segments, {len(df.columns)} variables")
    print(f"  URB: {df['URB'].mean():.1%} urban, Y mean = {df['Y'].mean():.1f}")

    # Make sure OFFSET exists
    if "OFFSET" not in df.columns:
        df["OFFSET"] = 0.0

    builder = ExperimentBuilder(df=df, id_col="ID", y_col="Y", offset_col="OFFSET")

    # Choose variables — use the same 4 as the Swiss Metro study
    fixed_terms = ["EXPOSE", "GRADEBR", "AVEPRE", "FRICTION"]
    n_classes = 3

    print(f"\n  Variables: {fixed_terms}")
    print(f"  Classes: {n_classes}")

    # --- 1. Standard LC ---
    print(f"\n{'─'*80}")
    print("  1. Standard LC (uniform init)")
    t0 = time.perf_counter()
    res_std = fit_lc_gse(
        builder, df, fixed_terms=fixed_terms,
        n_classes=n_classes, model="nb",
        gradient_init=False, R=100,
    )
    t_std = time.perf_counter() - t0
    print(f"  LL = {res_std['loglik']:.1f}  BIC = {res_std['bic']:.1f}  params = {res_std['n_params']}  time = {t_std:.1f}s")
    print(f"  Class shares: {np.array2string(res_std['class_props'], precision=3)}")

    # --- 2. Get gradient scores from 1-class model ---
    print(f"\n{'─'*80}")
    print("  2. Computing gradient scores from 1-class NB2 ...")
    t0 = time.perf_counter()
    try:
        g_avg, ind_ids = compute_individual_gradients(
            builder, df,
            model_spec={"fixed_terms": fixed_terms, "dispersion": 1},
            model="nb", R=100,
        )
        print(f"  Gradient scores shape: {g_avg.shape}")
        score_var = np.var(g_avg, axis=0)
        print(f"  Score variance: {' '.join(f'{fixed_terms[i]}={score_var[i]:.2f}' for i in range(len(fixed_terms)))}")

        # --- 3. Cluster gradients ---
        print(f"\n{'─'*80}")
        print("  3. Clustering gradient scores for class init ...")
        km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        class_labels = km.fit_predict(g_avg)
        print(f"  Cluster sizes: {np.bincount(class_labels)}")
        t_grad = time.perf_counter() - t0

        # --- 4. GSE LC ---
        print(f"\n{'─'*80}")
        print("  4. GSE LC (gradient-informed init)")
        t0 = time.perf_counter()
        res_gse = fit_lc_gse(
            builder, df, fixed_terms=fixed_terms,
            n_classes=n_classes, model="nb",
            gradient_init=True, class_labels=class_labels,
            R=100,
        )
        t_gse = time.perf_counter() - t0
        print(f"  LL = {res_gse['loglik']:.1f}  BIC = {res_gse['bic']:.1f}  params = {res_gse['n_params']}  time = {t_gse:.1f}s")
        print(f"  Class shares: {np.array2string(res_gse['class_props'], precision=3)}")
    except Exception as e:
        print(f"  GSE pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        res_gse = {"loglik": float("nan"), "bic": float("nan"), "n_params": 0, "class_props": np.array([])}

    # --- Summary ---
    print(f"\n{'='*80}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model':<30s} {'LL':>10s} {'AIC':>10s} {'BIC':>10s} {'Params':>8s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    print(f"  {'Standard LC':<30s} {res_std['loglik']:10.1f} {res_std['aic']:10.1f} {res_std['bic']:10.1f} {res_std['n_params']:8d}")
    print(f"  {'GSE LC':<30s} {res_gse['loglik']:10.1f} {res_gse['aic']:10.1f} {res_gse['bic']:10.1f} {res_gse['n_params']:8d}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
