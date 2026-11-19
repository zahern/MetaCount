"""Halton vs Sobol draw comparison on synthetic latent-class count data.

Generates a 2-class NB2 panel dataset with membership covariates,
fits the true model specification using Halton and Sobol draws across
a range of draw counts (R), and compares:

    - Log-likelihood at convergence
    - Parameter recovery (MSE vs true values)
    - Wall time per fit
    - Convergence stability (LL variance across seeds)

Run from the metacountregressor/ directory:
    python test_halton_vs_sobol.py [--N 400] [--T 4] [--R_min 50] [--R_max 500] [--n_seeds 5]
"""

from __future__ import annotations

import sys, os, time, argparse
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial
from jaxopt import LBFGS
from scipy import stats as scipy_stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from metacountregressor import main_hpc_lc_patch  # noqa
from metacountregressor.main_hpc import (
    CountModel,
    build_base_index,
    generate_master_halton,
    generate_master_sobol,
    generate_master_draws,
)
from metacountregressor.main_hpc_lc_patch import (
    ModelSpec,
    build_param_index,
    build_model_from_manual_spec,
    mixed_model_loglik,
    fit_em,
    _seed_classes_from_clusters,
    unpack_lc_params,
)
from metacountregressor.experiment_package import StructureEvaluatorLC

jax.config.update("jax_enable_x64", True)

# ────────────────────────────────────────────────────────────────────
# True DGP
# ────────────────────────────────────────────────────────────────────
TRUE = {
    "class1": {"intercept": -3.0, "x1": 1.0, "x2": -0.8, "x3": 0.3, "x4": 0.1, "alpha": 1.5},
    "class2": {"intercept": -0.8, "x1": -0.4, "x2": 0.6, "x3": 0.2, "x4": 0.9, "alpha": 0.6},
    "membership": {"g0": 0.0, "gz1": 1.5, "gz2": -1.0},
}
OUTCOME_VARS = ["x1", "x2", "x3", "x4"]
MEMBERSHIP_VARS = ["z1", "z2"]

TRUE_THETA = np.array([
    # class 1: intercept, x1, x2, x3, x4, log(alpha)
    -3.0, 1.0, -0.8, 0.3, 0.1, np.log(1.5),
    # class 2: intercept, x1, x2, x3, x4, log(alpha)
    -0.8, -0.4, 0.6, 0.2, 0.9, np.log(0.6),
    # gamma: g0, gz1, gz2
    0.0, 1.5, -1.0,
])


def generate_data(N=400, T=4, seed=42):
    rng = np.random.default_rng(seed)
    mem = TRUE["membership"]
    rows = []
    for i in range(N):
        x1, x2, x3, x4 = rng.normal(0, 1, 4)
        z1, z2 = rng.normal(0, 1, 2)
        log_odds = mem["g0"] + mem["gz1"] * z1 + mem["gz2"] * z2
        p_c2 = 1.0 / (1.0 + np.exp(-log_odds))
        c = 2 if rng.random() < p_c2 else 1
        p = TRUE[f"class{c}"]
        for t in range(T):
            eta = p["intercept"] + p["x1"] * x1 + p["x2"] * x2 + p["x3"] * x3 + p["x4"] * x4
            mu = np.exp(np.clip(eta, -20, 20))
            alpha = p["alpha"]
            size = alpha
            prob = alpha / (alpha + mu)
            y = rng.negative_binomial(size, prob)
            rows.append({
                "ID": i + 1, "t": t + 1,
                "x1": x1, "x2": x2, "x3": x3, "x4": x4,
                "z1": z1, "z2": z2, "Y": int(y), "true_class": c,
            })
    return pd.DataFrame(rows)


def fit_model(df, manual_spec, draw_method="halton", R=200, seed=0):
    """Fit a 2-class LC model with the given draw method and R. Returns (params, ll, elapsed)."""
    data, spec = build_model_from_manual_spec(
        df=df, manual_spec=manual_spec,
        id_col="ID", y_col="Y", offset_col=None,
        draw_method=draw_method, R=R,
    )
    C = spec.latent_classes
    K_mem = spec.K_membership
    base_spec = replace(spec, latent_classes=1)
    pindex = build_param_index(spec)
    class_K_base = list(pindex.get("class_K_base", [build_base_index(base_spec)["total_params"]] * C))
    K_base_0 = class_K_base[0]

    # -- single-class warm start --
    t0 = time.perf_counter()
    model_1 = CountModel(base_spec, data)
    result_1 = model_1.fit()
    theta_1 = np.array(result_1.params)

    # -- cluster seeding --
    rng = np.random.default_rng(42)
    try:
        per_class = _seed_classes_from_clusters(
            theta_1, data, base_spec, C, K_base_0, rng,
            class_K_base=class_K_base,
        )
        theta_init = np.concatenate(per_class)
    except Exception:
        theta_init = np.concatenate([
            theta_1[:k] + rng.normal(0, 0.05, k)
            for k in class_K_base
        ])
    gamma_init = np.zeros((C - 1) * (K_mem + 1))
    init_params = np.concatenate([theta_init, gamma_init])

    # -- EM --
    params_em = fit_em(
        init_params=init_params, data=data, spec=spec,
        max_iter=50, tol=1e-5, verbose=False,
    )

    # -- LBFGS polish --
    polish = LBFGS(
        fun=lambda p: mixed_model_loglik(p, data, spec),
        maxiter=300,
    )
    result = polish.run(jnp.array(params_em))
    params_all = np.array(result.params)
    ll = -float(result.state.value)
    elapsed = time.perf_counter() - t0

    return params_all, ll, elapsed


def compare_draws(df, manual_spec, R_values, n_seeds=3):
    """Compare Halton vs Sobol across a range of R values."""
    results = []
    for R in R_values:
        for draw_method in ["halton", "sobol"]:
            lls = []
            params_list = []
            times = []
            for seed in range(n_seeds):
                try:
                    p, ll, t = fit_model(df, manual_spec, draw_method=draw_method, R=R, seed=seed)
                    if np.isfinite(ll):
                        lls.append(ll)
                        params_list.append(p)
                        times.append(t)
                except Exception as e:
                    print(f"  R={R:4d} {draw_method:6s} seed={seed} FAILED: {e}")
            if not lls:
                continue
            # MSE vs true params (use shared param size via padding)
            mse = np.mean([
                np.mean((p[:len(TRUE_THETA)] - TRUE_THETA) ** 2)
                for p in params_list
            ])
            results.append({
                "R": R,
                "method": draw_method,
                "ll_mean": np.mean(lls),
                "ll_std": np.std(lls),
                "ll_best": np.max(lls),
                "mse": mse,
                "time_mean": np.mean(times),
                "time_std": np.std(times),
                "n_converged": len(lls),
            })
    return pd.DataFrame(results)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=400)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--R_min", type=int, default=50)
    parser.add_argument("--R_max", type=int, default=500)
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("HALTON vs SOBOL — Synthetic LC-2 NB2 Recovery")
    print("=" * 70)
    print(f"N={args.N}  T={args.T}  seeds={args.n_seeds}")
    print()

    # Generate data
    df = generate_data(N=args.N, T=args.T, seed=42)
    print(f"Data: {df['ID'].nunique()} IDs x {args.T} periods = {len(df)} rows")
    print(f"  true class mix: {dict(df.groupby('ID')['true_class'].first().value_counts().to_dict())}")
    print(f"  mean Y: {df['Y'].mean():.2f}  std Y: {df['Y'].std():.2f}")
    print()

    # Build spec using the true DGP
    manual_spec = {
        "fixed_terms": OUTCOME_VARS,
        "rdm_terms": [],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [],
        "zi_terms": [],
        "membership_terms": MEMBERSHIP_VARS,
        "group_id_col": None,
        "dispersion": 1,  # NB2
        "latent_classes": 2,
        "min_class_proportion": 0.10,
    }

    # First do a faithful fit to show recovery quality
    print("-" * 70)
    print("FULL FIT (R=500 Halton) — parameter recovery check")
    print("-" * 70)
    params, ll, elapsed = fit_model(df, manual_spec, draw_method="halton", R=500)
    n = len(df)
    k = len(params)
    bic = k * np.log(n) - 2 * ll
    print(f"  LL={ll:.2f}  BIC={bic:.2f}  params={k}  time={elapsed:.1f}s")
    theta_list, gamma, pindex = unpack_lc_params(params, spec=None)
    # spec=None fails for unpack; compute manually
    gamma_est = params[-3:].reshape(1, 3)
    print(f"\n  True params vs estimated:")
    labels = [
        "c1_intercept", "c1_x1", "c1_x2", "c1_x3", "c1_x4", "c1_log_alpha",
        "c2_intercept", "c2_x1", "c2_x2", "c2_x3", "c2_x4", "c2_log_alpha",
        "g0", "gz1", "gz2",
    ]
    for i, lab in enumerate(labels):
        true_v = TRUE_THETA[i]
        est_v = params[i]
        err = est_v - true_v
        print(f"  {lab:18s}  true={true_v:+8.4f}  est={est_v:+8.4f}  err={err:+8.4f}")
    print()

    # Now sweep R values comparing Halton vs Sobol
    print("=" * 70)
    print("R-SWEEP: Halton vs Sobol across draw counts")
    print("=" * 70)
    R_values = np.linspace(args.R_min, args.R_max, 6, dtype=int).tolist()
    print(f"  R values: {R_values}")
    print(f"  seeds:    {args.n_seeds}")
    print()

    df_cmp = compare_draws(df, manual_spec, R_values, n_seeds=args.n_seeds)

    print(f"\n{'R':>5s}  {'Method':>6s}  {'LL_mean':>10s}  {'LL_std':>8s}  {'LL_best':>10s}  "
          f"{'MSE':>8s}  {'Time(s)':>8s}  {'Conv':>5s}")
    print("-" * 75)
    for _, row in df_cmp.iterrows():
        print(f"{int(row['R']):5d}  {row['method']:>6s}  {row['ll_mean']:10.2f}  "
              f"{row['ll_std']:8.2f}  {row['ll_best']:10.2f}  "
              f"{row['mse']:8.4f}  {row['time_mean']:8.2f}  {int(row['n_converged']):5d}")

    # Summary comparison
    print()
    print("=" * 70)
    print("SUMMARY: Halton vs Sobol (aggregated across R)")
    hat = df_cmp[df_cmp["method"] == "halton"]
    sob = df_cmp[df_cmp["method"] == "sobol"]
    if len(hat) and len(sob):
        print(f"  {'':>20s}  {'Halton':>12s}  {'Sobol':>12s}  {'Delta':>10s}")
        print(f"  {'LL mean':>20s}  {hat['ll_mean'].mean():12.2f}  {sob['ll_mean'].mean():12.2f}  "
              f"{sob['ll_mean'].mean() - hat['ll_mean'].mean():+10.2f}")
        print(f"  {'LL std (stability)':>20s}  {hat['ll_std'].mean():12.2f}  {sob['ll_std'].mean():12.2f}  "
              f"{sob['ll_std'].mean() - hat['ll_std'].mean():+10.2f}")
        print(f"  {'MSE (param error)':>20s}  {hat['mse'].mean():12.4f}  {sob['mse'].mean():12.4f}  "
              f"{sob['mse'].mean() - hat['mse'].mean():+10.4f}")
        print(f"  {'Time (s)':>20s}  {hat['time_mean'].mean():12.2f}  {sob['time_mean'].mean():12.2f}  "
              f"{sob['time_mean'].mean() - hat['time_mean'].mean():+10.2f}")
    print()

    print("Done.")
