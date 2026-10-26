"""
Search demonstration: let the package FIND the two-tier model, then show it.

Runs the package's own multi-start simulated-annealing structure search over
tier assignments (which covariates are upper tier, which are AADT-elasticity
modifiers, which are random, and which family), refits the winner at high
Halton resolution, and prints the resulting hierarchical reading.

Usage:
    python cmf_search_demo.py [--upper ...] [--lower ...]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = r"C:\Users\ahernz\source\MetaCount"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from metacountregressor.GA_CMF_AADT_JAX import (  # noqa: E402
    build_summary_table, compute_se, fit_final_model, print_cmf_results,
    print_summary_table, run_ga,
)

DATA = os.path.join(REPO, "metacountregressor", "data", "Ex-16-3.csv")

UPPER_POOL = ["WIDTH", "CURVES", "GRADEBR", "FRICTION", "EXPOSE", "MXMEDSH", "URB"]
LOWER_POOL = ["SLOPE", "TANGENT", "MIMEDSH", "MXGRDIFF", "AVEPRE"]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["_log_len"] = np.log(df["LENGTH"].astype(float).clip(lower=1e-6))
    df["_id"] = np.arange(len(df))
    return df


def print_elasticity_random_interaction(result,
                                         selected_baseline, selected_local,
                                         rand_baseline, rand_local,
                                         AADT_mean):
    """
    Show how elasticity modifiers (lower tier) interact with upper-tier
    random parameters. In the hierarchical model:

      log(mu_i) = alpha0 + sum_k alpha_k x_ik + u_i' sigma_alpha
                + [beta0 + sum_m beta_m z_im + v_i' sigma_beta] * log(AADT_i)

    The site-specific elasticity is beta0 + sum_m beta_m z_im + v_i' sigma_beta,
    while the site-specific CMF for upper-tier variable k is exp(alpha_k + sigma_alpha_k * u_ik).

    This function prints the combined effect for a few representative sites.
    """
    params = np.array(result.params if hasattr(result, "params") else result.x)
    idx = 0
    alpha0 = params[idx]; idx += 1

    alphas = []
    sig_alphas = []
    for k, v in enumerate(selected_baseline):
        alphas.append(params[idx]); idx += 1
        if rand_baseline[k]:
            sig_alphas.append(params[idx]); idx += 1
        else:
            sig_alphas.append(0.0)

    beta0 = params[idx]; idx += 1

    betas = []
    sig_betas = []
    for k, v in enumerate(selected_local):
        betas.append(params[idx]); idx += 1
        if rand_local[k]:
            sig_betas.append(params[idx]); idx += 1
        else:
            sig_betas.append(0.0)

    print("\n" + "=" * 88)
    print("  ELASTICITY MODIFIER × RANDOM PARAMETER INTERACTION")
    print("=" * 88)
    print("  Site-specific model:")
    print("    log(mu_i) = alpha0 + sum_k (alpha_k + sigma_alpha_k * u_ik) * x_ik")
    print("            + [beta0 + sum_m (beta_m + sigma_beta_m * v_im) * z_im] * log(AADT_i)")
    print()

    # Show which variables are random in each tier
    rand_A = [v for v, r in zip(selected_baseline, rand_baseline) if r]
    rand_B = [v for v, r in zip(selected_local, rand_local) if r]
    print("  Random upper-tier (A):", rand_A if rand_A else "none")
    print("  Random lower-tier (B):", rand_B if rand_B else "none")
    print()

    # Representative sites: mean AADT, ±1 sigma on random effects
    print("  Representative site scenarios (AADT = {:,.0f}):".format(AADT_mean))
    print("  " + "-" * 84)
    print("  {:<28} {:>12} {:>12} {:>12} {:>12}".format(
        "Scenario", "log(mu)", "mu", "Elasticity", "AADT CMF"))

    # Baseline: all u=0, v=0
    log_mu_base = alpha0 + beta0 * np.log(AADT_mean)
    print("  {:<28} {:>12.4f} {:>12.2f} {:>12.4f} {:>12.4f}".format(
        "Mean site (u=v=0)", log_mu_base, np.exp(log_mu_base), beta0, 1.0))

    # Upper-tier random only — CHANGES BASE RATE, NOT ELASTICITY
    if rand_A:
        print("  --- Component A (upper tier): site characteristics → base CMF ---")
        for k, v in enumerate(selected_baseline):
            if rand_baseline[k]:
                sig = abs(sig_alphas[k])
                for sign, label in [(-1, "-1σ"), (+1, "+1σ")]:
                    log_mu = log_mu_base + sign * sig * 1.0
                    cmf = np.exp(sign * sig * 1.0)
                    print("  {:<28} {:>12.4f} {:>12.2f} {:>12.4f} {:>12.4f}".format(
                        f"{v} [R] {label}", log_mu, np.exp(log_mu), beta0, cmf))

    # Lower-tier random only — CHANGES ELASTICITY
    if rand_B:
        print("  --- Component B (lower tier): elasticity modifiers → AADT response ---")
        for k, v in enumerate(selected_local):
            if rand_local[k]:
                sig = abs(sig_betas[k])
                for sign, label in [(-1, "-1σ"), (+1, "+1σ")]:
                    elast = beta0 + betas[k] + sign * sig * 1.0
                    log_mu = alpha0 + elast * np.log(AADT_mean)
                    aadt_cmf = float(AADT_mean) ** betas[k]
                    print("  {:<28} {:>12.4f} {:>12.2f} {:>12.4f} {:>12.4f}".format(
                        f"{v} [R] {label}", log_mu, np.exp(log_mu), elast, aadt_cmf))

    # Combined: both tiers random (independent) — MULTIPLICATIVE EFFECT
    if rand_A and rand_B:
        print("  --- Combined: μ_i = exp(α_i) × AADT_i^(β_i) ---")
        for ka, va in enumerate(selected_baseline):
            if rand_baseline[ka]:
                sa = abs(sig_alphas[ka])
            else:
                sa = 0
            for kb, vb in enumerate(selected_local):
                if rand_local[kb]:
                    sb = abs(sig_betas[kb])
                else:
                    sb = 0
                if sa > 0 and sb > 0:
                    for sa_sign, sa_label in [(-1, "-1σ"), (+1, "+1σ")]:
                        for sb_sign, sb_label in [(-1, "-1σ"), (+1, "+1σ")]:
                            elast = beta0 + betas[kb] + sb_sign * sb * 1.0
                            log_mu = alpha0 + sa_sign * sa * 1.0 + elast * np.log(AADT_mean)
                            base_cmf = np.exp(sa_sign * sa * 1.0)
                            aadt_cmf = float(AADT_mean) ** betas[kb]
                            print("  {:<28} {:>12.4f} {:>12.2f} {:>12.4f} {:>12.4f}".format(
                                f"{va}[A]{sa_label} + {vb}[B]{sb_label}",
                                log_mu, np.exp(log_mu), elast, base_cmf * aadt_cmf))

    print()
    print("  Key insight: upper-tier RP scales the BASE crash rate (mu),")
    print("  lower-tier RP scales the AADT ELASTICITY (response to volume).")
    print("  They multiply: mu_i = exp(alpha_i) * AADT_i^(beta_i).")
    print()


def print_comprehensive_cmf_table(result,
                                   selected_baseline, selected_local,
                                   rand_baseline, rand_local,
                                   df, AADT_mean):
    """
    Show CMF for each variable individually — varying ONE variable
    while holding all others at their reference (mean/0) level.
    """
    params = np.array(result.params if hasattr(result, "params") else result.x)
    idx = 0
    alpha0 = params[idx]; idx += 1

    alphas = []
    sig_alphas = []
    for k, v in enumerate(selected_baseline):
        alphas.append(params[idx]); idx += 1
        if rand_baseline[k]:
            sig_alphas.append(params[idx]); idx += 1
        else:
            sig_alphas.append(0.0)

    beta0 = params[idx]; idx += 1

    betas = []
    sig_betas = []
    for k, v in enumerate(selected_local):
        betas.append(params[idx]); idx += 1
        if rand_local[k]:
            sig_betas.append(params[idx]); idx += 1
        else:
            sig_betas.append(0.0)

    print("\n" + "=" * 92)
    print("  COMPREHENSIVE CMF TABLE — One Variable at a Time")
    print("=" * 92)
    print("  Reference: all continuous vars at mean, binary at 0, AADT = {:,.0f}".format(AADT_mean))
    print("  " + "-" * 90)

    # Upper tier (Component A): CMF = exp(alpha_k * delta_x)
    if selected_baseline:
        print("  COMPONENT A — Base CMFs (volume-independent)")
        print("  {:<24} {:>8} {:>12} {:>14} {:>14} {:>12}".format(
            "Variable", "Type", "alpha_k", "CMF (1 unit)", "CMF (range)", "Random?"))
        print("  " + "-" * 90)

        for k, v in enumerate(selected_baseline):
            a = alphas[k]
            is_rand = rand_baseline[k]

            # Determine variable type and meaningful change
            col_data = df[v].dropna()
            if col_data.nunique() == 2:
                var_type = "Binary"
                delta = 1.0  # 0 -> 1
                range_str = "0→1"
            else:
                var_type = "Continuous"
                delta = 1.0  # 1 unit change
                rng = col_data.max() - col_data.min()
                range_str = f"min→max ({rng:.0f})"

            cmf_unit = np.exp(a * delta)
            cmf_range = np.exp(a * rng) if var_type == "Continuous" else cmf_unit

            # Random: show 95% CMF range
            rand_str = "Yes" if is_rand else "No"
            if is_rand:
                sig = abs(sig_alphas[k])
                lo = np.exp((a - 1.96*sig) * delta)
                hi = np.exp((a + 1.96*sig) * delta)
                rand_str = f"Yes (95%: {lo:.4f}-{hi:.4f})"

            print("  {:<24} {:>8} {:>+12.4f} {:>14.4f} {:>14.4f} {}".format(
                v, var_type, a, cmf_unit, cmf_range, rand_str))

    # Lower tier (Component B): CMF = AADT^beta_k
    if selected_local:
        print()
        print("  COMPONENT B — Elasticity Modifiers (AADT-dependent)")
        print("  {:<24} {:>8} {:>12} {:>14} {:>14} {:>12}".format(
            "Variable", "Type", "beta_k", "CMF@meanAADT", "CMF@2xAADT", "Random?"))
        print("  " + "-" * 90)

        for k, v in enumerate(selected_local):
            b = betas[k]
            is_rand = rand_local[k]

            col_data = df[v].dropna()
            if col_data.nunique() == 2:
                var_type = "Binary"
                delta = 1.0
                range_str = "0→1"
            else:
                var_type = "Continuous"
                delta = 1.0
                rng = col_data.max() - col_data.min()
                range_str = f"min→max ({rng:.0f})"

            cmf_mean = float(AADT_mean) ** (b * delta)
            cmf_2x = float(2 * AADT_mean) ** (b * delta)

            rand_str = "Yes" if is_rand else "No"
            if is_rand:
                sig = abs(sig_betas[k])
                elast_lo = beta0 + b*delta - 1.96*sig
                elast_hi = beta0 + b*delta + 1.96*sig
                cmf_lo = float(AADT_mean) ** (b*delta - 1.96*sig)
                cmf_hi = float(AADT_mean) ** (b*delta + 1.96*sig)
                rand_str = f"Yes (elast 95%: {elast_lo:.4f}-{elast_hi:.4f})"

            print("  {:<24} {:>8} {:>+12.4f} {:>14.4f} {:>14.4f} {}".format(
                v, var_type, b, cmf_mean, cmf_2x, rand_str))

    # Combined: reference site (all vars at reference)
    print()
    print("  REFERENCE SITE (all vars at reference, u=v=0):")
    base_mu = np.exp(alpha0 + beta0 * np.log(AADT_mean))
    print("    Base crash rate mu = exp({:.4f}) * {:.0f}^{:.4f} = {:.4f}"
          .format(alpha0, AADT_mean, beta0, base_mu))
    print("    Elasticity = {:.4f}  |  1% AADT increase → {:.4f}% crash change"
          .format(beta0, beta0 * 100))

    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", nargs="*", default=UPPER_POOL)
    ap.add_argument("--lower", nargs="*", default=LOWER_POOL)
    ap.add_argument("--R-search", type=int, default=120)
    ap.add_argument("--R-final", type=int, default=500)
    ap.add_argument("--min-upper", type=int, default=2)
    ap.add_argument("--min-lower", type=int, default=1)
    ap.add_argument("--max-upper", type=int, default=None)
    ap.add_argument("--max-lower", type=int, default=None)
    args = ap.parse_args()

    df = load()
    upper = [v for v in args.upper if v in df.columns]
    lower = [v for v in args.lower if v in df.columns]

    print("=" * 88)
    print("STRUCTURE SEARCH -- finding the two-tier model")
    print("=" * 88)
    print("upper-tier candidates (direct CMFs) :", upper)
    print("lower-tier candidates (AADT elasticity):", lower)
    print("constraints  K_U: %d .. %s   |   K_L: %d .. %s"
          % (args.min_upper, args.max_upper, args.min_lower, args.max_lower))
    print("decision space = 2*%d + %d = %d binary choices per candidate"
          % (len(upper) + len(lower), len(upper) + len(lower),
             2 * (len(upper) + len(lower))))

    t0 = time.time()
    sel_b, sel_l, rand_b, rand_l, use_halton, model, fitness = run_ga(
        df, upper, lower, R=args.R_search, corr_threshold=0.8,
        min_baseline=args.min_upper, min_local=args.min_lower,
        max_baseline=args.max_upper, max_local=args.max_lower)
    el = time.time() - t0

    print("\nsearch finished in %.1f s   (fitness = %.4f, family = %s)"
          % (el, fitness, model))
    print("\nSEARCH OUTCOME")
    print("  UPPER tier (conventional CMFs) :", sel_b)
    print("  LOWER tier (elasticity mods)  :", sel_l)
    print("  random upper coefficients     :", [v for v, r in zip(sel_b, rand_b) if r])
    print("  random lower coefficients     :", [v for v, r in zip(sel_l, rand_l) if r])
    print("  selection criterion           : BIC + %s * n_insig" % 5.0)

    print("\nrefitting the selected structure at R = %d ..." % args.R_final)
    res, y_jax, aadt_jax, b_jax, l_jax, draws, std_info, offset_jax = fit_final_model(
        df, sel_b, sel_l, rand_b, rand_l, model=model, use_halton=use_halton,
        R=args.R_final)
    se, cov = compute_se(res, y_jax, aadt_jax, b_jax, l_jax,
                         rand_b, rand_l, draws, args.R_final, model, offset_jax)
    table = build_summary_table(res, sel_b, sel_l, rand_b, rand_l, model, se)
    print_summary_table(table)

    print("HIERARCHICAL READING OF THE SEARCH RESULT")
    print_cmf_results(res, sel_b, sel_l, rand_b, rand_l,
                      AADT_mean=float(np.mean(df["AADT"])), model=model)

    if not sel_l:
        print("!! WARNING: an EMPTY lower tier; K_L^min >= 1 is not being")
        print("   enforced by run_ga with these arguments.")
    else:
        print("tier sizes: K_U = %d, K_L = %d  (both within constraints)." % (
            len(sel_b), len(sel_l)))

    print("\nThe upper tier supplies exp(delta*dx) CMFs; the lower tier supplies")
    print("gamma coefficients on the AADT elasticity, so the site-specific")
    print("elasticity is beta + sum_k gamma_k z_k.")

    # --- NEW: Elasticity modifier × Random parameter interaction ---
    if any(rand_b) and any(rand_l):
        print_elasticity_random_interaction(res, sel_b, sel_l, rand_b, rand_l,
                                             AADT_mean=float(np.mean(df["AADT"])))
    elif any(rand_b):
        print("\nNOTE: upper-tier random parameters present; lower tier is fixed.")
        print("      Site-specific elasticity = beta0 + sum_k beta_k z_k (no RP in B).")
    elif any(rand_l):
        print("\nNOTE: lower-tier random parameters present; upper tier is fixed.")
        print("      Site-specific CMF = exp(alpha0 + sum_k alpha_k x_k) (no RP in A).")
    else:
        print("\nNOTE: all parameters fixed — no site-level heterogeneity.")

    # --- NEW: Comprehensive CMF table for each variable ---
    print_comprehensive_cmf_table(res, sel_b, sel_l, rand_b, rand_l,
                                   df, AADT_mean=float(np.mean(df["AADT"])))

    print("\nDone.")


if __name__ == "__main__":
    main()