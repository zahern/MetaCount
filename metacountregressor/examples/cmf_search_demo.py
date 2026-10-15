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
    res, y_jax, aadt_jax, b_jax, l_jax, draws = fit_final_model(
        df, sel_b, sel_l, rand_b, rand_l, model=model, use_halton=use_halton,
        R=args.R_final)
    se, cov = compute_se(res, y_jax, aadt_jax, b_jax, l_jax,
                         rand_b, rand_l, draws, args.R_final, model)
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
    print("\nDone.")


if __name__ == "__main__":
    main()