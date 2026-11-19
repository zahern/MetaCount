"""
QLD Heavy-Vehicle Head-On Search Demo
Runs the hierarchical CMF structure search on QLD Stage-5A dataset (n=1848).
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

QLD_DATA = os.path.join(REPO, "data", "Stage5A_1848_All_Initial_Columns.xlsx")

# Core QLD candidate sets from generate_washington_hierarchical_cmf_assets.py
QLD_UPPER_CORE = [
    "US", "RSMS", "RSLS", "MCV", "Nlanes", "Lwidth",
    "Curve50", "Curve", "Median", "ATLM", "F_W", "SP"
]
QLD_LOWER_CORE = [
    "Curve50", "RSHS", "MCV", "ArtVeh", "Nlanes",
    "Lwidth", "ATLM", "SP"
]
QLD_UPPER_EXPANDED = [
    "HSP", "MSP", "LSP", "St_Curve", "M_Curve", "S_Curve",
    "Roll_Ter", "Mt_Ter", "WC_W", "SW_PS", "SW_DS", "SSW_PS", "SSW_DS"
]
QLD_LOWER_EXPANDED = [
    "US", "RSMS", "Median", "Curve", "ATLM_PS", "Roll_Ter", "Mt_Ter"
]


def load_qld(use_expanded: bool = False) -> pd.DataFrame:
    df = pd.read_excel(QLD_DATA, sheet_name="Stage5A_1848")

    # Target: heavy-vehicle head-on crashes
    df = df.rename(columns={"Headon": "FREQ"})

    # Exposure offset: log(Length) — the generator excludes LENGTH-like cols from predictors
    df["_log_len"] = np.log(df["Length"].astype(float).clip(lower=1e-6))

    # Ensure numeric
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill NaN with 0 for binary/indicator columns
    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    for c in binary_cols:
        df[c] = df[c].fillna(0).astype(int)

    upper = [v for v in QLD_UPPER_CORE if v in df.columns]
    lower = [v for v in QLD_LOWER_CORE if v in df.columns]
    if use_expanded:
        upper += [v for v in QLD_UPPER_EXPANDED if v in df.columns]
        lower += [v for v in QLD_LOWER_EXPANDED if v in df.columns]

    print(f"QLD data loaded: {len(df)} obs, FREQ mean={df['FREQ'].mean():.4f}")
    print(f"  Upper candidates ({len(upper)}): {upper}")
    print(f"  Lower candidates ({len(lower)}): {lower}")
    print(f"  AADT range: {df['AADT'].min():.0f} - {df['AADT'].max():.0f} (mean {df['AADT'].mean():.0f})")
    print(f"  Length range: {df['Length'].min():.2f} - {df['Length'].max():.2f}")

    return df, upper, lower


def main() -> None:
    ap = argparse.ArgumentParser(description="QLD Hierarchical CMF Structure Search")
    ap.add_argument("--expanded", action="store_true", help="Use expanded candidate pool")
    ap.add_argument("--R-search", type=int, default=80, help="Halton draws during search")
    ap.add_argument("--R-final", type=int, default=200, help="Halton draws for final refit")
    ap.add_argument("--min-upper", type=int, default=2)
    ap.add_argument("--min-lower", type=int, default=1)
    ap.add_argument("--max-upper", type=int, default=None)
    ap.add_argument("--max-lower", type=int, default=None)
    args = ap.parse_args()

    df, upper, lower = load_qld(use_expanded=args.expanded)

    print("=" * 88)
    print("QLD STRUCTURE SEARCH -- Heavy-Vehicle Head-On Crashes")
    print("=" * 88)

    t0 = time.time()
    sel_b, sel_l, rand_b, rand_l, use_halton, model, fitness = run_ga(
        df, upper, lower, R=args.R_search, corr_threshold=0.7,
        min_baseline=args.min_upper, min_local=args.min_lower,
        max_baseline=args.max_upper, max_local=args.max_lower)
    el = time.time() - t0

    print(f"\nsearch finished in {el:.1f} s   (fitness = {fitness:.4f}, family = {model})")
    print("\nSEARCH OUTCOME")
    print(f"  UPPER tier (conventional CMFs) : {sel_b}")
    print(f"  LOWER tier (elasticity mods)  : {sel_l}")
    print(f"  random upper coefficients     : {[v for v, r in zip(sel_b, rand_b) if r]}")
    print(f"  random lower coefficients     : {[v for v, r in zip(sel_l, rand_l) if r]}")
    print(f"  selection criterion           : BIC + 5.0 * n_insig")

    print(f"\nrefitting the selected structure at R = {args.R_final} ...")
    res, y_jax, aadt_jax, b_jax, l_jax, draws, std_info, offset_jax = fit_final_model(
        df, sel_b, sel_l, rand_b, rand_l, model=model, use_halton=use_halton,
        R=args.R_final)
    se, cov = compute_se(res, y_jax, aadt_jax, b_jax, l_jax,
                         rand_b, rand_l, draws, args.R_final, model, offset_jax)
    table = build_summary_table(res, sel_b, sel_l, rand_b, rand_l, model, se)
    print_summary_table(table)

    print("\nHIERARCHICAL READING OF THE SEARCH RESULT")
    AADT_mean = float(np.mean(df["AADT"]))
    print_cmf_results(res, sel_b, sel_l, rand_b, rand_l,
                      AADT_mean=AADT_mean, model=model)

    if not sel_l:
        print("\n!! WARNING: EMPTY lower tier!")
    else:
        print(f"\ntier sizes: K_U = {len(sel_b)}, K_L = {len(sel_l)}  (both within constraints).")

    # Comprehensive CMF table
    from metacountregressor.examples.cmf_search_demo import print_comprehensive_cmf_table
    print_comprehensive_cmf_table(res, sel_b, sel_l, rand_b, rand_l,
                                   df, AADT_mean=AADT_mean)

    print("\nDone.")


if __name__ == "__main__":
    main()