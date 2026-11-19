"""
manual_hierarchical_cmf_tutorial.py
====================================
A self-contained, runnable tutorial: how to fit a hierarchical CMF model
BY HAND with metacountregressor's ExperimentBuilder — no metaheuristic
structure search, no PBS/HPC submission, no cluster required. Just:

    python manual_hierarchical_cmf_tutorial.py

Background (see paper/Article Multilevel modelling in traffic safety
analysis.docx): the Highway Safety Manual's traditional approach treats
AADT as a fixed-elasticity offset and calibrates CMFs as a SEPARATE,
second-stage multiplicative adjustment on top of an already-fitted SPF.
That two-stage split can introduce omitted-variable bias, and it assumes
every observation shares exactly one data-generating process (an
assumption the paper calls "complete pooling").

This tutorial builds the same model the paper argues for, in one stage:

  1. COMPLETE POOLING       — the traditional baseline: one fixed
                               coefficient per variable, all crash counts
                               forced through the same "line".
  2. VARYING-ELASTICITY SPF — the paper's main innovation: AADT's own
                               elasticity is allowed to depend on other
                               covariates, so exposure isn't a fixed offset.
  3. PARTIAL POOLING         — random parameters: each observation gets its
                               own draw around a population mean, shrunk
                               toward that mean by how much data supports
                               it. This is what "hierarchical" buys you
                               over the plain fixed-effects fit in step 1.
  4. CMF DERIVATION           — reading a Crash Modification Factor straight
                               off the fitted coefficients, HSM-style.

Every step below produces a real fitted model on the Washington Ex-16-3
dataset bundled with this package (metacountregressor/data/Ex-16-3.csv).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from metacountregressor.experiment_package import ExperimentBuilder

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Ex-16-3.csv")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # A row per road segment. FREQ = observed crash count, AADT = traffic
    # volume, LENGTH = segment length (miles) — used for the exposure offset.
    df["_id"] = np.arange(len(df))
    df["log_len"] = np.log(df["LENGTH"].clip(lower=1e-6))
    return df


# ======================================================================
# STEP 1 — Complete pooling: the traditional fixed-effects SPF
# ======================================================================
def fit_complete_pooling(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("STEP 1 — COMPLETE POOLING (traditional fixed-effects SPF)")
    print("=" * 72)
    print("Every segment is forced through the SAME coefficients — no")
    print("group structure, no partial pooling. This is the paper's")
    print("'complete pooling' baseline (Eq. 16-20).")

    builder = ExperimentBuilder(df=df, id_col="_id", y_col="FREQ", offset_col="log_len")

    # AADT enters as a free covariate (not folded into the offset), so its
    # coefficient IS the traffic elasticity — this already goes one step
    # beyond the classic HSM offset-with-elasticity-fixed-at-1 formulation.
    spec = builder.make_manual_spec(
        fixed_terms=["LNAADT", "WIDTH", "CURVES", "GRADEBR", "SLOPE"],
        dispersion=1,  # 1 = Negative Binomial (NB2), 0 = Poisson
    )
    fit = builder.fit_manual_model(spec, model="nb", print_report=True)
    print(f"BIC = {fit['summary'].get('bic'):.2f}   "
          f"LL = {fit['summary'].get('loglik'):.2f}")
    return fit


# ======================================================================
# STEP 2 — Varying-elasticity SPF: exposure as a function of context
# ======================================================================
def fit_varying_elasticity(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("STEP 2 — VARYING-ELASTICITY SPF (AADT elasticity depends on context)")
    print("=" * 72)
    print("Paper Eq. 38-43: elasticity = beta_AADT + gamma * SLOPE, instead")
    print("of a single fixed elasticity for every segment. We build this by")
    print("adding a precomputed AADT x SLOPE interaction column and fitting")
    print("it as an ordinary fixed term — the coefficient on that column IS")
    print("gamma from the paper's varying-coefficient formulation.")

    df = df.copy()
    df["slope_x_lnaadt"] = df["SLOPE"] * df["LNAADT"]

    builder = ExperimentBuilder(df=df, id_col="_id", y_col="FREQ", offset_col="log_len")
    spec = builder.make_manual_spec(
        fixed_terms=["LNAADT", "WIDTH", "CURVES", "GRADEBR", "slope_x_lnaadt"],
        dispersion=1,
    )
    fit = builder.fit_manual_model(spec, model="nb", print_report=True)

    beta_aadt = float(fit["result"].params[list(fit["spec"].fixed_names).index("LNAADT")])
    gamma_slope = float(fit["result"].params[list(fit["spec"].fixed_names).index("slope_x_lnaadt")])
    print(f"\nBaseline AADT elasticity (beta_AADT): {beta_aadt:.4f}")
    print(f"Slope interaction (gamma):             {gamma_slope:.4f}")
    print(f"  -> elasticity on a flat segment  (SLOPE=1): {beta_aadt + gamma_slope:.4f}")
    print(f"  -> elasticity on a graded segment (SLOPE=0): {beta_aadt:.4f}")
    print("(This is exactly the paper's claim that traffic elasticity is")
    print(" context-dependent, not a single fixed number.)")
    return fit


# ======================================================================
# STEP 3 — Partial pooling: random parameters across segments
# ======================================================================
def fit_partial_pooling(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("STEP 3 — PARTIAL POOLING (random parameters = hierarchical shrinkage)")
    print("=" * 72)
    print("rdm_terms tells ExperimentBuilder to treat a coefficient as")
    print("normally distributed across observations (mean + SD estimated),")
    print("instead of one fixed number for everyone. This is the paper's")
    print("partial-pooling estimator (Eq. 22-25): each segment's effective")
    print("coefficient is shrunk toward the population mean by an amount")
    print("that depends on how much the data supports deviating from it.")

    builder = ExperimentBuilder(df=df, id_col="_id", y_col="FREQ", offset_col="log_len")
    spec = builder.make_manual_spec(
        fixed_terms=["LNAADT", "GRADEBR"],
        rdm_terms=["WIDTH:normal", "CURVES:normal"],   # <-- the partial-pooling terms
        dispersion=1,
    )
    fit = builder.fit_manual_model(spec, model="nb", print_report=True)

    print(f"\nBIC = {fit['summary'].get('bic'):.2f}   "
          f"LL = {fit['summary'].get('loglik'):.2f}")
    print("Compare this BIC to Step 1's complete-pooling BIC: a lower BIC")
    print("here means the added heterogeneity is worth its extra parameters.")
    print("A large SD on WIDTH/CURVES means their effect genuinely varies")
    print("segment-to-segment; a near-zero SD means complete pooling was")
    print("already a fine approximation for that variable.")
    return fit


# ======================================================================
# STEP 4 — Deriving a CMF from the fitted model (HSM-style)
# ======================================================================
def derive_cmf(fit: dict, variable: str, delta: float = 1.0):
    """
    HSM defines a CMF as the ratio of predicted crashes after vs. before a
    one-unit (or `delta`-unit) change in a design variable, holding
    everything else fixed (paper Eq. 32-34):

        CMF = exp(beta * delta)

    Because this model is log-linear (log(mu) = beta' x + offset), that
    ratio collapses to exp(beta * delta) directly off the fitted
    coefficient — no separate calibration stage required.

    GOTCHA: metacountregressor standardises continuous predictors internally
    before fitting (for numerical stability), so fit["result"].params are in
    PER-STANDARD-DEVIATION units, not per raw unit. A CMF for a real-world
    "delta"-unit change (e.g. +1 foot of lane width) needs the coefficient
    converted back first:  beta_original = beta_standardised / sd(variable).
    This mirrors exactly what metacountregressor's own printed model summary
    does before showing "Estimate" (see main_hpc.unstandardize_summary_df) —
    which is why the number below matches the printed summary's "Estimate"
    column, not the raw params array.
    """
    names = list(fit["spec"].fixed_names)
    if variable not in names:
        raise ValueError(f"{variable} is not a fixed term in this fit: {names}")
    beta_std = float(fit["result"].params[names.index(variable)])

    scaler = (fit.get("data", {}) or {}).get("scaler", {}) or {}
    if variable in scaler:
        _, sd = scaler[variable]
        beta = beta_std / float(sd)
    else:
        beta = beta_std  # not standardised (e.g. a 0/1 indicator variable)

    cmf = float(np.exp(beta * delta))
    pct = (cmf - 1.0) * 100.0
    print(f"\nCMF({variable}, delta={delta}) = exp({beta:.4f} * {delta}) = {cmf:.4f}")
    print(f"  -> a {delta:+g}-unit change in {variable} changes expected "
          f"crashes by {pct:+.1f}%, holding all else fixed.")
    return cmf


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} segments from {os.path.abspath(DATA_PATH)}")

    fit1 = fit_complete_pooling(df)
    fit2 = fit_varying_elasticity(df)
    fit3 = fit_partial_pooling(df)

    print("\n" + "=" * 72)
    print("STEP 4 — CMF DERIVATION")
    print("=" * 72)
    derive_cmf(fit1, "WIDTH", delta=1.0)
    derive_cmf(fit1, "CURVES", delta=1.0)

    print("\nDone. This whole tutorial ran as a single local Python process —")
    print("no search, no PBS submission, no cluster. For the metaheuristic")
    print("structure-search version of this same idea (which decides WHICH")
    print("variables and roles to use automatically), see:")
    print("  metacountregressor/scripts/generate_washington_hierarchical_cmf_assets.py")
    print("  --help")
