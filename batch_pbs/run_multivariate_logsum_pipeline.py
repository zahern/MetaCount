"""
run_multivariate_logsum_pipeline.py
====================================
Three-phase multivariate count model comparison pipeline.

PHASE 1 – Variable-selection search across ALL algorithms (SA, HS, DE)
         Run independently; compare BIC across all.

PHASE 2 – Logsum extraction
         Take the best-BIC model from Phase 1.
         Estimate the full multivariate model on the selected variables.
         Compute per-person composite logsum:

             logsum_i = log( sum_m  exp( x_i @ beta_m ) )
                      = log( sum_m  mu_hat_{i,m} / exp(offset_{i,m}) )

         This is the total predicted activity "attractiveness" across all M
         activity types — analogous to the inclusive value (logsum) used in
         nested discrete-choice / ABM utility frameworks.

PHASE 3 – Joint re-estimation with logsum feedback
         Add logsum_i as an additional covariate in every activity equation.
         Re-estimate the joint multivariate copula model (warm-started from
         Phase 1 estimates).  Compare BIC: Phase 3 vs Phase 1 best.

Usage (standalone):
    python run_multivariate_logsum_pipeline.py
    python run_multivariate_logsum_pipeline.py --algo sa hs de  # subset
    python run_multivariate_logsum_pipeline.py --max_time 3600 --seed 42

The script writes JSON summaries and a Markdown comparison table to
./results_pipeline/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize

warnings.filterwarnings("ignore")

# ── Make sure MetaCount repo is importable ─────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent  # one level up from batch_pbs/
for _p in [str(_REPO)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION (override with CLI flags)
# ─────────────────────────────────────────────────────────────────────────────
ALGOS           = ["sa", "hs", "de"]   # all three search algorithms
MAX_TIME        = 1800.0               # seconds per algorithm arm
MAX_IMP         = 300                  # max improving iterations per arm
SEED            = 42                   # RNG seed for reproducibility
N_PERSONS       = 600                  # synthetic dataset size
OUTPUT_DIR      = _HERE / "results_pipeline"

ACTIVITY_COLS  = ["n_work", "n_shop", "n_rec", "n_eat"]
COVARIATE_COLS = [
    "age", "income", "cars", "hhsize", "dist_cbd",
    "emp_density", "pop_density", "female", "employed", "student",
]

COPULA   = "gaussian"
MARGINAL = "nb"

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_data(n: int = N_PERSONS, seed: int = SEED) -> pd.DataFrame:
    """Generate the canonical synthetic ABM dataset (fixed DGP)."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({"person_id": range(n)})
    df["age"]         = rng.integers(18, 75, size=n).astype(float)
    df["income"]      = rng.lognormal(mean=10.5, sigma=0.6, size=n)
    df["cars"]        = rng.choice([0, 1, 2, 3], size=n, p=[.15, .40, .35, .10]).astype(float)
    df["hhsize"]      = rng.choice([1, 2, 3, 4, 5], size=n, p=[.20, .35, .25, .15, .05]).astype(float)
    df["dist_cbd"]    = rng.exponential(scale=8.0, size=n)
    df["emp_density"] = rng.exponential(scale=2000.0, size=n)
    df["pop_density"] = rng.exponential(scale=3000.0, size=n)
    df["female"]      = rng.choice([0, 1], size=n, p=[.49, .51]).astype(float)
    df["employed"]    = rng.choice([0, 1], size=n, p=[.35, .65]).astype(float)
    df["student"]     = rng.choice([0, 1], size=n, p=[.85, .15]).astype(float)

    def nb_draw(eta, alpha=1.0):
        mu = np.exp(np.clip(eta, -5, 5))
        p  = alpha / (alpha + mu)
        return rng.negative_binomial(alpha, p)

    df["n_work"] = nb_draw(-0.5 + 0.8 * df["employed"] - 0.3 * df["student"])
    df["n_shop"] = nb_draw(-0.8 + 0.2 * df["female"]   - 0.1 * np.log1p(df["dist_cbd"]))
    df["n_rec"]  = nb_draw(-1.0 + 0.1 * df["cars"]     + 0.05 * df["hhsize"])
    df["n_eat"]  = nb_draw(-1.2 + 0.15 * df["income"]  / df["income"].mean())
    return df


def _serialise(obj):
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, dict):            return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return [_serialise(v) for v in obj]
    return obj


def banner(text: str, width: int = 66) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — Search across all algorithms
# ─────────────────────────────────────────────────────────────────────────────

def phase1_search(df: pd.DataFrame, algos: list[str], max_time: float,
                  max_imp: int, output_dir: Path) -> dict:
    """
    Run variable-selection search for each algorithm independently.

    Returns a dict keyed by algo name; each value contains:
      'bic', 'best_decoded', 'result', 'elapsed_s'
    """
    from metacountregressor import ExperimentBuilder

    banner("PHASE 1  –  Variable-selection search  (all algorithms)")

    builder = ExperimentBuilder(
        df=df,
        id_col="person_id",
        y_col=ACTIVITY_COLS[0],
    )

    problem = builder.build_search(
        model_family          = "multivariate",
        variables             = COVARIATE_COLS,
        activity_cols         = ACTIVITY_COLS,
        maxiter               = 500,
        add_intercept         = True,
        fixed_copula          = COPULA,
        fixed_marginal        = MARGINAL,
        search_copula         = False,
        search_marginal       = False,
        min_vars_per_activity = 1,
    )

    phase1_results: dict = {}

    for algo in algos:
        print(f"\n  ── Algorithm : {algo.upper():<4}  max_time={max_time:.0f}s  max_imp={max_imp}")
        t0 = time.perf_counter()

        result = problem.run(
            algo             = algo,
            max_time         = max_time,
            max_imp          = max_imp,
            termination_iter = 200,
        )

        elapsed = time.perf_counter() - t0
        bic     = result.get("best_bic")
        decoded = result.get("best_decoded")

        print(f"     BIC = {bic}   elapsed = {elapsed:.1f}s")
        if decoded:
            for act, cols in zip(ACTIVITY_COLS, decoded.get("selected_per_activity", [])):
                print(f"       {act:<12}: {cols}")

        phase1_results[algo] = {
            "bic":          float(bic) if bic is not None else None,
            "best_decoded": _serialise(decoded),
            "elapsed_s":    elapsed,
        }

        # Save per-algo result
        out = output_dir / f"phase1_{algo}.json"
        with open(out, "w") as f:
            json.dump({"algo": algo, **phase1_results[algo]}, f, indent=2)
        print(f"     Saved → {out}")

    return phase1_results


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — Estimate best model → extract logsums
# ─────────────────────────────────────────────────────────────────────────────

def phase2_extract_logsum(df: pd.DataFrame,
                           phase1_results: dict,
                           output_dir: Path) -> tuple[np.ndarray, dict]:
    """
    1. Select the best-BIC result from Phase 1.
    2. Estimate the full multivariate model on the winning variable set.
    3. Compute per-person logsum:

           logsum_i = log( sum_{m=1}^{M}  exp( x_{i,m} @ beta_m ) )
                    = log( sum_m  mu_hat_{i,m} )          if offsets=0
                    = "total predicted activity volume"   (natural log)

       This composite value (the "inclusive value" across all activity types)
       captures how travel-intensive person i is predicted to be across the
       entire activity portfolio, before any feedback.

    Returns (logsum_vector [n,], fit_info dict).
    """
    from metacountregressor.multivariate_count_regressor import MultivariateCountRegressor

    banner("PHASE 2  –  Best model estimation  →  logsum extraction")

    # ── Pick best algo ──────────────────────────────────────────────────────
    valid = {a: r for a, r in phase1_results.items() if r.get("bic") is not None}
    best_algo = min(valid, key=lambda a: valid[a]["bic"])
    best_bic  = valid[best_algo]["bic"]
    decoded   = valid[best_algo]["best_decoded"]

    print(f"\n  Best algorithm  : {best_algo.upper()}  (BIC = {best_bic:.4f})")

    # ── Determine selected covariates ───────────────────────────────────────
    if decoded and "selected_per_activity" in decoded:
        selected_per_act = decoded["selected_per_activity"]
    else:
        # Fallback: use all covariates for every activity
        selected_per_act = [COVARIATE_COLS] * len(ACTIVITY_COLS)

    print("\n  Selected covariates (winning model):")
    for act, cols in zip(ACTIVITY_COLS, selected_per_act):
        print(f"    {act:<15}: {cols}")

    # ── Build design matrices (outcome-specific) ────────────────────────────
    add_intercept = True
    X_list: list[np.ndarray] = []
    feature_names: list[list[str]] = []
    for act_cols in selected_per_act:
        X_m = df[act_cols].to_numpy(dtype=np.float64)
        if add_intercept:
            X_m = np.column_stack([np.ones(len(X_m)), X_m])
            fn  = ["const"] + list(act_cols)
        else:
            fn = list(act_cols)
        X_list.append(X_m)
        feature_names.append(fn)

    Y = df[ACTIVITY_COLS].to_numpy(dtype=np.float64)

    # ── Fit multivariate model ──────────────────────────────────────────────
    print(f"\n  Fitting {COPULA}-copula {MARGINAL} model …")
    model = MultivariateCountRegressor(
        activity_names=ACTIVITY_COLS,
        copula=COPULA,
        marginal=MARGINAL,
        maxiter=2000,
        verbose=True,
    )
    fit = model.fit(X_list, Y)

    print(f"\n  Fit summary:")
    print(f"    log-lik  = {fit.loglik:.4f}")
    print(f"    AIC      = {fit.aic:.4f}")
    print(f"    BIC      = {fit.bic:.4f}")
    print(f"    converged= {fit.converged}")

    # ── Compute logsum ──────────────────────────────────────────────────────
    # Linear predictors V_{i,m} = x_{i,m} @ beta_m   (before exp)
    # logsum_i = log( sum_m exp(V_{i,m}) )
    # This is the LogSum (inclusive value) across activity types.
    V_list = []
    for m, (X_m, beta_m) in enumerate(zip(X_list, fit.coef)):
        V_list.append(X_m @ beta_m)          # (n,)

    V_mat      = np.column_stack(V_list)      # (n, M)
    # Numerically stable log-sum-exp
    V_max      = V_mat.max(axis=1, keepdims=True)
    logsum     = np.log(np.exp(V_mat - V_max).sum(axis=1)) + V_max.squeeze()  # (n,)

    print(f"\n  Logsum statistics:")
    print(f"    min  = {logsum.min():.4f}")
    print(f"    mean = {logsum.mean():.4f}")
    print(f"    max  = {logsum.max():.4f}")
    print(f"    std  = {logsum.std():.4f}")

    # Attach to dataframe for Phase 3
    df["logsum"] = logsum

    # Save
    np.save(output_dir / "logsum_phase2.npy", logsum)
    with open(output_dir / "phase2_fit_summary.json", "w") as f:
        json.dump({
            "best_algo":          best_algo,
            "phase1_bic":         best_bic,
            "phase2_loglik":      float(fit.loglik),
            "phase2_aic":         float(fit.aic),
            "phase2_bic":         float(fit.bic),
            "phase2_converged":   bool(fit.converged),
            "selected_per_act":   _serialise(selected_per_act),
            "logsum_mean":        float(logsum.mean()),
            "logsum_std":         float(logsum.std()),
        }, f, indent=2)

    print(f"\n  Saved logsum    → {output_dir / 'logsum_phase2.npy'}")
    print(f"  Saved fit info  → {output_dir / 'phase2_fit_summary.json'}")

    return logsum, {
        "best_algo":        best_algo,
        "phase1_bic":       best_bic,
        "phase2_bic":       float(fit.bic),
        "phase2_loglik":    float(fit.loglik),
        "selected_per_act": selected_per_act,
        "fit":              fit,
        "X_list":           X_list,
        "Y":                Y,
        "feature_names":    feature_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 — Re-estimate with logsum as link covariate
# ─────────────────────────────────────────────────────────────────────────────

def phase3_reestimate_with_logsum(df: pd.DataFrame,
                                   logsum: np.ndarray,
                                   phase2_info: dict,
                                   output_dir: Path) -> dict:
    """
    Add logsum_i as an additional covariate in each activity equation and
    re-estimate the joint multivariate copula model (warm-started from
    Phase 2 coefficients).

    The logsum enters as a standardised (z-scored) variable so its coefficient
    is comparable across activity types.

    After estimation:
      • Print per-activity logsum coefficients (z-stat, p-value)
      • Compare BIC: Phase 2 (no logsum) vs Phase 3 (with logsum)
      • Report ΔBIC and log-likelihood ratio test statistic
    """
    from metacountregressor.multivariate_count_regressor import MultivariateCountRegressor

    banner("PHASE 3  –  Re-estimation with logsum link covariate")

    selected_per_act = phase2_info["selected_per_act"]
    Y                = phase2_info["Y"]
    phase2_bic       = phase2_info["phase2_bic"]
    phase2_loglik    = phase2_info["phase2_loglik"]

    # ── Standardise logsum for numerical stability ──────────────────────────
    ls_mean  = logsum.mean()
    ls_std   = logsum.std() + 1e-12
    ls_z     = (logsum - ls_mean) / ls_std
    df["logsum_z"] = ls_z

    print(f"\n  Adding 'logsum_z' (standardised logsum) to each activity equation.")
    print(f"  logsum_z: mean={ls_z.mean():.4f}  std={ls_z.std():.4f}")

    # ── Build design matrices: original covariates + logsum_z ───────────────
    add_intercept = True
    X_list_new:    list[np.ndarray] = []
    feat_new:      list[list[str]]  = []
    for act_cols in selected_per_act:
        X_m = df[act_cols].to_numpy(dtype=np.float64)
        if add_intercept:
            X_m = np.column_stack([np.ones(len(X_m)), X_m, ls_z])
            fn  = ["const"] + list(act_cols) + ["logsum_z"]
        else:
            X_m = np.column_stack([X_m, ls_z])
            fn  = list(act_cols) + ["logsum_z"]
        X_list_new.append(X_m)
        feat_new.append(fn)

    # ── Fit with logsum ─────────────────────────────────────────────────────
    print(f"\n  Fitting {COPULA}-copula {MARGINAL} model WITH logsum_z …")
    model3 = MultivariateCountRegressor(
        activity_names=ACTIVITY_COLS,
        copula=COPULA,
        marginal=MARGINAL,
        maxiter=2000,
        verbose=True,
    )
    fit3 = model3.fit(X_list_new, Y)

    print(f"\n  Phase 3 fit summary:")
    print(f"    log-lik  = {fit3.loglik:.4f}")
    print(f"    AIC      = {fit3.aic:.4f}")
    print(f"    BIC      = {fit3.bic:.4f}")
    print(f"    converged= {fit3.converged}")

    # ── Extract logsum_z coefficient from each activity ─────────────────────
    print("\n  Logsum link coefficient per activity:")
    print(f"  {'Activity':<15} {'coef':>10} {'SE':>10} {'z':>8} {'p':>8}")
    print("  " + "-" * 55)

    logsum_coefs = {}
    for m, (act, coef_m, se_m, fn_m) in enumerate(
        zip(ACTIVITY_COLS, fit3.coef, fit3.se, feat_new)
    ):
        if "logsum_z" in fn_m:
            idx   = fn_m.index("logsum_z")
            c     = float(coef_m[idx])
            se    = float(se_m[idx]) if se_m is not None and not np.isnan(se_m[idx]) else np.nan
            z_val = c / se if se > 0 else np.nan
            from scipy.stats import norm
            p_val = 2.0 * (1.0 - norm.cdf(abs(z_val))) if not np.isnan(z_val) else np.nan
            stars = ("***" if p_val < 0.001 else "**" if p_val < 0.01 else
                     "*"   if p_val < 0.05  else "."  if p_val < 0.1  else "")
            print(f"  {act:<15} {c:>10.4f} {se:>10.4f} {z_val:>8.3f} {p_val:>8.4f} {stars}")
            logsum_coefs[act] = {"coef": c, "se": se, "z": z_val, "p": p_val}
        else:
            print(f"  {act:<15} {'(not added)':>32}")
            logsum_coefs[act] = None

    # ── BIC comparison ──────────────────────────────────────────────────────
    delta_bic  = fit3.bic - phase2_bic
    delta_ll   = fit3.loglik - phase2_loglik
    lrt_stat   = 2.0 * delta_ll          # χ² with M extra params (one logsum per activity)
    lrt_df     = len(ACTIVITY_COLS)
    from scipy.stats import chi2
    lrt_pval   = 1.0 - chi2.cdf(lrt_stat, df=lrt_df)

    print(f"\n  ─── BIC comparison ───────────────────────────────────────")
    print(f"  Phase 2  (no logsum)   BIC = {phase2_bic:.4f}")
    print(f"  Phase 3  (+ logsum_z)  BIC = {fit3.bic:.4f}")
    print(f"  ΔBIC (Phase3 − Phase2) = {delta_bic:+.4f}")
    if delta_bic < 0:
        print("  ✓  Adding logsum IMPROVES the model (lower BIC).")
    elif abs(delta_bic) < 2:
        print("  ≈  BIC difference < 2 → negligible effect of logsum.")
    else:
        print("  ✗  Adding logsum WORSENS the model (higher BIC).")
        print("     (Logsum may be collinear with covariates or mis-specified.)")

    print(f"\n  LR-test:  χ²({lrt_df}) = {lrt_stat:.4f}   p = {lrt_pval:.6f}")

    # ── Save ────────────────────────────────────────────────────────────────
    out = {
        "phase2_bic":    float(phase2_bic),
        "phase3_bic":    float(fit3.bic),
        "delta_bic":     float(delta_bic),
        "phase2_loglik": float(phase2_loglik),
        "phase3_loglik": float(fit3.loglik),
        "lrt_stat":      float(lrt_stat),
        "lrt_df":        int(lrt_df),
        "lrt_pval":      float(lrt_pval),
        "converged":     bool(fit3.converged),
        "logsum_coefs":  _serialise(logsum_coefs),
    }
    with open(output_dir / "phase3_logsum_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {output_dir / 'phase3_logsum_results.json'}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(phase1: dict, phase2_info: dict, phase3: dict,
                  output_dir: Path) -> None:
    banner("PIPELINE SUMMARY")

    lines = []
    lines.append("# MetaCount Multivariate + Logsum Pipeline — Summary\n")
    lines.append("## Phase 1: Algorithm comparison\n")
    lines.append(f"| Algorithm | BIC | Elapsed (s) |")
    lines.append(f"|-----------|-----|-------------|")
    for algo, r in phase1.items():
        bic_str = f"{r['bic']:.4f}" if r["bic"] is not None else "n/a"
        lines.append(f"| {algo.upper()} | {bic_str} | {r['elapsed_s']:.1f} |")

    best = phase2_info["best_algo"]
    p2b  = phase2_info["phase2_bic"]
    p3b  = phase3["phase3_bic"]
    db   = phase3["delta_bic"]
    lrt  = phase3["lrt_stat"]
    lrtp = phase3["lrt_pval"]

    lines.append(f"\n**Best algorithm (Phase 1):** {best.upper()}")
    lines.append(f"\n## Phase 2: Full model (no logsum)\n")
    lines.append(f"- BIC = {p2b:.4f}")
    lines.append(f"- log-lik = {phase2_info['phase2_loglik']:.4f}\n")

    lines.append(f"## Phase 3: Re-estimation with logsum link\n")
    lines.append(f"- BIC = {p3b:.4f}")
    lines.append(f"- log-lik = {phase3['phase3_loglik']:.4f}")
    lines.append(f"- ΔBIC (Phase 3 − Phase 2) = {db:+.4f}")
    lines.append(f"- LR test: χ²({phase3['lrt_df']}) = {lrt:.4f}, p = {lrtp:.6f}\n")

    verdict = "IMPROVES" if db < 0 else ("NEGLIGIBLE" if abs(db) < 2 else "WORSENS")
    lines.append(f"**Logsum link effect: {verdict}**\n")

    lines.append(f"## Logsum coefficients per activity\n")
    lines.append(f"| Activity | coef | SE | z | p |")
    lines.append(f"|----------|------|----|---|---|")
    for act, v in phase3["logsum_coefs"].items():
        if v:
            lines.append(
                f"| {act} | {v['coef']:.4f} | {v['se']:.4f} | {v['z']:.3f} | {v['p']:.4f} |"
            )
        else:
            lines.append(f"| {act} | — | — | — | — |")

    md = "\n".join(lines)
    out = output_dir / "pipeline_summary.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\n  Full summary saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(algos=None, max_time=MAX_TIME, max_imp=MAX_IMP,
         seed=SEED, n_persons=N_PERSONS):

    if algos is None:
        algos = ALGOS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    banner(f"MetaCount  –  Multivariate + Logsum Pipeline  (seed={seed})")
    print(f"  Algorithms  : {algos}")
    print(f"  Max time/arm: {max_time:.0f}s")
    print(f"  Max imp/arm : {max_imp}")
    print(f"  Persons (N) : {n_persons}")
    print(f"  Output dir  : {OUTPUT_DIR}")

    # ── Dataset ─────────────────────────────────────────────────────────────
    df = make_synthetic_data(n=n_persons, seed=seed)
    print(f"\n  Synthetic dataset: N={len(df)}")
    for act in ACTIVITY_COLS:
        print(f"    {act}: mean={df[act].mean():.2f}  max={df[act].max()}")

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    phase1 = phase1_search(df, algos, max_time, max_imp, OUTPUT_DIR)

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    logsum, phase2_info = phase2_extract_logsum(df, phase1, OUTPUT_DIR)

    # ── Phase 3 ─────────────────────────────────────────────────────────────
    phase3 = phase3_reestimate_with_logsum(df, logsum, phase2_info, OUTPUT_DIR)

    # ── Summary ─────────────────────────────────────────────────────────────
    write_summary(phase1, phase2_info, phase3, OUTPUT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MetaCount multivariate + logsum pipeline"
    )
    parser.add_argument(
        "--algo", nargs="+", default=ALGOS,
        choices=["sa", "hs", "de"],
        help="Algorithms to include in Phase 1 (default: all three)",
    )
    parser.add_argument(
        "--max_time", type=float, default=MAX_TIME,
        help="Wall-clock budget per algorithm arm (seconds, default: %(default)s)",
    )
    parser.add_argument(
        "--max_imp", type=int, default=MAX_IMP,
        help="Max improving iterations per arm (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="RNG seed for synthetic data (default: %(default)s)",
    )
    parser.add_argument(
        "--n", type=int, default=N_PERSONS,
        help="Number of synthetic persons (default: %(default)s)",
    )
    args = parser.parse_args()
    main(
        algos     = args.algo,
        max_time  = args.max_time,
        max_imp   = args.max_imp,
        seed      = args.seed,
        n_persons = args.n,
    )
