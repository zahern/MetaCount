#!/usr/bin/env python3
"""
MetaCount Latent-Class Experiment Runner
=========================================
Unified script for running Latent-Class count and Tobit models using the
metacountregressor package on an HPC Linux system.

Supports two experiment families:
  lc_count   -- Latent-class NB2 / Poisson count model  (crash frequency)
  lc_tobit   -- Latent-class Tobit model                 (crash rate, censored at 0)

Usage examples
--------------
# 1. Synthetic data quick-check (no data file needed):
python run_experiment.py --experiment lc_count   --synthetic
python run_experiment.py --experiment lc_tobit   --synthetic

# 2. Real data — both families:
python run_experiment.py --experiment lc_count   --data data/crashes.csv \\
    --id_col site_id --y_col crashes --n_classes 2 \\
    --variables aadt lanes speed shoulder \\
    --membership_vars ramp_flag citybound \\
    --output_dir results/

python run_experiment.py --experiment lc_tobit   --data data/crash_rates.csv \\
    --id_col site_id --y_col rate_per_100mveh --n_classes 2 \\
    --variables h085 aadt_per_lane citybound pct_heavy \\
    --membership_vars entry_ramp exit_ramp \\
    --output_dir results/

# 3. Automated specification search (SA over variable roles):
python run_experiment.py --experiment lc_count   --data data/crashes.csv \\
    --mode search --max_classes 3 --search_iter 3000 \\
    --id_col site_id --y_col crashes --output_dir results/

python run_experiment.py --experiment lc_tobit   --data data/crash_rates.csv \\
    --mode search --max_classes 2 --search_iter 3000 \\
    --id_col site_id --y_col rate_per_100mveh --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generators (self-contained — no external data required)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_lc_count(N: int = 800, T: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Two-class NB2 panel dataset.
    Class 1 (low severity):  intercept=-3.0  x1=+1.0  x2=-0.8  x3=+0.3  x4=+0.1
    Class 2 (high severity): intercept=-0.8  x1=-0.4  x2=+0.6  x3=+0.2  x4=+0.9
    Membership: log-odds = 0 + 1.5*z1 - 1.0*z2
    """
    rng = np.random.default_rng(seed)
    TRUE = {
        1: {"intercept": -3.0, "x1": 1.0, "x2": -0.8, "x3": 0.3, "x4": 0.1, "alpha": 1.5},
        2: {"intercept": -0.8, "x1": -0.4, "x2": 0.6, "x3": 0.2, "x4": 0.9, "alpha": 0.6},
    }
    rows = []
    for i in range(N):
        x1, x2, x3, x4 = rng.normal(0,1,4)
        z1, z2 = rng.normal(0,1), rng.normal(0,1)
        p2 = 1.0 / (1.0 + np.exp(-(0.0 + 1.5*z1 - 1.0*z2)))
        c = rng.choice([1,2], p=[1-p2, p2])
        p = TRUE[c]
        for t in range(T):
            eta = p["intercept"] + p["x1"]*x1 + p["x2"]*x2 + p["x3"]*x3 + p["x4"]*x4
            mu  = np.exp(np.clip(eta, -20, 20))
            a   = p["alpha"]
            y   = rng.negative_binomial(a, a/(a+mu))
            rows.append(dict(id=i, t=t, x1=x1, x2=x2, x3=x3, x4=x4,
                             z1=z1, z2=z2, y=int(y), true_class=c))
    return pd.DataFrame(rows)


def _synthetic_lc_tobit(N: int = 700, T: int = 3, seed: int = 0) -> pd.DataFrame:
    """
    Two-class Tobit panel dataset (left-censored at 0).
    Class 1 (low rate):  intercept=-1.0  x1=+0.8  x2=-0.5  x3=+0.2  sigma=1.0
    Class 2 (high rate): intercept=+2.0  x1=-0.3  x2=+0.7  x3=+0.1  sigma=1.5
    Membership: log-odds = 0 + 1.2*z1 - 0.9*z2
    """
    rng = np.random.default_rng(seed)
    TRUE = {
        1: {"intercept": -1.0, "x1": 0.8, "x2": -0.5, "x3": 0.2, "sigma": 1.0},
        2: {"intercept":  2.0, "x1": -0.3, "x2": 0.7, "x3": 0.1, "sigma": 1.5},
    }
    rows = []
    for i in range(N):
        x1, x2, x3 = rng.normal(0,1,3)
        z1, z2 = rng.normal(0,1), rng.normal(0,1)
        p2 = 1.0 / (1.0 + np.exp(-(0.0 + 1.2*z1 - 0.9*z2)))
        c = rng.choice([1,2], p=[1-p2, p2])
        p = TRUE[c]
        for t in range(T):
            eta   = p["intercept"] + p["x1"]*x1 + p["x2"]*x2 + p["x3"]*x3
            y_obs = max(0.0, eta + rng.normal(0, p["sigma"]))
            rows.append(dict(id=i, t=t, x1=x1, x2=x2, x3=x3,
                             z1=z1, z2=z2, y=float(y_obs), true_class=c))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(fit_result: dict, label: str, out_dir: Path):
    """Persist coefficient table, BIC, and predictions to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = fit_result.get("summary", {})

    # ── BIC / stats ──────────────────────────────────────────────
    stats = {k: summary.get(k) for k in ["loglik", "bic", "aic", "num_parm", "n_obs"]}
    stats["label"] = label
    json_path = out_dir / f"{label}_stats.json"
    with open(json_path, "w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    # ── Predictions ──────────────────────────────────────────────
    preds = fit_result.get("predictions")
    if preds is not None:
        pd.DataFrame({"prediction": np.array(preds).flatten()}).to_csv(
            out_dir / f"{label}_predictions.csv", index=False)

    # ── Raw params ───────────────────────────────────────────────
    params = np.array(fit_result["result"].params)
    pd.DataFrame({"param": params}).to_csv(
        out_dir / f"{label}_params.csv", index=False)

    print(f"  [saved] {out_dir / label}_(stats|predictions|params).csv")


def _print_summary_block(fit_result: dict, label: str):
    summary = fit_result.get("summary", {})
    ll  = summary.get("loglik",   float("nan"))
    bic = summary.get("bic",      float("nan"))
    k   = summary.get("num_parm", "?")
    print(f"\n  {label}: LL={ll:.2f}  k={k}  BIC={bic:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def run_lc_count(args, df: pd.DataFrame, out_dir: Path):
    print("\n" + "="*65)
    print("  LATENT-CLASS COUNT MODEL  (NB2)")
    print("="*65)

    from metacountregressor import ExperimentBuilder

    builder = ExperimentBuilder(
        df=df,
        id_col=args.id_col,
        y_col=args.y_col,
        offset_col=args.offset_col or None,
    )

    variables   = [v.strip() for v in args.variables.split(",")]   if args.variables else None
    mem_vars    = [v.strip() for v in args.membership_vars.split(",")] if args.membership_vars else []
    n_classes   = args.n_classes
    R           = args.n_draws
    results     = {}

    if args.mode == "manual":
        for C in range(1, n_classes + 1):
            label = f"lc{C}_nb"
            print(f"\n  Fitting {label} ...")
            spec = builder.make_manual_spec(
                fixed_terms      = variables,
                membership_terms = mem_vars if C > 1 else [],
                dispersion       = 1,
                latent_classes   = C,
            )
            fit = builder.fit_manual_model(
                manual_spec = spec,
                model       = "nb",
                R           = R,
                print_report= True,
            )
            results[label] = fit
            _save_results(fit, label, out_dir)

    elif args.mode == "search":
        print(f"\n  Running SA specification search  "
              f"(max_classes={args.max_classes}, iter={args.search_iter}) ...")
        evaluator = builder.build_search(
            model_family       = "count",
            variables          = variables,
            max_latent_classes = args.max_classes,
            R                  = R,
        )
        from metacountregressor import ExperimentBuilder as _EB
        result = builder.run(evaluator, algo="sa", max_iter=args.search_iter)
        (out_dir / "search_result.json").write_text(
            json.dumps(result, indent=2, default=str))
        print(f"  Search complete — saved to {out_dir / 'search_result.json'}")
        return

    # Comparison table
    print("\n" + "="*65 + "\n  MODEL COMPARISON  (BIC)\n" + "="*65)
    rows = []
    for lbl, fit in results.items():
        s = fit.get("summary", {})
        rows.append({"label": lbl,
                     "LL": s.get("loglik", float("nan")),
                     "k":  s.get("num_parm", "?"),
                     "BIC": s.get("bic", float("nan"))})
    cmp = pd.DataFrame(rows).sort_values("BIC")
    cmp["dBIC"] = cmp["BIC"] - cmp["BIC"].min()
    print(cmp.to_string(index=False))
    cmp.to_csv(out_dir / "lc_count_comparison.csv", index=False)


def run_lc_tobit(args, df: pd.DataFrame, out_dir: Path):
    print("\n" + "="*65)
    print("  LATENT-CLASS TOBIT MODEL  (left-censored at 0)")
    print("="*65)

    from metacountregressor import ExperimentBuilder

    builder = ExperimentBuilder(
        df=df,
        id_col=args.id_col,
        y_col=args.y_col,
        offset_col=None,        # Tobit models crash rate directly
    )

    variables   = [v.strip() for v in args.variables.split(",")]   if args.variables else None
    mem_vars    = [v.strip() for v in args.membership_vars.split(",")] if args.membership_vars else []
    n_classes   = args.n_classes
    R           = args.n_draws
    results     = {}

    if args.mode == "manual":
        for C in range(1, n_classes + 1):
            label = f"lc{C}_tobit"
            print(f"\n  Fitting {label} ...")
            spec = builder.make_manual_spec(
                fixed_terms      = variables,
                membership_terms = mem_vars if C > 1 else [],
                dispersion       = 0,
                latent_classes   = C,
            )
            fit = builder.fit_manual_model(
                manual_spec = spec,
                model       = "tobit",
                R           = R,
                print_report= True,
            )
            results[label] = fit
            _save_results(fit, label, out_dir)

    elif args.mode == "search":
        print(f"\n  Running SA specification search (Tobit)  "
              f"(max_classes={args.max_classes}, iter={args.search_iter}) ...")
        evaluator = builder.build_search(
            model_family       = "tobit",
            variables          = variables,
            max_latent_classes = args.max_classes,
            R                  = R,
        )
        result = builder.run(evaluator, algo="sa", max_iter=args.search_iter)
        (out_dir / "tobit_search_result.json").write_text(
            json.dumps(result, indent=2, default=str))
        print(f"  Search complete — saved to {out_dir / 'tobit_search_result.json'}")
        return

    # Comparison table
    print("\n" + "="*65 + "\n  MODEL COMPARISON  (BIC)\n" + "="*65)
    rows = []
    for lbl, fit in results.items():
        s = fit.get("summary", {})
        rows.append({"label": lbl,
                     "LL": s.get("loglik", float("nan")),
                     "k":  s.get("num_parm", "?"),
                     "BIC": s.get("bic", float("nan"))})
    cmp = pd.DataFrame(rows).sort_values("BIC")
    cmp["dBIC"] = cmp["BIC"] - cmp["BIC"].min()
    print(cmp.to_string(index=False))
    cmp.to_csv(out_dir / "lc_tobit_comparison.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MetaCount LC / LC-Tobit experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Experiment selection
    p.add_argument("--experiment", choices=["lc_count","lc_tobit","both"],
                   default="both",
                   help="Which model family to run. Default: both.")
    p.add_argument("--mode", choices=["manual","search"], default="manual",
                   help="'manual' fits 1..n_classes models; "
                        "'search' runs SA over variable structure. Default: manual.")

    # Data
    p.add_argument("--data",    default=None,
                   help="Path to CSV data file (relative to this script or absolute). "
                        "If omitted, synthetic data is generated.")
    p.add_argument("--synthetic", action="store_true",
                   help="Force use of synthetic data even if --data is given.")
    p.add_argument("--id_col",     default="id",   help="Panel ID column. Default: id.")
    p.add_argument("--y_col",      default="y",    help="Outcome column. Default: y.")
    p.add_argument("--offset_col", default=None,   help="Exposure offset column (count models).")
    p.add_argument("--variables",  default=None,
                   help="Comma-separated outcome variable names. "
                        "Default: all non-reserved columns.")
    p.add_argument("--membership_vars", default=None,
                   help="Comma-separated membership variable names (class probability eq.).")

    # Model options
    p.add_argument("--n_classes",   type=int, default=2,
                   help="Fit 1..n_classes models (manual mode). Default: 2.")
    p.add_argument("--max_classes", type=int, default=3,
                   help="Upper bound on classes in search mode. Default: 3.")
    p.add_argument("--n_draws",     type=int, default=200,
                   help="Halton draws for mixed models. Default: 200.")
    p.add_argument("--search_iter", type=int, default=3000,
                   help="SA iterations in search mode. Default: 3000.")

    # Output
    p.add_argument("--output_dir", default="results",
                   help="Directory for results. Default: results/")
    p.add_argument("--seed",  type=int, default=42, help="RNG seed. Default: 42.")
    p.add_argument("--tag",   default="",
                   help="Optional tag appended to output sub-directory name.")

    return p.parse_args()


def main():
    args    = parse_args()
    t_start = time.time()

    # ── Output directory ─────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag     = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.output_dir) / f"{ts}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  MetaCount LC Experiment Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  experiment : {args.experiment}")
    print(f"  mode       : {args.mode}")
    print(f"  output     : {out_dir}")
    print("=" * 65)

    # ── Load / generate data ──────────────────────────────────────
    if args.data and not args.synthetic:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = HERE / data_path
        print(f"\n  Loading data: {data_path}")
        df_count = pd.read_csv(data_path)
        df_tobit = df_count.copy()
    else:
        print("\n  Generating synthetic data ...")
        df_count = _synthetic_lc_count(N=800, T=3, seed=args.seed)
        df_tobit = _synthetic_lc_tobit(N=700, T=3, seed=args.seed)
        print(f"  Count dataset : {df_count['id'].nunique()} sites  "
              f"y mean={df_count['y'].mean():.2f}  "
              f"zeros={100*(df_count['y']==0).mean():.1f}%")
        print(f"  Tobit dataset : {df_tobit['id'].nunique()} sites  "
              f"y mean={df_tobit['y'].mean():.2f}  "
              f"zeros={100*(df_tobit['y']==0).mean():.1f}%")

        if args.membership_vars is None:
            args.membership_vars = "z1,z2"

    # ── Run experiments ───────────────────────────────────────────
    # For synthetic "both", each family has different predictor sets.
    # We clone args and set family-specific defaults to avoid cross-contamination.
    import copy

    if args.experiment in ("lc_count", "both"):
        args_count = copy.copy(args)
        if args_count.variables is None:
            args_count.variables = "x1,x2,x3,x4"
        run_lc_count(args_count, df_count, out_dir / "lc_count")

    if args.experiment in ("lc_tobit", "both"):
        args_tobit = copy.copy(args)
        if args_tobit.variables is None:
            args_tobit.variables = "x1,x2,x3"
        # Filter to columns actually present in the Tobit dataset
        avail = set(df_tobit.columns)
        reserved = {args_tobit.id_col, args_tobit.y_col}
        args_tobit.variables = ",".join(
            v for v in args_tobit.variables.split(",") if v.strip() in avail
        ) or args_tobit.variables
        run_lc_tobit(args_tobit, df_tobit, out_dir / "lc_tobit")

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"  Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
