"""
Benchmark: Standard EM vs SQUAREM – MetaCountRegressor Latent-Class NB Model
=============================================================================
Compares convergence speed (EM-call count, wall-clock time, final LL) between
``fit_em`` (standard EM) and ``fit_em_squarem`` (SQUAREM-accelerated EM) for
a 2-class Negative-Binomial latent-class count model.

SQUAREM reference:
  Varadhan, R. & Roland, C. (2008). Simple and globally convergent methods for
  accelerating the convergence of any EM algorithm.
  Scandinavian Journal of Statistics, 35(2), 335–353.
  R package: https://cran.r-project.org/web/packages/SQUAREM/index.html

Run from the metacountregressor/ directory:
    python benchmark_squarem_mcr.py

Requires: jax, jaxopt  (installed in the MetaCount environment)
"""

from __future__ import annotations

import sys
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ── Apply LC patch FIRST ─────────────────────────────────────────────────────
import main_hpc_lc_patch  # noqa: F401  (monkey-patches _hpc in place)
from main_hpc_lc_patch import (
    ModelSpec, build_jax_data, mixed_model_loglik,
    fit_em, fit_em_squarem, build_base_index,
    build_model_from_manual_spec,
)
from experiment_lc_model_comparison import generate_data
from dataclasses import replace
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char="=", w=72): print(char * w)

def _header(t): _sep(); print(f"  {t}"); _sep()

def _result_row(label, n_outer, em_calls, wall, ll, conv):
    c = "yes" if conv else "NO"
    print(
        f"  {label:<18}  outer={n_outer:>4}  em_calls={em_calls:>4}"
        f"  time={wall:>6.1f}s  LL={ll:>12.4f}  conv={c}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation helper
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_VARS    = ["x1", "x2", "x3", "x4"]
MEMBERSHIP_VARS = ["z1", "z2"]


def _make_data_and_spec(df, R=50):
    """Return (data dict, LC-2 NB ModelSpec with membership) from a panel DataFrame."""
    manual_spec = {
        "fixed_terms":      OUTCOME_VARS,
        "membership_terms": MEMBERSHIP_VARS,
        "dispersion":       1,   # NB2
        "latent_classes":   2,
    }
    data, spec = build_model_from_manual_spec(
        df, manual_spec, id_col="id", y_col="y", R=R
    )
    return data, spec


def _make_init_params(spec, rng):
    """Small random start for theta, zeros for gamma."""
    K_mem = spec.K_membership
    C = spec.latent_classes
    base_spec = replace(spec, latent_classes=1)
    K_base = build_base_index(base_spec, model=spec.model)["total_params"]
    gamma_size = (C - 1) * (K_mem + 1)
    theta = rng.normal(scale=0.05, size=C * K_base)
    gamma = np.zeros(gamma_size)
    return np.concatenate([theta, gamma])


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(N=300, T=3, max_iter=40, tol=1e-4, seeds=(42, 43, 44), R=50):
    _header(f"MetaCountRegressor  LC-2 NB + membership:  fit_em vs fit_em_squarem")
    print(f"\n  N={N}  T={T}  max_iter={max_iter}  tol={tol:.0e}  R={R}")
    print()

    results = {"standard": [], "squarem": []}

    for seed in seeds:
        df = generate_data(N=N, T=T, seed=seed)

        try:
            data, spec = _make_data_and_spec(df, R=R)
        except Exception as exc:
            print(f"  [SKIP seed={seed}] data/spec setup failed: {exc}")
            continue

        rng = np.random.default_rng(seed)
        init = _make_init_params(spec, rng)

        for label, fn in [("standard", fit_em), ("squarem", fit_em_squarem)]:
            t0 = time.perf_counter()
            try:
                out = fn(
                    init.copy(), data, spec,
                    max_iter=max_iter, tol=tol, verbose=False, return_trace=True,
                )
                elapsed = time.perf_counter() - t0
                best, trace = out

                if not trace:
                    ll = float("nan"); n_outer = 0; em_calls_total = 0; conv = False
                elif label == "squarem":
                    # trace: (outer_iter, em_calls, alpha, ll, delta_ll, shares)
                    n_outer       = trace[-1][0] + 1
                    em_calls_total = trace[-1][1]
                    ll             = trace[-1][3]
                    conv           = trace[-1][4] < tol
                else:
                    # trace: (iter, T, m_iters, ll, delta_ll, shares)
                    n_outer        = trace[-1][0] + 1
                    em_calls_total = n_outer
                    ll             = trace[-1][3]
                    conv           = trace[-1][4] < tol

                results[label].append({
                    "seed": seed, "n_outer": n_outer, "em_calls": em_calls_total,
                    "wall": elapsed, "loglik": ll, "converged": conv,
                })
                tag = f"{'Std EM' if label=='standard' else 'SQUAREM'} (seed={seed})"
                _result_row(tag, n_outer, em_calls_total, elapsed, ll, conv)

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                print(f"  {'Std EM' if label=='standard' else 'SQUAREM'} (seed={seed})  ERROR: {exc}")

    print()
    _sep("-")
    print("  --- Averages across seeds ---")
    for label in ("standard", "squarem"):
        rs = results[label]
        if not rs:
            print(f"  {'Standard EM' if label=='standard' else 'SQUAREM':<14}  no results")
            continue
        avg_calls = np.mean([r["em_calls"] for r in rs])
        avg_time  = np.mean([r["wall"]     for r in rs])
        avg_ll    = np.mean([r["loglik"]   for r in rs])
        conv_rate = np.mean([r["converged"] for r in rs])
        print(
            f"  {'Standard EM' if label=='standard' else 'SQUAREM':<14}"
            f"  avg_em_calls={avg_calls:>5.1f}"
            f"  avg_time={avg_time:>5.1f}s"
            f"  avg_LL={avg_ll:>12.4f}"
            f"  conv_rate={conv_rate:.0%}"
        )

    std_rs = results["standard"]; sq_rs = results["squarem"]
    if std_rs and sq_rs:
        avg_std = np.mean([r["em_calls"] for r in std_rs])
        avg_sq  = np.mean([r["em_calls"] for r in sq_rs])
        avg_ts  = np.mean([r["wall"]     for r in std_rs])
        avg_tq  = np.mean([r["wall"]     for r in sq_rs])
        print()
        if avg_sq > 0:
            print(f"  EM-call speedup  : {avg_std / avg_sq:.2f}x")
        if avg_tq > 0:
            print(f"  Wall-time speedup: {avg_ts / avg_tq:.2f}x")
    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# Convergence trace table (single seed)
# ─────────────────────────────────────────────────────────────────────────────

def convergence_table(N=300, T=3, max_iter=40, tol=1e-4, seed=42, R=50):
    _header("Convergence trace: LL per outer iteration (single run)")
    df   = generate_data(N=N, T=T, seed=seed)
    data, spec = _make_data_and_spec(df, R=R)
    rng  = np.random.default_rng(seed)
    init = _make_init_params(spec, rng)

    traces = {}
    for label, fn in [("standard", fit_em), ("squarem", fit_em_squarem)]:
        try:
            _, trace = fn(
                init.copy(), data, spec,
                max_iter=max_iter, tol=tol, verbose=False, return_trace=True,
            )
            traces[label] = trace
        except Exception as exc:
            print(f"  [SKIP {label}]: {exc}")
            traces[label] = []

    std = traces.get("standard", [])
    sq  = traces.get("squarem",  [])

    print(f"\n  {'Outer iter':<12}  {'EM calls (std)':>15}  {'LL (standard)':>15}"
          f"  {'EM calls (sq)':>14}  {'LL (SQUAREM)':>14}")
    print(f"  {'-'*12}  {'-'*15}  {'-'*15}  {'-'*14}  {'-'*14}")

    max_len = max(len(std), len(sq))
    for i in range(max_len):
        if i < len(std):
            # (iter, T, m_iters, ll, delta_ll, shares)
            s_ec = i + 1
            s_ll = f"{std[i][3]:15.4f}"
        else:
            s_ec = "—"; s_ll = f"{'(converged)':>15}"
        if i < len(sq):
            # (outer_iter, em_calls, alpha, ll, delta_ll, shares)
            q_ec = sq[i][1]
            q_ll = f"{sq[i][3]:14.4f}"
        else:
            q_ec = "—"; q_ll = f"{'(converged)':>14}"
        print(f"  {i+1:<12}  {str(s_ec):>15}  {s_ll}  {str(q_ec):>14}  {q_ll}")

    if std:
        print(f"\n  Standard EM : {len(std)} outer iters / {len(std)} EM calls")
    if sq:
        print(f"  SQUAREM     : {len(sq)} outer iters / {sq[-1][1]} EM calls total")
    _sep()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    _header("MetaCountRegressor – SQUAREM Benchmark")
    print("  Varadhan & Roland (2008) SQUAREM for EM acceleration")
    print("  https://cran.r-project.org/web/packages/SQUAREM/index.html")
    _sep()
    print()

    benchmark(N=300, T=3, max_iter=40, tol=1e-4, seeds=(42, 43, 44), R=50)
    print()
    convergence_table(N=300, T=3, max_iter=40, tol=1e-4, seed=42, R=50)
    print()
    _sep()
    print("  Benchmark complete.")
    _sep()
