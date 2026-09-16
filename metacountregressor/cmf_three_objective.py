"""Three-objective CMF programme + GSE helping-hands screen.

Mathematical programme (see paper, ``The Mathematical Programme''):

    minimise  F1 = BIC               (fit + parsimony, training MLE)
    minimise  F2 = n_insig            (significance: # of p > 0.05 terms)
    minimise  F3 = MAE^v              (validation mean absolute error)

subject to C1 monotonicity (eta_i > 0), C2 disjointness (upper XOR lower),
C3 dimension bounds, C4 correlation cap, C5 single family (NB/Poisson).

This module is solver-agnostic: it operates on plain candidate records so it
can be used by ``generator_new2.py`` (HPC workspace), by
``GA_CMF_AADT_JAX.py`` / ``family_search.py``, or in post-processing over
``search_history_full.csv`` files.

GSE helping-hands screen
------------------------
Mirrors the SearchLibrium ``MixedLogitGSE`` idea
(``src/SearchLibrium/MixedLogitGSE.py``: ``beta_nk = mu_k + gamma_k*g_nk +
sigma_k*eta_nk``) for count models.  A fixed NB/Poisson fit gives
per-observation score vectors ``s_n = d logL_n / d beta``; covariates whose
scores vary strongly across sites (large variance) or correlate strongly
with a candidate helping-hand variable are natural random-parameter and
heterogeneity-in-means/variance candidates.  ``suggest_random_params_gse``
implements that screen with finite-difference scores so it needs only a
fitted fixed model, not a full random-parameters fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Objective evaluation
# ---------------------------------------------------------------------------

@dataclass
class ThreeObjectiveScore:
    bic: float
    n_insig: int
    mae_val: float

    def as_vector(self) -> tuple[float, float, float]:
        return (float(self.bic), float(self.n_insig), float(self.mae_val))


def dominates_3d(a: Sequence[float], b: Sequence[float]) -> bool:
    """Return True if a weakly dominates b on all 3 objectives, strictly on one."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return False
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_3d(
    df: pd.DataFrame,
    bic_col: str = "BIC",
    insig_col: str = "N Insignificant (p>0.05)",
    mae_col: str = "Val MAE",
) -> np.ndarray:
    """Boolean mask of non-dominated rows under (BIC, n_insig, MAE)."""
    for col in (bic_col, insig_col, mae_col):
        if col not in df.columns:
            raise KeyError(f"pareto_3d: missing column {col!r}")
    scores = df[[bic_col, insig_col, mae_col]].to_numpy(dtype=float)
    n = scores.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i] or not np.all(np.isfinite(scores[i])):
            if not np.all(np.isfinite(scores[i])):
                efficient[i] = False
            continue
        dom = (
            np.all(scores <= scores[i], axis=1)
            & np.any(scores < scores[i], axis=1)
            & np.all(np.isfinite(scores), axis=1)
        )
        dom[i] = False
        if np.any(dom):
            efficient[i] = False
    return efficient


def pareto_2d_fallback(
    df: pd.DataFrame,
    bic_col: str = "BIC",
    rmse_col: str = "Val RMSE",
) -> np.ndarray:
    """Legacy 2-objective (BIC, RMSE) Pareto mask for histories without Val MAE."""
    scores = df[[bic_col, rmse_col]].to_numpy(dtype=float)
    n = scores.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        bi, ri = scores[i]
        dominated = (
            (scores[:, 0] <= bi)
            & (scores[:, 1] <= ri)
            & ((scores[:, 0] < bi) | (scores[:, 1] < ri))
        )
        dominated[i] = False
        if np.any(dominated):
            efficient[i] = False
    return efficient


def select_three_objective(
    history: pd.DataFrame,
    benchmark_bic: float,
    benchmark_mae: float,
    require_benchmark_dominance: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """Rank history on (BIC, n_insig, MAE); significance leads, MAE breaks ties.

    Mirrors the paper's final-selection safeguards: eligible candidates beat
    the benchmark on selection BIC and validation MAE; ranking leads with
    n_insig, then MAE improvement, then BIC.  Returns (selected_row, scored).
    """
    scored = history.copy()
    mae_col = "Val MAE" if "Val MAE" in scored.columns else "Val RMSE"
    insig_col = "N Insignificant (p>0.05)"
    scored["Pareto Efficient (3obj)"] = (
        pareto_3d(scored) if mae_col == "Val MAE"
        else pareto_2d_fallback(scored)
    )
    scored["Beats benchmark (BIC+MAE)"] = (
        (pd.to_numeric(scored["BIC"], errors="coerce") < float(benchmark_bic))
        & (pd.to_numeric(scored[mae_col], errors="coerce") < float(benchmark_mae))
    )
    pool = scored[scored["Pareto Efficient (3obj)"]].copy()
    beating = pool[pool["Beats benchmark (BIC+MAE)"]]
    if require_benchmark_dominance and beating.empty:
        raise RuntimeError("No 3-objective Pareto candidate beats the benchmark.")
    pool = beating if not beating.empty else pool
    if pool.empty:
        raise RuntimeError("Three-objective selection failed: empty pool.")
    pool = pool.sort_values(
        [insig_col, mae_col, "BIC"], ascending=[True, True, True]
    )
    return pool.iloc[0], scored


# ---------------------------------------------------------------------------
# GSE helping-hands screen for random parameters
# ---------------------------------------------------------------------------

def _nb_loglik_row(y: float, mu: float, alpha: float) -> float:
    from math import lgamma, log

    mu = min(max(mu, 1e-10), 1e10)
    alpha = max(alpha, 1e-8)
    return (
        lgamma(y + alpha)
        - lgamma(y + 1)
        - lgamma(alpha)
        + alpha * log(alpha / (alpha + mu))
        + y * log(mu / (alpha + mu))
    )


def suggest_random_params_gse(
    df: pd.DataFrame,
    y_col: str,
    mu_col: str = "mu_fixed",
    x_cols: Sequence[str] | None = None,
    helper_cols: Sequence[str] | None = None,
    alpha: float = 1.0,
    family: str = "nb",
    top_k: int = 4,
    eps: float = 1e-5,
) -> pd.DataFrame:
    """Score-based screen for random-parameter + helping-hand candidates.

    Parameters
    ----------
    df : frame containing the response, fitted fixed-model means (``mu_col``),
        design columns (``x_cols``) and candidate helpers (``helper_cols``).
    mu_col : column of fixed-model fitted means.  If absent, the caller must
        supply it (fit the fixed spec first, then call this function).
    x_cols : tested random-parameter candidates (defaults: all numeric except
        response/mu/offset columns).
    helper_cols : variables tested as heterogeneity-in-means/variance drivers
        (defaults to ``x_cols``).
    Returns a ranked frame with per-candidate score variance, dispersion
    index, and best-helper correlations, mirroring MixedLogitGSE's
    ``gamma_k * g_nk`` loadings: a large ``|corr(score_k, helper)|`` means the
    helper explains that coefficient's instability and should enter as a
    heterogeneity (helping-hand) term.
    """
    import math

    if mu_col not in df.columns:
        raise KeyError(f"suggest_random_params_gse: {mu_col!r} not in df")
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mu = np.clip(
        pd.to_numeric(df[mu_col], errors="coerce").to_numpy(dtype=float),
        1e-10,
        1e10,
    )
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    skip = {y_col, mu_col, "OFFSET", "offset", "FREQ", "Headon", "ID"}
    if x_cols is None:
        x_cols = [c for c in numeric if c not in skip and df[c].std() > 0]
    if helper_cols is None:
        helper_cols = list(x_cols)
    X = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in x_cols]
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n, k = X.shape
    log_mu = np.log(mu)

    def total_ll(b: np.ndarray) -> float:
        eta = log_mu + X @ b
        m = np.clip(np.exp(eta), 1e-10, 1e10)
        if family == "poisson":
            return float(np.sum(y * np.log(m) - m))
        return float(sum(_nb_loglik_row(yi, mi, alpha) for yi, mi in zip(y, m)))

    # Central finite-difference scores: s_nk = d logL_n / d b_k at b = 0.
    S = np.zeros((n, k))
    for j in range(k):
        dp = np.zeros(k)
        dm = np.zeros(k)
        dp[j] = eps
        dm[j] = -eps
        eta_p = log_mu + X @ dp
        eta_m = log_mu + X @ dm
        mu_p = np.clip(np.exp(eta_p), 1e-10, 1e10)
        mu_m = np.clip(np.exp(eta_m), 1e-10, 1e10)
        if family == "poisson":
            ll_p = y * np.log(mu_p) - mu_p
            ll_m = y * np.log(mu_m) - mu_m
        else:
            ll_p = np.array(
                [_nb_loglik_row(yi, mi, alpha) for yi, mi in zip(y, mu_p)]
            )
            ll_m = np.array(
                [_nb_loglik_row(yi, mi, alpha) for yi, mi in zip(y, mu_m)]
            )
        S[:, j] = (ll_p - ll_m) / (2.0 * eps)

    H = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in helper_cols]
    )
    H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
    rows = []
    for j, var in enumerate(x_cols):
        s = S[:, j]
        svar = float(np.var(s))
        # Dispersion index: Var(score) / (1 + |mean|) — high means unstable coef.
        disp = svar / (1.0 + abs(float(np.mean(s))))
        best_h, best_c = "-", 0.0
        for h_i, hvar in enumerate(helper_cols):
            h = H[:, h_i]
            if np.std(h) <= 0 or np.std(s) <= 0:
                continue
            c = float(np.corrcoef(s, h)[0, 1])
            if abs(c) > abs(best_c):
                best_c, best_h = c, str(hvar)
        rows.append(
            {
                "candidate": var,
                "score_var": svar,
                "dispersion_index": disp,
                "best_helper": best_h,
                "corr(score,best_helper)": best_c,
                "recommend_random": disp > float(np.median([r["dispersion_index"] for r in rows] + [disp])),
            }
        )
    out = pd.DataFrame(rows).sort_values("dispersion_index", ascending=False)
    if top_k and len(out) > top_k:
        out = out.head(top_k).copy()
    # Suppress unused-variable warning for total_ll helper (kept for callers
    # that want full-likelihood checks).
    _ = total_ll
    return out.reset_index(drop=True)
