"""
Pavement Clusterwise Linear Regression (CLR) with Temporal Error Structures
============================================================================
Extends metacountregressor with a specialized SA-based CLR framework for
pavement deterioration modelling.

Models the power (log-log) deterioration equation:
    ln(PSI) = β₀ + Σᵢ βᵢ ln(xᵢ) + Σⱼ γⱼ Dⱼ + εₜ

where ε can follow i.i.d. (OLS), AR(1), random walk, or near-unit-root structure.

Key classes
-----------
PavementCLROptimizer
    SA-based simultaneous cluster assignment + variable selection, minimising BIC.

PavementTemporalComparison
    Fits and compares four temporal error models within the same CLR structure.

Helper functions
----------------
fit_cluster_ols / fit_cluster_ar1 / fit_cluster_random_walk / fit_cluster_nur
    Per-cluster model fitters (log-transformed data assumed as input).
"""

from __future__ import annotations

import warnings
from itertools import combinations as _it_comb
from math import log, exp, pi, sqrt
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize as _scipy_minimize
from scipy.stats import t as _tdist

__all__ = [
    "PavementCLROptimizer",
    "PavementTemporalComparison",
    "fit_cluster_ols",
    "fit_cluster_ar1",
    "fit_cluster_random_walk",
    "fit_cluster_nur",
    "log_transform_pavement",
]


# ============================================================
# Data transformation
# ============================================================

def log_transform_pavement(
    df: pd.DataFrame,
    psi_col: str = "psi",
    continuous_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Apply log transform to PSI and continuous predictors.

    This converts the power-form model to a linear model in log space:
        ln(PSI) = β₀ + Σᵢ βᵢ ln(xᵢ) + categorical + ε

    Parameters
    ----------
    df : DataFrame
        Original (untransformed) pavement data.
    psi_col : str
        Name of the PSI column.
    continuous_cols : list[str], optional
        Continuous predictor column names to log-transform.
        If None, all non-categorical float columns (except IDs/time) are used.

    Returns
    -------
    DataFrame with ln(PSI) and ln(continuous predictors). Zero values are
    replaced with NaN before logging and then filled with 0.
    """
    df = df.copy()
    cols_to_log = [psi_col] + (continuous_cols or [])
    for col in cols_to_log:
        if col not in df.columns:
            continue
        vals = df[col].astype(float)
        vals = vals.replace(0.0, np.nan)
        df[col] = np.log(vals)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ============================================================
# OLS utilities (pure NumPy, no JAX dependency)
# ============================================================

def _solve_ols(X: np.ndarray, y: np.ndarray):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(np.sum(resid ** 2))
    return beta, resid, rss


def _fit_lm(X: np.ndarray, y: np.ndarray, col_names: list[str]):
    n, p = X.shape
    if n <= p:
        return None
    beta, resid, rss = _solve_ols(X, y)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 1e-15 else 0.0
    df_resid = max(n - p, 1)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid
    sigma2 = rss / df_resid
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    t_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p_vals = 2.0 * (1.0 - _tdist.cdf(np.abs(t_stats), df=df_resid))
    return {
        "coefficients": beta,
        "coeff_dict": {col_names[i]: float(beta[i]) for i in range(len(col_names))},
        "residuals": resid,
        "rss": rss,
        "tss": tss,
        "r2": r2,
        "adj_r2": adj_r2,
        "se": se,
        "p_values": p_vals,
        "n": n,
        "df_resid": df_resid,
        "col_names": list(col_names),
    }


def _bic(n: int, p: int, rss: float) -> float:
    if n <= 0 or rss <= 0:
        return np.inf
    sigma2 = rss / n
    return n + n * log(2 * pi) + n * log(sigma2) + p * log(max(n, 1))


def _build_design_matrix(
    df_log: pd.DataFrame,
    psi_col: str,
    selected_vars: list[str],
    all_vars: list[str],
    cat_vars: set[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n = len(df_log)
    y = df_log[psi_col].values.astype(float)
    col_names = ["Intercept"]
    X_blocks = [np.ones((n, 1))]
    for vname in selected_vars:
        vals = df_log[vname].values.astype(float)
        if vname in cat_vars:
            uniq = sorted(set(vals))
            if len(uniq) >= 2:
                for lev in uniq[1:]:
                    X_blocks.append((vals == lev).astype(float).reshape(-1, 1))
                    col_names.append(f"{vname}_{int(lev)}")
        else:
            X_blocks.append(vals.reshape(-1, 1))
            col_names.append(vname)
    return np.column_stack(X_blocks), y, col_names


def _calc_vif(X_no_intercept: np.ndarray, col_names: list[str]) -> dict:
    n, p = X_no_intercept.shape
    if p < 2:
        return {}
    vif = {}
    for i, name in enumerate(col_names):
        y_i = X_no_intercept[:, i]
        X_oth = np.delete(X_no_intercept, i, axis=1)
        X_reg = np.column_stack([np.ones(n), X_oth])
        try:
            b = np.linalg.lstsq(X_reg, y_i, rcond=None)[0]
        except np.linalg.LinAlgError:
            vif[name] = np.inf
            continue
        ss_res = np.sum((y_i - X_reg @ b) ** 2)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        vif[name] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vif


# ============================================================
# Per-cluster model fitters
# ============================================================

def fit_cluster_ols(
    df_log: pd.DataFrame,
    psi_col: str,
    variable_names: list[str],
    categorical_vars: set[str],
    level_of_significance: float = 0.05,
    max_vif: float = 10.0,
) -> Optional[dict]:
    """
    Fit an OLS cluster model with VIF screening and all-significant subset selection.

    Parameters
    ----------
    df_log : DataFrame
        Log-transformed cluster data (ln PSI as dependent variable).
    psi_col : str
        Name of the (already log-transformed) PSI column.
    variable_names : list[str]
        All candidate predictor names.
    categorical_vars : set[str]
        Subset of variable_names that are categorical.
    level_of_significance : float
        p-value threshold for coefficient significance.
    max_vif : float
        VIF threshold; variables exceeding this are iteratively removed.

    Returns
    -------
    dict with keys: coefficients, coeff_dict, r2, adj_r2, rss, bic,
    col_names, variables_used, n, p_values, se.
    None if no valid model found.
    """
    active = list(variable_names)

    # Drop categorical vars with < 2 levels in this cluster
    for i, vname in enumerate(variable_names):
        if vname in categorical_vars:
            if df_log[vname].nunique() < 2:
                active[i] = None

    # Iterative VIF removal
    prev_na = -1
    while True:
        used = [nm for nm in active if nm is not None]
        if len(used) < 1:
            return None
        cur_na = sum(1 for x in active if x is None)
        if cur_na == prev_na:
            break
        prev_na = cur_na
        X, y, col_names = _build_design_matrix(df_log, psi_col, used, variable_names, categorical_vars)
        if X.shape[0] <= X.shape[1]:
            return None
        vif = _calc_vif(X[:, 1:], col_names[1:])
        if not vif:
            break
        max_v = max(vif.values())
        if max_v <= max_vif:
            break
        worst = max(vif, key=vif.get)
        for i, nm in enumerate(active):
            if nm is None:
                continue
            if nm in categorical_vars and worst.startswith(nm + "_"):
                active[i] = None
                break
            if worst == nm:
                active[i] = None
                break

    used = [nm for nm in active if nm is not None]
    if len(used) < 1:
        return None

    # All-subsets BIC search with significance constraint
    all_combos = []
    for r in range(1, len(used) + 1):
        all_combos.extend(_it_comb(used, r))

    results = []
    for combo in all_combos:
        X, y, col_names = _build_design_matrix(
            df_log, psi_col, list(combo), variable_names, categorical_vars)
        if X.shape[0] <= X.shape[1]:
            continue
        mf = _fit_lm(X, y, col_names)
        if mf is None:
            continue
        bic_val = _bic(mf["n"], len(mf["coefficients"]), mf["rss"])
        results.append((bic_val, mf, list(combo)))

    if not results:
        return None
    results.sort(key=lambda x: x[0])
    for bic_val, mf, combo in results:
        if max(mf["p_values"]) <= level_of_significance:
            mf["variables_used"] = combo
            mf["bic"] = bic_val
            return mf
    return None


def fit_cluster_ar1(
    df_log: pd.DataFrame,
    psi_col: str,
    variable_names: list[str],
    categorical_vars: set[str],
    segment_col: str = "sample_id",
) -> Optional[dict]:
    """
    Fit an AR(1) error model per cluster via maximum likelihood.

    Model: ε_t = ρ ε_{t-1} + ν_t,  ν_t ~ N(0, σ²_ν)
    where ε_t = ln(PSI_t) - X_t β

    Parameters
    ----------
    df_log : DataFrame
        Log-transformed cluster data.
    psi_col : str
        Name of the ln(PSI) column.
    variable_names : list[str]
        All candidate predictor names.
    categorical_vars : set[str]
        Subset of variable_names that are categorical.
    segment_col : str
        Column identifying pavement segments (panel structure).

    Returns
    -------
    dict with r2, adj_r2, rss, bic, rho_ar1, sigma_ar1, and model_type.
    """
    X, y, col_names = _build_design_matrix(
        df_log, psi_col, variable_names, variable_names, categorical_vars)
    n, p = X.shape
    if n <= p + 2:
        return None

    ols = _fit_lm(X, y, col_names)
    if ols is None:
        return None
    beta0 = ols["coefficients"][:p].copy()
    sigma0 = sqrt(max(ols["rss"] / (n - p), 1e-8))

    seg_ids = df_log[segment_col].values.astype(int)
    unique_segs = np.unique(seg_ids)
    seg_idx = {s: np.where(seg_ids == s)[0] for s in unique_segs}

    def neg_ll(params):
        beta = params[:p]
        rho = np.clip(params[p], -0.999, 0.999)
        log_sig = params[p + 1]
        sigma2 = np.exp(2 * log_sig)
        resid = y - X @ beta
        total = 0.0
        for s, idx in seg_idx.items():
            r = resid[idx]
            T = len(r)
            if T < 2:
                continue
            innovations = r[1:] - rho * r[:-1]
            total += 0.5 * (T - 1) * log(2 * pi * sigma2)
            total += 0.5 * np.sum(innovations ** 2) / sigma2
            # Stationary initial condition
            var_init = sigma2 / max(1.0 - rho ** 2, 1e-10)
            total += 0.5 * log(2 * pi * var_init) + 0.5 * r[0] ** 2 / var_init
        return total

    x0 = np.zeros(p + 2)
    x0[:p] = beta0
    x0[p] = 0.5
    x0[p + 1] = log(max(sigma0, 1e-4))
    try:
        res = _scipy_minimize(neg_ll, x0, method="L-BFGS-B",
                              bounds=[(-np.inf, np.inf)] * p + [(-0.999, 0.999), (-10, 10)],
                              options={"maxiter": 300, "ftol": 1e-7})
        params = res.x
    except Exception:
        return None

    beta = params[:p]
    rho = float(np.clip(params[p], -0.999, 0.999))
    sigma = float(np.exp(params[p + 1]))

    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / max(tss, 1e-15)
    n_params = p + 2
    bic_val = _bic(n, n_params, rss)

    return {
        "coefficients": beta,
        "coeff_dict": {col_names[i]: float(beta[i]) for i in range(len(col_names))},
        "residuals": resid,
        "rss": rss,
        "tss": tss,
        "r2": r2,
        "adj_r2": 1.0 - (1.0 - r2) * (n - 1) / max(n - n_params, 1),
        "se": np.full(p, np.nan),
        "p_values": np.full(p, np.nan),
        "n": n,
        "col_names": col_names,
        "variables_used": list(variable_names),
        "rho_ar1": rho,
        "sigma_ar1": sigma,
        "bic": bic_val,
        "n_params_total": n_params,
        "model_type": "AR(1)",
    }


def fit_cluster_random_walk(
    df_log: pd.DataFrame,
    psi_col: str,
    variable_names: list[str],
    categorical_vars: set[str],
    segment_col: str = "sample_id",
) -> Optional[dict]:
    """
    Fit a random walk with drift model per cluster (first-difference form).

    Model (first-differenced):
        Δln(PSI_t) = δ + ΔX_t β + ε_t,  ε_t ~ N(0, σ²)

    The drift δ captures systematic deterioration trend.
    OLS on first differences — no numerical optimisation required.

    Returns
    -------
    dict with r2 on original scale, bic on differenced scale, drift_rw.
    """
    X_full, y_full, col_names = _build_design_matrix(
        df_log, psi_col, variable_names, variable_names, categorical_vars)

    seg_ids = df_log[segment_col].values.astype(int)
    unique_segs = np.unique(seg_ids)

    dy_list, dX_list = [], []
    for s in unique_segs:
        idx = np.where(seg_ids == s)[0]
        if len(idx) < 2:
            continue
        dy_list.append(y_full[idx[1:]] - y_full[idx[:-1]])
        dX_list.append(X_full[idx[1:], :] - X_full[idx[:-1], :])

    if not dy_list:
        return None

    Dy = np.concatenate(dy_list)
    DX = np.concatenate(dX_list, axis=0)
    n_diff = len(Dy)

    # Differenced intercept column is all-zero → becomes the drift
    DX_noint = DX[:, 1:]  # drop original intercept (all zeros after differencing)
    X_diff = np.column_stack([np.ones(n_diff), DX_noint])
    p_diff = X_diff.shape[1]

    try:
        beta_diff = np.linalg.lstsq(X_diff, Dy, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    drift = float(beta_diff[0])
    beta_rw = beta_diff[1:]
    resid_diff = Dy - X_diff @ beta_diff
    rss_diff = float(np.sum(resid_diff ** 2))
    tss_diff = float(np.sum((Dy - Dy.mean()) ** 2))
    r2_diff = 1.0 - rss_diff / max(tss_diff, 1e-15)
    n_params = p_diff + 1  # include sigma

    # Reconstruct original-scale predictions for interpretable R²
    y_hat = np.full(len(y_full), np.nan)
    for s in unique_segs:
        idx = np.where(seg_ids == s)[0]
        if len(idx) < 1:
            continue
        y_hat[idx[0]] = y_full[idx[0]]
        for t in range(1, len(idx)):
            dX_t = X_full[idx[t], 1:] - X_full[idx[t - 1], 1:]
            y_hat[idx[t]] = y_hat[idx[t - 1]] + drift + dX_t @ beta_rw

    valid = ~np.isnan(y_hat)
    resid_orig = y_full[valid] - y_hat[valid]
    rss_orig = float(np.sum(resid_orig ** 2))
    tss_orig = float(np.sum((y_full[valid] - y_full[valid].mean()) ** 2))
    r2_orig = 1.0 - rss_orig / max(tss_orig, 1e-15)
    n_valid = int(valid.sum())
    bic_val = _bic(n_diff, n_params, rss_diff)

    col_names_rw = ["drift"] + [f"Δ{c}" for c in col_names[1:]]
    coeff_dict = {"drift": drift}
    for i, cn in enumerate(col_names_rw[1:]):
        if i < len(beta_rw):
            coeff_dict[cn] = float(beta_rw[i])

    return {
        "coefficients": np.concatenate([[drift], beta_rw]),
        "coeff_dict": coeff_dict,
        "residuals": y_full - y_hat,
        "rss": rss_orig,
        "tss": tss_orig,
        "r2": r2_orig,
        "adj_r2": 1.0 - (1.0 - r2_orig) * (n_valid - 1) / max(n_valid - n_params, 1),
        "se": np.full(p_diff, np.nan),
        "p_values": np.full(p_diff, np.nan),
        "n": n_diff,
        "n_orig": n_valid,
        "col_names": col_names_rw,
        "variables_used": list(variable_names),
        "drift_rw": drift,
        "sigma_rw": float(sqrt(max(rss_diff / max(n_diff - p_diff, 1), 1e-8))),
        "bic": bic_val,
        "n_params_total": n_params,
        "model_type": "Random_Walk",
    }


def fit_cluster_nur(
    df_log: pd.DataFrame,
    psi_col: str,
    variable_names: list[str],
    categorical_vars: set[str],
    segment_col: str = "sample_id",
    rho_bounds: tuple[float, float] = (0.85, 1.0),
) -> Optional[dict]:
    """
    Fit a near-unit-root (NUR) model per cluster via constrained MLE.

    Model:
        ln(PSI_t) = ρ ln(PSI_{t-1}) + (1-ρ) μ + X_t β + ε_t
    with ρ constrained to rho_bounds (default [0.85, 1.0]).

    When ρ → 1 this reduces to the random walk; ρ < 1 gives mean reversion.

    Returns
    -------
    dict with rho_nur, drift (long-run mean), r2, bic, model_type.
    """
    X, y, col_names = _build_design_matrix(
        df_log, psi_col, variable_names, variable_names, categorical_vars)
    n, p = X.shape
    if n <= p + 3:
        return None

    ols = _fit_lm(X, y, col_names)
    if ols is None:
        return None
    beta0 = ols["coefficients"][:p].copy()
    sigma0 = sqrt(max(ols["rss"] / (n - p), 1e-8))

    seg_ids = df_log[segment_col].values.astype(int)
    unique_segs = np.unique(seg_ids)
    seg_starts = {s: int(np.where(seg_ids == s)[0][0]) for s in unique_segs}
    seg_ends = {s: int(np.where(seg_ids == s)[0][-1]) for s in unique_segs}
    lo, hi = rho_bounds

    def neg_ll(params):
        beta = params[:p]
        # Map unconstrained → [lo, hi]
        rho = lo + (hi - lo) / (1.0 + np.exp(-params[p]))
        log_sig = params[p + 1]
        sigma2 = np.exp(2 * log_sig)
        mu = params[p + 2]

        total = 0.0
        for s in unique_segs:
            i0, i1 = seg_starts[s], seg_ends[s]
            T = i1 - i0 + 1
            for t in range(i0, i1 + 1):
                if t == i0:
                    r = y[t] - mu - X[t] @ beta
                else:
                    r = y[t] - rho * y[t - 1] - (1.0 - rho) * mu - X[t] @ beta
                total += r ** 2 / (2 * sigma2)
            total += 0.5 * (T - 1) * log(2 * pi * sigma2)
            if abs(rho) < 0.999:
                var_init = sigma2 / max(1.0 - rho ** 2, 1e-10)
                r0 = y[i0] - mu - X[i0] @ beta
                total += 0.5 * log(2 * pi * var_init) + 0.5 * r0 ** 2 / var_init
        return total

    x0 = np.zeros(p + 3)
    x0[:p] = beta0
    x0[p] = 0.0
    x0[p + 1] = log(max(sigma0, 1e-4))
    x0[p + 2] = float(np.mean(y))

    try:
        res = _scipy_minimize(neg_ll, x0, method="L-BFGS-B",
                              options={"maxiter": 400, "ftol": 1e-7})
        params = res.x
    except Exception:
        return None

    beta = params[:p]
    rho = float(lo + (hi - lo) / (1.0 + np.exp(-params[p])))
    sigma = float(np.exp(params[p + 1]))
    mu = float(params[p + 2])

    # Reconstruct predictions
    est = np.full(n, np.nan)
    for s in unique_segs:
        i0, i1 = seg_starts[s], seg_ends[s]
        for t in range(i0, i1 + 1):
            if t == i0:
                est[t] = mu + X[t] @ beta
            else:
                est[t] = rho * y[t - 1] + (1.0 - rho) * mu + X[t] @ beta

    valid = ~np.isnan(est)
    resid = y[valid] - est[valid]
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y[valid] - y[valid].mean()) ** 2))
    r2 = 1.0 - rss / max(tss, 1e-15)
    n_params = p + 3
    bic_val = _bic(int(valid.sum()), n_params, rss)

    coeff_dict = {col_names[i]: float(beta[i]) for i in range(min(p, len(col_names)))}
    coeff_dict.update({"_rho": rho, "_mu": mu, "_sigma": sigma})

    return {
        "coefficients": beta,
        "coeff_dict": coeff_dict,
        "residuals": y - est,
        "rss": rss,
        "tss": tss,
        "r2": r2,
        "adj_r2": 1.0 - (1.0 - r2) * (valid.sum() - 1) / max(valid.sum() - n_params, 1),
        "se": np.full(p, np.nan),
        "p_values": np.full(p, np.nan),
        "n": int(valid.sum()),
        "col_names": col_names,
        "variables_used": list(variable_names),
        "rho_nur": rho,
        "drift_nur": mu,
        "sigma_nur": sigma,
        "bic": bic_val,
        "n_params_total": n_params,
        "model_type": "Near_Unit_Root",
    }


# ============================================================
# PavementCLROptimizer — SA-based cluster + variable search
# ============================================================

class PavementCLROptimizer:
    """
    Simultaneous cluster assignment and variable selection for pavement CLR.

    Uses simulated annealing (SA) to minimise total BIC across clusters by
    jointly searching over:
    - Segment-to-cluster assignments
    - Per-cluster variable inclusion masks

    This implements the methodology in Khadka & Paz (2017/2018) extended with
    the joint variable–cluster SA state and BIC objective.

    Parameters
    ----------
    variable_names : list[str]
        Names of all candidate predictor variables.
    categorical_vars : set[str]
        Subset of variable_names that are categorical.
    psi_col : str
        Column name of the (log-transformed) PSI.
    segment_col : str
        Column identifying pavement segments.
    min_observations : int
        Minimum number of observations per cluster.
    level_of_significance : float
        p-value threshold for coefficient retention.
    max_vif : float
        VIF threshold for multicollinearity removal.
    temp_init : float
        Initial SA temperature.
    cooling_rate : float
        Geometric cooling rate α: T ← T × α.
    boltzmann : float
        BIC scaling constant in the acceptance criterion.
    n_changes : int
        Number of segment reassignments per SA neighbour.

    Examples
    --------
    >>> opt = PavementCLROptimizer(
    ...     variable_names=["age", "aadt", "rut_depth", "f_class"],
    ...     categorical_vars={"f_class"},
    ...     psi_col="psi",
    ... )
    >>> result = opt.fit(df_log, n_clusters=4, seed=42)
    >>> print(result["bic"], result["fits"][0]["r2"])
    """

    def __init__(
        self,
        variable_names: list[str],
        categorical_vars: set[str],
        psi_col: str = "psi",
        segment_col: str = "sample_id",
        min_observations: int = 300,
        level_of_significance: float = 0.05,
        max_vif: float = 10.0,
        temp_init: float = 10.0,
        cooling_rate: float = 0.97,
        boltzmann: float = 80.0,
        n_changes: int = 80,
        n_neighbors: int = 3,
    ):
        self.variable_names = list(variable_names)
        self.categorical_vars = set(categorical_vars)
        self.psi_col = psi_col
        self.segment_col = segment_col
        self.min_observations = min_observations
        self.level_of_significance = level_of_significance
        self.max_vif = max_vif
        self.temp_init = temp_init
        self.cooling_rate = cooling_rate
        self.boltzmann = boltzmann
        self.n_changes = n_changes
        self.n_neighbors = n_neighbors

    # ----------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------

    def fit(
        self,
        df_log: pd.DataFrame,
        n_clusters: int,
        seed: int = 42,
        max_iterations: int = 1000,
        verbose: bool = True,
    ) -> dict:
        """
        Run SA to find optimal cluster assignments and variable masks for K clusters.

        Parameters
        ----------
        df_log : DataFrame
            Log-transformed pavement data (ln PSI column already applied).
        n_clusters : int
            Number of clusters K.
        seed : int
            Random seed.
        max_iterations : int
            Maximum SA iterations (overrides temperature-based stopping if hit).
        verbose : bool
            Print progress every 10 iterations.

        Returns
        -------
        dict with keys: clusters (array), fits (list), bic (float),
        rss (float), iterations (int), convergence (list of (iter, bic) tuples).
        """
        rng = np.random.default_rng(seed)
        seg_ids, obs_per_seg = self._build_segment_index(df_log)
        n_seg = len(seg_ids)

        clusters = self._random_init(seg_ids, obs_per_seg, n_clusters, rng)
        fits = self._fit_all_clusters(df_log, clusters, n_clusters)
        best_bic = self._total_bic(fits)

        best_clusters = clusters.copy()
        best_fits = [f.copy() if f else None for f in fits]
        convergence = [(0, best_bic)]
        temp = self.temp_init
        iteration = 0

        while temp > 1e-15 and iteration < max_iterations:
            for _ in range(self.n_neighbors):
                nbr_c = self._perturb_clusters(clusters, seg_ids, obs_per_seg,
                                               n_clusters, rng)
                nbr_fits = self._fit_all_clusters(df_log, nbr_c, n_clusters)
                if any(f is None for f in nbr_fits):
                    continue
                nbr_bic = self._total_bic(nbr_fits)
                diff = nbr_bic - best_bic
                if diff < 0 or rng.random() < exp(-abs(diff / self.boltzmann) / max(temp, 1e-30)):
                    clusters = nbr_c
                    fits = nbr_fits
                    if nbr_bic < best_bic:
                        best_bic = nbr_bic
                        best_clusters = nbr_c.copy()
                        best_fits = [f.copy() if f else None for f in nbr_fits]

            temp *= self.cooling_rate
            iteration += 1
            convergence.append((iteration, best_bic))
            if verbose and iteration % 10 == 0:
                print(f"    SA iter {iteration:4d} | T={temp:.2e} | BIC={best_bic:.2f}")

        total_rss = sum(f["rss"] for f in best_fits if f is not None)
        return {
            "clusters": best_clusters,
            "fits": best_fits,
            "bic": best_bic,
            "rss": total_rss,
            "iterations": iteration,
            "convergence": convergence,
            "n_clusters": n_clusters,
        }

    def search_k(
        self,
        df_log: pd.DataFrame,
        k_min: int = 2,
        k_max: int = 6,
        seed: int = 42,
        max_iterations: int = 500,
        verbose: bool = True,
    ) -> dict:
        """
        Run `fit` for K = k_min, ..., k_max and select optimal K by BIC.

        Returns dict: best_K (int), per_K (dict of K → result), bic_by_K (dict).
        """
        per_K = {}
        bic_by_K = {}
        for k in range(k_min, k_max + 1):
            if verbose:
                print(f"\n  K = {k}")
            result = self.fit(df_log, k, seed=seed + k,
                              max_iterations=max_iterations, verbose=verbose)
            per_K[k] = result
            bic_by_K[k] = result["bic"]

        best_K = min(bic_by_K, key=bic_by_K.get)
        return {"best_K": best_K, "per_K": per_K, "bic_by_K": bic_by_K,
                "best_result": per_K[best_K]}

    def predict(
        self,
        df_orig: pd.DataFrame,
        fits: list[dict],
        clusters: np.ndarray,
    ) -> np.ndarray:
        """
        Predict ln(PSI) for each row using cluster models.

        Parameters
        ----------
        df_orig : DataFrame
            Original (untransformed) data. Log transform is applied inside.
        fits : list[dict]
            Per-cluster model fits from `fit`.
        clusters : ndarray
            Cluster assignment array, indexed by segment_id.

        Returns
        -------
        ndarray of predicted ln(PSI) values (NaN for unassigned observations).
        """
        est = np.full(len(df_orig), np.nan)
        for i, row in enumerate(df_orig.itertuples(index=False)):
            sid = int(getattr(row, self.segment_col))
            if sid >= len(clusters) or clusters[sid] == 0:
                continue
            cid = clusters[sid]
            fit = fits[cid - 1]
            if fit is None:
                continue
            coeffs = fit["coeff_dict"]
            val = coeffs.get("Intercept", 0.0)
            for vname in self.variable_names:
                raw = float(getattr(row, vname, 0.0))
                if vname in self.categorical_vars:
                    val += coeffs.get(f"{vname}_{int(raw)}", 0.0)
                else:
                    beta = coeffs.get(vname, 0.0)
                    if beta != 0.0 and raw > 0.0:
                        val += beta * log(raw)
            est[i] = val
        return est

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _build_segment_index(self, df: pd.DataFrame):
        seg_ids = df[self.segment_col].unique()
        obs_per_seg = {int(s): int((df[self.segment_col] == s).sum()) for s in seg_ids}
        return [int(s) for s in seg_ids], obs_per_seg

    def _random_init(self, seg_ids, obs_per_seg, n_cl, rng):
        max_id = max(seg_ids) + 1
        clusters = np.zeros(max_id, dtype=int)
        for _ in range(100):
            for s in seg_ids:
                clusters[s] = int(rng.integers(1, n_cl + 1))
            if self._valid(clusters, seg_ids, obs_per_seg, n_cl):
                return clusters
        # Fallback: round-robin
        for i, s in enumerate(seg_ids):
            clusters[s] = (i % n_cl) + 1
        return clusters

    def _valid(self, clusters, seg_ids, obs_per_seg, n_cl):
        for c in range(1, n_cl + 1):
            total = sum(obs_per_seg[s] for s in seg_ids if clusters[s] == c)
            if total < self.min_observations:
                return False
        return True

    def _perturb_clusters(self, clusters, seg_ids, obs_per_seg, n_cl, rng):
        for _ in range(50):
            new_c = clusters.copy()
            picks = rng.choice(seg_ids, size=min(self.n_changes, len(seg_ids)),
                               replace=False)
            for s in picks:
                old = new_c[s]
                if n_cl > 1:
                    cand = int(rng.integers(1, n_cl + 1))
                    while cand == old:
                        cand = int(rng.integers(1, n_cl + 1))
                    new_c[s] = cand
            if self._valid(new_c, seg_ids, obs_per_seg, n_cl):
                return new_c
        return clusters.copy()

    def _fit_cluster(self, df_log: pd.DataFrame, cid: int, clusters: np.ndarray):
        segs = set(np.where(clusters == cid)[0])
        dfc = df_log[df_log[self.segment_col].isin(segs)]
        if len(dfc) < self.min_observations:
            return None
        return fit_cluster_ols(dfc, self.psi_col, self.variable_names,
                               self.categorical_vars,
                               self.level_of_significance, self.max_vif)

    def _fit_all_clusters(self, df_log, clusters, n_cl):
        return [self._fit_cluster(df_log, ci, clusters) for ci in range(1, n_cl + 1)]

    def _total_bic(self, fits):
        total = 0.0
        for f in fits:
            if f is None:
                return np.inf
            total += f.get("bic", np.inf)
        return total


# ============================================================
# PavementTemporalComparison — OLS / AR1 / RW / NUR comparison
# ============================================================

class PavementTemporalComparison:
    """
    Compare four temporal error models within a fixed CLR cluster structure.

    Models: OLS (i.i.d.), AR(1), Random Walk with drift, Near-Unit-Root.

    Usage
    -----
    >>> cmp = PavementTemporalComparison(variable_names, categorical_vars)
    >>> df_results = cmp.compare(df_log, clusters, n_clusters=4)
    >>> print(df_results[["model", "r2", "bic", "rho_ar1", "drift_rw", "rho_nur"]])
    """

    def __init__(
        self,
        variable_names: list[str],
        categorical_vars: set[str],
        psi_col: str = "psi",
        segment_col: str = "sample_id",
    ):
        self.variable_names = list(variable_names)
        self.categorical_vars = set(categorical_vars)
        self.psi_col = psi_col
        self.segment_col = segment_col

    def compare(
        self,
        df_log: pd.DataFrame,
        clusters: np.ndarray,
        n_clusters: int,
    ) -> pd.DataFrame:
        """
        Fit all four temporal models for each cluster and return a comparison DataFrame.

        Parameters
        ----------
        df_log : DataFrame
            Log-transformed pavement data.
        clusters : ndarray
            Cluster assignment array (from PavementCLROptimizer.fit).
        n_clusters : int
            Number of clusters K.

        Returns
        -------
        DataFrame with columns: cluster, n_obs, model, r2, adj_r2, bic,
        rho_ar1, drift_rw, rho_nur, sigma.
        """
        rows = []
        for ci in range(1, n_clusters + 1):
            segs = set(np.where(clusters == ci)[0])
            dfc = df_log[df_log[self.segment_col].isin(segs)]
            n_obs = len(dfc)

            models = {
                "OLS": fit_cluster_ols(
                    dfc, self.psi_col, self.variable_names,
                    self.categorical_vars),
                "AR(1)": fit_cluster_ar1(
                    dfc, self.psi_col, self.variable_names,
                    self.categorical_vars, self.segment_col),
                "Random_Walk": fit_cluster_random_walk(
                    dfc, self.psi_col, self.variable_names,
                    self.categorical_vars, self.segment_col),
                "Near_Unit_Root": fit_cluster_nur(
                    dfc, self.psi_col, self.variable_names,
                    self.categorical_vars, self.segment_col),
            }

            for mname, mf in models.items():
                row = {"cluster": ci, "n_obs": n_obs, "model": mname}
                if mf is None:
                    for k in ["r2", "adj_r2", "bic", "rho_ar1",
                              "drift_rw", "rho_nur", "sigma"]:
                        row[k] = np.nan
                else:
                    row["r2"] = mf["r2"]
                    row["adj_r2"] = mf["adj_r2"]
                    row["bic"] = mf.get("bic", np.nan)
                    row["rho_ar1"] = mf.get("rho_ar1", np.nan)
                    row["drift_rw"] = mf.get("drift_rw", np.nan)
                    row["rho_nur"] = mf.get("rho_nur", np.nan)
                    row["sigma"] = mf.get("sigma_ar1",
                                  mf.get("sigma_rw",
                                  mf.get("sigma_nur", np.nan)))
                rows.append(row)

        return pd.DataFrame(rows)

    def summary(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate comparison results by model (mean across clusters)."""
        numeric_cols = ["r2", "adj_r2", "bic", "rho_ar1", "drift_rw", "rho_nur", "sigma"]
        return df_results.groupby("model")[numeric_cols].mean().reset_index()
