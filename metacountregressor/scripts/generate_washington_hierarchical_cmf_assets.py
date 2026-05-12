from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ── JAX NB2 engine (the package's own implementation) ─────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGS as _LBFGS
    from main_hpc_lc_patch import (
        mixed_model_loglik as _jax_loglik,
        ModelSpec as _ModelSpec,
    )
    jax.config.update("jax_enable_x64", True)
    _JAX_OK = True
except Exception:
    _JAX_OK = False


@dataclass
class FittedModel:
    family: str
    upper_vars: list[str]
    lower_vars: list[str]
    result: Any
    aadt_col: str
    offset_col: str | None


# ── Human-readable variable labels ────────────────────────────────────────────
VARIABLE_LABELS: dict[str, str] = {
    # ── Response / exposure ──────────────────────────────────────────────
    "FREQ":     "Crash Count",
    "LENGTH":   "Segment Length (mi)",
    "AADT":     "Annual Average Daily Traffic (veh/day)",
    "LNAADT":   "Log of AADT",
    "OFFSET":   "Log-Exposure Offset: log(Segment Length)",
    # ── Lane geometry ────────────────────────────────────────────────────
    "LANES":    "Total Lanes, Both Directions (lanes)",
    "INCLANES": "Lanes, Increasing Direction (lanes)",
    "DECLANES": "Lanes, Decreasing Direction (lanes)",
    "WIDTH":    "Lane Width (ft)",
    # ── Median / shoulder ────────────────────────────────────────────────
    "MIMEDSH":  "Minimum Median Shoulder Width (ft)",
    "MXMEDSH":  "Maximum Median Shoulder Width (ft)",
    "MEDWIDTH": "Median Width (ft)",
    # ── Speed / classification ───────────────────────────────────────────
    "SPEED":    "Posted Speed Limit (mph)",
    "URB":      "Urban/Rural Indicator (1=Urban, 0=Rural)",
    "FC":       "Functional Classification (1=Interstate to 6=Local)",
    # ── Traffic composition ──────────────────────────────────────────────
    "SINGLE":   "Single-Unit Trucks, % of AADT",
    "DOUBLE":   "Double-Unit Trucks, % of AADT",
    "TRAIN":    "Road Trains / Multi-Unit Trucks, % of AADT",
    "PEAKHR":   "Peak Hour Factor",
    # ── Vertical alignment ───────────────────────────────────────────────
    "GRADEBR":  "Number of Grade Breaks (count)",
    "MIGRADE":  "Minimum Vertical Grade (%)",
    "MXGRADE":  "Maximum Vertical Grade (%)",
    "MXGRDIFF": "Max-Min Grade Difference (%) - range of grades on segment",
    "SLOPE":    "Average Longitudinal Slope (%)",
    "INTECHAG": "Number of Grade-Change Points (count)",
    "GBRPM":    "Grade-Break Rate per Mile",
    # ── Horizontal alignment ─────────────────────────────────────────────
    "TANGENT":  "Proportion of Segment That Is Tangent (0 to 1)",
    "CURVES":   "Number of Horizontal Curves (count)",
    "MINRAD":   "Minimum Curve Radius (ft) - smaller = sharper bend",
    "CPM":      "Horizontal Curve Density (curves per mile)",
    # ── Roadside / access ────────────────────────────────────────────────
    "ACCESS":   "Access-Point Density (driveways + intersections per mile)",
    "FRICTION": "Pavement Friction, Skid Number (higher = more grip)",
    "ADTLANE":  "ADT per Lane (1,000 veh/day per lane)",
    # ── Climate / weather ────────────────────────────────────────────────
    "AVEPRE":   "Average Annual Precipitation (in/yr)",
    "AVESNOW":  "Average Annual Snowfall (in/yr)",
    "LOWPRE":   "Low-Precipitation Days per Year (precip < 1.5 in/month)",
    "HISNOW":   "High-Snow Days per Year (snowfall > 1 in/day)",
    # ── Derived / auxiliary ──────────────────────────────────────────────
    "EXPOSE":   "Exposure Index",
    "INTPM":    "Intersection Density (intersections per mile)",
}

# Tooltip descriptions shown in the dashboard on hover
VARIABLE_DESCRIPTIONS: dict[str, str] = {
    "LANES":    "Total number of lanes in both directions. More lanes = greater road capacity and less per-lane traffic demand.",
    "WIDTH":    "Width of each travel lane in feet. Wider lanes generally reduce sideswipe and run-off-road crashes.",
    "SPEED":    "Posted speed limit in mph. Higher-speed roads are typically built to higher geometric standards.",
    "CURVES":   "Count of horizontal curves on the segment. More curves = more steering demand and crash exposure.",
    "MXGRDIFF": "Difference between the steepest and flattest grade on the segment (%). Larger values signal complex vertical alignment.",
    "MIMEDSH":  "Minimum median shoulder width in feet. Narrower shoulders leave less recovery space for errant vehicles.",
    "MXMEDSH":  "Maximum median shoulder width in feet. Wide medians separate opposing traffic flows.",
    "HISNOW":   "Days per year with snowfall exceeding 1 inch. Captures severe winter conditions (derived from AVESNOW > 1).",
    "AVESNOW":  "Average annual snowfall in inches. Indicates typical winter severity for the segment.",
    "LOWPRE":   "Days per year with precipitation below 1.5 inches per month. Captures extended dry periods.",
    "AVEPRE":   "Average annual precipitation in inches. Reflects overall wetness and pavement drainage demands.",
    "FRICTION": "Skid number from friction testing. Higher values mean better pavement grip and shorter wet-road stopping distances.",
    "ACCESS":   "Number of driveways and intersections per mile. High-access roads have more conflict points.",
    "SLOPE":    "Average longitudinal slope (%). Steeper grades increase braking demands and truck-speed differentials.",
    "MINRAD":   "Smallest horizontal curve radius on the segment in feet. Tighter curves require lower safe operating speeds.",
    "FC":       "FHWA functional class: 1=Interstate, 2=Principal Arterial, 3=Minor Arterial, 4=Major Collector, 5=Minor Collector, 6=Local.",
    "URB":      "1 if the segment is in an urbanized area, 0 if rural. Urban and rural roads have different crash patterns.",
    "MIGRADE":  "Minimum vertical grade (%) on the segment. Flat sections typically have lower crash rates.",
    "MXGRADE":  "Maximum vertical grade (%) on the segment. Steep grades affect stopping and sight distances.",
    "CPM":      "Number of horizontal curves per mile. High-curvature roads demand sustained driver attention.",
    "INTPM":    "Number of intersections per mile. Each intersection is a potential crash conflict point.",
    "MEDWIDTH": "Width of the median in feet. Wider medians improve separation of opposing traffic.",
}

FC_LABELS = {
    1: "Principal Arterial — Interstate",
    2: "Principal Arterial — Other",
    3: "Minor Arterial",
    4: "Major Collector",
    5: "Minor Collector",
    6: "Local Road",
}


def _label(var: str) -> str:
    """Return the human-readable label for a variable name."""
    raw = var.replace("_Z", "").replace("_X_logaadt", "").replace("_inter_", "")
    return VARIABLE_LABELS.get(raw, raw)


def _to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(row[c]) for c in cols) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, divider] + rows)


def _is_binary(series: pd.Series) -> bool:
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        return False
    unique = set(np.unique(values).tolist())
    return unique.issubset({0.0, 1.0})


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    out = [item.strip() for item in value.split(",")]
    return [item for item in out if item]


def _prepare_washington_df(df: pd.DataFrame, aadt_col: str, y_col: str) -> pd.DataFrame:
    out = df.copy()

    if "rumble_install_year" in out.columns:
        out["has_rumble"] = pd.to_numeric(out["rumble_install_year"], errors="coerce").notna().astype(int)

    # Always recompute OFFSET as log(LENGTH) — the correct log-exposure offset
    # for segment-level crash-frequency models.  The raw OFFSET column in this
    # dataset is log(AADT)/LENGTH, a traffic-intensity metric that is not a
    # valid log-exposure offset and causes exploding predictions if used as-is.
    length_col = next((c for c in ["LENGTH", "segment_length"] if c in out.columns), None)
    if length_col is not None:
        exposure = pd.to_numeric(out[length_col], errors="coerce")
        exposure = np.clip(exposure.to_numpy(dtype=float), 1e-9, None)
        out["OFFSET"] = np.log(exposure)

    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out[aadt_col] = pd.to_numeric(out[aadt_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[y_col, aadt_col])
    out = out[out[aadt_col] > 0].copy()

    # Composite LANES = INCLANES + DECLANES (total lane count both directions)
    if "INCLANES" in out.columns and "DECLANES" in out.columns:
        inc = pd.to_numeric(out["INCLANES"], errors="coerce").fillna(0)
        dec = pd.to_numeric(out["DECLANES"], errors="coerce").fillna(0)
        out["LANES"] = (inc + dec).clip(lower=1)

    out = out.reset_index(drop=True)
    out["OBS_ID"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def _split_indices(n: int, train_frac: float, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _build_scaler_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        mu = float(np.nanmean(values.to_numpy(dtype=float)))
        sd = float(np.nanstd(values.to_numpy(dtype=float), ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        stats[col] = (mu, sd)
    return stats


def _apply_standardization(df: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> pd.DataFrame:
    out = df.copy()
    for col, (mu, sd) in stats.items():
        out[f"{col}_Z"] = (pd.to_numeric(out[col], errors="coerce") - mu) / sd
    return out


def _safe_predict(result: Any, exog: pd.DataFrame, offset: np.ndarray | None) -> np.ndarray:
    if offset is None:
        pred = result.predict(exog)
    else:
        pred = result.predict(exog, offset=offset)
    arr = np.asarray(pred, dtype=float).reshape(-1)
    return np.clip(arr, 1e-9, None)


def _poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    mu = np.clip(np.asarray(y_pred, dtype=float).reshape(-1), 1e-12, None)
    y_safe = np.clip(y, 1e-12, None)
    terms = np.where(y > 0, y * np.log(y_safe / mu) - (y - mu), -(-mu))
    return float(2.0 * np.sum(terms))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(y.size, p.size)
    y = y[:n]
    p = p[:n]
    p = np.clip(p, 1e-9, None)

    residual = p - y
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    corr = float(np.corrcoef(y, p)[0, 1]) if n > 1 else float("nan")

    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    return {
        "n": int(n),
        "obs_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(p)),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "corr": corr,
        "r2": r2,
        "poisson_dev": _poisson_deviance(y, p),
    }


# ─────────────────────────────────────────────────────────────────────────────
# JAX NB2 fitter — wraps the package's own mixed_model_loglik.
#
# Uses _design_matrix as Xf directly so the parameter names stay identical
# to the statsmodels convention ("const", "log_aadt", "upper::VAR", ...).
# _JAXResult exposes .params, .bse, .aic, .bic, .predict() so all downstream
# functions (coefficient_report, aadt_elasticity, dashboard payload, …) work
# unchanged.
# ─────────────────────────────────────────────────────────────────────────────

class _JAXResult:
    """Statsmodels-compatible wrapper around a JAX NB2 fit."""

    def __init__(
        self,
        params_np: np.ndarray,
        col_names: list[str],
        ll: float,
        n: int,
        family: str = "nb",
    ):
        Kf   = len(col_names)           # fixed params (incl. intercept "const")
        k    = Kf + (1 if family == "nb" else 0)   # +1 for log_alpha
        self._beta   = params_np[:Kf]   # fixed betas (no alpha)
        self._Kf     = Kf
        # Named series matching statsmodels convention
        self.params  = pd.Series(self._beta, index=col_names, dtype=float)
        self.bse     = pd.Series(np.full(Kf, np.nan), index=col_names, dtype=float)
        self.aic     = float(2 * k - 2 * ll)
        self.bic     = float(k * np.log(max(n, 1)) - 2 * ll)

    def predict(self, exog, offset=None):
        X = exog.to_numpy(dtype=float) if hasattr(exog, "to_numpy") else np.asarray(exog, dtype=float)
        eta = X @ self._beta
        if offset is not None:
            eta = eta + np.asarray(offset, dtype=float)
        return np.exp(eta)


def _build_jax_data(X_np: np.ndarray, y_np: np.ndarray, offset_np: np.ndarray | None) -> dict:
    """Minimal data dict for a fixed-only count model (no random effects)."""
    N, Kf = X_np.shape
    off   = offset_np[:, None, None] if offset_np is not None else np.zeros((N, 1, 1))
    return {
        "Xf":        jnp.array(X_np[:, None, :]),
        "Xr_ind":    jnp.zeros((N, 1, 0)),
        "Xr_cor":    jnp.zeros((N, 1, 0)),
        "Xg":        jnp.zeros((N, 1, 0)),
        "Xh":        jnp.zeros((N, 1, 0)),
        "Xzi":       jnp.zeros((N, 1, 0)),
        "Xmem":      jnp.zeros((N, 1, 0)),
        "y":         jnp.array(y_np[:, None, None]),
        "mask":      jnp.ones((N, 1, 1)),
        "offset":    jnp.array(off.astype(float)),
        "draws_ind": jnp.zeros((N, 0, 1)),
        "draws_cor": jnp.zeros((N, 0, 1)),
        "draws_g":   jnp.zeros((N, 0, 1)),
        "group_ids": jnp.zeros(N, dtype=int),
    }


def _make_spec(Kf: int, col_names: list[str], family: str) -> "_ModelSpec":
    return _ModelSpec(
        Kf=Kf, Kr_ind=0, Kr_cor=0, Kg=0, Kh=0, Kzi=0,
        model=family, zero_inflated=False,
        fixed_names=tuple(col_names),
        zi_names=(), random_ind_names=(), random_cor_names=(),
        grouped_names=(), hetro_names=(),
        random_ind_dists=(), random_cor_dists=(), grouped_dists=(),
        latent_classes=1, membership_names=(), K_membership=0,
    )


def _jax_fit(X_df: pd.DataFrame, y_np: np.ndarray,
             offset_np: np.ndarray | None, family: str,
             n_restarts: int = 1) -> _JAXResult | None:
    """
    Fit NB2 or Poisson via the package's own JAX LBFGS engine.

    The objective is evaluated by mixed_model_loglik which is @jit-cached
    on (spec, indivi).  Different Kf values (different variable counts) each
    get their own compiled version — subsequent calls with the same Kf reuse
    the compiled code, so the search gets faster after the first few unique
    column counts.
    """
    if not _JAX_OK:
        return None
    try:
        X_np = X_df.to_numpy(dtype=float)
        N, Kf = X_np.shape
        data  = _build_jax_data(X_np, y_np, offset_np)
        # Use anonymous names so JAX reuses its compiled trace for all specs
        # with the same Kf — the actual column names are stored in _JAXResult.
        spec  = _make_spec(Kf, [f"_x{i}" for i in range(Kf)], family)
        K     = Kf + (1 if family == "nb" else 0)

        # Warm start: intercept ≈ log(mean(y)), log_alpha ≈ 0
        p0    = np.zeros(K)
        p0[0] = float(np.log(np.clip(float(np.mean(y_np)), 1e-8, None)))

        # Use functools.partial so JAX sees a stable callable and can reuse
        # its compiled trace for calls with the same Kf (same spec hash).
        from functools import partial as _partial
        obj  = _partial(_jax_loglik, data=data, spec=spec, indivi=False)
        best = _LBFGS(fun=obj, maxiter=300).run(jnp.array(p0))

        if n_restarts > 1:
            rng = np.random.default_rng(1)
            for _ in range(n_restarts - 1):
                p_try = jnp.array(p0 + rng.normal(0, 0.1, K))
                cand  = _LBFGS(fun=obj, maxiter=300).run(p_try)
                if float(cand.state.value) < float(best.state.value):
                    best = cand

        ll = float(-best.state.value)
        if not np.isfinite(ll):
            return None
        return _JAXResult(np.array(best.params), list(X_df.columns), ll, N, family)
    except Exception:
        return None


def _design_matrix(df: pd.DataFrame, aadt_col: str, upper_vars: list[str], lower_vars: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(index=df.index)
    mat["const"] = 1.0
    log_aadt = np.log(np.clip(pd.to_numeric(df[aadt_col], errors="coerce").to_numpy(dtype=float), 1e-12, None))
    mat["log_aadt"] = log_aadt

    for var in upper_vars:
        mat[f"upper::{var}"] = pd.to_numeric(df[var], errors="coerce").to_numpy(dtype=float)

    for var in lower_vars:
        x = pd.to_numeric(df[var], errors="coerce").to_numpy(dtype=float)
        mat[f"lower::{var}*log_aadt"] = x * log_aadt

    return mat


def _fit_model(
    df_train: pd.DataFrame,
    aadt_col: str,
    y_col: str,
    upper_vars: list[str],
    lower_vars: list[str],
    family: str,
    offset_col: str | None,
) -> FittedModel | None:
    X_train = _design_matrix(df_train, aadt_col, upper_vars, lower_vars)
    y_train = pd.to_numeric(df_train[y_col], errors="coerce").to_numpy(dtype=float)

    offset = None
    if offset_col is not None and offset_col in df_train.columns:
        offset = pd.to_numeric(df_train[offset_col], errors="coerce").to_numpy(dtype=float)

    result = _jax_fit(X_train, y_train, offset, family)
    if result is None:
        return None

    return FittedModel(
        family=family,
        upper_vars=list(upper_vars),
        lower_vars=list(lower_vars),
        result=result,
        aadt_col=aadt_col,
        offset_col=offset_col,
    )


def _predict(df: pd.DataFrame, fitted: FittedModel) -> np.ndarray:
    X = _design_matrix(df, fitted.aadt_col, fitted.upper_vars, fitted.lower_vars)
    offset = None
    if fitted.offset_col is not None and fitted.offset_col in df.columns:
        offset = pd.to_numeric(df[fitted.offset_col], errors="coerce").to_numpy(dtype=float)
    return _safe_predict(fitted.result, X, offset)


def _aadt_elasticity(df: pd.DataFrame, fitted: FittedModel) -> np.ndarray:
    params = pd.Series(fitted.result.params)
    base = float(params.get("log_aadt", 0.0))
    elasticity = np.full(len(df), base, dtype=float)

    for var in fitted.lower_vars:
        term_name = f"lower::{var}*log_aadt"
        coef = float(params.get(term_name, 0.0))
        x = pd.to_numeric(df[var], errors="coerce").to_numpy(dtype=float)
        elasticity += coef * x

    return np.asarray(elasticity, dtype=float)


def _elasticity_stats(elasticity: np.ndarray) -> dict[str, float]:
    e = np.asarray(elasticity, dtype=float)
    return {
        "aadt_elasticity_min": float(np.min(e)),
        "aadt_elasticity_p10": float(np.quantile(e, 0.10)),
        "aadt_elasticity_median": float(np.quantile(e, 0.50)),
        "aadt_elasticity_p90": float(np.quantile(e, 0.90)),
        "aadt_elasticity_share_positive": float(np.mean(e > 0.0)),
    }


def _random_search(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    aadt_col: str,
    y_col: str,
    upper_candidates: list[str],
    lower_candidates: list[str],
    families: list[str],        # e.g. ["nb"], ["poisson"], or ["nb","poisson"]
    offset_col: str | None,
    search_iter: int,
    max_upper_terms: int,
    max_lower_terms: int,
    seed: int,
    enforce_aadt_increase: bool,
    min_aadt_elasticity: float,
    allow_nonmonotonic_fallback: bool,
    top_k: int = 5,             # keep top-K models for random-params sweep
) -> tuple[FittedModel, pd.DataFrame, list[FittedModel]]:
    rng = np.random.default_rng(seed)
    tested: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()
    history_rows: list[dict[str, Any]] = []

    best_fit: FittedModel | None = None
    best_score = float("inf")
    best_fit_any: FittedModel | None = None
    best_score_any = float("inf")

    # Priority queue (min-heap) for top-K: (bic, fit)
    import heapq as _hq
    top_k_heap: list[tuple[float, int, FittedModel]] = []  # (bic, counter, fit)
    _counter = [0]   # mutable counter for heap tie-breaking

    # Seeds: use at least MIN_UPPER/MIN_LOWER variables to match main search
    baseline_upper = upper_candidates[:2] if len(upper_candidates) >= 2 else upper_candidates
    baseline_lower = lower_candidates[:1] if lower_candidates else []

    seeds = [
        (tuple(baseline_upper), tuple(baseline_lower)),
        (tuple(upper_candidates[:2]), tuple(lower_candidates[:1])),  # different pair
    ]

    def _eval(fit: FittedModel, y_val_arr: np.ndarray) -> None:
        """Score fit, append to history, update bests and top-K heap."""
        pred_val   = _predict(df_val, fit)
        score      = _poisson_deviance(y_val_arr, pred_val)
        val_rmse   = _metrics(y_val_arr, pred_val)["rmse"]
        bic        = float(getattr(fit.result, "bic", np.nan))
        aic        = float(getattr(fit.result, "aic", np.nan))
        elasticity = _aadt_elasticity(df_val, fit)
        es         = _elasticity_stats(elasticity)
        mono_ok    = bool(es["aadt_elasticity_min"] > float(min_aadt_elasticity))
        history_rows.append({
            "Iteration": len(history_rows) + 1,
            "Family":    fit.family,
            "Phase":     "random",
            "Upper Vars": ", ".join(fit.upper_vars) if fit.upper_vars else "(none)",
            "Lower Vars": ", ".join(fit.lower_vars) if fit.lower_vars else "(none)",
            "Val Poisson Dev": score, "Val RMSE": val_rmse,
            "AIC": aic, "BIC": bic,
            "AADT elasticity min":            es["aadt_elasticity_min"],
            "AADT elasticity p10":            es["aadt_elasticity_p10"],
            "AADT elasticity median":         es["aadt_elasticity_median"],
            "AADT elasticity p90":            es["aadt_elasticity_p90"],
            "AADT elasticity share positive": es["aadt_elasticity_share_positive"],
            "Monotonic AADT OK": "yes" if mono_ok else "no",
        })
        nonlocal best_fit, best_score, best_fit_any, best_score_any
        if bic < best_score_any:
            best_score_any = bic
            best_fit_any   = fit
        if (not enforce_aadt_increase or mono_ok) and bic < best_score:
            best_score = bic
            best_fit   = fit
        # top-K heap maintenance (max-heap by negating bic for min-heap)
        _counter[0] += 1
        if np.isfinite(bic) and (not enforce_aadt_increase or mono_ok):
            if len(top_k_heap) < top_k:
                _hq.heappush(top_k_heap, (-bic, _counter[0], fit))
            elif -bic > top_k_heap[0][0]:
                _hq.heapreplace(top_k_heap, (-bic, _counter[0], fit))

    y_val_np = pd.to_numeric(df_val[y_col], errors="coerce").to_numpy(dtype=float)

    for init_upper, init_lower in seeds:
        for fam in families:
            key = (tuple(sorted(init_upper)), tuple(sorted(init_lower)), fam)
            if key in tested:
                continue
            tested.add(key)
            fit = _fit_model(df_train, aadt_col=aadt_col, y_col=y_col,
                             upper_vars=list(init_upper), lower_vars=list(init_lower),
                             family=fam, offset_col=offset_col)
            if fit is None:
                continue
            _eval(fit, y_val_np)

    # ── Random search — BIC-primary, multi-family ─────────────────────────
    # Min complexity: always at least 2 upper + 1 lower variable.
    MIN_UPPER = 2
    MIN_LOWER = 1

    for _ in range(search_iter):
        k_upper = int(rng.integers(MIN_UPPER, max_upper_terms + 1))
        k_lower = int(rng.integers(MIN_LOWER, max_lower_terms + 1))
        fam     = families[int(rng.integers(0, len(families)))]  # pick family randomly

        pick_upper: list[str] = []
        pick_lower: list[str] = []
        if upper_candidates:
            pick_upper = sorted(rng.choice(upper_candidates,
                                           size=min(k_upper, len(upper_candidates)),
                                           replace=False).tolist())
        if lower_candidates:
            pick_lower = sorted(rng.choice(lower_candidates,
                                           size=min(k_lower, len(lower_candidates)),
                                           replace=False).tolist())

        key = (tuple(pick_upper), tuple(pick_lower), fam)
        if key in tested:
            continue
        tested.add(key)

        fit = _fit_model(df_train, aadt_col=aadt_col, y_col=y_col,
                         upper_vars=pick_upper, lower_vars=pick_lower,
                         family=fam, offset_col=offset_col)
        if fit is None:
            continue

        _eval(fit, y_val_np)  # updates bests and top-K heap via nonlocal + closure

    # Hill-climb phase intentionally omitted — pure random search is used
    # to avoid getting trapped in local optima and to keep exploration broad.

    if best_fit is None and best_fit_any is None:
        raise RuntimeError("Search failed to fit any candidate model.")
    if best_fit is None:
        if enforce_aadt_increase and not allow_nonmonotonic_fallback:
            raise RuntimeError(
                "No candidate met the required positive AADT elasticity threshold. "
                "Increase --search-iter, adjust candidate variables, or run with "
                "--allow-nonmonotonic-fallback (or --no-enforce-aadt-increase) to inspect unconstrained fits."
            )
        best_fit = best_fit_any

    # Extract top-K list (sorted best BIC first)
    top_k_list = [fit for (_, _, fit) in sorted(top_k_heap, key=lambda x: x[0])]

    history_df        = pd.DataFrame(history_rows)
    history_df_sorted = history_df.sort_values("Val Poisson Dev").reset_index(drop=True)
    return best_fit, history_df_sorted, history_df, top_k_list


def _coefficient_report(
    fitted: FittedModel,
    scaler_stats: dict[str, tuple[float, float]],
    binary_vars: set[str],
) -> pd.DataFrame:
    params = pd.Series(fitted.result.params)
    std_err = pd.Series(getattr(fitted.result, "bse", np.nan), index=params.index)

    rows: list[dict[str, Any]] = []
    for name, value in params.items():
        if name in {"const", "log_aadt"}:
            source_var = name
            component = "core"
            scale_type = "native"
            coef_orig = float(value)
        elif name.startswith("upper::"):
            source_var = name.replace("upper::", "")
            component = "upper"
            if source_var.endswith("_Z"):
                raw = source_var[:-2]
                sd = scaler_stats.get(raw, (0.0, 1.0))[1]
                coef_orig = float(value) / sd
                scale_type = "standardized-continuous"
                source_var = raw
            else:
                coef_orig = float(value)
                scale_type = "binary" if source_var in binary_vars else "native"
        elif name.startswith("lower::") and name.endswith("*log_aadt"):
            core = name.replace("lower::", "").replace("*log_aadt", "")
            source_var = core
            component = "lower"
            if core.endswith("_Z"):
                raw = core[:-2]
                sd = scaler_stats.get(raw, (0.0, 1.0))[1]
                coef_orig = float(value) / sd
                scale_type = "standardized-continuous"
                source_var = raw
            else:
                coef_orig = float(value)
                scale_type = "binary" if core in binary_vars else "native"
        else:
            source_var = name
            component = "other"
            scale_type = "native"
            coef_orig = float(value)

        rows.append(
            {
                "Parameter": name,
                "Component": component,
                "Variable": source_var,
                "Scale Type": scale_type,
                "Estimate (standardized fit scale)": float(value),
                "Std. Error": float(std_err.get(name, np.nan)),
                "Estimate (original variable scale)": float(coef_orig),
            }
        )

    return pd.DataFrame(rows)


def _obs_pred_plot(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    lim = float(max(np.max(y), np.max(p), 1.0)) * 1.05

    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=160)
    ax.scatter(y, p, s=9, alpha=0.28, color="#1f77b4", edgecolors="none")
    ax.plot([0, lim], [0, lim], "--", color="#d62728", linewidth=1.2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed crashes")
    ax.set_ylabel("Predicted crashes")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    table = pd.DataFrame({"Observed": y, "Predicted": p})
    try:
        table["Predicted Bin"] = pd.qcut(table["Predicted"], q=n_bins, duplicates="drop")
    except Exception:
        table["Predicted Bin"] = "all"

    out = (
        table.groupby("Predicted Bin", observed=False)
        .agg(
            N=("Observed", "size"),
            **{
                "Observed Mean": ("Observed", "mean"),
                "Predicted Mean": ("Predicted", "mean"),
            },
        )
        .reset_index()
    )
    out["Observed Mean"] = out["Observed Mean"].astype(float).round(4)
    out["Predicted Mean"] = out["Predicted Mean"].astype(float).round(4)
    return out


def _representative_profile(
    df: pd.DataFrame,
    variables: list[str],
    binary_vars: set[str],
    include_columns: list[str] | None = None,
) -> dict[str, float]:
    profile: dict[str, float] = {}
    ordered_vars = list(dict.fromkeys(list(variables) + list(include_columns or [])))
    for var in ordered_vars:
        if var not in df.columns:
            continue
        series = pd.to_numeric(df[var], errors="coerce")
        if var in binary_vars:
            mode = series.dropna().mode()
            profile[var] = float(mode.iloc[0]) if not mode.empty else 0.0
        else:
            profile[var] = float(series.median())
    return profile


def _scenario_row(base: dict[str, float], feature: str, value: float, aadt_col: str, aadt_value: float) -> dict[str, float]:
    row = dict(base)
    row[feature] = float(value)
    row[aadt_col] = float(aadt_value)
    return row


def _make_single_row_df(row: dict[str, float], all_columns: list[str]) -> pd.DataFrame:
    data = {col: row.get(col, 0.0) for col in all_columns}
    return pd.DataFrame([data])


def _dashboard_payload(
    fitted: FittedModel,
    df_reference_raw: pd.DataFrame,
    all_columns: list[str],
    selected_raw_vars: list[str],
    binary_vars: set[str],
    scaler_stats: dict[str, tuple[float, float]],
    aadt_col: str,
) -> dict[str, Any]:
    quant = df_reference_raw[aadt_col].quantile([0.25, 0.5, 0.75]).to_numpy(dtype=float)
    aadt_levels = {
        "Low AADT": float(quant[0]),
        "Median AADT": float(quant[1]),
        "High AADT": float(quant[2]),
    }

    include_columns = [fitted.offset_col] if fitted.offset_col is not None else []
    profile = _representative_profile(df_reference_raw, selected_raw_vars, binary_vars, include_columns=include_columns)
    profile[aadt_col] = float(aadt_levels["Median AADT"])

    variable_payload: dict[str, Any] = {}

    for raw_var in selected_raw_vars:
        is_binary = raw_var in binary_vars
        if is_binary:
            grid = [0.0, 1.0]
        else:
            q10 = float(df_reference_raw[raw_var].quantile(0.10))
            q90 = float(df_reference_raw[raw_var].quantile(0.90))
            if not np.isfinite(q10) or not np.isfinite(q90) or q10 == q90:
                center = float(df_reference_raw[raw_var].median())
                q10 = center - 1.0
                q90 = center + 1.0
            grid = np.linspace(q10, q90, 41).tolist()

        level_payload: dict[str, Any] = {}
        for level_name, level_aadt in aadt_levels.items():
            pred_values: list[float] = []
            for x in grid:
                row = _scenario_row(profile, raw_var, x, aadt_col, level_aadt)
                scenario_raw = _make_single_row_df(row, all_columns)
                scenario_std = _apply_standardization(scenario_raw, scaler_stats)
                pred = float(_predict(scenario_std, fitted)[0])
                pred_values.append(pred)

            base_idx = 0
            if is_binary:
                base_idx = 0
            else:
                base_val = profile.get(raw_var, grid[len(grid) // 2])
                base_idx = int(np.argmin(np.abs(np.asarray(grid, dtype=float) - float(base_val))))

            base_pred = max(pred_values[base_idx], 1e-9)
            cmf_values = [float(v / base_pred) for v in pred_values]

            level_payload[level_name] = {
                "x": [float(v) for v in grid],
                "pred": pred_values,
                "cmf": cmf_values,
                "base_index": int(base_idx),
            }

        variable_payload[raw_var] = {
            "is_binary": is_binary,
            "levels": level_payload,
        }

    return {
        "aadt_col": aadt_col,
        "aadt_levels": aadt_levels,
        "profile": profile,
        "variables": variable_payload,
    }


def _interactive_model_payload(
    fitted: FittedModel,
    df_reference_raw: pd.DataFrame,
    selected_raw_vars: list[str],
    binary_vars: set[str],
    scaler_stats: dict[str, tuple[float, float]],
    aadt_col: str,
) -> dict[str, Any]:
    include_columns = [fitted.offset_col] if fitted.offset_col is not None else []
    profile = _representative_profile(df_reference_raw, selected_raw_vars, binary_vars, include_columns=include_columns)

    aadt_series = pd.to_numeric(df_reference_raw[aadt_col], errors="coerce")
    aadt_q05 = float(aadt_series.quantile(0.05))
    aadt_q50 = float(aadt_series.quantile(0.50))
    aadt_q95 = float(aadt_series.quantile(0.95))
    if not np.isfinite(aadt_q05) or not np.isfinite(aadt_q95) or aadt_q05 >= aadt_q95:
        aadt_q05 = float(max(aadt_series.min(), 1.0))
        aadt_q95 = float(max(aadt_series.max(), aadt_q05 + 1.0))

    variable_specs: dict[str, Any] = {}
    for raw_var in selected_raw_vars:
        series = pd.to_numeric(df_reference_raw[raw_var], errors="coerce")
        default_value = float(profile.get(raw_var, float(series.median())))

        # Compute quantile ticks for slider display
        q10_tick = float(series.quantile(0.10)) if not _is_binary(series) else 0.0
        q50_tick = float(series.quantile(0.50)) if not _is_binary(series) else 0.5
        q90_tick = float(series.quantile(0.90)) if not _is_binary(series) else 1.0

        if raw_var in binary_vars:
            variable_specs[raw_var] = {
                "is_binary": True,
                "min": 0.0,
                "max": 1.0,
                "step": 1.0,
                "default": float(round(default_value)),
                "q10": 0.0,
                "q50": 0.5,
                "q90": 1.0,
                "label": VARIABLE_LABELS.get(raw_var, raw_var),
                "unit": "",
                "description": VARIABLE_DESCRIPTIONS.get(raw_var, ""),
            }
            continue

        q10 = float(series.quantile(0.10))
        q90 = float(series.quantile(0.90))
        if not np.isfinite(q10) or not np.isfinite(q90) or q10 == q90:
            center = float(series.median()) if np.isfinite(series.median()) else default_value
            q10 = center - 1.0
            q90 = center + 1.0
        step = float((q90 - q10) / 100.0)
        if not np.isfinite(step) or step <= 0:
            step = 0.1

        # Determine display unit hint
        unit = ""
        if raw_var == aadt_col or "AADT" in raw_var.upper():
            unit = "veh/day"
        elif raw_var in ("SPEED",):
            unit = "mph"
        elif raw_var in ("LENGTH",):
            unit = "mi"
        elif raw_var in ("WIDTH", "MEDWIDTH"):
            unit = "ft"

        full_label = VARIABLE_LABELS.get(raw_var, raw_var)
        description = VARIABLE_DESCRIPTIONS.get(raw_var, "")
        variable_specs[raw_var] = {
            "is_binary": False,
            "min": float(q10),
            "max": float(q90),
            "step": float(step),
            "default": float(np.clip(default_value, q10, q90)),
            "q10": float(q10_tick),
            "q50": float(q50_tick),
            "q90": float(q90_tick),
            "label": full_label,
            "unit": unit,
            "description": description,
        }

    params = pd.Series(fitted.result.params)
    upper_terms: list[dict[str, Any]] = []
    lower_terms: list[dict[str, Any]] = []
    for name, value in params.items():
        if name in {"const", "log_aadt"}:
            continue
        if name.startswith("upper::"):
            raw_name = name.replace("upper::", "")
            standardized = raw_name.endswith("_Z")
            variable = raw_name[:-2] if standardized else raw_name
            upper_terms.append(
                {
                    "variable": variable,
                    "coefficient": float(value),
                    "standardized": bool(standardized),
                }
            )
        elif name.startswith("lower::") and name.endswith("*log_aadt"):
            raw_name = name.replace("lower::", "").replace("*log_aadt", "")
            standardized = raw_name.endswith("_Z")
            variable = raw_name[:-2] if standardized else raw_name
            lower_terms.append(
                {
                    "variable": variable,
                    "coefficient": float(value),
                    "standardized": bool(standardized),
                }
            )

    scaler_payload = {
        var: {"mean": float(mu), "sd": float(sd)}
        for var, (mu, sd) in scaler_stats.items()
    }

    return {
        "aadt_col": aadt_col,
        "aadt_range": {
            "min": float(max(aadt_q05, 1.0)),
            "max": float(max(aadt_q95, aadt_q05 + 1.0)),
            "default": float(max(aadt_q50, 1.0)),
            "step": float(max((aadt_q95 - aadt_q05) / 120.0, 1.0)),
        },
        "default_profile": profile,
        "variables": variable_specs,
        "binary_variables": sorted(binary_vars),
        "scaler_stats": scaler_payload,
        "coefficients": {
            "const": float(params.get("const", 0.0)),
            "log_aadt": float(params.get("log_aadt", 0.0)),
        },
        "upper_terms": upper_terms,
        "lower_terms": lower_terms,
        "offset_col": fitted.offset_col,
        "offset_value": float(profile.get(fitted.offset_col, 0.0)) if fitted.offset_col else 0.0,
    }


def _save_curve_plot(
    path: Path,
    payload: dict[str, Any],
    variable: str,
    aadt_levels: list[str],
    title: str,
    ylabel: str,
    mode: str,
) -> None:
    var_data = payload["variables"].get(variable)
    if var_data is None:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for idx, level_name in enumerate(aadt_levels):
        level = var_data["levels"][level_name]
        x = np.asarray(level["x"], dtype=float)
        y = np.asarray(level[mode], dtype=float)
        ax.plot(x, y, linewidth=2.2, color=colors[idx % len(colors)], label=level_name)

    ax.set_title(title)
    ax.set_xlabel(variable)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.24)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_binary_bar(path: Path, payload: dict[str, Any], binary_vars: list[str]) -> None:
    rows: list[dict[str, float]] = []
    for var in binary_vars:
        var_payload = payload["variables"].get(var)
        if var_payload is None or not var_payload.get("is_binary", False):
            continue
        level = var_payload["levels"]["Median AADT"]
        cmf = level["cmf"]
        if len(cmf) >= 2:
            rows.append({"Variable": var, "CMF 0->1": float(cmf[1])})

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("CMF 0->1")
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    colors = ["#2ca02c" if value <= 1.0 else "#d62728" for value in df["CMF 0->1"]]
    ax.barh(df["Variable"], df["CMF 0->1"], color=colors)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel("CMF (toggle 0 to 1) at median AADT")
    ax.set_title("Binary variable crash modification effects")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_aadt_curve_plot(
    path: Path,
    fitted: FittedModel,
    df_reference_raw: pd.DataFrame,
    all_columns: list[str],
    selected_raw_vars: list[str],
    binary_vars: set[str],
    scaler_stats: dict[str, tuple[float, float]],
    aadt_col: str,
) -> None:
    include_columns = [fitted.offset_col] if fitted.offset_col is not None else []
    profile = _representative_profile(df_reference_raw, selected_raw_vars, binary_vars, include_columns=include_columns)
    aadt_series = pd.to_numeric(df_reference_raw[aadt_col], errors="coerce")
    q05 = float(aadt_series.quantile(0.05))
    q95 = float(aadt_series.quantile(0.95))
    if not np.isfinite(q05) or not np.isfinite(q95) or q05 >= q95:
        q05 = float(max(aadt_series.min(), 1.0))
        q95 = float(max(aadt_series.max(), q05 + 1.0))

    grid = np.linspace(max(q05, 1.0), max(q95, q05 + 1.0), 61)
    preds: list[float] = []
    for aadt_value in grid:
        row = dict(profile)
        row[aadt_col] = float(aadt_value)
        scenario_raw = _make_single_row_df(row, all_columns)
        scenario_std = _apply_standardization(scenario_raw, scaler_stats)
        preds.append(float(_predict(scenario_std, fitted)[0]))

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    ax.plot(grid, preds, linewidth=2.4, color="#0a6c74")
    ax.set_title("Crash-risk change as AADT changes")
    ax.set_xlabel(aadt_col)
    ax.set_ylabel("Predicted crashes")
    ax.grid(alpha=0.24)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_dashboard_html(path: Path, model_payload: dict[str, Any], dataset_label: str) -> None:
    html = f"""<!doctype html>
<html>
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Washington Hierarchical CMF Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --ink: #152238;
            --teal: #0a6c74;
            --orange: #d96f32;
            --paper: #f8f6f2;
            --panel: #ffffff;
            --line: #d4d8de;
            --muted: #5d6b79;
            --green: #2a7f4f;
            --red: #b93a2b;
        }}
        body {{
            margin: 0;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 0% 0%, #dceff1 0%, transparent 45%),
                radial-gradient(circle at 100% 100%, #f4ddcf 0%, transparent 45%),
                var(--paper);
        }}
        .wrap {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
        .hero {{
            background: linear-gradient(120deg, #0a6c74 0%, #18435a 45%, #8a4e2c 100%);
            color: #fff;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 12px 28px rgba(0,0,0,0.14);
        }}
        .hero h1 {{ margin: 0; font-size: 1.45rem; letter-spacing: 0.2px; }}
        .hero p {{ margin: 6px 0 0; color: rgba(255,255,255,0.94); font-size: 0.95rem; }}
        .grid {{
            margin-top: 16px;
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 14px;
        }}
        .card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 8px 18px rgba(21,34,56,0.08);
        }}
        .control-row {{ display: grid; gap: 6px; margin-bottom: 14px; }}
        .label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--muted);
            font-weight: 700;
        }}
        select {{ width: 100%; padding: 4px 6px; border-radius: 6px; border: 1px solid var(--line); }}
        /* Gradient-track slider */
        input[type="range"] {{
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            border-radius: 4px;
            outline: none;
            cursor: pointer;
            background: linear-gradient(to right, #0a6c74 0%, #d96f32 100%);
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px; height: 16px;
            border-radius: 50%;
            background: var(--teal);
            border: 2px solid #fff;
            box-shadow: 0 1px 4px rgba(0,0,0,0.25);
        }}
        .slider-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 2px;
        }}
        .slider-value {{
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--teal);
        }}
        .cmf-badge {{
            font-size: 0.78rem;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 10px;
            background: #e8f5ee;
            color: var(--green);
        }}
        .cmf-badge.over {{ background: #fdecea; color: var(--red); }}
        .tick-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.68rem;
            color: var(--muted);
            margin-top: 1px;
        }}
        /* Toggle switch for binary */
        .toggle-wrap {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 4px;
        }}
        .toggle-switch {{
            position: relative;
            width: 46px; height: 24px;
        }}
        .toggle-switch input {{ display: none; }}
        .toggle-slider {{
            position: absolute;
            inset: 0;
            background: #ccc;
            border-radius: 24px;
            cursor: pointer;
            transition: 0.25s;
        }}
        .toggle-slider:before {{
            content: "";
            position: absolute;
            left: 3px; top: 3px;
            width: 18px; height: 18px;
            border-radius: 50%;
            background: #fff;
            transition: 0.25s;
        }}
        .toggle-switch input:checked + .toggle-slider {{ background: var(--teal); }}
        .toggle-switch input:checked + .toggle-slider:before {{ transform: translateX(22px); }}
        .toggle-label {{ font-size: 0.82rem; font-weight: 600; }}
        /* Prediction Summary */
        .pred-summary {{
            background: linear-gradient(135deg, #f0fafa 0%, #fff8f3 100%);
            border: 1px solid #c5dfe0;
            border-radius: 10px;
            padding: 10px 12px;
            margin-bottom: 14px;
        }}
        .pred-title {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 700; margin-bottom: 6px; }}
        .pred-number {{ font-size: 2rem; font-weight: 800; color: var(--teal); line-height: 1; }}
        .pred-label {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 8px; }}
        .pred-row {{ display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 4px; }}
        .pred-row .pk {{ color: var(--muted); }}
        .pred-row .pv {{ font-weight: 600; }}
        .pv.good {{ color: var(--green); }}
        .pv.bad {{ color: var(--red); }}
        #chart {{ height: 58vh; }}
        .kpi {{ margin-top: 8px; font-size: 0.88rem; color: var(--ink); line-height: 1.6; }}
        .warn {{
            margin-top: 8px;
            color: #8b3e12;
            font-size: 0.85rem;
            background: #fff1e9;
            border: 1px solid #f1c9b4;
            border-radius: 8px;
            padding: 8px;
        }}
        button#resetButton {{
            margin-top: 10px;
            padding: 6px 14px;
            background: var(--teal);
            color: #fff;
            border: none;
            border-radius: 7px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        button#resetButton:hover {{ background: #085a61; }}
        .controls-scroll {{ max-height: 55vh; overflow-y: auto; padding-right: 4px; }}
        @media (max-width: 960px) {{
            .grid {{ grid-template-columns: 1fr; }}
            #chart {{ height: 50vh; }}
        }}
    </style>
</head>
<body>
<div class="wrap">
    <div class="hero">
        <h1>{dataset_label} Hierarchical CMF Explorer</h1>
        <p>Adjust AADT and road geometry variables to explore how the model changes predicted crash likelihood. CMF badges show the effect relative to the current profile baseline.</p>
    </div>

    <div class="grid">
        <div class="card">
            <div class="control-row">
                <span class="label">Variable</span>
                <select id="varSelect"></select>
            </div>
            <div class="control-row">
                <span class="label">Display</span>
                <select id="modeSelect">
                    <option value="pred">Predicted crashes</option>
                    <option value="cmf">CMF (relative to profile baseline)</option>
                </select>
            </div>

            <!-- Prediction Summary Panel -->
            <div class="pred-summary" id="predSummary">
                <div class="pred-title">Prediction Summary</div>
                <div class="pred-number" id="predNumber">--</div>
                <div class="pred-label">predicted crashes</div>
                <div class="pred-row"><span class="pk">AADT elasticity</span><span class="pv" id="summElast">--</span></div>
                <div class="pred-row"><span class="pk">CMF vs baseline AADT</span><span class="pv" id="summCmf">--</span></div>
                <div class="pred-row"><span class="pk">Upper-level contribution</span><span class="pv" id="summUpper">--</span></div>
                <div class="pred-row"><span class="pk">Lower-level contribution</span><span class="pv" id="summLower">--</span></div>
            </div>

            <!-- AADT Slider -->
            <div class="control-row">
                <span class="label" id="aadtLabel">AADT</span>
                <input id="aadtSlider" type="range" />
                <div class="slider-footer">
                    <span class="slider-value" id="aadtValueDisplay">--</span>
                    <span class="cmf-badge" id="aadtCmfBadge">CMF: --</span>
                </div>
            </div>

            <div class="controls-scroll" id="dynamicControls"></div>

            <button id="resetButton" type="button">Reset Profile</button>
            <div class="warn" id="binaryNote" style="display:none;">Binary feature: interpret as a class toggle.</div>
            <div class="kpi" id="hierarchyInfo"></div>
        </div>

        <div class="card">
            <div id="chart"></div>
            <div id="aadtChart" style="height: 36vh; margin-top: 12px;"></div>
        </div>
    </div>
</div>

<script>
const model = {json.dumps(model_payload)};
const variables = Object.keys(model.variables);

const varSelect = document.getElementById('varSelect');
const modeSelect = document.getElementById('modeSelect');
const aadtSlider = document.getElementById('aadtSlider');
const aadtLabel = document.getElementById('aadtLabel');
const aadtValueDisplay = document.getElementById('aadtValueDisplay');
const aadtCmfBadge = document.getElementById('aadtCmfBadge');
const dynamicControls = document.getElementById('dynamicControls');
const hierarchyInfo = document.getElementById('hierarchyInfo');
const binaryNote = document.getElementById('binaryNote');
const resetButton = document.getElementById('resetButton');
const predNumber = document.getElementById('predNumber');
const summElast = document.getElementById('summElast');
const summCmf = document.getElementById('summCmf');
const summUpper = document.getElementById('summUpper');
const summLower = document.getElementById('summLower');

for (const v of variables) {{
    const o = document.createElement('option');
    o.value = v; o.textContent = v;
    varSelect.appendChild(o);
}}

aadtSlider.min = String(model.aadt_range.min);
aadtSlider.max = String(model.aadt_range.max);
aadtSlider.step = String(model.aadt_range.step);
aadtSlider.value = String(model.aadt_range.default);

function fmt(value, unit) {{
    if (!Number.isFinite(value)) return 'n/a';
    let s;
    if (Math.abs(value) >= 10000) s = value.toLocaleString('en-US', {{maximumFractionDigits: 0}});
    else if (Math.abs(value) >= 100) s = value.toFixed(1);
    else if (Math.abs(value) >= 1) s = value.toFixed(3);
    else s = value.toFixed(4);
    return unit ? s + ' ' + unit : s;
}}

function normalizeValue(variable, rawValue) {{
    const stats = model.scaler_stats[variable];
    if (!stats) return rawValue;
    return (rawValue - stats.mean) / stats.sd;
}}

function currentProfile() {{
    const profile = {{ ...model.default_profile }};
    profile[model.aadt_col] = Number(aadtSlider.value);
    for (const variable of variables) {{
        const spec = model.variables[variable];
        const el = document.getElementById('control-' + variable);
        if (!el) continue;
        if (spec.is_binary) {{
            profile[variable] = el.checked ? 1 : 0;
        }} else {{
            profile[variable] = Number(el.value);
        }}
    }}
    return profile;
}}

function predict(profile, aadtValue) {{
    const safeAadt = Math.max(aadtValue, 1e-9);
    const logAadt = Math.log(safeAadt);
    let linear = model.coefficients.const + model.coefficients.log_aadt * logAadt + (model.offset_value || 0);
    for (const term of model.upper_terms) {{
        const rawValue = Number(profile[term.variable] ?? model.default_profile[term.variable] ?? 0);
        const modelValue = term.standardized ? normalizeValue(term.variable, rawValue) : rawValue;
        linear += term.coefficient * modelValue;
    }}
    for (const term of model.lower_terms) {{
        const rawValue = Number(profile[term.variable] ?? model.default_profile[term.variable] ?? 0);
        const modelValue = term.standardized ? normalizeValue(term.variable, rawValue) : rawValue;
        linear += term.coefficient * modelValue * logAadt;
    }}
    return Math.exp(linear);
}}

function decompose(profile, aadtValue) {{
    const safeAadt = Math.max(aadtValue, 1e-9);
    const logAadt = Math.log(safeAadt);
    const coreConst = model.coefficients.const;
    const coreAadt = model.coefficients.log_aadt * logAadt;
    let upperSum = 0.0, lowerSum = 0.0;
    const upperItems = [], lowerItems = [];
    for (const term of model.upper_terms) {{
        const rawValue = Number(profile[term.variable] ?? model.default_profile[term.variable] ?? 0);
        const modelValue = term.standardized ? normalizeValue(term.variable, rawValue) : rawValue;
        const contribution = term.coefficient * modelValue;
        upperSum += contribution;
        upperItems.push({{ variable: term.variable, coefficient: term.coefficient, value: rawValue, contribution, standardized: term.standardized }});
    }}
    for (const term of model.lower_terms) {{
        const rawValue = Number(profile[term.variable] ?? model.default_profile[term.variable] ?? 0);
        const modelValue = term.standardized ? normalizeValue(term.variable, rawValue) : rawValue;
        const contribution = term.coefficient * modelValue * logAadt;
        lowerSum += contribution;
        lowerItems.push({{ variable: term.variable, coefficient: term.coefficient, value: rawValue, contribution, standardized: term.standardized }});
    }}
    const offset = model.offset_value || 0;
    const linear = coreConst + coreAadt + upperSum + lowerSum + offset;
    const prediction = Math.exp(linear);
    const aadtElasticity = model.coefficients.log_aadt + lowerItems.reduce((acc, item) => {{
        const stats = model.scaler_stats[item.variable];
        const modelValue = item.standardized && stats ? (item.value - stats.mean) / stats.sd : item.value;
        return acc + item.coefficient * modelValue;
    }}, 0);
    return {{ coreConst, coreAadt, upperSum, lowerSum, linear, prediction, aadtElasticity, upperItems, lowerItems }};
}}

function buildGrid(spec) {{
    if (spec.is_binary) return [0, 1];
    const values = [];
    const points = 41;
    const width = spec.max - spec.min;
    for (let i = 0; i < points; i++) values.push(spec.min + (width * i) / (points - 1));
    return values;
}}

function cmfBadgeHTML(cmf) {{
    const pct = ((cmf - 1) * 100).toFixed(1);
    const sign = cmf >= 1 ? '+' : '';
    const cls = cmf > 1 ? ' over' : '';
    return '<span class="cmf-badge' + cls + '">CMF: ' + cmf.toFixed(3) + ' (' + sign + pct + '%)</span>';
}}

let baselinePred = null;

function draw() {{
    const variable = varSelect.value;
    const mode = modeSelect.value;
    const spec = model.variables[variable];
    const profile = currentProfile();
    const aadtValue = Number(aadtSlider.value);
    const x = buildGrid(spec);
    const pred = x.map(value => predict({{ ...profile, [variable]: value }}, aadtValue));
    const currentValue = spec.is_binary ? (profile[variable] || 0) : Number(profile[variable] ?? spec.default);
    const decomp = decompose(profile, aadtValue);
    const baseline = Math.max(decomp.prediction, 1e-9);
    const y = mode === 'pred' ? pred : pred.map(v => v / baseline);
    const selectedY = mode === 'pred' ? baseline : 1.0;
    binaryNote.style.display = spec.is_binary ? 'block' : 'none';

    // Update CMF badges on all sliders
    for (const v of variables) {{
        const badgeEl = document.getElementById('cmf-badge-' + v);
        if (!badgeEl) continue;
        const vspec = model.variables[v];
        const baselineProf = {{ ...model.default_profile }};
        baselineProf[model.aadt_col] = model.aadt_range.default;
        const baseVal = predict(baselineProf, model.aadt_range.default);
        const curVal = predict(profile, aadtValue);
        // per-var CMF: prediction with this var at current vs at default
        const profileWithDefault = {{ ...profile }};
        profileWithDefault[v] = vspec.default;
        const predWithDefault = predict(profileWithDefault, aadtValue);
        const predWithCurrent = predict(profile, aadtValue);
        const varCmf = predWithDefault > 1e-9 ? predWithCurrent / predWithDefault : 1.0;
        badgeEl.className = 'cmf-badge' + (varCmf > 1 ? ' over' : '');
        const pct = ((varCmf - 1) * 100).toFixed(1);
        const sign = varCmf >= 1 ? '+' : '';
        badgeEl.textContent = 'CMF: ' + varCmf.toFixed(3) + ' (' + sign + pct + '%)';
    }}

    // Update AADT slider display
    const aadtUnit = 'veh/day';
    aadtValueDisplay.textContent = fmt(aadtValue, aadtUnit);
    const aadtBaselinePred = predict(profile, model.aadt_range.default);
    const aadtCmf = aadtBaselinePred > 1e-9 ? baseline / aadtBaselinePred : 1.0;
    aadtCmfBadge.className = 'cmf-badge' + (aadtCmf > 1 ? ' over' : '');
    aadtCmfBadge.textContent = 'CMF: ' + aadtCmf.toFixed(3);

    // Update prediction summary
    predNumber.textContent = baseline.toFixed(3);
    const elast = decomp.aadtElasticity;
    summElast.textContent = elast.toFixed(4);
    summElast.className = 'pv' + (elast > 0 ? ' good' : ' bad');
    summCmf.textContent = aadtCmf.toFixed(4);
    summCmf.className = 'pv' + (aadtCmf <= 1 ? ' good' : ' bad');
    summUpper.textContent = decomp.upperSum.toFixed(4);
    summUpper.className = 'pv';
    summLower.textContent = decomp.lowerSum.toFixed(4);
    summLower.className = 'pv';

    const mainTrace = {{
        x, y,
        type: 'scatter', mode: 'lines+markers',
        marker: {{ size: 5, color: '#0a6c74' }},
        line: {{ width: 3, color: '#0a6c74' }},
        name: mode === 'pred' ? 'Predicted crashes' : 'CMF'
    }};
    const markerTrace = {{
        x: [currentValue], y: [selectedY],
        type: 'scatter', mode: 'markers',
        marker: {{ size: 12, color: '#d96f32', symbol: 'diamond' }},
        name: 'Current profile'
    }};
    const yTitle = mode === 'pred' ? 'Predicted crashes' : 'CMF';
    Plotly.react('chart', [mainTrace, markerTrace], {{
        template: 'plotly_white',
        title: variable + ' response at AADT ' + fmt(aadtValue, ''),
        xaxis: {{ title: variable }},
        yaxis: {{ title: yTitle }},
        margin: {{ l: 64, r: 20, t: 56, b: 52 }},
        legend: {{ orientation: 'h' }}
    }}, {{ responsive: true }});

    const aadtGrid = [], aadtPred = [];
    const aadtPoints = 61;
    for (let i = 0; i < aadtPoints; i++) {{
        const v = model.aadt_range.min + ((model.aadt_range.max - model.aadt_range.min) * i) / (aadtPoints - 1);
        aadtGrid.push(v);
        aadtPred.push(predict(profile, v));
    }}
    const aadtDefaultPred = Math.max(predict(profile, model.aadt_range.default), 1e-9);
    const aadtY = mode === 'pred' ? aadtPred : aadtPred.map(v => v / aadtDefaultPred);
    Plotly.react('aadtChart', [
        {{ x: aadtGrid, y: aadtY, type: 'scatter', mode: 'lines', line: {{ width: 3, color: '#18435a' }}, name: mode === 'pred' ? 'AADT response' : 'AADT CMF' }},
        {{ x: [aadtValue], y: [mode === 'pred' ? baseline : baseline / aadtDefaultPred], type: 'scatter', mode: 'markers', marker: {{ size: 12, color: '#d96f32', symbol: 'diamond' }}, name: 'Current AADT' }}
    ], {{
        template: 'plotly_white',
        title: 'AADT response for current profile',
        xaxis: {{ title: model.aadt_col }},
        yaxis: {{ title: yTitle }},
        margin: {{ l: 64, r: 20, t: 56, b: 52 }},
        legend: {{ orientation: 'h' }}
    }}, {{ responsive: true }});

    aadtLabel.textContent = 'AADT (' + fmt(aadtValue, '') + ')';
    const topUpper = [...decomp.upperItems]
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .slice(0, 5)
        .map(item => item.variable + ': ' + fmt(item.value, '') + ' → ' + fmt(item.contribution, ''))
        .join('<br/>') || '(none)';
    const topLower = [...decomp.lowerItems]
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .slice(0, 5)
        .map(item => item.variable + ': ' + fmt(item.value, '') + ' → ' + fmt(item.contribution, ''))
        .join('<br/>') || '(none)';
    hierarchyInfo.innerHTML = [
        '<strong>Top Upper-Level Terms</strong><br/>' + topUpper,
        '<strong>Top Lower-Level Terms (×log AADT)</strong><br/>' + topLower,
    ].join('<br/>');
}}

function buildControls() {{
    dynamicControls.innerHTML = '';
    for (const variable of variables) {{
        const spec = model.variables[variable];
        const wrapper = document.createElement('div');
        wrapper.className = 'control-row';
        const label = document.createElement('span');
        label.className = 'label';
        // Show full name + unit; tooltip shows description on hover
        const unitTxt = spec.unit ? ' (' + spec.unit + ')' : '';
        label.textContent = (spec.label || variable) + unitTxt;
        if (spec.description) {{
          label.title = spec.description;
          label.style.cursor = 'help';
          label.style.borderBottom = '1px dotted var(--muted)';
        }}
        wrapper.appendChild(label);

        if (spec.is_binary) {{
            const toggleWrap = document.createElement('div');
            toggleWrap.className = 'toggle-wrap';
            const switchEl = document.createElement('label');
            switchEl.className = 'toggle-switch';
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.id = 'control-' + variable;
            input.checked = spec.default === 1;
            const slider = document.createElement('span');
            slider.className = 'toggle-slider';
            switchEl.appendChild(input);
            switchEl.appendChild(slider);
            const toggleLabel = document.createElement('span');
            toggleLabel.className = 'toggle-label';
            toggleLabel.id = 'toggle-label-' + variable;
            toggleLabel.textContent = input.checked ? 'Present (1)' : 'Absent (0)';
            input.addEventListener('change', () => {{
                toggleLabel.textContent = input.checked ? 'Present (1)' : 'Absent (0)';
                draw();
            }});
            const badgeSpan = document.createElement('span');
            badgeSpan.id = 'cmf-badge-' + variable;
            badgeSpan.className = 'cmf-badge';
            badgeSpan.textContent = 'CMF: --';
            toggleWrap.appendChild(switchEl);
            toggleWrap.appendChild(toggleLabel);
            toggleWrap.appendChild(badgeSpan);
            wrapper.appendChild(toggleWrap);
        }} else {{
            const input = document.createElement('input');
            input.type = 'range';
            input.id = 'control-' + variable;
            input.min = String(spec.min);
            input.max = String(spec.max);
            input.step = String(spec.step);
            input.value = String(spec.default);

            const footer = document.createElement('div');
            footer.className = 'slider-footer';
            const valueSpan = document.createElement('span');
            valueSpan.className = 'slider-value';
            valueSpan.id = 'value-' + variable;
            valueSpan.textContent = fmt(spec.default, spec.unit || '');
            const badgeSpan = document.createElement('span');
            badgeSpan.id = 'cmf-badge-' + variable;
            badgeSpan.className = 'cmf-badge';
            badgeSpan.textContent = 'CMF: --';
            footer.appendChild(valueSpan);
            footer.appendChild(badgeSpan);

            const tickRow = document.createElement('div');
            tickRow.className = 'tick-row';
            const u = spec.unit || '';
            tickRow.innerHTML = '<span>p10: ' + fmt(spec.q10, u) + '</span><span>p50: ' + fmt(spec.q50, u) + '</span><span>p90: ' + fmt(spec.q90, u) + '</span>';

            input.addEventListener('input', () => {{
                valueSpan.textContent = fmt(Number(input.value), spec.unit || '');
                draw();
            }});
            wrapper.appendChild(input);
            wrapper.appendChild(footer);
            wrapper.appendChild(tickRow);
        }}
        dynamicControls.appendChild(wrapper);
    }}
}}

function resetProfile() {{
    aadtSlider.value = String(model.aadt_range.default);
    for (const variable of variables) {{
        const spec = model.variables[variable];
        const control = document.getElementById('control-' + variable);
        if (!control) continue;
        if (spec.is_binary) {{
            control.checked = spec.default === 1;
            const lbl = document.getElementById('toggle-label-' + variable);
            if (lbl) lbl.textContent = control.checked ? 'Present (1)' : 'Absent (0)';
        }} else {{
            control.value = String(spec.default);
            const valueText = document.getElementById('value-' + variable);
            if (valueText) valueText.textContent = fmt(Number(spec.default), spec.unit || '');
        }}
    }}
    draw();
}}

varSelect.value = variables[0];
buildControls();
draw();
varSelect.addEventListener('change', draw);
modeSelect.addEventListener('change', draw);
aadtSlider.addEventListener('input', () => {{
    aadtValueDisplay.textContent = fmt(Number(aadtSlider.value), 'veh/day');
    draw();
}});
resetButton.addEventListener('click', resetProfile);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _save_convergence_html(path: Path, history_df: pd.DataFrame, dataset_label: str) -> None:
    """
    Convergence dashboard with:
      - Proposed-path trace (thin line connecting all candidates in order)
      - Running-best envelope (thick step-function line)
      - Dual objectives: BIC (left) and Val RMSE (right)
      - Phase colouring (random = teal, hillclimb = orange)
      - Fixed-height window so it fits inside a presentation slide
    """
    try:
        df = history_df.copy()
        if "Iteration" in df.columns:
            df = df.sort_values("Iteration").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
            df["Iteration"] = df.index + 1

        iters      = df["Iteration"].tolist()
        bic_raw    = [float(v) if np.isfinite(float(v)) else float("nan") for v in df["BIC"]]
        rmse_raw   = [float(v) if np.isfinite(float(v)) else float("nan") for v in df.get("Val RMSE", [float("nan")] * len(df))]
        dev_raw    = [float(v) if np.isfinite(float(v)) else float("nan") for v in df["Val Poisson Dev"]]
        upper_vars = df["Upper Vars"].tolist() if "Upper Vars" in df.columns else [""] * len(df)
        lower_vars = df["Lower Vars"].tolist() if "Lower Vars" in df.columns else [""] * len(df)
        monotonic  = df["Monotonic AADT OK"].tolist() if "Monotonic AADT OK" in df.columns else ["yes"] * len(df)
        phases     = df["Phase"].tolist() if "Phase" in df.columns else ["random"] * len(df)

        def _running_min(vals: list[float]) -> list[float]:
            cur = float("inf")
            out = []
            for v in vals:
                if np.isfinite(v):
                    cur = min(cur, v)
                out.append(cur if np.isfinite(cur) else float("nan"))
            return out

        bic_rmin  = _running_min(bic_raw)
        rmse_rmin = _running_min(rmse_raw)

        # Colours: phase (random=teal, hillclimb=orange) × monotonic (filled/hollow)
        def _dot_color(phase: str, mono: str) -> str:
            base = "#7fdee8" if phase == "random" else "#d96f32"
            return base if mono == "yes" else "#8c959f"

        dot_colors = [_dot_color(p, m) for p, m in zip(phases, monotonic)]

        best_bic_idx  = int(np.nanargmin(bic_raw))
        best_rmse_idx = int(np.nanargmin(rmse_raw))

        payload = {
            "iters": iters,
            "bic": bic_raw, "bic_rmin": bic_rmin,
            "rmse": rmse_raw, "rmse_rmin": rmse_rmin,
            "dev": dev_raw,
            "upper_vars": upper_vars, "lower_vars": lower_vars,
            "monotonic": monotonic, "phases": phases,
            "dot_colors": dot_colors,
            "best_bic_idx": best_bic_idx, "best_rmse_idx": best_rmse_idx,
            "label": dataset_label,
        }

        html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Search Convergence</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing:border-box; }}
  html, body {{
    margin:0; padding:0; width:100%; height:100%;
    background:#0f1923; color:#e8edf2;
    font-family:'Segoe UI',sans-serif; overflow:hidden;
  }}
  .hdr {{ padding:10px 18px 3px; }}
  .hdr h2 {{ margin:0; font-size:1.0rem; color:#7fdee8; white-space:nowrap; }}
  .hdr p  {{ margin:2px 0 0; font-size:0.76rem; color:#8fa3b3; }}
  .leg {{
    display:flex; gap:14px; padding:3px 18px 4px;
    font-size:0.74rem; color:#8fa3b3; flex-wrap:nowrap; align-items:center;
  }}
  .ld {{ width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:3px; flex-shrink:0; }}
  /* Two charts side by side, each exactly half width */
  .charts {{
    display:flex; flex-direction:row; width:100%;
    height:calc(100vh - 72px);
  }}
  #bc, #rc {{ flex:1 1 50%; min-width:0; height:100%; }}
</style>
</head>
<body>
<div class="hdr">
  <h2>{dataset_label} — Search Convergence</h2>
  <p>Dotted = proposed path in order &nbsp;·&nbsp; Solid = running best &nbsp;·&nbsp;
     Green = monotonic AADT &nbsp;·&nbsp; Grey = non-monotonic &nbsp;·&nbsp; Star = global best</p>
</div>
<div class="leg">
  <span><span class="ld" style="background:#2a7f4f"></span>Monotonic</span>
  <span><span class="ld" style="background:#8c959f"></span>Non-monotonic</span>
  <span style="color:#ffd700">★ best</span>
  <span>— running min</span>
  <span style="color:#8fa3b3">····· proposed path</span>
</div>
<div class="charts">
  <div id="bc"></div>
  <div id="rc"></div>
</div>
<script>
const D = {json.dumps(payload)};

const BASE = {{
  template:'plotly_dark',
  paper_bgcolor:'#0f1923',
  plot_bgcolor:'#172232',
  font:{{color:'#c8d6e0', size:10}},
  margin:{{l:60, r:18, t:38, b:44}},
  legend:{{orientation:'h', y:-0.15, font:{{size:9}}, traceorder:'normal'}},
  xaxis:{{title:'Search iteration', gridcolor:'#1e3045', zeroline:false}},
}};

const cd = D.iters.map((_,i) =>
  [D.upper_vars[i], D.lower_vars[i], D.monotonic[i]]);
const dot_hover =
  'Iter %{{x}}<br>Value: %{{y:.3f}}<br>' +
  'Upper: %{{customdata[0]}}<br>Lower: %{{customdata[1]}}<br>' +
  'Monotonic: %{{customdata[2]}}<extra></extra>';

function pathT(y, name) {{
  return {{
    x:D.iters, y, type:'scatter', mode:'lines', name,
    line:{{color:'rgba(160,200,240,0.15)', width:1.2, dash:'dot'}},
    showlegend:false, hoverinfo:'skip'
  }};
}}
function dotsT(y, name) {{
  return {{
    x:D.iters, y, type:'scatter', mode:'markers', name,
    marker:{{color:D.dot_colors, size:6, opacity:0.85}},
    customdata:cd, hovertemplate:dot_hover
  }};
}}
function envT(y, color, name) {{
  return {{
    x:D.iters, y, type:'scatter', mode:'lines', name,
    line:{{color, width:3, shape:'hv'}}
  }};
}}
function starT(idx, y, name) {{
  return {{
    x:[D.iters[idx]], y:[y[idx]], type:'scatter', mode:'markers', name,
    marker:{{symbol:'star', size:16, color:'#ffd700', line:{{color:'#fff',width:1}}}},
    hovertemplate:'BEST<br>Iter %{{x}}<br>%{{y:.3f}}<extra></extra>'
  }};
}}

Plotly.newPlot('bc',
  [pathT(D.bic,'proposed'), dotsT(D.bic,'BIC'), envT(D.bic_rmin,'#7fdee8','Running best'), starT(D.best_bic_idx,D.bic,'Best')],
  {{...BASE, title:{{text:'BIC over iterations',font:{{size:12}}}},
    yaxis:{{title:'BIC', gridcolor:'#1e3045', zeroline:false}}}},
  {{responsive:true, displayModeBar:false}});

Plotly.newPlot('rc',
  [pathT(D.rmse,'proposed'), dotsT(D.rmse,'Val RMSE'), envT(D.rmse_rmin,'#f0b060','Running best'), starT(D.best_rmse_idx,D.rmse,'Best')],
  {{...BASE, title:{{text:'Validation RMSE over iterations',font:{{size:12}}}},
    yaxis:{{title:'Val RMSE', gridcolor:'#1e3045', zeroline:false}}}},
  {{responsive:true, displayModeBar:false}});
</script>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"[WARNING] Could not save convergence HTML: {exc}")


def _save_obs_pred_aadt_html(
    path: Path,
    df_all: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aadt: np.ndarray,
    title: str,
    split_labels: list[str],
) -> None:
    """Save a self-contained Plotly HTML with observed vs predicted by AADT and residual panel."""
    try:
        y = np.asarray(y_true, dtype=float).reshape(-1)
        p = np.asarray(y_pred, dtype=float).reshape(-1)
        a = np.asarray(aadt, dtype=float).reshape(-1)
        n = min(y.size, p.size, a.size)
        y, p, a = y[:n], p[:n], a[:n]
        labels = list(split_labels)[:n] if split_labels else ["all"] * n

        # Work in log(AADT) space throughout — the model is linear in log(AADT)
        log_a  = np.log(np.clip(a, 1.0, None))          # log(AADT) for each obs
        p_clip = np.clip(p, 0, np.quantile(p, 0.99))
        try:
            coeffs        = np.polyfit(log_a, p_clip, deg=3)
            log_a_fine    = np.linspace(log_a.min(), log_a.max(), 200)
            p_fine        = np.maximum(np.polyval(coeffs, log_a_fine), 0.0)
            smooth_logaadt = log_a_fine.tolist()
            smooth_pred    = p_fine.tolist()
        except Exception:
            si             = np.argsort(log_a)
            smooth_logaadt = log_a[si].tolist()
            smooth_pred    = p_clip[si].tolist()

        residuals    = (p - y) / np.maximum(y, 1.0)
        laadt_p10    = float(np.quantile(log_a, 0.10))
        laadt_p25    = float(np.quantile(log_a, 0.25))
        laadt_p75    = float(np.quantile(log_a, 0.75))
        laadt_p90    = float(np.quantile(log_a, 0.90))

        color_map    = {"train": "#0a6c74", "val": "#d96f32", "validation": "#d96f32",
                        "test": "#7a3c8c", "all": "#1f5f8b"}
        obs_colors   = [color_map.get(str(lbl).lower(), "#4a6785") for lbl in labels]
        resid_colors = ["#d96f32" if r > 0 else "#0a6c74" for r in residuals]

        hover_text = [
            f"log(AADT): {log_a[i]:.3f}  (AADT≈{a[i]:,.0f})<br>"
            f"Observed: {y[i]:.1f}<br>Predicted: {p[i]:.3f}<br>"
            f"Resid%: {residuals[i]*100:+.1f}%<br>Split: {labels[i]}"
            for i in range(n)
        ]

        payload = {
            "log_aadt":    log_a.tolist(),           # x values for scatter
            "y_true":      y.tolist(),
            "y_pred":      p.tolist(),
            "residuals":   residuals.tolist(),
            "labels":      labels,
            "obs_colors":  obs_colors,
            "resid_colors": resid_colors,
            "smooth_logaadt": smooth_logaadt,        # x values for smooth line
            "smooth_pred": smooth_pred,
            "hover_text":  hover_text,
            "laadt_p10": laadt_p10, "laadt_p25": laadt_p25,
            "laadt_p75": laadt_p75, "laadt_p90": laadt_p90,
            "title": title,
        }

        html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ margin: 0; background: #f8f6f2; font-family: 'Segoe UI', sans-serif; color: #152238; }}
  .header {{ padding: 16px 22px 4px; background: linear-gradient(120deg,#0a6c74,#18435a); color:#fff; }}
  .header h2 {{ margin:0; font-size:1.2rem; }}
  .header p {{ margin:4px 0 0; font-size:0.85rem; color:rgba(255,255,255,0.85); }}
  #scatter_chart {{ height: 52vh; }}
  #resid_chart {{ height: 36vh; }}
</style>
</head>
<body>
<div class="header">
  <h2>{title}</h2>
  <p>Top: observed (dots) and smoothed predicted (line) vs AADT. Bottom: normalized residuals = (pred-obs)/max(obs,1).</p>
</div>
<div id="scatter_chart"></div>
<div id="resid_chart"></div>
<script>
const D = {json.dumps(payload)};

// Band shape helper
function bandShape(x0, x1, color) {{
  return {{
    type: 'rect', xref: 'x', yref: 'paper',
    x0: x0, x1: x1, y0: 0, y1: 1,
    fillcolor: color, opacity: 0.08, line: {{ width: 0 }}
  }};
}}

const shapes_scatter = [
  bandShape(D.laadt_p10, D.laadt_p25, '#0a6c74'),
  bandShape(D.laadt_p75, D.laadt_p90, '#d96f32'),
];

const obsTrace = {{
  x: D.log_aadt, y: D.y_true,
  type: 'scatter', mode: 'markers',
  marker: {{ color: D.obs_colors, size: 7, opacity: 0.6, line: {{ color: 'rgba(255,255,255,0.4)', width: 0.5 }} }},
  name: 'Observed crashes',
  hovertext: D.hover_text, hoverinfo: 'text'
}};

const predLine = {{
  x: D.smooth_logaadt, y: D.smooth_pred,
  type: 'scatter', mode: 'lines',
  line: {{ color: '#d96f32', width: 3 }},
  name: 'Predicted (polynomial smooth)'
}};

Plotly.newPlot('scatter_chart', [obsTrace, predLine], {{
  template: 'plotly_white',
  title: D.title,
  xaxis: {{ title: 'log(AADT)', gridcolor: '#e0e4e8' }},
  yaxis: {{ title: 'Crash count', gridcolor: '#e0e4e8' }},
  margin: {{ l: 70, r: 20, t: 60, b: 50 }},
  legend: {{ orientation: 'h' }},
  shapes: shapes_scatter,
}}, {{ responsive: true }});

const residTrace = {{
  x: D.log_aadt, y: D.residuals,
  type: 'scatter', mode: 'markers',
  marker: {{ color: D.resid_colors, size: 7, opacity: 0.65 }},
  name: 'Normalized residual',
  hovertext: D.hover_text, hoverinfo: 'text'
}};

const zeroLine = {{
  x: [Math.min(...D.log_aadt), Math.max(...D.log_aadt)],
  y: [0, 0],
  type: 'scatter', mode: 'lines',
  line: {{ color: '#333', width: 1.5, dash: 'dash' }},
  name: 'Zero'
}};

Plotly.newPlot('resid_chart', [residTrace, zeroLine], {{
  template: 'plotly_white',
  title: 'Normalized residuals by log(AADT)',
  xaxis: {{ title: 'log(AADT)', gridcolor: '#e0e4e8' }},
  yaxis: {{ title: '(pred - obs) / max(obs,1)', gridcolor: '#e0e4e8', zeroline: true }},
  margin: {{ l: 70, r: 20, t: 50, b: 50 }},
  legend: {{ orientation: 'h' }},
  shapes: [
    bandShape(D.laadt_p10, D.laadt_p25, '#0a6c74'),
    bandShape(D.laadt_p75, D.laadt_p90, '#d96f32'),
  ],
}}, {{ responsive: true }});

function bandShape(x0, x1, color) {{
  return {{
    type: 'rect', xref: 'x', yref: 'paper',
    x0: x0, x1: x1, y0: 0, y1: 1,
    fillcolor: color, opacity: 0.08, line: {{ width: 0 }}
  }};
}}
</script>
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"[WARNING] Could not save AADT obs/pred HTML: {exc}")


def _jax_random_params_refit(
    df_trainval_raw: pd.DataFrame,
    best_upper_raw: list[str],
    best_lower_raw: list[str],
    y_col: str,
    aadt_col: str,
    offset_col: str | None,
    scaler_stats: dict[str, tuple[float, float]],
    binary_vars: set[str],
) -> dict[str, Any] | None:
    """Attempt a random-parameters refit using ExperimentBuilder if available."""
    # Ensure the package root is on sys.path regardless of how the script is invoked
    import sys as _sys
    _pkg_root = str(Path(__file__).resolve().parent.parent)
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)

    try:
        from experiment_package import ExperimentBuilder  # type: ignore
    except ImportError:
        try:
            from ..experiment_package import ExperimentBuilder  # type: ignore
        except ImportError:
            return None
    except Exception:
        return None

    try:
        df = df_trainval_raw.copy()
        df["_id"] = np.arange(len(df), dtype=int)

        log_aadt = np.log(np.clip(
            pd.to_numeric(df[aadt_col], errors="coerce").to_numpy(dtype=float), 1e-12, None
        ))
        df["_log_aadt"] = log_aadt

        for var in best_lower_raw:
            x = pd.to_numeric(df[var], errors="coerce").to_numpy(dtype=float)
            df[f"_inter_{var}"] = x * log_aadt

        for var, (mu, sd) in scaler_stats.items():
            if var in df.columns:
                df[f"{var}_Z"] = (pd.to_numeric(df[var], errors="coerce") - mu) / sd

        def _model_name(v: str) -> str:
            return v if v in binary_vars else f"{v}_Z"

        continuous_upper = [v for v in best_upper_raw if v not in binary_vars]
        continuous_lower = [v for v in best_lower_raw if v not in binary_vars]

        # Variable name mapper (raw → model column)
        def _model_name(v: str) -> str:
            return v if v in binary_vars else f"{v}_Z"

        rdm_upper_names = [_model_name(v) for v in continuous_upper
                           if _model_name(v) in df.columns]

        # Fixed terms exclude the continuous upper vars (they enter via random)
        fixed_binary_upper = [_model_name(v) for v in best_upper_raw
                               if _model_name(v) not in rdm_upper_names]
        fixed_terms = (
            ["_log_aadt"]
            + [t for t in fixed_binary_upper if t in df.columns]
            + [f"_inter_{v}" for v in best_lower_raw if f"_inter_{v}" in df.columns]
        )

        builder = ExperimentBuilder(df=df, id_col="_id", y_col=y_col,
                                    offset_col=offset_col)

        def _extract_result(fit: dict, cor_names: list[str] = ()) -> dict:
            """Extract readable coefficient table, corr matrix, BIC from a fit."""
            spec   = fit["spec"]
            params = np.array(fit["result"].params)
            try:
                from main_hpc_lc_patch import build_base_index as _bidx
                from main_hpc import unpack_params as _unpack
                import jax as _jax
                idx    = _bidx(spec)
                blocks = _unpack(params, spec)
                rows   = []

                # Fixed params
                for k, name in enumerate(spec.fixed_names):
                    lbl = VARIABLE_LABELS.get(
                        name.replace("_Z","").replace("__INTERCEPT__","").strip("_"),
                        name.replace("__INTERCEPT__","Intercept"))
                    rows.append({"Parameter": lbl, "Type": "Fixed",
                                 "Mean": round(float(np.array(blocks["beta_f"])[k]), 5), "SD": ""})

                # Independent random params
                if spec.Kr_ind > 0 and blocks.get("mean_ind") is not None:
                    means = np.array(blocks["mean_ind"])
                    sds   = np.abs(np.array(blocks["sd_ind"]))
                    for j, rname in enumerate(spec.random_ind_names):
                        lbl = VARIABLE_LABELS.get(rname.replace("_Z",""), rname)
                        rows.append({"Parameter": f"{lbl} [random, independent]",
                                     "Type": "Random-Ind",
                                     "Mean": round(float(means[j]), 5),
                                     "SD":   round(float(sds[j]),   5)})

                # Correlated random params
                corr_matrix_str = ""
                if spec.Kr_cor > 0 and blocks.get("mean_cor") is not None:
                    means_c = np.array(blocks["mean_cor"])
                    chol_v  = np.array(blocks["chol"])
                    K       = spec.Kr_cor
                    L       = np.zeros((K, K))
                    i = 0
                    for r in range(K):
                        for c in range(r + 1):
                            L[r, c] = chol_v[i]; i += 1
                    L[np.diag_indices(K)] = np.abs(np.diag(L))
                    Sigma = L @ L.T
                    sds_c = np.sqrt(np.diag(Sigma))
                    corr  = Sigma / (sds_c[:, None] * sds_c[None, :] + 1e-12)
                    for j, rname in enumerate(spec.random_cor_names):
                        lbl = VARIABLE_LABELS.get(rname.replace("_Z",""), rname)
                        rows.append({"Parameter": f"{lbl} [random, correlated]",
                                     "Type": "Random-Cor",
                                     "Mean": round(float(means_c[j]), 5),
                                     "SD":   round(float(sds_c[j]),   5)})
                    # Format correlation matrix as text
                    cor_lbls = [VARIABLE_LABELS.get(n.replace("_Z",""), n)
                                for n in spec.random_cor_names]
                    corr_lines = ["  Correlation matrix of correlated random parameters:"]
                    header = "    " + "  ".join(f"{l[:14]:>14}" for l in cor_lbls)
                    corr_lines.append(header)
                    for r, rl in enumerate(cor_lbls):
                        row_s = "  ".join(f"{corr[r,c]:+14.4f}" for c in range(K))
                        corr_lines.append(f"  {rl[:20]:20}  {row_s}")
                    corr_matrix_str = "\n".join(corr_lines)

                # NB2 dispersion
                if blocks.get("alpha") is not None:
                    rows.append({"Parameter": "NB2 Dispersion (alpha)",
                                 "Type": "Fixed",
                                 "Mean": round(float(_jax.nn.softplus(blocks["alpha"])), 5),
                                 "SD": ""})

                coef_df = pd.DataFrame(rows)
                summary_d = fit.get("summary", {})
                ll  = summary_d.get("loglik", float("nan")) if isinstance(summary_d, dict) else float("nan")
                bic = summary_d.get("bic",    float("nan")) if isinstance(summary_d, dict) else float("nan")
                return {"fit": fit, "coef_df": coef_df,
                        "coef_str": coef_df.to_string(index=False),
                        "corr_matrix_str": corr_matrix_str,
                        "loglik": ll, "bic": bic,
                        "summary": str(fit.get("summary", {}))}
            except Exception as exc:
                summary_d = fit.get("summary", {})
                ll  = summary_d.get("loglik", float("nan")) if isinstance(summary_d, dict) else float("nan")
                bic = summary_d.get("bic",    float("nan")) if isinstance(summary_d, dict) else float("nan")
                return {"fit": fit, "coef_df": pd.DataFrame(),
                        "coef_str": f"(extraction failed: {exc})",
                        "corr_matrix_str": "",
                        "loglik": ll, "bic": bic, "summary": ""}

        best_result: dict | None = None

        # ── Try 1: all continuous upper as INDEPENDENT random ────────────
        if rdm_upper_names:
            try:
                spec_ind = builder.make_manual_spec(
                    fixed_terms = fixed_terms,
                    rdm_terms   = [f"{n}:normal" for n in rdm_upper_names],
                    dispersion  = 1,
                )
                fit_ind = builder.fit_manual_model(spec_ind, model="nb", print_report=False)
                res_ind = _extract_result(fit_ind)
                if np.isfinite(res_ind["bic"]):
                    print(f"    [RP] Independent random params BIC: {res_ind['bic']:.2f}")
                    best_result = res_ind
            except Exception:
                pass

        # ── Try 2: top-2 continuous upper as CORRELATED random ───────────
        if len(rdm_upper_names) >= 2:
            try:
                cor_names  = rdm_upper_names[:2]   # correlated pair
                ind_rest   = rdm_upper_names[2:]    # remaining as independent
                spec_cor = builder.make_manual_spec(
                    fixed_terms   = fixed_terms,
                    rdm_cor_terms = [f"{n}:normal" for n in cor_names],
                    rdm_terms     = [f"{n}:normal" for n in ind_rest],
                    dispersion    = 1,
                )
                fit_cor = builder.fit_manual_model(spec_cor, model="nb", print_report=False)
                res_cor = _extract_result(fit_cor, cor_names=cor_names)
                if np.isfinite(res_cor["bic"]):
                    print(f"    [RP] Correlated random params BIC: {res_cor['bic']:.2f}")
                    if best_result is None or res_cor["bic"] < best_result["bic"]:
                        best_result = res_cor
            except Exception:
                pass

        if best_result is None:
            return None
        return best_result

    except Exception:
        return None


def _print_readable_model(
    fitted: FittedModel,
    scaler_stats: dict[str, tuple[float, float]],
    binary_vars: set[str],
    save_path: Path | None = None,
) -> None:
    """
    Print (and optionally save) a plain-English model summary with:
      - the full variable name beside each coefficient
      - the original-scale effect (un-standardised)
      - the implied CMF at median AADT for a 1-unit or 1-SD change
    """
    params = pd.Series(fitted.result.params)
    lines: list[str] = []

    w = 70
    lines.append("=" * w)
    lines.append("  FINAL HIERARCHICAL CMF MODEL — READABLE SUMMARY")
    lines.append("=" * w)
    lines.append("")
    lines.append("  Model: log(crashes) = Intercept + b_AADT·log(AADT)")
    lines.append("       + sum [Upper terms: direct effect on crash rate]")
    lines.append("       + sum [Lower terms: modify AADT elasticity]")
    lines.append("       + log(segment length)  [exposure offset]")
    lines.append("")

    # ── Core ──────────────────────────────────────────────────────────────
    lines.append("  CORE PARAMETERS")
    lines.append("  " + "-" * (w - 2))
    for pname in ["const", "log_aadt"]:
        if pname in params.index:
            lbl = "Intercept" if pname == "const" else "log(AADT) elasticity"
            lines.append(f"  {lbl:<45}  {params[pname]:>+9.4f}")
    lines.append("")

    # ── Upper terms ────────────────────────────────────────────────────────
    upper_rows = [(n, v) for n, v in params.items() if str(n).startswith("upper::")]
    if upper_rows:
        lines.append("  UPPER-LEVEL TERMS  (direct effect — AADT-independent)")
        lines.append(f"  {'Variable':<45}  {'Coef':>9}  {'Orig scale':>11}  {'CMF (1-unit)':>12}")
        lines.append("  " + "-" * (w - 2))
        for pname, coef in upper_rows:
            raw = pname.replace("upper::", "")
            is_std = raw.endswith("_Z")
            base_var = raw[:-2] if is_std else raw
            lbl = _label(base_var)
            sd = scaler_stats.get(base_var, (0.0, 1.0))[1]
            coef_orig = float(coef) / sd if is_std and sd > 0 else float(coef)
            cmf = float(np.exp(coef_orig))
            tag = " (per SD)" if is_std else ""
            lines.append(
                f"  {lbl:<45}  {coef:>+9.4f}  {coef_orig:>+11.4f}{tag}  CMF={cmf:>7.4f}"
            )
        lines.append("")

    # ── Lower terms ────────────────────────────────────────────────────────
    lower_rows = [(n, v) for n, v in params.items()
                  if str(n).startswith("lower::") and str(n).endswith("*log_aadt")]

    # Extract the base AADT elasticity for elasticity range display
    beta_aadt = float(params.get("log_aadt", params.get("log(AADT)", 0.0)))

    if lower_rows:
        lines.append("  LOWER-LEVEL TERMS  (modify AADT elasticity — x × log(AADT) interaction)")
        lines.append("")
        lines.append("  These do NOT affect the baseline crash rate directly.  They shift")
        lines.append("  how steeply crash risk scales with AADT:  a negative coefficient means")
        lines.append("  segments with high values of that variable are LESS sensitive to AADT.")
        lines.append("")
        lines.append(f"  {'Variable':<45}  {'Coef':>9}  {'Interpretation'}")
        lines.append("  " + "-" * (w - 2))
        for pname, coef in lower_rows:
            raw = pname.replace("lower::", "").replace("*log_aadt", "")
            is_std = raw.endswith("_Z")
            base_var = raw[:-2] if is_std else raw
            lbl = _label(base_var)
            sd = scaler_stats.get(base_var, (0.0, 1.0))[1]
            coef_orig = float(coef) / sd if is_std and sd > 0 else float(coef)
            direction = ("lower" if float(coef) < 0
                         else "higher") + " AADT sensitivity"
            lines.append(
                f"  {lbl:<45}  {coef:>+9.4f}  => {direction}"
            )
        lines.append("")
        lines.append(f"  Base AADT elasticity (log_aadt coeff): {beta_aadt:+.4f}")
        lines.append("  Local elasticity = base + sum(lower_coeff_i * variable_i_std)")
        lines.append("  Positive elasticity for all segments = AADT monotonicity holds.")
        lines.append("")

    # ── FC note ────────────────────────────────────────────────────────────
    if any("FC" in str(n) for n in params.index):
        lines.append("  NOTE — Functional Classification (FC) coding:")
        for code, desc in FC_LABELS.items():
            lines.append(f"    FC = {code}  →  {desc}")
        lines.append("")

    lines.append("=" * w)
    text = "\n".join(lines)
    print(text)
    if save_path is not None:
        save_path.write_text(text, encoding="utf-8")


def _fit_benchmark(
    df_tv: pd.DataFrame,
    df_test: pd.DataFrame,
    y_col: str,
    aadt_col: str,
    offset_col: str | None,
    family: str,
) -> dict[str, Any]:
    """
    Fit the simplest possible SPF (Safety Performance Function):
        log(mu) = alpha + beta_AADT * log(AADT) + offset
    No geometry, weather, or interaction terms.
    Returns a dict of metrics on both train+val and test splits.
    """
    bench_fit = _fit_model(
        df_tv,
        aadt_col   = aadt_col,
        y_col      = y_col,
        upper_vars = [],
        lower_vars = [],
        family     = family,
        offset_col = offset_col,
    )
    if bench_fit is None:
        return {}

    y_tv   = pd.to_numeric(df_tv[y_col],   errors="coerce").to_numpy(float)
    y_test = pd.to_numeric(df_test[y_col], errors="coerce").to_numpy(float)

    pred_tv   = _predict(df_tv,   bench_fit)
    pred_test = _predict(df_test, bench_fit)

    m_tv   = _metrics(y_tv,   pred_tv)
    m_test = _metrics(y_test, pred_test)

    return {
        "family":           family,
        "benchmark_model":  "Intercept + log(AADT) + offset only",
        "tv_ll":            float(-getattr(bench_fit.result, "aic", np.nan) / 2 + 1) if hasattr(bench_fit.result, "aic") else float("nan"),
        "tv_bic":           float(getattr(bench_fit.result, "bic", np.nan)),
        "tv_aic":           float(getattr(bench_fit.result, "aic", np.nan)),
        "tv_rmse":          m_tv["rmse"],
        "tv_poisson_dev":   _poisson_deviance(y_tv, pred_tv),
        "test_rmse":        m_test["rmse"],
        "test_poisson_dev": _poisson_deviance(y_test, pred_test),
        "test_r2":          m_test["r2"],
        "fitted":           bench_fit,
    }


def _resolve_default_candidates(df: pd.DataFrame, profile: str = "core") -> tuple[list[str], list[str]]:
    if {"FREQ", "LENGTH", "AADT", "CURVES"}.issubset(df.columns):
        # Expanded core_upper: all relevant road geometry and traffic variables
        core_upper = [
            # LENGTH intentionally excluded — it is the exposure offset, not a predictor
            # INCLANES/DECLANES replaced by composite LANES (total lanes both directions)
            "LANES",
            "WIDTH",
            "MIMEDSH",
            "MXMEDSH",
            "SPEED",
            "URB",
            "FC",
            "SINGLE",
            "DOUBLE",
            "TRAIN",
            "PEAKHR",
            "GRADEBR",
            "MIGRADE",
            "MXGRADE",
            "MXGRDIFF",
            "TANGENT",
            "CURVES",
            "MINRAD",
            "ACCESS",
            "MEDWIDTH",
            "FRICTION",
            # ADTLANE excluded — collinear with AADT / lane count
            "SLOPE",
            "AVEPRE",
            "AVESNOW",
            "LOWPRE",
            "HISNOW",
            "INTPM",
            "CPM",
            "EXPOSE",
        ]
        core_lower = [
            "SPEED",
            "CURVES",
            "LANES",       # replaces INCLANES / DECLANES
            "FRICTION",
            "SLOPE",
            "AVEPRE",
            "AVESNOW",
            "HISNOW",
        ]
        expanded_only_upper: list[str] = [
            "INTECHAG",
            "GBRPM",
            "LNAADT",
        ]
        expanded_only_lower: list[str] = [
            # LENGTH excluded — it is the exposure offset
            "WIDTH",
            "MEDWIDTH",
            "MINRAD",
            "MIMEDSH",
            "MXMEDSH",
            "ACCESS",
            "PEAKHR",
            "URB",
            # ADTLANE excluded — collinear with AADT and lane count; causes
            # instability and multicollinearity in the hierarchical CMF model.
        ]
    else:
        core_upper = [
            "segment_length",
            "curve",
            "speed",
            "paved_shoulder",
            "num_lanes",
            "left_shoulder_width",
            "right_shoulder_width",
            "dummy_winter",
            "has_rumble",
            "DP01",
            "DX32",
            "PRCP",
            "SNOW",
            "TAVG",
        ]
        core_lower = [
            "curve",
            "speed",
            "paved_shoulder",
            "num_lanes",
            "left_shoulder_width",
            "right_shoulder_width",
            "has_rumble",
            "DP01",
            "DX32",
            "dummy_winter",
        ]
        expanded_only_upper = [
            "DP10",
            "DSND",
            "DSNW",
            "max_prcp",
            "max_snow",
            "TMAX",
            "TMIN",
        ]
        expanded_only_lower = [
            "segment_length",
            "PRCP",
            "SNOW",
        ]

    if profile == "expanded":
        default_upper = core_upper + expanded_only_upper
        default_lower = core_lower + expanded_only_lower
    else:
        default_upper = core_upper
        default_lower = core_lower

    upper = [v for v in dict.fromkeys(default_upper) if v in df.columns]
    lower = [v for v in dict.fromkeys(default_lower) if v in df.columns]
    return upper, lower


# ─────────────────────────────────────────────────────────────────────────────
# Static PNG generators (for PPTX / offline use — parallel to Plotly HTMLs)
# ─────────────────────────────────────────────────────────────────────────────

def _save_convergence_png(path: Path, history_df: pd.DataFrame, dataset_label: str) -> None:
    """Matplotlib version of the convergence plot for embedding in PPTX."""
    df = history_df.copy()
    n  = len(df)
    iters     = list(range(1, n + 1))
    bic_vals  = df["BIC"].tolist()
    val_dev   = df["Val Poisson Dev"].tolist()
    mono_ok   = [str(v).lower() == "yes" for v in df.get("Monotonic AADT OK", ["yes"] * n)]
    colors    = ["#2a7f4f" if ok else "#9e9e9e" for ok in mono_ok]

    run_min_bic = [min(bic_vals[: i + 1]) for i in range(n)]
    run_min_dev = [min(val_dev[: i + 1]) for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    ax1.scatter(iters, bic_vals, c=colors, s=22, alpha=0.65, zorder=3)
    ax1.plot(iters, run_min_bic, color="#0a6c74", lw=2.5, label="Running min")
    ax1.set_title("BIC over Search Iterations", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("BIC")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    ax2.scatter(iters, val_dev, c=colors, s=22, alpha=0.65, zorder=3,
                label="Green = monotonic AADT")
    ax2.plot(iters, run_min_dev, color="#d96f32", lw=2.5, label="Running min")
    ax2.set_title("Validation Deviance over Iterations", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Val Poisson Deviance")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    fig.suptitle(f"{dataset_label} — Search Convergence", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_obs_pred_aadt_png(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aadt: np.ndarray,
    title: str,
    split_labels: list[str],
) -> None:
    """Matplotlib AADT obs/pred plot for PPTX."""
    y, p, a  = (np.asarray(x, float) for x in (y_true, y_pred, aadt))
    p_clip   = np.clip(p, 0, np.quantile(p, 0.99))
    labels   = list(split_labels)

    log_a = np.log(np.clip(a, 1.0, None))   # compute before loop

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=150)
    color_map = {"train": "#0a6c74", "val": "#d96f32", "validation": "#d96f32",
                 "test": "#7a3c8c", "all": "#1f5f8b"}

    for lbl in sorted(set(labels)):
        mask = np.array([l == lbl for l in labels])
        ax1.scatter(log_a[mask], y[mask], s=18, alpha=0.55,
                    color=color_map.get(lbl, "#4a6785"), label=f"Observed ({lbl})", zorder=3)
    try:
        coeffs      = np.polyfit(log_a, p_clip, deg=3)
        log_a_fine  = np.linspace(log_a.min(), log_a.max(), 200)
        p_fine      = np.maximum(np.polyval(coeffs, log_a_fine), 0.0)
    except Exception:
        si         = np.argsort(log_a)
        log_a_fine = log_a[si]
        p_fine     = p_clip[si]
    ax1.plot(log_a_fine, p_fine, color="#d96f32", lw=2.5, label="Predicted (polynomial smooth)")
    ax1.set_xlabel("log(AADT)")
    ax1.set_ylabel("Crashes"); ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    resid        = (p - y) / np.maximum(y, 1.0)
    resid_colors = ["#d96f32" if r > 0 else "#0a6c74" for r in resid]
    ax2.scatter(log_a, resid, c=resid_colors, s=14, alpha=0.55)
    ax2.axhline(0, color="#333", lw=1.5)
    ax2.set_xlabel("log(AADT)")
    ax2.set_ylabel("Norm. Residual"); ax2.set_title("Residuals = (pred-obs)/max(obs,1)", fontsize=10)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_model_comparison_html(
    path: Path,
    y_all: np.ndarray,
    aadt_all: np.ndarray,
    pred_benchmark: np.ndarray,
    pred_hierarchical: np.ndarray,
    split_labels: list[str],
    title: str,
) -> None:
    """
    Single Plotly HTML showing observed crashes, benchmark predictions, and
    hierarchical model predictions — all vs AADT on the same axes.

    Top panel   : observed points + both smooth predicted curves
    Bottom panel: per-model normalised residuals side by side
    """
    try:
        y = np.asarray(y_all, float)
        a = np.asarray(aadt_all, float)
        pb = np.clip(np.asarray(pred_benchmark, float),    0, np.quantile(pred_benchmark, 0.99))
        ph = np.clip(np.asarray(pred_hierarchical, float), 0, np.quantile(pred_hierarchical, 0.99))
        labels = list(split_labels)
        n      = len(y)
        log_a  = np.log(np.clip(a, 1.0, None))

        def _psmooth(loga, pred, deg=3):
            try:
                c  = np.polyfit(loga, pred, deg=deg)
                xf = np.linspace(loga.min(), loga.max(), 200)
                return xf.tolist(), np.maximum(np.polyval(c, xf), 0.0).tolist()
            except Exception:
                si = np.argsort(loga)
                return loga[si].tolist(), pred[si].tolist()

        sla_b, sp_b = _psmooth(log_a, pb)
        sla_h, sp_h = _psmooth(log_a, ph)
        rb = (pb - y) / np.maximum(y, 1.0)
        rh = (ph - y) / np.maximum(y, 1.0)
        y_max = float(max(y.max(), pb.max(), ph.max())) * 1.05

        color_map  = {"train": "#0a6c74", "val": "#d96f32", "validation": "#d96f32",
                      "test": "#7a3c8c", "all": "#4a6785"}
        obs_colors = [color_map.get(str(l).lower(), "#4a6785") for l in labels]

        hover_b = [f"log(AADT):{log_a[i]:.3f}<br>Obs:{y[i]:.0f}<br>Benchmark:{pb[i]:.2f}<br>Split:{labels[i]}" for i in range(n)]
        hover_h = [f"log(AADT):{log_a[i]:.3f}<br>Obs:{y[i]:.0f}<br>Hierarchical:{ph[i]:.2f}<br>Split:{labels[i]}" for i in range(n)]

        payload = {
            "log_aadt": log_a.tolist(), "y": y.tolist(),
            "obs_colors": obs_colors,
            "hover_b": hover_b, "hover_h": hover_h,
            "sla_b": sla_b, "sp_b": sp_b,
            "sla_h": sla_h, "sp_h": sp_h,
            "rb": rb.tolist(), "rh": rh.tolist(),
            "y_max": y_max, "title": title,
        }

        html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; padding:0; width:100%; height:100%;
    background:#f8f6f2; font-family:'Segoe UI',sans-serif; overflow:hidden; }}
  .hdr {{ background:linear-gradient(120deg,#0a6c74,#18435a);
    color:#fff; padding:8px 18px 4px; }}
  .hdr h2 {{ margin:0; font-size:1.0rem; white-space:nowrap; }}
  .hdr p  {{ margin:2px 0 0; font-size:0.78rem; color:rgba(255,255,255,0.85); }}
  /* Two columns for side-by-side, two rows per column */
  .grid {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:58% 42%;
    height:calc(100vh - 56px); gap:0; }}
  .panel {{ position:relative; }}
  .label {{ position:absolute; top:6px; left:50%; transform:translateX(-50%);
    background:rgba(255,255,255,0.92); padding:2px 10px; border-radius:12px;
    font-size:0.78rem; font-weight:600; z-index:10; white-space:nowrap; }}
  .benchmark {{ color:#1f6bb5; border:1px solid #1f6bb5; }}
  .hierarchical {{ color:#d96f32; border:1px solid #d96f32; }}
  #bL, #bR, #hL, #hR {{ width:100%; height:100%; }}
</style>
</head>
<body>
<div class="hdr">
  <h2>{title}</h2>
  <p>Side by side: Benchmark (left) vs Hierarchical CMF (right) — x axis = log(AADT)</p>
</div>
<div class="grid">
  <div class="panel"><div class="label benchmark">Benchmark — AADT-only SPF</div><div id="bL"></div></div>
  <div class="panel"><div class="label hierarchical">Hierarchical CMF model</div><div id="hL"></div></div>
  <div id="bR"></div>
  <div id="hR"></div>
</div>
<script>
const D = {json.dumps(payload)};
const BASE = {{
  template:'plotly_white', paper_bgcolor:'#f8f6f2', plot_bgcolor:'#fff',
  font:{{size:10, color:'#152238'}},
  margin:{{l:54, r:8, t:28, b:38}},
  showlegend:false,
  xaxis:{{title:'log(AADT)', gridcolor:'#eee', zeroline:false}},
}};
const yax_top = {{title:'Crashes', gridcolor:'#eee', range:[0, D.y_max]}};
const yax_bot = {{title:'Norm. residual', gridcolor:'#eee', zeroline:false}};
const xr = [Math.min(...D.log_aadt), Math.max(...D.log_aadt)];

// Benchmark top
Plotly.newPlot('bL', [
  {{x:D.log_aadt, y:D.y, type:'scatter', mode:'markers', name:'Observed',
    marker:{{color:D.obs_colors, size:6, opacity:0.6}},
    hovertext:D.hover_b, hoverinfo:'text'}},
  {{x:D.sla_b, y:D.sp_b, mode:'lines',
    line:{{color:'#1f6bb5', width:3}}, name:'Benchmark predicted'}},
], {{...BASE, title:{{text:'Observed vs Benchmark',font:{{size:11}}}}, yaxis:yax_top}},
{{responsive:true, displayModeBar:false}});

// Hierarchical top
Plotly.newPlot('hL', [
  {{x:D.log_aadt, y:D.y, type:'scatter', mode:'markers', name:'Observed',
    marker:{{color:D.obs_colors, size:6, opacity:0.6}},
    hovertext:D.hover_h, hoverinfo:'text'}},
  {{x:D.sla_h, y:D.sp_h, mode:'lines',
    line:{{color:'#d96f32', width:3}}, name:'Hierarchical predicted'}},
], {{...BASE, title:{{text:'Observed vs Hierarchical CMF',font:{{size:11}}}}, yaxis:yax_top}},
{{responsive:true, displayModeBar:false}});

// Benchmark residuals
Plotly.newPlot('bR', [
  {{x:D.log_aadt, y:D.rb, mode:'markers',
    marker:{{color:D.rb.map(r=>r>0?'#1f6bb5':'#93c5e8'), size:5, opacity:0.65}},
    hovertemplate:'log(AADT) %{{x:.3f}}<br>Resid: %{{y:.2f}}<extra></extra>'}},
  {{x:xr, y:[0,0], mode:'lines', line:{{color:'#999', width:1.2}}, hoverinfo:'skip'}},
], {{...BASE, title:{{text:'Benchmark residuals',font:{{size:10}}}}, yaxis:yax_bot}},
{{responsive:true, displayModeBar:false}});

// Hierarchical residuals
Plotly.newPlot('hR', [
  {{x:D.log_aadt, y:D.rh, mode:'markers',
    marker:{{color:D.rh.map(r=>r>0?'#d96f32':'#0a6c74'), size:5, opacity:0.65}},
    hovertemplate:'log(AADT) %{{x:.3f}}<br>Resid: %{{y:.2f}}<extra></extra>'}},
  {{x:xr, y:[0,0], mode:'lines', line:{{color:'#999', width:1.2}}, hoverinfo:'skip'}},
], {{...BASE, title:{{text:'Hierarchical residuals',font:{{size:10}}}}, yaxis:yax_bot}},
{{responsive:true, displayModeBar:false}});
</script>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"[WARNING] Could not save model comparison HTML: {exc}")


def _save_model_comparison_png(
    path: Path,
    y_all: np.ndarray,
    aadt_all: np.ndarray,
    pred_benchmark: np.ndarray,
    pred_hierarchical: np.ndarray,
    split_labels: list[str],
    title: str,
) -> None:
    """Matplotlib version of the comparison plot for PPTX."""
    try:
        y  = np.asarray(y_all, float)
        a  = np.asarray(aadt_all, float)
        pb = np.clip(np.asarray(pred_benchmark, float),    0, np.quantile(pred_benchmark, 0.99))
        ph = np.clip(np.asarray(pred_hierarchical, float), 0, np.quantile(pred_hierarchical, 0.99))

        log_a = np.log(np.clip(a, 1.0, None))   # log(AADT) as x

        def _smooth_log(log_aadt, pred):
            """Return (log_aadt_fine, pred_fine) on log(AADT) linear axis."""
            try:
                coeffs     = np.polyfit(log_aadt, pred, deg=3)
                la_fine    = np.linspace(log_aadt.min(), log_aadt.max(), 200)
                p_fine     = np.maximum(np.polyval(coeffs, la_fine), 0.0)
                return la_fine, p_fine
            except Exception:
                si = np.argsort(log_aadt)
                return log_aadt[si], pred[si]

        sla_b, sp_b = _smooth_log(log_a, pb)
        sla_h, sp_h = _smooth_log(log_a, ph)

        color_map = {"train": "#0a6c74", "val": "#d96f32", "validation": "#d96f32",
                     "test": "#7a3c8c", "all": "#4a6785"}
        obs_c = [color_map.get(str(l).lower(), "#4a6785") for l in split_labels]

        rb     = (pb - y) / np.maximum(y, 1.0)
        rh     = (ph - y) / np.maximum(y, 1.0)
        y_max  = max(y.max(), pb.max(), ph.max()) * 1.05

        # 2×2 grid: top-left=benchmark obs/pred, top-right=hierarchical obs/pred,
        #           bottom-left=benchmark residuals, bottom-right=hierarchical residuals
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
        (ax_bl, ax_hr), (ax_br, ax_hrr) = axes

        ax_bl.scatter(log_a, y, c=obs_c, s=14, alpha=0.55, zorder=3, label="Observed")
        ax_bl.plot(sla_b, sp_b, "-", color="#1f6bb5", lw=2.5, label="Predicted")
        ax_bl.set_ylim(0, y_max)
        ax_bl.set_xlabel("log(AADT)"); ax_bl.set_ylabel("Crashes")
        ax_bl.set_title("Benchmark: AADT-only SPF", fontsize=10, fontweight="bold", color="#1f6bb5")
        ax_bl.legend(fontsize=7); ax_bl.grid(alpha=0.18)

        ax_hr.scatter(log_a, y, c=obs_c, s=14, alpha=0.55, zorder=3, label="Observed")
        ax_hr.plot(sla_h, sp_h, "-", color="#d96f32", lw=2.5, label="Predicted")
        ax_hr.set_ylim(0, y_max)
        ax_hr.set_xlabel("log(AADT)"); ax_hr.set_ylabel("Crashes")
        ax_hr.set_title("Hierarchical CMF model", fontsize=10, fontweight="bold", color="#d96f32")
        ax_hr.legend(fontsize=7); ax_hr.grid(alpha=0.18)

        ax_br.scatter(log_a, rb, c=["#1f6bb5" if r > 0 else "#93c5e8" for r in rb],
                      s=12, alpha=0.55)
        ax_br.axhline(0, color="#555", lw=1.2)
        ax_br.set_xlabel("log(AADT)"); ax_br.set_ylabel("(pred-obs)/max(obs,1)")
        ax_br.set_title("Benchmark residuals", fontsize=9); ax_br.grid(alpha=0.18)

        ax_hrr.scatter(log_a, rh, c=["#d96f32" if r > 0 else "#0a6c74" for r in rh],
                       s=12, alpha=0.55)
        ax_hrr.axhline(0, color="#555", lw=1.2)
        ax_hrr.set_xlabel("log(AADT)"); ax_hrr.set_ylabel("(pred-obs)/max(obs,1)")
        ax_hrr.set_title("Hierarchical residuals", fontsize=9); ax_hrr.grid(alpha=0.18)

        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARNING] Could not save comparison PNG: {exc}")


def _save_readable_coef_csv(
    path: Path,
    fitted: FittedModel,
    scaler_stats: dict[str, tuple[float, float]],
    binary_vars: set[str],
) -> pd.DataFrame:
    """Save a human-readable coefficient CSV with full variable names and CMF."""
    params = pd.Series(fitted.result.params)
    rows: list[dict] = []

    for pname, coef in params.items():
        pname = str(pname)
        if pname == "const":
            rows.append({"Component": "Core", "Variable": "Intercept",
                         "Full Name": "Intercept",
                         "Coefficient": round(float(coef), 5), "CMF": ""})
        elif pname == "log_aadt":
            rows.append({"Component": "Core", "Variable": "log(AADT)",
                         "Full Name": "log(AADT) Elasticity",
                         "Coefficient": round(float(coef), 5), "CMF": ""})
        elif pname.startswith("upper::"):
            raw     = pname.replace("upper::", "")
            is_std  = raw.endswith("_Z")
            base    = raw[:-2] if is_std else raw
            sd      = scaler_stats.get(base, (0.0, 1.0))[1]
            orig    = float(coef) / sd if is_std and sd > 0 else float(coef)
            rows.append({"Component": "Upper (AADT-independent)",
                         "Variable": base, "Full Name": _label(base),
                         "Coefficient": round(float(coef), 5),
                         "CMF": round(float(np.exp(orig)), 4)})
        elif pname.startswith("lower::") and pname.endswith("*log_aadt"):
            raw    = pname.replace("lower::", "").replace("*log_aadt", "")
            is_std = raw.endswith("_Z")
            base   = raw[:-2] if is_std else raw
            sd     = scaler_stats.get(base, (0.0, 1.0))[1]
            orig   = float(coef) / sd if is_std and sd > 0 else float(coef)
            rows.append({"Component": "Lower (AADT interaction)",
                         "Variable": base, "Full Name": _label(base),
                         "Coefficient": round(float(coef), 5),
                         "CMF": "varies with AADT"})

    df = pd.DataFrame(rows)
    path.write_text(df.to_csv(index=False), encoding="utf-8")

    # Write Quarto-compatible pipe tables (no tabulate needed) ─────────────
    # Split into upper and lower tables to keep each slide uncluttered.

    def _pipe_table(sub: pd.DataFrame, cols: list[str]) -> str:
        """Build a plain-pipe Pandoc markdown table without tabulate."""
        sub = sub[cols].copy()
        # Format numeric columns
        for c in cols:
            sub[c] = sub[c].apply(
                lambda x: (f"{x:+.4f}" if isinstance(x, float) and np.isfinite(x)
                           else str(x) if not (isinstance(x, float)) else str(x))
            )
        widths = {c: max(len(c), sub[c].str.len().max()) for c in cols}
        header  = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
        divider = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
        data_rows = [
            "| " + " | ".join(str(row[c]).ljust(widths[c]) for c in cols) + " |"
            for _, row in sub.iterrows()
        ]
        return "\n".join([header, divider] + data_rows)

    upper_df = df[df["Component"].str.startswith("Core") |
                  df["Component"].str.startswith("Upper")].copy()
    upper_df["CMF"] = upper_df["CMF"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    )

    lower_df = df[df["Component"].str.startswith("Lower")].copy()
    lower_df["CMF"] = lower_df["CMF"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    )

    upper_cols = ["Variable", "Full Name", "Coefficient", "CMF"]
    lower_cols = ["Variable", "Full Name", "Coefficient"]

    (path.parent / "readable_coef_upper.md").write_text(
        _pipe_table(upper_df, upper_cols), encoding="utf-8"
    )
    (path.parent / "readable_coef_lower.md").write_text(
        _pipe_table(lower_df, lower_cols), encoding="utf-8"
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up and run a hierarchical CMF experiment with validation outputs and an interactive AADT dashboard."
    )
    parser.add_argument("--input", default="data/Ex-16-3.csv", help="Path to crash dataset CSV.")
    parser.add_argument("--output-dir", default="results/ex16_3_cmf", help="Output directory.")
    parser.add_argument("--y-col", default="FREQ", help="Crash frequency response column.")
    parser.add_argument("--aadt-col", default="AADT", help="AADT traffic-response column.")
    parser.add_argument("--offset-col", default="OFFSET", help="Offset column name to use if available.")
    parser.add_argument("--disable-offset", action="store_true", help="Disable offset during fitting.")
    parser.add_argument("--family", choices=["nb", "poisson"], default="nb",
                        help="Count family for final refit (default: nb).")
    parser.add_argument("--families", default=None,
                        help="Comma-separated families to search over, or 'both'/'all' "
                             "for nb+poisson. Overrides --family for the search phase. "
                             "Example: --families both  or  --families nb,poisson")

    parser.add_argument("--train-frac", type=float, default=0.60, help="Training split fraction.")
    parser.add_argument("--val-frac", type=float, default=0.20, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")

    parser.add_argument("--search-iter", type=int, default=300,
                        help="Random search iterations (default 300 — more = better BIC exploration).")
    parser.add_argument("--max-upper-terms", type=int, default=6,
                        help="Max upper-level terms per candidate (default 6).")
    parser.add_argument("--max-lower-terms", type=int, default=4,
                        help="Max lower-level terms per candidate (default 4).")
    parser.add_argument(
        "--candidate-profile",
        choices=["core", "expanded"],
        default="core",
        help="Default candidate pool profile when --upper-vars/--lower-vars are not provided.",
    )
    parser.add_argument("--upper-vars", default=None, help="Comma-separated upper-level candidate variables.")
    parser.add_argument("--lower-vars", default=None, help="Comma-separated lower-level candidate variables.")
    parser.add_argument(
        "--enforce-aadt-increase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require positive local AADT elasticity during candidate selection (default: enabled).",
    )
    parser.add_argument(
        "--min-aadt-elasticity",
        type=float,
        default=0.0,
        help="Minimum accepted local elasticity for AADT effect.",
    )
    parser.add_argument(
        "--allow-nonmonotonic-fallback",
        action="store_true",
        help="If set, fall back to best unconstrained model when no monotonic candidate exists.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = input_path.stem.replace("_", " ")

    raw = pd.read_csv(input_path)
    df = _prepare_washington_df(raw, aadt_col=args.aadt_col, y_col=args.y_col)

    auto_upper, auto_lower = _resolve_default_candidates(df, profile=args.candidate_profile)
    upper_raw = _parse_csv_list(args.upper_vars) or auto_upper
    lower_raw = _parse_csv_list(args.lower_vars) or auto_lower

    upper_raw = [v for v in upper_raw if v in df.columns]
    lower_raw = [v for v in lower_raw if v in df.columns]

    if not upper_raw or not lower_raw:
        raise ValueError("No valid upper/lower candidate variables found after filtering.")

    binary_vars = {v for v in set(upper_raw + lower_raw) if _is_binary(df[v])}
    continuous_vars = sorted([v for v in set(upper_raw + lower_raw) if v not in binary_vars])

    train_idx, val_idx, test_idx = _split_indices(len(df), args.train_frac, args.val_frac, args.seed)
    df_train_raw = df.iloc[train_idx].reset_index(drop=True)
    df_val_raw = df.iloc[val_idx].reset_index(drop=True)
    df_test_raw = df.iloc[test_idx].reset_index(drop=True)

    scaler_train = _build_scaler_stats(df_train_raw, continuous_vars)
    df_train = _apply_standardization(df_train_raw, scaler_train)
    df_val = _apply_standardization(df_val_raw, scaler_train)
    df_test = _apply_standardization(df_test_raw, scaler_train)

    model_name_map = {v: (v if v in binary_vars else f"{v}_Z") for v in set(upper_raw + lower_raw)}
    upper_model_vars = [model_name_map[v] for v in upper_raw]
    lower_model_vars = [model_name_map[v] for v in lower_raw]

    offset_col = None if args.disable_offset else (args.offset_col if args.offset_col in df_train.columns else None)

    # Build family list from --families argument (e.g. "nb", "poisson", "both")
    _fam_arg = getattr(args, "families", args.family)
    if _fam_arg in ("both", "all"):
        search_families = ["nb", "poisson"]
    elif "," in str(_fam_arg):
        search_families = [f.strip() for f in str(_fam_arg).split(",")]
    else:
        search_families = [str(_fam_arg)]

    best_fit_train, search_history, search_history_ordered, top_k_fits = _random_search(
        df_train=df_train,
        df_val=df_val,
        aadt_col=args.aadt_col,
        y_col=args.y_col,
        upper_candidates=upper_model_vars,
        lower_candidates=lower_model_vars,
        families=search_families,
        offset_col=offset_col,
        search_iter=int(args.search_iter),
        max_upper_terms=int(args.max_upper_terms),
        max_lower_terms=int(args.max_lower_terms),
        seed=int(args.seed),
        enforce_aadt_increase=bool(args.enforce_aadt_increase),
        min_aadt_elasticity=float(args.min_aadt_elasticity),
        allow_nonmonotonic_fallback=bool(args.allow_nonmonotonic_fallback),
    )

    pred_train = _predict(df_train, best_fit_train)
    pred_val = _predict(df_val, best_fit_train)

    y_train = pd.to_numeric(df_train[args.y_col], errors="coerce").to_numpy(dtype=float)
    y_val = pd.to_numeric(df_val[args.y_col], errors="coerce").to_numpy(dtype=float)

    metrics_train = _metrics(y_train, pred_train)
    metrics_val = _metrics(y_val, pred_val)

    # Final refit on train+validation, then test on held-out test split.
    df_trainval_raw = pd.concat([df_train_raw, df_val_raw], axis=0, ignore_index=True)
    scaler_trainval = _build_scaler_stats(df_trainval_raw, continuous_vars)
    df_trainval = _apply_standardization(df_trainval_raw, scaler_trainval)
    df_test_final = _apply_standardization(df_test_raw, scaler_trainval)

    final_fit = _fit_model(
        df_train=df_trainval,
        aadt_col=args.aadt_col,
        y_col=args.y_col,
        upper_vars=best_fit_train.upper_vars,
        lower_vars=best_fit_train.lower_vars,
        family=args.family,
        offset_col=offset_col,
    )
    if final_fit is None:
        raise RuntimeError("Final model fit failed on train+validation data.")

    pred_trainval = _predict(df_trainval, final_fit)
    pred_test = _predict(df_test_final, final_fit)

    y_trainval = pd.to_numeric(df_trainval[args.y_col], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(df_test_final[args.y_col], errors="coerce").to_numpy(dtype=float)

    metrics_trainval = _metrics(y_trainval, pred_trainval)
    metrics_test = _metrics(y_test, pred_test)
    elasticity_trainval = _elasticity_stats(_aadt_elasticity(df_trainval, final_fit))
    elasticity_test = _elasticity_stats(_aadt_elasticity(df_test_final, final_fit))

    selected_upper_raw = sorted(
        {name[:-2] if name.endswith("_Z") else name for name in best_fit_train.upper_vars}
    )
    selected_lower_raw = sorted(
        {name[:-2] if name.endswith("_Z") else name for name in best_fit_train.lower_vars}
    )
    selected_raw = sorted(set(selected_upper_raw + selected_lower_raw))

    coef_df = _coefficient_report(final_fit, scaler_trainval, binary_vars=binary_vars)

    metrics_df = pd.DataFrame(
        [
            {"Split": "train (selection fit)", **metrics_train},
            {"Split": "validation (selection fit)", **metrics_val},
            {"Split": "train+validation (final fit)", **metrics_trainval},
            {"Split": "test (held out)", **metrics_test},
        ]
    )

    # ── Benchmark comparison ──────────────────────────────────────────────
    bench = _fit_benchmark(
        df_tv=df_trainval, df_test=df_test_final,
        y_col=args.y_col, aadt_col=args.aadt_col,
        offset_col=offset_col, family=args.family,
    )
    bench_compare_df = pd.DataFrame([
        {
            "Model":           "Benchmark (AADT-only SPF)",
            "BIC":             round(bench.get("tv_bic", float("nan")), 2),
            "AIC":             round(bench.get("tv_aic", float("nan")), 2),
            "Test RMSE":       round(bench.get("test_rmse", float("nan")), 4),
            "Test Poisson Dev":round(bench.get("test_poisson_dev", float("nan")), 2),
            "Test R2":         round(bench.get("test_r2", float("nan")), 4),
        },
        {
            "Model":           "Hierarchical CMF (selected)",
            "BIC":             round(float(getattr(final_fit.result, "bic", np.nan)), 2),
            "AIC":             round(float(getattr(final_fit.result, "aic", np.nan)), 2),
            "Test RMSE":       round(metrics_test["rmse"], 4),
            "Test Poisson Dev":round(_poisson_deviance(y_test, pred_test), 2),
            "Test R2":         round(metrics_test["r2"], 4),
        },
    ])
    print("\n  BENCHMARK VS HIERARCHICAL CMF:")
    print(bench_compare_df.to_string(index=False))

    settings_df = pd.DataFrame(
        [
            {"Setting": "Rows", "Value": int(len(df))},
            {"Setting": "Train rows", "Value": int(len(df_train_raw))},
            {"Setting": "Validation rows", "Value": int(len(df_val_raw))},
            {"Setting": "Test rows", "Value": int(len(df_test_raw))},
            {"Setting": "Search iterations", "Value": int(args.search_iter)},
            {"Setting": "Candidate profile", "Value": args.candidate_profile},
            {"Setting": "Family", "Value": args.family},
            {"Setting": "AADT column", "Value": args.aadt_col},
            {"Setting": "Offset used", "Value": "yes" if offset_col else "no"},
            {"Setting": "Enforce AADT increase", "Value": "yes" if args.enforce_aadt_increase else "no"},
            {"Setting": "Min AADT elasticity", "Value": float(args.min_aadt_elasticity)},
            {"Setting": "Allow nonmonotonic fallback", "Value": "yes" if args.allow_nonmonotonic_fallback else "no"},
            {"Setting": "Upper candidate count", "Value": int(len(upper_raw))},
            {"Setting": "Lower candidate count", "Value": int(len(lower_raw))},
            {"Setting": "Selected upper vars", "Value": ", ".join(selected_upper_raw) if selected_upper_raw else "(none)"},
            {"Setting": "Selected lower vars", "Value": ", ".join(selected_lower_raw) if selected_lower_raw else "(none)"},
        ]
    )

    elasticity_df = pd.DataFrame(
        [
            {"Split": "train+validation", **elasticity_trainval},
            {"Split": "test", **elasticity_test},
        ]
    )

    calibration_test_df = _calibration_table(y_test, pred_test, n_bins=10)
    search_top_df = search_history.head(25).copy()

    _obs_pred_plot(output_dir / "validation_observed_vs_predicted.png", y_val, pred_val, "Validation: observed vs predicted")
    _obs_pred_plot(output_dir / "test_observed_vs_predicted.png", y_test, pred_test, "Test: observed vs predicted")

    # Build scenario payload and requested crash-risk sensitivity visualizations.
    scenario_columns = list(df_trainval_raw.columns)
    payload = _dashboard_payload(
        fitted=final_fit,
        df_reference_raw=df_trainval_raw,
        all_columns=scenario_columns,
        selected_raw_vars=selected_raw,
        binary_vars=binary_vars,
        scaler_stats=scaler_trainval,
        aadt_col=args.aadt_col,
    )
    interactive_payload = _interactive_model_payload(
        fitted=final_fit,
        df_reference_raw=df_trainval_raw,
        selected_raw_vars=selected_raw,
        binary_vars=binary_vars,
        scaler_stats=scaler_trainval,
        aadt_col=args.aadt_col,
    )

    curve_var = "CURVES" if "CURVES" in payload["variables"] else ("curve" if "curve" in payload["variables"] else None)
    if curve_var is not None:
        _save_curve_plot(
            output_dir / "curve_crash_risk_sensitivity.png",
            payload=payload,
            variable=curve_var,
            aadt_levels=["Low AADT", "Median AADT", "High AADT"],
            title=f"Crash-risk change as {curve_var} changes",
            ylabel="Predicted crashes",
            mode="pred",
        )
        _save_curve_plot(
            output_dir / "curve_cmf_sensitivity.png",
            payload=payload,
            variable=curve_var,
            aadt_levels=["Low AADT", "Median AADT", "High AADT"],
            title=f"CMF change as {curve_var} changes",
            ylabel="CMF (relative to profile baseline)",
            mode="cmf",
        )

    _save_aadt_curve_plot(
        output_dir / "aadt_crash_risk_sensitivity.png",
        fitted=final_fit,
        df_reference_raw=df_trainval_raw,
        all_columns=scenario_columns,
        selected_raw_vars=selected_raw,
        binary_vars=binary_vars,
        scaler_stats=scaler_trainval,
        aadt_col=args.aadt_col,
    )

    selected_binary = [v for v in selected_raw if v in binary_vars]
    if selected_binary:
        _save_binary_bar(output_dir / "binary_toggle_cmf_effects.png", payload, selected_binary)

    _save_dashboard_html(output_dir / "hierarchical_cmf_dashboard.html", interactive_payload, dataset_label)

    # --- New outputs ---

    # ── Random-parameters sweep over top-K specs ─────────────────────────
    # For each of the top-K fixed-effects models found by the search, attempt
    # a random-params NB2 refit.  The spec whose random-params BIC is lowest
    # becomes the "promoted" final model (reported instead of fixed-effects).
    best_rp_result: dict | None = None
    best_rp_bic = float("nan")

    print(f"  Running random-params sweep on {len(top_k_fits)} top candidates ...")
    for _rp_candidate in top_k_fits:
        _cand_upper = sorted({n[:-2] if n.endswith("_Z") else n for n in _rp_candidate.upper_vars})
        _cand_lower = sorted({n[:-2] if n.endswith("_Z") else n for n in _rp_candidate.lower_vars})
        _rp = _jax_random_params_refit(
            df_trainval_raw=df_trainval_raw,
            best_upper_raw=_cand_upper,
            best_lower_raw=_cand_lower,
            y_col=args.y_col,
            aadt_col=args.aadt_col,
            offset_col=offset_col,
            scaler_stats=scaler_trainval,
            binary_vars=binary_vars,
        )
        if _rp is None:
            continue
        _rp_bic = _rp.get("bic", float("nan"))
        if not np.isfinite(_rp_bic):
            continue
        if best_rp_result is None or _rp_bic < best_rp_bic:
            best_rp_bic    = _rp_bic
            best_rp_result = _rp
            # Promote the best fixed-effects candidate to match this random spec
            selected_upper_raw = _cand_upper
            selected_lower_raw = _cand_lower

    # Use the best random-params result (from sweep) as jax_result
    jax_result = best_rp_result
    if jax_result is None:
        # Fallback: try random-params on the primary selected spec
        jax_result = _jax_random_params_refit(
            df_trainval_raw=df_trainval_raw,
            best_upper_raw=selected_upper_raw,
            best_lower_raw=selected_lower_raw,
            y_col=args.y_col,
            aadt_col=args.aadt_col,
            offset_col=offset_col,
            scaler_stats=scaler_trainval,
            binary_vars=binary_vars,
        )
    if jax_result is not None:
        try:
            coef_str  = jax_result.get("coef_str", "(unavailable)")
            corr_str  = jax_result.get("corr_matrix_str", "")
            rp_ll     = jax_result.get("loglik", float("nan"))
            rp_bic    = jax_result.get("bic",    float("nan"))

            print("\n" + "="*72)
            print("  RANDOM-PARAMETERS NB2 — FINAL COEFFICIENTS")
            print(f"  LL={rp_ll:.2f}   BIC={rp_bic:.2f}")
            print("="*72)
            print(coef_str)
            if corr_str:
                print()
                print(corr_str)
            print()
            print("  INTERPRETATION:")
            print("    Fixed         — same effect for every road segment")
            print("    Random-Ind    — Mean = population average; SD = site-to-site variability")
            print("    Random-Cor    — correlated pair; correlation matrix shows co-movement")
            print("    A large SD means the effect differs substantially across segments")
            print("="*72 + "\n")

            rp_txt = (
                f"Random-parameters NB2  |  LL={rp_ll:.2f}  BIC={rp_bic:.2f}\n\n"
                "COEFFICIENT TABLE\n"
                "  Fixed:       same value for every segment\n"
                "  Random-Ind:  independently distributed across segments (Mean +/- SD)\n"
                "  Random-Cor:  jointly distributed (see correlation matrix below)\n\n"
                + coef_str
                + ("\n\n" + corr_str if corr_str else "")
                + "\n\nINTERPRETATION\n"
                "  A large SD on a variable means its effect varies substantially\n"
                "  across road segments — some sites respond much more (or less)\n"
                "  to that characteristic than the population average.\n"
                "  The correlation matrix shows whether two variables' effects\n"
                "  tend to move together across sites."
            )

            (output_dir / "random_params_note.md").write_text(rp_txt, encoding="utf-8")
            (output_dir / "random_params_summary.json").write_text(
                json.dumps({"summary": rp_txt, "loglik": rp_ll, "bic": rp_bic}, indent=2),
                encoding="utf-8",
            )
            print("  random_params_summary.json written.")

            # Use random-params predictions for the AADT obs/pred plot
            rp_fit = jax_result.get("fit")
            if rp_fit is not None:
                rp_preds = np.asarray(rp_fit.get("predictions", [])).squeeze()
                if rp_preds.shape == (len(df_trainval_raw),):
                    pred_all_rp_tv = rp_preds
                    # For full-dataset predictions, fall back to fixed-effects
                    # (random-params builder only covers trainval)
        except Exception as exc:
            print(f"[WARNING] Could not write random_params_summary.json: {exc}")
    else:
        # Write a placeholder so the QMD include never fails
        note = (
            "**Note:** The search finds the best *fixed-effects* NB2 specification "
            "(which variables to include and whether they enter as upper or lower terms). "
            "After the best specification is selected, a random-parameters refit "
            "is attempted using the MetaCount JAX engine — giving each site its own "
            "draw of the continuous predictor coefficients, capturing unobserved "
            "heterogeneity across road segments.\n\n"
            "The random-parameters model (`random_params_summary.json`) was not "
            "generated this run because the `ExperimentBuilder` dependency was not "
            "available in the subprocess path. Run from within the package directory "
            "for the full mixed-model refit."
        )
        (output_dir / "random_params_note.md").write_text(note, encoding="utf-8")

    # Convergence HTML (uses original iteration order from search_history)
    # Reconstruct in-order history from the sorted df by sorting on Iteration
    # Pass iteration-ordered history so the convergence trace is meaningful
    _save_convergence_html(output_dir / "search_convergence.html",
                           search_history_ordered, dataset_label)
    _save_convergence_png(output_dir / "search_convergence.png", search_history_ordered, dataset_label)
    print("  search_convergence.html / .png written.")

    # AADT obs/pred HTML + PNG over full dataset
    df_all_std = _apply_standardization(df, scaler_trainval)
    pred_all = _predict(df_all_std, final_fit)
    y_all = pd.to_numeric(df[args.y_col], errors="coerce").to_numpy(float)
    aadt_all = pd.to_numeric(df[args.aadt_col], errors="coerce").to_numpy(float)

    split_label_arr = np.array(["test"] * len(df), dtype=object)
    split_label_arr[train_idx] = "train"
    split_label_arr[val_idx] = "val"
    split_labels_list = split_label_arr.tolist()

    _aadt_title = f"{dataset_label} — Observed vs Predicted by AADT"
    _save_obs_pred_aadt_html(
        output_dir / "aadt_obs_pred.html",
        df_all=df, y_true=y_all, y_pred=pred_all, aadt=aadt_all,
        title=_aadt_title, split_labels=split_labels_list,
    )
    _save_obs_pred_aadt_png(
        output_dir / "aadt_obs_pred.png",
        y_true=y_all, y_pred=pred_all, aadt=aadt_all,
        title=_aadt_title, split_labels=split_labels_list,
    )
    print("  aadt_obs_pred.html / .png written.")

    # Comparison plot: observed vs benchmark vs hierarchical on same axes
    if bench and bench.get("fitted") is not None:
        pred_bench_all = _predict(
            _apply_standardization(df, scaler_trainval), bench["fitted"]
        )
        _comp_title = f"{dataset_label} — Observed vs Benchmark vs Hierarchical"
        _save_model_comparison_html(
            output_dir / "model_comparison.html",
            y_all=y_all, aadt_all=aadt_all,
            pred_benchmark=pred_bench_all,
            pred_hierarchical=pred_all,
            split_labels=split_labels_list,
            title=_comp_title,
        )
        _save_model_comparison_png(
            output_dir / "model_comparison.png",
            y_all=y_all, aadt_all=aadt_all,
            pred_benchmark=pred_bench_all,
            pred_hierarchical=pred_all,
            split_labels=split_labels_list,
            title=_comp_title,
        )
        print("  model_comparison.html / .png written.")

    # Readable coefficient CSV (human-readable names + CMF)
    readable_coef_df = _save_readable_coef_csv(
        output_dir / "readable_coefficients.csv", final_fit, scaler_trainval, binary_vars
    )

    # Benchmark comparison CSV + markdown include
    bench_compare_df.to_csv(output_dir / "benchmark_comparison.csv", index=False)
    # Build pipe-table for QMD include
    def _pipe(df: pd.DataFrame) -> str:
        cols   = list(df.columns)
        widths = {c: max(len(c), df[c].astype(str).str.len().max()) for c in cols}
        header  = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
        divider = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
        rows    = ["| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |"
                   for _, r in df.iterrows()]
        return "\n".join([header, divider] + rows)
    (output_dir / "benchmark_comparison.md").write_text(
        _pipe(bench_compare_df), encoding="utf-8"
    )

    # Write tabular outputs.
    (output_dir / "search_history_full.csv").write_text(search_history.to_csv(index=False), encoding="utf-8")
    (output_dir / "search_history_top25.md").write_text(_to_markdown(search_top_df), encoding="utf-8")
    (output_dir / "search_history_top25.csv").write_text(search_top_df.to_csv(index=False), encoding="utf-8")

    (output_dir / "model_settings.md").write_text(_to_markdown(settings_df), encoding="utf-8")
    (output_dir / "validation_metrics.md").write_text(_to_markdown(metrics_df), encoding="utf-8")
    (output_dir / "validation_metrics.csv").write_text(metrics_df.to_csv(index=False), encoding="utf-8")

    (output_dir / "test_calibration_deciles.md").write_text(_to_markdown(calibration_test_df), encoding="utf-8")
    (output_dir / "test_calibration_deciles.csv").write_text(calibration_test_df.to_csv(index=False), encoding="utf-8")
    (output_dir / "aadt_monotonicity_diagnostics.md").write_text(_to_markdown(elasticity_df), encoding="utf-8")
    (output_dir / "aadt_monotonicity_diagnostics.csv").write_text(elasticity_df.to_csv(index=False), encoding="utf-8")

    (output_dir / "coefficients_standardized_and_original.md").write_text(_to_markdown(coef_df), encoding="utf-8")
    (output_dir / "coefficients_standardized_and_original.csv").write_text(coef_df.to_csv(index=False), encoding="utf-8")

    # ── Human-readable model summary printed to console and saved ──────────
    _print_readable_model(final_fit, scaler_trainval, binary_vars,
                          output_dir / "model_summary_readable.txt")

    scaler_rows = [
        {"Variable": var, "Mean (train/final)": mu, "Std (train/final)": sd}
        for var, (mu, sd) in sorted(scaler_trainval.items())
    ]
    scaler_df = pd.DataFrame(scaler_rows)
    (output_dir / "standardization_reference.csv").write_text(scaler_df.to_csv(index=False), encoding="utf-8")

    model_spec = {
        "family": args.family,
        "aadt_col": args.aadt_col,
        "y_col": args.y_col,
        "offset_col": offset_col,
        "selected_upper_model_vars": best_fit_train.upper_vars,
        "selected_lower_model_vars": best_fit_train.lower_vars,
        "selected_upper_raw_vars": selected_upper_raw,
        "selected_lower_raw_vars": selected_lower_raw,
        "coefficients": {k: float(v) for k, v in pd.Series(final_fit.result.params).to_dict().items()},
        "aadt_monotonicity": {
            "trainval": elasticity_trainval,
            "test": elasticity_test,
        },
    }
    (output_dir / "final_model_spec.json").write_text(json.dumps(model_spec, indent=2), encoding="utf-8")
    with (output_dir / "final_model_fit.pkl").open("wb") as f:
        pickle.dump(final_fit, f)
    with (output_dir / "selection_model_fit.pkl").open("wb") as f:
        pickle.dump(best_fit_train, f)

    summary_lines = [
        f"{dataset_label} Hierarchical CMF Experiment",
        "",
        "Configuration",
        _to_markdown(settings_df),
        "",
        "Validation and held-out crash-frequency metrics",
        _to_markdown(metrics_df),
        "",
        "Test calibration by predicted decile",
        _to_markdown(calibration_test_df),
        "",
        "Outputs",
        "- search_history_full.csv",
        "- search_history_top25.md / .csv",
        "- model_settings.md",
        "- validation_metrics.md / .csv",
        "- test_calibration_deciles.md / .csv",
        "- aadt_monotonicity_diagnostics.md / .csv",
        "- coefficients_standardized_and_original.md / .csv",
        "- standardization_reference.csv",
        "- final_model_spec.json",
        "- final_model_fit.pkl",
        "- selection_model_fit.pkl",
        "- validation_observed_vs_predicted.png",
        "- test_observed_vs_predicted.png",
        "- aadt_crash_risk_sensitivity.png",
        "- curve_crash_risk_sensitivity.png (if curve selected)",
        "- curve_cmf_sensitivity.png (if curve selected)",
        "- binary_toggle_cmf_effects.png (if binary vars selected)",
        "- hierarchical_cmf_dashboard.html",
        "- search_convergence.html",
        "- aadt_obs_pred.html",
        "- random_params_summary.json (if ExperimentBuilder available)",
    ]
    (output_dir / "hierarchical_cmf_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Assets written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
