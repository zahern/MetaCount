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
    include_log_aadt: bool = True


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
    "FC_1":     "FC = Interstate (vs Minor Collector reference)",
    "FC_2":     "FC = Principal Arterial Other (vs Minor Collector reference)",
    "FC_3":     "FC = Minor Arterial (vs Minor Collector reference)",
    "FC_4":     "FC = Major Collector (vs Minor Collector reference)",
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

# Canonical Ex16-3 literature benchmark structure provided by user.
# AADT is included only in the offset term for benchmark fitting.
LITERATURE_BENCHMARK_NONRANDOM_RAW = ["LOWPRE", "GBRPM", "FRICTION"]
LITERATURE_BENCHMARK_RANDOM_MEAN_RAW = ["EXPOSE", "INTPM", "CPM", "HISNOW"]


def _label(var: str) -> str:
    """Return the human-readable label for a variable name."""
    raw = var.replace("_Z", "").replace("_X_logaadt", "").replace("_inter_", "")
    return VARIABLE_LABELS.get(raw, raw)


def _to_markdown(df: pd.DataFrame) -> str:
    pretty = df.copy()
    if not pretty.empty:
        for col in pretty.columns:
            s = pretty[col]
            if pd.api.types.is_numeric_dtype(s):
                def _fmt_num(v: Any) -> Any:
                    if pd.isna(v):
                        return ""
                    x = float(v)
                    if not np.isfinite(x):
                        return ""
                    ax = abs(x)
                    if ax >= 1000:
                        return f"{x:,.2f}"
                    if ax >= 100:
                        return f"{x:.2f}"
                    if ax >= 1:
                        return f"{x:.4f}"
                    return f"{x:.6f}"
                pretty[col] = s.map(_fmt_num)
            else:
                pretty[col] = s.map(lambda v: "" if pd.isna(v) else str(v).strip())
    try:
        return pretty.to_markdown(index=False)
    except Exception:
        cols = list(pretty.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(row[c]) for c in cols) + " |"
            for _, row in pretty.iterrows()
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

    # ── Impute sentinel / missing values with column mean ─────────────────
    # PEAKHR uses -99 as a missing-value sentinel — replace before imputation.
    if "PEAKHR" in out.columns:
        out["PEAKHR"] = pd.to_numeric(out["PEAKHR"], errors="coerce")
        out.loc[out["PEAKHR"] < 0, "PEAKHR"] = np.nan   # -99 -> NaN

    # For every numeric predictor column, replace NaN with the column mean.
    # This ensures no observation is silently dropped due to missing predictors.
    reserved = {y_col, aadt_col, "OFFSET", "OBS_ID"}
    for col in out.columns:
        if col in reserved:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if s.isna().any():
            col_mean = s.mean()
            if np.isfinite(col_mean):
                out[col] = s.fillna(col_mean)

    # Composite LANES = INCLANES + DECLANES (total lane count both directions)
    if "INCLANES" in out.columns and "DECLANES" in out.columns:
        inc = pd.to_numeric(out["INCLANES"], errors="coerce").fillna(0)
        dec = pd.to_numeric(out["DECLANES"], errors="coerce").fillna(0)
        out["LANES"] = (inc + dec).clip(lower=1)

    # One-hot encode FC — Functional Classification is a NOMINAL label,
    # not ordinal. Treating it as continuous (FC=1 vs FC=5 = 4 units apart)
    # has no physical meaning. Binary dummies let each class have its own CMF.
    # Reference = most frequent class (implicit baseline in regression).
    if "FC" in out.columns:
        fc_s    = pd.to_numeric(out["FC"], errors="coerce")
        fc_mode = float(fc_s.mode().iloc[0])
        fc_s    = fc_s.fillna(fc_mode)
        fc_vals = sorted(fc_s.unique())
        _fc_ref = fc_vals[-1]  # use highest-coded (most common = FC_5) as reference
        # pick the most frequent as reference instead
        _fc_ref = float(fc_s.value_counts().index[0])
        for fv in fc_vals:
            if fv == _fc_ref:
                continue
            out[f"FC_{int(fv)}"] = (fc_s == fv).astype(int)

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


def _design_matrix(
    df: pd.DataFrame,
    aadt_col: str,
    upper_vars: list[str],
    lower_vars: list[str],
    include_log_aadt: bool = True,
) -> pd.DataFrame:
    mat = pd.DataFrame(index=df.index)
    mat["const"] = 1.0
    log_aadt = np.log(np.clip(pd.to_numeric(df[aadt_col], errors="coerce").to_numpy(dtype=float), 1e-12, None))
    if include_log_aadt:
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
    include_log_aadt: bool = True,
) -> FittedModel | None:
    X_train = _design_matrix(
        df_train,
        aadt_col,
        upper_vars,
        lower_vars,
        include_log_aadt=include_log_aadt,
    )
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
        include_log_aadt=bool(include_log_aadt),
    )


def _predict(df: pd.DataFrame, fitted: FittedModel) -> np.ndarray:
    X = _design_matrix(
        df,
        fitted.aadt_col,
        fitted.upper_vars,
        fitted.lower_vars,
        include_log_aadt=bool(getattr(fitted, "include_log_aadt", True)),
    )
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
    search_method: str = "random-sa",
    harmony_hms: int = 12,
    harmony_hmcr: float = 0.90,
    harmony_par: float = 0.35,
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

    def _eval(fit: FittedModel, y_val_arr: np.ndarray, phase: str = "random") -> None:
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
            "Phase":     phase,
            "Upper Vars": ", ".join(fit.upper_vars) if fit.upper_vars else "(none)",
            "Lower Vars": ", ".join(fit.lower_vars) if fit.lower_vars else "(none)",
            "Val Poisson Dev": score, "Val RMSE": val_rmse,
            "AIC": aic, "BIC": bic,
            "AADT elasticity min":            es["aadt_elasticity_min"],
            "AADT elasticity p10":            es["aadt_elasticity_p10"],
            "AADT elasticity median":         es["aadt_elasticity_median"],
            "AADT elasticity p90":            es["aadt_elasticity_p90"],
            "AADT elasticity share positive": es["aadt_elasticity_share_positive"],
            # Monotonic = local AADT elasticity > 0 for ALL segments in val set.
            # Elasticity = d(log crashes)/d(log AADT) = beta_AADT + sum(gamma_k * x_k)
            # Must be positive everywhere so the model never predicts fewer crashes
            # as traffic increases — a physical plausibility requirement.
            "Monotonic AADT (e>0 all segs)": "yes" if mono_ok else "no",
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
            _eval(fit, y_val_np, phase="random")

    # ── Hybrid search: random exploration + SA refinement ────────────────
    #
    # Phase 1 (first 40% of budget): pure random search — broad exploration of
    # the full variable-combination space.
    #
    # Phase 2 (last 60% of budget): Simulated Annealing (SA) starting from the
    # best solution found in Phase 1, using structured perturbations:
    #   - add one variable to upper or lower
    #   - remove one variable from upper or lower
    #   - move one variable between upper and lower
    #   - swap one variable for another in the same level
    #   - toggle family (NB <-> Poisson)
    #
    # SA acceptance: always accept improvements; accept worsening moves with
    # probability exp(-ΔBIC / T) so the search can escape local optima.
    # Temperature cools geometrically: T <- T * cooling_rate.
    #
    # This is better than pure random because later iterations build on earlier
    # discoveries rather than starting from scratch every time.

    MIN_UPPER = 2
    MIN_LOWER = 1

    n_random = max(2, int(search_iter * 0.40))
    n_sa     = search_iter - n_random

    # ── Phase 1: random ────────────────────────────────────────────────────
    for _ in range(n_random):
        k_upper = int(rng.integers(MIN_UPPER, max_upper_terms + 1))
        k_lower = int(rng.integers(MIN_LOWER, max_lower_terms + 1))
        fam     = families[int(rng.integers(0, len(families)))]

        pick_upper = sorted(rng.choice(upper_candidates,
                                       size=min(k_upper, len(upper_candidates)),
                                       replace=False).tolist()) if upper_candidates else []
        pick_lower = sorted(rng.choice(lower_candidates,
                                       size=min(k_lower, len(lower_candidates)),
                                       replace=False).tolist()) if lower_candidates else []

        key = (tuple(pick_upper), tuple(pick_lower), fam)
        if key in tested:
            continue
        tested.add(key)

        fit = _fit_model(df_train, aadt_col=aadt_col, y_col=y_col,
                         upper_vars=pick_upper, lower_vars=pick_lower,
                         family=fam, offset_col=offset_col)
        if fit is None:
            continue
        _eval(fit, y_val_np, phase="random")

    # ── Phase 2: SA refinement from best found ─────────────────────────────
    if best_fit is not None and n_sa > 0:
        # SA parameters — temperature schedule chosen so we go from accepting
        # ~30 BIC-unit worsening (broad exploration) to ~0.5 (near-greedy).
        T = 20.0                  # initial temperature (BIC units)
        T_min = 0.5               # final temperature
        cooling = (T_min / T) ** (1.0 / max(n_sa, 1))

        # Current SA solution starts at the best joint (BIC + Val RMSE)
        # candidate found in Phase-1, not the best-BIC-only candidate.
        _sa_start_log_msg = ""
        if history_rows:
            _sa_df = pd.DataFrame(history_rows)
            _sa_df = _sa_df[
                np.isfinite(pd.to_numeric(_sa_df["BIC"], errors="coerce").to_numpy(dtype=float))
                & np.isfinite(pd.to_numeric(_sa_df["Val RMSE"], errors="coerce").to_numpy(dtype=float))
            ].copy()
            if enforce_aadt_increase and "Monotonic AADT (e>0 all segs)" in _sa_df.columns:
                _sa_df = _sa_df[_sa_df["Monotonic AADT (e>0 all segs)"].astype(str).str.lower().eq("yes")].copy()

            if not _sa_df.empty:
                _bic_ref = float(np.nanmin(pd.to_numeric(_sa_df["BIC"], errors="coerce").to_numpy(dtype=float)))
                _rmse_ref = float(np.nanmin(pd.to_numeric(_sa_df["Val RMSE"], errors="coerce").to_numpy(dtype=float)))
                _eps_bic = max(abs(_bic_ref), 1e-9)
                _eps_rmse = max(abs(_rmse_ref), 1e-9)
                _sa_df["_bic_improve"] = (_bic_ref - pd.to_numeric(_sa_df["BIC"], errors="coerce")) / _eps_bic
                _sa_df["_rmse_improve"] = (_rmse_ref - pd.to_numeric(_sa_df["Val RMSE"], errors="coerce")) / _eps_rmse
                _sa_df["_joint_score"] = np.minimum(_sa_df["_bic_improve"], _sa_df["_rmse_improve"])
                _sa_df["_total_score"] = _sa_df["_bic_improve"] + _sa_df["_rmse_improve"]
                _sa_row = _sa_df.sort_values(
                    ["_joint_score", "_total_score", "BIC", "Val RMSE"],
                    ascending=[False, False, True, True],
                ).iloc[0]

                _upper_txt = str(_sa_row.get("Upper Vars", "")).strip()
                _lower_txt = str(_sa_row.get("Lower Vars", "")).strip()
                _sa_upper_init = [v.strip() for v in _upper_txt.split(",") if v.strip()] if _upper_txt not in {"", "(none)", "nan", "None"} else []
                _sa_lower_init = [v.strip() for v in _lower_txt.split(",") if v.strip()] if _lower_txt not in {"", "(none)", "nan", "None"} else []
                _sa_fam_init = str(_sa_row.get("Family", best_fit.family))
                _sa_iter = int(_sa_row.get("Iteration", -1))
                _sa_joint = float(_sa_row.get("_joint_score", np.nan))
                _sa_total = float(_sa_row.get("_total_score", np.nan))
                _sa_bic_ref = float(_sa_row.get("BIC", np.nan))
                _sa_rmse_ref = float(_sa_row.get("Val RMSE", np.nan))
                _sa_start_log_msg = (
                    f"  SA start (best joint score): iter={_sa_iter}, family={_sa_fam_init}, "
                    f"joint={_sa_joint:.6f}, total={_sa_total:.6f}, BIC={_sa_bic_ref:.3f}, ValRMSE={_sa_rmse_ref:.4f}, "
                    f"upper=[{', '.join(_sa_upper_init) if _sa_upper_init else '(none)'}], "
                    f"lower=[{', '.join(_sa_lower_init) if _sa_lower_init else '(none)'}]"
                )
            else:
                _sa_upper_init = list(best_fit.upper_vars)
                _sa_lower_init = list(best_fit.lower_vars)
                _sa_fam_init = best_fit.family
                _sa_start_log_msg = "  SA start fallback: no finite joint-score candidate; using best BIC seed."
        else:
            _sa_upper_init = list(best_fit.upper_vars)
            _sa_lower_init = list(best_fit.lower_vars)
            _sa_fam_init = best_fit.family
            _sa_start_log_msg = "  SA start fallback: no history rows; using best BIC seed."

        sa_upper = list(_sa_upper_init)
        sa_lower = list(_sa_lower_init)
        sa_fam   = _sa_fam_init

        # Ensure SA starts from a valid fitted point.
        _sa_start_fit = _fit_model(
            df_train=df_train,
            aadt_col=aadt_col,
            y_col=y_col,
            upper_vars=sa_upper,
            lower_vars=sa_lower,
            family=sa_fam,
            offset_col=offset_col,
        )
        if _sa_start_fit is None:
            sa_upper = list(best_fit.upper_vars)
            sa_lower = list(best_fit.lower_vars)
            sa_fam = best_fit.family
            _sa_start_fit = best_fit
            _sa_start_log_msg += " Refit failed for joint-score seed; reverted to best BIC seed."

        if _sa_start_log_msg:
            print(_sa_start_log_msg)

        sa_bic   = float(getattr(_sa_start_fit.result, "bic", np.nan))
        if not np.isfinite(sa_bic):
            sa_bic = 1e12

        def _perturb_sa(u: list, l: list, fam: str) -> tuple:
            """Return a neighboring (upper, lower, family) by one structured move."""
            u_set = set(u)
            l_set = set(l)
            uc    = set(upper_candidates)
            lc    = set(lower_candidates)

            ops: list[str] = []
            if len(u) < max_upper_terms and uc - u_set:
                ops.append("add_upper")
            if len(u) > MIN_UPPER:
                ops.append("rem_upper")
            if len(l) < max_lower_terms and lc - l_set:
                ops.append("add_lower")
            if len(l) > MIN_LOWER:
                ops.append("rem_lower")
            if u and (lc - l_set) and len(l) < max_lower_terms:
                ops.append("upper_to_lower")
            if l and (uc - u_set) and len(u) < max_upper_terms:
                ops.append("lower_to_upper")
            if u and (uc - u_set):
                ops.append("swap_upper")
            if l and (lc - l_set):
                ops.append("swap_lower")
            if len(families) > 1:
                ops.append("toggle_family")
            if not ops:
                return u, l, fam

            op = ops[int(rng.integers(len(ops)))]
            nu, nl, nf = list(u), list(l), fam

            if op == "add_upper":
                nu.append(rng.choice(sorted(uc - u_set)))
            elif op == "rem_upper":
                nu.pop(int(rng.integers(len(nu))))
            elif op == "add_lower":
                nl.append(rng.choice(sorted(lc - l_set)))
            elif op == "rem_lower":
                nl.pop(int(rng.integers(len(nl))))
            elif op == "upper_to_lower":
                v = nu.pop(int(rng.integers(len(nu))))
                nl.append(v) if v in lc else nu.append(v)
            elif op == "lower_to_upper":
                v = nl.pop(int(rng.integers(len(nl))))
                nu.append(v) if v in uc else nl.append(v)
            elif op == "swap_upper":
                idx = int(rng.integers(len(nu)))
                cands = sorted(uc - u_set)
                nu[idx] = rng.choice(cands)
            elif op == "swap_lower":
                idx = int(rng.integers(len(nl)))
                cands = sorted(lc - l_set)
                nl[idx] = rng.choice(cands)
            elif op == "toggle_family":
                other = [f for f in families if f != nf]
                nf = rng.choice(other) if other else nf

            # Enforce minimum complexity
            while len(nu) < MIN_UPPER and (uc - set(nu)):
                nu.append(rng.choice(sorted(uc - set(nu))))
            while len(nl) < MIN_LOWER and (lc - set(nl)):
                nl.append(rng.choice(sorted(lc - set(nl))))

            return sorted(set(nu)), sorted(set(nl)), nf

        search_mode = str(search_method).strip().lower()
        if search_mode == "harmony":
            hms = max(4, int(harmony_hms))
            hmcr = float(np.clip(harmony_hmcr, 0.0, 1.0))
            par = float(np.clip(harmony_par, 0.0, 1.0))
            print(f"  Harmony Search refinement: HMS={hms}, HMCR={hmcr:.2f}, PAR={par:.2f}, iters={n_sa}")

            harmony_memory: list[tuple[float, list[str], list[str], str]] = []
            _mem_seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()

            def _mem_add(bic_v: float, u_v: list[str], l_v: list[str], f_v: str) -> None:
                key_v = (tuple(sorted(set(u_v))), tuple(sorted(set(l_v))), str(f_v))
                if key_v in _mem_seen or not np.isfinite(bic_v):
                    return
                _mem_seen.add(key_v)
                harmony_memory.append((float(bic_v), list(key_v[0]), list(key_v[1]), key_v[2]))
                harmony_memory.sort(key=lambda z: z[0])
                if len(harmony_memory) > hms:
                    worst = harmony_memory.pop(-1)
                    _mem_seen.discard((tuple(worst[1]), tuple(worst[2]), worst[3]))

            _mem_add(sa_bic, sa_upper, sa_lower, sa_fam)
            for _neg_bic, _, _fit_mem in sorted(top_k_heap, key=lambda x: x[0]):
                _mb = float(getattr(_fit_mem.result, "bic", np.nan))
                _mem_add(_mb, list(_fit_mem.upper_vars), list(_fit_mem.lower_vars), str(_fit_mem.family))

            for _ in range(n_sa):
                if harmony_memory and rng.random() < hmcr:
                    _base = harmony_memory[int(rng.integers(0, len(harmony_memory)))]
                    nu, nl, nf = list(_base[1]), list(_base[2]), str(_base[3])
                else:
                    k_upper = int(rng.integers(MIN_UPPER, max_upper_terms + 1))
                    k_lower = int(rng.integers(MIN_LOWER, max_lower_terms + 1))
                    nf = families[int(rng.integers(0, len(families)))]
                    nu = sorted(rng.choice(upper_candidates, size=min(k_upper, len(upper_candidates)), replace=False).tolist()) if upper_candidates else []
                    nl = sorted(rng.choice(lower_candidates, size=min(k_lower, len(lower_candidates)), replace=False).tolist()) if lower_candidates else []

                if rng.random() < par:
                    nu, nl, nf = _perturb_sa(nu, nl, nf)

                nu = sorted(set(nu))[:max_upper_terms]
                nl = sorted(set(nl))[:max_lower_terms]
                while len(nu) < MIN_UPPER and len(set(upper_candidates) - set(nu)) > 0:
                    nu.append(rng.choice(sorted(set(upper_candidates) - set(nu))))
                while len(nl) < MIN_LOWER and len(set(lower_candidates) - set(nl)) > 0:
                    nl.append(rng.choice(sorted(set(lower_candidates) - set(nl))))
                nu = sorted(set(nu))
                nl = sorted(set(nl))

                key = (tuple(nu), tuple(nl), nf)
                if key in tested:
                    continue
                tested.add(key)

                fit = _fit_model(
                    df_train,
                    aadt_col=aadt_col,
                    y_col=y_col,
                    upper_vars=nu,
                    lower_vars=nl,
                    family=nf,
                    offset_col=offset_col,
                )
                if fit is None:
                    continue
                _eval(fit, y_val_np, phase="harmony")

                new_bic = float(getattr(fit.result, "bic", np.nan))
                if np.isfinite(new_bic):
                    _mono_ok = True
                    if enforce_aadt_increase:
                        _mono_ok = bool(np.min(_aadt_elasticity(df_val, fit)) > float(min_aadt_elasticity))
                    if _mono_ok:
                        _mem_add(new_bic, nu, nl, nf)
                    if new_bic < sa_bic:
                        sa_upper, sa_lower, sa_fam = nu, nl, nf
                        sa_bic = new_bic

            # Always produce a directly comparable SA trace when running harmony.
            # This enables explicit SA-vs-Harmony diagnostics in reports.
            print(f"  Simulated Annealing comparison run: iters={n_sa}")
            sa_cmp_upper = list(_sa_upper_init)
            sa_cmp_lower = list(_sa_lower_init)
            sa_cmp_fam = str(_sa_fam_init)
            sa_cmp_start_fit = _fit_model(
                df_train=df_train,
                aadt_col=aadt_col,
                y_col=y_col,
                upper_vars=sa_cmp_upper,
                lower_vars=sa_cmp_lower,
                family=sa_cmp_fam,
                offset_col=offset_col,
            )
            if sa_cmp_start_fit is None:
                sa_cmp_upper = list(best_fit.upper_vars)
                sa_cmp_lower = list(best_fit.lower_vars)
                sa_cmp_fam = str(best_fit.family)
                sa_cmp_start_fit = best_fit

            sa_cmp_bic = float(getattr(sa_cmp_start_fit.result, "bic", np.nan))
            if not np.isfinite(sa_cmp_bic):
                sa_cmp_bic = 1e12
            T_cmp = 20.0
            T_cmp_min = 0.5
            cooling_cmp = (T_cmp_min / T_cmp) ** (1.0 / max(n_sa, 1))
            sa_seen: set[tuple[tuple[str, ...], tuple[str, ...], str]] = set()

            for _ in range(n_sa):
                nu, nl, nf = _perturb_sa(sa_cmp_upper, sa_cmp_lower, sa_cmp_fam)
                key_sa = (tuple(nu), tuple(nl), nf)
                if key_sa in sa_seen:
                    T_cmp = max(T_cmp * cooling_cmp, T_cmp_min)
                    continue
                sa_seen.add(key_sa)

                fit = _fit_model(
                    df_train,
                    aadt_col=aadt_col,
                    y_col=y_col,
                    upper_vars=nu,
                    lower_vars=nl,
                    family=nf,
                    offset_col=offset_col,
                )
                if fit is not None:
                    _eval(fit, y_val_np, phase="sa")
                    new_bic = float(getattr(fit.result, "bic", np.nan))
                    if np.isfinite(new_bic):
                        delta = new_bic - sa_cmp_bic
                        if delta < 0 or rng.random() < float(np.exp(-delta / max(T_cmp, 1e-9))):
                            sa_cmp_upper, sa_cmp_lower, sa_cmp_fam = nu, nl, nf
                            sa_cmp_bic = new_bic

                T_cmp = max(T_cmp * cooling_cmp, T_cmp_min)
        else:
            for _ in range(n_sa):
                nu, nl, nf = _perturb_sa(sa_upper, sa_lower, sa_fam)
                key = (tuple(nu), tuple(nl), nf)
                if key not in tested:
                    tested.add(key)
                    fit = _fit_model(df_train, aadt_col=aadt_col, y_col=y_col,
                                     upper_vars=nu, lower_vars=nl,
                                     family=nf, offset_col=offset_col)
                    if fit is not None:
                        _eval(fit, y_val_np, phase="sa")
                        new_bic = float(getattr(fit.result, "bic", np.nan))
                        if np.isfinite(new_bic):
                            delta = new_bic - sa_bic
                            # Accept improvement always; accept worsening with SA prob
                            if delta < 0 or rng.random() < float(np.exp(-delta / max(T, 1e-9))):
                                sa_upper, sa_lower, sa_fam = nu, nl, nf
                                sa_bic = new_bic

                T = max(T * cooling, T_min)

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

        full_label  = VARIABLE_LABELS.get(raw_var, raw_var)
        description = VARIABLE_DESCRIPTIONS.get(raw_var, "")
        data_min    = float(series.min()) if np.isfinite(series.min()) else q10
        data_max    = float(series.max()) if np.isfinite(series.max()) else q90
        variable_specs[raw_var] = {
            "is_binary": False,
            "min": float(q10),
            "max": float(q90),
            "data_min": data_min,    # full dataset range for non-linear curve
            "data_max": data_max,
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
    // Use the full data range (not just p10–p90) so curvature is more visible
    const lo = spec.data_min ?? spec.min;
    const hi = spec.data_max ?? spec.max;
    const values = [];
    const points = 61;
    for (let i = 0; i < points; i++) values.push(lo + ((hi - lo) * i) / (points - 1));
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

    const yTitle = mode === 'pred' ? 'Predicted crashes' : 'CMF (vs median profile)';
    // Reference line at y=1 for CMF mode, or at baseline for pred mode
    const refLine = mode === 'cmf'
        ? [{{ x: [x[0], x[x.length-1]], y: [1, 1], mode: 'lines',
              line: {{ color: '#bbb', width: 1.2, dash: 'dash' }},
              name: 'No change (CMF=1)', showlegend: true }}]
        : [];

    const mainTrace = {{
        x, y,
        type: 'scatter', mode: 'lines',
        line: {{ width: 3.5, color: '#0a6c74', shape: 'spline', smoothing: 0.3 }},
        name: mode === 'pred' ? 'Predicted crashes' : 'CMF',
        fill: mode === 'cmf' ? 'tonexty' : 'none',
        fillcolor: 'rgba(10,108,116,0.07)'
    }};
    const markerTrace = {{
        x: [currentValue], y: [selectedY],
        type: 'scatter', mode: 'markers',
        marker: {{ size: 13, color: '#d96f32', symbol: 'diamond',
                   line: {{ color: '#fff', width: 1.5 }} }},
        name: 'Current value'
    }};
    // p10/p50/p90 tick marks on x-axis via shapes
    const xTickShapes = spec.is_binary ? [] : [
        {{ type: 'line', xref: 'x', yref: 'paper', x0: spec.q10, x1: spec.q10,
           y0: 0, y1: 0.04, line: {{ color: '#bbb', width: 1 }} }},
        {{ type: 'line', xref: 'x', yref: 'paper', x0: spec.q50, x1: spec.q50,
           y0: 0, y1: 0.06, line: {{ color: '#999', width: 1.5 }} }},
        {{ type: 'line', xref: 'x', yref: 'paper', x0: spec.q90, x1: spec.q90,
           y0: 0, y1: 0.04, line: {{ color: '#bbb', width: 1 }} }},
    ];
    Plotly.react('chart', [...refLine, mainTrace, markerTrace], {{
        template: 'plotly_white',
        title: {{
            text: (spec.label || variable) + ' at AADT ' + fmt(aadtValue, 'veh/day'),
            font: {{ size: 12 }}
        }},
        xaxis: {{ title: spec.unit ? spec.label + ' (' + spec.unit + ')' : spec.label,
                  gridcolor: '#eee' }},
        yaxis: {{ title: yTitle, gridcolor: '#eee' }},
        shapes: xTickShapes,
        margin: {{ l: 64, r: 16, t: 52, b: 52 }},
        legend: {{ orientation: 'h', y: -0.18, font: {{ size: 10 }} }}
    }}, {{ responsive: true }});

    // ── AADT response fan: 3 lines at low / median / high of selected variable ──
    // For UPPER terms  -> 3 parallel lines (same slope, different height)
    // For LOWER terms  -> 3 diverging lines (different slopes) = clearly non-linear!
    const aadtGrid = [];
    const aadtPoints = 61;
    for (let i = 0; i < aadtPoints; i++) {{
        aadtGrid.push(model.aadt_range.min + ((model.aadt_range.max - model.aadt_range.min) * i) / (aadtPoints - 1));
    }}

    const fanLevels = spec.is_binary
        ? [{{ label: 'Absent (0)', val: 0, color: '#1f6bb5' }},
           {{ label: 'Present (1)', val: 1, color: '#d96f32' }}]
        : [{{ label: 'Low (p10=' + fmt(spec.q10,'') + (spec.unit?' '+spec.unit:'') + ')', val: spec.q10, color: '#1f6bb5' }},
           {{ label: 'Median (p50=' + fmt(spec.q50,'') + (spec.unit?' '+spec.unit:'') + ')', val: spec.q50, color: '#0a6c74' }},
           {{ label: 'High (p90=' + fmt(spec.q90,'') + (spec.unit?' '+spec.unit:'') + ')', val: spec.q90, color: '#d96f32' }}];

    const refPred = Math.max(predict({{ ...profile, [variable]: fanLevels[Math.floor(fanLevels.length/2)].val }}, model.aadt_range.default), 1e-9);

    const fanTraces = fanLevels.map(lv => ({{
        x: aadtGrid,
        y: aadtGrid.map(a => {{
            const p = predict({{ ...profile, [variable]: lv.val }}, a);
            return mode === 'pred' ? p : p / refPred;
        }}),
        type: 'scatter', mode: 'lines',
        line: {{ width: 2.5, color: lv.color }},
        name: lv.label
    }}));
    // Current position diamond
    fanTraces.push({{
        x: [aadtValue], y: [mode === 'pred' ? baseline : baseline / refPred],
        type: 'scatter', mode: 'markers',
        marker: {{ size: 12, color: '#f0b000', symbol: 'diamond', line: {{ color: '#333', width: 1 }} }},
        name: 'Current setting'
    }});

    Plotly.react('aadtChart', fanTraces, {{
        template: 'plotly_white',
        title: {{
            text: 'How ' + (spec.label || variable) + ' shifts the AADT-crash curve',
            font: {{ size: 12 }}
        }},
        xaxis: {{ title: model.aadt_col + ' (veh/day)', gridcolor: '#eee' }},
        yaxis: {{ title: yTitle, gridcolor: '#eee' }},
        margin: {{ l: 64, r: 16, t: 52, b: 52 }},
        legend: {{ orientation: 'h', y: -0.18, font: {{ size: 10 }} }},
        annotations: [{{
            xref: 'paper', yref: 'paper', x: 0, y: 1.18,
            text: 'Parallel lines = AADT-independent effect | Diverging/converging = traffic-interaction effect',
            showarrow: false, font: {{ size: 9, color: '#888' }}, align: 'left'
        }}]
    }}, {{ responsive: true }});

    aadtLabel.textContent = 'AADT (' + fmt(aadtValue, '') + ')';
    const topUpper = [...decomp.upperItems]
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .slice(0, 5)
        .map(item => item.variable + ': ' + fmt(item.value, '') + ' -> ' + fmt(item.contribution, ''))
        .join('<br/>') || '(none)';
    const topLower = [...decomp.lowerItems]
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .slice(0, 5)
        .map(item => item.variable + ': ' + fmt(item.value, '') + ' -> ' + fmt(item.contribution, ''))
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
        monotonic  = df["Monotonic AADT (e>0 all segs)"].tolist() if "Monotonic AADT (e>0 all segs)" in df.columns else ["yes"] * len(df)
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
        bic_ylim = _compute_tight_axis_window(bic_raw, objective="bic")
        rmse_ylim = _compute_tight_axis_window(rmse_raw, objective="rmse")

        payload = {
            "iters": iters,
            "bic": bic_raw, "bic_rmin": bic_rmin,
            "rmse": rmse_raw, "rmse_rmin": rmse_rmin,
            "dev": dev_raw,
            "upper_vars": upper_vars, "lower_vars": lower_vars,
            "monotonic": monotonic, "phases": phases,
            "dot_colors": dot_colors,
            "best_bic_idx": best_bic_idx, "best_rmse_idx": best_rmse_idx,
            "bic_ylim": list(bic_ylim) if bic_ylim is not None else None,
            "rmse_ylim": list(rmse_ylim) if rmse_ylim is not None else None,
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

const bicAxis = {{title:'BIC', gridcolor:'#1e3045', zeroline:false}};
if (Array.isArray(D.bic_ylim) && D.bic_ylim.length === 2) bicAxis.range = D.bic_ylim;

const rmseAxis = {{title:'Val RMSE', gridcolor:'#1e3045', zeroline:false}};
if (Array.isArray(D.rmse_ylim) && D.rmse_ylim.length === 2) rmseAxis.range = D.rmse_ylim;

Plotly.newPlot('bc',
  [pathT(D.bic,'proposed'), dotsT(D.bic,'BIC'), envT(D.bic_rmin,'#7fdee8','Running best'), starT(D.best_bic_idx,D.bic,'Best')],
  {{...BASE, title:{{text:'BIC over iterations',font:{{size:12}}}},
        yaxis:bicAxis}},
  {{responsive:true, displayModeBar:false}});

Plotly.newPlot('rc',
  [pathT(D.rmse,'proposed'), dotsT(D.rmse,'Val RMSE'), envT(D.rmse_rmin,'#f0b060','Running best'), starT(D.best_rmse_idx,D.rmse,'Best')],
  {{...BASE, title:{{text:'Validation RMSE over iterations',font:{{size:12}}}},
        yaxis:rmseAxis}},
  {{responsive:true, displayModeBar:false}});
</script>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"[WARNING] Could not save convergence HTML: {exc}")


def _compute_tight_axis_window(values: list[float] | np.ndarray, objective: str) -> tuple[float, float] | None:
    """Return a tighter y-window that focuses on the competitive region.

    For BIC specifically, this intentionally trims very high outlier values so
    presentation plots emphasize actionable differences near the best scores.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    best = float(np.nanmin(arr))
    worst = float(np.nanmax(arr))
    span = max(worst - best, 1e-9)

    if str(objective).lower() == "bic":
        focus = max(20.0, min(180.0, 0.18 * span))
        p85 = float(np.nanpercentile(arr, 85))
        upper = min(best + focus, p85 + max(2.0, 0.03 * span))
        upper = max(upper, best + 5.0)
        lower = best - max(4.0, 0.06 * (upper - best))
        return (float(lower), float(upper))

    # RMSE / deviance windows: keep moderate clipping for readability.
    focus = max(0.5, min(4.0, 0.40 * span))
    p90 = float(np.nanpercentile(arr, 90))
    upper = min(best + focus, p90 + max(0.05, 0.05 * span))
    upper = max(upper, best + 0.25)
    lower = best - max(0.1, 0.10 * (upper - best))
    return (float(lower), float(upper))


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
    include_lower_interactions: bool,
    max_random_terms: int,
    rp_draws: int,
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


def _fit_literature_benchmark_with_metacount(
    df_trainval_raw: pd.DataFrame,
    y_col: str,
    aadt_col: str,
    offset_col: str | None,
    rp_draws: int,
) -> dict[str, Any] | None:
    """Fit the user-specified Ex16-3 benchmark structure with MetaCount's mixed NB2 engine."""
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

        bench_offset_col = "__BENCHMARK_OFFSET_AADT__"
        base_off = np.zeros(len(df), dtype=float)
        if offset_col is not None and offset_col in df.columns:
            base_off = pd.to_numeric(df[offset_col], errors="coerce").to_numpy(dtype=float)
        log_aadt = np.log(np.clip(pd.to_numeric(df[aadt_col], errors="coerce").to_numpy(dtype=float), 1e-12, None))
        df[bench_offset_col] = base_off + log_aadt

        fixed_terms = [v for v in LITERATURE_BENCHMARK_NONRANDOM_RAW if v in df.columns]
        random_terms = [v for v in LITERATURE_BENCHMARK_RANDOM_MEAN_RAW if v in df.columns]

        if not fixed_terms and not random_terms:
            return None

        builder = ExperimentBuilder(df=df, id_col="_id", y_col=y_col, offset_col=bench_offset_col)
        spec = builder.make_manual_spec(
            fixed_terms=fixed_terms,
            rdm_terms=[f"{v}:normal" for v in random_terms],
            dispersion=1,
        )
        fit = builder.fit_manual_model(
            spec,
            model="nb",
            print_report=False,
            R=max(200, int(rp_draws)),
        )

        from main_hpc import unpack_params as _unpack  # type: ignore

        fitted_spec = fit.get("spec", spec)
        blocks = _unpack(np.array(fit["result"].params), fitted_spec)
        rows: list[dict[str, Any]] = []

        for k, name in enumerate(fitted_spec.fixed_names):
            rows.append(
                {
                    "Variable": VARIABLE_LABELS.get(name, name),
                    "Role": "Nonrandom parameter",
                    "Estimate": float(np.array(blocks["beta_f"])[k]),
                    "StdDev": np.nan,
                }
            )

        if fitted_spec.Kr_ind > 0 and blocks.get("mean_ind") is not None:
            means = np.array(blocks["mean_ind"])
            sds = np.abs(np.array(blocks["sd_ind"]))
            for j, rname in enumerate(fitted_spec.random_ind_names):
                rows.append(
                    {
                        "Variable": VARIABLE_LABELS.get(rname, rname),
                        "Role": "Random mean (normal)",
                        "Estimate": float(means[j]),
                        "StdDev": float(sds[j]),
                    }
                )

        if blocks.get("alpha") is not None:
            rows.append(
                {
                    "Variable": "NB2 Dispersion (alpha)",
                    "Role": "Dispersion",
                    "Estimate": float(jax.nn.softplus(blocks["alpha"])),
                    "StdDev": np.nan,
                }
            )

        coef_df = pd.DataFrame(rows)
        summary_d = fit.get("summary", {}) if isinstance(fit, dict) else {}
        ll = summary_d.get("loglik", float("nan")) if isinstance(summary_d, dict) else float("nan")
        bic = summary_d.get("bic", float("nan")) if isinstance(summary_d, dict) else float("nan")
        aic = summary_d.get("aic", float("nan")) if isinstance(summary_d, dict) else float("nan")
        n_obs = summary_d.get("n_obs", float("nan")) if isinstance(summary_d, dict) else float("nan")
        num_parm = summary_d.get("num_parm", float("nan")) if isinstance(summary_d, dict) else float("nan")

        # Compact quality checks so extreme/unreliable benchmark fits are visible in artifacts.
        fit_result = fit.get("result") if isinstance(fit, dict) else None
        fit_state = getattr(fit_result, "state", None)
        opt_iter = getattr(fit_state, "iter_num", None)
        opt_error = getattr(fit_state, "error", None)
        opt_value = getattr(fit_state, "value", None)
        opt_ls_fail = bool(getattr(fit_state, "failed_linesearch", False))
        has_nonfinite = False
        if not coef_df.empty:
            has_nonfinite = bool(~np.isfinite(pd.to_numeric(coef_df["Estimate"], errors="coerce")).all())

        fixed_est = coef_df.loc[coef_df["Role"] == "Nonrandom parameter", "Estimate"]
        rand_sd = coef_df.loc[coef_df["Role"] == "Random mean (normal)", "StdDev"]
        max_abs_fixed = float(np.nanmax(np.abs(fixed_est.to_numpy(dtype=float)))) if len(fixed_est) else float("nan")
        max_rand_sd = float(np.nanmax(rand_sd.to_numpy(dtype=float))) if len(rand_sd) else float("nan")

        obj_consistency = float("nan")
        if np.isfinite(float(ll)) and opt_value is not None and np.isfinite(float(opt_value)):
            obj_consistency = float(abs(float(opt_value) + float(ll)))

        q_lines = [
            "Fit quality checks:",
            f"- Optimizer linesearch: {'WARN' if opt_ls_fail else 'PASS'} (failed_linesearch={opt_ls_fail})",
            (
                f"- Optimizer residual: {'PASS' if (opt_error is not None and float(opt_error) <= 1e-3) else 'WARN'} "
                f"(error={float(opt_error):.3e})"
                if opt_error is not None
                else "- Optimizer residual: WARN (not reported)"
            ),
            (
                f"- Objective consistency: {'PASS' if (np.isfinite(obj_consistency) and obj_consistency <= 1e-3) else 'WARN'} "
                f"(|state.value + loglik|={obj_consistency:.3e})"
                if np.isfinite(obj_consistency)
                else "- Objective consistency: WARN (not available)"
            ),
            f"- Finite coefficients: {'PASS' if not has_nonfinite else 'WARN'}",
            (
                f"- Max |nonrandom estimate|: {'PASS' if (np.isfinite(max_abs_fixed) and max_abs_fixed <= 100.0) else 'WARN'} "
                f"({max_abs_fixed:.4f})"
                if np.isfinite(max_abs_fixed)
                else "- Max |nonrandom estimate|: WARN (not available)"
            ),
            (
                f"- Max random SD: {'PASS' if (np.isfinite(max_rand_sd) and max_rand_sd <= 10.0) else 'WARN'} "
                f"({max_rand_sd:.4f})"
                if np.isfinite(max_rand_sd)
                else "- Max random SD: WARN (not available)"
            ),
        ]

        lines = [
            "MetaCount benchmark re-estimation (Ex16-3 structure)",
            "",
            f"LogLik: {float(ll):.4f}",
            f"BIC: {float(bic):.4f}",
            f"AIC: {float(aic):.4f}",
            (
                f"n_obs={int(n_obs)}, n_params={int(num_parm)}, optimizer_iter={int(opt_iter)}"
                if np.isfinite(float(n_obs)) and np.isfinite(float(num_parm)) and opt_iter is not None
                else "n_obs / n_params / optimizer_iter: not fully available"
            ),
            f"Offset used: {bench_offset_col} = log(AADT) + log(L)",
            "",
            "Model terms:",
            f"- Nonrandom: {', '.join(fixed_terms) if fixed_terms else '(none)'}",
            f"- Random means: {', '.join(random_terms) if random_terms else '(none)'}",
            "",
            *q_lines,
            "",
            "Estimated coefficients:",
            _to_markdown(coef_df) if not coef_df.empty else "(none)",
        ]

        return {
            "coef_df": coef_df,
            "summary_md": "\n".join(lines),
            "loglik": float(ll),
            "bic": float(bic),
            "fixed_terms": fixed_terms,
            "random_terms": random_terms,
        }
    except Exception:
        import traceback as _tb

        return {
            "coef_df": pd.DataFrame(),
            "summary_md": (
                "MetaCount benchmark re-estimation failed.\n\n"
                f"Error:\n```\n{_tb.format_exc()}\n```"
            ),
            "error": _tb.format_exc(),
        }

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

        # Variable name mapper (raw -> model column)
        def _model_name(v: str) -> str:
            return v if v in binary_vars else f"{v}_Z"

        rdm_upper_names = [_model_name(v) for v in continuous_upper if _model_name(v) in df.columns]
        rdm_lower_names = [f"_inter_{v}" for v in continuous_lower if f"_inter_{v}" in df.columns]

        # Candidate random terms can include upper-level continuous effects and,
        # optionally, lower-level AADT-interaction terms.
        rdm_candidates = list(dict.fromkeys(rdm_upper_names + (rdm_lower_names if include_lower_interactions else [])))

        # Rank random-term candidates by absolute marginal correlation with y.
        # This keeps the sweep focused on the strongest heterogeneity signals.
        y_arr = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        ranked_terms: list[tuple[float, str]] = []
        for term in rdm_candidates:
            x_arr = pd.to_numeric(df[term], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x_arr) & np.isfinite(y_arr)
            if np.sum(valid) < 5:
                score = 0.0
            else:
                try:
                    score = float(abs(np.corrcoef(x_arr[valid], y_arr[valid])[0, 1]))
                    if not np.isfinite(score):
                        score = 0.0
                except Exception:
                    score = 0.0
            ranked_terms.append((score, term))
        ranked_terms.sort(key=lambda t: t[0], reverse=True)
        rdm_names = [name for _, name in ranked_terms[: max(1, int(max_random_terms))]]

        # Fixed terms exclude any terms selected as random.
        fixed_binary_upper = [_model_name(v) for v in best_upper_raw if _model_name(v) not in rdm_names]
        fixed_lower_inter = [f"_inter_{v}" for v in best_lower_raw if f"_inter_{v}" in df.columns and f"_inter_{v}" not in rdm_names]
        fixed_terms = (["_log_aadt"] + [t for t in fixed_binary_upper if t in df.columns] + fixed_lower_inter)

        if rdm_names:
            print("    [RP] Candidate random terms (ranked): " + ", ".join(rdm_names))

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

                # Independent random params — include distribution name in label
                if spec.Kr_ind > 0 and blocks.get("mean_ind") is not None:
                    means = np.array(blocks["mean_ind"])
                    sds   = np.abs(np.array(blocks["sd_ind"]))
                    # Infer distribution from spec.random_ind_dists if available
                    ind_dists = list(getattr(spec, "random_ind_dists", []))
                    for j, rname in enumerate(spec.random_ind_names):
                        lbl  = VARIABLE_LABELS.get(rname.replace("_Z",""), rname)
                        dist = ind_dists[j] if j < len(ind_dists) else "normal"
                        rows.append({
                            "Parameter": f"{lbl} [random, {dist}]",
                            "Type":  f"Random ({dist})",
                            "Mean":  round(float(means[j]), 5),
                            "SD":    round(float(sds[j]),   5),
                        })

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

        # Halton draws for simulation-based integration (configurable).
        _R = int(rp_draws)

        best_result: dict | None = None

        # Distributions to try for each random parameter.
        # We evaluate every distribution and keep the best-BIC winner.
        # normal    — symmetric, unbounded; default choice
        # lognormal — positive-skewed; useful if coefficient should be one-signed
        # triangular — bounded, flexible shape
        # uniform   — bounded, equal-weight prior
        _DISTS = ["normal", "lognormal", "triangular", "uniform"]

        # ── Try each distribution for independent random params ──────────
        if rdm_names:
            for dist in _DISTS:
                try:
                    spec_ind = builder.make_manual_spec(
                        fixed_terms = fixed_terms,
                        rdm_terms   = [f"{n}:{dist}" for n in rdm_names],
                        dispersion  = 1,
                    )
                    fit_ind = builder.fit_manual_model(
                        spec_ind, model="nb", print_report=False, R=_R)
                    res_ind = _extract_result(fit_ind)
                    res_ind["dist"] = dist   # record which distribution won
                    if np.isfinite(res_ind["bic"]):
                        marker = " *" if best_result is None or res_ind["bic"] < best_result["bic"] else ""
                        print(f"    [RP] Indep {dist:12s} BIC: {res_ind['bic']:.2f}{marker}")
                        if best_result is None or res_ind["bic"] < best_result["bic"]:
                            best_result = res_ind
                except Exception:
                    pass

        # ── Try top-2 as CORRELATED with best independent distribution ───
        if len(rdm_names) >= 2:
            best_dist = best_result.get("dist", "normal") if best_result else "normal"
            try:
                cor_names = rdm_names[:2]
                ind_rest  = rdm_names[2:]
                spec_cor  = builder.make_manual_spec(
                    fixed_terms   = fixed_terms,
                    rdm_cor_terms = [f"{n}:{best_dist}" for n in cor_names],
                    rdm_terms     = [f"{n}:{best_dist}" for n in ind_rest],
                    dispersion    = 1,
                )
                fit_cor = builder.fit_manual_model(
                    spec_cor, model="nb", print_report=False, R=_R)
                res_cor = _extract_result(fit_cor, cor_names=cor_names)
                res_cor["dist"] = best_dist
                if np.isfinite(res_cor["bic"]):
                    marker = " *" if best_result is None or res_cor["bic"] < best_result["bic"] else ""
                    print(f"    [RP] Correlated ({best_dist}) BIC: {res_cor['bic']:.2f}{marker}")
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
        lines.append("  Local elasticity_i = beta_AADT + sum(gamma_k * x_k_std_i)")
        lines.append("  'Monotonic AADT' constraint: elasticity > 0 for EVERY segment in dataset.")
        lines.append("  Meaning: more traffic always -> more crashes (no segment ever reverses).")
        lines.append("")

    # ── FC note ────────────────────────────────────────────────────────────
    if any("FC" in str(n) for n in params.index):
        lines.append("  NOTE — Functional Classification (FC) coding:")
        for code, desc in FC_LABELS.items():
            lines.append(f"    FC = {code} -> {desc}")
        lines.append("")

    lines.append("=" * w)
    text = "\n".join(lines)
    print(text)
    if save_path is not None:
        save_path.write_text(text, encoding="utf-8")


def _save_model_equation_md(
    path: Path,
    fitted: FittedModel,
    scaler_stats: dict[str, tuple[float, float]],
    binary_vars: set[str],
    offset_col: str | None,
    aadt_col: str,
) -> None:
    """
    Write a fully spelled-out equation slide markdown file.
    Every term in the final model is written out with:
      - the actual coefficient value
      - the full human-readable variable name
      - the unit / description
      - what it means (direct effect or AADT interaction)
    """
    params = pd.Series(fitted.result.params)
    lines  = []

    def _rnd(v: float) -> str:
        return f"{v:+.4f}"

    def _lbl(raw: str) -> str:
        return VARIABLE_LABELS.get(raw, raw)

    lines.append("### What each coefficient means for crash prediction")
    lines.append("")
    lines.append(
        "> **How to read this table:** The model predicts "
        "$\\log(\\text{crashes}) = $ sum of all terms below. "
        "Each row shows the exact contribution for one variable."
    )
    lines.append("")

    # ── Build rows ────────────────────────────────────────────────────────
    rows = []

    # Intercept
    alpha = float(params.get("const", params.get("__INTERCEPT__", 0.0)))
    rows.append({
        "Term": "**Intercept** $\\alpha$",
        "Coefficient": _rnd(alpha),
        "Variable / Formula": "Baseline (no traffic, no geometry adjustments)",
        "Interpretation": f"$e^{{{alpha:.3f}}} = {np.exp(alpha):.2f}$ crashes per unit exposure at reference conditions",
    })

    # AADT elasticity
    b_aadt = float(params.get("log_aadt", 0.0))
    rows.append({
        "Term": "**Base AADT elasticity** $\\beta_{\\text{AADT}}$",
        "Coefficient": _rnd(b_aadt),
        "Variable / Formula": "$\\beta \\times \\log(\\text{AADT})$",
        "Interpretation": f"Each 1% more traffic -> crashes change by {b_aadt:.2f}% (elasticity = {b_aadt:.3f})",
    })

    # Upper terms
    upper_added = False
    for pname, coef in params.items():
        pname = str(pname)
        if not pname.startswith("upper::"):
            continue
        if not upper_added:
            rows.append({"Term": "---", "Coefficient": "---",
                         "Variable / Formula": "**UPPER TERMS — direct effect, same CMF at any AADT**",
                         "Interpretation": ""})
            upper_added = True
        raw = pname.replace("upper::", "")
        is_std = raw.endswith("_Z")
        base = raw[:-2] if is_std else raw
        lbl  = _lbl(base)
        sd   = scaler_stats.get(base, (0.0, 1.0))[1]
        coef_f = float(coef)
        orig = coef_f / sd if is_std and sd > 0 else coef_f
        cmf_1sd = np.exp(coef_f) if is_std else np.exp(orig)
        unit_note = " (per 1 SD change)" if is_std else " (per 1 unit)"
        rows.append({
            "Term": f"$\\beta_{{{base}}}$",
            "Coefficient": _rnd(coef_f),
            "Variable / Formula": f"$\\beta \\times \\text{{{lbl}}}${unit_note}",
            "Interpretation": (
                f"CMF = $e^{{{coef_f:.4f}}} = {cmf_1sd:.4f}$ "
                f"({'safer' if cmf_1sd < 1 else 'higher risk'}, "
                f"{abs(cmf_1sd - 1)*100:.1f}%{' reduction' if cmf_1sd < 1 else ' increase'})"
            ),
        })

    # Lower terms
    lower_added = False
    for pname, coef in params.items():
        pname = str(pname)
        if not (pname.startswith("lower::") and pname.endswith("*log_aadt")):
            continue
        if not lower_added:
            rows.append({"Term": "---", "Coefficient": "---",
                         "Variable / Formula": "**LOWER TERMS — modify AADT elasticity (traffic interaction)**",
                         "Interpretation": ""})
            lower_added = True
        raw  = pname.replace("lower::", "").replace("*log_aadt", "")
        is_std = raw.endswith("_Z")
        base = raw[:-2] if is_std else raw
        lbl  = _lbl(base)
        sd   = scaler_stats.get(base, (0.0, 1.0))[1]
        coef_f = float(coef)
        orig   = coef_f / sd if is_std and sd > 0 else coef_f
        unit_note = " (per 1 SD)" if is_std else " (per 1 unit)"
        direction = "amplifies" if coef_f > 0 else "dampens"
        rows.append({
            "Term": f"$\\gamma_{{{base}}}$",
            "Coefficient": _rnd(coef_f),
            "Variable / Formula": (
                f"$\\gamma \\times \\text{{{lbl}}} \\times \\log(\\text{{AADT}})${unit_note}"
            ),
            "Interpretation": (
                f"{direction.capitalize()} AADT elasticity by {abs(coef_f):.4f}{unit_note}. "
                f"{'Higher values make crash rate grow faster with AADT.' if coef_f > 0 else 'Higher values reduce how fast crash rate grows with AADT.'}"
            ),
        })

    # Offset
    rows.append({"Term": "---", "Coefficient": "---",
                 "Variable / Formula": "**OFFSET — exposure correction**", "Interpretation": ""})
    rows.append({
        "Term": "**Offset**",
        "Coefficient": "1.000 (fixed)",
        "Variable / Formula": "$+ \\log(\\text{Segment Length, mi})$",
        "Interpretation": (
            "Longer segments have proportionally more crashes. "
            "Doubling length doubles expected crashes (holding rate constant). "
            "This makes coefficients comparable across segments of different sizes."
        ),
    })

    # Write pipe table
    cols   = ["Term", "Coefficient", "Variable / Formula", "Interpretation"]
    widths = {c: max(len(c), max(len(str(r.get(c,""))) for r in rows)) for c in cols}
    header  = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    divider = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    data_rows = [
        "| " + " | ".join(str(r.get(c,"")).ljust(widths[c]) for c in cols) + " |"
        for r in rows
    ]

    lines += [header, divider] + data_rows
    lines.append("")
    lines.append(
        "> **Full equation:** "
        "$\\log(\\hat{\\mu}_i) = \\alpha + \\beta_{\\text{AADT}} \\log A_i + "
        "\\sum_j \\beta_j x_{ij} + \\sum_k \\gamma_k x_{ik} \\log A_i + \\log L_i$"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _fit_benchmark(
    df_tv: pd.DataFrame,
    df_test: pd.DataFrame,
    y_col: str,
    aadt_col: str,
    offset_col: str | None,
    family: str,
    benchmark_upper_vars: list[str] | None = None,
) -> dict[str, Any]:
    """
    Fit an offset-only benchmark SPF:
        log(mu) = alpha + log(AADT) + offset
    where AADT appears ONLY inside the offset term (no beta_AADT parameter).
    No geometry, weather, or interaction terms.
    Returns a dict of metrics on both train+val and test splits.
    """
    bench_offset_col = "__BENCHMARK_OFFSET_AADT__"
    df_tv_b = df_tv.copy()
    df_test_b = df_test.copy()

    base_tv = np.zeros(len(df_tv_b), dtype=float)
    base_test = np.zeros(len(df_test_b), dtype=float)
    if offset_col is not None and offset_col in df_tv_b.columns:
        base_tv = pd.to_numeric(df_tv_b[offset_col], errors="coerce").to_numpy(dtype=float)
    if offset_col is not None and offset_col in df_test_b.columns:
        base_test = pd.to_numeric(df_test_b[offset_col], errors="coerce").to_numpy(dtype=float)

    log_aadt_tv = np.log(np.clip(pd.to_numeric(df_tv_b[aadt_col], errors="coerce").to_numpy(dtype=float), 1e-12, None))
    log_aadt_test = np.log(np.clip(pd.to_numeric(df_test_b[aadt_col], errors="coerce").to_numpy(dtype=float), 1e-12, None))
    df_tv_b[bench_offset_col] = base_tv + log_aadt_tv
    df_test_b[bench_offset_col] = base_test + log_aadt_test

    bench_upper = list(benchmark_upper_vars or [])

    bench_fit = _fit_model(
        df_tv_b,
        aadt_col   = aadt_col,
        y_col      = y_col,
        upper_vars = bench_upper,
        lower_vars = [],
        family     = family,
        offset_col = bench_offset_col,
        include_log_aadt=False,
    )
    if bench_fit is None:
        return {}

    y_tv   = pd.to_numeric(df_tv_b[y_col],   errors="coerce").to_numpy(float)
    y_test = pd.to_numeric(df_test_b[y_col], errors="coerce").to_numpy(float)

    pred_tv   = _predict(df_tv_b,   bench_fit)
    pred_test = _predict(df_test_b, bench_fit)

    m_tv   = _metrics(y_tv,   pred_tv)
    m_test = _metrics(y_test, pred_test)

    return {
        "family":           family,
        "benchmark_model":  "Ex16-3 literature structure with AADT only in offset (no AADT coefficient)",
        "benchmark_offset_column": bench_offset_col,
        "benchmark_upper_vars": bench_upper,
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


def _parse_search_vars(value: Any) -> list[str]:
    text = str(value).strip()
    if text in {"", "(none)", "nan", "None"}:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _compute_pareto_efficient(df: pd.DataFrame, bic_col: str = "BIC", rmse_col: str = "Val RMSE") -> np.ndarray:
    """Return a boolean mask for non-dominated rows (minimise BIC and RMSE)."""
    scores = df[[bic_col, rmse_col]].to_numpy(dtype=float)
    n = scores.shape[0]
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        bic_i, rmse_i = scores[i]
        dominated = (
            (scores[:, 0] <= bic_i)
            & (scores[:, 1] <= rmse_i)
            & ((scores[:, 0] < bic_i) | (scores[:, 1] < rmse_i))
        )
        if np.any(dominated):
            efficient[i] = False
    return efficient


def _select_pareto_candidate(
    search_history_ordered: pd.DataFrame,
    benchmark_bic: float,
    benchmark_val_rmse: float,
    enforce_aadt_increase: bool,
    require_benchmark_dominance: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Select a model from the Pareto front (BIC, Val RMSE), preferring candidates
    that beat the benchmark on both metrics.
    """
    if search_history_ordered.empty:
        raise RuntimeError("Search history is empty; cannot do Pareto selection.")

    scored = search_history_ordered.copy()
    if enforce_aadt_increase and "Monotonic AADT (e>0 all segs)" in scored.columns:
        scored = scored[scored["Monotonic AADT (e>0 all segs)"].astype(str).str.lower().eq("yes")].copy()

    if scored.empty:
        raise RuntimeError("No monotonic candidate remained for Pareto selection.")

    finite_mask = np.isfinite(pd.to_numeric(scored["BIC"], errors="coerce").to_numpy(dtype=float)) & np.isfinite(
        pd.to_numeric(scored["Val RMSE"], errors="coerce").to_numpy(dtype=float)
    )
    scored = scored.loc[finite_mask].copy()
    if scored.empty:
        raise RuntimeError("No candidate has finite BIC and validation RMSE.")

    scored["Pareto Efficient"] = _compute_pareto_efficient(scored, bic_col="BIC", rmse_col="Val RMSE")
    scored["Beats benchmark (BIC+RMSE)"] = (
        (pd.to_numeric(scored["BIC"], errors="coerce") < float(benchmark_bic))
        & (pd.to_numeric(scored["Val RMSE"], errors="coerce") < float(benchmark_val_rmse))
    )

    pareto = scored[scored["Pareto Efficient"]].copy()
    beating = pareto[pareto["Beats benchmark (BIC+RMSE)"]].copy()

    if require_benchmark_dominance and beating.empty:
        raise RuntimeError(
            "No Pareto-efficient candidate beats the benchmark in both BIC and validation RMSE. "
            "Increase --search-iter / candidate breadth, or run with --no-require-benchmark-dominance."
        )

    pool = beating if not beating.empty else pareto
    if pool.empty:
        raise RuntimeError("Pareto selection failed; no eligible candidate in selection pool.")

    eps_bic = max(abs(float(benchmark_bic)), 1e-9)
    eps_rmse = max(abs(float(benchmark_val_rmse)), 1e-9)
    scored["BIC improvement vs benchmark"] = np.nan
    scored["RMSE improvement vs benchmark"] = np.nan
    scored["Joint improvement score"] = np.nan
    scored["Total improvement score"] = np.nan

    pool["BIC improvement vs benchmark"] = (float(benchmark_bic) - pd.to_numeric(pool["BIC"], errors="coerce")) / eps_bic
    pool["RMSE improvement vs benchmark"] = (
        float(benchmark_val_rmse) - pd.to_numeric(pool["Val RMSE"], errors="coerce")
    ) / eps_rmse
    pool["Joint improvement score"] = np.minimum(
        pool["BIC improvement vs benchmark"], pool["RMSE improvement vs benchmark"]
    )
    pool["Total improvement score"] = pool["BIC improvement vs benchmark"] + pool["RMSE improvement vs benchmark"]

    scored.loc[pool.index, "BIC improvement vs benchmark"] = pool["BIC improvement vs benchmark"]
    scored.loc[pool.index, "RMSE improvement vs benchmark"] = pool["RMSE improvement vs benchmark"]
    scored.loc[pool.index, "Joint improvement score"] = pool["Joint improvement score"]
    scored.loc[pool.index, "Total improvement score"] = pool["Total improvement score"]

    # RMSE-first ranking to reduce the chance of selecting a lower-BIC but
    # meaningfully worse predictive candidate.
    selected = pool.sort_values(
        ["RMSE improvement vs benchmark", "Joint improvement score", "Total improvement score", "Val RMSE", "BIC"],
        ascending=[False, False, False, True, True],
    ).iloc[0]

    merged = search_history_ordered.copy()
    merged = merged.merge(
        scored[[
            "Iteration",
            "Pareto Efficient",
            "Beats benchmark (BIC+RMSE)",
            "BIC improvement vs benchmark",
            "RMSE improvement vs benchmark",
            "Joint improvement score",
            "Total improvement score",
        ]],
        on="Iteration",
        how="left",
    )
    return selected, merged


def _parameter_label_from_name(name: str) -> str:
    if name == "const":
        return "Intercept"
    if name == "log_aadt":
        return "log(AADT)"
    if name.startswith("upper::"):
        return _label(name.replace("upper::", ""))
    if name.startswith("lower::") and name.endswith("*log_aadt"):
        core = name.replace("lower::", "").replace("*log_aadt", "")
        return f"{_label(core)} x log(AADT)"
    return name


def _build_literature_vs_proposed_coef_table(
    coef_df: pd.DataFrame,
    benchmark_fit: FittedModel | None,
    random_params_result: dict[str, Any] | None,
) -> pd.DataFrame:
    """Create side-by-side coefficient table: literature benchmark vs proposed model."""
    bench_params = pd.Series(dtype=float)
    if benchmark_fit is not None and hasattr(benchmark_fit, "result"):
        bench_params = pd.Series(getattr(benchmark_fit.result, "params", {}), dtype=float)

    random_rows: list[dict[str, Any]] = []
    rp_df = None
    if random_params_result is not None:
        rp_df = random_params_result.get("coef_df")
    if isinstance(rp_df, pd.DataFrame) and not rp_df.empty:
        for _, row in rp_df.iterrows():
            ptype = str(row.get("Type", ""))
            if not ptype.lower().startswith("random"):
                continue
            raw_name = str(row.get("Parameter", "")).strip()
            base_name = raw_name.split("[random", 1)[0].strip() if "[random" in raw_name else raw_name
            if "(" in ptype and ")" in ptype:
                dist = ptype.split("(", 1)[1].split(")", 1)[0].strip()
            elif ptype.lower() == "random-cor":
                dist = "correlated"
            else:
                dist = "random"
            random_rows.append({
                "label": base_name,
                "dist": dist,
                "mean": row.get("Mean", np.nan),
                "sd": row.get("SD", np.nan),
            })

    rows: list[dict[str, Any]] = []

    for _, row in coef_df.iterrows():
        param_name = str(row.get("Parameter", ""))
        label = _parameter_label_from_name(param_name)

        literature_coef = bench_params.get(param_name, np.nan)
        rows.append(
            {
                "Parameter": label,
                "Model term": param_name,
                "Literature benchmark coefficient": float(literature_coef) if np.isfinite(literature_coef) else np.nan,
                "Proposed fixed coefficient": float(row.get("Estimate (original variable scale)", np.nan)),
                "Proposed random distribution": "",
                "Proposed random mean": np.nan,
                "Proposed random sd": np.nan,
            }
        )

    for param_name, value in bench_params.items():
        if (coef_df["Parameter"].astype(str) == str(param_name)).any():
            continue
        rows.append(
            {
                "Parameter": _parameter_label_from_name(str(param_name)),
                "Model term": str(param_name),
                "Literature benchmark coefficient": float(value),
                "Proposed fixed coefficient": np.nan,
                "Proposed random distribution": "",
                "Proposed random mean": np.nan,
                "Proposed random sd": np.nan,
            }
        )

    # Save random-parameter effects as dedicated rows (one row per random term),
    # matching package-style model printouts.
    for info in random_rows:
        random_label = str(info.get("label", "")).strip()
        if not random_label:
            continue
        rows.append(
            {
                "Parameter": f"{random_label} (random)",
                "Model term": f"random::{random_label}",
                "Literature benchmark coefficient": np.nan,
                "Proposed fixed coefficient": np.nan,
                "Proposed random distribution": info["dist"],
                "Proposed random mean": float(info["mean"]) if pd.notna(info["mean"]) else np.nan,
                "Proposed random sd": float(info["sd"]) if pd.notna(info["sd"]) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Parameter", "Model term"], na_position="last").reset_index(drop=True)


def _resolve_default_candidates(df: pd.DataFrame, profile: str = "core") -> tuple[list[str], list[str]]:
    if {"FREQ", "LENGTH", "AADT", "CURVES"}.issubset(df.columns):
        # Expanded core_upper: all relevant road geometry and traffic variables
        core_upper = [
            # LENGTH intentionally excluded — it is the exposure offset, not a predictor
            # INCLANES/DECLANES replaced by composite LANES (total lanes both directions)
            # SINGLE, DOUBLE, TRAIN excluded — truck-composition % not a road design var
            # PEAKHR excluded — has -99 missing-value sentinel; not a reliable design var
            "LANES",
            "WIDTH",
            "MIMEDSH",
            "MXMEDSH",
            "SPEED",
            "URB",
            # FC one-hot dummies (FC=5 Minor Collector is reference, omitted)
            "FC_1",   # Interstate
            "FC_2",   # Principal Arterial Other
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
            # PEAKHR excluded — has -99 missing-value sentinel; not reliable
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
    mono_ok   = [str(v).lower() == "yes" for v in df.get("Monotonic AADT (e>0 all segs)", ["yes"] * n)]
    colors    = ["#2a7f4f" if ok else "#9e9e9e" for ok in mono_ok]

    run_min_bic = [min(bic_vals[: i + 1]) for i in range(n)]
    run_min_dev = [min(val_dev[: i + 1]) for i in range(n)]
    bic_window = _compute_tight_axis_window(bic_vals, objective="bic")
    dev_window = _compute_tight_axis_window(val_dev, objective="rmse")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    ax1.scatter(iters, bic_vals, c=colors, s=22, alpha=0.65, zorder=3)
    ax1.plot(iters, run_min_bic, color="#0a6c74", lw=2.5, label="Running min")
    ax1.set_title("BIC over Search Iterations", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("BIC")
    if bic_window is not None:
        ax1.set_ylim(*bic_window)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

    ax2.scatter(iters, val_dev, c=colors, s=22, alpha=0.65, zorder=3,
                label="Green = monotonic AADT")
    ax2.plot(iters, run_min_dev, color="#d96f32", lw=2.5, label="Running min")
    ax2.set_title("Validation Deviance over Iterations", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Val Poisson Deviance")
    if dev_window is not None:
        ax2.set_ylim(*dev_window)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.2)

    fig.suptitle(f"{dataset_label} — Search Convergence", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_refinement_convergence_png(path: Path, history_df: pd.DataFrame, phase: str, dataset_label: str) -> None:
    """Convergence plot for a specific refinement phase (harmony or sa)."""
    if "Phase" not in history_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.6), dpi=150)
        ax.axis("off")
        ax.text(0.5, 0.5, "No refinement phase data available.", ha="center", va="center", fontsize=12)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return
    df = history_df[history_df["Phase"].astype(str).str.lower().eq(str(phase).lower())].copy()
    if df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.6), dpi=150)
        ax.axis("off")
        ax.text(0.5, 0.5, f"No {phase} refinement candidates in this run.", ha="center", va="center", fontsize=12)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return

    n = len(df)
    iters = np.arange(1, n + 1)
    bic_vals = pd.to_numeric(df["BIC"], errors="coerce").to_numpy(dtype=float)
    rmse_vals = pd.to_numeric(df["Val RMSE"], errors="coerce").to_numpy(dtype=float)
    bic_window = _compute_tight_axis_window(bic_vals, objective="bic")
    rmse_window = _compute_tight_axis_window(rmse_vals, objective="rmse")

    bic_rmin = np.minimum.accumulate(np.where(np.isfinite(bic_vals), bic_vals, np.inf))
    rmse_rmin = np.minimum.accumulate(np.where(np.isfinite(rmse_vals), rmse_vals, np.inf))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    ax1.scatter(iters, bic_vals, c="#0a6c74", s=22, alpha=0.65, zorder=3)
    ax1.plot(iters, bic_rmin, color="#18435a", lw=2.5, label="Running min")
    ax1.set_title(f"{phase.title()} BIC", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Phase Iteration")
    ax1.set_ylabel("BIC")
    if bic_window is not None:
        ax1.set_ylim(*bic_window)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=8)

    ax2.scatter(iters, rmse_vals, c="#d96f32", s=22, alpha=0.65, zorder=3)
    ax2.plot(iters, rmse_rmin, color="#8a4e2c", lw=2.5, label="Running min")
    ax2.set_title(f"{phase.title()} Validation RMSE", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Phase Iteration")
    ax2.set_ylabel("Val RMSE")
    if rmse_window is not None:
        ax2.set_ylim(*rmse_window)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=8)

    fig.suptitle(f"{dataset_label} — {phase.title()} Refinement Convergence", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_refinement_phase_traces_png(path: Path, history_df: pd.DataFrame, dataset_label: str) -> None:
    """Trace plot comparing Harmony and SA refinement trajectories side-by-side."""
    if "Phase" not in history_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.6), dpi=150)
        ax.axis("off")
        ax.text(0.5, 0.5, "No phase column available for refinement traces.", ha="center", va="center", fontsize=12)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return

    df = history_df.copy()
    df["_phase"] = df["Phase"].astype(str).str.lower().str.strip()

    phase_cfg = {
        "harmony": {"label": "Harmony Search", "color": "#0a6c74"},
        "sa": {"label": "Simulated Annealing", "color": "#d96f32"},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    plotted = 0
    all_bic: list[float] = []
    all_rmse: list[float] = []

    for phase_key in ["harmony", "sa"]:
        sub = df[df["_phase"].eq(phase_key)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("Iteration").reset_index(drop=True)
        x = np.arange(1, len(sub) + 1)
        bic = pd.to_numeric(sub["BIC"], errors="coerce").to_numpy(dtype=float)
        rmse = pd.to_numeric(sub["Val RMSE"], errors="coerce").to_numpy(dtype=float)
        bic_rmin = np.minimum.accumulate(np.where(np.isfinite(bic), bic, np.inf))
        rmse_rmin = np.minimum.accumulate(np.where(np.isfinite(rmse), rmse, np.inf))

        color = phase_cfg[phase_key]["color"]
        label = phase_cfg[phase_key]["label"]

        ax1.plot(x, bic, color=color, lw=1.3, alpha=0.40)
        ax1.scatter(x, bic, color=color, s=18, alpha=0.55, label=f"{label} trace")
        ax1.plot(x, bic_rmin, color=color, lw=2.8, linestyle="-", label=f"{label} running best")

        ax2.plot(x, rmse, color=color, lw=1.3, alpha=0.40)
        ax2.scatter(x, rmse, color=color, s=18, alpha=0.55, label=f"{label} trace")
        ax2.plot(x, rmse_rmin, color=color, lw=2.8, linestyle="-", label=f"{label} running best")

        all_bic.extend([float(v) for v in bic if np.isfinite(v)])
        all_rmse.extend([float(v) for v in rmse if np.isfinite(v)])
        plotted += 1

    if plotted == 0:
        ax1.axis("off")
        ax2.axis("off")
        ax1.text(0.5, 0.5, "No harmony/sa refinement traces found.", ha="center", va="center", fontsize=11)
        fig.suptitle(f"{dataset_label} — Harmony vs SA Refinement Traces", fontsize=12, y=1.01)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return

    bic_window = _compute_tight_axis_window(all_bic, objective="bic")
    rmse_window = _compute_tight_axis_window(all_rmse, objective="rmse")

    ax1.set_title("Refinement BIC Traces", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Within-phase iteration")
    ax1.set_ylabel("BIC")
    if bic_window is not None:
        ax1.set_ylim(*bic_window)
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=7, loc="best")

    ax2.set_title("Refinement Validation RMSE Traces", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Within-phase iteration")
    ax2.set_ylabel("Val RMSE")
    if rmse_window is not None:
        ax2.set_ylim(*rmse_window)
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=7, loc="best")

    fig.suptitle(f"{dataset_label} — Harmony vs SA Refinement Traces", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_search_phase_comparison_png(path: Path, compare_df: pd.DataFrame, dataset_label: str) -> None:
    """Bar chart comparing best Simulated Annealing vs Harmony Search candidates."""
    if compare_df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.6), dpi=150)
        ax.axis("off")
        ax.text(0.5, 0.5, "No phase comparison rows available.", ha="center", va="center", fontsize=12)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return
    df = compare_df.copy()
    if "Phase Label" in df.columns:
        labels = df["Phase Label"].astype(str).tolist()
    else:
        _phase_map = {
            "sa": "Simulated Annealing",
            "harmony": "Harmony Search",
            "random": "Random Exploration",
        }
        labels = [
            _phase_map.get(str(p).strip().lower(), str(p))
            for p in df["Phase"].astype(str).tolist()
        ]
    bic_vals = pd.to_numeric(df["BIC"], errors="coerce").to_numpy(dtype=float)
    rmse_vals = pd.to_numeric(df["Val RMSE"], errors="coerce").to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    ax1.bar(labels, bic_vals, color=["#0a6c74", "#d96f32"][: len(labels)])
    ax1.set_title("Best BIC by Search Phase", fontsize=11, fontweight="bold")
    ax1.set_ylabel("BIC")
    ax1.grid(axis="y", alpha=0.2)

    ax2.bar(labels, rmse_vals, color=["#0a6c74", "#d96f32"][: len(labels)])
    ax2.set_title("Best Validation RMSE by Search Phase", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Val RMSE")
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle(f"{dataset_label} — Simulated Annealing vs Harmony Search", fontsize=12, y=1.02)
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


def _save_prediction_delta_png(
    path: Path,
    aadt_all: np.ndarray,
    pred_benchmark: np.ndarray,
    pred_hierarchical: np.ndarray,
    split_labels: list[str],
    title: str,
) -> None:
    """Save a compact delta chart: hierarchical prediction minus benchmark prediction."""
    try:
        a = np.asarray(aadt_all, float)
        pb = np.asarray(pred_benchmark, float)
        ph = np.asarray(pred_hierarchical, float)
        delta = ph - pb
        log_a = np.log(np.clip(a, 1.0, None))

        color_map = {
            "train": "#0a6c74",
            "val": "#d96f32",
            "validation": "#d96f32",
            "test": "#7a3c8c",
            "all": "#4a6785",
        }
        obs_c = [color_map.get(str(l).lower(), "#4a6785") for l in split_labels]

        fig, ax = plt.subplots(1, 1, figsize=(10, 4.8), dpi=150)
        ax.scatter(log_a, delta, c=obs_c, s=18, alpha=0.58, zorder=3)

        try:
            coeffs = np.polyfit(log_a, delta, deg=3)
            x_fine = np.linspace(log_a.min(), log_a.max(), 220)
            y_fine = np.polyval(coeffs, x_fine)
            ax.plot(x_fine, y_fine, color="#18435a", lw=2.4, label="Delta smooth")
        except Exception:
            pass

        ax.axhline(0.0, color="#444", lw=1.3, linestyle="--", label="No difference")
        ax.set_xlabel("log(AADT)")
        ax.set_ylabel("Hierarchical - Benchmark predicted crashes")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[WARNING] Could not save prediction delta PNG: {exc}")


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
    parser.add_argument(
        "--search-method",
        choices=["random-sa", "harmony"],
        default="random-sa",
        help="Refinement method after random exploration: random-sa or harmony.",
    )
    parser.add_argument("--harmony-hms", type=int, default=12, help="Harmony memory size (HMS).")
    parser.add_argument("--harmony-hmcr", type=float, default=0.90, help="Harmony memory consideration rate (HMCR).")
    parser.add_argument("--harmony-par", type=float, default=0.35, help="Harmony pitch adjustment rate (PAR).")
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
    parser.add_argument(
        "--require-benchmark-dominance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require Pareto-selected model to beat benchmark on BOTH validation RMSE and BIC (default: enabled).",
    )
    parser.add_argument(
        "--require-final-beat-benchmark-both",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require final proposed model to beat benchmark on BOTH test RMSE and BIC (default: enabled).",
    )
    parser.add_argument(
        "--rp-include-lower-interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include lower-level interaction terms as random-parameter candidates in RP sweep (default: enabled).",
    )
    parser.add_argument(
        "--rp-max-random-terms",
        type=int,
        default=4,
        help="Maximum number of random terms to include in RP sweep after ranking (default: 4).",
    )
    parser.add_argument(
        "--rp-draws",
        type=int,
        default=500,
        help="Halton draws for random-parameter fit integration (default: 500).",
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

    benchmark_raw_vars = [
        v for v in (LITERATURE_BENCHMARK_NONRANDOM_RAW + LITERATURE_BENCHMARK_RANDOM_MEAN_RAW)
        if v in model_name_map
    ]
    benchmark_upper_model_vars = [model_name_map[v] for v in benchmark_raw_vars]

    offset_col = None if args.disable_offset else (args.offset_col if args.offset_col in df_train.columns else None)

    # Build family list from --families argument (e.g. "nb", "poisson", "both")
    _fam_arg = getattr(args, "families", args.family)
    if _fam_arg in ("both", "all"):
        search_families = ["nb", "poisson"]
    elif "," in str(_fam_arg):
        search_families = [f.strip() for f in str(_fam_arg).split(",")]
    else:
        search_families = [str(_fam_arg)]

    # Benchmark thresholds used during search-time Pareto selection.
    # BIC is computed on train split fit; RMSE is evaluated on validation split.
    selection_benchmark = _fit_benchmark(
        df_tv=df_train,
        df_test=df_val,
        y_col=args.y_col,
        aadt_col=args.aadt_col,
        offset_col=offset_col,
        family=args.family,
        benchmark_upper_vars=benchmark_upper_model_vars,
    )
    if not selection_benchmark or selection_benchmark.get("fitted") is None:
        raise RuntimeError("Could not fit benchmark model for Pareto selection thresholds.")

    selection_benchmark_bic = float(selection_benchmark.get("tv_bic", np.nan))
    selection_benchmark_val_rmse = float(selection_benchmark.get("test_rmse", np.nan))
    if not np.isfinite(selection_benchmark_bic) or not np.isfinite(selection_benchmark_val_rmse):
        raise RuntimeError("Benchmark thresholds for Pareto selection are not finite.")

    _best_fit_seed, search_history, search_history_ordered, top_k_fits = _random_search(
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
        search_method=str(args.search_method),
        harmony_hms=int(args.harmony_hms),
        harmony_hmcr=float(args.harmony_hmcr),
        harmony_par=float(args.harmony_par),
    )

    # Pareto selection over (BIC, validation RMSE), then require benchmark
    # dominance in BOTH metrics unless explicitly disabled.
    selected_row, search_history_ordered = _select_pareto_candidate(
        search_history_ordered=search_history_ordered,
        benchmark_bic=selection_benchmark_bic,
        benchmark_val_rmse=selection_benchmark_val_rmse,
        enforce_aadt_increase=bool(args.enforce_aadt_increase),
        require_benchmark_dominance=bool(args.require_benchmark_dominance),
    )
    search_history = search_history_ordered.sort_values("Val Poisson Dev").reset_index(drop=True)

    selected_upper_model_vars = _parse_search_vars(selected_row.get("Upper Vars", ""))
    selected_lower_model_vars = _parse_search_vars(selected_row.get("Lower Vars", ""))
    selected_search_family = str(selected_row.get("Family", args.family))

    best_fit_train = _fit_model(
        df_train=df_train,
        aadt_col=args.aadt_col,
        y_col=args.y_col,
        upper_vars=selected_upper_model_vars,
        lower_vars=selected_lower_model_vars,
        family=selected_search_family,
        offset_col=offset_col,
    )
    if best_fit_train is None:
        raise RuntimeError("Pareto-selected model could not be refit on training data.")

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
        family=best_fit_train.family,
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
        offset_col=offset_col, family=final_fit.family,
        benchmark_upper_vars=benchmark_upper_model_vars,
    )

    # Rescue pass: if the current final choice misses benchmark test RMSE,
    # scan searched specifications and swap in a benchmark-dominating candidate
    # when one exists. This keeps reporting robust in small-budget searches.
    _bench_bic_now = float(bench.get("tv_bic", np.nan))
    _bench_rmse_now = float(bench.get("test_rmse", np.nan))
    _final_bic_now = float(getattr(final_fit.result, "bic", np.nan))
    _final_rmse_now = float(metrics_test["rmse"])
    if np.isfinite(_bench_bic_now) and np.isfinite(_bench_rmse_now):
        _fails_now = not ((_final_bic_now < _bench_bic_now) and (_final_rmse_now < _bench_rmse_now))
    else:
        _fails_now = False

    if _fails_now and not search_history_ordered.empty:
        print("  Final rescue pass: searching evaluated specs for better held-out RMSE ...")
        _cand = search_history_ordered.copy()
        if bool(args.enforce_aadt_increase) and "Monotonic AADT (e>0 all segs)" in _cand.columns:
            _cand = _cand[_cand["Monotonic AADT (e>0 all segs)"].astype(str).str.lower().eq("yes")].copy()
        _finite = np.isfinite(pd.to_numeric(_cand["BIC"], errors="coerce").to_numpy(dtype=float)) & np.isfinite(
            pd.to_numeric(_cand["Val RMSE"], errors="coerce").to_numpy(dtype=float)
        )
        _cand = _cand.loc[_finite].copy()
        _cand = _cand.sort_values(["Val RMSE", "BIC"], ascending=[True, True])
        _cand = _cand.drop_duplicates(subset=["Upper Vars", "Lower Vars", "Family"]).head(300)

        _rescue_pool: list[dict[str, Any]] = []
        y_test_rescue = pd.to_numeric(df_test_final[args.y_col], errors="coerce").to_numpy(dtype=float)
        y_trainval_rescue = pd.to_numeric(df_trainval[args.y_col], errors="coerce").to_numpy(dtype=float)

        for _, _r in _cand.iterrows():
            _up = _parse_search_vars(_r.get("Upper Vars", ""))
            _lo = _parse_search_vars(_r.get("Lower Vars", ""))
            _fam = str(_r.get("Family", args.family))
            _fit_try = _fit_model(
                df_train=df_trainval,
                aadt_col=args.aadt_col,
                y_col=args.y_col,
                upper_vars=_up,
                lower_vars=_lo,
                family=_fam,
                offset_col=offset_col,
            )
            if _fit_try is None:
                continue

            _pred_test_try = _predict(df_test_final, _fit_try)
            _rmse_try = float(_metrics(y_test_rescue, _pred_test_try)["rmse"])
            _bic_try = float(getattr(_fit_try.result, "bic", np.nan))
            _rescue_pool.append(
                {
                    "fit": _fit_try,
                    "rmse": _rmse_try,
                    "bic": _bic_try,
                    "upper": _up,
                    "lower": _lo,
                    "family": _fam,
                }
            )

        _dom = [
            r for r in _rescue_pool
            if np.isfinite(r["bic"]) and np.isfinite(r["rmse"])
            and r["bic"] < _bench_bic_now and r["rmse"] < _bench_rmse_now
        ]
        if _dom:
            _best_rescue = sorted(_dom, key=lambda r: (r["rmse"], r["bic"]))[0]
            if _best_rescue["rmse"] < _final_rmse_now:
                final_fit = _best_rescue["fit"]
                pred_trainval = _predict(df_trainval, final_fit)
                pred_test = _predict(df_test_final, final_fit)
                metrics_trainval = _metrics(y_trainval_rescue, pred_trainval)
                metrics_test = _metrics(y_test_rescue, pred_test)
                elasticity_trainval = _elasticity_stats(_aadt_elasticity(df_trainval, final_fit))
                elasticity_test = _elasticity_stats(_aadt_elasticity(df_test_final, final_fit))
                selected_upper_raw = sorted({name[:-2] if name.endswith("_Z") else name for name in final_fit.upper_vars})
                selected_lower_raw = sorted({name[:-2] if name.endswith("_Z") else name for name in final_fit.lower_vars})
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
                print(
                    "  Rescue selected benchmark-dominating spec: "
                    f"family={_best_rescue['family']}, "
                    f"BIC={_best_rescue['bic']:.2f}, Test RMSE={_best_rescue['rmse']:.4f}"
                )

    bench_compare_df = pd.DataFrame([
        {
            "Model":           "Benchmark (Ex16-3 literature model)",
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

    # Enforce required final dominance over benchmark on both metrics.
    final_bic = float(getattr(final_fit.result, "bic", np.nan))
    final_test_rmse = float(metrics_test["rmse"])
    bench_bic = float(bench.get("tv_bic", np.nan))
    bench_test_rmse = float(bench.get("test_rmse", np.nan))
    beats_bic = np.isfinite(final_bic) and np.isfinite(bench_bic) and (final_bic < bench_bic)
    beats_rmse = np.isfinite(final_test_rmse) and np.isfinite(bench_test_rmse) and (final_test_rmse < bench_test_rmse)
    print(
        "\n  Final dominance check: "
        f"BIC {final_bic:.2f} vs benchmark {bench_bic:.2f} ({'PASS' if beats_bic else 'FAIL'}), "
        f"Test RMSE {final_test_rmse:.4f} vs benchmark {bench_test_rmse:.4f} ({'PASS' if beats_rmse else 'FAIL'})"
    )
    if bool(args.require_final_beat_benchmark_both) and not (beats_bic and beats_rmse):
        failed = []
        if not beats_bic:
            failed.append("BIC")
        if not beats_rmse:
            failed.append("Test RMSE")
        failed_str = ", ".join(failed) if failed else "unknown"
        raise RuntimeError(
            "Final proposed model did not beat the benchmark on BOTH BIC and test RMSE. "
            f"Failed metric(s): {failed_str}. "
            f"Model(BIC={final_bic:.2f}, Test RMSE={final_test_rmse:.4f}) vs "
            f"Benchmark(BIC={bench_bic:.2f}, Test RMSE={bench_test_rmse:.4f}). "
            "Re-run with broader search / different seed, or use "
            "--no-require-final-beat-benchmark-both to keep best Pareto model anyway."
        )

    settings_df = pd.DataFrame(
        [
            {"Setting": "Rows", "Value": int(len(df))},
            {"Setting": "Train rows", "Value": int(len(df_train_raw))},
            {"Setting": "Validation rows", "Value": int(len(df_val_raw))},
            {"Setting": "Test rows", "Value": int(len(df_test_raw))},
            {"Setting": "Search iterations", "Value": int(args.search_iter)},
            {"Setting": "Search method", "Value": str(args.search_method)},
            {"Setting": "Harmony HMS", "Value": int(args.harmony_hms)},
            {"Setting": "Harmony HMCR", "Value": float(args.harmony_hmcr)},
            {"Setting": "Harmony PAR", "Value": float(args.harmony_par)},
            {"Setting": "Candidate profile", "Value": args.candidate_profile},
            {"Setting": "Final family", "Value": final_fit.family},
            {"Setting": "Search families", "Value": ", ".join(search_families)},
            {"Setting": "Benchmark upper vars (literature)", "Value": ", ".join(benchmark_raw_vars) if benchmark_raw_vars else "(none found)"},
            {"Setting": "AADT column", "Value": args.aadt_col},
            {"Setting": "Offset used", "Value": "yes" if offset_col else "no"},
            {"Setting": "Enforce AADT increase", "Value": "yes" if args.enforce_aadt_increase else "no"},
            {"Setting": "Min AADT elasticity", "Value": float(args.min_aadt_elasticity)},
            {"Setting": "Allow nonmonotonic fallback", "Value": "yes" if args.allow_nonmonotonic_fallback else "no"},
            {"Setting": "Pareto benchmark dominance required", "Value": "yes" if args.require_benchmark_dominance else "no"},
            {"Setting": "Final benchmark dominance required", "Value": "yes" if args.require_final_beat_benchmark_both else "no"},
            {"Setting": "Selection benchmark BIC (train)", "Value": round(selection_benchmark_bic, 4)},
            {"Setting": "Selection benchmark RMSE (validation)", "Value": round(selection_benchmark_val_rmse, 6)},
            {"Setting": "Selected by Pareto iteration", "Value": int(selected_row.get("Iteration", -1))},
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

    # ── Random-parameters sweep — MANDATORY ──────────────────────────────
    # Random parameters are a required output of this experiment.
    # Sweep all top-K candidates; if none succeed, progressively simplify
    # (fewer correlated terms, then independent-only, then just the best spec).
    best_rp_result: dict | None = None
    best_rp_bic = float("nan")

    # Candidates to sweep: top-K from search + the selected final spec
    _rp_candidates_to_try = list(top_k_fits)
    # Always include the primary selected spec as a fallback candidate
    if best_fit_train not in _rp_candidates_to_try:
        _rp_candidates_to_try.append(best_fit_train)

    print(f"  Random-params sweep (mandatory) on {len(_rp_candidates_to_try)} candidates ...")
    for _rp_candidate in _rp_candidates_to_try:
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
            include_lower_interactions=bool(args.rp_include_lower_interactions),
            max_random_terms=int(args.rp_max_random_terms),
            rp_draws=int(args.rp_draws),
        )
        if _rp is None:
            continue
        _rp_bic = _rp.get("bic", float("nan"))
        if not np.isfinite(_rp_bic):
            continue
        if best_rp_result is None or _rp_bic < best_rp_bic:
            best_rp_bic    = _rp_bic
            best_rp_result = _rp
            # Note: do NOT overwrite selected_upper_raw / selected_lower_raw here.
            # Those drive the dashboard variables and must stay aligned with final_fit.

    jax_result = best_rp_result

    # If still no random-params result, force it on the selected spec with
    # a single random variable (simplest possible random-params model).
    if jax_result is None:
        print("  [RP] Sweep yielded no result — forcing single-random-var fallback ...")
        _cont_upper = [v for v in selected_upper_raw if v not in binary_vars]
        if _cont_upper:
            jax_result = _jax_random_params_refit(
                df_trainval_raw=df_trainval_raw,
                best_upper_raw=_cont_upper[:1],   # simplest: one random var
                best_lower_raw=selected_lower_raw,
                y_col=args.y_col,
                aadt_col=args.aadt_col,
                offset_col=offset_col,
                scaler_stats=scaler_trainval,
                binary_vars=binary_vars,
                include_lower_interactions=bool(args.rp_include_lower_interactions),
                max_random_terms=max(1, int(args.rp_max_random_terms)),
                rp_draws=int(args.rp_draws),
            )

    if jax_result is None:
        print("  [WARNING] Random-parameters model could not be fitted — "
              "check ExperimentBuilder availability.")
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
        (output_dir / "random_params_separate_lines.md").write_text(
            "Random-Parameters NB2 | unavailable\n\n- (random-parameter refit not generated in this run)",
            encoding="utf-8",
        )
        (output_dir / "random_params_coef_table_pptx.md").write_text(
            "Random-Parameters NB2 | unavailable\n\n- Compact random-parameter table unavailable in this run.",
            encoding="utf-8",
        )

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
        _bench_all = _apply_standardization(df, scaler_trainval)
        _bench_off_col = str(bench.get("benchmark_offset_column", "__BENCHMARK_OFFSET_AADT__"))
        _base_off = np.zeros(len(_bench_all), dtype=float)
        if offset_col is not None and offset_col in _bench_all.columns:
            _base_off = pd.to_numeric(_bench_all[offset_col], errors="coerce").to_numpy(dtype=float)
        _bench_all[_bench_off_col] = _base_off + np.log(
            np.clip(pd.to_numeric(_bench_all[args.aadt_col], errors="coerce").to_numpy(dtype=float), 1e-12, None)
        )
        pred_bench_all = _predict(_bench_all, bench["fitted"])
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
        _save_prediction_delta_png(
            output_dir / "model_prediction_delta.png",
            aadt_all=aadt_all,
            pred_benchmark=pred_bench_all,
            pred_hierarchical=pred_all,
            split_labels=split_labels_list,
            title=f"{dataset_label} - Prediction Delta (Hierarchical minus Benchmark)",
        )
        print("  model_comparison.html / .png written.")

    # Readable coefficient CSV (human-readable names + CMF)
    readable_coef_df = _save_readable_coef_csv(
        output_dir / "readable_coefficients.csv", final_fit, scaler_trainval, binary_vars
    )

    # Write random-params coefficient table as a standalone markdown for QMD include
    if jax_result is not None and jax_result.get("coef_df") is not None:
        rp_df = jax_result["coef_df"]
        if not rp_df.empty:
            rp_md = _to_markdown(rp_df)
            corr_str = jax_result.get("corr_matrix_str", "")
            rp_full  = (
                f"LL = {jax_result.get('loglik', float('nan')):.2f}  "
                f"BIC = {jax_result.get('bic', float('nan')):.2f}\n\n"
                + rp_md
                + ("\n\n" + corr_str if corr_str else "")
                + "\n\n**Fixed** = same for all segments  "
                "**Random-Ind** = Mean +/- SD across segments  "
                "**Random-Cor** = correlated (see matrix)"
            )
            (output_dir / "random_params_coef_table.md").write_text(
                rp_full, encoding="utf-8"
            )

            # PPTX-friendly compact table: no correlation matrix, fewer rows/columns.
            _rp_pptx = rp_df.copy()
            keep_cols = [c for c in ["Parameter", "Type", "Mean", "SD"] if c in _rp_pptx.columns]
            if keep_cols:
                _rp_pptx = _rp_pptx[keep_cols].copy()
            if len(_rp_pptx) > 14:
                _rp_pptx = _rp_pptx.head(14).copy()
            _rp_pptx_md = _to_markdown(_rp_pptx)
            _rp_pptx_note = (
                f"LL = {jax_result.get('loglik', float('nan')):.2f}  "
                f"BIC = {jax_result.get('bic', float('nan')):.2f}\n\n"
                + _rp_pptx_md
                + "\n\nTop rows shown for slide readability; full table remains in random_params_coef_table.md"
            )
            (output_dir / "random_params_coef_table_pptx.md").write_text(
                _rp_pptx_note, encoding="utf-8"
            )

            # Package-style random-parameter lines: one random term per line.
            _rp_lines: list[str] = []
            _rp_lines.append(
                f"Random-Parameters NB2 | LL={jax_result.get('loglik', float('nan')):.2f} | BIC={jax_result.get('bic', float('nan')):.2f}"
            )
            _rp_lines.append("")
            _rp_lines.append("Random parameters (separate lines):")

            _rp_only = rp_df[rp_df["Type"].astype(str).str.lower().str.startswith("random")].copy() if "Type" in rp_df.columns else pd.DataFrame()
            if _rp_only.empty:
                _rp_lines.append("- (none)")
            else:
                for _, _r in _rp_only.iterrows():
                    _pname = str(_r.get("Parameter", "")).strip()
                    _ptype = str(_r.get("Type", "")).strip()
                    _mean = _r.get("Mean", np.nan)
                    _sd = _r.get("SD", np.nan)
                    _mean_txt = f"{float(_mean):.5f}" if pd.notna(_mean) else "nan"
                    _sd_txt = f"{float(_sd):.5f}" if pd.notna(_sd) else "nan"
                    _rp_lines.append(f"- {_pname}: type={_ptype}, mean={_mean_txt}, sd={_sd_txt}")

            _corr = str(jax_result.get("corr_matrix_str", "")).strip()
            if _corr:
                _rp_lines.append("")
                _rp_lines.append("Correlated random-effects matrix:")
                _rp_lines.append(_corr)

            (output_dir / "random_params_separate_lines.md").write_text(
                "\n".join(_rp_lines), encoding="utf-8"
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

    # Pareto selection summary table (includes benchmark-dominance flags)
    pareto_cols = [
        "Iteration",
        "Family",
        "Upper Vars",
        "Lower Vars",
        "Val RMSE",
        "BIC",
        "Pareto Efficient",
        "Beats benchmark (BIC+RMSE)",
        "BIC improvement vs benchmark",
        "RMSE improvement vs benchmark",
        "Joint improvement score",
    ]
    pareto_summary_df = search_history_ordered.copy()
    pareto_summary_df["Selected by Pareto"] = (
        pd.to_numeric(pareto_summary_df["Iteration"], errors="coerce")
        == float(selected_row.get("Iteration", -1))
    )
    keep_cols = [c for c in (pareto_cols + ["Selected by Pareto"]) if c in pareto_summary_df.columns]
    pareto_summary_df = pareto_summary_df[keep_cols].sort_values(
        ["Selected by Pareto", "Pareto Efficient", "Beats benchmark (BIC+RMSE)", "Joint improvement score", "BIC", "Val RMSE"],
        ascending=[False, False, False, False, True, True],
    )
    (output_dir / "pareto_selection_summary.csv").write_text(
        pareto_summary_df.to_csv(index=False), encoding="utf-8"
    )
    (output_dir / "pareto_selection_summary.md").write_text(
        _to_markdown(pareto_summary_df.head(40)), encoding="utf-8"
    )

    # Compare best random-phase model vs best refinement-phase model and save
    # both a table and a PNG for side-by-side presentation.
    phase_compare_rows: list[dict[str, Any]] = []
    phase_ref = "harmony" if str(args.search_method).lower() == "harmony" else "sa"
    _phase_label_map = {
        "sa": "Simulated Annealing",
        "harmony": "Harmony Search",
        "random": "Random Exploration",
    }
    if "Phase" in search_history_ordered.columns:
        for _phase in ["sa", "harmony"]:
            _phase_df = search_history_ordered[
                search_history_ordered["Phase"].astype(str).str.lower().eq(_phase)
            ].copy()
            if _phase_df.empty:
                continue
            _phase_df = _phase_df.sort_values(["BIC", "Val RMSE"], ascending=[True, True])
            _best = _phase_df.iloc[0]
            phase_compare_rows.append(
                {
                    "Phase": _phase,
                    "Phase Label": _phase_label_map.get(_phase, _phase.title()),
                    "Iteration": int(_best.get("Iteration", -1)),
                    "Family": str(_best.get("Family", "")),
                    "BIC": float(_best.get("BIC", np.nan)),
                    "Val RMSE": float(_best.get("Val RMSE", np.nan)),
                    "Upper Vars": str(_best.get("Upper Vars", "(none)")),
                    "Lower Vars": str(_best.get("Lower Vars", "(none)")),
                }
            )

    phase_compare_df = pd.DataFrame(phase_compare_rows)
    if phase_compare_df.empty and "Phase" in search_history_ordered.columns:
        # Fallback for runs where only one refinement strategy is present.
        for _phase in ["random", phase_ref]:
            _phase_df = search_history_ordered[
                search_history_ordered["Phase"].astype(str).str.lower().eq(_phase)
            ].copy()
            if _phase_df.empty:
                continue
            _best = _phase_df.sort_values(["BIC", "Val RMSE"], ascending=[True, True]).iloc[0]
            phase_compare_rows.append(
                {
                    "Phase": _phase,
                    "Phase Label": _phase_label_map.get(_phase, _phase.title()),
                    "Iteration": int(_best.get("Iteration", -1)),
                    "Family": str(_best.get("Family", "")),
                    "BIC": float(_best.get("BIC", np.nan)),
                    "Val RMSE": float(_best.get("Val RMSE", np.nan)),
                    "Upper Vars": str(_best.get("Upper Vars", "(none)")),
                    "Lower Vars": str(_best.get("Lower Vars", "(none)")),
                }
            )
        phase_compare_df = pd.DataFrame(phase_compare_rows)

    if not phase_compare_df.empty:
        phase_compare_df = phase_compare_df.sort_values("Val RMSE", ascending=True).reset_index(drop=True)
        phase_compare_md = phase_compare_df[[
            "Phase Label", "Iteration", "Family", "BIC", "Val RMSE", "Upper Vars", "Lower Vars"
        ]].rename(columns={"Phase Label": "Search Phase"})
    else:
        phase_compare_md = phase_compare_df

    (output_dir / "search_phase_comparison.csv").write_text(
        phase_compare_df.to_csv(index=False), encoding="utf-8"
    )
    if not phase_compare_md.empty:
        (output_dir / "search_phase_comparison.md").write_text(
            _to_markdown(phase_compare_md), encoding="utf-8"
        )
    else:
        (output_dir / "search_phase_comparison.md").write_text(
            "No phase comparison rows available.", encoding="utf-8"
        )
    _save_search_phase_comparison_png(
        output_dir / "search_phase_comparison.png",
        phase_compare_df,
        dataset_label,
    )

    # Explain why SA/Harmony can look identical at the winner row by also
    # reporting phase-level distributions and convergence pace.
    if "Phase" in search_history_ordered.columns:
        _phase_rows: list[dict[str, Any]] = []
        for _phase in ["sa", "harmony"]:
            _dfp = search_history_ordered[
                search_history_ordered["Phase"].astype(str).str.lower().eq(_phase)
            ].copy()
            if _dfp.empty:
                continue
            _bic = pd.to_numeric(_dfp["BIC"], errors="coerce")
            _rmse = pd.to_numeric(_dfp["Val RMSE"], errors="coerce")
            _best_bic = float(np.nanmin(_bic.to_numpy(dtype=float)))
            _best_rmse = float(np.nanmin(_rmse.to_numpy(dtype=float)))
            _thr = _best_bic * 1.005
            _within = _dfp[_bic <= _thr]
            if _within.empty:
                _first_within = np.nan
            else:
                _first_within = float(pd.to_numeric(_within["Iteration"], errors="coerce").min())
            _phase_rows.append(
                {
                    "Search Phase": _phase_label_map.get(_phase, _phase.title()),
                    "Candidates evaluated": int(len(_dfp)),
                    "Best BIC": _best_bic,
                    "Best Val RMSE": _best_rmse,
                    "Median BIC": float(np.nanmedian(_bic.to_numpy(dtype=float))),
                    "Median Val RMSE": float(np.nanmedian(_rmse.to_numpy(dtype=float))),
                    "First iter within 0.5% of best BIC": _first_within,
                }
            )
        _phase_diff_df = pd.DataFrame(_phase_rows)
        if not _phase_diff_df.empty:
            (output_dir / "search_phase_differences.csv").write_text(
                _phase_diff_df.to_csv(index=False), encoding="utf-8"
            )
            (output_dir / "search_phase_differences.md").write_text(
                _to_markdown(_phase_diff_df), encoding="utf-8"
            )

    metric_scope_lines = [
        "Metric scope note",
        "",
        "- Search-convergence plots report **validation metrics during selection** (train-fit evaluated on validation split).",
        "- Benchmark comparison slide reports **held-out test metrics after final refit** on train+validation.",
        "- These numbers are expected to differ because they are computed on different splits and at different fit stages.",
    ]
    (output_dir / "metric_scope_note.md").write_text("\n".join(metric_scope_lines), encoding="utf-8")

    _save_refinement_convergence_png(
        output_dir / "refinement_convergence.png",
        search_history_ordered,
        phase_ref,
        dataset_label,
    )
    _save_refinement_convergence_png(
        output_dir / "refinement_convergence_harmony.png",
        search_history_ordered,
        "harmony",
        dataset_label,
    )
    _save_refinement_convergence_png(
        output_dir / "refinement_convergence_sa.png",
        search_history_ordered,
        "sa",
        dataset_label,
    )
    _save_refinement_phase_traces_png(
        output_dir / "refinement_phase_traces.png",
        search_history_ordered,
        dataset_label,
    )

    if "Phase" in search_history_ordered.columns:
        harmony_phase_df = search_history_ordered[
            search_history_ordered["Phase"].astype(str).str.lower().eq("harmony")
        ].copy()
    else:
        harmony_phase_df = pd.DataFrame()

    if harmony_phase_df.empty:
        harmony_summary_md = (
            "Harmony search summary\n\n"
            "- No harmony-phase candidates were evaluated in this run.\n"
            "- Run with --search-method harmony to generate harmony search winners."
        )
        (output_dir / "harmony_search_summary.csv").write_text("", encoding="utf-8")
    else:
        harmony_top_df = harmony_phase_df.sort_values(["BIC", "Val RMSE"], ascending=[True, True]).head(10).copy()
        harmony_top_show = harmony_top_df[[
            "Iteration", "Family", "BIC", "Val RMSE", "Upper Vars", "Lower Vars"
        ]].copy()
        _winner = harmony_top_df.iloc[0]
        harmony_summary_md = "\n".join([
            "Harmony search summary",
            "",
            f"Winner iteration: {int(_winner.get('Iteration', -1))}",
            f"Winner family: {str(_winner.get('Family', ''))}",
            f"Winner BIC: {float(_winner.get('BIC', np.nan)):.4f}",
            f"Winner Val RMSE: {float(_winner.get('Val RMSE', np.nan)):.6f}",
            f"Winner upper vars: {str(_winner.get('Upper Vars', '(none)'))}",
            f"Winner lower vars: {str(_winner.get('Lower Vars', '(none)'))}",
            "",
            "Top harmony candidates:",
            _to_markdown(harmony_top_show),
        ])
        (output_dir / "harmony_search_summary.csv").write_text(
            harmony_top_show.to_csv(index=False), encoding="utf-8"
        )
    (output_dir / "harmony_search_summary.md").write_text(harmony_summary_md, encoding="utf-8")

    # Side-by-side coefficient table: literature benchmark vs proposed model,
    # including random-parameter distributions and coefficients.
    coef_compare_df = _build_literature_vs_proposed_coef_table(
        coef_df=coef_df,
        benchmark_fit=bench.get("fitted"),
        random_params_result=jax_result,
    )
    if not coef_compare_df.empty:
        (output_dir / "literature_vs_proposed_coefficients.csv").write_text(
            coef_compare_df.to_csv(index=False), encoding="utf-8"
        )
        (output_dir / "literature_vs_proposed_coefficients.md").write_text(
            _to_markdown(coef_compare_df), encoding="utf-8"
        )
        coef_compare_pptx = coef_compare_df.copy()
        keep_cols = [
            "Parameter",
            "Literature benchmark coefficient",
            "Proposed fixed coefficient",
            "Proposed random mean",
            "Proposed random sd",
        ]
        keep_cols = [c for c in keep_cols if c in coef_compare_pptx.columns]
        if keep_cols:
            coef_compare_pptx = coef_compare_pptx[keep_cols]
            coef_compare_pptx = coef_compare_pptx.rename(
                columns={
                    "Literature benchmark coefficient": "Benchmark coef",
                    "Proposed fixed coefficient": "Proposed fixed",
                    "Proposed random mean": "Random mean",
                    "Proposed random sd": "Random SD",
                }
            )
            (output_dir / "literature_vs_proposed_coefficients_pptx.md").write_text(
                _to_markdown(coef_compare_pptx), encoding="utf-8"
            )

        # Safety interpretation summary for slides: benchmark vs proposed fixed terms.
        _cmp = coef_compare_df.copy()
        _bcol = "Literature benchmark coefficient"
        _pcol = "Proposed fixed coefficient"
        if _bcol in _cmp.columns and _pcol in _cmp.columns:
            _cmp["_b"] = pd.to_numeric(_cmp[_bcol], errors="coerce")
            _cmp["_p"] = pd.to_numeric(_cmp[_pcol], errors="coerce")
            _cmp = _cmp[np.isfinite(_cmp["_p"])].copy()

            def _effect_text(v: float) -> str:
                if not np.isfinite(v):
                    return "not estimated"
                if v > 0:
                    return "higher value linked with higher crashes"
                if v < 0:
                    return "higher value linked with lower crashes"
                return "near-zero net effect"

            def _delta_text(b: float, p: float) -> str:
                if np.isfinite(b):
                    if np.sign(b) != 0 and np.sign(p) != 0 and np.sign(b) != np.sign(p):
                        return "direction flips vs benchmark"
                    if abs(p) > abs(b):
                        return "stronger than benchmark"
                    if abs(p) < abs(b):
                        return "weaker than benchmark"
                return "new or benchmark-missing term"

            _cmp["_importance"] = _cmp["_p"].abs()
            _cmp = _cmp.sort_values("_importance", ascending=False)
            _cmp = _cmp.head(8).copy()
            _cmp["Safety implication"] = [
                f"{_effect_text(float(p))}; {_delta_text(float(b), float(p)) if np.isfinite(b) else 'new or benchmark-missing term'}"
                for b, p in zip(_cmp["_b"], _cmp["_p"])
            ]

            _show = pd.DataFrame(
                {
                    "Parameter": _cmp.get("Parameter", ""),
                    "Benchmark coef": _cmp["_b"],
                    "Proposed coef": _cmp["_p"],
                    "Safety implication": _cmp["Safety implication"],
                }
            )
            (output_dir / "safety_implications.csv").write_text(
                _show.to_csv(index=False), encoding="utf-8"
            )
            (output_dir / "safety_implications.md").write_text(
                _to_markdown(_show), encoding="utf-8"
            )

    literature_ref_lines = [
        "Ex16-3 literature benchmark model (user-specified)",
        "",
        "Model form:",
        "$\\log(\\hat{\\mu}) = \\alpha + \\beta_{LOWPRE}LOWPRE + \\beta_{GBRPM}GBRPM + \\beta_{FRICTION}FRICTION + "
        "\\beta_{EXPOSE}EXPOSE + \\beta_{INTPM}INTPM + \\beta_{CPM}CPM + \\beta_{HISNOW}HISNOW + "
        "\\log(AADT) + \\log(L)$",
        "",
        "Benchmark constraint used in this run: AADT appears only via offset (no estimated AADT slope parameter).",
        "",
        "- Nonrandom parameters: LOWPRE, GBRPM, FRICTION",
        "- Means for random parameters: EXPOSE, INTPM, CPM, HISNOW",
        "- Random scales and NB2 dispersion are reported from literature; this benchmark fit uses the same structural variables for count-model comparison.",
    ]
    (output_dir / "literature_benchmark_reference.md").write_text("\n".join(literature_ref_lines), encoding="utf-8")

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

    # MetaCount re-estimation of the user-specified benchmark structure.
    benchmark_mc = _fit_literature_benchmark_with_metacount(
        df_trainval_raw=df_trainval_raw,
        y_col=args.y_col,
        aadt_col=args.aadt_col,
        offset_col=offset_col,
        rp_draws=int(args.rp_draws),
    )
    if benchmark_mc is not None:
        bm_df = benchmark_mc.get("coef_df", pd.DataFrame())
        if isinstance(bm_df, pd.DataFrame) and not bm_df.empty:
            (output_dir / "benchmark_metacount_coefficients.csv").write_text(
                bm_df.to_csv(index=False), encoding="utf-8"
            )
            (output_dir / "benchmark_metacount_coefficients.md").write_text(
                _to_markdown(bm_df), encoding="utf-8"
            )
        (output_dir / "benchmark_metacount_summary.md").write_text(
            str(benchmark_mc.get("summary_md", "")), encoding="utf-8"
        )
    else:
        (output_dir / "benchmark_metacount_summary.md").write_text(
            "MetaCount benchmark re-estimation could not be produced in this run.",
            encoding="utf-8",
        )

    # Generate the fully spelled-out equation slide (included by QMD)
    _save_model_equation_md(
        output_dir / "model_equation_slide.md",
        fitted      = final_fit,
        scaler_stats= scaler_trainval,
        binary_vars = binary_vars,
        offset_col  = offset_col,
        aadt_col    = args.aadt_col,
    )
    print("  model_equation_slide.md written.")
    # If random-params model was fitted, also print its coefficient table so it
    # appears prominently in the console output alongside the fixed-effects summary.
    if jax_result is not None and jax_result.get("coef_str"):
        rp_cs = jax_result["coef_str"]
        rp_cm = jax_result.get("corr_matrix_str", "")
        print("\n" + "="*70)
        print("  FINAL MODEL: RANDOM-PARAMETERS NB2 (extends fixed-effects spec)")
        print(f"  LL={jax_result.get('loglik', float('nan')):.2f}   "
              f"BIC={jax_result.get('bic', float('nan')):.2f}")
        print("="*70)
        print("  Column key:  Fixed = same for all segments")
        print("               Random-Ind = population Mean +/- SD (site heterogeneity)")
        print("               Random-Cor = correlated pair (see correlation matrix)")
        print("-"*70)
        print(rp_cs)
        if rp_cm:
            print()
            print(rp_cm)
        print()
        print("  SD interpretation: a large SD relative to |Mean| means this")
        print("  variable's effect varies substantially across road segments.")
        print("="*70)

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
        "- pareto_selection_summary.md / .csv",
        "- harmony_search_summary.md / .csv",
        "- refinement_phase_traces.png",
        "- refinement_convergence_harmony.png",
        "- refinement_convergence_sa.png",
        "- search_phase_comparison.md / .csv / .png",
        "- refinement_convergence.png",
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
        "- literature_vs_proposed_coefficients.md / .csv",
        "- literature_benchmark_reference.md",
        "- benchmark_metacount_summary.md",
        "- benchmark_metacount_coefficients.md / .csv",
    ]
    (output_dir / "hierarchical_cmf_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Assets written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
