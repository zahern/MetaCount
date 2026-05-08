from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings


warnings.simplefilter("ignore", ConvergenceWarning)


@dataclass
class FittedModel:
    family: str
    upper_vars: list[str]
    lower_vars: list[str]
    result: Any
    aadt_col: str
    offset_col: str | None


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

    if "segment_length" in out.columns and aadt_col in out.columns:
        exposure = pd.to_numeric(out["segment_length"], errors="coerce") * pd.to_numeric(out[aadt_col], errors="coerce")
        exposure = np.clip(exposure.to_numpy(dtype=float), 1e-9, None)
        out["OFFSET"] = np.log(exposure)

    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out[aadt_col] = pd.to_numeric(out[aadt_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[y_col, aadt_col])
    out = out[out[aadt_col] > 0].copy()

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

    try:
        if family == "nb":
            model = NegativeBinomial(y_train, X_train, offset=offset)
            result = model.fit(disp=False, maxiter=200)
        else:
            model = Poisson(y_train, X_train, offset=offset)
            result = model.fit(disp=False, maxiter=200)
    except Exception:
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


def _random_search(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    aadt_col: str,
    y_col: str,
    upper_candidates: list[str],
    lower_candidates: list[str],
    family: str,
    offset_col: str | None,
    search_iter: int,
    max_upper_terms: int,
    max_lower_terms: int,
    seed: int,
) -> tuple[FittedModel, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tested: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    history_rows: list[dict[str, Any]] = []

    best_fit: FittedModel | None = None
    best_score = float("inf")

    # Always include at least one simple baseline model.
    baseline_upper = upper_candidates[:1] if upper_candidates else []
    baseline_lower = lower_candidates[:1] if lower_candidates else []

    seeds = [
        (tuple(baseline_upper), tuple(baseline_lower)),
        (tuple(), tuple()),
    ]

    for init_upper, init_lower in seeds:
        key = (tuple(sorted(init_upper)), tuple(sorted(init_lower)))
        tested.add(key)
        fit = _fit_model(
            df_train,
            aadt_col=aadt_col,
            y_col=y_col,
            upper_vars=list(init_upper),
            lower_vars=list(init_lower),
            family=family,
            offset_col=offset_col,
        )
        if fit is None:
            continue
        pred_val = _predict(df_val, fit)
        y_val = pd.to_numeric(df_val[y_col], errors="coerce").to_numpy(dtype=float)
        score = _poisson_deviance(y_val, pred_val)
        history_rows.append(
            {
                "Iteration": len(history_rows) + 1,
                "Upper Vars": ", ".join(fit.upper_vars) if fit.upper_vars else "(none)",
                "Lower Vars": ", ".join(fit.lower_vars) if fit.lower_vars else "(none)",
                "Val Poisson Dev": score,
                "Val RMSE": _metrics(y_val, pred_val)["rmse"],
                "AIC": float(getattr(fit.result, "aic", np.nan)),
                "BIC": float(getattr(fit.result, "bic", np.nan)),
            }
        )
        if score < best_score:
            best_score = score
            best_fit = fit

    for _ in range(search_iter):
        k_upper = int(rng.integers(0, max_upper_terms + 1))
        k_lower = int(rng.integers(0, max_lower_terms + 1))

        pick_upper = []
        pick_lower = []
        if k_upper > 0 and upper_candidates:
            size = min(k_upper, len(upper_candidates))
            pick_upper = sorted(rng.choice(upper_candidates, size=size, replace=False).tolist())
        if k_lower > 0 and lower_candidates:
            size = min(k_lower, len(lower_candidates))
            pick_lower = sorted(rng.choice(lower_candidates, size=size, replace=False).tolist())

        key = (tuple(pick_upper), tuple(pick_lower))
        if key in tested:
            continue
        tested.add(key)

        fit = _fit_model(
            df_train,
            aadt_col=aadt_col,
            y_col=y_col,
            upper_vars=pick_upper,
            lower_vars=pick_lower,
            family=family,
            offset_col=offset_col,
        )
        if fit is None:
            continue

        pred_val = _predict(df_val, fit)
        y_val = pd.to_numeric(df_val[y_col], errors="coerce").to_numpy(dtype=float)
        score = _poisson_deviance(y_val, pred_val)
        val_stats = _metrics(y_val, pred_val)

        history_rows.append(
            {
                "Iteration": len(history_rows) + 1,
                "Upper Vars": ", ".join(fit.upper_vars) if fit.upper_vars else "(none)",
                "Lower Vars": ", ".join(fit.lower_vars) if fit.lower_vars else "(none)",
                "Val Poisson Dev": score,
                "Val RMSE": val_stats["rmse"],
                "AIC": float(getattr(fit.result, "aic", np.nan)),
                "BIC": float(getattr(fit.result, "bic", np.nan)),
            }
        )

        if score < best_score:
            best_score = score
            best_fit = fit

    if best_fit is None:
        raise RuntimeError("Search failed to fit any candidate model.")

    history_df = pd.DataFrame(history_rows).sort_values("Val Poisson Dev").reset_index(drop=True)
    return best_fit, history_df


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


def _representative_profile(df: pd.DataFrame, variables: list[str], binary_vars: set[str]) -> dict[str, float]:
    profile: dict[str, float] = {}
    for var in variables:
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

    profile = _representative_profile(df_reference_raw, selected_raw_vars, binary_vars)
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


def _save_dashboard_html(path: Path, payload: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Washington Hierarchical CMF Dashboard</title>
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <style>
        :root {{
            --ink: #152238;
            --teal: #0a6c74;
            --orange: #d96f32;
            --paper: #f8f6f2;
            --panel: #ffffff;
            --line: #d4d8de;
            --muted: #5d6b79;
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
        .wrap {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .hero {{
            background: linear-gradient(120deg, #0a6c74 0%, #18435a 45%, #8a4e2c 100%);
            color: #fff;
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.14);
        }}
        .hero h1 {{
            margin: 0;
            font-size: 1.45rem;
            letter-spacing: 0.2px;
        }}
        .hero p {{
            margin: 6px 0 0;
            color: rgba(255, 255, 255, 0.94);
            font-size: 0.95rem;
        }}
        .grid {{
            margin-top: 16px;
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 14px;
        }}
        .card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px;
            box-shadow: 0 8px 18px rgba(21, 34, 56, 0.08);
        }}
        .control-row {{
            display: grid;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .label {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: var(--muted);
            font-weight: 600;
        }}
        select, input[type=\"range\"] {{
            width: 100%;
        }}
        #chart {{
            height: 62vh;
        }}
        .kpi {{
            margin-top: 8px;
            font-size: 0.95rem;
            color: var(--ink);
            line-height: 1.5;
        }}
        .warn {{
            margin-top: 8px;
            color: #8b3e12;
            font-size: 0.88rem;
            background: #fff1e9;
            border: 1px solid #f1c9b4;
            border-radius: 8px;
            padding: 8px;
        }}
        @media (max-width: 920px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            #chart {{
                height: 54vh;
            }}
        }}
    </style>
</head>
<body>
<div class=\"wrap\">
    <div class=\"hero\">
        <h1>Washington Hierarchical CMF Explorer</h1>
        <p>Adjust one variable at a time to inspect predicted crashes and CMF response. Binary variables are evaluated as a 0 to 1 toggle.</p>
    </div>

    <div class=\"grid\">
        <div class=\"card\">
            <div class=\"control-row\">
                <span class=\"label\">Variable</span>
                <select id=\"varSelect\"></select>
            </div>
            <div class=\"control-row\">
                <span class=\"label\">AADT Scenario</span>
                <select id=\"aadtSelect\"></select>
            </div>
            <div class=\"control-row\">
                <span class=\"label\">Display</span>
                <select id=\"modeSelect\">
                    <option value=\"pred\">Predicted crashes</option>
                    <option value=\"cmf\">CMF (relative to profile baseline)</option>
                </select>
            </div>
            <div class=\"control-row\">
                <span class=\"label\" id=\"sliderLabel\">Value</span>
                <input id=\"valueSlider\" type=\"range\" min=\"0\" max=\"40\" value=\"20\" step=\"1\" />
            </div>
            <div class=\"kpi\" id=\"kpi\"></div>
            <div class=\"warn\" id=\"binaryNote\" style=\"display:none;\">Binary feature: interpret as a class toggle, not a one-unit continuous increment.</div>
        </div>

        <div class=\"card\">
            <div id=\"chart\"></div>
        </div>
    </div>
</div>

<script>
const payload = {json.dumps(payload)};
const variables = Object.keys(payload.variables);
const aadtLevels = Object.keys(payload.aadt_levels);

const varSelect = document.getElementById('varSelect');
const aadtSelect = document.getElementById('aadtSelect');
const modeSelect = document.getElementById('modeSelect');
const valueSlider = document.getElementById('valueSlider');
const sliderLabel = document.getElementById('sliderLabel');
const kpi = document.getElementById('kpi');
const binaryNote = document.getElementById('binaryNote');

for (const v of variables) {{
    const o = document.createElement('option');
    o.value = v;
    o.textContent = v;
    varSelect.appendChild(o);
}}
for (const level of aadtLevels) {{
    const o = document.createElement('option');
    o.value = level;
    o.textContent = `${{level}} (${{payload.aadt_levels[level].toFixed(0)}})`;
    aadtSelect.appendChild(o);
}}

function currentSeries() {{
    const variable = varSelect.value;
    const level = aadtSelect.value;
    const mode = modeSelect.value;
    return payload.variables[variable].levels[level][mode];
}}

function currentX() {{
    const variable = varSelect.value;
    const level = aadtSelect.value;
    return payload.variables[variable].levels[level].x;
}}

function refreshSlider() {{
    const variable = varSelect.value;
    const level = aadtSelect.value;
    const data = payload.variables[variable].levels[level];
    const maxIndex = Math.max(data.x.length - 1, 0);
    valueSlider.min = 0;
    valueSlider.max = maxIndex;
    valueSlider.step = 1;
    valueSlider.value = Math.min(Number(valueSlider.value), maxIndex);
    sliderLabel.textContent = `Value Index (0-${{maxIndex}})`;
    binaryNote.style.display = payload.variables[variable].is_binary ? 'block' : 'none';
}}

function draw() {{
    const variable = varSelect.value;
    const level = aadtSelect.value;
    const mode = modeSelect.value;

    const x = currentX();
    const y = currentSeries();
    const idx = Math.min(Number(valueSlider.value), y.length - 1);

    const mainTrace = {{
        x,
        y,
        type: 'scatter',
        mode: 'lines+markers',
        marker: {{size: 5, color: '#0a6c74'}},
        line: {{width: 3, color: '#0a6c74'}},
        name: mode === 'pred' ? 'Predicted crashes' : 'CMF'
    }};

    const markerTrace = {{
        x: [x[idx]],
        y: [y[idx]],
        type: 'scatter',
        mode: 'markers',
        marker: {{size: 11, color: '#d96f32'}},
        name: 'Selected point'
    }};

    const yTitle = mode === 'pred' ? 'Predicted crashes' : 'CMF';
    const layout = {{
        template: 'plotly_white',
        title: `${{variable}} response at ${{level}}`,
        xaxis: {{title: variable}},
        yaxis: {{title: yTitle}},
        margin: {{l: 64, r: 20, t: 56, b: 52}},
        legend: {{orientation: 'h'}}
    }};

    Plotly.react('chart', [mainTrace, markerTrace], layout, {{responsive: true}});

    const aadtValue = payload.aadt_levels[level];
    const xVal = x[idx];
    const yVal = y[idx];
    kpi.innerHTML = [
        `<strong>${{variable}}</strong> = <strong>${{xVal.toFixed(4)}}</strong>`,
        `AADT level = <strong>${{aadtValue.toFixed(0)}}</strong>`,
        mode === 'pred'
            ? `Predicted crashes = <strong>${{yVal.toFixed(4)}}</strong>`
            : `CMF = <strong>${{yVal.toFixed(4)}}</strong>`
    ].join('<br/>');
}}

varSelect.value = variables[0];
aadtSelect.value = aadtLevels[1] || aadtLevels[0];
refreshSlider();
draw();

varSelect.addEventListener('change', () => {{ refreshSlider(); draw(); }});
aadtSelect.addEventListener('change', () => {{ refreshSlider(); draw(); }});
modeSelect.addEventListener('change', draw);
valueSlider.addEventListener('input', draw);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _resolve_default_candidates(df: pd.DataFrame, profile: str = "core") -> tuple[list[str], list[str]]:
    # Tuned profile keeps engineering signal strong and avoids very collinear weather variants by default.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up and run a Washington hierarchical CMF experiment with validation outputs and dashboard assets."
    )
    parser.add_argument("--input", default="data/rural_int.csv", help="Path to Washington crash dataset CSV.")
    parser.add_argument("--output-dir", default="results/washington_cmf", help="Output directory.")
    parser.add_argument("--y-col", default="crashes", help="Crash frequency response column.")
    parser.add_argument("--aadt-col", default="monthly_AADT", help="AADT traffic-response column.")
    parser.add_argument("--offset-col", default="OFFSET", help="Offset column name to use if available.")
    parser.add_argument("--disable-offset", action="store_true", help="Disable offset during fitting.")
    parser.add_argument("--family", choices=["nb", "poisson"], default="nb", help="Count family for fitting.")

    parser.add_argument("--train-frac", type=float, default=0.60, help="Training split fraction.")
    parser.add_argument("--val-frac", type=float, default=0.20, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")

    parser.add_argument("--search-iter", type=int, default=120, help="Random search iterations.")
    parser.add_argument("--max-upper-terms", type=int, default=6, help="Max upper-level terms per candidate.")
    parser.add_argument("--max-lower-terms", type=int, default=5, help="Max lower-level terms per candidate.")
    parser.add_argument(
        "--candidate-profile",
        choices=["core", "expanded"],
        default="core",
        help="Default candidate pool profile when --upper-vars/--lower-vars are not provided.",
    )
    parser.add_argument("--upper-vars", default=None, help="Comma-separated upper-level candidate variables.")
    parser.add_argument("--lower-vars", default=None, help="Comma-separated lower-level candidate variables.")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    best_fit_train, search_history = _random_search(
        df_train=df_train,
        df_val=df_val,
        aadt_col=args.aadt_col,
        y_col=args.y_col,
        upper_candidates=upper_model_vars,
        lower_candidates=lower_model_vars,
        family=args.family,
        offset_col=offset_col,
        search_iter=int(args.search_iter),
        max_upper_terms=int(args.max_upper_terms),
        max_lower_terms=int(args.max_lower_terms),
        seed=int(args.seed),
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
            {"Setting": "Upper candidate count", "Value": int(len(upper_raw))},
            {"Setting": "Lower candidate count", "Value": int(len(lower_raw))},
            {"Setting": "Selected upper vars", "Value": ", ".join(selected_upper_raw) if selected_upper_raw else "(none)"},
            {"Setting": "Selected lower vars", "Value": ", ".join(selected_lower_raw) if selected_lower_raw else "(none)"},
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

    if "curve" in payload["variables"]:
        _save_curve_plot(
            output_dir / "curve_crash_risk_sensitivity.png",
            payload=payload,
            variable="curve",
            aadt_levels=["Low AADT", "Median AADT", "High AADT"],
            title="Crash-risk change as curvature changes",
            ylabel="Predicted crashes",
            mode="pred",
        )
        _save_curve_plot(
            output_dir / "curve_cmf_sensitivity.png",
            payload=payload,
            variable="curve",
            aadt_levels=["Low AADT", "Median AADT", "High AADT"],
            title="CMF change as curvature changes",
            ylabel="CMF (relative to profile baseline)",
            mode="cmf",
        )

    selected_binary = [v for v in selected_raw if v in binary_vars]
    if selected_binary:
        _save_binary_bar(output_dir / "binary_toggle_cmf_effects.png", payload, selected_binary)

    _save_dashboard_html(output_dir / "washington_hierarchical_cmf_dashboard.html", payload)

    # Write tabular outputs.
    (output_dir / "search_history_top25.md").write_text(_to_markdown(search_top_df), encoding="utf-8")
    (output_dir / "search_history_top25.csv").write_text(search_top_df.to_csv(index=False), encoding="utf-8")

    (output_dir / "model_settings.md").write_text(_to_markdown(settings_df), encoding="utf-8")
    (output_dir / "validation_metrics.md").write_text(_to_markdown(metrics_df), encoding="utf-8")
    (output_dir / "validation_metrics.csv").write_text(metrics_df.to_csv(index=False), encoding="utf-8")

    (output_dir / "test_calibration_deciles.md").write_text(_to_markdown(calibration_test_df), encoding="utf-8")
    (output_dir / "test_calibration_deciles.csv").write_text(calibration_test_df.to_csv(index=False), encoding="utf-8")

    (output_dir / "coefficients_standardized_and_original.md").write_text(_to_markdown(coef_df), encoding="utf-8")
    (output_dir / "coefficients_standardized_and_original.csv").write_text(coef_df.to_csv(index=False), encoding="utf-8")

    scaler_rows = [
        {"Variable": var, "Mean (train/final)": mu, "Std (train/final)": sd}
        for var, (mu, sd) in sorted(scaler_trainval.items())
    ]
    scaler_df = pd.DataFrame(scaler_rows)
    (output_dir / "standardization_reference.csv").write_text(scaler_df.to_csv(index=False), encoding="utf-8")

    summary_lines = [
        "Washington Hierarchical CMF Experiment",
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
        "- search_history_top25.md / .csv",
        "- model_settings.md",
        "- validation_metrics.md / .csv",
        "- test_calibration_deciles.md / .csv",
        "- coefficients_standardized_and_original.md / .csv",
        "- standardization_reference.csv",
        "- validation_observed_vs_predicted.png",
        "- test_observed_vs_predicted.png",
        "- curve_crash_risk_sensitivity.png (if curve selected)",
        "- curve_cmf_sensitivity.png (if curve selected)",
        "- binary_toggle_cmf_effects.png (if binary vars selected)",
        "- washington_hierarchical_cmf_dashboard.html",
    ]
    (output_dir / "washington_hierarchical_cmf_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Assets written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
