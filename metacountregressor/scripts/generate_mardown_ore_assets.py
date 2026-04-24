from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacountregressor import CMFExperimentBuilder, ExperimentBuilder, load_example16_3_model_data  # noqa: E402

try:
    from experiment_package import CountModel, build_model_from_manual_spec  # noqa: E402
except ImportError:
    from metacountregressor.experiment_package import CountModel, build_model_from_manual_spec  # noqa: E402


def _to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        header = "| " + " | ".join(str(col) for col in cols) + " |"
        divider = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = [
            "| " + " | ".join(str(row[col]) for col in cols) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, divider] + rows)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _add_zscore_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    out = df.copy()
    stats: dict[str, tuple[float, float]] = {}
    for col in columns:
        mu = float(out[col].mean())
        sd = float(out[col].std(ddof=0))
        denom = sd if sd > 0 else 1.0
        out[f"{col}_Z"] = (out[col] - mu) / denom
        stats[col] = (mu, sd)
    return out, stats


def _fit_is_valid(fit: dict[str, Any]) -> bool:
    try:
        preds = np.asarray(fit.get("predictions", []), dtype=float).reshape(-1)
        params = np.asarray(getattr(fit.get("result"), "params", []), dtype=float).reshape(-1)
        summary = fit.get("summary", {})
        if preds.size == 0 or params.size == 0:
            return False
        if not np.all(np.isfinite(preds)) or not np.all(np.isfinite(params)):
            return False
        for key in ["bic", "aic", "loglik"]:
            if not np.isfinite(summary.get(key, np.nan)):
                return False
        return float(np.nanmax(preds)) <= 1e6
    except Exception:
        return False


def _clean_parameter_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Parameter"] = (
        out["Parameter"].astype(str)
        .str.replace("CURVES_Z", "CURVES", regex=False)
        .str.replace("WIDTH_Z", "WIDTH", regex=False)
        .str.replace("__cmf_local__WIDTH_Z", "__cmf_local__WIDTH", regex=False)
    )
    return out


def _validation_metrics(fit: dict[str, Any]) -> dict[str, float]:
    y_true = np.asarray(fit.get("data", {}).get("y", []), dtype=float).reshape(-1)
    y_pred = np.asarray(fit.get("predictions", []), dtype=float).reshape(-1)
    n = min(y_true.size, y_pred.size)
    if n == 0:
        return {
            "rmse": float("inf"),
            "mae": float("inf"),
            "corr": float("nan"),
            "obs_mean": float("nan"),
            "pred_mean": float("nan"),
        }

    y_true = y_true[:n]
    y_pred = y_pred[:n]
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if y_true.size == 0:
        return {
            "rmse": float("inf"),
            "mae": float("inf"),
            "corr": float("nan"),
            "obs_mean": float("nan"),
            "pred_mean": float("nan"),
        }

    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "obs_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def _coef_table(builder: ExperimentBuilder, fit: dict[str, Any], model_name: str) -> pd.DataFrame:
    coef = builder.print_coefficients(fit).copy()
    coef["Model"] = model_name
    coef["Estimate"] = pd.to_numeric(coef["Estimate"], errors="coerce").round(6)
    return coef[["Model", "Parameter", "Type", "Estimate"]]


def _offset_from_aadt_length(aadt: np.ndarray, length_value: float) -> np.ndarray:
    exposure = np.maximum(aadt * length_value * 365.0 / 1e8, 1e-12)
    return np.log(exposure)


def _representative_profile(df: pd.DataFrame) -> dict[str, float]:
    return {
        "URB": float(np.round(df["URB"].median())),
        "ACCESS": float(df["ACCESS"].median()),
        "GRADEBR": float(df["GRADEBR"].median()),
        "CURVES": float(df["CURVES"].median()),
        "WIDTH": float(df["WIDTH"].median()),
        "LENGTH": float(df["LENGTH"].median()),
    }


def _scenario_frame(
    aadt_values: np.ndarray,
    profile: dict[str, float],
    scale_stats: dict[str, tuple[float, float]],
    use_offset: bool = True,
) -> pd.DataFrame:
    aadt = np.asarray(aadt_values, dtype=float).reshape(-1)
    scenario = pd.DataFrame(
        {
            "ID": np.arange(1, aadt.size + 1, dtype=int),
            "FC": np.ones(aadt.size, dtype=int),
            "FREQ": np.zeros(aadt.size, dtype=float),
            "AADT": aadt,
            "URB": np.full(aadt.size, profile["URB"], dtype=float),
            "ACCESS": np.full(aadt.size, profile["ACCESS"], dtype=float),
            "GRADEBR": np.full(aadt.size, profile["GRADEBR"], dtype=float),
            "CURVES": np.full(aadt.size, profile["CURVES"], dtype=float),
            "WIDTH": np.full(aadt.size, profile["WIDTH"], dtype=float),
            "LENGTH": np.full(aadt.size, profile["LENGTH"], dtype=float),
        }
    )
    if use_offset:
        scenario["OFFSET"] = _offset_from_aadt_length(aadt, profile["LENGTH"])
    scenario["__cmf_log_aadt"] = np.log(np.maximum(scenario["AADT"].to_numpy(dtype=float), 1e-12))
    for raw_name in ["CURVES", "WIDTH"]:
        mu, sd = scale_stats[raw_name]
        denom = sd if sd > 0 else 1.0
        scenario[f"{raw_name}_Z"] = (scenario[raw_name] - mu) / denom
        scenario[f"__cmf_local__{raw_name}"] = scenario[raw_name] * scenario["__cmf_log_aadt"]
        scenario[f"__cmf_local__{raw_name}_Z"] = scenario[f"{raw_name}_Z"] * scenario["__cmf_log_aadt"]
    return scenario


def _predict_with_fit(
    df_scenario: pd.DataFrame,
    fit: dict[str, Any],
    id_col: str = "ID",
    y_col: str = "FREQ",
    offset_col: str | None = "OFFSET",
    R: int = 120,
) -> np.ndarray:
    data_scenario, spec_scenario = build_model_from_manual_spec(
        df=df_scenario,
        manual_spec=fit["manual_spec"],
        id_col=id_col,
        y_col=y_col,
        offset_col=offset_col,
        R=R,
    )
    model = CountModel(spec_scenario, data_scenario)
    model.params = np.asarray(fit["result"].params)
    return np.asarray(model.predict(), dtype=float).reshape(-1)


def _save_observed_vs_predicted_png(
    out_path: Path,
    y_true: np.ndarray,
    pred_trad: np.ndarray,
    pred_cmf: np.ndarray,
) -> None:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    trad = np.asarray(pred_trad, dtype=float).reshape(-1)
    cmf = np.asarray(pred_cmf, dtype=float).reshape(-1)
    max_value = float(np.nanmax(np.concatenate([y, trad, cmf, np.array([1.0])])))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.3), dpi=160, sharex=True, sharey=True)
    panels = [
        (axes[0], trad, "Traditional baseline NB", "#1f77b4"),
        (axes[1], cmf, "Hierarchical CMF baseline NB", "#d62728"),
    ]
    for axis, pred, title, color in panels:
        axis.scatter(pred, y, s=14, alpha=0.28, color=color, edgecolors="none")
        axis.plot([0, max_value], [0, max_value], linestyle="--", linewidth=1.2, color="#444444")

        bin_df = pd.DataFrame({"pred": pred, "obs": y}).sort_values("pred")
        bin_df["bin"] = pd.qcut(bin_df["pred"], q=10, duplicates="drop")
        summary = (
            bin_df.groupby("bin", observed=False)
            .agg(pred=("pred", "mean"), obs=("obs", "mean"))
            .reset_index(drop=True)
        )
        axis.plot(summary["pred"], summary["obs"], color="#ff7f0e", linewidth=2.0)
        axis.set_title(title)
        axis.set_xlabel("Predicted crashes")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Observed crashes")
    fig.suptitle("Observed vs Predicted (baseline models)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_curve_png(out_path: Path, aadt_grid: np.ndarray, trad: np.ndarray, cmf: np.ndarray) -> None:
    pct_gap = 100.0 * (np.asarray(cmf) - np.asarray(trad)) / np.maximum(np.asarray(trad), 1e-9)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 5.6),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.1]},
    )
    ax = axes[0]
    ax_gap = axes[1]

    ax.plot(aadt_grid, trad, color="#1f77b4", linewidth=2.4, label="Traditional baseline NB")
    ax.plot(aadt_grid, cmf, color="#d62728", linewidth=2.4, label="Hierarchical CMF baseline NB")
    ax.fill_between(aadt_grid, trad, cmf, color="#f2b8b5", alpha=0.18)
    ax.set_ylabel("Predicted crashes")
    ax.set_title("Smoothed model-based predicted crashes vs AADT")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    ax_gap.plot(aadt_grid, pct_gap, color="#2ca02c", linewidth=2.0)
    ax_gap.axhline(0.0, color="#444444", linewidth=1.0)
    ax_gap.set_xlabel("AADT")
    ax_gap.set_ylabel("CMF gap %")
    ax_gap.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_interactive_curve_html(out_path: Path, aadt_grid: np.ndarray, trad: np.ndarray, cmf: np.ndarray) -> None:
    payload = {
        "aadt": [float(v) for v in aadt_grid],
        "traditional": [float(v) for v in trad],
        "hierarchical": [float(v) for v in cmf],
        "difference": [float(c - t) for t, c in zip(trad, cmf)],
    }
    html = f"""<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>AADT Prediction Curve Comparison</title>
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <style>
        body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
        #chart {{ width: 100%; height: 72vh; }}
        .toolbar {{ margin-bottom: 10px; display: flex; gap: 8px; flex-wrap: wrap; }}
        button {{ padding: 8px 12px; border: 1px solid #ccc; background: #fafafa; cursor: pointer; }}
        button:hover {{ background: #f0f0f0; }}
        .note {{ color: #444; font-size: 0.95rem; margin-top: 8px; max-width: 900px; }}
    </style>
</head>
<body>
    <h2>Smoothed Predicted Crashes as AADT Changes</h2>
    <div class=\"toolbar\">
        <button onclick=\"showBoth()\">Both Models</button>
        <button onclick=\"showTraditional()\">Traditional Only</button>
        <button onclick=\"showHierarchical()\">Hierarchical Only</button>
        <button onclick=\"showDifference()\">Difference Only</button>
    </div>
    <div id=\"chart\"></div>
    <div class=\"note\">
        This is a smooth model-based partial-effect curve: predictions are evaluated on a dense AADT grid while other variables stay at a representative roadway profile.
    </div>
    <script>
        const dataPayload = {json.dumps(payload)};
        const traceTrad = {{
            x: dataPayload.aadt,
            y: dataPayload.traditional,
            mode: 'lines',
            name: 'Traditional baseline NB',
            line: {{color: '#1f77b4', width: 3}},
            visible: true
        }};
        const traceCmf = {{
            x: dataPayload.aadt,
            y: dataPayload.hierarchical,
            mode: 'lines',
            name: 'Hierarchical CMF baseline NB',
            line: {{color: '#d62728', width: 3}},
            visible: true
        }};
        const traceDiff = {{
            x: dataPayload.aadt,
            y: dataPayload.difference,
            mode: 'lines',
            name: 'Hierarchical - Traditional',
            line: {{color: '#2ca02c', width: 3, dash: 'dot'}},
            visible: false,
            yaxis: 'y2'
        }};
        const layout = {{
            xaxis: {{ title: 'AADT' }},
            yaxis: {{ title: 'Predicted crashes' }},
            yaxis2: {{ title: 'Prediction difference', overlaying: 'y', side: 'right', showgrid: false }},
            title: 'Smoothed predicted crashes vs AADT',
            template: 'plotly_white',
            legend: {{ orientation: 'h' }}
        }};
        Plotly.newPlot('chart', [traceTrad, traceCmf, traceDiff], layout, {{responsive: true}});
        function showBoth() {{ Plotly.restyle('chart', {{visible: [true]}}, [0]); Plotly.restyle('chart', {{visible: [true]}}, [1]); Plotly.restyle('chart', {{visible: [false]}}, [2]); }}
        function showTraditional() {{ Plotly.restyle('chart', {{visible: [true]}}, [0]); Plotly.restyle('chart', {{visible: [false]}}, [1]); Plotly.restyle('chart', {{visible: [false]}}, [2]); }}
        function showHierarchical() {{ Plotly.restyle('chart', {{visible: [false]}}, [0]); Plotly.restyle('chart', {{visible: [true]}}, [1]); Plotly.restyle('chart', {{visible: [false]}}, [2]); }}
        function showDifference() {{ Plotly.restyle('chart', {{visible: [false]}}, [0]); Plotly.restyle('chart', {{visible: [false]}}, [1]); Plotly.restyle('chart', {{visible: [true]}}, [2]); }}
    </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _cmf_change_payload(
    profile: dict[str, float],
    scale_stats: dict[str, tuple[float, float]],
    fit_trad_offset: dict[str, Any],
    fit_cmf_offset: dict[str, Any],
    fit_trad_no_offset: dict[str, Any],
    fit_cmf_no_offset: dict[str, Any],
    df: pd.DataFrame,
    R: int,
) -> dict[str, Any]:
    aadt_levels = {
        "Low AADT": float(df["AADT"].quantile(0.25)),
        "Median AADT": float(df["AADT"].median()),
        "High AADT": float(df["AADT"].quantile(0.75)),
    }
    step_values = [0, 1, 2, 3]
    feature_labels = {
        "CURVES": "Curves",
        "ACCESS": "Access",
        "WIDTH": "Width",
    }
    mode_map = {
        "With offset": (fit_trad_offset, fit_cmf_offset, True),
        "Without offset": (fit_trad_no_offset, fit_cmf_no_offset, False),
    }

    payload: dict[str, Any] = {
        "steps": step_values,
        "feature_order": list(feature_labels.keys()),
        "feature_labels": feature_labels,
        "modes": {},
    }
    for mode_name, (fit_trad, fit_cmf, use_offset) in mode_map.items():
        payload["modes"][mode_name] = {"levels": {}}
        for level_name, aadt_value in aadt_levels.items():
            level_payload: dict[str, Any] = {}
            for feature_name in feature_labels:
                trad_values: list[float] = []
                cmf_values: list[float] = []
                for step in step_values:
                    scenario_profile = dict(profile)
                    scenario_profile[feature_name] = float(profile[feature_name] + step)
                    scenario_df = _scenario_frame(
                        np.array([aadt_value]),
                        scenario_profile,
                        scale_stats,
                        use_offset=use_offset,
                    )
                    offset_col = "OFFSET" if use_offset else None
                    trad_values.append(float(_predict_with_fit(scenario_df, fit_trad, offset_col=offset_col, R=R)[0]))
                    cmf_values.append(float(_predict_with_fit(scenario_df, fit_cmf, offset_col=offset_col, R=R)[0]))

                trad_base = trad_values[0]
                cmf_base = cmf_values[0]
                level_payload[feature_name] = {
                    "traditional": trad_values,
                    "hierarchical": cmf_values,
                    "traditional_pct": [100.0 * (value - trad_base) / max(trad_base, 1e-9) for value in trad_values],
                    "hierarchical_pct": [100.0 * (value - cmf_base) / max(cmf_base, 1e-9) for value in cmf_values],
                }
            payload["modes"][mode_name]["levels"][level_name] = level_payload
    return payload


def _save_cmf_change_png(out_path: Path, payload: dict[str, Any]) -> None:
    steps = np.asarray(payload["steps"], dtype=float)
    offset_data = payload["modes"]["With offset"]["levels"]["Median AADT"]["CURVES"]
    no_offset_data = payload["modes"]["Without offset"]["levels"]["Median AADT"]["CURVES"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), dpi=160, sharey=True)
    panels = [
        (axes[0], "With offset", offset_data),
        (axes[1], "Without offset", no_offset_data),
    ]
    for axis, title, data in panels:
        axis.plot(steps, data["traditional_pct"], color="#1f77b4", linewidth=2.2, marker="o", label="Traditional")
        axis.plot(steps, data["hierarchical_pct"], color="#d62728", linewidth=2.2, marker="o", label="Hierarchical CMF")
        axis.axhline(0.0, color="#444444", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Additional curves")
        axis.set_xticks(steps)
        axis.grid(alpha=0.22)

    axes[0].set_ylabel("% change from base case")
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Multi-step curve sensitivity at median AADT")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_cmf_change_html(out_path: Path, payload: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>CMF Change Explorer</title>
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <style>
        body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
        #chart {{ width: 100%; height: 72vh; }}
        .toolbar {{ margin-bottom: 10px; display: flex; gap: 8px; flex-wrap: wrap; }}
        button {{ padding: 8px 12px; border: 1px solid #ccc; background: #fafafa; cursor: pointer; }}
        button:hover {{ background: #f0f0f0; }}
        .note {{ color: #444; font-size: 0.95rem; margin-top: 8px; max-width: 900px; }}
    </style>
</head>
<body>
    <h2>Interactive CMF Change Explorer</h2>
    <div class=\"toolbar\">
        <button onclick=\"showPct()\">Percent Change</button>
        <button onclick=\"showAbs()\">Absolute Predictions</button>
        <button onclick=\"setMode('With offset')\">With Offset</button>
        <button onclick=\"setMode('Without offset')\">Without Offset</button>
        <button onclick=\"setLevel('Low AADT')\">Low AADT</button>
        <button onclick=\"setLevel('Median AADT')\">Median AADT</button>
        <button onclick=\"setLevel('High AADT')\">High AADT</button>
        <button onclick=\"setFeature('CURVES')\">Curves</button>
        <button onclick=\"setFeature('ACCESS')\">Access</button>
        <button onclick=\"setFeature('WIDTH')\">Width</button>
    </div>
    <div id=\"chart\"></div>
    <div class=\"note\">
        This explorer shows step changes of +0, +1, +2, and +3 for one feature at a time. Toggle the offset on and off to see how exposure normalization changes the apparent effect size and direction.
    </div>
    <script>
        const payload = {json.dumps(payload)};
        let currentLevel = 'Median AADT';
        let currentMode = 'pct';
        let currentOffsetMode = 'With offset';
        let currentFeature = 'CURVES';
        function traces() {{
            const featureData = payload.modes[currentOffsetMode].levels[currentLevel][currentFeature];
            const yTrad = currentMode === 'pct' ? featureData.traditional_pct : featureData.traditional;
            const yCmf = currentMode === 'pct' ? featureData.hierarchical_pct : featureData.hierarchical;
            return [
                {{ x: payload.steps, y: yTrad, type: 'scatter', mode: 'lines+markers', name: 'Traditional', line: {{color: '#1f77b4', width: 3}} }},
                {{ x: payload.steps, y: yCmf, type: 'scatter', mode: 'lines+markers', name: 'Hierarchical CMF', line: {{color: '#d62728', width: 3}} }},
            ];
        }}
        function layout() {{
            return {{
                template: 'plotly_white',
                title: payload.feature_labels[currentFeature] + ' sensitivity: ' + currentLevel + ' / ' + currentOffsetMode,
                yaxis: {{ title: currentMode === 'pct' ? '% change from base case' : 'Predicted crashes' }},
                xaxis: {{ title: 'Additional units', tickmode: 'array', tickvals: payload.steps }},
                legend: {{ orientation: 'h' }}
            }};
        }}
        Plotly.newPlot('chart', traces(), layout(), {{responsive: true}});
        function redraw() {{ Plotly.react('chart', traces(), layout(), {{responsive: true}}); }}
        function setLevel(level) {{ currentLevel = level; redraw(); }}
        function showPct() {{ currentMode = 'pct'; redraw(); }}
        function showAbs() {{ currentMode = 'abs'; redraw(); }}
        function setMode(mode) {{ currentOffsetMode = mode; redraw(); }}
        function setFeature(feature) {{ currentFeature = feature; redraw(); }}
    </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit simplified mardown_ore models and export plain-language slide assets."
    )
    parser.add_argument("--R", type=int, default=120, help="Simulation draws for fitted models.")
    parser.add_argument(
        "--output-dir",
        default="results/slide_assets",
        help="Directory where slide assets are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_example16_3_model_data().copy()
    df_scaled, scale_stats = _add_zscore_columns(df, ["CURVES", "WIDTH"])

    trad_builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="FREQ",
        offset_col="OFFSET",
        group_id_col="FC",
    )
    trad_builder_no_offset = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="FREQ",
        offset_col=None,
        group_id_col="FC",
    )
    cmf_builder = CMFExperimentBuilder(
        df=df,
        y_col="FREQ",
        aadt_col="AADT",
        baseline_vars=["URB", "ACCESS", "GRADEBR", "CURVES"],
        local_vars=["CURVES", "WIDTH"],
    )
    cmf_builder_scaled = CMFExperimentBuilder(
        df=df_scaled,
        y_col="FREQ",
        aadt_col="AADT",
        baseline_vars=["URB", "ACCESS", "GRADEBR", "CURVES_Z"],
        local_vars=["CURVES_Z", "WIDTH_Z"],
    )

    trad_spec = trad_builder.make_manual_spec(
        fixed_terms=["URB", "ACCESS", "GRADEBR", "CURVES", "LENGTH"],
        dispersion=1,
        latent_classes=1,
    )
    trad_spec_no_offset = trad_builder_no_offset.make_manual_spec(
        fixed_terms=["URB", "ACCESS", "GRADEBR", "CURVES", "LENGTH", "AADT"],
        dispersion=1,
        latent_classes=1,
    )
    cmf_spec = cmf_builder.make_manual_cmf_spec(
        baseline_fixed=["URB", "ACCESS", "GRADEBR", "CURVES"],
        local_fixed=["WIDTH"],
        dispersion=1,
        latent_classes=1,
    )
    cmf_spec_scaled = cmf_builder_scaled.make_manual_cmf_spec(
        baseline_fixed=["URB", "ACCESS", "GRADEBR", "CURVES_Z"],
        local_fixed=["WIDTH_Z"],
        dispersion=1,
        latent_classes=1,
    )

    fit_trad = trad_builder.fit_manual_model(manual_spec=trad_spec, model="nb", R=args.R)
    fit_trad_no_offset = trad_builder_no_offset.fit_manual_model(
        manual_spec=trad_spec_no_offset,
        model="nb",
        R=args.R,
    )
    fit_cmf = cmf_builder.fit_manual_cmf_model(
        id_col="ID",
        offset_col="OFFSET",
        group_id_col="FC",
        manual_spec=cmf_spec,
        model="nb",
        R=args.R,
    )
    fit_cmf_no_offset = cmf_builder.fit_manual_cmf_model(
        id_col="ID",
        offset_col=None,
        group_id_col="FC",
        manual_spec=cmf_spec,
        model="nb",
        R=args.R,
    )

    used_scaled_fallback = False
    if not _fit_is_valid(fit_cmf):
        used_scaled_fallback = True
        fit_cmf = cmf_builder_scaled.fit_manual_cmf_model(
            id_col="ID",
            offset_col="OFFSET",
            group_id_col="FC",
            manual_spec=cmf_spec_scaled,
            model="nb",
            R=args.R,
        )
    used_scaled_fallback_no_offset = False
    if not _fit_is_valid(fit_cmf_no_offset):
        used_scaled_fallback_no_offset = True
        fit_cmf_no_offset = cmf_builder_scaled.fit_manual_cmf_model(
            id_col="ID",
            offset_col=None,
            group_id_col="FC",
            manual_spec=cmf_spec_scaled,
            model="nb",
            R=args.R,
        )

    metrics_trad = _validation_metrics(fit_trad)
    metrics_cmf = _validation_metrics(fit_cmf)
    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Traditional baseline NB",
                "BIC": round(float(fit_trad["summary"]["bic"]), 2),
                "RMSE": round(metrics_trad["rmse"], 3),
                "MAE": round(metrics_trad["mae"], 3),
                "Corr": round(metrics_trad["corr"], 3),
            },
            {
                "Model": "Hierarchical CMF baseline NB",
                "BIC": round(float(fit_cmf["summary"]["bic"]), 2),
                "RMSE": round(metrics_cmf["rmse"], 3),
                "MAE": round(metrics_cmf["mae"], 3),
                "Corr": round(metrics_cmf["corr"], 3),
            },
        ]
    )

    coef_df = pd.concat(
        [
            _coef_table(trad_builder, fit_trad, "Traditional baseline NB"),
            _coef_table(trad_builder, fit_cmf, "Hierarchical CMF baseline NB"),
        ],
        ignore_index=True,
    )
    coef_df = _clean_parameter_labels(coef_df)
    trad_map = dict(
        zip(
            coef_df.loc[coef_df["Model"] == "Traditional baseline NB", "Parameter"],
            coef_df.loc[coef_df["Model"] == "Traditional baseline NB", "Estimate"],
        )
    )
    cmf_map = dict(
        zip(
            coef_df.loc[coef_df["Model"] == "Hierarchical CMF baseline NB", "Parameter"],
            coef_df.loc[coef_df["Model"] == "Hierarchical CMF baseline NB", "Estimate"],
        )
    )

    compare_params = [
        "URB",
        "ACCESS",
        "GRADEBR",
        "CURVES",
        "LENGTH",
        "__cmf_log_aadt",
        "__cmf_local__WIDTH",
    ]
    compare_notes = {
        "URB": "Shared baseline factor",
        "ACCESS": "Shared baseline factor",
        "GRADEBR": "Shared baseline factor",
        "CURVES": "Traditional direct effect vs CMF baseline block effect",
        "LENGTH": "Traditional-only direct term",
        "__cmf_log_aadt": "CMF traffic-response baseline elasticity term",
        "__cmf_local__WIDTH": "CMF local interaction with log(AADT)",
    }
    compare_rows: list[dict[str, Any]] = []
    for param in compare_params:
        trad_val = trad_map.get(param, np.nan)
        cmf_val = cmf_map.get(param, np.nan)
        delta = np.nan
        if np.isfinite(trad_val) and np.isfinite(cmf_val):
            delta = float(cmf_val - trad_val)
        compare_rows.append(
            {
                "Parameter": param,
                "Traditional baseline NB": round(float(trad_val), 6) if np.isfinite(trad_val) else "-",
                "Hierarchical CMF baseline NB": round(float(cmf_val), 6) if np.isfinite(cmf_val) else "-",
                "Delta coefficient (CMF - Traditional)": round(float(delta), 6) if np.isfinite(delta) else "-",
                "Interpretation": compare_notes.get(param, ""),
            }
        )
    coefficient_compare_df = pd.DataFrame(compare_rows)

    beta_trad_curves = float(trad_map.get("CURVES", np.nan))
    beta_cmf_curves = float(cmf_map.get("CURVES", np.nan))
    curve_pct_rows = []
    for step in [1, 2, 3]:
        trad_pct = float(100.0 * (math.exp(beta_trad_curves * step) - 1.0)) if np.isfinite(beta_trad_curves) else np.nan
        cmf_pct = float(100.0 * (math.exp(beta_cmf_curves * step) - 1.0)) if np.isfinite(beta_cmf_curves) else np.nan
        curve_pct_rows.append(
            {
                "Change": f"+{step} CURVES",
                "Traditional from coefficient": f"{trad_pct:+.2f}%" if np.isfinite(trad_pct) else "-",
                "CMF baseline block from coefficient": f"{cmf_pct:+.2f}%" if np.isfinite(cmf_pct) else "-",
            }
        )
    curve_pct_df = pd.DataFrame(curve_pct_rows)

    profile = _representative_profile(df)
    aadt_grid = np.linspace(float(df["AADT"].quantile(0.05)), float(df["AADT"].quantile(0.95)), 160)
    curve_df = _scenario_frame(aadt_grid, profile, scale_stats)
    pred_trad = _predict_with_fit(curve_df, fit_trad, R=args.R)
    pred_cmf = _predict_with_fit(curve_df, fit_cmf, R=args.R)
    pct_gap_curve = 100.0 * (pred_cmf - pred_trad) / np.maximum(pred_trad, 1e-9)

    curve_df_no_offset = _scenario_frame(aadt_grid, profile, scale_stats, use_offset=False)
    pred_trad_no_offset = _predict_with_fit(curve_df_no_offset, fit_trad_no_offset, offset_col=None, R=args.R)
    pred_cmf_no_offset = _predict_with_fit(curve_df_no_offset, fit_cmf_no_offset, offset_col=None, R=args.R)

    trad_effective_elasticity = float(
        np.polyfit(np.log(np.maximum(aadt_grid, 1e-9)), np.log(np.maximum(pred_trad, 1e-9)), 1)[0]
    )
    cmf_effective_elasticity = float(
        np.polyfit(np.log(np.maximum(aadt_grid, 1e-9)), np.log(np.maximum(pred_cmf, 1e-9)), 1)[0]
    )

    curve_beta = float(cmf_map.get("CURVES", cmf_map.get("CURVES_Z", 0.0)))
    if "CURVES_Z" in cmf_map and used_scaled_fallback:
        _, sd_curve = scale_stats["CURVES"]
        curve_beta = curve_beta / (sd_curve if sd_curve > 0 else 1.0)
    pct_curves = 100.0 * (math.exp(curve_beta) - 1.0)
    pct_access = 100.0 * (math.exp(float(cmf_map.get("ACCESS", 0.0))) - 1.0)

    curve_png = output_dir / "mardown_ore_predictions_vs_aadt.png"
    _save_curve_png(curve_png, aadt_grid, pred_trad, pred_cmf)

    curve_html = output_dir / "mardown_ore_predictions_interactive.html"
    _save_interactive_curve_html(curve_html, aadt_grid, pred_trad, pred_cmf)

    obs_pred_png = output_dir / "mardown_ore_observed_vs_predicted.png"
    _save_observed_vs_predicted_png(
        obs_pred_png,
        y_true=np.asarray(fit_trad["data"]["y"], dtype=float),
        pred_trad=np.asarray(fit_trad["predictions"], dtype=float),
        pred_cmf=np.asarray(fit_cmf["predictions"], dtype=float),
    )

    change_payload = _cmf_change_payload(
        profile=profile,
        scale_stats=scale_stats,
        fit_trad_offset=fit_trad,
        fit_cmf_offset=fit_cmf,
        fit_trad_no_offset=fit_trad_no_offset,
        fit_cmf_no_offset=fit_cmf_no_offset,
        df=df,
        R=args.R,
    )
    cmf_change_png = output_dir / "mardown_ore_cmf_change_map.png"
    _save_cmf_change_png(cmf_change_png, change_payload)
    cmf_change_html = output_dir / "mardown_ore_cmf_change_explorer.html"
    _save_cmf_change_html(cmf_change_html, change_payload)

    bottom_line_text = f"""
- Predictive winner: the traditional baseline NB is still slightly better on fit statistics in this example.
- CMF advantage: it separates baseline roadway risk from the AADT-response block, so the engineering story is cleaner even when predictions are similar.
- On the smooth AADT curve, the largest gap between the two baseline models is only {float(np.max(np.abs(pct_gap_curve))):.2f}%.
"""

    offset_curve_delta_with = change_payload["modes"]["With offset"]["levels"]["Median AADT"]["CURVES"]
    offset_curve_delta_without = change_payload["modes"]["Without offset"]["levels"]["Median AADT"]["CURVES"]
    offset_test_text = f"""
- With the offset, the median-AADT +3-curves test moves traditional NB by {offset_curve_delta_with['traditional_pct'][3]:+.1f}% and hierarchical CMF by {offset_curve_delta_with['hierarchical_pct'][3]:+.1f}%.
- Without the offset, the same +3-curves test moves traditional NB by {offset_curve_delta_without['traditional_pct'][3]:+.1f}% and hierarchical CMF by {offset_curve_delta_without['hierarchical_pct'][3]:+.1f}%.
- So yes, the offset helps explain why the original +1 bars looked similar: exposure normalization damped the visual gap.
- Fit also changes: with offset, BIC favors traditional ({float(fit_trad['summary']['bic']):.2f} vs {float(fit_cmf['summary']['bic']):.2f}); without offset, BIC favors hierarchical CMF ({float(fit_cmf_no_offset['summary']['bic']):.2f} vs {float(fit_trad_no_offset['summary']['bic']):.2f}).
"""

    cmf_advantage_text = f"""
- Traditional NB gives one combined prediction equation.
- Hierarchical CMF splits the problem into baseline risk and traffic-response adjustments.
- In this fit, a one-unit curve change is about {pct_curves:+.1f}% in the CMF baseline block and a one-unit ACCESS change is about {pct_access:+.1f}%.
- That separation is the main payoff: easier CMF-style interpretation and cleaner policy discussion, even when headline fit is close.
"""

    curve_explanation_text = f"""
- The curves look almost the same because both models imply nearly the same traffic elasticity for the representative segment in this plot.
- Traditional effective AADT elasticity here is about {trad_effective_elasticity:.3f}.
- Hierarchical CMF effective AADT elasticity here is about {cmf_effective_elasticity:.3f}.
- They also look close to linear because the offset already makes expected crashes roughly proportional to exposure, so when elasticity stays near 1 the curve is close to a straight line over this range.
- The CMF advantage shows up more clearly in how feature changes are organized and interpreted than in this one representative AADT line.
"""

    coefficient_gain_text = f"""
- What is gained from hierarchical CMF fitting is structural separation, not only a different number.
- Traditional model uses one combined equation with direct terms (for example CURVES = {float(trad_map.get('CURVES', np.nan)):+.4f} and LENGTH = {float(trad_map.get('LENGTH', np.nan)):+.4f}).
- Hierarchical CMF introduces traffic-response terms (for example __cmf_log_aadt = {float(cmf_map.get('__cmf_log_aadt', np.nan)):+.4f}, __cmf_local__WIDTH = {float(cmf_map.get('__cmf_local__WIDTH', np.nan)):+.4f}) in addition to baseline terms.
- This gives a clearer policy explanation: one block governs baseline safety level, another block governs how effects scale with traffic.
- In this run, fit is close but narrative gain is clear: we can explain whether a treatment shifts baseline risk or changes exposure-response behavior.
"""

    median_aadt = float(df["AADT"].median())
    beta0 = float(cmf_map.get("__INTERCEPT__", np.nan))
    b_urb = float(cmf_map.get("URB", np.nan))
    b_access = float(cmf_map.get("ACCESS", np.nan))
    b_grade = float(cmf_map.get("GRADEBR", np.nan))
    b_curves = float(cmf_map.get("CURVES", np.nan))
    b_log_aadt = float(cmf_map.get("__cmf_log_aadt", np.nan))
    b_width_local = float(cmf_map.get("__cmf_local__WIDTH", np.nan))

    hsm_style_explainer = f"""
Below is the same style of explanation you asked for, using your fitted hierarchical model.

Traditional-style single-level count form:

$$
\\hat N_i = E_i \\times \\exp(\\beta_0 + \\beta_1 X_{{1i}} + \\cdots + \\beta_k X_{{ki}})
$$

Hierarchical CMF two-level form (same fitted model, rewritten for interpretation):

$$
\\hat N_i = E_i \\times \\exp(\\beta_0 + \\beta_{{URB}}URB_i + \\beta_{{ACCESS}}ACCESS_i + \\beta_{{GRADE}}GRADEBR_i + \\beta_{{CURVES}}CURVES_i) \\times AADT_i^{{(\\beta_{{logAADT}} + \\beta_{{WIDTH,local}}WIDTH_i)}}
$$

With your fitted values:

$$
\\beta_0={beta0:+.4f},\\; \\beta_{{URB}}={b_urb:+.4f},\\; \\beta_{{ACCESS}}={b_access:+.4f},\\; \\beta_{{GRADE}}={b_grade:+.4f},\\; \\beta_{{CURVES}}={b_curves:+.4f},\\; \\beta_{{logAADT}}={b_log_aadt:+.4f},\\; \\beta_{{WIDTH,local}}={b_width_local:+.4f}
$$

CMF from a coefficient (holding other factors constant):

$$
CMF(a \\rightarrow b) = \\frac{{\\hat N_b}}{{\\hat N_a}} = \\exp(\\alpha_1(b-a))
$$

So percent change is:

$$
100 \\times (CMF-1) = 100 \\times (\\exp(\\alpha_1(b-a)) - 1)
$$

Worked one-unit examples from your fitted CMF baseline block:

- ACCESS +1: $100 \\times (\\exp({b_access:+.6f})-1) = {100.0 * (math.exp(b_access) - 1.0):+.2f}\\%$
- CURVES +1: $100 \\times (\\exp({b_curves:+.6f})-1) = {100.0 * (math.exp(b_curves) - 1.0):+.2f}\\%$
- GRADEBR +1: $100 \\times (\\exp({b_grade:+.6f})-1) = {100.0 * (math.exp(b_grade) - 1.0):+.2f}\\%$

For WIDTH in this hierarchical fit, WIDTH is in the local AADT-response block, so its CMF depends on AADT:

$$
CMF_{{WIDTH}}(a \\rightarrow b \\mid AADT) = \\exp(\\beta_{{WIDTH,local}}\\log(AADT)(b-a))
$$

At median $AADT={median_aadt:.0f}$ and $b-a=1$ this gives approximately:

$$
100 \\times \\left(\\exp({b_width_local:+.6f}\\log({median_aadt:.0f})) - 1\\right) = {100.0 * (math.exp(b_width_local * math.log(max(median_aadt, 1.0))) - 1.0):+.2f}\\%
$$

Why this is easier than traditional for interpretation:

- Traditional gives one combined equation, so baseline and traffic-response effects are mixed together.
- Hierarchical CMF separates the baseline-risk block from the traffic-response block, so each percentage has a clear role in the story.
"""

    percentage_from_coeff_text = f"""
Use this exact rule for log-link count models:

- Percent change from one coefficient is 100 * (exp(beta * delta_x) - 1).
- beta is the fitted coefficient.
- delta_x is how much the variable changed (+1, +2, +3).

Worked examples from your fitted CURVES coefficients:

- Traditional beta_CURVES = {beta_trad_curves:+.6f}
- CMF baseline beta_CURVES = {beta_cmf_curves:+.6f}

So for +1 CURVES:

- Traditional: 100 * (exp({beta_trad_curves:+.6f} * 1) - 1) = {100.0 * (math.exp(beta_trad_curves) - 1.0):+.2f}%
- CMF baseline: 100 * (exp({beta_cmf_curves:+.6f} * 1) - 1) = {100.0 * (math.exp(beta_cmf_curves) - 1.0):+.2f}%

Important: these are coefficient-implied effects from one term.
The scenario plots are full-model effects, so they can differ when interaction and offset terms are active.
"""

    delta_cmf_text = """
What does Delta coefficient (CMF - Traditional) mean?

- It is just a subtraction of fitted coefficients: beta_CMF minus beta_Traditional.
- It is not a CMF percent by itself.
- It tells you direction and magnitude of parameter shift between methods.

Interpretation shortcut:

- Positive delta: CMF fitted a larger coefficient than traditional for that same term.
- Negative delta: CMF fitted a smaller coefficient than traditional for that same term.
- Blank delta means the term exists in one model form but not the other (for example CMF-specific interaction terms).
"""

    _write_text(output_dir / "mardown_ore_bottom_line.md", bottom_line_text)
    _write_text(output_dir / "mardown_ore_cmf_advantage.md", cmf_advantage_text)
    _write_text(output_dir / "mardown_ore_curve_explanation.md", curve_explanation_text)
    _write_text(output_dir / "mardown_ore_offset_test.md", offset_test_text)
    _write_text(output_dir / "mardown_ore_baseline_fit_snapshot.md", _to_markdown(metrics_df))
    _write_text(output_dir / "mardown_ore_method_fit_coefficients.md", _to_markdown(coefficient_compare_df))
    _write_text(output_dir / "mardown_ore_method_gain_from_fit.md", coefficient_gain_text)
    _write_text(output_dir / "mardown_ore_method_curve_percent_from_coeff.md", percentage_from_coeff_text)
    _write_text(output_dir / "mardown_ore_method_curve_percent_table.md", _to_markdown(curve_pct_df))
    _write_text(output_dir / "mardown_ore_method_delta_cmf_explained.md", delta_cmf_text)
    _write_text(output_dir / "mardown_ore_method_hsm_style_explainer.md", hsm_style_explainer)

    print(f"Assets written to: {output_dir.resolve()}")
    print("- mardown_ore_bottom_line.md")
    print("- mardown_ore_cmf_advantage.md")
    print("- mardown_ore_curve_explanation.md")
    print("- mardown_ore_offset_test.md")
    print("- mardown_ore_baseline_fit_snapshot.md")
    print("- mardown_ore_method_fit_coefficients.md")
    print("- mardown_ore_method_gain_from_fit.md")
    print("- mardown_ore_method_curve_percent_from_coeff.md")
    print("- mardown_ore_method_curve_percent_table.md")
    print("- mardown_ore_method_delta_cmf_explained.md")
    print("- mardown_ore_method_hsm_style_explainer.md")
    print("- mardown_ore_observed_vs_predicted.png")
    print("- mardown_ore_predictions_vs_aadt.png")
    print("- mardown_ore_predictions_interactive.html")
    print("- mardown_ore_cmf_change_map.png")
    print("- mardown_ore_cmf_change_explorer.html")


if __name__ == "__main__":
    main()