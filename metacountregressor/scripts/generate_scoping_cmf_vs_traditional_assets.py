from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metacountregressor import (  # noqa: E402
    CMFExperimentBuilder,
    ExperimentBuilder,
    load_example16_3_model_data,
)


def _add_zscore_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        mu = float(out[col].mean())
        sd = float(out[col].std(ddof=0))
        denom = sd if sd > 0 else 1.0
        out[f"{col}_Z"] = (out[col] - mu) / denom
    return out


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


def _summary_or_nan(fit: dict) -> dict:
    s = fit.get("summary")
    if isinstance(s, dict):
        return {
            "bic": s.get("bic", float("nan")),
            "aic": s.get("aic", float("nan")),
            "loglik": s.get("loglik", float("nan")),
            "num_parm": s.get("num_parm", float("nan")),
        }
    return {
        "bic": float("nan"),
        "aic": float("nan"),
        "loglik": float("nan"),
        "num_parm": float("nan"),
    }


def _extract_obs_pred(fit: dict) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(fit.get("data", {}).get("y", []), dtype=float).reshape(-1)
    y_pred = np.asarray(fit.get("predictions", []), dtype=float).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    n = min(y_true.size, y_pred.size)
    return y_true[:n], y_pred[:n]


def _validation_metrics_from_fit(fit: dict) -> dict:
    y_true, y_pred = _extract_obs_pred(fit)
    if y_true.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
            "corr": float("nan"),
            "poisson_dev": float("nan"),
            "obs_mean": float("nan"),
            "pred_mean": float("nan"),
            "n_obs": 0,
        }

    y_pred = np.clip(y_pred, 1e-9, None)
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    bias = float(np.mean(residual))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.size > 1 else float("nan")

    # Poisson deviance for count fit quality diagnostics.
    yt_safe = np.clip(y_true, 1e-12, None)
    dev_terms = np.where(
        y_true > 0,
        y_true * np.log(yt_safe / y_pred) - (y_true - y_pred),
        -(-y_pred),
    )
    poisson_dev = float(2.0 * np.sum(dev_terms))

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "corr": corr,
        "poisson_dev": poisson_dev,
        "obs_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "n_obs": int(y_true.size),
    }


def _obs_pred_bins_df(model_name: str, fit: dict, n_bins: int = 10) -> pd.DataFrame:
    y_true, y_pred = _extract_obs_pred(fit)
    if y_true.size == 0:
        return pd.DataFrame(columns=["Model", "Predicted Decile", "N", "Observed Mean", "Predicted Mean"])

    df_bins = pd.DataFrame({"Observed": y_true, "Predicted": y_pred})
    try:
        df_bins["Predicted Decile"] = pd.qcut(df_bins["Predicted"], q=n_bins, duplicates="drop")
    except Exception:
        df_bins["Predicted Decile"] = "all"

    out = (
        df_bins
        .groupby("Predicted Decile", observed=False)
        .agg(N=("Observed", "size"), **{"Observed Mean": ("Observed", "mean"), "Predicted Mean": ("Predicted", "mean")})
        .reset_index()
    )
    out.insert(0, "Model", model_name)
    out["Observed Mean"] = out["Observed Mean"].astype(float).round(4)
    out["Predicted Mean"] = out["Predicted Mean"].astype(float).round(4)
    return out


def _safe_slug(text: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "_" for ch in text]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _save_obs_pred_plot(model_name: str, fit: dict, output_dir: Path) -> str | None:
    y_true, y_pred = _extract_obs_pred(fit)
    if y_true.size == 0:
        return None

    max_v = float(max(np.max(y_true), np.max(y_pred), 1.0))
    lim = max_v * 1.05
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=150)
    ax.scatter(y_true, y_pred, s=10, alpha=0.35, color="#1f77b4", edgecolors="none")
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.2, color="#d62728", label="45-degree")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Observed vs Predicted\\n{model_name}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()

    file_name = f"obs_vs_pred_{_safe_slug(model_name)}.png"
    out_path = output_dir / file_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return file_name


def _save_obs_pred_panel(fit_map: dict[str, dict], output_dir: Path) -> str | None:
    items = list(fit_map.items())
    if not items:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), dpi=150)
    axes = axes.flatten()

    wrote_any = False
    for i, (name, fit) in enumerate(items[:4]):
        ax = axes[i]
        y_true, y_pred = _extract_obs_pred(fit)
        if y_true.size == 0:
            ax.set_title(f"{name} (no data)")
            ax.axis("off")
            continue

        wrote_any = True
        max_v = float(max(np.max(y_true), np.max(y_pred), 1.0))
        lim = max_v * 1.05
        ax.scatter(y_true, y_pred, s=7, alpha=0.30, color="#1f77b4", edgecolors="none")
        ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0, color="#d62728")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Observed", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.grid(alpha=0.2)

    for j in range(len(items), 4):
        axes[j].axis("off")

    if not wrote_any:
        plt.close(fig)
        return None

    fig.suptitle("Observed vs Predicted (Manual Fits)", fontsize=12)
    fig.tight_layout()
    file_name = "obs_vs_pred_panel_manual_fits.png"
    out_path = output_dir / file_name
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return file_name


def _fit_rows(label: str, family: str, fit: dict, validation: dict | None = None) -> dict:
    s = _summary_or_nan(fit)
    v = validation or {}
    return {
        "Stage": "Manual fit",
        "Family": family,
        "Model": label,
        "BIC": s["bic"],
        "AIC": s["aic"],
        "Log-Likelihood": s["loglik"],
        "Parameters": s["num_parm"],
        "RMSE": v.get("rmse", float("nan")),
        "MAE": v.get("mae", float("nan")),
        "Bias": v.get("bias", float("nan")),
        "R2": v.get("r2", float("nan")),
        "Corr": v.get("corr", float("nan")),
        "Poisson Dev": v.get("poisson_dev", float("nan")),
        "Search Iterations": np.nan,
    }


def _search_row(label: str, family: str, result: dict) -> dict:
    return {
        "Stage": "Search (quick)",
        "Family": family,
        "Model": label,
        "BIC": result.get("best_score", float("nan")),
        "AIC": np.nan,
        "Log-Likelihood": np.nan,
        "Parameters": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "Bias": np.nan,
        "R2": np.nan,
        "Corr": np.nan,
        "Poisson Dev": np.nan,
        "Search Iterations": result.get("iteration", np.nan),
    }


def _collect_coef(builder: ExperimentBuilder, fit: dict, model_name: str) -> pd.DataFrame:
    coef = builder.print_coefficients(fit).copy()
    coef["Model"] = model_name
    coef["Estimate"] = pd.to_numeric(coef["Estimate"], errors="coerce").round(4)
    return coef[["Model", "Parameter", "Type", "Estimate"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scoping CMF vs traditional comparison fits and quick searches."
    )
    parser.add_argument("--fit-draws", type=int, default=120, help="Simulation draws used for manual fits.")
    parser.add_argument("--search-draws", type=int, default=60, help="Simulation draws used in quick search.")
    parser.add_argument("--search-iter", type=int, default=20, help="Small SA iteration budget for quick end-to-end run.")
    parser.add_argument(
        "--search-mode",
        choices=["micro", "sa"],
        default="micro",
        help=(
            "Search strategy: 'micro' runs a tiny hierarchical SA budget for fast end-to-end runs; "
            "'sa' runs a larger hierarchical SA budget."
        ),
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for searches.")
    parser.add_argument(
        "--output-dir",
        default="results/slide_assets",
        help="Directory where markdown/html assets are written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep CMF magnitudes interpretable by scaling local continuous terms
    # before the CMF interaction with log(AADT). Apply on base df so both
    # the shared builder and CMF wrapper see the same derived columns.
    df = _add_zscore_columns(load_example16_3_model_data(), ["CURVES", "WIDTH"])
    df_cmf = df

    # Traditional count builder
    trad_builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="FREQ",
        offset_col="OFFSET",
        group_id_col="FC",
    )

    # CMF builder
    baseline_vars = ["URB", "ACCESS", "GRADEBR", "CURVES_Z"]
    local_vars = ["CURVES_Z", "WIDTH_Z"]
    cmf_builder = CMFExperimentBuilder(
        df=df_cmf,
        y_col="FREQ",
        aadt_col="AADT",
        baseline_vars=baseline_vars,
        local_vars=local_vars,
    )

    print("Fitting scoping traditional baseline NB...")
    trad_baseline_spec = trad_builder.make_manual_spec(
        fixed_terms=["URB", "ACCESS", "GRADEBR", "CURVES", "LENGTH"],
        dispersion=1,
        latent_classes=1,
    )
    fit_trad_baseline = trad_builder.fit_manual_model(
        manual_spec=trad_baseline_spec,
        model="nb",
        R=args.fit_draws,
    )

    print("Fitting scoping traditional random-parameter NB...")
    trad_random_spec = trad_builder.make_manual_spec(
        fixed_terms=["URB", "ACCESS", "GRADEBR", "LENGTH"],
        rdm_terms=["CURVES:lognormal"],
        dispersion=1,
        latent_classes=1,
    )
    fit_trad_random = trad_builder.fit_manual_model(
        manual_spec=trad_random_spec,
        model="nb",
        R=args.fit_draws,
    )

    print("Fitting scoping CMF baseline NB...")
    cmf_baseline_spec = cmf_builder.make_manual_cmf_spec(
        baseline_fixed=["URB", "ACCESS", "GRADEBR"],
        local_fixed=["WIDTH_Z"],
        dispersion=1,
        latent_classes=1,
    )
    fit_cmf_baseline = cmf_builder.fit_manual_cmf_model(
        id_col="ID",
        offset_col="OFFSET",
        group_id_col="FC",
        manual_spec=cmf_baseline_spec,
        model="nb",
        R=args.fit_draws,
    )

    print("Fitting scoping CMF random-parameter NB...")
    cmf_random_spec = cmf_builder.make_manual_cmf_spec(
        baseline_fixed=["URB", "ACCESS", "GRADEBR"],
        baseline_random=["CURVES_Z"],
        local_fixed=["WIDTH_Z"],
        dispersion=1,
        latent_classes=1,
    )
    fit_cmf_random = cmf_builder.fit_manual_cmf_model(
        id_col="ID",
        offset_col="OFFSET",
        group_id_col="FC",
        manual_spec=cmf_random_spec,
        model="nb",
        R=args.fit_draws,
    )

    validation_map = {
        "Traditional baseline NB": _validation_metrics_from_fit(fit_trad_baseline),
        "Traditional random-parameter NB": _validation_metrics_from_fit(fit_trad_random),
        "CMF baseline NB": _validation_metrics_from_fit(fit_cmf_baseline),
        "CMF random-parameter NB": _validation_metrics_from_fit(fit_cmf_random),
    }

    fit_map = {
        "Traditional baseline NB": fit_trad_baseline,
        "Traditional random-parameter NB": fit_trad_random,
        "CMF baseline NB": fit_cmf_baseline,
        "CMF random-parameter NB": fit_cmf_random,
    }

    if args.search_mode == "micro":
        effective_search_iter = max(2, min(int(args.search_iter), 5))
        search_stage = "Search (hierarchical micro-SA)"
        print(
            "Running micro hierarchical SA search "
            f"(iter={effective_search_iter}, draws={args.search_draws})..."
        )
    else:
        effective_search_iter = int(args.search_iter)
        search_stage = "Search (hierarchical SA quick)"
        print(
            "Running hierarchical SA search "
            f"(iter={effective_search_iter}, draws={args.search_draws})..."
        )

    trad_search_eval = trad_builder.build_evaluator(
        variables=["URB", "ACCESS", "GRADEBR", "CURVES", "LENGTH", "WIDTH"],
        mode="single",
        max_latent_classes=1,
        R=args.search_draws,
        default_roles=[0, 1, 2, 3, 5],
    )
    trad_search = trad_builder.run(
        evaluator=trad_search_eval,
        algo="sa",
        max_iter=effective_search_iter,
        seed=args.seed,
    )

    cmf_search_problem = trad_builder.build_evaluator(
        model_family="cmf",
        cmf_driver="jax_count",
        aadt_col="AADT",
        baseline_vars=baseline_vars,
        local_vars=local_vars,
        variables=["URB", "ACCESS", "GRADEBR", "CURVES_Z", "WIDTH_Z"],
        mode="single",
        max_latent_classes=1,
        R=args.search_draws,
        default_roles=[0, 1, 2, 3, 5],
    )
    cmf_search = trad_builder.run_search(
        cmf_search_problem,
        algo="sa",
        max_iter=effective_search_iter,
        seed=args.seed + 1,
    )

    trad_search_iters = trad_search.get("iteration", effective_search_iter)
    cmf_search_iters = cmf_search.get("iteration", effective_search_iter)

    metrics_rows = [
        _fit_rows(
            "Traditional baseline NB",
            "Traditional",
            fit_trad_baseline,
            validation_map["Traditional baseline NB"],
        ),
        _fit_rows(
            "Traditional random-parameter NB",
            "Traditional",
            fit_trad_random,
            validation_map["Traditional random-parameter NB"],
        ),
        _fit_rows(
            "CMF baseline NB",
            "CMF",
            fit_cmf_baseline,
            validation_map["CMF baseline NB"],
        ),
        _fit_rows(
            "CMF random-parameter NB",
            "CMF",
            fit_cmf_random,
            validation_map["CMF random-parameter NB"],
        ),
        {
            **_search_row("Traditional search best (quick)", "Traditional", trad_search),
            "Stage": search_stage,
            "Search Iterations": trad_search_iters,
        },
        {
            **_search_row("CMF search best (quick)", "CMF", cmf_search),
            "Stage": search_stage,
            "Search Iterations": cmf_search_iters,
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows).sort_values("BIC", na_position="last").reset_index(drop=True)

    obs_pred_summary_df = pd.DataFrame([
        {
            "Model": name,
            "N": stats["n_obs"],
            "Observed Mean": round(stats["obs_mean"], 4) if np.isfinite(stats["obs_mean"]) else np.nan,
            "Predicted Mean": round(stats["pred_mean"], 4) if np.isfinite(stats["pred_mean"]) else np.nan,
            "Bias": round(stats["bias"], 4) if np.isfinite(stats["bias"]) else np.nan,
            "RMSE": round(stats["rmse"], 4) if np.isfinite(stats["rmse"]) else np.nan,
            "MAE": round(stats["mae"], 4) if np.isfinite(stats["mae"]) else np.nan,
        }
        for name, stats in validation_map.items()
    ])

    obs_pred_bins_df = pd.concat([
        _obs_pred_bins_df("Traditional baseline NB", fit_trad_baseline),
        _obs_pred_bins_df("Traditional random-parameter NB", fit_trad_random),
        _obs_pred_bins_df("CMF baseline NB", fit_cmf_baseline),
        _obs_pred_bins_df("CMF random-parameter NB", fit_cmf_random),
    ], ignore_index=True)

    plot_files = {}
    for model_name, fit in fit_map.items():
        plot_files[model_name] = _save_obs_pred_plot(model_name, fit, output_dir)
    panel_plot = _save_obs_pred_panel(fit_map, output_dir)

    coef_tables = [
        _collect_coef(trad_builder, fit_trad_baseline, "Traditional baseline NB"),
        _collect_coef(trad_builder, fit_trad_random, "Traditional random-parameter NB"),
        _collect_coef(trad_builder, fit_cmf_baseline, "CMF baseline NB"),
        _collect_coef(trad_builder, fit_cmf_random, "CMF random-parameter NB"),
    ]
    coef_df = pd.concat(coef_tables, ignore_index=True)

    quick_config = pd.DataFrame(
        [
            {"Setting": "Search iterations (SA)", "Value": int(args.search_iter)},
            {"Setting": "Search draws", "Value": int(args.search_draws)},
            {"Setting": "Search mode", "Value": args.search_mode},
            {"Setting": "Fit draws", "Value": int(args.fit_draws)},
            {"Setting": "Traditional search best BIC", "Value": float(trad_search.get("best_score", np.nan))},
            {"Setting": "CMF search best BIC", "Value": float(cmf_search.get("best_score", np.nan))},
        ]
    )

    rmse_best = min(validation_map.items(), key=lambda kv: kv[1]["rmse"] if np.isfinite(kv[1]["rmse"]) else np.inf)
    mae_best = min(validation_map.items(), key=lambda kv: kv[1]["mae"] if np.isfinite(kv[1]["mae"]) else np.inf)
    corr_best = max(validation_map.items(), key=lambda kv: kv[1]["corr"] if np.isfinite(kv[1]["corr"]) else -np.inf)

    rmse_trad_rand = validation_map["Traditional random-parameter NB"]["rmse"]
    rmse_cmf_rand = validation_map["CMF random-parameter NB"]["rmse"]
    rmse_gain = float("nan")
    if np.isfinite(rmse_trad_rand) and np.isfinite(rmse_cmf_rand) and rmse_trad_rand > 0:
        rmse_gain = 100.0 * (rmse_trad_rand - rmse_cmf_rand) / rmse_trad_rand

    key_insights = [
        f"Best RMSE (manual fits): {rmse_best[0]} ({rmse_best[1]['rmse']:.4f})",
        f"Best MAE (manual fits): {mae_best[0]} ({mae_best[1]['mae']:.4f})",
        f"Best observed-predicted correlation: {corr_best[0]} ({corr_best[1]['corr']:.4f})",
    ]
    if np.isfinite(rmse_gain):
        direction = "improvement" if rmse_gain >= 0 else "decline"
        key_insights.append(
            f"CMF random vs Traditional random RMSE: {rmse_gain:+.2f}% ({direction})."
        )

    (output_dir / "scoping_cmf_vs_traditional_metrics.md").write_text(
        _to_markdown(metrics_df),
        encoding="utf-8",
    )
    (output_dir / "scoping_cmf_vs_traditional_coefficients.md").write_text(
        _to_markdown(coef_df),
        encoding="utf-8",
    )
    (output_dir / "scoping_cmf_vs_traditional_coefficients.html").write_text(
        coef_df.to_html(index=False),
        encoding="utf-8",
    )
    (output_dir / "scoping_cmf_vs_traditional_observed_vs_predicted.md").write_text(
        _to_markdown(obs_pred_summary_df),
        encoding="utf-8",
    )
    (output_dir / "scoping_cmf_vs_traditional_observed_vs_predicted_bins.md").write_text(
        _to_markdown(obs_pred_bins_df),
        encoding="utf-8",
    )
    (output_dir / "scoping_cmf_vs_traditional_observed_vs_predicted_bins.html").write_text(
        obs_pred_bins_df.to_html(index=False),
        encoding="utf-8",
    )

    summary_text = (
        "Scoping CMF vs Traditional comparison (quick task run)\n\n"
        "Quick-run configuration\n\n"
        f"{_to_markdown(quick_config)}\n\n"
        "Key insights (Observed vs Predicted + Validation)\n\n"
        + "\n".join(f"- {line}" for line in key_insights)
        + "\n\n"
        "Model comparison metrics\n\n"
        f"{_to_markdown(metrics_df)}\n\n"
        "Observed vs Predicted summary (manual fits)\n\n"
        f"{_to_markdown(obs_pred_summary_df)}\n\n"
        "Observed vs Predicted by predicted-decile (manual fits)\n\n"
        f"{_to_markdown(obs_pred_bins_df)}\n\n"
        "Observed vs Predicted charts (manual fits)\n\n"
        + (f"![Observed vs Predicted Panel](results/slide_assets/{panel_plot})\n\n" if panel_plot else "")
        + "\n".join(
            f"![{name}](results/slide_assets/{file_name})"
            for name, file_name in plot_files.items()
            if file_name
        )
    )
    (output_dir / "scoping_cmf_vs_traditional_differences.md").write_text(summary_text, encoding="utf-8")

    print(f"Assets written to: {output_dir.resolve()}")
    print("Created files:")
    print("- scoping_cmf_vs_traditional_metrics.md")
    print("- scoping_cmf_vs_traditional_coefficients.md")
    print("- scoping_cmf_vs_traditional_coefficients.html")
    print("- scoping_cmf_vs_traditional_observed_vs_predicted.md")
    print("- scoping_cmf_vs_traditional_observed_vs_predicted_bins.md")
    print("- scoping_cmf_vs_traditional_observed_vs_predicted_bins.html")
    if panel_plot:
        print(f"- {panel_plot}")
    for file_name in plot_files.values():
        if file_name:
            print(f"- {file_name}")
    print("- scoping_cmf_vs_traditional_differences.md")


if __name__ == "__main__":
    main()
