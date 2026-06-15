from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _compute_tight_axis_window(values, objective="bic", padding=0.08):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    v_min, v_max = float(np.min(values)), float(np.max(values))
    span = v_max - v_min
    if span <= 0:
        return None
    return (v_min - span * padding, v_max + span * padding)


# ---------------------------------------------------------------------------
# Figure 1 — Search convergence (BIC + Validation RMSE over iterations)
# ---------------------------------------------------------------------------

def plot_cmf_search_convergence(
    history_csv: str,
    output_path: Optional[str] = None,
    dataset_label: str = "Ex16-3 Washington",
):
    df = pd.read_csv(history_csv)
    n = len(df)
    iters = list(range(1, n + 1))

    if "BIC" not in df.columns:
        print(f"  [skip] BIC column not in {history_csv}")
        return None

    bic_vals = df["BIC"].tolist()

    if "Val RMSE" in df.columns:
        val_rmse = df["Val RMSE"].tolist()
    elif "Val Poisson Dev" in df.columns:
        val_rmse = df["Val Poisson Dev"].tolist()
    else:
        val_rmse = None

    has_mono = "Monotonic AADT (e>0 all segs)" in df.columns
    mono_ok = [str(v).lower() == "yes" for v in df.get("Monotonic AADT (e>0 all segs)", ["yes"] * n)] if has_mono else [True] * n
    colors = ["#2a7f4f" if ok else "#9e9e9e" for ok in mono_ok]

    run_min_bic = [min(bic_vals[:i + 1]) for i in range(n)]

    if output_path is None:
        output_path = str(Path(history_csv).parent / "cmf_search_convergence.png")

    if val_rmse is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

        bic_window = _compute_tight_axis_window(bic_vals, objective="bic")
        ax1.scatter(iters, bic_vals, c=colors, s=22, alpha=0.65, zorder=3)
        ax1.plot(iters, run_min_bic, color="#0a6c74", lw=2.5, label="Running min")
        ax1.set_title("BIC over Search Iterations", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("BIC")
        if bic_window is not None:
            ax1.set_ylim(*bic_window)
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.2)

        run_min_rmse = [min(val_rmse[:i + 1]) for i in range(n)]
        rmse_window = _compute_tight_axis_window(val_rmse, objective="rmse")
        ax2.scatter(iters, val_rmse, c=colors, s=22, alpha=0.65, zorder=3,
                    label="Green = monotonic AADT" if has_mono else None)
        ax2.plot(iters, run_min_rmse, color="#d96f32", lw=2.5, label="Running min")
        ax2.set_title("Validation RMSE over Iterations", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Val RMSE")
        if rmse_window is not None:
            ax2.set_ylim(*rmse_window)
        if has_mono:
            ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2)

        fig.suptitle(f"{dataset_label} \u2014 Search Convergence", fontsize=12, y=1.01)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 4.5), dpi=150)
        ax1.scatter(iters, bic_vals, c=colors, s=22, alpha=0.65, zorder=3)
        ax1.plot(iters, run_min_bic, color="#0a6c74", lw=2.5, label="Running min")
        ax1.set_title(f"{dataset_label} \u2014 Search Convergence", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("BIC")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 2 — Observed vs Predicted crashes (by log AADT, with residuals)
# ---------------------------------------------------------------------------

def plot_cmf_obs_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aadt: np.ndarray,
    output_path: str,
    title: str = "Observed vs Predicted Crashes",
    split_labels: Optional[list[str]] = None,
):
    y, p, a = (np.asarray(x, float) for x in (y_true, y_pred, aadt))
    p_clip = np.clip(p, 0, np.quantile(p, 0.99))
    log_a = np.log(np.clip(a, 1.0, None))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=150)

    color_map = {
        "train": "#0a6c74", "val": "#d96f32", "validation": "#d96f32",
        "test": "#7a3c8c", "all": "#1f5f8b",
    }

    if split_labels:
        labels = list(split_labels)
        for lbl in sorted(set(labels)):
            mask = np.array([l == lbl for l in labels])
            ax1.scatter(log_a[mask], y[mask], s=18, alpha=0.55,
                        color=color_map.get(lbl, "#4a6785"),
                        label=f"Observed ({lbl})", zorder=3)
    else:
        ax1.scatter(log_a, y, s=18, alpha=0.55, color="#0a6c74",
                    label="Observed", zorder=3)

    try:
        coeffs = np.polyfit(log_a, p_clip, deg=3)
        log_a_fine = np.linspace(log_a.min(), log_a.max(), 200)
        p_fine = np.maximum(np.polyval(coeffs, log_a_fine), 0.0)
    except Exception:
        si = np.argsort(log_a)
        log_a_fine = log_a[si]
        p_fine = p_clip[si]

    ax1.plot(log_a_fine, p_fine, color="#d96f32", lw=2.5, label="Predicted (polynomial smooth)")
    ax1.set_xlabel("log(AADT)")
    ax1.set_ylabel("Crashes")
    ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    resid = (p - y) / np.maximum(y, 1.0)
    resid_colors = ["#d96f32" if r > 0 else "#0a6c74" for r in resid]
    ax2.scatter(log_a, resid, c=resid_colors, s=14, alpha=0.55)
    ax2.axhline(0, color="#333", lw=1.5)
    ax2.set_xlabel("log(AADT)")
    ax2.set_ylabel("Norm. Residual")
    ax2.set_title("Residuals = (pred-obs)/max(obs,1)", fontsize=10)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 3 — Model comparison: Benchmark vs Hierarchical CMF
# ---------------------------------------------------------------------------

def plot_cmf_model_comparison(
    y_all: np.ndarray,
    aadt_all: np.ndarray,
    pred_benchmark: np.ndarray,
    pred_hierarchical: np.ndarray,
    output_path: str,
    title: str = "Benchmark vs Hierarchical CMF Comparison",
    split_labels: Optional[list[str]] = None,
):
    y = np.asarray(y_all, float)
    a = np.asarray(aadt_all, float)
    pb = np.clip(np.asarray(pred_benchmark, float), 0, np.quantile(pred_benchmark, 0.99))
    ph = np.clip(np.asarray(pred_hierarchical, float), 0, np.quantile(pred_hierarchical, 0.99))
    log_a = np.log(np.clip(a, 1.0, None))

    def _smooth_log(log_aadt, pred):
        try:
            coeffs = np.polyfit(log_aadt, pred, deg=3)
            la_fine = np.linspace(log_aadt.min(), log_aadt.max(), 200)
            p_fine = np.maximum(np.polyval(coeffs, la_fine), 0.0)
            return la_fine, p_fine
        except Exception:
            si = np.argsort(log_aadt)
            return log_aadt[si], pred[si]

    sla_b, sp_b = _smooth_log(log_a, pb)
    sla_h, sp_h = _smooth_log(log_a, ph)

    color_map = {
        "train": "#0a6c74", "val": "#d96f32", "validation": "#d96f32",
        "test": "#7a3c8c", "all": "#4a6785",
    }
    obs_c = [color_map.get(str(l).lower(), "#4a6785") for l in split_labels] if split_labels else ["#0a6c74"] * len(y)

    rb = (pb - y) / np.maximum(y, 1.0)
    rh = (ph - y) / np.maximum(y, 1.0)
    y_max = max(y.max(), pb.max(), ph.max()) * 1.05

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    (ax_bl, ax_hr), (ax_br, ax_hrr) = axes

    ax_bl.scatter(log_a, y, c=obs_c, s=14, alpha=0.55, zorder=3, label="Observed")
    ax_bl.plot(sla_b, sp_b, "-", color="#1f6bb5", lw=2.5, label="Predicted")
    ax_bl.set_ylim(0, y_max)
    ax_bl.set_xlabel("log(AADT)")
    ax_bl.set_ylabel("Crashes")
    ax_bl.set_title("Benchmark: AADT-only SPF", fontsize=10, fontweight="bold", color="#1f6bb5")
    ax_bl.legend(fontsize=7)
    ax_bl.grid(alpha=0.18)

    ax_hr.scatter(log_a, y, c=obs_c, s=14, alpha=0.55, zorder=3, label="Observed")
    ax_hr.plot(sla_h, sp_h, "-", color="#d96f32", lw=2.5, label="Predicted")
    ax_hr.set_ylim(0, y_max)
    ax_hr.set_xlabel("log(AADT)")
    ax_hr.set_ylabel("Crashes")
    ax_hr.set_title("Hierarchical CMF model", fontsize=10, fontweight="bold", color="#d96f32")
    ax_hr.legend(fontsize=7)
    ax_hr.grid(alpha=0.18)

    ax_br.scatter(log_a, rb, c=["#1f6bb5" if r > 0 else "#93c5e8" for r in rb], s=12, alpha=0.55)
    ax_br.axhline(0, color="#555", lw=1.2)
    ax_br.set_xlabel("log(AADT)")
    ax_br.set_ylabel("(pred-obs)/max(obs,1)")
    ax_br.set_title("Benchmark residuals", fontsize=9)
    ax_br.grid(alpha=0.18)

    ax_hrr.scatter(log_a, rh, c=["#d96f32" if r > 0 else "#0a6c74" for r in rh], s=12, alpha=0.55)
    ax_hrr.axhline(0, color="#555", lw=1.2)
    ax_hrr.set_xlabel("log(AADT)")
    ax_hrr.set_ylabel("(pred-obs)/max(obs,1)")
    ax_hrr.set_title("Hierarchical residuals", fontsize=9)
    ax_hrr.grid(alpha=0.18)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Convenience: generate all 3 CMF figures from an output directory
# ---------------------------------------------------------------------------

def generate_all_cmf_figures(
    output_dir: str,
    dataset_label: str = "Ex16-3 Washington",
):
    output_dir = str(output_dir)

    history_csv = os.path.join(output_dir, "search_history_full.csv")
    fig1_path = os.path.join(output_dir, "fig1_search_convergence.png")

    if os.path.exists(history_csv):
        plot_cmf_search_convergence(history_csv, fig1_path, dataset_label)
        print(f"  Figure 1 (search convergence): {fig1_path}")
    else:
        print(f"  [skip] Figure 1: {history_csv} not found")

    # Figure 2 is already produced as validation_observed_vs_predicted.png
    # by generate_washington_hierarchical_cmf_assets.py
    fig2_src = os.path.join(output_dir, "validation_observed_vs_predicted.png")
    fig2_path = os.path.join(output_dir, "fig2_obs_vs_pred.png")

    if os.path.exists(fig2_src):
        import shutil
        shutil.copy(fig2_src, fig2_path)
        print(f"  Figure 2 (obs vs pred): {fig2_path}")
    else:
        print(f"  [skip] Figure 2: {fig2_src} not found. Run the full CMF pipeline first.")

    # Figure 3 is already produced as model_comparison.png
    fig3_src = os.path.join(output_dir, "model_comparison.png")
    fig3_path = os.path.join(output_dir, "fig3_model_comparison.png")

    if os.path.exists(fig3_src):
        import shutil
        shutil.copy(fig3_src, fig3_path)
        print(f"  Figure 3 (model comparison): {fig3_path}")
    else:
        print(f"  [skip] Figure 3: {fig3_src} not found. Run the full CMF pipeline first.")

    return {
        "fig1": fig1_path,
        "fig2": fig2_path,
        "fig3": fig3_path,
    }
