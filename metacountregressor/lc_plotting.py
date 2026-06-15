from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Figure 1 — SA structure-search convergence
# ---------------------------------------------------------------------------

def plot_search_convergence(trace_csv_path: str, output_path: Optional[str] = None):
    trace_df = pd.read_csv(trace_csv_path)
    bics = trace_df["bic"].values
    fits = trace_df["fit"].values
    best_so_far = np.minimum.accumulate(bics)

    if output_path is None:
        output_path = str(Path(trace_csv_path).with_suffix(".png"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(fits, bics, s=12, alpha=0.6, color="steelblue", edgecolors="none")
    ax1.plot(fits, best_so_far, "r-", linewidth=1.5, label="Best so far")
    ax1.set_xlabel("Fit evaluation #")
    ax1.set_ylabel("BIC (lower = better)")
    ax1.set_title(
        f"SA Structure Search: BIC per Evaluation\n"
        f"(2-class latent-class model, DE warm-up, best={best_so_far[-1]:.1f})"
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    finite_bics = [b for b in bics if b < 1e11]
    ax2.hist(
        finite_bics,
        bins=min(30, len(finite_bics)),
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
    )
    ax2.axvline(
        best_so_far[-1],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Best = {best_so_far[-1]:.1f}",
    )
    ax2.set_xlabel("BIC")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of BIC Values")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "SA Structure Search Convergence\n(2-class latent-class, FC held out)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 2 — Objective function trace (DE warm-up + EM + LBFGS polish)
# ---------------------------------------------------------------------------

def plot_objective_trace(trace_csv_path: str, output_path: Optional[str] = None):
    trace_df = pd.read_csv(trace_csv_path)
    trace_df["step_global"] = range(len(trace_df))

    if output_path is None:
        output_path = str(Path(trace_csv_path).with_suffix(".png"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    phases = trace_df["phase"].unique()
    colors = {"seed": "gray", "DE": "blue", "EM": "green", "LBFGS": "red"}
    for ph in phases:
        mask = trace_df["phase"] == ph
        ax1.plot(
            trace_df.loc[mask, "step_global"],
            trace_df.loc[mask, "neg_loglik"],
            marker="." if ph == "DE" else "o" if ph == "LBFGS" else None,
            markersize=4,
            color=colors.get(ph, "black"),
            label=ph,
            alpha=0.8,
            linewidth=1.5,
        )

    ax1.set_xlabel("Cumulative step")
    ax1.set_ylabel("Negative Log-Likelihood (lower = better)")
    ax1.set_title("LC Model Objective Trace (all phases)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    em_mask = trace_df["phase"] == "EM"
    em_df = trace_df[em_mask]
    if len(em_df) > 0:
        ax2.plot(em_df["step"], em_df["loglik"], "g.-", linewidth=1.5, markersize=4)
        ax2.set_xlabel("EM Iteration")
        ax2.set_ylabel("Log-Likelihood (higher = better)")
        ax2.set_title("EM Convergence: Log-Likelihood by Iteration")
        ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "LC Model: Objective Function Over Time\n"
        "(2 classes, no membership covariates, FC & URB held out)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 3 — Class assignment confusion matrix heatmap + covariate profiles
# ---------------------------------------------------------------------------

def plot_class_profiles(
    assignments_csv_path: str,
    output_path: Optional[str] = None,
    *,
    fc_col: str = "FC",
    class_col: str = "class",
    confidence_col: str = "class_confidence",
):
    df = pd.read_csv(assignments_csv_path)

    if output_path is None:
        output_path = str(Path(assignments_csv_path).with_suffix(".png"))

    profile_vars = [
        "SPEED", "AADT", "WIDTH", "CURVES", "MINRAD",
        "ACCESS", "MEDWIDTH", "Y", "LENGTH", "SINGLE",
        "DOUBLE", "TRAIN", "GRADEBR", "TANGENT", "SLOPE",
        "FRICTION", "EXPOSE", "INCLANES", "GBRPM", "INTPM", "CPM",
    ]
    profile_vars = [v for v in profile_vars if v in df.columns]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # --- Subplot 1: Confusion matrix heatmap (FC vs assigned class) ---
    ax_cm = fig.add_subplot(gs[0, 0])
    if fc_col in df.columns:
        classes_sorted = sorted(df[class_col].dropna().unique())
        fc_vals = sorted(df[fc_col].dropna().unique())
        cm = np.zeros((len(fc_vals), len(classes_sorted)), dtype=int)
        fc_labels = [f"FC={int(v)}" for v in fc_vals]
        class_labels = [f"Class {int(c)}" for c in classes_sorted]

        for i, fc_val in enumerate(fc_vals):
            for j, cl_val in enumerate(classes_sorted):
                cm[i, j] = ((df[fc_col] == fc_val) & (df[class_col] == cl_val)).sum()

        # Row-normalize for percentages
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(
            cm_pct,
            annot=np.char.add(cm.astype(str), np.char.add("\n(", np.char.add(cm_pct.round(1).astype(str), "%)"))),
            fmt="",
            xticklabels=class_labels,
            yticklabels=fc_labels,
            cmap="Blues",
            ax=ax_cm,
            cbar_kws={"label": "% of FC group"},
            linewidths=0.5,
        )
        ax_cm.set_title("Latent Class vs Functional Classification (FC)\n(% of FC group assigned to each class)", fontsize=10)
        ax_cm.set_ylabel("Functional Classification")
        ax_cm.set_xlabel("Assigned Latent Class")
    else:
        ax_cm.text(0.5, 0.5, "FC column not available", ha="center", va="center", transform=ax_cm.transAxes)
        ax_cm.set_title("Confusion Matrix")

    # --- Subplot 2: Class proportions bar chart ---
    ax_prop = fig.add_subplot(gs[0, 1])
    class_counts = df[class_col].value_counts().sort_index()
    bars = ax_prop.bar(
        class_counts.index.astype(str),
        class_counts.values,
        color=["#3498db", "#e74c3c", "#2ecc71"][:len(class_counts)],
        edgecolor="white",
    )
    for bar, val in zip(bars, class_counts.values):
        ax_prop.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(class_counts.values) * 0.01,
            f"{val}\n({val / class_counts.sum():.1%})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax_prop.set_title("Segment Count per Latent Class", fontsize=10)
    ax_prop.set_xlabel("Assigned Latent Class")
    ax_prop.set_ylabel("Number of Segments")
    ax_prop.grid(axis="y", alpha=0.3)

    # --- Subplot 3: Posterior confidence histogram ---
    ax_conf = fig.add_subplot(gs[0, 2])
    if confidence_col in df.columns:
        ax_conf.hist(df[confidence_col], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax_conf.axvline(df[confidence_col].mean(), color="red", linestyle="--",
                        linewidth=1.5, label=f"Mean = {df[confidence_col].mean():.3f}")
        ax_conf.set_title(f"Posterior Assignment Confidence\n(mean = {df[confidence_col].mean():.3f})", fontsize=10)
        ax_conf.set_xlabel("Max Posterior Probability")
        ax_conf.set_ylabel("Number of Segments")
        ax_conf.legend(fontsize=8)
        ax_conf.grid(True, alpha=0.3)

    # --- Subplot 4: Covariate profiles by class (bar chart, top discriminators) ---
    ax_prof = fig.add_subplot(gs[1, :])
    if len(profile_vars) > 0:
        # Compute means per class
        class_vals = sorted(df[class_col].dropna().unique())
        means_list = []
        for cv in class_vals:
            mask = df[class_col] == cv
            means_list.append(df.loc[mask, profile_vars].mean())

        # Z-score normalize each variable for visual comparison
        all_data = df[profile_vars].values
        all_mean = np.nanmean(all_data, axis=0)
        all_std = np.nanstd(all_data, axis=0)
        all_std = np.where(all_std == 0, 1.0, all_std)

        normalized_means = {}
        for cv, means in zip(class_vals, means_list):
            normalized_means[cv] = (means.values - all_mean) / all_std

        # Select top discriminators by max absolute z-difference
        if len(class_vals) >= 2:
            diffs = np.abs(normalized_means[class_vals[0]] - normalized_means[class_vals[-1]])
            top_k = min(12, len(profile_vars))
            top_idx = np.argsort(diffs)[-top_k:][::-1]
            top_vars = [profile_vars[i] for i in top_idx]
        else:
            top_vars = profile_vars[:12]
            top_idx = np.arange(len(top_vars))

        x = np.arange(len(top_vars))
        width = 0.35
        colors_c = ["#3498db", "#e74c3c", "#2ecc71"]
        for j, cv in enumerate(class_vals):
            vals = [normalized_means[cv][i] for i in top_idx]
            offset = (j - (len(class_vals) - 1) / 2) * width
            ax_prof.bar(x + offset, vals, width, label=f"Class {int(cv)}",
                        color=colors_c[j % len(colors_c)], edgecolor="white", alpha=0.85)

        ax_prof.set_xticks(x)
        ax_prof.set_xticklabels(top_vars, rotation=45, ha="right", fontsize=8)
        ax_prof.set_ylabel("Standardized Mean (z-score)", fontsize=9)
        ax_prof.set_title("Covariate Profiles by Latent Class\n(top discriminators, z-score normalized)", fontsize=10)
        ax_prof.legend(fontsize=8)
        ax_prof.axhline(0, color="black", linewidth=0.5)
        ax_prof.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Latent Class Analysis: Validation and Profiling\n"
        "(FC & URB held out, recovered via latent classes)",
        fontsize=13,
        fontweight="bold",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Convenience: generate all 3 figures from results directory
# ---------------------------------------------------------------------------

def generate_all_lc_figures(
    results_dir: str,
    output_dir: Optional[str] = None,
):
    results_dir = str(results_dir)
    if output_dir is None:
        output_dir = results_dir
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # Figure 1 — Search convergence trace
    search_csv = os.path.join(results_dir, "lc_search_trace.csv")
    if os.path.exists(search_csv):
        fig1_path = os.path.join(output_dir, "lc_search_trace.png")
        paths["fig1_search_trace"] = plot_search_convergence(search_csv, fig1_path)
    else:
        print(f"  [skip] Figure 1: {search_csv} not found. Run Phase 1 search first.")

    # Figure 2 — Objective trace
    obj_csv = os.path.join(results_dir, "lc_objective_trace.csv")
    if os.path.exists(obj_csv):
        fig2_path = os.path.join(output_dir, "lc_objective_trace.png")
        paths["fig2_objective_trace"] = plot_objective_trace(obj_csv, fig2_path)
    else:
        print(f"  [skip] Figure 2: {obj_csv} not found. Run Phase 2 analysis first.")

    # Figure 3 — Class profiles & confusion
    assignments_csv = os.path.join(results_dir, "lc_class_assignments.csv")
    if os.path.exists(assignments_csv):
        fig3_path = os.path.join(output_dir, "lc_class_profiles.png")
        paths["fig3_class_profiles"] = plot_class_profiles(assignments_csv, fig3_path)
    else:
        print(f"  [skip] Figure 3: {assignments_csv} not found. Run class profiling first.")

    return paths
