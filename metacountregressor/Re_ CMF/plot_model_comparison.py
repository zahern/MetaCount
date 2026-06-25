"""
CMF Model Comparison Plots
Parameters taken directly from the paper tables, NOT from the CSV
(the CSV has at least one confirmed error: Washington SLOPE = -0.0076 → -0.0769).

Table 1 — Washington (Washington 2020 benchmark vs hierarchical)
Table 2 — Queensland HV (fixed benchmark vs hierarchical)
Table 3 — Maine (Islam 2023 benchmark vs hierarchical)

Layout per dataset (2 × 2):
  [0,0]  Benchmark   — scatter observed & predicted + OLS line vs log(AADT)
  [0,1]  Hierarchical — scatter observed & predicted + OLS line vs log(AADT)
  [1,0]  Benchmark residuals vs log(AADT)
  [1,1]  Hierarchical residuals vs log(AADT)
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

HERE      = Path(__file__).parent
WASH_FILE = HERE / "Ex-16-3.csv"
MAINE_FILE = HERE / "rural_int.xlsx"
QLD_FILE  = HERE / "Stage5A_1848_All_Initial_Columns.xlsx"

# ── colours ──────────────────────────────────────────────────────────────────
C_OBS  = "#2e86ab"   # observed (blue)
C_BMARK = "#e07b00"  # benchmark / literature (orange)
C_HIER = "#b22222"   # hierarchical (firebrick)
AP, AB, LW = 0.45, 0.15, 1.8

matplotlib.rc("font", family="serif", size=10)
matplotlib.rc("axes", titlesize=11)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_val(v):
    """Parse comma-decimal strings to float."""
    return float(str(v).strip().replace(",", "."))


def ols_ci(x, y, xg, alpha=0.05):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        nan = np.full_like(xg, np.nan)
        return nan, nan, nan
    sl, ic, *_ = stats.linregress(x, y)
    yhat = sl * x + ic
    se = np.sqrt(np.sum((y - yhat) ** 2) / max(n - 2, 1))
    xb = x.mean()
    ssx = max(np.sum((x - xb) ** 2), 1e-12)
    tc = stats.t.ppf(1 - alpha / 2, df=max(n - 2, 1))
    yg = sl * xg + ic
    seg = se * np.sqrt(1 / max(n, 1) + (xg - xb) ** 2 / ssx)
    return yg, yg - tc * seg, yg + tc * seg


def winsorise(a, pct=97.5):
    pos = a[np.isfinite(a) & (a > 0)]
    hi = np.nanpercentile(pos, pct) if len(pos) else 1e9
    return np.clip(np.where(np.isfinite(a), a, 0.0), 0.0, hi)


def _smart_ylim(*arrs, pct=99, headroom=1.15):
    vals = [np.nanpercentile(a[np.isfinite(a) & (a >= 0)], pct)
            for a in arrs if np.any(np.isfinite(a) & (a >= 0))]
    return max(vals) * headroom if vals else 1.0


def _pred_panel(ax, la, obs, pred, col, label, ylim):
    pw = winsorise(pred)
    xg = np.linspace(la.min() - 0.1, la.max() + 0.1, 300)
    ax.scatter(la, obs, s=9, c=C_OBS, alpha=AP, linewidths=0, zorder=2, label="Observed")
    ax.scatter(la, pw, s=9, c=col, alpha=AP + 0.1, linewidths=0, zorder=3, label="Predicted")
    yg, lo, hi = ols_ci(la, pw, xg)
    ax.plot(xg, yg, c=col, lw=LW, zorder=4)
    ax.fill_between(xg, np.clip(lo, 0, None), hi, color=col, alpha=AB)
    ax.set_ylim(0, ylim)
    ax.set_xlim(xg[[0, -1]])
    ax.set_xlabel("log(AADT)")
    ax.set_ylabel("Crashes")
    ax.legend(fontsize=8, loc="upper left", markerscale=1.5)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(label, fontweight="bold")


def _resid_panel(ax, la, obs, pred, col, label):
    pw = winsorise(pred)
    safe = np.where(pw > 0, pw, np.nanmedian(pw[pw > 0]) if np.any(pw > 0) else 1e-6)
    res = np.clip((obs - pw) / np.sqrt(safe), -10, 10)
    xg = np.linspace(la.min() - 0.1, la.max() + 0.1, 300)
    ax.scatter(la, res, s=7, c=col, alpha=0.35, linewidths=0, zorder=2)
    ax.axhline(0, color="grey", lw=1.0, ls="--")
    yg, lo, hi = ols_ci(la, res, xg)
    ax.plot(xg, yg, c=col, lw=LW, zorder=3)
    ax.fill_between(xg, lo, hi, color=col, alpha=AB)
    ax.set_xlabel("log(AADT)")
    ax.set_ylabel("(obs − pred) / √pred")
    ax.set_title(label, fontweight="bold")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def make_fig(la, obs, bmark, hier, bmark_label, hier_label, suptitle, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)
    ylim = _smart_ylim(obs, bmark, hier)
    _pred_panel(axes[0, 0], la, obs, bmark, C_BMARK, bmark_label, ylim)
    _pred_panel(axes[0, 1], la, obs, hier,  C_HIER,  hier_label,  ylim)
    _resid_panel(axes[1, 0], la, obs, bmark, C_BMARK, f"{bmark_label} — residuals")
    _resid_panel(axes[1, 1], la, obs, hier,  C_HIER,  f"{hier_label} — residuals")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {Path(out).name}")


def _summary(label, obs, pred):
    pw = winsorise(pred)
    res = obs - pw
    rmse = np.sqrt(np.nanmean(res ** 2))
    mae  = np.nanmean(np.abs(res))
    print(f"  {label:30s}  mean_pred={np.nanmean(pw):.4f}  "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  WASHINGTON  (Table 1)
# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark (Washington 2020, RPNB fixed-effect predictions):
#   Np = exp(2.81 − 0.41*LOWPRE − 0.05*GRADEBR − 0.01*FRICTION
#             + 2.70*EXPOSE + 0.15*INTPM − 0.16*CURVES − 0.18*HISNOW)
# (random effects have zero mean, so fixed-effect-only prediction used)
#
# Hierarchical (from paper Table 1):
#   A  params: a0=−10.3774, WIDTH_PER_LANE=−0.0534, CURVES=0.0954,
#              TANGENT_LENGTH=0.1014, INTERCHANGES=0.4192,
#              MXGRDIFF=0.0769, SHOULDER_WIDTH=−0.4338
#   B  params: b0=5.3944, SLOPE_FLAT=−0.0769   ← SLOPE = −0.0769, not −0.0076
#   Formula: Np = exp(Xa @ A) × log(AADT)^(b0 × exp(SLOPE_FLAT × −0.0769))

WASH_BMARK_A = np.array([2.81, -0.41, -0.05, -0.01, 2.70, 0.15, -0.16, -0.18])
WASH_HIER_A  = np.array([-10.3774, -0.0534, 0.0954, 0.1014, 0.4192, 0.0769, -0.4338])
WASH_HIER_b0    = 5.3944
WASH_HIER_slope = -0.0769   # corrected from CSV typo −0.0076


def plot_washington():
    print("\n── Washington ──────────────────────────────────────────────────")
    df = pd.read_csv(WASH_FILE)
    df["WIDTH_PER_LANE"] = df["WIDTH"] / (df["INCLANES"] + df["DECLANES"])
    df["SLOPE_FLAT"]     = (df["SLOPE"] == 0).astype(float)
    df["AADTmaj"]        = np.log(df["AADT"].astype(float))
    df["const"]          = 1.0
    df["SHOULDER_WIDTH"] = df["MIMEDSH"] / df["WIDTH"]
    for c in ["TANGENT", "INTECHAG", "MXGRDIFF", "LOWPRE", "GRADEBR",
              "FRICTION", "EXPOSE", "INTPM", "CURVES", "HISNOW"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Benchmark (fixed-effect prediction only; random-effect mean = 0)
    Xb = df[["const", "LOWPRE", "GRADEBR", "FRICTION",
              "EXPOSE", "INTPM", "CURVES", "HISNOW"]].astype(float).values
    bmark = np.exp(Xb @ WASH_BMARK_A)

    # Hierarchical
    Xa = df[["const", "WIDTH_PER_LANE", "CURVES", "TANGENT",
              "INTECHAG", "MXGRDIFF", "SHOULDER_WIDTH"]].astype(float).values
    A_base = np.exp(Xa @ WASH_HIER_A)
    B_base = np.exp(df["SLOPE_FLAT"].values * WASH_HIER_slope)
    hier   = A_base * np.power(df["AADTmaj"].values, WASH_HIER_b0 * B_base)

    obs = df["FREQ"].values.astype(float)
    la  = df["AADTmaj"].values

    _summary("Benchmark (Washington 2020)", obs, bmark)
    _summary("Hierarchical",                obs, hier)

    make_fig(la, obs, bmark, hier,
             "Benchmark (Washington 2020)",
             "Hierarchical CMF",
             "Washington: Benchmark vs Hierarchical CMF  [Table 1]",
             HERE / "washington_comparison_ols.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  QUEENSLAND  (Table 2)
# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark (fixed effects, stepwise BIC NB, with AADT exposure offset):
#   Formula: Np = AADT × exp(−12.3962 − 1.2646*Nlanes + 0.5442*RS_HS + 2.2823*LNMCV)
#   (offset = log(AADT) → coefficient on log(AADT) forced to 1)
#
# Hierarchical:
#   A params: a0=−10.859, Total_width=−0.2758, RS_HS=1.378, LNMCV=0.9219
#   B params: b0=2.1655, RS=0.1179
#   Formula: Np = exp(Xa @ A) × log(AADT)^(b0 × exp(RS × 0.1179))

QLD_BMARK_A  = np.array([-12.3962, -1.2646, 0.5442, 2.2823])
QLD_HIER_A   = np.array([-10.8590, -0.2758, 1.3780, 0.9219])
QLD_HIER_b0  = 2.1655
QLD_HIER_rs  = 0.1179


def plot_queensland():
    print("\n── Queensland ──────────────────────────────────────────────────")
    df = pd.read_excel(QLD_FILE, sheet_name="Stage5A_1848")
    df["const"]       = 1.0
    df["AADTmaj"]     = np.log(df["AADT"].astype(float))
    df["Total_width"] = df["Lwidth"].astype(float) * df["Nlanes"].astype(float)
    df["RS_HS"]       = df["RS"].astype(float) * df["HSP"].astype(float)
    df["RS"]          = df["RS"].astype(float)
    df["Nlanes"]      = df["Nlanes"].astype(float)
    df["LNMCV"]       = pd.to_numeric(df["LNMCV"], errors="coerce").fillna(0)

    # Benchmark (Table 2) — NB model without exposure offset.
    # Verified: applying these params WITHOUT any offset gives mean=0.058 ≈ Table 4's 0.06.
    # Adding log(AADT) or log(km-years) as offset inflates predictions to ~300+.
    Xb    = df[["const", "Nlanes", "RS_HS", "LNMCV"]].astype(float).values
    bmark = np.exp(Xb @ QLD_BMARK_A)

    # Hierarchical (Table 2)
    Xa   = df[["const", "Total_width", "RS_HS", "LNMCV"]].astype(float).values
    hier = (np.exp(Xa @ QLD_HIER_A) *
            np.power(df["AADTmaj"].values,
                     QLD_HIER_b0 * np.exp(df["RS"].values * QLD_HIER_rs)))

    obs = df["Headon"].astype(float).values
    la  = df["AADTmaj"].values

    _summary("Benchmark (stepwise NB, Table 2)", obs, bmark)
    _summary("Hierarchical",                     obs, hier)

    make_fig(la, obs, bmark, hier,
             "Benchmark (stepwise NB)",
             "Hierarchical CMF",
             "Queensland HV: Benchmark vs Hierarchical CMF  [Table 2]",
             HERE / "queensland_comparison_ols.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAINE  (Table 3)
# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark (Islam 2023, fixed-effect prediction only; random effects = 0):
#   Fixed: intercept=−10.36, MADT=0.805, segment_length=0.955,
#          speed_limit=0.030, shoulder_width=−0.165, curve=0.204
#   Formula: Np = exp(Xa @ A)
#   Note: random parameters on DP01 and DX32 have zero mean → use zero
#
# Hierarchical (Table 3):
#   A params: a0=−57.2378, speed=0.4949, right_shoulder_width=−0.0234,
#             DP01=0.0440, DX32=0.0455
#   B params: b0=7.5549, dummy_winter=0.2350
#   Formula: Np = exp(Xa @ A) × log(AADT)^(b0 × exp(dummy_winter × 0.2350))

MAINE_BMARK_A = np.array([-10.36, 0.805, 0.955, 0.030, -0.165, 0.204])
MAINE_HIER_A  = np.array([-57.2378, 0.4949, -0.0234, 0.0440, 0.0455])
MAINE_HIER_b0     = 7.5549
MAINE_HIER_winter = 0.2350


def plot_maine():
    print("\n── Maine ───────────────────────────────────────────────────────")
    df = pd.read_excel(MAINE_FILE, sheet_name="rural_int")
    df["const"]   = 1.0
    df["AADTmaj"] = np.log(df["monthly_AADT"].astype(float))
    for c in ["speed", "right_shoulder_width", "DP01", "DX32",
              "dummy_winter", "segment_length", "curve"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Benchmark (Islam 2023, fixed effects only; Table 3)
    Xb    = df[["const", "AADTmaj", "segment_length", "speed",
                 "right_shoulder_width", "curve"]].astype(float).values
    bmark = np.exp(Xb @ MAINE_BMARK_A)

    # Hierarchical (Table 3)
    Xa   = df[["const", "speed", "right_shoulder_width",
                "DP01", "DX32"]].astype(float).values
    hier = (np.exp(Xa @ MAINE_HIER_A) *
            np.power(df["AADTmaj"].values,
                     MAINE_HIER_b0 * np.exp(df["dummy_winter"].values * MAINE_HIER_winter)))

    obs = df["crashes"].astype(float).values
    la  = df["AADTmaj"].values

    _summary("Benchmark (Islam 2023, Table 3)", obs, bmark)
    _summary("Hierarchical",                    obs, hier)

    make_fig(la, obs, bmark, hier,
             "Benchmark (Islam 2023)",
             "Hierarchical CMF",
             "Maine: Benchmark vs Hierarchical CMF  [Table 3]",
             HERE / "maine_comparison_ols.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  COMBINED  (all three side-by-side, predictions + residuals)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_combined():
    print("\n── Combined ────────────────────────────────────────────────────")
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(
        "CMF Benchmark vs Hierarchical — OLS trend ± 95% CI  |  Pearson residuals",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Washington ──
    df = pd.read_csv(WASH_FILE)
    df["WIDTH_PER_LANE"] = df["WIDTH"] / (df["INCLANES"] + df["DECLANES"])
    df["SLOPE_FLAT"]     = (df["SLOPE"] == 0).astype(float)
    df["AADTmaj"]        = np.log(df["AADT"].astype(float))
    df["const"]          = 1.0
    df["SHOULDER_WIDTH"] = df["MIMEDSH"] / df["WIDTH"]
    for c in ["TANGENT","INTECHAG","MXGRDIFF","LOWPRE","GRADEBR","FRICTION","EXPOSE","INTPM","CURVES","HISNOW"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    bmark_w = np.exp(df[["const","LOWPRE","GRADEBR","FRICTION","EXPOSE","INTPM","CURVES","HISNOW"]].astype(float).values @ WASH_BMARK_A)
    hier_w  = (np.exp(df[["const","WIDTH_PER_LANE","CURVES","TANGENT","INTECHAG","MXGRDIFF","SHOULDER_WIDTH"]].astype(float).values @ WASH_HIER_A) *
               np.power(df["AADTmaj"].values, WASH_HIER_b0 * np.exp(df["SLOPE_FLAT"].values * WASH_HIER_slope)))
    obs_w, la_w = df["FREQ"].values.astype(float), df["AADTmaj"].values

    # ── Queensland ──
    df2 = pd.read_excel(QLD_FILE, sheet_name="Stage5A_1848")
    df2["const"]       = 1.0
    df2["AADTmaj"]     = np.log(df2["AADT"].astype(float))
    df2["Total_width"] = df2["Lwidth"].astype(float) * df2["Nlanes"].astype(float)
    df2["RS_HS"]       = df2["RS"].astype(float) * df2["HSP"].astype(float)
    df2["RS"]          = df2["RS"].astype(float)
    df2["Nlanes"]      = df2["Nlanes"].astype(float)
    df2["LNMCV"]       = pd.to_numeric(df2["LNMCV"], errors="coerce").fillna(0)
    bmark_q = np.exp(df2[["const","Nlanes","RS_HS","LNMCV"]].astype(float).values @ QLD_BMARK_A)
    hier_q  = (np.exp(df2[["const","Total_width","RS_HS","LNMCV"]].astype(float).values @ QLD_HIER_A) *
               np.power(df2["AADTmaj"].values, QLD_HIER_b0 * np.exp(df2["RS"].values * QLD_HIER_rs)))
    obs_q, la_q = df2["Headon"].astype(float).values, df2["AADTmaj"].values

    # ── Maine ──
    df3 = pd.read_excel(MAINE_FILE, sheet_name="rural_int")
    df3["const"]   = 1.0
    df3["AADTmaj"] = np.log(df3["monthly_AADT"].astype(float))
    for c in ["speed","right_shoulder_width","DP01","DX32","dummy_winter","segment_length","curve"]:
        df3[c] = pd.to_numeric(df3[c], errors="coerce").fillna(0)
    bmark_m = np.exp(df3[["const","AADTmaj","segment_length","speed","right_shoulder_width","curve"]].astype(float).values @ MAINE_BMARK_A)
    hier_m  = (np.exp(df3[["const","speed","right_shoulder_width","DP01","DX32"]].astype(float).values @ MAINE_HIER_A) *
               np.power(df3["AADTmaj"].values, MAINE_HIER_b0 * np.exp(df3["dummy_winter"].values * MAINE_HIER_winter)))
    obs_m, la_m = df3["crashes"].astype(float).values, df3["AADTmaj"].values

    datasets = [
        (la_w, obs_w, bmark_w, hier_w, "Washington 2020", "Washington"),
        (la_q, obs_q, bmark_q, hier_q, "Stepwise NB",     "Queensland HV"),
        (la_m, obs_m, bmark_m, hier_m, "Islam 2023",      "Maine"),
    ]

    for ci, (la, obs, bmark, hier, bmlbl, ds) in enumerate(datasets):
        ylim = _smart_ylim(obs, bmark, hier)
        xg   = np.linspace(la.min() - 0.1, la.max() + 0.1, 300)

        ax_top = axes[0, ci]
        ax_top.scatter(la, obs,             s=7, c=C_OBS,   alpha=0.35, linewidths=0, zorder=1, label="Observed")
        ax_top.scatter(la, winsorise(bmark), s=6, c=C_BMARK, alpha=0.35, linewidths=0, zorder=2)
        ax_top.scatter(la, winsorise(hier),  s=6, c=C_HIER,  alpha=0.35, linewidths=0, zorder=2)
        for preds, col, lbl in [(bmark, C_BMARK, bmlbl), (hier, C_HIER, "Hierarchical")]:
            pw = winsorise(preds)
            yg, lo, hi = ols_ci(la, pw, xg)
            ax_top.plot(xg, yg, c=col, lw=LW, label=lbl)
            ax_top.fill_between(xg, np.clip(lo, 0, None), hi, color=col, alpha=AB)
        ax_top.set_ylim(0, ylim)
        ax_top.set_xlim(xg[[0, -1]])
        ax_top.set_title(ds, fontweight="bold")
        ax_top.set_xlabel("log(AADT)")
        ax_top.set_ylabel("Crashes")
        ax_top.legend(fontsize=8, loc="upper left", markerscale=1.5)
        ax_top.grid(True, alpha=0.2, lw=0.5)
        ax_top.spines[["top", "right"]].set_visible(False)

        ax_bot = axes[1, ci]
        ax_bot.axhline(0, color="grey", lw=1.0, ls="--")
        for preds, col, lbl in [(bmark, C_BMARK, bmlbl), (hier, C_HIER, "Hierarchical")]:
            pw   = winsorise(preds)
            safe = np.where(pw > 0, pw, np.nanmedian(pw[pw > 0]) if np.any(pw > 0) else 1e-6)
            res  = np.clip((obs - pw) / np.sqrt(safe), -10, 10)
            ax_bot.scatter(la, res, s=5, c=col, alpha=0.28, linewidths=0, zorder=2)
            yg, lo, hi = ols_ci(la, res, xg)
            ax_bot.plot(xg, yg, c=col, lw=LW, label=lbl)
            ax_bot.fill_between(xg, lo, hi, color=col, alpha=AB)
        ax_bot.set_title(f"{ds} — Pearson residuals", fontweight="bold")
        ax_bot.set_xlabel("log(AADT)")
        ax_bot.set_ylabel("(obs − pred) / √pred")
        ax_bot.legend(fontsize=8, loc="upper right")
        ax_bot.grid(True, alpha=0.2, lw=0.5)
        ax_bot.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(HERE / "all_datasets_comparison_ols.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: all_datasets_comparison_ols.png")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    plot_washington()
    plot_queensland()
    plot_maine()
    plot_combined()
    print("\nDone — four PNGs in", HERE)
