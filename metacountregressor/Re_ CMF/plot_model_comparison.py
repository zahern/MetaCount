"""
CMF Model Comparison Plots — OLS straight-line + residuals edition.

Layout per dataset (2 × 2):
  [0,0] Hierarchical  — observed & predicted scatter + OLS trend
  [0,1] Benchmark     — observed & predicted scatter + OLS trend
  [1,0] Hierarchical residuals vs log(AADT)
  [1,1] Benchmark residuals vs log(AADT)

Benchmark models for Maine and Queensland are fitted fresh using JAX autodiff
+ jaxopt LBFGS (matching metacountregressor's NB2 fitting pipeline).
Washington literature parameters come from the CSV (externally estimated).

Outputs:
  washington_comparison_ols.png
  maine_comparison_ols.png
  queensland_comparison_ols.png
  all_datasets_comparison_ols.png   (1 × 3, predictions only, for paper)
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

HERE        = Path(__file__).parent
PARAMS_FILE = HERE / "Parameters_hierarchical_models.csv"
WASH_FILE   = HERE / "Ex-16-3.csv"
MAINE_FILE  = HERE / "rural_int.xlsx"
QLD_FILE    = HERE / "Stage5A_1848_All_Initial_Columns.xlsx"

C_OBS  = "#2e86ab"   # observed — steel blue
C_HIER = "#b22222"   # hierarchical — firebrick
C_COMP = "#e07b00"   # benchmark / metacount — burnt orange
ALPHA_PT = 0.45
ALPHA_BAND = 0.15
LW = 1.8

matplotlib.rc("font", family="serif", size=10)
matplotlib.rc("axes", titlesize=11)


# ── Parameter helpers ────────────────────────────────────────────────────────

def load_params() -> pd.DataFrame:
    df = pd.read_csv(PARAMS_FILE, sep=";")
    df.columns = df.columns.str.strip()
    df["parameter"] = (df["parameter"].astype(str).str.strip()
                       .str.replace(",", "."))
    df["parameter"] = pd.to_numeric(df["parameter"], errors="coerce")
    return df


def _coefs(params, model, analysis):
    return params[
        (params["model"].str.strip() == model) &
        (params["analysis"].str.strip() == analysis)
    ].copy()


def _scalar(series):
    return float(series.dropna().values[0])


# ── OLS line + 95 % CI ───────────────────────────────────────────────────────

def ols_ci(x, y, x_grid, alpha=0.05):
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    n = len(xm)
    if n < 3:
        nan = np.full_like(x_grid, np.nan)
        return nan, nan, nan
    slope, intercept, *_ = stats.linregress(xm, ym)
    y_hat = slope * xm + intercept
    se   = np.sqrt(np.sum((ym - y_hat) ** 2) / max(n - 2, 1))
    xbar = xm.mean()
    ssx  = max(np.sum((xm - xbar) ** 2), 1e-12)
    tc   = stats.t.ppf(1 - alpha / 2, df=max(n - 2, 1))
    yg   = slope * x_grid + intercept
    seg  = se * np.sqrt(1 / max(n, 1) + (x_grid - xbar) ** 2 / ssx)
    return yg, yg - tc * seg, yg + tc * seg


def winsorise(arr, pct=97.5):
    pos = arr[np.isfinite(arr) & (arr > 0)]
    upper = np.nanpercentile(pos, pct) if len(pos) else 1e6
    return np.clip(np.where(np.isfinite(arr), arr, 0.0), 0.0, upper)


def _smart_ylim(obs, *preds, headroom=1.15):
    """y-axis upper bound = headroom × 99th-pct of obs and winsorised preds."""
    cands = [np.nanpercentile(obs[obs >= 0], 99)]
    for p in preds:
        pw = winsorise(p)
        cands.append(np.nanpercentile(pw[pw > 0], 99) if np.any(pw > 0) else 0)
    return max(cands) * headroom


# ── JAX / jaxopt NB2 fitting (metacountregressor-style) ─────────────────────

def fit_nb2_jaxopt(X: np.ndarray, y: np.ndarray,
                   offset: np.ndarray | None = None,
                   maxiter: int = 2000) -> tuple[np.ndarray, np.ndarray, float]:
    """
    NB2 MLE via JAX autodiff + jaxopt LBFGS.

    Mirrors metacountregressor's CountModel.fit() pattern:
      - JAX 64-bit, automatic gradient via reverse-mode AD
      - jaxopt LBFGS solver, tol=1e-8
      - Dispersion α parameterised as exp(log_alpha)

    Returns (fitted_mu, beta, alpha).
    """
    import jax
    import jax.numpy as jnp
    from jaxopt import LBFGS

    jax.config.update("jax_enable_x64", True)

    n, p = X.shape
    off  = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)

    X_j   = jnp.array(X,   dtype=jnp.float64)
    y_j   = jnp.array(y,   dtype=jnp.float64)
    off_j = jnp.array(off, dtype=jnp.float64)

    def nll(params):
        beta      = params[:p]
        log_alpha = jnp.clip(params[p], -6.0, 5.0)
        alpha     = jnp.exp(log_alpha)
        eta       = jnp.clip(X_j @ beta + off_j, -15.0, 15.0)
        mu        = jnp.exp(eta)
        inv_a     = 1.0 / alpha
        ll = (jax.scipy.special.gammaln(y_j + inv_a)
              - jax.scipy.special.gammaln(inv_a)
              - jax.scipy.special.gammaln(y_j + 1.0)
              + inv_a * jnp.log(inv_a / (inv_a + mu))
              + y_j   * jnp.log(mu / (mu + inv_a)))
        return -jnp.sum(ll)

    # Warm-start: intercept ≈ log(mean(y)) so we start near the MLE
    init_np = np.zeros(p + 1)
    mean_y  = np.mean(y[y > 0]) if np.any(y > 0) else 0.1
    init_np[0] = np.log(mean_y + 1e-6) - (np.mean(off) if offset is not None else 0.0)
    init_np[p] = 0.5   # log(alpha) starting point (~alpha=1.65, reasonable for NB2)
    init   = jnp.array(init_np, dtype=jnp.float64)
    solver = LBFGS(fun=nll, maxiter=maxiter, tol=1e-8, history_size=30)
    result = solver.run(init)

    beta  = np.array(result.params[:p])
    alpha = float(np.exp(np.clip(float(result.params[p]), -6.0, 5.0)))
    mu    = np.exp(X @ beta + off)
    print(f"    NB2 fit: LL={-float(result.state.value):.1f}  "
          f"alpha={alpha:.4f}  iters={int(result.state.iter_num)}")
    return mu, beta, alpha


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _pred_panel(ax, log_aadt, obs, preds,
                pred_label, title, ylim_top):
    """Scatter of observed + predicted, with OLS line through predicted."""
    pw = winsorise(preds)
    x_g = np.linspace(log_aadt.min() - 0.05, log_aadt.max() + 0.05, 300)

    # Observed
    ax.scatter(log_aadt, obs, s=8, c=C_OBS, alpha=ALPHA_PT,
               linewidths=0, zorder=2, label="Observed")
    # Predicted scatter
    ax.scatter(log_aadt, pw, s=8, c=C_HIER if "Hier" in pred_label else C_COMP,
               alpha=ALPHA_PT + 0.1, linewidths=0, zorder=3, label="Predicted")
    # OLS line through predictions
    col = C_HIER if "Hier" in pred_label else C_COMP
    yg, lo, hi = ols_ci(log_aadt, pw, x_g)
    ax.plot(x_g, yg, c=col, lw=LW, zorder=4)
    ax.fill_between(x_g, np.clip(lo, 0, None), hi,
                    color=col, alpha=ALPHA_BAND, zorder=1)

    ax.set_ylim(0, ylim_top)
    ax.set_xlim(x_g[[0, -1]])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("log(AADT)")
    ax.set_ylabel("Crashes")
    ax.legend(fontsize=8, loc="upper left", markerscale=1.5)
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def _resid_panel(ax, log_aadt, obs, preds, title):
    """Pearson residuals (obs - pred) / sqrt(pred) vs log(AADT) + OLS."""
    pw = winsorise(preds)
    safe = np.where(pw > 0, pw, np.nanmedian(pw[pw > 0]))
    resid = (obs - pw) / np.sqrt(safe)
    resid = np.clip(resid, -10, 10)

    col = C_HIER if "Hier" in title else C_COMP
    x_g = np.linspace(log_aadt.min() - 0.05, log_aadt.max() + 0.05, 300)

    ax.scatter(log_aadt, resid, s=7, c=col, alpha=0.35,
               linewidths=0, zorder=2)
    ax.axhline(0, color="grey", lw=1.0, ls="--", zorder=1)

    yg, lo, hi = ols_ci(log_aadt, resid, x_g)
    ax.plot(x_g, yg, c=col, lw=LW, zorder=4)
    ax.fill_between(x_g, lo, hi, color=col, alpha=ALPHA_BAND, zorder=1)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("log(AADT)")
    ax.set_ylabel("(obs - pred) / sqrt(pred)")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def make_2x2(log_aadt, obs, hier, comp,
             hier_title, comp_title, suptitle, out_path):
    """Full 2×2 figure: predictions + residuals for two models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    ylim = _smart_ylim(obs, hier, comp)
    _pred_panel(axes[0, 0], log_aadt, obs, hier,
                "Hier", hier_title, ylim)
    _pred_panel(axes[0, 1], log_aadt, obs, comp,
                "Comp", comp_title, ylim)
    _resid_panel(axes[1, 0], log_aadt, obs, hier, "Hierarchical residuals")
    _resid_panel(axes[1, 1], log_aadt, obs, comp, f"{comp_title} residuals")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {Path(out_path).name}")


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {Path(path).name}")


# ── Hierarchical prediction formulas ────────────────────────────────────────

def _hier_washington(df, params):
    h = _coefs(params, "hierarchical", "washington")
    A_feat = ["const", "WIDTH_PER_LANE", "CURVES", "TANGENT",
              "INTECHAG", "MXGRDIFF", "SHOULDER_WIDTH"]
    A  = h[~h["feature"].isin(["b0", "SLOPE"])]["parameter"].values
    b0 = _scalar(h.loc[h["feature"] == "b0", "parameter"])
    sl = _scalar(h.loc[h["feature"] == "SLOPE", "parameter"])
    hier = (np.exp(df[A_feat].astype(float).values @ A) *
            np.power(df["AADTmaj"].values,
                     b0 * np.exp(df["SLOPE_FLAT"].values * sl)))
    return hier


def _hier_maine(df, params):
    h = _coefs(params, "hierarchical", "Maine")
    A_feat = ["const", "speed", "right_shoulder_width", "DP01", "DX32"]
    A   = h[~h["feature"].isin(["b0", "dummy_winter"])]["parameter"].values
    b0  = _scalar(h.loc[h["feature"] == "b0", "parameter"])
    win = _scalar(h.loc[h["feature"] == "dummy_winter", "parameter"])
    hier = (np.exp(df[A_feat].astype(float).values @ A) *
            np.power(df["AADTmaj"].values,
                     b0 * np.exp(df["dummy_winter"].values * win)))
    return hier


def _hier_queensland(df, params):
    h = _coefs(params, "hierarchical", "QLD HV")
    A_feat = ["const", "Total_width", "RS_HS", "LNMCV"]
    A  = h[~h["feature"].isin(["b0", "RS"])]["parameter"].values
    b0 = _scalar(h.loc[h["feature"] == "b0", "parameter"])
    rs = _scalar(h.loc[h["feature"] == "RS", "parameter"])
    hier = (np.exp(df[A_feat].astype(float).values @ A) *
            np.power(df["AADTmaj"].values,
                     b0 * np.exp(df["RS"].values * rs)))
    return hier


# ═══════════════════════════════════════════════════════════════════════════
# WASHINGTON
# ═══════════════════════════════════════════════════════════════════════════

def plot_washington(params):
    print("\n-- Washington")
    df = pd.read_csv(WASH_FILE)
    df["WIDTH_PER_LANE"] = df["WIDTH"] / (df["INCLANES"] + df["DECLANES"])
    df["SLOPE_FLAT"]     = (df["SLOPE"] == 0).astype(float)
    df["AADTmaj"]        = np.log(df["AADT"].astype(float))
    df["const"]          = 1.0
    df["SHOULDER_WIDTH"] = df["MIMEDSH"] / df["WIDTH"]
    for c in ["TANGENT", "INTECHAG", "MXGRDIFF", "LOWPRE", "GRADEBR",
              "FRICTION", "EXPOSE", "INTPM", "CURVES", "HISNOW"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Hierarchical
    hier = _hier_washington(df, params)

    # Literature (from parameters CSV — externally estimated coefficients)
    l = _coefs(params, "literature", "washington")
    L_feat = ["const", "LOWPRE", "GRADEBR", "FRICTION",
              "EXPOSE", "INTPM", "CURVES", "HISNOW"]
    lit = np.exp(df[L_feat].astype(float).values @ l["parameter"].values)

    obs        = df["FREQ"].values.astype(float)
    log_aadt   = df["AADTmaj"].values

    make_2x2(log_aadt, obs, hier, lit,
             "Hierarchical CMF model",
             "Literature SPF",
             "Washington: Hierarchical vs Literature CMF",
             HERE / "washington_comparison_ols.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAINE  — benchmark fitted fresh with metacountregressor JAX NB2
# ═══════════════════════════════════════════════════════════════════════════

def plot_maine(params):
    print("\n-- Maine")
    df = pd.read_excel(MAINE_FILE, sheet_name="rural_int")
    df["const"]   = 1.0
    df["AADTmaj"] = np.log(df["monthly_AADT"].astype(float))
    for c in ["speed", "right_shoulder_width", "DP01", "DX32",
              "dummy_winter", "segment_length", "curve", "crashes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Hierarchical (from parameters CSV)
    hier = _hier_maine(df, params)

    # Metacountregressor NB2 benchmark
    # Variables matching the literature specification in the R script
    bm_feat = ["const", "AADTmaj", "segment_length", "speed",
               "right_shoulder_width", "curve", "DP01", "DX32"]
    X_bm = df[bm_feat].astype(float).values
    y_bm = df["crashes"].values.astype(float)
    print("  Fitting Maine NB2 (JAX + jaxopt LBFGS)...")
    meta_mu, _, _ = fit_nb2_jaxopt(X_bm, y_bm)

    obs      = df["crashes"].values.astype(float)
    log_aadt = df["AADTmaj"].values

    make_2x2(log_aadt, obs, hier, meta_mu,
             "Hierarchical CMF model",
             "Metacount NB2",
             "Maine: Hierarchical vs Metacount NB2 Benchmark",
             HERE / "maine_comparison_ols.png")


# ═══════════════════════════════════════════════════════════════════════════
# QUEENSLAND  — analyst NB2 fitted fresh with metacountregressor JAX NB2
# ═══════════════════════════════════════════════════════════════════════════

def plot_queensland(params):
    print("\n-- Queensland")
    df = pd.read_excel(QLD_FILE, sheet_name="Stage5A_1848")
    df["const"]       = 1.0
    df["AADTmaj"]     = np.log(df["AADT"].astype(float))
    df["Total_width"] = df["Lwidth"].astype(float) * df["Nlanes"].astype(float)
    df["RS_HS"]       = df["RS"].astype(float)  * df["HSP"].astype(float)
    df["RS"]          = df["RS"].astype(float)
    df["LNMCV"]       = pd.to_numeric(df["LNMCV"], errors="coerce").fillna(0)

    for c in ["SP", "Curve", "Nlanes", "Median", "US", "RD", "UD"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["MSP"] = pd.to_numeric(df["MSP"], errors="coerce").fillna(0)
    df["RS_MSP"] = df["RS"] * df["MSP"]
    df["US_MSP"] = df["US"] * df["MSP"]
    df["RD_HS"]  = df["RD"] * df["HSP"].astype(float)
    df["RD_MSP"] = df["RD"] * df["MSP"]

    # Hierarchical (from parameters CSV)
    hier = _hier_queensland(df, params)

    # Metacountregressor NB2 analyst model
    # Variables matching R's MASS::glm.nb specification
    bm_vars = ["SP", "Curve", "Nlanes", "Total_width", "Median",
               "LNMCV", "RS_HS", "RS_MSP", "US_MSP", "RD_HS", "RD_MSP", "US", "RS"]
    avail   = ["const"] + [v for v in bm_vars if v in df.columns]
    X_bm    = df[avail].astype(float).values
    y_bm    = df["Headon"].astype(float).values
    offset  = df["AADTmaj"].values           # log-AADT as exposure offset

    print("  Fitting Queensland NB2 (JAX + jaxopt LBFGS)...")
    meta_mu, _, _ = fit_nb2_jaxopt(X_bm, y_bm, offset=offset)

    obs      = y_bm
    log_aadt = df["AADTmaj"].values

    make_2x2(log_aadt, obs, hier, meta_mu,
             "Hierarchical CMF model",
             "Metacount NB2 Analyst",
             "Queensland (HV): Hierarchical vs Metacount NB2",
             HERE / "queensland_comparison_ols.png")


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED — 1 × 3, predictions only (paper figure)
# ═══════════════════════════════════════════════════════════════════════════

def plot_combined(params):
    print("\n-- Combined (paper figure)")

    # Pre-compute all predictions (reuse logic above; skip refitting NB2)
    # ── Washington ──
    df_w = pd.read_csv(WASH_FILE)
    df_w["WIDTH_PER_LANE"] = df_w["WIDTH"] / (df_w["INCLANES"] + df_w["DECLANES"])
    df_w["SLOPE_FLAT"]     = (df_w["SLOPE"] == 0).astype(float)
    df_w["AADTmaj"]        = np.log(df_w["AADT"].astype(float))
    df_w["const"]          = 1.0
    df_w["SHOULDER_WIDTH"] = df_w["MIMEDSH"] / df_w["WIDTH"]
    for c in ["TANGENT","INTECHAG","MXGRDIFF","LOWPRE","GRADEBR",
              "FRICTION","EXPOSE","INTPM","CURVES","HISNOW"]:
        df_w[c] = pd.to_numeric(df_w[c], errors="coerce").fillna(0)
    hier_w  = _hier_washington(df_w, params)
    l_w     = _coefs(params, "literature", "washington")
    comp_w  = np.exp(
        df_w[["const","LOWPRE","GRADEBR","FRICTION","EXPOSE","INTPM","CURVES","HISNOW"]]
        .astype(float).values @ l_w["parameter"].values
    )
    obs_w   = df_w["FREQ"].values.astype(float)
    la_w    = df_w["AADTmaj"].values

    # ── Maine ──
    df_m = pd.read_excel(MAINE_FILE, sheet_name="rural_int")
    df_m["const"]   = 1.0
    df_m["AADTmaj"] = np.log(df_m["monthly_AADT"].astype(float))
    for c in ["speed","right_shoulder_width","DP01","DX32","dummy_winter",
              "segment_length","curve","crashes"]:
        df_m[c] = pd.to_numeric(df_m[c], errors="coerce").fillna(0)
    hier_m  = _hier_maine(df_m, params)
    bm_m    = ["const","AADTmaj","segment_length","speed",
               "right_shoulder_width","curve","DP01","DX32"]
    print("  Fitting Maine NB2 for combined figure...")
    comp_m, *_ = fit_nb2_jaxopt(
        df_m[bm_m].astype(float).values, df_m["crashes"].values.astype(float)
    )
    obs_m  = df_m["crashes"].values.astype(float)
    la_m   = df_m["AADTmaj"].values

    # ── Queensland ──
    df_q = pd.read_excel(QLD_FILE, sheet_name="Stage5A_1848")
    df_q["const"]       = 1.0
    df_q["AADTmaj"]     = np.log(df_q["AADT"].astype(float))
    df_q["Total_width"] = df_q["Lwidth"].astype(float) * df_q["Nlanes"].astype(float)
    df_q["RS_HS"]       = df_q["RS"].astype(float) * df_q["HSP"].astype(float)
    df_q["RS"]          = df_q["RS"].astype(float)
    df_q["LNMCV"]       = pd.to_numeric(df_q["LNMCV"], errors="coerce").fillna(0)
    for c in ["SP","Curve","Nlanes","Median","US","RD","UD"]:
        if c in df_q.columns:
            df_q[c] = pd.to_numeric(df_q[c], errors="coerce").fillna(0)
    df_q["MSP"]   = pd.to_numeric(df_q["MSP"], errors="coerce").fillna(0)
    df_q["RS_MSP"]= df_q["RS"] * df_q["MSP"]
    df_q["US_MSP"]= df_q["US"] * df_q["MSP"]
    df_q["RD_HS"] = df_q["RD"] * df_q["HSP"].astype(float)
    df_q["RD_MSP"]= df_q["RD"] * df_q["MSP"]
    hier_q = _hier_queensland(df_q, params)
    bm_q_vars = ["SP","Curve","Nlanes","Total_width","Median",
                 "LNMCV","RS_HS","RS_MSP","US_MSP","RD_HS","RD_MSP","US","RS"]
    avail_q = ["const"] + [v for v in bm_q_vars if v in df_q.columns]
    print("  Fitting Queensland NB2 for combined figure...")
    comp_q, *_ = fit_nb2_jaxopt(
        df_q[avail_q].astype(float).values,
        df_q["Headon"].astype(float).values,
        offset=df_q["AADTmaj"].values
    )
    obs_q = df_q["Headon"].astype(float).values
    la_q  = df_q["AADTmaj"].values

    # ── Draw combined 2×3: top=predictions, bottom=residuals ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "CMF Model Comparison: Hierarchical vs Benchmark\n"
        "OLS linear trend (line) ± 95% CI band  |  Pearson residuals below",
        fontsize=13, fontweight="bold", y=1.01,
    )

    datasets = [
        (la_w, obs_w, hier_w, comp_w,  "Hierarchical", "Literature",   "Washington"),
        (la_m, obs_m, hier_m, comp_m,  "Hierarchical", "Metacount NB2","Maine"),
        (la_q, obs_q, hier_q, comp_q,  "Hierarchical", "Metacount NB2","Queensland (HV)"),
    ]

    for col_idx, (la, obs, hier, comp, hl, cl, ds) in enumerate(datasets):
        ylim = _smart_ylim(obs, hier, comp)

        # Predictions: one combined panel per dataset showing both models
        ax_top = axes[0, col_idx]
        xg = np.linspace(la.min() - 0.05, la.max() + 0.05, 300)
        ax_top.scatter(la, obs, s=7, c=C_OBS, alpha=0.4,
                       linewidths=0, zorder=1, label="Observed")
        for preds, col, lbl in [(winsorise(hier), C_HIER, hl),
                                 (winsorise(comp), C_COMP, cl)]:
            ax_top.scatter(la, preds, s=6, c=col, alpha=0.35,
                           linewidths=0, zorder=2)
            yg, lo, hi = ols_ci(la, preds, xg)
            ax_top.plot(xg, yg, c=col, lw=LW, zorder=3, label=lbl)
            ax_top.fill_between(xg, np.clip(lo, 0, None), hi,
                                color=col, alpha=ALPHA_BAND, zorder=0)
        ax_top.set_ylim(0, ylim)
        ax_top.set_xlim(xg[[0, -1]])
        ax_top.set_title(ds, fontweight="bold")
        ax_top.set_xlabel("log(AADT)")
        ax_top.set_ylabel("Crashes")
        ax_top.legend(fontsize=8, loc="upper left", markerscale=1.8)
        ax_top.grid(True, alpha=0.2, lw=0.5)
        ax_top.spines[["top", "right"]].set_visible(False)

        # Residuals
        ax_bot = axes[1, col_idx]
        ax_bot.axhline(0, color="grey", lw=1.0, ls="--", zorder=1)
        for preds, col, lbl in [(hier, C_HIER, hl), (comp, C_COMP, cl)]:
            pw   = winsorise(preds)
            safe = np.where(pw > 0, pw, np.nanmedian(pw[pw > 0]))
            res  = np.clip((obs - pw) / np.sqrt(safe), -10, 10)
            ax_bot.scatter(la, res, s=6, c=col, alpha=0.30, linewidths=0, zorder=2)
            yg, lo, hi = ols_ci(la, res, xg)
            ax_bot.plot(xg, yg, c=col, lw=LW, zorder=3, label=lbl)
            ax_bot.fill_between(xg, lo, hi, color=col, alpha=ALPHA_BAND)
        ax_bot.set_title(f"{ds} — Pearson residuals", fontweight="bold")
        ax_bot.set_xlabel("log(AADT)")
        ax_bot.set_ylabel("(obs - pred) / sqrt(pred)")
        ax_bot.legend(fontsize=8, loc="upper right")
        ax_bot.grid(True, alpha=0.2, lw=0.5)
        ax_bot.spines[["top", "right"]].set_visible(False)

    save_fig(fig, HERE / "all_datasets_comparison_ols.png")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    params = load_params()
    plot_washington(params)
    plot_maine(params)
    plot_queensland(params)
    plot_combined(params)
    print("\nDone — four PNGs written to", HERE)
