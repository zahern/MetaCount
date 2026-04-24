"""
Latent-Class Model Comparison Experiment
==========================================
Fits the same synthetic 2-class dataset under every meaningful
structural variant and prints a ranked comparison table.

Variants tested
---------------
  Single-class baselines
    1.  Poisson (no dispersion)
    2.  NB2

  Two-class  —  no membership covariates
    3.  LC-2  Poisson
    4.  LC-2  NB2

  Two-class  —  WITH membership covariates (z1, z2)
    5.  LC-2  Poisson  + membership
    6.  LC-2  NB2      + membership  ← matches the true DGP

  Two-class  —  random effect on x1 (individual heterogeneity)
    7.  LC-2  Poisson  + random x1
    8.  LC-2  NB2      + random x1

  Three-class  —  over-specified
    9.  LC-3  NB2
   10.  LC-3  NB2      + membership

Run from the metacountregressor/ directory:
    python experiment_lc_model_comparison.py
"""

from __future__ import annotations

import sys, os, warnings, traceback
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# TRUE DGP  (same as test_latent_class_synthetic.py)
# ─────────────────────────────────────────────────────────────────────────────

TRUE = {
    "class1": {
        "intercept": -3.0,
        "x1": +1.0, "x2": -0.8, "x3": +0.3, "x4": +0.1,
        "alpha": 1.5,
    },
    "class2": {
        "intercept": -0.8,
        "x1": -0.4, "x2": +0.6, "x3": +0.2, "x4": +0.9,
        "alpha": 0.6,
    },
    "membership": {"g0": 0.0, "gz1": +1.5, "gz2": -1.0},
}

OUTCOME_VARS    = ["x1", "x2", "x3", "x4"]
MEMBERSHIP_VARS = ["z1", "z2"]


def generate_data(N: int = 800, T: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mem = TRUE["membership"]
    rows = []
    for i in range(N):
        x1 = rng.normal(0, 1); x2 = rng.normal(0, 1)
        x3 = rng.normal(0, 1); x4 = rng.normal(0, 1)
        z1 = rng.normal(0, 1); z2 = rng.normal(0, 1)
        log_odds = mem["g0"] + mem["gz1"] * z1 + mem["gz2"] * z2
        prob_c2  = 1.0 / (1.0 + np.exp(-log_odds))
        c = rng.choice([1, 2], p=[1 - prob_c2, prob_c2])
        p = TRUE[f"class{c}"]
        for t in range(T):
            eta   = (p["intercept"] + p["x1"]*x1 + p["x2"]*x2
                     + p["x3"]*x3 + p["x4"]*x4)
            mu    = np.exp(np.clip(eta, -20, 20))
            alpha = p["alpha"]
            p_nb  = alpha / (alpha + mu)
            y     = rng.negative_binomial(alpha, p_nb)
            rows.append({"id": i, "t": t,
                         "x1": x1, "x2": x2, "x3": x3, "x4": x4,
                         "z1": z1, "z2": z2,
                         "y": int(y), "true_class": c})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    # (label, model_family, latent_classes, membership, random_ind, n_classes_note)
    ("1-class  Poisson",                "poisson", 1, False, False),
    ("1-class  NB2",                    "nb",      1, False, False),
    ("LC-2  Poisson",                   "poisson", 2, False, False),
    ("LC-2  NB2",                       "nb",      2, False, False),
    ("LC-2  Poisson + membership",      "poisson", 2, True,  False),
    ("LC-2  NB2    + membership  [DGP]","nb",      2, True,  False),
    ("LC-2  Poisson + random x1",       "poisson", 2, False, True),
    ("LC-2  NB2    + random x1",        "nb",      2, False, True),
    ("LC-3  NB2",                       "nb",      3, False, False),
    ("LC-3  NB2    + membership",       "nb",      3, True,  False),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _class_assignment_accuracy(builder, fit, df, C):
    """Posterior class assignment accuracy vs true_class column."""
    try:
        lc_df     = builder.compute_latent_class_probabilities(fit)
        prob_cols = [f"class_{c+1}_prob" for c in range(C)]
        lc_df["pred"] = (
            lc_df[prob_cols].idxmax(axis=1)
            .str.extract(r"(\d+)").astype(int)
        )
        true_per_id = df.groupby("id")["true_class"].first().reset_index()
        m    = lc_df[["id", "pred"]].merge(true_per_id, on="id")
        acc  = (m["pred"] == m["true_class"]).mean()
        acc2 = (m["pred"] != m["true_class"]).mean()
        return max(acc, acc2)
    except Exception:
        return float("nan")


def _recovery_summary(fit, C):
    """
    Return a short string showing recovered x1/x2 per class — the two
    variables with opposing signs across classes in the true DGP.
    """
    try:
        from main_hpc_lc_patch import build_base_index
        from dataclasses import replace

        spec      = fit["spec"]
        params_np = np.array(fit["result"].params)
        base_spec = replace(spec, latent_classes=1)
        K_base    = build_base_index(base_spec)["total_params"]
        theta_all = params_np[:C * K_base].reshape(C, K_base)
        names     = list(spec.fixed_names)          # __INTERCEPT__, x1, x2 …

        idx_x1 = names.index("x1") if "x1" in names else None
        idx_x2 = names.index("x2") if "x2" in names else None

        parts = []
        for c in range(C):
            seg = f"C{c+1}["
            if idx_x1 is not None:
                seg += f"x1={theta_all[c, idx_x1]:+.2f}"
            if idx_x2 is not None:
                seg += f" x2={theta_all[c, idx_x2]:+.2f}"
            seg += "]"
            parts.append(seg)
        return "  ".join(parts)
    except Exception:
        return "—"


def _membership_summary(fit, C):
    """Short string of gamma (membership) coefficients for z1/z2."""
    try:
        from main_hpc_lc_patch import build_base_index
        from dataclasses import replace

        spec      = fit["spec"]
        if spec.K_membership == 0:
            return "—"
        params_np = np.array(fit["result"].params)
        base_spec = replace(spec, latent_classes=1)
        K_base    = build_base_index(base_spec)["total_params"]
        gamma     = params_np[C * K_base:].reshape(C - 1, spec.K_membership + 1)
        mem_names = ["intercept"] + list(spec.membership_names)
        parts = []
        for k, nm in enumerate(mem_names):
            if nm in ("z1", "z2", "intercept"):
                parts.append(f"{nm}={gamma[0, k]:+.2f}")
        return "  ".join(parts)
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(N: int = 800, T: int = 3, seed: int = 42):
    print("=" * 80)
    print("  LATENT CLASS MODEL COMPARISON  —  functional variants")
    print("=" * 80)

    df = generate_data(N=N, T=T, seed=seed)
    print(f"\n  Dataset: {df['id'].nunique()} individuals x {T} periods  "
          f"y mean={df['y'].mean():.2f}  zeros={100*(df['y']==0).mean():.1f}%")
    true_shares = df.groupby("true_class")["id"].nunique()
    print(f"  True class shares: " +
          "  ".join(f"class{c}={n/df['id'].nunique():.2f}" for c, n in true_shares.items()))
    print()

    from experiment_package import ExperimentBuilder
    builder = ExperimentBuilder(df=df, id_col="id", y_col="y")

    results = []

    for label, model_fam, n_classes, use_mem, use_rdm in VARIANTS:
        print(f"  Fitting: {label} ...", end="", flush=True)

        dispersion = 1 if model_fam == "nb" else 0

        spec_kwargs = dict(
            fixed_terms  = OUTCOME_VARS,
            dispersion   = dispersion,
            latent_classes = n_classes,
        )
        if use_mem:
            spec_kwargs["membership_terms"] = MEMBERSHIP_VARS
        if use_rdm:
            spec_kwargs["rdm_terms"] = ["x1:normal"]

        try:
            manual_spec = builder.make_manual_spec(**spec_kwargs)
            fit = builder.fit_manual_model(
                manual_spec = manual_spec,
                model       = model_fam,
                print_report= False,
            )
            summary = fit.get("summary", {})
            ll   = summary.get("loglik",  float("nan"))
            bic  = summary.get("bic",     float("nan"))
            k    = summary.get("num_parm", "?")
            n_obs= summary.get("n_obs",    "?")

            acc  = _class_assignment_accuracy(builder, fit, df, n_classes) \
                   if n_classes > 1 else float("nan")
            rec  = _recovery_summary(fit, n_classes)
            mem  = _membership_summary(fit, n_classes)

            results.append({
                "label":   label,
                "C":       n_classes,
                "model":   model_fam,
                "mem":     use_mem,
                "rdm":     use_rdm,
                "LL":      ll,
                "BIC":     bic,
                "k":       k,
                "acc":     acc,
                "x1x2_recovery": rec,
                "gamma":   mem,
                "status":  "OK",
            })
            print(f"  BIC={bic:.1f}  LL={ll:.1f}  acc={acc:.1%}" if not np.isnan(acc)
                  else f"  BIC={bic:.1f}  LL={ll:.1f}")

        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append({
                "label": label, "C": n_classes, "model": model_fam,
                "mem": use_mem, "rdm": use_rdm,
                "LL": float("nan"), "BIC": float("nan"),
                "k": "?", "acc": float("nan"),
                "x1x2_recovery": "FAILED", "gamma": "—",
                "status": f"FAILED: {exc}",
            })

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  RESULTS  (ranked by BIC)")
    print("=" * 80)

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("BIC")
    best_bic = df_res["BIC"].min()

    hdr = (f"  {'Rank':<5}  {'Model Variant':<38}  {'C':>2}  "
           f"{'LL':>10}  {'k':>4}  {'BIC':>10}  {'dBIC':>7}  {'Acc':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for rank, (_, row) in enumerate(df_res.iterrows(), 1):
        dbic = row["BIC"] - best_bic if not np.isnan(row["BIC"]) else float("nan")
        acc_str = f"{row['acc']:.1%}" if not np.isnan(row["acc"]) else "  n/a"
        bic_str = f"{row['BIC']:>10.1f}" if not np.isnan(row["BIC"]) else f"{'FAILED':>10}"
        ll_str  = f"{row['LL']:>10.1f}"  if not np.isnan(row["LL"])  else f"{'—':>10}"
        dbic_str= f"{dbic:>+7.1f}"       if not np.isnan(dbic)        else f"{'—':>7}"
        marker  = "  << TRUE DGP" if "DGP" in row["label"] else ""
        print(f"  {rank:<5}  {row['label']:<38}  {row['C']:>2}  "
              f"{ll_str}  {row['k']:>4}  {bic_str}  {dbic_str}  {acc_str:>6}{marker}")

    # ── x1/x2 sign recovery per model ─────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  CLASS-SPECIFIC COEFFICIENT RECOVERY  (x1 / x2)")
    print(f"  True DGP: C1[x1=+1.00 x2=-0.80]  C2[x1=-0.40 x2=+0.60]")
    print("=" * 80)
    for _, row in df_res.iterrows():
        if row["C"] > 1:
            print(f"  {row['label']:<40}  {row['x1x2_recovery']}")

    # ── Membership gamma recovery ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  MEMBERSHIP EQUATION RECOVERY  (true: intercept=0.00  z1=+1.50  z2=-1.00)")
    print("=" * 80)
    for _, row in df_res.iterrows():
        if row["mem"]:
            print(f"  {row['label']:<40}  {row['gamma']}")

    print("\n  Done.\n")
    return df_res


if __name__ == "__main__":
    run_experiment()
