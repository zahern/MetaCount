"""
Synthetic Tobit model recovery test
=====================================
Generates a left-censored crash-rate dataset with known true parameters,
fits three models (1-class Tobit, 2-class LC-Tobit, 2-class LC-Tobit with
membership covariates) and reports parameter recovery.

True DGP (2-class, left-censored at 0)
---------------------------------------
Class 1  ("low crash-rate" sites):
    intercept = -1.0, x1 = +0.8, x2 = -0.5, x3 = +0.2,  sigma = 1.0
Class 2  ("high crash-rate" sites):
    intercept =  2.0, x1 = -0.3, x2 = +0.7, x3 = +0.1,  sigma = 1.5
Membership: log-odds(C2/C1) = 0.0 + 1.2*z1 - 0.9*z2

Run from the metacountregressor/ directory:
    python test_tobit_synthetic.py
"""

from __future__ import annotations
import sys, os, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# TRUE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TRUE = {
    "class1": {
        "label":     "low-rate",
        "intercept": -1.0,
        "x1": +0.8, "x2": -0.5, "x3": +0.2,
        "sigma": 1.0,
    },
    "class2": {
        "label":     "high-rate",
        "intercept":  2.0,
        "x1": -0.3, "x2": +0.7, "x3": +0.1,
        "sigma": 1.5,
    },
    "membership": {"g0": 0.0, "gz1": +1.2, "gz2": -0.9},
}

OUTCOME_VARS    = ["x1", "x2", "x3"]
MEMBERSHIP_VARS = ["z1", "z2"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_tobit_panel(N: int = 700, T: int = 3, seed: int = 0) -> pd.DataFrame:
    """
    Balanced panel with left-censored crash rates.
    Censoring fraction is ~30-40% (zeros in outcome).
    """
    rng = np.random.default_rng(seed)
    mem = TRUE["membership"]
    rows = []
    class_counts = {1: 0, 2: 0}

    for i in range(N):
        x1 = rng.normal(0, 1); x2 = rng.normal(0, 1); x3 = rng.normal(0, 1)
        z1 = rng.normal(0, 1); z2 = rng.normal(0, 1)

        log_odds = mem["g0"] + mem["gz1"] * z1 + mem["gz2"] * z2
        prob_c2  = 1.0 / (1.0 + np.exp(-log_odds))
        c = rng.choice([1, 2], p=[1 - prob_c2, prob_c2])
        class_counts[c] += 1

        p = TRUE[f"class{c}"]
        for t in range(T):
            eta    = p["intercept"] + p["x1"]*x1 + p["x2"]*x2 + p["x3"]*x3
            y_star = eta + rng.normal(0, p["sigma"])
            y_obs  = max(0.0, y_star)           # left-censor at 0

            rows.append({"id": i, "t": t,
                         "x1": x1, "x2": x2, "x3": x3,
                         "z1": z1, "z2": z2,
                         "y": float(y_obs), "true_class": c})

    df = pd.DataFrame(rows)
    n  = df["id"].nunique()
    print(f"  Generated: class1={class_counts[1]/n:.2f}  class2={class_counts[2]/n:.2f}  "
          f"(membership driven by z1, z2)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RECOVERY DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def _show_recovery(fit, label: str, C: int):
    from main_hpc_lc_patch import build_base_index
    from dataclasses import replace

    spec      = fit["spec"]
    params_np = np.array(fit["result"].params)
    base_spec = replace(spec, latent_classes=1)
    K_base    = build_base_index(base_spec)["total_params"]
    theta_all = params_np[:C * K_base].reshape(C, K_base)
    gamma_flat = params_np[C * K_base:]

    names = list(spec.fixed_names)   # __INTERCEPT__, x1, x2, x3

    # Match class labels by intercept proximity
    true_intc = [TRUE["class1"]["intercept"], TRUE["class2"]["intercept"]]
    if C == 2:
        e0, e1 = float(theta_all[0, 0]), float(theta_all[1, 0])
        swap = (abs(e0 - true_intc[1]) + abs(e1 - true_intc[0]) <
                abs(e0 - true_intc[0]) + abs(e1 - true_intc[1]))
        mapping = [1, 0] if swap else [0, 1]
    else:
        mapping = [0]

    # sigma is last param in each class block
    sigma_idx = K_base - 1

    print(f"\n  {label}")
    col = 20
    hdr = f"    {'Variable':<{col}}"
    for tc in range(1, C + 1):
        hdr += f"  {'True C'+str(tc):>10}  {'Est C'+str(tc):>10}  {'Diff':>7}"
    print(hdr)
    print("    " + "-" * (col + C * 31))

    for k, name in enumerate(names):
        lbl = name.replace("__INTERCEPT__", "intercept")
        row = f"    {lbl:<{col}}"
        for tc_idx in range(C):
            tc   = tc_idx + 1
            ec   = mapping[tc_idx]
            tv   = TRUE.get(f"class{tc}", {}).get(lbl, None)
            ev   = float(theta_all[ec, k])
            if tv is None:
                row += f"  {'n/a':>10}  {ev:>+10.4f}  {'':>7}"
            else:
                row += f"  {tv:>+10.4f}  {ev:>+10.4f}  {ev-tv:>+7.3f}"
        print(row)

    # sigma
    row = f"    {'sigma':<{col}}"
    for tc_idx in range(C):
        tc  = tc_idx + 1
        ec  = mapping[tc_idx]
        tv  = TRUE[f"class{tc}"]["sigma"]
        raw = float(theta_all[ec, sigma_idx])
        import jax
        ev  = float(jax.nn.softplus(raw))
        row += f"  {tv:>+10.4f}  {ev:>+10.4f}  {ev-tv:>+7.3f}"
    print(row)

    if C == 2 and spec.K_membership > 0:
        gamma = gamma_flat.reshape(C - 1, spec.K_membership + 1)[0]
        mem   = TRUE["membership"]
        true_g = [mem["g0"], mem["gz1"], mem["gz2"]]
        print(f"\n    Membership  (true: g0={mem['g0']:.2f}  "
              f"gz1={mem['gz1']:.2f}  gz2={mem['gz2']:.2f})")
        for k, (nm, tv) in enumerate(zip(["intercept", "z1", "z2"], true_g)):
            ev = float(gamma[k])
            print(f"      {nm:<12}  true={tv:+.2f}  est={ev:+.4f}  diff={ev-tv:+.3f}")

    summary = fit.get("summary", {})
    if isinstance(summary, dict):
        ll  = summary.get("loglik", float("nan"))
        bic = summary.get("bic",    float("nan"))
        k   = summary.get("num_parm", "?")
        print(f"\n    LL={ll:.2f}   k={k}   BIC={bic:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_test():
    print("=" * 70)
    print("  TOBIT MODEL  —  SYNTHETIC RECOVERY TEST")
    print("=" * 70)
    print()

    df = generate_tobit_panel(N=700, T=3, seed=0)
    print(f"  Dataset : {df['id'].nunique()} sites x {df['t'].nunique()} periods")
    print(f"  y       : mean={df['y'].mean():.2f}  "
          f"zeros={100*(df['y']==0).mean():.1f}%  "
          f"max={df['y'].max():.2f}")

    from experiment_package import ExperimentBuilder
    builder = ExperimentBuilder(df=df, id_col="id", y_col="y")

    results = {}

    # ── Model A: 1-class Tobit ────────────────────────────────────────
    print("\n  Fitting Model A: 1-class Tobit ...")
    spec_a = builder.make_manual_spec(
        fixed_terms=OUTCOME_VARS, dispersion=0, latent_classes=1)
    results["A"] = builder.fit_manual_model(
        spec_a, model="tobit", print_report=True)

    # ── Model B: 2-class LC-Tobit (no membership) ─────────────────────
    print("\n  Fitting Model B: 2-class LC-Tobit ...")
    spec_b = builder.make_manual_spec(
        fixed_terms=OUTCOME_VARS, dispersion=0, latent_classes=2)
    results["B"] = builder.fit_manual_model(
        spec_b, model="tobit", print_report=True)

    # ── Model C: 2-class LC-Tobit + membership ────────────────────────
    print("\n  Fitting Model C: 2-class LC-Tobit + membership (z1, z2) ...")
    spec_c = builder.make_manual_spec(
        fixed_terms=OUTCOME_VARS, membership_terms=MEMBERSHIP_VARS,
        dispersion=0, latent_classes=2)
    results["C"] = builder.fit_manual_model(
        spec_c, model="tobit", print_report=True)

    # ── Recovery tables ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PARAMETER RECOVERY")
    print("=" * 70)

    _show_recovery(results["A"], "Model A — 1-class Tobit", C=1)
    _show_recovery(results["B"], "Model B — LC-2 Tobit (no membership)", C=2)
    _show_recovery(results["C"], "Model C — LC-2 Tobit + membership  [TRUE DGP]", C=2)

    # ── Model comparison ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON  (lower BIC = better)")
    print("=" * 70)
    rows = []
    for lbl, fit in results.items():
        s   = fit.get("summary", {})
        rows.append({"Model": lbl,
                     "Description": {"A": "1-class Tobit",
                                     "B": "LC-2 Tobit",
                                     "C": "LC-2 Tobit + membership"}[lbl],
                     "LL":  s.get("loglik", float("nan")),
                     "k":   s.get("num_parm", "?"),
                     "BIC": s.get("bic",    float("nan"))})
    cmp = pd.DataFrame(rows).sort_values("BIC")
    best = cmp["BIC"].min()
    cmp["dBIC"] = cmp["BIC"] - best
    print(cmp[["Model","Description","LL","k","BIC","dBIC"]].to_string(index=False))

    # ── Class assignment accuracy for Model C ────────────────────────
    try:
        lc_df = builder.compute_latent_class_probabilities(results["C"])
        prob_cols = ["class_1_prob", "class_2_prob"]
        lc_df["pred"] = lc_df[prob_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
        true_id = df.groupby("id")["true_class"].first().reset_index()
        m   = lc_df[["id","pred"]].merge(true_id, on="id")
        acc = max((m["pred"] == m["true_class"]).mean(),
                  (m["pred"] != m["true_class"]).mean())
        print(f"\n  Class assignment accuracy (Model C): {acc:.1%}")
    except Exception as exc:
        print(f"\n  [class probs] {exc}")

    print("\n  Done.\n")
    return results


if __name__ == "__main__":
    run_test()
