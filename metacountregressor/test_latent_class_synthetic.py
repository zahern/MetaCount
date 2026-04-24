"""
Synthetic latent-class recovery test  (rich specification)
============================================================
Two-class NB2 panel dataset with:
  - 4 outcome covariates (x1..x4) whose effects differ markedly by class
  - 2 membership covariates (z1, z2) that drive class assignment
  - Clear sign reversals between classes on x1 and x2 to make
    the two sub-populations identifiably distinct

True DGP summary
----------------
Class 1 ("low-severity"):  intercept=-3.0  x1=+1.0  x2=-0.8  x3=+0.3  x4=+0.1  alpha=1.5
Class 2 ("high-severity"): intercept=-0.8  x1=-0.4  x2=+0.6  x3=+0.2  x4=+0.9  alpha=0.6
Membership logit (class 2 vs 1):  g0=0.0  gz1=+1.5  gz2=-1.0
  -> z1 high  => more likely class 2
  -> z2 high  => more likely class 1
Class shares driven by membership: roughly 55 / 45 in expectation.

Run from the metacountregressor/ directory:
    python test_latent_class_synthetic.py
"""

import numpy as np
import pandas as pd
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ─────────────────────────────────────────────────────────────────────────────
# TRUE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

TRUE = {
    "class1": {
        "label":     "low-severity",
        "intercept": -3.0,
        "x1":        +1.0,   # strong positive  (sign-flips vs class 2)
        "x2":        -0.8,   # strong negative  (sign-flips vs class 2)
        "x3":        +0.3,   # moderate, similar across classes
        "x4":        +0.1,   # near-zero        (clear contrast with class 2)
        "alpha":      1.5,   # NB dispersion
    },
    "class2": {
        "label":     "high-severity",
        "intercept": -0.8,
        "x1":        -0.4,   # negative         (sign-flip vs class 1)
        "x2":        +0.6,   # positive         (sign-flip vs class 1)
        "x3":        +0.2,   # moderate, similar
        "x4":        +0.9,   # large            (clear contrast with class 1)
        "alpha":      0.6,
    },
    # Membership logit: log-odds(class 2 / class 1) = g0 + g_z1*z1 + g_z2*z2
    "membership": {
        "g0":  0.0,
        "gz1": +1.5,   # z1 increases prob of class 2
        "gz2": -1.0,   # z2 decreases prob of class 2
    },
}

OUTCOME_VARS   = ["x1", "x2", "x3", "x4"]
MEMBERSHIP_VARS = ["z1", "z2"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_latent_class_panel(
    N: int = 800,
    T: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a balanced panel for a 2-class NB model with membership covariates.

    Each individual i has:
      - outcome covariates x1..x4 (fixed across t, drawn once per individual)
      - membership covariates z1, z2 (fixed across t)
      - class assignment drawn from Bernoulli(softmax(g0 + gz1*z1 + gz2*z2))
      - counts y drawn from NB2 with class-specific parameters

    Returns a DataFrame with columns:
        id, t, x1, x2, x3, x4, z1, z2, y, true_class
    """
    rng = np.random.default_rng(seed)
    mem = TRUE["membership"]

    rows = []
    true_classes = []

    for i in range(N):
        # Draw individual-level covariates (constant across t)
        x1 = rng.normal(0.0, 1.0)
        x2 = rng.normal(0.0, 1.0)
        x3 = rng.normal(0.0, 1.0)
        x4 = rng.normal(0.0, 1.0)

        z1 = rng.normal(0.0, 1.0)   # membership covariate 1
        z2 = rng.normal(0.0, 1.0)   # membership covariate 2

        # Class assignment probability via logistic
        log_odds_c2 = mem["g0"] + mem["gz1"] * z1 + mem["gz2"] * z2
        prob_c2 = 1.0 / (1.0 + np.exp(-log_odds_c2))
        c = rng.choice([1, 2], p=[1 - prob_c2, prob_c2])
        true_classes.append(c)

        p = TRUE[f"class{c}"]
        for t in range(T):
            eta = (
                p["intercept"]
                + p["x1"] * x1
                + p["x2"] * x2
                + p["x3"] * x3
                + p["x4"] * x4
            )
            mu    = np.exp(np.clip(eta, -20, 20))
            alpha = p["alpha"]
            p_nb  = alpha / (alpha + mu)
            y     = rng.negative_binomial(alpha, p_nb)

            rows.append({
                "id": i, "t": t,
                "x1": x1, "x2": x2, "x3": x3, "x4": x4,
                "z1": z1, "z2": z2,
                "y":  int(y),
                "true_class": c,
            })

    df = pd.DataFrame(rows)
    n1 = sum(1 for c in true_classes if c == 1)
    n2 = N - n1
    print(f"  Generated class shares: class1={n1/N:.3f}  class2={n2/N:.3f}  "
          f"(driven by membership z1, z2)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RECOVERY TABLE PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _print_recovery_table(
    est_params: np.ndarray,
    fixed_names: list,
    K_base: int,
    C: int,
    gamma_flat: np.ndarray,
    membership_names: tuple,
):
    """
    Side-by-side table: true vs estimated for each class.
    Labels are matched by comparing intercept sign to the true classes.
    """
    theta_all = est_params[:C * K_base].reshape(C, K_base)

    # ------------------------------------------------------------------
    # Match estimated classes to true classes by intercept proximity
    # ------------------------------------------------------------------
    true_intercepts = [TRUE["class1"]["intercept"], TRUE["class2"]["intercept"]]
    est_intercepts  = [theta_all[c, 0] for c in range(C)]

    if C == 2:
        # pick assignment that minimises total intercept gap
        err_direct = abs(est_intercepts[0] - true_intercepts[0]) + \
                     abs(est_intercepts[1] - true_intercepts[1])
        err_swap   = abs(est_intercepts[0] - true_intercepts[1]) + \
                     abs(est_intercepts[1] - true_intercepts[0])
        mapping = [0, 1] if err_direct <= err_swap else [1, 0]
    else:
        mapping = list(range(C))

    print("\n" + "=" * 72)
    print("  OUTCOME EQUATION RECOVERY")
    print("=" * 72)

    # Header
    col = 22
    print(f"  {'Variable':<{col}}", end="")
    for true_c in range(1, C + 1):
        print(f"  {'True C'+str(true_c):>10}  {'Est C'+str(true_c):>10}  {'Diff':>7}", end="")
    print()
    print("  " + "-" * (col + C * 31))

    # Outcome params
    param_names = list(fixed_names)  # already has __INTERCEPT__

    for k, name in enumerate(param_names):
        # Collect true values for each class (by display order)
        label = name.replace("__INTERCEPT__", "intercept")
        true_vals = []
        est_vals  = []
        for true_c_idx in range(C):
            tc  = true_c_idx + 1
            est_c = mapping[true_c_idx]
            tv  = TRUE[f"class{tc}"].get(label, None)
            ev  = float(theta_all[est_c, k])
            true_vals.append(tv)
            est_vals.append(ev)

        row = f"  {label:<{col}}"
        for tv, ev in zip(true_vals, est_vals):
            if tv is None:
                row += f"  {'n/a':>10}  {ev:>+10.4f}  {'':>7}"
            else:
                row += f"  {tv:>+10.4f}  {ev:>+10.4f}  {ev-tv:>+7.3f}"
        print(row)

    # Dispersion (last base param)
    alpha_idx = K_base - 1
    row = f"  {'alpha (NB disp)':<{col}}"
    for true_c_idx in range(C):
        tc    = true_c_idx + 1
        est_c = mapping[true_c_idx]
        tv    = TRUE[f"class{tc}"]["alpha"]
        ev    = float(np.exp(theta_all[est_c, alpha_idx]))
        row  += f"  {tv:>+10.4f}  {ev:>+10.4f}  {ev-tv:>+7.3f}"
    print(row)

    # ------------------------------------------------------------------
    # Membership equation
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  MEMBERSHIP EQUATION  (log-odds class 2 vs class 1)")
    print("  True: g0=0.0  gz1=+1.5  gz2=-1.0")
    print("=" * 72)

    mem_labels = ["(intercept)"] + list(membership_names)
    true_gamma = [
        TRUE["membership"]["g0"],
        TRUE["membership"]["gz1"],
        TRUE["membership"]["gz2"],
    ]

    gamma_full = gamma_flat.reshape(C - 1, -1)[0]  # row for class 2 vs ref

    print(f"  {'Variable':<20}  {'True':>10}  {'Estimated':>10}  {'Diff':>7}")
    print("  " + "-" * 52)
    for k, lbl in enumerate(mem_labels):
        tv = true_gamma[k] if k < len(true_gamma) else None
        ev = float(gamma_full[k])
        if tv is None:
            print(f"  {lbl:<20}  {'n/a':>10}  {ev:>+10.4f}")
        else:
            print(f"  {lbl:<20}  {tv:>+10.4f}  {ev:>+10.4f}  {ev-tv:>+7.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_test(verbose: bool = True):
    print("=" * 72)
    print("  LATENT CLASS SYNTHETIC RECOVERY TEST  (rich spec)")
    print("=" * 72)

    # ── generate data ──────────────────────────────────────────────────
    print()
    df = generate_latent_class_panel(N=800, T=3, seed=42)

    print(f"  Dataset : {df['id'].nunique()} individuals x {df['t'].nunique()} periods")
    print(f"  y       : mean={df['y'].mean():.2f}  zeros={100*(df['y']==0).mean():.1f}%")

    # True class shares actually realised in the sample
    n_per_class = df.groupby("true_class")["id"].nunique()
    for c, n in n_per_class.items():
        print(f"  Class {c} : {n} individuals  "
              f"(mean y={df[df['true_class']==c]['y'].mean():.2f})")

    # ── fit ────────────────────────────────────────────────────────────
    from experiment_package import ExperimentBuilder

    builder = ExperimentBuilder(df=df, id_col="id", y_col="y")

    manual_spec = builder.make_manual_spec(
        fixed_terms     = OUTCOME_VARS,         # x1 x2 x3 x4 in outcome eq.
        membership_terms= MEMBERSHIP_VARS,      # z1 z2 drive class probs
        dispersion      = 1,                    # NB2
        latent_classes  = 2,
    )

    print(f"\n  Fitting 2-class NB model with membership (z1, z2) ...")
    fit = builder.fit_manual_model(
        manual_spec=manual_spec,
        model="nb",
        print_report=verbose,
    )

    # ── recovery table ─────────────────────────────────────────────────
    from main_hpc_lc_patch import build_base_index
    from dataclasses import replace

    spec      = fit["spec"]
    params_np = np.array(fit["result"].params)
    C         = spec.latent_classes
    base_spec = replace(spec, latent_classes=1)
    base_idx  = build_base_index(base_spec)
    K_base    = base_idx["total_params"]
    gamma_flat = params_np[C * K_base:]

    _print_recovery_table(
        est_params      = params_np,
        fixed_names     = list(spec.fixed_names),
        K_base          = K_base,
        C               = C,
        gamma_flat      = gamma_flat,
        membership_names= spec.membership_names,
    )

    # ── class assignment accuracy ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CLASS ASSIGNMENT")
    print("=" * 72)
    try:
        lc_df = builder.compute_latent_class_probabilities(fit)
        prob_cols   = [f"class_{c+1}_prob" for c in range(C)]
        lc_df["pred_class"] = (
            lc_df[prob_cols].idxmax(axis=1)
            .str.extract(r"(\d+)").astype(int)
        )
        true_per_id = df.groupby("id")["true_class"].first().reset_index()
        merged      = lc_df[["id", "pred_class"]].merge(true_per_id, on="id")

        acc_direct = (merged["pred_class"] == merged["true_class"]).mean()
        acc_swap   = (merged["pred_class"] != merged["true_class"]).mean()
        acc        = max(acc_direct, acc_swap)
        label_swap = acc_swap > acc_direct
        print(f"  Assignment accuracy: {acc:.1%}  "
              f"({'labels swapped' if label_swap else 'labels match'})")

        # Show confusion matrix style summary
        merged["match"] = (merged["pred_class"] == merged["true_class"]) ^ label_swap
        for tc in [1, 2]:
            sub  = merged[merged["true_class"] == tc]
            corr = sub["match"].sum()
            print(f"    True class {tc}: {corr}/{len(sub)} correctly assigned")
    except Exception as exc:
        print(f"  [class probs] {exc}")

    # ── model fit stats ────────────────────────────────────────────────
    summary = fit.get("summary", {})
    if isinstance(summary, dict):
        bic = summary.get("bic")
        ll  = summary.get("loglik")
        k   = summary.get("num_parm")
        if bic is not None:
            print(f"\n  LL={ll:.2f}   k={k}   BIC={bic:.2f}")

    print("\n  Done.\n")
    return fit


if __name__ == "__main__":
    run_test(verbose=True)
