"""
Latent-Class Model Comparison Experiment
==========================================
PRIMARY HYPOTHESIS
------------------
We hypothesise that crash frequency data contains TWO latent sub-populations
corresponding to **urban** and **rural/suburban** road types.

  Class 1  (low-baseline, traffic-sensitive):
      High AADT sensitivity; typical of urban arterials with high exposure.
      x1 large positive, x4 large positive.

  Class 2  (high-baseline, geometry-driven):
      Weaker AADT effect; dominated by road geometry / alignment.
      x1 negative or near-zero, x2 positive.

Membership (class 1 vs 2) is expected to be predicted by:
  - z1  (urban indicator proxy)  →  higher z1 → more likely urban (class 2)
  - z2  (speed / rurality proxy) →  higher z2 → more likely rural (class 1)

WHAT THIS SCRIPT TESTS
-----------------------
1. Pre-fit validation  —  standardisation warnings, AADT log-transform check,
   VIF collinearity screen, overdispersion assessment.
2. Smoke test  —  confirms the full fitting chain works before a long run.
3. Structural comparison  —  10 model variants ranked by BIC:
     single-class baselines → LC-2 (no/with membership) → random-param
     → LC-3 (over-specified)
4. Split-class validation  —  two separate single-class NB models fit on
   z1-split sub-samples (high-z1 ≈ urban, low-z1 ≈ rural) are compared
   to the joint LC-2 model. Summed split BIC vs joint BIC indicates
   whether the observed urban/rural split is informationally sufficient.

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

import sys, os, traceback
import warnings as _warnings_module
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Only suppress numerical/convergence noise; keep UserWarning for data issues.
_warnings_module.filterwarnings("ignore", category=RuntimeWarning)
_warnings_module.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# TRUE DGP  — encodes the urban / rural latent class hypothesis
# ─────────────────────────────────────────────────────────────────────────────

HYPOTHESIS = """
URBAN / RURAL LATENT CLASS HYPOTHESIS
======================================
Class 1  ("rural / geometry-driven")
  intercept = -3.0  (low baseline crash rate)
  x1 = +1.0        [traffic volume / AADT proxy]: strongly positive
  x2 = -0.8        [alignment / horizontal curvature]: negative (safer roads)
  x3 = +0.3        [lane width or gradient]: mild positive
  x4 = +0.1        [near-zero, negligible differentiator]
  alpha (NB dispersion) = 1.5

Class 2  ("urban / exposure-driven")
  intercept = -0.8  (higher baseline crash rate)
  x1 = -0.4        [AADT effect reverses: congested urban roads self-regulate]
  x2 = +0.6        [intersection / conflict-point density]: positive
  x3 = +0.2        [similar mild gradient effect]
  x4 = +0.9        [access points / driveways]: large positive urban effect
  alpha = 0.6       [less dispersed, more homogeneous urban crash patterns]

Membership logit (class 2 vs class 1):
  g0  = 0.0
  gz1 = +1.5   z1 high (urban indicator) -> more likely class 2 (urban)
  gz2 = -1.0   z2 high (rurality/speed)  -> more likely class 1 (rural)

Validation strategy
-------------------
  (a) A joint LC-2 model should recover the sign pattern above.
  (b) Two single-class NB models fit on high-z1 vs low-z1 sub-samples
      serve as the "observed-split" baseline.  If the joint LC model
      has lower summed BIC, the latent structure is justified.
"""

TRUE = {
    "class1": {
        "label":     "rural / geometry-driven",
        "intercept": -3.0,
        "x1":        +1.0,
        "x2":        -0.8,
        "x3":        +0.3,
        "x4":        +0.1,
        "alpha":      1.5,
    },
    "class2": {
        "label":     "urban / exposure-driven",
        "intercept": -0.8,
        "x1":        -0.4,
        "x2":        +0.6,
        "x3":        +0.2,
        "x4":        +0.9,
        "alpha":      0.6,
    },
    "membership": {"g0": 0.0, "gz1": +1.5, "gz2": -1.0},
}

OUTCOME_VARS    = ["x1", "x2", "x3", "x4"]
MEMBERSHIP_VARS = ["z1", "z2"]


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    # (label, model_family, latent_classes, use_membership, use_random)
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
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_data(N: int = 800, T: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    Simulate a panel dataset matching the urban/rural latent class DGP.

    Variables
    ---------
    x1  AADT proxy  (log-transformed; ~ N(0,1) on log scale)
    x2  Horizontal curvature / alignment index
    x3  Gradient / vertical grade
    x4  Access-point density (urban differentiator)
    z1  Urban indicator proxy (drives class 2 membership)
    z2  Rurality / posted-speed proxy (drives class 1 membership)

    NOTE: x1 is generated on a log scale, i.e. it represents log(AADT).
    If you use a raw AADT column in real data, apply log(AADT) first.
    """
    rng = np.random.default_rng(seed)
    mem = TRUE["membership"]
    rows = []
    for i in range(N):
        # x1 = log(AADT) proxy — already on log scale
        x1 = rng.normal(0, 1)
        x2 = rng.normal(0, 1)
        x3 = rng.normal(0, 1)
        x4 = rng.normal(0, 1)
        z1 = rng.normal(0, 1)   # urban indicator proxy
        z2 = rng.normal(0, 1)   # rurality / speed proxy

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
            rows.append({
                "id": i, "t": t,
                "x1": x1, "x2": x2, "x3": x3, "x4": x4,
                "z1": z1, "z2": z2,
                # binary urban indicator derived from z1 median split
                "urban": int(z1 > 0),
                "y": int(y), "true_class": c,
            })
    return pd.DataFrame(rows)


def _check_collinearity(df: pd.DataFrame, cols: list) -> None:
    """Print pairwise Pearson correlations and flag |r| > 0.7."""
    print("\n  Pairwise correlations (outcome + membership vars):")
    print(f"  {'':>6}", end="")
    for c in cols:
        print(f"  {c:>6}", end="")
    print()
    high_pairs = []
    for ci in cols:
        print(f"  {ci:>6}", end="")
        for cj in cols:
            r = df[ci].corr(df[cj])
            flag = "*" if abs(r) > 0.7 and ci != cj else " "
            print(f"  {r:>+6.2f}{flag}", end="")
            if abs(r) > 0.7 and ci < cj:
                high_pairs.append((ci, cj, r))
        print()
    if high_pairs:
        print("\n  HIGH COLLINEARITY (|r| > 0.7):")
        for ci, cj, r in high_pairs:
            _warnings_module.warn(
                f"  [WARN] '{ci}' and '{cj}': r={r:.3f} — "
                "consider dropping one or using orthogonal contrasts.",
                UserWarning, stacklevel=3,
            )
            print(f"    {ci} ↔ {cj} : r={r:.3f}  ← high correlation")
    else:
        print("\n  No high collinearity detected (|r| <= 0.70 for all pairs).")


def _validate_aadt_columns(df: pd.DataFrame, cols: list) -> None:
    """Warn if any column looks like raw (un-logged) AADT."""
    for col in cols:
        if any(kw in col.upper() for kw in ("AADT", "ADT", "VOLUME", "EXPO")):
            med = df[col].median()
            mn  = df[col].min()
            if med > 500 and mn >= 0:
                msg = (
                    f"  [WARN] '{col}': median={med:,.1f}  min={mn:.1f}  "
                    "→ looks like RAW traffic volume. Apply log() before fitting."
                )
                _warnings_module.warn(msg, UserWarning, stacklevel=3)
                print(msg)
            else:
                print(f"  [OK]   '{col}': median={med:.3f}  "
                      "(appears log-transformed or standardised)")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
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
    """x1/x2 per-class estimates for sign-recovery check."""
    try:
        from main_hpc_lc_patch import build_param_index
        from dataclasses import replace

        spec      = fit["spec"]
        params_np = np.array(fit["result"].params)
        pindex    = build_param_index(spec)
        class_offsets = list(pindex.get("class_offsets", []))
        class_K_base  = list(pindex.get("class_K_base", []))

        if not class_offsets:
            base_spec = replace(spec, latent_classes=1)
            K_base    = build_param_index(base_spec)["total_params"]
            class_offsets = [i * K_base for i in range(C)]
            class_K_base  = [K_base] * C

        parts = []
        for c in range(C):
            oc = class_offsets[c]
            kc = class_K_base[c]
            thetas_c = params_np[oc:oc + kc]
            names     = list(spec.fixed_names)
            idx_x1 = names.index("x1") if "x1" in names else None
            idx_x2 = names.index("x2") if "x2" in names else None
            seg = f"C{c+1}["
            if idx_x1 is not None and idx_x1 < len(thetas_c):
                seg += f"x1={thetas_c[idx_x1]:+.2f}"
            if idx_x2 is not None and idx_x2 < len(thetas_c):
                seg += f" x2={thetas_c[idx_x2]:+.2f}"
            seg += "]"
            parts.append(seg)
        return "  ".join(parts)
    except Exception:
        return "—"


def _membership_summary(fit, C):
    """Gamma (membership) coefficients for z1/z2."""
    try:
        from main_hpc_lc_patch import build_param_index
        from dataclasses import replace

        spec      = fit["spec"]
        if spec.K_membership == 0:
            return "—"
        params_np = np.array(fit["result"].params)
        pindex    = build_param_index(spec)
        class_offsets = list(pindex.get("class_offsets", []))
        class_K_base  = list(pindex.get("class_K_base", []))

        if not class_offsets:
            base_spec = replace(spec, latent_classes=1)
            K_base    = build_param_index(base_spec)["total_params"]
            total_theta = C * K_base
        else:
            total_theta = class_offsets[-1] + class_K_base[-1] if C > 0 else 0

        gamma_raw = params_np[total_theta:]
        gamma_size = (C - 1) * (spec.K_membership + 1)
        if len(gamma_raw) >= gamma_size:
            gamma = gamma_raw[:gamma_size].reshape(C - 1, spec.K_membership + 1)
        else:
            gamma = np.zeros((C - 1, spec.K_membership + 1))
            if len(gamma_raw) > 0:
                gamma.flat[:len(gamma_raw)] = gamma_raw
        mem_names = ["intercept"] + list(spec.membership_names)
        parts = []
        for k, nm in enumerate(mem_names):
            if nm in ("z1", "z2", "intercept"):
                parts.append(f"{nm}={gamma[0, k]:+.2f}")
        return "  ".join(parts)
    except Exception:
        return "—"


def _random_stddev_summary(fit, C):
    """StdDev of random x1 per class."""
    try:
        from main_hpc_lc_patch import build_param_index
        from dataclasses import replace
        spec      = fit["spec"]
        params_np = np.array(fit["result"].params)
        pindex    = build_param_index(spec)
        class_offsets = list(pindex.get("class_offsets", []))
        class_K_base  = list(pindex.get("class_K_base", []))

        if not class_offsets:
            base_spec = replace(spec, latent_classes=1)
            K_base    = build_param_index(base_spec)["total_params"]
            class_offsets = [i * K_base for i in range(C)]
            class_K_base  = [K_base] * C

        if not hasattr(spec, "random_ind_names") or len(spec.random_ind_names) == 0:
            return "—"
        idx_sd = None
        for i, n in enumerate(spec.random_ind_names):
            if n == "x1":
                idx_sd = len(spec.fixed_names) + i
                break
        if idx_sd is None:
            return "—"
        parts = []
        for c in range(C):
            oc = class_offsets[c]
            kc = class_K_base[c]
            thetas_c = params_np[oc:oc + kc]
            if idx_sd < len(thetas_c):
                parts.append(f"C{c+1}[StdDev(x1)={thetas_c[idx_sd]:+.2f}]")
            else:
                parts.append(f"C{c+1}[StdDev(x1)=—]")
        return "  ".join(parts)
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(N: int = 800, T: int = 3, seed: int = 42):

    # ── Print hypothesis ───────────────────────────────────────────
    print("=" * 80)
    print("  LATENT CLASS MODEL COMPARISON  —  urban/rural road class hypothesis")
    print("=" * 80)
    print(HYPOTHESIS)

    # ── Generate data ──────────────────────────────────────────────
    df = generate_data(N=N, T=T, seed=seed)
    print(f"\n  Dataset: {df['id'].nunique()} individuals x {T} periods  "
          f"y mean={df['y'].mean():.2f}  zeros={100*(df['y']==0).mean():.1f}%")
    true_shares = df.groupby("true_class")["id"].nunique()
    print(f"  True class shares: " +
          "  ".join(
              f"class{c}={n/df['id'].nunique():.2f} ({TRUE[f'class{c}']['label']})"
              for c, n in true_shares.items()
          ))

    # ── Pre-fit validation ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PRE-FIT VALIDATION")
    print("=" * 80)

    all_vars = OUTCOME_VARS + MEMBERSHIP_VARS
    _validate_aadt_columns(df, all_vars)

    print("\n  Variable summary (outcome vars + membership vars):")
    print(f"  {'Variable':<8}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}  "
          f"{'Zeros%':>7}")
    print("  " + "-" * 55)
    for col in all_vars:
        s = df[col]
        print(f"  {col:<8}  {s.mean():>8.3f}  {s.std():>8.3f}  "
              f"{s.min():>8.3f}  {s.max():>8.3f}  {(s==0).mean()*100:>6.1f}%")

    _check_collinearity(df, all_vars)

    y_vr = df["y"].var() / df["y"].mean() if df["y"].mean() > 0 else 0
    print(f"\n  Outcome 'y': var/mean = {y_vr:.2f}  "
          f"({'overdispersed → NB recommended' if y_vr > 1.5 else 'near-Poisson'})")

    # ── Smoke test ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SMOKE TEST  —  verify fitting pipeline before full search")
    print("=" * 80)

    from experiment_package import ExperimentBuilder
    builder = ExperimentBuilder(df=df, id_col="id", y_col="y")

    smoke_ok = builder.smoke_test(
        fixed_terms=OUTCOME_VARS[:2],
        model="nb",
        latent_classes=2,
        R=50,
    )
    if not smoke_ok:
        print("  *** Smoke test FAILED — aborting full experiment ***")
        return None

    # ── Full variant comparison ────────────────────────────────────
    print("\n" + "=" * 80)
    print("  STRUCTURAL VARIANT COMPARISON")
    print("=" * 80)

    results = []

    for label, model_fam, n_classes, use_mem, use_rdm in VARIANTS:
        print(f"  Fitting: {label} ...", end="", flush=True)

        dispersion = 1 if model_fam == "nb" else 0
        spec_kwargs = dict(
            fixed_terms    = OUTCOME_VARS,
            dispersion     = dispersion,
            latent_classes = n_classes,
        )
        if use_mem:
            spec_kwargs["membership_terms"] = MEMBERSHIP_VARS
        if use_rdm:
            spec_kwargs["rdm_terms"] = ["x1:normal"]

        try:
            manual_spec = builder.make_manual_spec(**spec_kwargs)
            fit = builder.fit_manual_model(
                manual_spec=manual_spec,
                model=model_fam,
                print_report=False,
            )
            summary = fit.get("summary", {})
            ll   = summary.get("loglik",  float("nan"))
            bic  = summary.get("bic",     float("nan"))
            k    = summary.get("num_parm", "?")

            acc    = _class_assignment_accuracy(builder, fit, df, n_classes) \
                     if n_classes > 1 else float("nan")
            rec    = _recovery_summary(fit, n_classes)
            mem    = _membership_summary(fit, n_classes)
            stddev = _random_stddev_summary(fit, n_classes) if use_rdm else "—"

            results.append({
                "label": label, "C": n_classes, "model": model_fam,
                "mem": use_mem, "rdm": use_rdm,
                "LL": ll, "BIC": bic, "k": k, "acc": acc,
                "x1x2_recovery": rec, "gamma": mem, "stddev": stddev,
                "status": "OK",
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
                "x1x2_recovery": "FAILED", "gamma": "—", "stddev": "—",
                "status": f"FAILED: {exc}",
            })

    # ── Summary table ─────────────────────────────────────────────
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
        dbic    = row["BIC"] - best_bic if not np.isnan(row["BIC"]) else float("nan")
        acc_str = f"{row['acc']:.1%}" if not np.isnan(row["acc"]) else "  n/a"
        bic_str = f"{row['BIC']:>10.1f}" if not np.isnan(row["BIC"]) else f"{'FAILED':>10}"
        ll_str  = f"{row['LL']:>10.1f}"  if not np.isnan(row["LL"])  else f"{'—':>10}"
        dbic_str= f"{dbic:>+7.1f}"       if not np.isnan(dbic)        else f"{'—':>7}"
        marker  = "  << TRUE DGP" if "DGP" in row["label"] else ""
        print(f"  {rank:<5}  {row['label']:<38}  {row['C']:>2}  "
              f"{ll_str}  {row['k']:>4}  {bic_str}  {dbic_str}  {acc_str:>6}{marker}")

    # ── x1/x2 sign recovery ───────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  CLASS-SPECIFIC COEFFICIENT RECOVERY  (x1 / x2)")
    print(f"  True DGP: C1[x1=+1.00 x2=-0.80]  C2[x1=-0.40 x2=+0.60]")
    print(f"  Hypothesis: C1=rural (traffic-sensitive), C2=urban (geometry-driven)")
    print("=" * 80)
    for _, row in df_res.iterrows():
        if row["C"] > 1:
            print(f"  {row['label']:<40}  {row['x1x2_recovery']}")

    # ── Membership gamma recovery ─────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  MEMBERSHIP EQUATION RECOVERY")
    print("  True: intercept=0.00  z1=+1.50  z2=-1.00")
    print("  Hypothesis: z1 (urban proxy) → class 2; z2 (rural/speed) → class 1")
    print("=" * 80)
    for _, row in df_res.iterrows():
        if row["mem"]:
            print(f"  {row['label']:<40}  {row['gamma']}")

    # ── Random param recovery ─────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  RANDOM PARAMETER StdDev RECOVERY (per class, for x1)")
    print("=" * 80)
    for _, row in df_res.iterrows():
        if row["rdm"]:
            print(f"  {row['label']:<40}  {row['stddev']}")

    # ── Split-class validation (urban/rural parallel models) ──────
    print("\n\n" + "=" * 80)
    print("  SPLIT-CLASS VALIDATION")
    print("  Observed split: z1 > 0  (urban proxy) vs z1 ≤ 0  (rural proxy)")
    print("  Fits two separate NB models on each sub-sample and compares")
    print("  summed BIC to the joint LC-2 + membership model.")
    print("=" * 80)

    split_result = builder.fit_split_class_models(
        split_col="urban",                    # binary indicator derived from z1 median split
        fixed_terms=OUTCOME_VARS,
        model="nb",
        dispersion=1,
        membership_terms=MEMBERSHIP_VARS,
        R=100,
        hypothesis_label="urban vs rural road class (z1 median split)",
    )

    print("\n  Done.\n")
    return df_res


if __name__ == "__main__":
    run_experiment()
