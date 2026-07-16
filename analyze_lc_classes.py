"""Fit a 2-class LC-NB model on Washington data & profile what each class means.

Run from repo root with venv activated:
    python analyze_lc_classes.py

Phases:
    1. Single-class NB warm-start
    2. Cluster-based seeding for per-class thetas
    3. DE warm-up  --  population search on the full LC objective
    4. EM  --  refines class assignments + gamma
    5. LBFGS polish
    6. Posterior class probabilities + profiling

Outputs:
    - Console summary: class profiles, per-class coefficients, membership gamma
    - lc_class_assignments.csv with posterior probs + hard assignments
    - lc_objective_trace.csv: LL at every EM iteration + DE/LBFGS endpoints
    - lc_objective_trace.png: plot of objective value over time

If DE warm-up is slow or unstable, set  SKIP_DE = True  below.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial
from jaxopt import LBFGS
from scipy import stats as scipy_stats

SKIP_DE = False          # Set to True to skip DE warm-up and go straight to EM

# -------------------------------------------------------------------
# Path & import setup
# -------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "metacountregressor"))

from metacountregressor import main_hpc_lc_patch  # noqa: E402, F401

from metacountregressor.main_hpc import (  # noqa: E402
    experiment_washington,
    CountModel,
    build_base_index,
    run_with_oom_recovery,
)
from metacountregressor.main_hpc_lc_patch import (  # noqa: E402
    ModelSpec,
    build_param_index,
    build_model_from_manual_spec,
    mixed_model_loglik,
    fit_em,
    compute_standard_errors,
    _seed_classes_from_clusters,
    unpack_lc_params,
    compute_lc_posteriors,
)

jax.config.update("jax_enable_x64", True)


# =====================================================================
# MODEL SPECIFICATION  --  loaded from Phase 1 search result, or default
# =====================================================================
SPEC_PATH = os.path.join(REPO_ROOT, "results", "best_model_spec.json")
if os.path.exists(SPEC_PATH):
    import json as _json
    with open(SPEC_PATH) as f:
        manual_spec = _json.load(f)
    manual_spec["latent_classes"] = manual_spec.get("latent_classes", 2)  # read from spec, default 2
    manual_spec.setdefault("min_class_proportion", 0.15)
    print(f"\n  Loaded best model spec from Phase 1 search:  {SPEC_PATH}")
    print(f"    fixed_terms:      {manual_spec.get('fixed_terms', [])}")
    print(f"    membership_terms: {manual_spec.get('membership_terms', [])}")
    print(f"    dispersion:       {'NB2' if manual_spec.get('dispersion') else 'Poisson'}")
else:
    print(f"\n  [warn] No best_model_spec.json found at {SPEC_PATH}")
    print(f"  Using default model spec (all fixed vars, no membership).")
    print(f"  Run Phase 1 search first to populate this.")
    FIXED_VARS = [
        "SPEED", "WIDTH", "CURVES", "MINRAD", "ACCESS", "GRADEBR",
        "FRICTION", "EXPOSE", "SINGLE", "TANGENT", "SLOPE",
        "GBRPM", "INTPM", "CPM", "MEDWIDTH", "INCLANES",
        "DOUBLE", "TRAIN", "PEAKHR",
    ]
    manual_spec: dict = {
        "fixed_terms":       FIXED_VARS,
        "rdm_terms":         [],
        "rdm_cor_terms":     [],
        "grouped_terms":     [],
        "hetro_in_means":    [],
        "zi_terms":          [],
        "membership_terms":  [],
        "group_id_col":      None,
        "dispersion":        1,
        "latent_classes":    2,
        "min_class_proportion": 0.15,
    }


# -------------------------------------------------------------------
# DE warm-up for the LC objective
# -------------------------------------------------------------------

def de_warmup_lc(
    init: np.ndarray,
    *,
    objective,
    data,
    spec,
    maxiter: int = 15,
    popsize: int = 12,
    rel_span: float = 1.5,
    abs_span: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, list]:
    """Population-based DE warm-up on the full latent-class log-likelihood.

    Returns (best_params, trace) where trace is a list of
    (iteration, best_objective_so_far) tuples.
    """
    import jax as _jax

    center = np.asarray(init, dtype=float).reshape(-1)
    center = np.where(np.isfinite(center), center, 0.0)
    center_jnp = jnp.array(center)
    n_params = len(center)
    n_pop = max(8, int(popsize) * 2)
    n_elite = max(2, n_pop // 4)
    large_val = 1e20

    span = np.maximum(float(abs_span), np.abs(center) * float(rel_span))
    lower = center_jnp - span
    upper = center_jnp + span

    def _obj(x):
        try:
            return float(objective(x, data, spec))
        except Exception:
            return large_val

    def _eval_pop(pop):
        try:
            vals = _jax.vmap(objective, in_axes=(0, None, None))(pop, data, spec)
            vals = jnp.where(jnp.isfinite(vals), vals, large_val)
            return np.asarray(vals, dtype=float)
        except Exception:
            return np.array([_obj(row) for row in np.asarray(pop)], dtype=float)

    incumbent_obj = _obj(center_jnp)
    trace = [(0, incumbent_obj)]  # negative LL at iteration 0

    try:
        key = _jax.random.PRNGKey(int(seed))
        key, sub = _jax.random.split(key)
        pop = center_jnp + _jax.random.normal(sub, (n_pop, n_params)) * span
        pop = jnp.clip(pop, lower, upper)
        pop = pop.at[0].set(center_jnp)

        best_x = center_jnp
        best_obj = incumbent_obj

        for it in range(int(maxiter)):
            vals = _eval_pop(pop)
            order = np.argsort(vals)

            if vals[order[0]] < best_obj:
                best_obj = float(vals[order[0]])
                best_x = pop[order[0]]

            trace.append((it + 1, best_obj))

            elite_idx = order[:n_elite]
            elite = pop[elite_idx]
            elite_mean = jnp.mean(elite, axis=0)
            elite_std = jnp.std(elite, axis=0)
            anneal = max(0.15, 0.92 ** (it + 1))
            step = jnp.maximum(elite_std, span * 0.05) * anneal

            key, sub = _jax.random.split(key)
            noise = _jax.random.normal(sub, (n_pop, n_params))
            offspring = elite_mean + noise * step
            offspring = jnp.clip(offspring, lower, upper)
            offspring = offspring.at[0].set(best_x)

            num_keep = min(n_elite, n_pop - 1)
            if num_keep > 0:
                offspring = offspring.at[1:1 + num_keep].set(elite[:num_keep])

            pop = offspring

        delta = incumbent_obj - best_obj
        if np.isfinite(best_obj) and delta > 1e-6:
            print(f"  DE warm-up:  obj {incumbent_obj:.2f} -> {best_obj:.2f}  "
                  f"(delta={delta:.2f})")
            return np.asarray(best_x, dtype=float), trace
        else:
            print(f"  DE warm-up:  no improvement (obj={incumbent_obj:.2f})")
            return center, trace

    except Exception as exc:
        print(f"  DE warm-up:  failed ({exc}), returning init")
        return center, trace


# Extract from loaded/default spec for display
FIXED_VARS = manual_spec.get("fixed_terms", [])
MEMBERSHIP_VARS = manual_spec.get("membership_terms", [])
DISPERSION = manual_spec.get("dispersion", 1)

# -------------------------------------------------------------------
# 1.  Load & prepare data
# -------------------------------------------------------------------
_, df, _all_vars = experiment_washington()
print(f"Loaded: {len(df)} segments  |  {df['ID'].nunique()} unique IDs")
print(f"URB: {df['URB'].mean():.1%} urban  |  "
      f"FC distribution: {dict(df['FC'].value_counts().sort_index())}")
print(f"Mean FREQ(Y): {df['Y'].mean():.2f}  |  "
      f"Mean AADT: {df['AADT'].mean():.0f}")

# -------------------------------------------------------------------
# MODEL IDENTITY BLOCK
# -------------------------------------------------------------------
print(f"\n{'=' * 72}")
print("MODEL IDENTIFICATION")
print(f"{'=' * 72}")
print(f"  Model type:          Latent-Class Count Model")
print(f"  Latent classes:      2")
print(f"  Outcome covariates:  {len(FIXED_VARS)} fixed effects")
print(f"    {', '.join(FIXED_VARS)}")
if MEMBERSHIP_VARS:
    print(f"  Membership covariates: {MEMBERSHIP_VARS}")
else:
    print(f"  Membership covariates: NONE  (constant class priors)")
print(f"  Random effects:       {manual_spec.get('rdm_terms', []) or 'NONE'}")
print(f"  Zero-inflation:       {manual_spec.get('zi_terms', []) or 'NONE'}")
print(f"  Dispersion:           {'NB2' if DISPERSION else 'Poisson'}")
print(f"  Per-class models:     {manual_spec.get('class_models', 'all uniform')}")
print()
print(f"  *** NEITHER FC NOR URB ARE PREDICTORS IN THIS MODEL ***")
print(f"  Both are held out entirely and used ONLY for external")
print(f"  validation to assess whether the latent classes")
print(f"  recover the known roadway functional classification.")
if os.path.exists(SPEC_PATH):
    print(f"  Model spec loaded from:  {SPEC_PATH}")
print(f"{'=' * 72}")

# -------------------------------------------------------------------
# 2.  Build model from spec
# -------------------------------------------------------------------
data, spec = build_model_from_manual_spec(
    df=df, manual_spec=manual_spec,
    id_col="ID", y_col="Y", offset_col="OFFSET",
    R=200,
)

C = spec.latent_classes
K_mem = spec.K_membership
base_spec = replace(spec, latent_classes=1)
pindex = build_param_index(spec)
class_K_base = list(pindex.get("class_K_base", [build_base_index(base_spec)["total_params"]] * C))
K_base_0 = class_K_base[0]
gamma_size = (C - 1) * (K_mem + 1)
n_obs = int(data["y"].shape[0])

print(f"\nSpec: {spec.Kf} fixed + {K_mem} membership  |  "
      f"latent classes={C}  |  per-class K_base={class_K_base}"
      f"  |  total params={pindex['total_params']}")
print(f"Observations: {n_obs}")

# -------------------------------------------------------------------
# 3.  Pre-fit: single-class NB  (warm-start for seeding)
# -------------------------------------------------------------------
print("\n--- Phase 1: single-class NB warm-start ---")
model_1 = CountModel(base_spec, data)
result_1 = run_with_oom_recovery(model_1.fit, label="single-class warm-start fit")
theta_1 = np.array(result_1.params)
ll_1 = -float(result_1.state.value)
k_1 = len(theta_1)
print(f"  Single-class LL = {ll_1:.2f}  |  "
      f"BIC = {k_1 * np.log(n_obs) - 2 * ll_1:.2f}")

# -------------------------------------------------------------------
# 4.  Cluster-based seeding of per-class thetas
# -------------------------------------------------------------------
print("\n--- Phase 2: cluster seeding ---")
rng = np.random.default_rng(42)
try:
    per_class_thetas = _seed_classes_from_clusters(
        theta_1, data, base_spec, C, K_base_0, rng,
        class_K_base=class_K_base,
    )
    theta_init = np.concatenate(per_class_thetas)
    print(f"  Cluster seeding OK  (class sizes from k-means)")
except Exception:
    print("  [warn] cluster seeding failed; falling back to jitter")
    theta_init = np.concatenate([
        theta_1[:k] + rng.normal(0, 0.05, k)
        if len(theta_1) >= k
        else np.pad(theta_1, (0, k - len(theta_1))) + rng.normal(0, 0.05, k)
        for k in class_K_base
    ])

gamma_init = np.zeros(gamma_size)
init_params = np.concatenate([theta_init, gamma_init])
init_ll = -float(mixed_model_loglik(jnp.array(init_params), data, spec))
print(f"  Initial LC-2 LL = {init_ll:.2f}")

# -- Initialize trace DataFrame
trace_records = []  # list of (phase, step, neg_loglik, ll, description)

# Record seed point
trace_records.append(("seed", 0, float(init_ll), -float(init_ll),
                      "cluster-seeded initial params"))

# -------------------------------------------------------------------
# 5.  DE warm-up: population search on full LC objective
# -------------------------------------------------------------------
de_trace = []
if not SKIP_DE:
    print("\n--- Phase 3: DE warm-up on full LC objective ---")
    params_de, de_trace = run_with_oom_recovery(
        de_warmup_lc,
        init_params,
        objective=mixed_model_loglik,
        data=data,
        spec=spec,
        maxiter=15,
        popsize=12,
        rel_span=1.5,
        abs_span=1.0,
        seed=42,
        label="DE warm-up",
    )
    de_ll = -float(mixed_model_loglik(jnp.array(params_de), data, spec))
    print(f"  DE-best LC-2 LL = {de_ll:.2f}  "
          f"(delta from seed = {de_ll - init_ll:+.2f})")
    for it, obj in de_trace:
        trace_records.append(("DE", it, float(obj), -float(obj),
                              f"DE iter {it}"))
    params_pre_em = params_de
else:
    print("\n--- Phase 3: DE warm-up  SKIPPED ---")
    params_pre_em = init_params

# -------------------------------------------------------------------
# 6.  EM: refine class assignments + gamma
# -------------------------------------------------------------------
print("\n--- Phase 4: EM ---")
em_step_offset = len(trace_records)  # where EM steps start in the trace
params_em, em_trace = run_with_oom_recovery(
    fit_em,
    init_params=params_pre_em, data=data, spec=spec,
    max_iter=100, tol=1e-6, verbose=True, return_trace=True,
    label="EM",
)

for (iteration, T, m_iters, ll_val, delta_ll, shares) in em_trace:
    trace_records.append(("EM", iteration, float(ll_val), -float(ll_val),
                          f"EM iter {iteration} T={T:.2f} shares={shares}"))

# -------------------------------------------------------------------
# 7.  MLE polish
# -------------------------------------------------------------------
print("\n--- Phase 5: LBFGS polish ---")
polish = LBFGS(
    fun=lambda p: mixed_model_loglik(p, data, spec),
    maxiter=500,
)
result = run_with_oom_recovery(
    polish.run, jnp.array(params_em), label="LBFGS polish"
)
params = np.array(result.params)
final_ll = -float(result.state.value)
k_lc = len(params)
bic_lc = k_lc * np.log(n_obs) - 2 * final_ll
bic_1 = k_1 * np.log(n_obs) - 2 * ll_1
print(f"  LC-2 LL = {final_ll:.2f}  |  BIC = {bic_lc:.2f}")
print(f"  delta-BIC vs single-class = {bic_lc - bic_1:+.2f}")
print(f"  (delta-BIC < 0 -> LC-2 preferred;  delta-BIC > 0 -> single-class sufficient)")

trace_records.append(("LBFGS", 0, float(final_ll), -float(final_ll),
                      "LBFGS polished final"))

# Build trace DataFrame
trace_df = pd.DataFrame(trace_records, columns=["phase", "step", "neg_loglik", "loglik", "description"])
trace_df["step_global"] = range(len(trace_df))

# -------------------------------------------------------------------
# 8.  Compute standard errors for ALL parameters
# -------------------------------------------------------------------
print("\n--- Phase 6: standard errors ---")
se_all = compute_standard_errors(params, partial(mixed_model_loglik, data=data, spec=spec))
print(f"  Standard errors computed for all {len(se_all)} parameters")

# -------------------------------------------------------------------
# 9.  Posterior class probabilities
# -------------------------------------------------------------------
N = data["y"].shape[0]
posterior, log_pi, logL = compute_lc_posteriors(params, data, spec)
hard_class = np.argmax(posterior, axis=1) + 1

# Extract per-class thetas and gamma for display
theta_list, gamma, pindex = unpack_lc_params(params, spec)
se_theta_list = []
for c in range(C):
    oc = pindex["class_offsets"][c]
    kc = pindex["class_K_base"][c]
    se_theta_list.append(se_all[oc:oc + kc])
se_gamma = se_all[pindex["class_gamma"][0]:pindex["class_gamma"][1]].reshape(C - 1, K_mem + 1)

# -------------------------------------------------------------------
# 10.  Build output DataFrame
# -------------------------------------------------------------------
df_out = df.copy()
df_out["class"] = hard_class
df_out["class_1_posterior"] = posterior[:, 0]
df_out["class_2_posterior"] = posterior[:, 1]
df_out["class_confidence"] = posterior.max(axis=1)

# =====================================================================
#                           FINAL MODEL RESULTS
# =====================================================================
print(f"\n{'=' * 72}")
print("FINAL MODEL: 2-Class LC Model (no membership covariates)")
print(f"{'=' * 72}")
print(f"  Model: LC, 2 classes, {len(FIXED_VARS)} outcome predictors")
print(f"  Per-class dist: {list(spec.models)}")
print(f"  Membership: NONE (constant class priors)")
print(f"  FC and URB are NOT in the model (held out for validation)")
print(f"  Log-Likelihood: {final_ll:.4f}")
print(f"  Parameters:     {k_lc}")
print(f"  AIC:            {2 * k_lc - 2 * final_ll:.4f}")
print(f"  BIC:            {bic_lc:.4f}")
print(f"  delta-BIC vs 1-class: {bic_lc - bic_1:+.4f}")
print(f"{'=' * 72}")

print(f"\n{'=' * 72}")
print("CLASS ASSIGNMENT SUMMARY")
print(f"{'=' * 72}")
print(f"  Class 1:  {np.mean(hard_class == 1):.1%}  "
      f"(n={np.sum(hard_class == 1)})")
print(f"  Class 2:  {np.mean(hard_class == 2):.1%}  "
      f"(n={np.sum(hard_class == 2)})")
print(f"  Mean posterior certainty: {df_out['class_confidence'].mean():.3f}")

print(f"\n  Class assignment vs URB (urban indicator):")
print(pd.crosstab(df_out["URB"], df_out["class"], margins=True).to_string())

# =====================================================================
# 11.  VALIDATION: comparison to known Functional Class (FC)
# =====================================================================
print(f"\n{'=' * 72}")
print("VALIDATION vs FUNCTIONAL CLASS (FC)")
print(f"{'=' * 72}")
print(f"  FC is NOT a predictor.  It is used ONLY to assess whether")
print(f"  the latent classes recover the known road classification.")
print()

print(f"  Class assignment vs FC:")
ct_fc = pd.crosstab(df_out["FC"], df_out["class"], margins=True)
print(ct_fc.to_string())
print()

# For validation: FC = 1 is "principal arterial", FC = 2 is "minor arterial",
# FC = 5 is "collector". We test two mappings:
#   Mapping A: FC in {1, 2} -> class 1 (higher-type roads), FC=5 -> class 2
#   Mapping B: FC in {1, 2} -> class 2, FC=5 -> class 1
# For each we compute accuracy, precision, recall, F1 (treating FC=5 as positive=class2)
print(f"  FC RECOVERY VALIDATION METRICS")
print(f"  {'-' * 68}")
print(f"  Target grouping:  FC in {{1, 2}} vs FC = 5")
print(f"  (Principal/minor arterials vs collectors)")

# Mapping A: FC={1,2} -> class 1, FC=5 -> class 2
fc_target_a = np.where(df_out["FC"].isin([1, 2]), 1, 2)
acc_a = np.mean(hard_class == fc_target_a)

# Mapping B: FC={1,2} -> class 2, FC=5 -> class 1
fc_target_b = np.where(df_out["FC"].isin([1, 2]), 2, 1)
acc_b = np.mean(hard_class == fc_target_b)

# Pick the better mapping
if acc_a >= acc_b:
    fc_target = fc_target_a
    mapping_desc = "FC={1,2} -> Class 1,  FC=5 -> Class 2"
else:
    fc_target = fc_target_b
    mapping_desc = "FC={1,2} -> Class 2,  FC=5 -> Class 1"

accuracy = np.mean(hard_class == fc_target)
print(f"  Best mapping:  {mapping_desc}")
print(f"  Accuracy:      {accuracy:.4f}  ({accuracy:.1%})")
print()

# Detailed per-FC-class accuracy
for fc_val in sorted(df_out["FC"].unique()):
    mask_fc = df_out["FC"] == fc_val
    n_fc = mask_fc.sum()
    n_correct = np.sum(hard_class[mask_fc] == fc_target[mask_fc])
    print(f"    FC={int(fc_val)}:  {n_correct}/{n_fc} correct  "
          f"({n_correct/max(n_fc,1):.1%})")

# Confusion matrix with percentages
print(f"\n  CONFUSION MATRIX (counts and row %)")
print(f"  {'-' * 42}")
fc_values = sorted(df_out["FC"].unique())
class_values = [1, 2]
print(f"  {'':>6}  {'Class 1':>10} {'Class 2':>10}")
for fc_val in fc_values:
    mask = df_out["FC"] == fc_val
    n = mask.sum()
    c1 = np.sum(hard_class[mask] == 1)
    c2 = np.sum(hard_class[mask] == 2)
    print(f"  {'FC=' + str(int(fc_val)):>6}  "
          f"{c1:>5d} ({c1/max(n,1)*100:5.1f}%)  "
          f"{c2:>5d} ({c2/max(n,1)*100:5.1f}%)")

# Precision, Recall, F1 for each class (treating class 2 as "positive")
# Under the best mapping
tp = np.sum((hard_class == 2) & (fc_target == 2))
fp = np.sum((hard_class == 2) & (fc_target == 1))
fn = np.sum((hard_class == 1) & (fc_target == 2))
tn = np.sum((hard_class == 1) & (fc_target == 1))

precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-12)

print(f"\n  CLASSIFICATION METRICS (treating assigned Class 2 as 'collector')")
print(f"    True Positives:  {tp}")
print(f"    False Positives: {fp}")
print(f"    True Negatives:  {tn}")
print(f"    False Negatives: {fn}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall:    {recall:.4f}")
print(f"    F1 Score:  {f1:.4f}")

# =====================================================================
# 12.  Covariate profiles by class
# =====================================================================
PROFILE_VARS = [
    "URB", "FC", "SPEED", "AADT", "WIDTH", "CURVES", "MINRAD",
    "ACCESS", "MEDWIDTH", "Y", "LENGTH", "SINGLE",
    "DOUBLE", "TRAIN", "GRADEBR", "TANGENT", "SLOPE",
    "FRICTION", "EXPOSE", "INCLANES", "GBRPM", "INTPM", "CPM",
]

print(f"\n{'=' * 72}")
print("COVARIATE PROFILES BY ASSIGNED CLASS")
print(f"{'=' * 72}")
print(f"  {'Variable':<14} {'Class 1':>10}  {'Class 2':>10}  "
      f"{'Diff (1-2)':>11}  {'p-value'}")
print(f"  {'-' * 14} {'-' * 10}  {'-' * 10}  {'-' * 11}  {'-' * 8}")

for v in PROFILE_VARS:
    c1 = df_out.loc[df_out["class"] == 1, v]
    c2 = df_out.loc[df_out["class"] == 2, v]
    try:
        _, p = scipy_stats.mannwhitneyu(c1, c2, alternative="two-sided")
    except Exception:
        p = np.nan
    diff = c1.mean() - c2.mean()
    sig = ("  ***" if p < 0.001 else "  **" if p < 0.01
           else "  *" if p < 0.05 else "")
    print(f"  {v:<14} {c1.mean():>10.3f}  {c2.mean():>10.3f}  "
          f"{diff:>+11.3f}  {p:>8.4f}{sig}")

# =====================================================================
# 13.  Per-class outcome coefficients WITH STANDARD ERRORS
# =====================================================================
print(f"\n{'=' * 72}")
print("OUTCOME MODEL COEFFICIENTS WITH STANDARD ERRORS")
print(f"{'=' * 72}")
print(f"  Per-class distributions: {list(spec.models)}")
print(f"  These are the coefficients of the FIXED-EFFECT outcome model")
print(f"  for each latent class. FC is NOT among these predictors.")
print()

base_idx = build_base_index(base_spec)
fixed_names = list(base_spec.fixed_names)

for c_idx in range(C):
    theta_c = theta_list[c_idx]
    se_c = se_theta_list[c_idx]
    _model_c = spec.models[c_idx]
    _base_idx_c = build_base_index(base_spec, model=_model_c)

    print(f"\n  {'=' * 60}")
    print(f"  CLASS {c_idx + 1}  OUTCOME MODEL  [{_model_c.upper()}]")
    print(f"  {'=' * 60}")
    print(f"  {'Parameter':<16} {'Estimate':>10}  {'Std.Err':>10}  "
          f"{'z-value':>9}  {'p-value':>9}  {'exp(beta)':>10}")
    print(f"  {'-' * 16} {'-' * 10}  {'-' * 10}  "
          f"{'-' * 9}  {'-' * 9}  {'-' * 10}")

    f_start = _base_idx_c["fixed"][0]
    for i, name in enumerate(fixed_names):
        idx = f_start + i
        val = theta_c[idx]
        se_val = se_c[idx]
        z_val = val / max(se_val, 1e-12)
        p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z_val)))
        stars = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                 else "*" if p_val < 0.05 else "")
        print(f"  {name:<16} {val:>10.4f}  {se_val:>10.4f}  "
              f"{z_val:>9.3f}  {p_val:>9.4f}{stars}  {np.exp(val):>10.4f}")

    if "dispersion" in _base_idx_c:
        disp_idx = _base_idx_c["dispersion"]
        # alpha is stored as raw parameter; compute SE via delta method
        alpha_raw = theta_c[disp_idx]
        se_alpha_raw = se_c[disp_idx]
        alpha = np.exp(alpha_raw)
        se_alpha = alpha * se_alpha_raw  # delta method: d(exp(x))/dx = exp(x)
        z_alpha = alpha_raw / max(se_alpha_raw, 1e-12)
        p_alpha = 2 * (1 - scipy_stats.norm.cdf(abs(z_alpha)))
        print(f"  {'alpha (disp)':<16} {alpha:>10.4f}  {se_alpha:>10.4f}  "
              f"{z_alpha:>9.3f}  {p_alpha:>9.4f}")

# =====================================================================
# 14.  Membership equation WITH STANDARD ERRORS
# =====================================================================
print(f"\n{'=' * 72}")
print("MEMBERSHIP EQUATION  (with standard errors)")
print(f"{'=' * 72}")
if K_mem > 0:
    print(f"  log[pi_2(n) / pi_1(n)] = g_0 + g_1 * {' + '.join(spec.membership_names)}")
    print(f"  These parameters determine how covariates drive")
    print(f"  class membership probabilities for each observation.")
else:
    print(f"  log[pi_2(n) / pi_1(n)] = g_0   (constant, no membership covariates)")
    print(f"  All observations share the same prior class probabilities.")
print()

mem_names = ["(intercept)"] + list(spec.membership_names)
print(f"  {'Parameter':<16} {'Estimate':>10}  {'Std.Err':>10}  "
      f"{'z-value':>9}  {'p-value':>9}  {'exp(beta)':>10}")
print(f"  {'-' * 16} {'-' * 10}  {'-' * 10}  "
      f"{'-' * 9}  {'-' * 9}  {'-' * 10}")

for k, name in enumerate(mem_names):
    g_val = gamma[0, k]
    sg_val = se_gamma[0, k]
    z_val = g_val / max(sg_val, 1e-12)
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z_val)))
    stars = ("***" if p_val < 0.001 else "**" if p_val < 0.01
             else "*" if p_val < 0.05 else "")
    print(f"  {name:<16} {g_val:>10.4f}  {sg_val:>10.4f}  "
          f"{z_val:>9.3f}  {p_val:>9.4f}{stars}  {np.exp(g_val):>10.4f}")

# Marginal class shares
if K_mem > 0:
    logits_marg = np.concatenate([np.zeros(1), gamma[:, 0]])
else:
    logits_marg = np.array([0.0, float(gamma[0, 0])])
pi_marg = np.exp(logits_marg) / np.exp(logits_marg).sum()
print(f"\n  Marginal class shares (constant priors):")
for c in range(C):
    print(f"    pi_{c + 1} = {pi_marg[c]:.6f}  ({pi_marg[c]:.1%})")

# Class probabilities by URB (post-hoc profile only, URB not in model)
if K_mem > 0:
    print(f"\n  Prior class probabilities by membership covariates:")
    for urb_val in [0, 1]:
        z_row = np.array([[1.0, float(urb_val)]])
        logits_row = z_row @ gamma.T
        logits_full_row = np.concatenate([np.zeros((1, 1)), logits_row], axis=1)
        pi_row = (np.exp(logits_full_row)
                  / np.exp(logits_full_row).sum(axis=1, keepdims=True))
        print(f"    value={urb_val}:  pi_1={pi_row[0, 0]:.6f}  pi_2={pi_row[0, 1]:.6f}")

# =====================================================================
# 15.  PLOT objective trace
# =====================================================================
print(f"\n{'=' * 72}")
print("OBJECTIVE TRACE PLOT")
print(f"{'=' * 72}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: grouped by phase (LL maximizing, so neg_loglik descending)
    phases = trace_df["phase"].unique()
    colors = {"seed": "gray", "DE": "blue", "EM": "green", "LBFGS": "red"}
    for ph in phases:
        mask = trace_df["phase"] == ph
        ax1.plot(trace_df.loc[mask, "step_global"],
                 trace_df.loc[mask, "neg_loglik"],
                 marker="." if ph == "DE" else "o" if ph == "LBFGS" else None,
                 markersize=4, color=colors.get(ph, "black"),
                 label=ph, alpha=0.8, linewidth=1.5)

    ax1.set_xlabel("Cumulative step")
    ax1.set_ylabel("Negative Log-Likelihood (lower = better)")
    ax1.set_title("LC Model Objective Trace (all phases)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # -- Right: EM iterations only (LL, maximizing)
    em_mask = trace_df["phase"] == "EM"
    em_df = trace_df[em_mask]
    if len(em_df) > 0:
        ax2.plot(em_df["step"], em_df["loglik"], "g.-", linewidth=1.5, markersize=4)
        ax2.set_xlabel("EM Iteration")
        ax2.set_ylabel("Log-Likelihood (higher = better)")
        ax2.set_title("EM Convergence: Log-Likelihood by Iteration")
        ax2.grid(True, alpha=0.3)

    fig.suptitle("LC Model: Objective Function Over Time\n"
                 "(2 classes, no membership covariates, FC & URB held out)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    plot_path = os.path.join(REPO_ROOT, "results", "lc_objective_trace.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved to:  {plot_path}")
except Exception as exc:
    print(f"  [warn] Plot failed: {exc}")

# =====================================================================
# 16.  Save outputs
# =====================================================================

# -- Class assignments CSV
COLUMNS_TO_SAVE = [
    "ID", "class", "class_1_posterior", "class_2_posterior",
    "class_confidence",
] + PROFILE_VARS

out_path = os.path.join(REPO_ROOT, "lc_class_assignments.csv")
df_out[COLUMNS_TO_SAVE].to_csv(out_path, index=False)
print(f"\nClass assignments + profiles saved to:  {out_path}")

# -- Objective trace CSV
trace_path = os.path.join(REPO_ROOT, "results", "lc_objective_trace.csv")
trace_df.to_csv(trace_path, index=False)
print(f"Objective trace saved to:  {trace_path}")

# =====================================================================
# 17.  FINAL MODEL SUMMARY (compact, single block)
# =====================================================================
print(f"\n{'=' * 72}")
print("COMPACT FINAL MODEL SUMMARY")
print(f"{'=' * 72}")
print(f"  MODEL:  2-Class LC  |  Per-class: {list(spec.models)}  |  Membership: NONE  |  FC & URB: HELD OUT")
print(f"  N = {n_obs}  |  LL = {final_ll:.2f}  |  params = {k_lc}")
print(f"  AIC = {2 * k_lc - 2 * final_ll:.2f}  |  BIC = {bic_lc:.2f}")
print(f"  delta-BIC vs 1-class = {bic_lc - bic_1:+.2f}")
print(f"  Class shares:  {pi_marg[0]:.3f} / {pi_marg[1]:.3f}")
print(f"  FC recovery accuracy: {accuracy:.3f}")
print(f"{'=' * 72}")

print("\nDone.")
