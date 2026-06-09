"""Fit a 2-class LC-NB model on Washington data & profile what each class means.

Run from repo root with venv activated:
    python analyze_lc_classes.py

Phases:
    1. Single-class NB warm-start
    2. Cluster-based seeding for per-class thetas
    3. DE warm-up  —  population search on the full LC objective
    4. EM  —  refines class assignments + gamma
    5. LBFGS polish
    6. Posterior class probabilities + profiling

Outputs:
    - Console summary: class profiles, per-class coefficients, membership gamma
    - lc_class_assignments.csv with posterior probs + hard assignments

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
from jaxopt import LBFGS

SKIP_DE = False          # Set to True to skip DE warm-up and go straight to EM

# ────────────────────────────────────────────────────────────────
# Path & import setup
# ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "metacountregressor"))

from metacountregressor import main_hpc_lc_patch  # noqa: E402, F401

from metacountregressor.main_hpc import (  # noqa: E402
    experiment_washington,
    CountModel,
    build_base_index,
)
from metacountregressor.main_hpc_lc_patch import (  # noqa: E402
    ModelSpec,
    build_param_index,
    build_model_from_manual_spec,
    mixed_model_loglik,
    fit_em,
    _seed_classes_from_clusters,
)

jax.config.update("jax_enable_x64", True)


# ────────────────────────────────────────────────────────────────
#  DE warm-up for the LC objective
# ────────────────────────────────────────────────────────────────

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
) -> np.ndarray:
    """Population-based DE warm-up on the full latent-class log-likelihood.

    Creates a population around *init*, evaluates via ``jax.vmap``, and
    iteratively replaces the population with elite-mean + scaled noise.
    Returns the best params seen (or *init* if no improvement).
    """
    import jax

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
        """Best-effort evaluation — returns large_val on failure."""
        try:
            return float(objective(x, data, spec))
        except Exception:
            return large_val

    def _eval_pop(pop):
        try:
            vals = jax.vmap(objective, in_axes=(0, None, None))(pop, data, spec)
            vals = jnp.where(jnp.isfinite(vals), vals, large_val)
            return np.asarray(vals, dtype=float)
        except Exception:
            return np.array([_obj(row) for row in np.asarray(pop)], dtype=float)

    incumbent_obj = _obj(center_jnp)

    try:
        key = jax.random.PRNGKey(int(seed))
        key, sub = jax.random.split(key)
        pop = center_jnp + jax.random.normal(sub, (n_pop, n_params)) * span
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

            elite_idx = order[:n_elite]
            elite = pop[elite_idx]
            elite_mean = jnp.mean(elite, axis=0)
            elite_std = jnp.std(elite, axis=0)
            anneal = max(0.15, 0.92 ** (it + 1))
            step = jnp.maximum(elite_std, span * 0.05) * anneal

            key, sub = jax.random.split(key)
            noise = jax.random.normal(sub, (n_pop, n_params))
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
            return np.asarray(best_x, dtype=float)
        else:
            print(f"  DE warm-up:  no improvement (obj={incumbent_obj:.2f})")
            return center

    except Exception as exc:
        print(f"  DE warm-up:  failed ({exc}), returning init")
        return center


# ────────────────────────────────────────────────────────────────
# 1.  Load & prepare data
# ────────────────────────────────────────────────────────────────
_, df, _all_vars = experiment_washington()
print(f"Loaded: {len(df)} segments  |  {df['ID'].nunique()} unique IDs")
print(f"URB: {df['URB'].mean():.1%} urban  |  "
      f"FC distribution: {dict(df['FC'].value_counts().sort_index())}")
print(f"Mean FREQ(Y): {df['Y'].mean():.2f}  |  "
      f"Mean AADT: {df['AADT'].mean():.0f}")

# ────────────────────────────────────────────────────────────────
# 2.  Model specification
# ────────────────────────────────────────────────────────────────
FIXED_VARS = [
    "SPEED", "WIDTH", "CURVES", "MINRAD", "ACCESS", "GRADEBR",
    "FRICTION", "EXPOSE", "SINGLE", "TANGENT", "SLOPE",
    "GBRPM", "INTPM", "CPM", "MEDWIDTH", "INCLANES",
    "DOUBLE", "TRAIN", "PEAKHR",
]
MEMBERSHIP_VARS = ["URB"]

manual_spec: dict = {
    "fixed_terms":       FIXED_VARS,
    "rdm_terms":         [],
    "rdm_cor_terms":     [],
    "grouped_terms":     [],
    "hetro_in_means":    [],
    "zi_terms":          [],
    "membership_terms":  MEMBERSHIP_VARS,
    "group_id_col":      None,
    "dispersion":        1,     # 1 = NB2, 0 = Poisson
    "latent_classes":    2,
}

data, spec = build_model_from_manual_spec(
    df=df, manual_spec=manual_spec,
    id_col="ID", y_col="Y", offset_col="OFFSET",
    R=200,
)

C = spec.latent_classes
K_mem = spec.K_membership
base_spec = replace(spec, latent_classes=1)
K_base = build_base_index(base_spec)["total_params"]
gamma_size = (C - 1) * (K_mem + 1)

print(f"\nSpec: {spec.Kf} fixed + {K_mem} membership  |  "
      f"latent classes={C}  |  total params={C * K_base + gamma_size}")

# ────────────────────────────────────────────────────────────────
# 3.  Pre-fit: single-class NB  (warm-start for seeding)
# ────────────────────────────────────────────────────────────────
print("\n--- Phase 1: single-class NB warm-start ---")
model_1 = CountModel(base_spec, data)
result_1 = model_1.fit()
theta_1 = np.array(result_1.params)
ll_1 = -float(result_1.state.value)
n_obs = int(data["y"].shape[0])
k_1 = len(theta_1)
print(f"  Single-class LL = {ll_1:.2f}  |  "
      f"BIC = {k_1 * np.log(n_obs) - 2 * ll_1:.2f}")

# ────────────────────────────────────────────────────────────────
# 4.  Cluster-based seeding of per-class thetas
# ────────────────────────────────────────────────────────────────
print("\n--- Phase 2: cluster seeding ---")
rng = np.random.default_rng(42)
try:
    per_class_thetas = _seed_classes_from_clusters(
        theta_1, data, base_spec, C, K_base, rng,
    )
    theta_init = np.concatenate(per_class_thetas)
    print(f"  Cluster seeding OK  (class sizes from k-means)")
except Exception:
    print("  [warn] cluster seeding failed; falling back to jitter")
    theta_init = np.concatenate([
        theta_1 + rng.normal(0, 0.05, K_base) for _ in range(C)
    ])

gamma_init = np.zeros(gamma_size)
init_params = np.concatenate([theta_init, gamma_init])
init_ll = -float(mixed_model_loglik(jnp.array(init_params), data, spec))
print(f"  Initial LC-2 LL = {init_ll:.2f}")

# ────────────────────────────────────────────────────────────────
# 5.  DE warm-up: population search on full LC objective
# ────────────────────────────────────────────────────────────────
if not SKIP_DE:
    print("\n--- Phase 3: DE warm-up on full LC objective ---")
    params_de = de_warmup_lc(
        init_params,
        objective=mixed_model_loglik,
        data=data,
        spec=spec,
        maxiter=15,
        popsize=12,
        rel_span=1.5,
        abs_span=1.0,
        seed=42,
    )
    de_ll = -float(mixed_model_loglik(jnp.array(params_de), data, spec))
    print(f"  DE-best LC-2 LL = {de_ll:.2f}  "
          f"(delta from seed = {de_ll - init_ll:+.2f})")
    params_pre_em = params_de
else:
    print("\n--- Phase 3: DE warm-up  SKIPPED ---")
    params_pre_em = init_params

# ────────────────────────────────────────────────────────────────
# 6.  EM: refine class assignments + gamma
# ────────────────────────────────────────────────────────────────
print("\n--- Phase 4: EM ---")
params_em = fit_em(
    init_params=params_pre_em, data=data, spec=spec,
    max_iter=100, tol=1e-6, verbose=True,
)

# ────────────────────────────────────────────────────────────────
# 7.  MLE polish
# ────────────────────────────────────────────────────────────────
print("\n--- Phase 5: LBFGS polish ---")
polish = LBFGS(
    fun=lambda p: mixed_model_loglik(p, data, spec),
    maxiter=500,
)
result = polish.run(jnp.array(params_em))
params = np.array(result.params)
final_ll = -float(result.state.value)
k_lc = len(params)
bic_lc = k_lc * np.log(n_obs) - 2 * final_ll
bic_1 = k_1 * np.log(n_obs) - 2 * ll_1
print(f"  LC-2 LL = {final_ll:.2f}  |  BIC = {bic_lc:.2f}")
print(f"  delta-BIC vs single-class = {bic_lc - bic_1:+.2f}")
print(f"  (delta-BIC < 0 -> LC-2 preferred;  delta-BIC > 0 -> single-class sufficient)")

# ────────────────────────────────────────────────────────────────
# 8.  Posterior class probabilities
# ────────────────────────────────────────────────────────────────
N = data["y"].shape[0]
theta_all = params[:C * K_base].reshape(C, K_base)
gamma = params[C * K_base:].reshape(C - 1, K_mem + 1)

# Prior log-probabilities  pi_c(n) = softmax( Z_full @ gamma.T )
Xmem = np.array(data["Xmem"])
Z = np.mean(Xmem, axis=1)
Z_full = np.concatenate([np.ones((N, 1)), Z], axis=1)
logits_i = Z_full @ gamma.T
logits_full = np.concatenate([np.zeros((N, 1)), logits_i], axis=1)
log_pi = np.array(jax.nn.log_softmax(jnp.array(logits_full), axis=1))

# Per-class individual log-likelihoods
logL = np.zeros((N, C))
for c in range(C):
    ll_ind = mixed_model_loglik(jnp.array(theta_all[c]), data, base_spec, indivi=True)
    logL[:, c] = np.array(ll_ind)

# Full posterior  p(class=c | y, z, theta, gamma)
log_joint = logL + log_pi
max_log = log_joint.max(axis=1, keepdims=True)
posterior = np.exp(log_joint - max_log)
posterior /= posterior.sum(axis=1, keepdims=True)

hard_class = np.argmax(posterior, axis=1) + 1

# ────────────────────────────────────────────────────────────────
# 9.  Build output DataFrame
# ────────────────────────────────────────────────────────────────
df_out = df.copy()
df_out["class"] = hard_class
df_out["class_1_posterior"] = posterior[:, 0]
df_out["class_2_posterior"] = posterior[:, 1]
df_out["class_confidence"] = posterior.max(axis=1)

print(f"\n{'=' * 65}")
print("CLASS ASSIGNMENT SUMMARY")
print(f"{'=' * 65}")
print(f"  Class 1:  {np.mean(hard_class == 1):.1%}  "
      f"(n={np.sum(hard_class == 1)})")
print(f"  Class 2:  {np.mean(hard_class == 2):.1%}  "
      f"(n={np.sum(hard_class == 2)})")
print(f"  Mean posterior certainty: {df_out['class_confidence'].mean():.3f}")

print(f"\n  Class assignment vs URB (urban indicator):")
print(pd.crosstab(df_out["URB"], df_out["class"], margins=True).to_string())

print(f"\n  Class assignment vs FC (functional class):")
print(pd.crosstab(df_out["FC"], df_out["class"], margins=True).to_string())

# ────────────────────────────────────────────────────────────────
# 10.  Covariate profiles by class
# ────────────────────────────────────────────────────────────────
PROFILE_VARS = [
    "URB", "FC", "SPEED", "AADT", "WIDTH", "CURVES", "MINRAD",
    "ACCESS", "MEDWIDTH", "Y", "LENGTH", "SINGLE",
    "DOUBLE", "TRAIN", "GRADEBR", "TANGENT", "SLOPE",
    "FRICTION", "EXPOSE", "INCLANES", "GBRPM", "INTPM", "CPM",
]

print(f"\n{'=' * 65}")
print("COVARIATE PROFILES BY CLASS")
print(f"{'=' * 65}")
print(f"  {'Variable':<14} {'Class 1':>10}  {'Class 2':>10}  "
      f"{'Diff (1-2)':>11}  {'p-value'}")
print(f"  {'-' * 14} {'-' * 10}  {'-' * 10}  {'-' * 11}  {'-' * 8}")

from scipy import stats as scipy_stats  # noqa: E402

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

# ────────────────────────────────────────────────────────────────
# 11.  Per-class outcome coefficients
# ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("OUTCOME MODEL COEFFICIENTS  (NB2, by class)")
print(f"{'=' * 65}")

base_idx = build_base_index(base_spec)
fixed_names = list(spec.fixed_names)

for c_idx in range(C):
    theta_c = theta_all[c_idx]
    f_start, f_end = base_idx["fixed"]
    print(f"\n  --- CLASS {c_idx + 1} ---")
    print(f"  {'Parameter':<14} {'Estimate':>10}  {'exp(beta)':>10}")
    print(f"  {'-' * 14} {'-' * 10}  {'-' * 10}")
    for i, name in enumerate(fixed_names):
        val = theta_c[f_start + i]
        print(f"  {name:<14} {val:>10.4f}  {np.exp(val):>10.4f}")
    if base_spec.model == "nb":
        disp_idx = base_idx["dispersion"]
        disp = theta_c[disp_idx]
        print(f"  {'alpha (disp)':<14} {disp:>10.4f}")

# ────────────────────────────────────────────────────────────────
# 12.  Membership equation
# ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("MEMBERSHIP EQUATION  log[pi_2(n) / pi_1(n)] = g_0 + g_1 * URB")
print(f"{'=' * 65}")

mem_names = ["(intercept)"] + list(spec.membership_names)
for k, name in enumerate(mem_names):
    g_val = gamma[0, k]
    print(f"  {name:<14} ~ {g_val:+.4f}   ->  odds ratio = {np.exp(g_val):.4f}")

logits_marg = np.concatenate([np.zeros(1), gamma[:, 0]])
pi_marg = np.exp(logits_marg) / np.exp(logits_marg).sum()
print(f"\n  Marginal class shares (sample-average intercept):")
for c in range(C):
    print(f"    pi_{c + 1} = {pi_marg[c]:.4f}  ({pi_marg[c]:.1%})")

# Class probabilities by URB value
print(f"\n  Prior class probabilities by URB:")
for urb_val in [0, 1]:
    z_row = np.array([[1.0, float(urb_val)]])
    logits_row = z_row @ gamma.T
    logits_full_row = np.concatenate([np.zeros((1, 1)), logits_row], axis=1)
    pi_row = (np.exp(logits_full_row)
              / np.exp(logits_full_row).sum(axis=1, keepdims=True))
    print(f"    URB={urb_val}:  pi_1={pi_row[0, 0]:.4f}  pi_2={pi_row[0, 1]:.4f}")

# ────────────────────────────────────────────────────────────────
# 13.  Save output
# ────────────────────────────────────────────────────────────────
COLUMNS_TO_SAVE = [
    "ID", "class", "class_1_posterior", "class_2_posterior",
    "class_confidence",
] + PROFILE_VARS

out_path = os.path.join(REPO_ROOT, "lc_class_assignments.csv")
df_out[COLUMNS_TO_SAVE].to_csv(out_path, index=False)
print(f"\nClass assignments + profiles saved to  {out_path}")
print("Done.")
