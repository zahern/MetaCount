"""Run latent-class structure search on Washington data with DE warm-up.

Search dimensions
=================
  Decision vector = [roles(D) | dist_codes(D) | dispersion_bit | lc_code | class_mask(D) | membership_mask(D)]
  dimension = 4*D + 2
  class_mask:   per-class outcome variable assignments (0=all, 1=class1, 2=class2, ...)
  membership_mask: per-class membership assignments  (0=all, 1=class2, 2=class3, ...)

  lc_code  ->  actual classes = lc_code % max_latent_classes + 1
  So with max_latent_classes=2:  lc_code=0 -> 1 class,  lc_code=1 -> 2 classes

Fitness pipeline (per candidate structure)
===========================================
  Single-class warm start
  -> cluster-based class seeding
  -> DE warm-up on full LC objective
  -> EM (≤30 iterations)
  -> LBFGS polish

Usage
======
  python run_lc_patch.py [--max_iter N] [--n_starts K]
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial
from jaxopt import LBFGS
from typing import Union, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
_pkg_dir = os.path.join(REPO_ROOT, "metacountregressor")
if os.path.isdir(_pkg_dir):
    os.chdir(_pkg_dir)

jax.config.update("jax_enable_x64", True)

# ────────────────────────────────────────────────────────────────
# Apply LC patches
# ────────────────────────────────────────────────────────────────
from metacountregressor import main_hpc_lc_patch  # noqa: E402, F401

from metacountregressor.main_hpc import (  # noqa: E402
    experiment_washington,
    build_base_index,
    populate_allowed_roles,
    populate_allowed_distributions,
    CountModel,
    run_with_oom_recovery,
)
from metacountregressor.main_hpc_lc_patch import (  # noqa: E402
    build_param_index,
    mixed_model_loglik,
    mixed_model_loglik_reg,
    fit_em,
    _seed_classes_from_clusters,
    unpack_lc_params,
    compute_lc_posteriors,
)
from metacountregressor.experiment_package import StructureEvaluatorLC  # noqa: E402

# ── Boost membership role probabilities so the search explores roles 7 & 8 ──
#    Membership covariates are crucial for class separation; weight them heavily.
import metacountregressor.Solvers_METAJAX as _solvers  # noqa: E402
_solvers.ROLE_PROBS = np.array([
    0.15,  # 0 - Excluded
    0.08,  # 1 - Fixed
    0.08,  # 2 - Random Independent
    0.08,  # 3 - Random Correlated
    0.00,  # 4 - Grouped
    0.00,  # 5 - Heterogeneity in means
    0.02,  # 6 - Zero Inflation
    0.24,  # 7 - Membership only        <-- strongly boosted
    0.35,  # 8 - Membership + fixed     <-- strongly boosted
], dtype=float)
_solvers.ROLE_PROBS /= _solvers.ROLE_PROBS.sum()


# ────────────────────────────────────────────────────────────────
# DE warm-up helper
# ────────────────────────────────────────────────────────────────

def _de_warmup_lc(
    center: np.ndarray,
    data: dict,
    spec,
    *,
    maxiter: int = 12,
    popsize: int = 8,
    rel_span: float = 1.5,
    abs_span: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """Population-based DE warm-up on the full latent-class log-likelihood.

    Returns the best params found, or *center* if no improvement.
    """
    center = np.asarray(center, dtype=float).reshape(-1)
    center = np.where(np.isfinite(center), center, 0.0)
    center_jnp = jnp.array(center)
    n_params = len(center)
    n_pop = max(8, int(popsize) * 2)
    n_elite = max(2, n_pop // 4)
    large_val = 1e20

    span = np.maximum(float(abs_span), np.abs(center) * float(rel_span))
    # Hard clamp: DE cannot explore outside [-MAX_COEF, +MAX_COEF]
    max_c = 20.0
    lower = jnp.maximum(center_jnp - span, -max_c)
    upper = jnp.minimum(center_jnp + span, +max_c)

    def _eval_pop(pop):
        try:
            vals = jax.vmap(mixed_model_loglik, in_axes=(0, None, None))(
                pop, data, spec
            )
            vals = jnp.where(jnp.isfinite(vals), vals, large_val)
            return np.asarray(vals, dtype=float)
        except Exception:
            return np.array([
                _safe_obj(row) for row in np.asarray(pop)
            ], dtype=float)

    def _safe_obj(x):
        try:
            val = float(mixed_model_loglik(x, data, spec))
            return val if np.isfinite(val) else large_val
        except Exception:
            return large_val

    incumbent_obj = _safe_obj(center_jnp)

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
            if verbose:
                print(f"    DE warm-up: {incumbent_obj:.2f} -> {best_obj:.2f} "
                      f"(delta={delta:.2f})")
            return np.asarray(best_x, dtype=float)
        else:
            if verbose:
                print(f"    DE warm-up: no improvement "
                      f"(obj={incumbent_obj:.2f})")
            return center

    except Exception as exc:
        if verbose:
            print(f"    DE warm-up: failed ({exc}), returning init")
        return center


# ────────────────────────────────────────────────────────────────
# FC -> truth-class mapping (matches the ground truth to the number
# of latent classes being searched).
#
# The FC column is the observed roadway functional class.  When the
# search fits C latent classes we want an apple-to-apples recovery
# comparison, so the FC ground truth is re-coded to exactly C classes:
#   * more distinct FCs than C  -> merge the most-similar FCs first
#   * fewer distinct FCs than C -> keep every present FC
# Similarity is measured on standardised covariate profiles so the
# merged classes are the ones that look most alike in the data.
# ────────────────────────────────────────────────────────────────

_FC_PROFILE_COLS = ["URB", "SPEED", "AADT", "CURVES", "MINRAD",
                    "ACCESS", "FRICTION"]


def _fc_truth_merge(df, C):
    """Map observed functional classes to ``C`` truth labels.

    Returns ``(mapping, truth)`` where ``mapping`` is a dict
    ``{FC_value: truth_label}`` and ``truth`` is the per-row label
    array aligned with ``df``.  Returns ``(None, None)`` if there are
    no usable FC values.
    """
    fc_vals = sorted(int(v) for v in df["FC"].dropna().unique())
    if not fc_vals:
        return None, None
    if C is None or C < 1:
        C = len(fc_vals)

    cols = [c for c in _FC_PROFILE_COLS if c in df.columns]
    if cols:
        prof_df = df[df["FC"].isin(fc_vals)].groupby("FC")[cols].mean()
        prof = prof_df.apply(
            lambda s: (s - s.mean()) / s.std() if s.std() > 1e-12 else s * 0.0,
            axis=0,
        ).reindex(fc_vals).fillna(0.0).to_numpy()
    else:
        prof = np.zeros((len(fc_vals), 1))

    # Agglomerative merging of the closest FC pairs until C groups remain.
    groups = [[i] for i in range(len(fc_vals))]      # member profile-row idxs
    vecs   = [prof[i].copy() for i in range(len(fc_vals))]
    while len(groups) > C:
        best = min(
            ((np.linalg.norm(vecs[i] - vecs[j]), i, j)
             for i in range(len(groups))
             for j in range(i + 1, len(groups))),
            key=lambda t: t[0],
        )
        _, i, j = best
        members = groups[i] + groups[j]
        groups[i] = members
        vecs[i] = prof[members].mean(axis=0)
        del groups[j]
        del vecs[j]

    mapping = {}
    for label, group in enumerate(groups, start=1):
        for row_idx in group:
            mapping[fc_vals[row_idx]] = label
    truth = df["FC"].map(mapping).fillna(-1).astype(int).to_numpy()
    return mapping, truth


def _fc_recovery_stats(hard_class, truth, C):
    """Best RMSE / accuracy over all label permutations (1..C)."""
    hard_class = np.asarray(hard_class, dtype=int)
    truth = np.asarray(truth, dtype=int)
    valid = truth >= 1
    if not valid.any():
        return float("nan"), float("nan")
    hc = hard_class[valid]
    tr = truth[valid]
    import itertools
    best_rmse, best_acc = None, 0.0
    for perm in itertools.permutations(range(1, C + 1)):
        mapped = np.array([perm[v - 1] for v in tr])
        rmse = float(np.sqrt(np.mean((hc - mapped) ** 2)))
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_acc = float(np.mean(hc == mapped))
    return best_rmse, best_acc


# ────────────────────────────────────────────────────────────────
# Evaluator with DE warm-up injected into the LC fitness pipeline
# ────────────────────────────────────────────────────────────────

class StructureEvaluatorLC_DE(StructureEvaluatorLC):
    """LC evaluator that always fits 2-class models with DE warm-up + prints."""

    DE_MAXITER: int = 12
    DE_POPSIZE: int = 8
    DE_REL_SPAN: float = 1.5
    DE_ABS_SPAN: float = 1.0
    MAX_ABS_COEF: float = 30.0       # reject fits with |coef| above this
    MIN_DISPERSION: float = 0.01     # dispersion must be positive
    MIN_CLASS_PROP: float = 0.15     # base value; EM engine scales internally by C

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fit_count = 0
        self._fit_trace = []  # (fit_number, BIC, LL, n_params, n_obs)

    def _coefs_valid(self, params: np.ndarray) -> bool:
        """Check no coefficient is absurdly large and dispersion > 0."""
        return bool(np.all(np.abs(params) < self.MAX_ABS_COEF))

    def fitness(self, decision) -> Union[float, np.ndarray]:

        key = tuple(decision.tolist())
        if key in self.cache:
            return self.cache[key]

        spec_dict = self.build_spec(decision)
        if spec_dict is None:
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

        # ── Use LC code from decision vector (no hardcoded class count) ──
        spec_dict["latent_classes"] = spec_dict.get("latent_classes", 2)

        sig = self.structural_signature(spec_dict)
        if sig is None:
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

        if sig in self.structure_cache:
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

        self.structure_cache.clear()
        self.structure_cache.add(sig)

        C = int(spec_dict.get("latent_classes", 2))

        try:
            data_train, spec = self.build_data(
                self.df_train, spec_dict, self.master_halton_train
            )
            if spec_dict.get("model") is not None:
                spec = replace(spec, model=spec_dict["model"])

            K_mem = spec.K_membership
            spec_1 = replace(spec, latent_classes=1)
            K_base_0 = build_param_index(spec_1)["total_params"]

            # Step 1 — fit single-class warm-start
            model_1 = CountModel(spec_1, data_train)
            result_1 = model_1.fit()
            theta_1 = np.array(result_1.params)
            # Bail early if single-class fit already exploded
            if not self._coefs_valid(theta_1):
                self._fit_count += 1
                print(f"  fit #{self._fit_count:4d}  SKIPPED: "
                      f"single-class coefficients out of bounds "
                      f"(max |coef| = {np.max(np.abs(theta_1)):.1f})")
                return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

            spec_c = replace(spec, latent_classes=C,
                             min_class_proportion=self.MIN_CLASS_PROP,
                             l2_penalty=0.1)
            pindex_c = build_param_index(spec_c)
            _class_K_base = list(pindex_c.get("class_K_base", [K_base_0] * C))

            # Step 2 — cluster-based seeding
            rng = np.random.default_rng(abs(hash(sig)) % (2**31))
            try:
                per_class_thetas = _seed_classes_from_clusters(
                    theta_1, data_train, spec_1, C, K_base_0, rng,
                    class_K_base=_class_K_base,
                )
                theta_init = np.concatenate(per_class_thetas)
            except Exception:
                theta_init = np.concatenate([
                    theta_1[:k] + rng.normal(0, 0.05, k)
                    if len(theta_1) >= k
                    else np.pad(theta_1, (0, k - len(theta_1))) + rng.normal(0, 0.05, k)
                    for k in _class_K_base
                ])

            # ── Per-class gamma sizes from class_membership (if available) ──
            if hasattr(spec_c, 'class_membership_idx') and spec_c.class_membership_idx:
                per_class_gamma_sizes = [len(idx) + 1 for idx in spec_c.class_membership_idx]
            else:
                per_class_gamma_sizes = [K_mem + 1] * (C - 1) if C > 1 else []
            gamma_total = sum(per_class_gamma_sizes)
            gamma_init = np.zeros(gamma_total)
            init_params = np.concatenate([theta_init, gamma_init])

            # Step 3 — DE warm-up
            seed_de = abs(hash(sig) * 3 + 1) % (2**31)
            init_params = _de_warmup_lc(
                init_params,
                data=data_train,
                spec=spec_c,
                maxiter=self.DE_MAXITER,
                popsize=self.DE_POPSIZE,
                rel_span=self.DE_REL_SPAN,
                abs_span=self.DE_ABS_SPAN,
                seed=seed_de,
                verbose=False,
            )

            # Step 4 — EM
            try:
                params_em = fit_em(
                    init_params=init_params,
                    data=data_train,
                    spec=spec_c,
                    max_iter=30,
                    tol=1e-4,
                    verbose=False,
                )
            except Exception:
                params_em = init_params

            # Step 5 — LBFGS polish (with L2 penalty)
            polish = LBFGS(
                fun=lambda p: mixed_model_loglik_reg(p, data_train, spec_c),
                maxiter=500,
            )
            result_c = polish.run(jnp.array(params_em))
            params_c = np.array(result_c.params)
            ll = -float(result_c.state.value)
            n = data_train["y"].shape[0]
            k = len(params_c)
            bic = k * np.log(n) - 2.0 * ll

            # ── Validity check: reject exploded coefficients ──────
            if not (np.isfinite(ll) and self._coefs_valid(params_c)):
                self._fit_count += 1
                print(f"  fit #{self._fit_count:4d}  REJECTED: "
                      f"coefficients out of bounds "
                      f"(max |coef| = {np.max(np.abs(params_c)):.1f})")
                return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

            # ── Class balance check: penalise extreme imbalance ──
            try:
                posterior, _, _ = compute_lc_posteriors(params_c, data_train, spec_c)
                class_props = np.asarray(posterior).mean(axis=0)  # (C,) mean posterior weight
                min_prop = float(class_props.min())
                # Severe imbalance penalty: if any class < MIN_CLASS_PROP,
                # add a penalty to BIC proportional to the shortfall.
                below = max(0.0, self.MIN_CLASS_PROP - min_prop)
                bic_penalty = below * n * 10.0  # ~10 BIC points per 1% below 20%
                bic += bic_penalty
            except Exception:
                min_prop = 0.0
                bic_penalty = 0.0
                class_props = np.full(C, 1.0 / max(C, 1))

            self._last_fit_cache = {
                "params": params_c,
                "spec":   spec_c,
                "data":   data_train,
                "bic":    float(bic),
            }
            self._update_role_memory_from_fit(
                params   = params_c,
                spec     = spec_c,
                decision = decision,
            )

            class _Model:
                params = params_c
            model = _Model()

            # ── Print full model summary ──────────────────────────
            self._fit_count += 1
            class_prop_str = " / ".join(f"{p:.2f}" for p in class_props) if min_prop > 0 else "?"
            print(f"\n{'=' * 65}")
            print(f"FIT #{self._fit_count}   *GLOBAL* LC-2 BIC={bic:.1f}   "
                  f"LL={ll:.1f}   k={k}   n={n}")
            print(f"fixed={spec_dict.get('fixed_terms', [])}")
            print(f"membership={spec_dict.get('membership_terms', [])}")
            print(f"dispersion={'NB2' if spec_dict.get('dispersion') else 'Poisson'}")
            print(f"class props={{{class_prop_str}}}  "
                  f"min={min_prop:.3f}  balance_penalty={bic_penalty:+.1f}")
            print(f"{'=' * 65}")
            _objective = partial(mixed_model_loglik, data=data_train, spec=spec_c)
            _pindex = build_param_index(spec_c)
            from metacountregressor.main_hpc_lc_patch import print_summary
            lc_summary = print_summary(
                result_c, _objective, data_train, spec_c, _pindex
            )
            if lc_summary and isinstance(lc_summary, dict):
                print(f"\n{'=' * 65}")
                print(f"  GLOBAL LC-{C} FIT")
                print(f"  Log-Likelihood : {lc_summary.get('loglik', float('nan')):.4f}")
                print(f"  AIC            : {lc_summary.get('aic', float('nan')):.4f}")
                print(f"  BIC            : {lc_summary.get('bic', float('nan')):.4f}")
                print(f"  Params         : {lc_summary.get('num_parm', '?')}")
                print(f"  Latent Classes : {lc_summary.get('latent_classes', C)}")
                print(f"  Class Probs    : {lc_summary.get('class_probs', [])}")
                print(f"{'=' * 65}\n")

                # ── Posterior class assignments vs FC ───────────
                import pandas as pd
                try:
                    posterior, log_pi, logL = compute_lc_posteriors(
                        params_c, data_train, spec_c
                    )
                    hard_class = np.argmax(posterior, axis=1) + 1

                    # Align with original df
                    ids = np.array(data_train.get(
                        "ids",
                        self.df_train[[self.id_col]]
                        .drop_duplicates()[self.id_col].to_numpy()
                    )).ravel()
                    df_posterior = pd.DataFrame({
                        self.id_col: ids,
                        "assigned_class": hard_class,
                        "class_1_post": posterior[:, 0],
                        "class_2_post": posterior[:, 1],
                    })
                    df_join = self.df_train.merge(
                        df_posterior, on=self.id_col, how="left"
                    )

                    print(f"  CLASS ASSIGNMENT vs FUNCTIONAL CLASS (FC)")
                    ct_fc = pd.crosstab(
                        df_join["FC"], df_join["assigned_class"],
                        margins=True
                    )
                    print(ct_fc.to_string())
                    print()

                    print(f"  CLASS ASSIGNMENT vs URBAN (URB)")
                    ct_urb = pd.crosstab(
                        df_join["URB"], df_join["assigned_class"],
                        margins=True
                    )
                    print(ct_urb.to_string())
                    print()

                    # Mean covariate values per assigned class
                    print(f"  MEAN COVARIATES BY ASSIGNED CLASS")
                    profile_cols = ["URB", "SPEED", "AADT",
                                    "CURVES", "MINRAD", "ACCESS",
                                    "FRICTION", "Y"]
                    class_means = df_join.groupby(
                        "assigned_class"
                    )[profile_cols].mean()
                    print(class_means.to_string())
                    print()

                    # ── FC classification RMSE ─────────────────
                    # Re-code the observed FC ground truth to C classes
                    # (merging the most-similar FCs), then compare against
                    # the latent-class assignments under the best label
                    # permutation.
                    fc_mapping, fc_truth = _fc_truth_merge(df_join, C)
                    if fc_mapping is not None:
                        fc_rmse, fc_acc = _fc_recovery_stats(
                            hard_class, fc_truth, C
                        )
                        print(f"  FC-LC RECOVERY  RMSE={fc_rmse:.4f}  "
                              f"Accuracy={fc_acc:.1%}")
                        merged_desc = ", ".join(
                            f"{{FC {','.join(str(v) for v in
                            sorted(k for k, lbl in fc_mapping.items()
                                   if lbl == label))}}}->{label}"
                            for label in sorted(set(fc_mapping.values()))
                        )
                        print(f"  FC truth re-code: {merged_desc}")
                    print()
                except Exception as exc:
                    import traceback
                    print(f"  [posterior computation error] {exc}\n")
                    traceback.print_exc()
            else:
                print()

            # ── Return ──────────────────────────────────────────
            # Record trace for convergence plot
            self._fit_trace.append({
                "fit": self._fit_count,
                "bic": float(bic),
                "ll": float(ll),
                "n_params": int(k),
                "n_obs": int(n),
                "fixed": spec_dict.get("fixed_terms", []),
                "membership": spec_dict.get("membership_terms", []),
                "dispersion": spec_dict.get("dispersion", 0),
            })

            if self.mode == "single":
                value = float(bic)
                self.cache[key] = value
                return value

            data_test, _ = self.build_data(
                self.df_test, spec_dict, self.master_halton_test
            )
            model_test = CountModel(spec, data_test)
            model_test.params = np.array(model.params)
            preds = model_test.predict()
            y_true = np.array(data_test["y"]).squeeze()
            rmse = np.sqrt(np.mean((preds - y_true) ** 2))

            value = np.array([float(bic), float(rmse)])
            self.cache[key] = value
            return value

        except Exception as e:
            print(f"  [fitness error] {e}")
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="sa",
                        choices=["sa", "hs", "de"],
                        help="Metaheuristic: sa, hs (harmony), de")
    parser.add_argument("--max_iter", type=int, default=200,
                        help="Search iterations / generations")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pop_size", type=int, default=20,
                        help="Population size (NSGA2 mode only)")
    parser.add_argument("--de_warmup_iters", type=int, default=12,
                        help="DE warm-up iterations inside each fitness eval")
    parser.add_argument("--verbose_freq", type=int, default=10,
                        help="Print search progress every N iterations")
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────
    _, df, all_vars = experiment_washington()
    D = len(all_vars)
    print(f"Data:  {len(df)} segments  |  {D} variables")
    print(f"URB:   {df['URB'].mean():.1%} urban   |  "
          f"mean FREQ = {df['Y'].mean():.1f}")

    # ── Build evaluator ──────────────────────────────────────────
    allowed_roles = populate_allowed_roles(
        all_vars,
        {"EXPOSE": [1], "FC": [0], "URB": [0]},  # FC & URB fully excluded — recover via latent classes
        default_roles=[0, 1, 2, 3, 5, 7, 8],  # includes membership roles
    )
    allowed_dists = populate_allowed_distributions(all_vars, None)

    evaluator = StructureEvaluatorLC_DE(
        df=df,
        id_col="ID",
        y_col="Y",
        offset_col="OFFSET",
        all_variables=all_vars,
        allowed_roles=allowed_roles,
        allowed_distributions=allowed_dists,
        group_id_col="FC",
        mode="multi",
        R=200,
        max_latent_classes=2,
    )
    evaluator.DE_MAXITER = args.de_warmup_iters

    # ── Dimension ────────────────────────────────────────────────
    dim = 4 * D + 2  # 4D+2 = roles(D) + dists(D) + disp(1) + lc(1) + class_mask(D) + membership_mask(D)
    print(f"Dimension: {dim}  = 4*{D}+2  (roles, dists, dispersion, lc_code, class_mask, membership_mask)")
    print(f"max_latent_classes = {evaluator.max_latent_classes}")

    # ── Run search ───────────────────────────────────────────────
    start = time.time()

    if args.algo == "sa":
        from metacountregressor.Solvers_METAJAX import (  # noqa: E402
            MultiStartSA,
        )
        solver = MultiStartSA(
            evaluator=evaluator,
            dimension=dim,
            n_starts=1,
            n_jobs=args.n_jobs,
            max_iter=args.max_iter,
            mutation_rate=0.5,
            step_size=2,
            alpha=0.99,
        )
        print(f"\nSearch: SA  |  max_iter={args.max_iter}  |  "
              f"DE warm-up iters={args.de_warmup_iters}")
        print("-" * 60)
        solutions, scores = solver.optimize()
        print("-" * 60)

    elif args.algo in ("hs", "de"):
        from metacountregressor.Solvers_METAJAX import (  # noqa: E402
            NSGA2Engine,
            DynamicHarmony,
            AdaptiveDE,
        )
        if args.algo == "hs":
            operator = DynamicHarmony(
                hmcr=0.9, par_min=0.05, par_max=0.8,
                bw_min=1, bw_max=4,
            )
        else:
            operator = AdaptiveDE(F=0.5, CR=0.7)

        engine = NSGA2Engine(
            evaluator=evaluator,
            operator=operator,
            dimension=dim,
            pop_size=args.pop_size,
            max_iter=args.max_iter,
            n_jobs=args.n_jobs,
        )
        print(f"\nSearch: NSGA2+{args.algo.upper()}  |  "
              f"max_iter={args.max_iter}  |  pop={args.pop_size}  |  "
              f"DE warm-up iters={args.de_warmup_iters}")
        solutions, scores = engine.optimize()

    elapsed = time.time() - start
    print(f"\nSearch finished in {elapsed:.0f}s")
    print(f"Evaluated {len(solutions)} solutions")

    # ── Best solution ────────────────────────────────────────────
    if scores.ndim == 1:
        best_idx = np.argmin(scores)
    else:
        best_idx = np.argmin(scores[:, 0])

    best_sol = solutions[best_idx]
    best_score = float(scores[best_idx]) if scores.ndim == 1 else float(scores[best_idx, 0])
    best_rmse = None if scores.ndim == 1 else float(scores[best_idx, 1])
    print(f"\nBest BIC: {best_score:.2f}")
    if best_rmse is not None:
        print(f"Best RMSE (test): {best_rmse:.4f}")

    # ── Save & plot SA fit trace ─────────────────────────────────
    nclass = int(getattr(evaluator, "max_latent_classes", 2))
    out_dir = os.path.join(REPO_ROOT, f"results_{nclass}class")
    os.makedirs(out_dir, exist_ok=True)

    if evaluator._fit_trace:
        import csv as _csv
        import json as _json
        trace_csv = os.path.join(out_dir, "lc_search_trace.csv")
        os.makedirs(os.path.dirname(trace_csv), exist_ok=True)
        with open(trace_csv, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=evaluator._fit_trace[0].keys())
            writer.writeheader()
            for row in evaluator._fit_trace:
                writer.writerow({
                    k: _json.dumps(v) if isinstance(v, list) else v
                    for k, v in row.items()
                })
        print(f"Search trace saved to:  {trace_csv}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt

            bics = [r["bic"] for r in evaluator._fit_trace]
            fits = [r["fit"] for r in evaluator._fit_trace]
            best_so_far = np.minimum.accumulate(bics)

            fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(14, 5))

            ax1.scatter(fits, bics, s=10, alpha=0.6, color="steelblue")
            ax1.plot(fits, best_so_far, "r-", linewidth=1.5, label="Best so far")
            ax1.set_xlabel("Fit evaluation #")
            ax1.set_ylabel("BIC (lower = better)")
            ax1.set_title(f"SA Structure Search: BIC per Evaluation\n"
                          f"({nclass}-class LC-NB2, DE warm-up, best={best_score:.1f})")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Distribution of BIC values
            finite_bics = [b for b in bics if b < 1e11]
            ax2.hist(finite_bics, bins=min(30, len(finite_bics)),
                     color="steelblue", edgecolor="white", alpha=0.8)
            ax2.axvline(best_score, color="red", linestyle="--",
                        linewidth=1.5, label=f"Best = {best_score:.1f}")
            ax2.set_xlabel("BIC")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Distribution of BIC Values")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            fig.suptitle("SA Structure Search Convergence\n"
                         f"({nclass}-class LC-NB2, FC held out)",
                         fontsize=11, fontweight="bold")
            _plt.tight_layout()
            trace_png = os.path.join(out_dir, "lc_search_trace.png")
            _plt.savefig(trace_png, dpi=150, bbox_inches="tight")
            _plt.close()
            print(f"Search trace plot saved to:  {trace_png}")
        except Exception as exc:
            print(f"  [warn] Search trace plot failed: {exc}")

        try:
            import json as _jcsv
            bics_ser = [r["bic"] for r in evaluator._fit_trace]
            fits_ser = [r["fit"] for r in evaluator._fit_trace]
            best_sofar = list(map(float, np.minimum.accumulate(bics_ser)))
            conv_path = os.path.join(out_dir, "convergence.json")
            with open(conv_path, "w") as fc:
                _jcsv.dump({
                    "latent_classes": nclass,
                    "n_evaluations": len(evaluator._fit_trace),
                    "best_bic": best_score,
                    "best_iteration": int(np.argmin(best_sofar)),
                    "best_so_far_bic_series": best_sofar,
                    "bic_series": [float(b) for b in bics_ser],
                    "fit_series": [float(b) for b in fits_ser],
                }, fc, indent=2)
            print(f"Convergence summary saved to:  {conv_path}")
        except Exception as exc:
            print(f"  [warn] Convergence summary failed: {exc}")

    spec_dict = evaluator.build_spec(best_sol)
    if spec_dict:
        spec_dict["latent_classes"] = spec_dict.get("latent_classes", 2)
        # ── Save best spec for Phase 2 ─────────────────────────────
        import json as _json
        spec_path = os.path.join(out_dir, "best_model_spec.json")
        os.makedirs(os.path.dirname(spec_path), exist_ok=True)
        with open(spec_path, "w") as fp:
            _json.dump(spec_dict, fp, indent=2, default=str)
        print(f"Best model spec saved to:  {spec_path}")
        print(f"  dispersion:      {'NB2' if spec_dict.get('dispersion') else 'Poisson'}")
        print(f"  fixed_terms:     {spec_dict.get('fixed_terms', [])}")
        print(f"  rdm_terms:       {spec_dict.get('rdm_terms', [])}")
        print(f"  rdm_cor_terms:   {spec_dict.get('rdm_cor_terms', [])}")
        print(f"  membership_terms:{spec_dict.get('membership_terms', [])}")
        print(f"  zi_terms:        {spec_dict.get('zi_terms', [])}")

    # ── Refit best with full EM + print summary ─────────────────
    print(f"\n{'=' * 65}")
    print("REFIT BEST MODEL WITH FULL EM + SUMMARY")
    print(f"{'=' * 65}")
    if spec_dict:
        from metacountregressor.main_hpc_lc_patch import (  # noqa: E402
            build_model_from_manual_spec,
            print_summary,
        )
        data_best, spec_best = build_model_from_manual_spec(
            df=df,
            manual_spec=spec_dict,
            id_col="ID",
            y_col="Y",
            offset_col="OFFSET",
            R=200,
        )
        C = int(spec_dict.get("latent_classes", 2))
        spec_best = replace(spec_best, min_class_proportion=0.15)
        K_mem = spec_best.K_membership
        base_spec = replace(spec_best, latent_classes=1)
        pindex = build_param_index(spec_best)
        class_K_base = list(pindex.get("class_K_base", [build_base_index(base_spec)["total_params"]] * C))
        K_base_0 = class_K_base[0]

        # -- warm-start single-class --
        model_1 = CountModel(base_spec, data_best)
        res_1 = run_with_oom_recovery(
            model_1.fit, label="REFIT single-class warm-start fit"
        )
        theta_1 = np.array(res_1.params)

        rng = np.random.default_rng(42)
        try:
            per_class = _seed_classes_from_clusters(
                theta_1, data_best, base_spec, C, K_base_0, rng,
                class_K_base=class_K_base,
            )
            theta_init = np.concatenate(per_class)
        except Exception:
            theta_init = np.concatenate([
                theta_1[:class_K_base[c]]
                if len(theta_1) >= class_K_base[c]
                else np.pad(theta_1, (0, class_K_base[c] - len(theta_1)))
                + rng.normal(0, 0.05, class_K_base[c])
                for c in range(C)
            ])

        gamma_init = np.zeros((C - 1) * (K_mem + 1))
        init_p = np.concatenate([theta_init, gamma_init])

        # -- DE warm-up --
        init_p = run_with_oom_recovery(
            _de_warmup_lc, init_p, data=data_best, spec=spec_best,
            maxiter=12, popsize=8, seed=42,
            label="REFIT DE warm-up",
        )

        # -- EM --
        params_em = run_with_oom_recovery(
            fit_em,
            init_params=init_p, data=data_best, spec=spec_best,
            max_iter=100, tol=1e-6, verbose=True,
            label="REFIT EM",
        )

        # -- Polish --
        polish = LBFGS(
            fun=lambda p: mixed_model_loglik(p, data_best, spec_best),
            maxiter=800,
        )
        result = run_with_oom_recovery(
            polish.run, jnp.array(params_em), label="REFIT LBFGS polish"
        )

        objective = partial(mixed_model_loglik, data=data_best, spec=spec_best)
        param_index = build_param_index(spec_best)
        print_summary(result, objective, data_best, spec_best, param_index)

        # ── Test-set validation ────────────────────────────────
        if evaluator.df_test is not None:
            data_test_v, spec_test = build_model_from_manual_spec(
                df=evaluator.df_test,
                manual_spec=spec_dict,
                id_col="ID", y_col="Y", offset_col="OFFSET",
                R=200,
            )
            model_test = CountModel(spec_test, data_test_v)
            model_test.params = np.array(result.params)
            preds = model_test.predict()
            y_true = np.array(data_test_v["y"]).squeeze()
            rmse_v = np.sqrt(np.mean((preds - y_true) ** 2))
            mae_v = np.mean(np.abs(preds - y_true))
            print(f"\nTEST-SET VALIDATION  RMSE={rmse_v:.4f}  MAE={mae_v:.4f}")

            # ── FC classification RMSE on test set ──────────
            params_all = np.array(result.params)
            spec_c_test = replace(spec_test, latent_classes=C)
            posterior_test, _, _ = compute_lc_posteriors(
                params_all, data_test_v, spec_c_test
            )
            hard_test = np.argmax(posterior_test, axis=1) + 1

            test_ids = np.array(data_test_v.get(
                "ids", evaluator.df_test[["ID"]]
                .drop_duplicates()["ID"].to_numpy()
            )).ravel()
            fc_mapping_test, _ = _fc_truth_merge(evaluator.df_test, C)
            if fc_mapping_test is not None:
                _id_to_fc = dict(zip(evaluator.df_test["ID"],
                                     evaluator.df_test["FC"]))
                fc_by_id = np.array([_id_to_fc.get(int(i), np.nan)
                                     for i in test_ids])
                fc_truth_test = np.array([
                    fc_mapping_test.get(int(v), -1) if pd.notna(v) else -1
                    for v in fc_by_id
                ], dtype=int)
                fc_rmse_v, _acc = _fc_recovery_stats(
                    hard_test, fc_truth_test, C
                )
                print(f"  FC-LC RECOVERY (test)  RMSE={fc_rmse_v:.4f}")

    print("\nDone.")
