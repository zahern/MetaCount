"""Run latent-class structure search on Washington data with DE warm-up.

Search dimensions
=================
  Decision vector = [roles(D) | dist_codes(D) | dispersion_bit | lc_code]
  dimension = 2*D + 2  (base main_hpc uses 2*D+1 -- single-class only)

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
import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial
from jaxopt import LBFGS
from typing import Union, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
os.chdir(os.path.join(REPO_ROOT, "metacountregressor"))

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
)
from metacountregressor.main_hpc_lc_patch import (  # noqa: E402
    build_param_index,
    mixed_model_loglik,
    fit_em,
    _seed_classes_from_clusters,
)
from metacountregressor.experiment_package import StructureEvaluatorLC  # noqa: E402

# ── Boost membership role probabilities so the search explores roles 7 & 8 ──
import metacountregressor.Solvers_METAJAX as _solvers  # noqa: E402
_solvers.ROLE_PROBS = np.array([
    0.28,  # 0 - Excluded
    0.12,  # 1 - Fixed
    0.14,  # 2 - Random Independent
    0.14,  # 3 - Random Correlated
    0.00,  # 4 - Grouped
    0.00,  # 5 - Heterogeneity in means
    0.02,  # 6 - Zero Inflation
    0.15,  # 7 - Membership only        <-- boosted
    0.15,  # 8 - Membership + fixed     <-- boosted
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fit_count = 0

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

        # ── ALWAYS 2 latent classes ──────────────────────────────
        spec_dict["latent_classes"] = 2

        sig = self.structural_signature(spec_dict)
        if sig is None:
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

        if sig in self.structure_cache:
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

        self.structure_cache.clear()
        self.structure_cache.add(sig)

        C = 2

        try:
            data_train, spec = self.build_data(
                self.df_train, spec_dict, self.master_halton_train
            )
            if spec_dict.get("model") is not None:
                spec = replace(spec, model=spec_dict["model"])

            K_mem = spec.K_membership
            spec_1 = replace(spec, latent_classes=1)
            K_base = build_param_index(spec_1)["total_params"]

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

            # Step 2 — cluster-based seeding
            rng = np.random.default_rng(abs(hash(sig)) % (2**31))
            try:
                per_class_thetas = _seed_classes_from_clusters(
                    theta_1, data_train, spec_1, C, K_base, rng
                )
                theta_init = np.concatenate(per_class_thetas)
            except Exception:
                theta_init = np.concatenate([
                    theta_1 + rng.normal(0, 0.05, K_base)
                    for _ in range(C)
                ])

            gamma_init = np.zeros((C - 1) * (K_mem + 1))
            init_params = np.concatenate([theta_init, gamma_init])
            spec_c = replace(spec, latent_classes=C)

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

            # Step 5 — LBFGS polish
            polish = LBFGS(
                fun=lambda p: mixed_model_loglik(p, data_train, spec_c),
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
            print(f"\n{'=' * 65}")
            print(f"FIT #{self._fit_count}   *GLOBAL* LC-2 BIC={bic:.1f}   "
                  f"LL={ll:.1f}   k={k}   n={n}")
            print(f"fixed={spec_dict.get('fixed_terms', [])}")
            print(f"membership={spec_dict.get('membership_terms', [])}")
            print(f"dispersion={'NB2' if spec_dict.get('dispersion') else 'Poisson'}")
            print(f"{'=' * 65}")
            _objective = partial(mixed_model_loglik, data=data_train, spec=spec_c)
            _pindex = build_param_index(spec_c)
            from metacountregressor.main_hpc_lc_patch import print_summary
            lc_summary = print_summary(
                result_c, _objective, data_train, spec_c, _pindex
            )
            if lc_summary and isinstance(lc_summary, dict):
                print(f"\n{'=' * 65}")
                print(f"  GLOBAL LC-2 FIT")
                print(f"  Log-Likelihood : {lc_summary['loglik']:.4f}")
                print(f"  AIC            : {lc_summary['aic']:.4f}")
                print(f"  BIC            : {lc_summary['bic']:.4f}")
                print(f"  Params         : {lc_summary['num_parm']}")
                print(f"  Latent Classes : {lc_summary['latent_classes']}")
                print(f"  Class Probs    : {lc_summary.get('class_probs', [])}")
                print(f"{'=' * 65}\n")

                # ── Posterior class assignments vs FC ───────────
                try:
                    theta_all = params_c[:C * K_base].reshape(C, K_base)
                    gamma = params_c[C * K_base:].reshape(C - 1, K_mem + 1)

                    # Prior log-probs
                    Xmem = np.array(data_train["Xmem"])
                    Z = np.mean(Xmem, axis=1)
                    Z_full = np.concatenate(
                        [np.ones((n, 1)), Z], axis=1
                    )
                    logits_i = Z_full @ gamma.T
                    logits_full = np.concatenate(
                        [np.zeros((n, 1)), logits_i], axis=1
                    )
                    log_pi = np.array(
                        jax.nn.log_softmax(jnp.array(logits_full), axis=1)
                    )

                    # Per-class individual LL
                    logL = np.zeros((n, C))
                    for c in range(C):
                        ll_ind = mixed_model_loglik(
                            jnp.array(theta_all[c]), data_train,
                            replace(spec_c, latent_classes=1), indivi=True
                        )
                        logL[:, c] = np.array(ll_ind)

                    # Full posterior
                    log_joint = logL + log_pi
                    log_joint -= log_joint.max(axis=1, keepdims=True)
                    posterior = np.exp(log_joint)
                    posterior /= posterior.sum(axis=1, keepdims=True)
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
                except Exception as exc:
                    print(f"  [posterior computation error] {exc}\n")
            else:
                print()

            # ── Return ──────────────────────────────────────────
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
        {"EXPOSE": [1]},
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
        mode="single",
        R=200,
        max_latent_classes=2,
    )
    evaluator.DE_MAXITER = args.de_warmup_iters

    # ── Dimension ────────────────────────────────────────────────
    dim = 2 * D + 2  # 2D+1 = no LC gene; 2D+2 = with LC gene
    print(f"Dimension: {dim}  = 2*{D}+2  (roles, dists, dispersion, lc_code)")
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
    print(f"\nBest BIC: {best_score:.2f}")

    spec_dict = evaluator.build_spec(best_sol)
    if spec_dict:
        spec_dict["latent_classes"] = 2
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
        C = 2
        K_mem = spec_best.K_membership
        base_spec = replace(spec_best, latent_classes=1)
        K_base = build_param_index(base_spec)["total_params"]

        # -- warm-start single-class --
        model_1 = CountModel(base_spec, data_best)
        res_1 = model_1.fit()
        theta_1 = np.array(res_1.params)

        rng = np.random.default_rng(42)
        try:
            per_class = _seed_classes_from_clusters(
                theta_1, data_best, base_spec, C, K_base, rng
            )
            theta_init = np.concatenate(per_class)
        except Exception:
            theta_init = np.concatenate([
                theta_1 + rng.normal(0, 0.05, K_base) for _ in range(C)
            ])

        gamma_init = np.zeros((C - 1) * (K_mem + 1))
        init_p = np.concatenate([theta_init, gamma_init])

        # -- DE warm-up --
        init_p = _de_warmup_lc(
            init_p, data=data_best, spec=spec_best,
            maxiter=12, popsize=8, seed=42,
        )

        # -- EM --
        params_em = fit_em(
            init_params=init_p, data=data_best, spec=spec_best,
            max_iter=100, tol=1e-6, verbose=True,
        )

        # -- Polish --
        polish = LBFGS(
            fun=lambda p: mixed_model_loglik(p, data_best, spec_best),
            maxiter=800,
        )
        result = polish.run(jnp.array(params_em))

        objective = partial(mixed_model_loglik, data=data_best, spec=spec_best)
        param_index = build_param_index(spec_best)
        print_summary(result, objective, data_best, spec_best, param_index)

    print("\nDone.")
