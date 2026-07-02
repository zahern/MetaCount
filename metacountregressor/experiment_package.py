# =======================================================================
# experiment_package.py
# =======================================================================
#
# IMPORT ORDER (critical):
#   import main_hpc_lc_patch          # 1. patches main_hpc in place
#   from experiment_package import ExperimentBuilder   # 2. uses patched hpc
#
# ROLE SCHEME (complete)
# ──────────────────────
#   0  Excluded
#   1  Fixed (constant coefficient, same across sites)
#   2  Random Independent  (individual-level draws, independent)
#   3  Random Correlated   (individual-level draws, joint covariance)
#   4  Grouped             (group-level draws, shared within group)
#   5  Heterogeneity       (modifies means of random effects)
#   6  Zero Inflation      (enters class-prob of zero-excess)
#   7  Membership only     (enters class-membership eq. ONLY; role in outcome = 0)
#   8  Membership + fixed  (enters class-membership eq. AND outcome eq. as fixed)
#
# Roles 7 and 8 are ignored (treated as 0 / 1 respectively) when the
# latent-class code in the decision vector resolves to 1 class.
#
# DECISION VECTOR LAYOUT (dimension = 2·D + 2)
# ─────────────────────────────────────────────
#   [roles(D) | dist_codes(D) | dispersion_bit | latent_class_code]
#
# MEMBERSHIP LOGIC
# ─────────────────
# Role 7 variable:
#   → appears in spec["membership_terms"]
#   → does NOT appear in any outcome list
#   → its gamma coefficient lets individual-level z-values shift class
#     probability
#
# Role 8 variable:
#   → appears in spec["membership_terms"]  (class-prob equation)
#   → ALSO appears in spec["fixed_terms"]  (outcome equation)
#   → because the model has C classes, each class automatically gets its
#     own fixed beta for this variable (class-specific outcome + membership)
#
# =======================================================================

from __future__ import annotations

# ── Apply patches FIRST ─────────────────────────────────────────────────
try:
    from . import main_hpc_lc_patch as _patch  # type: ignore[attr-defined]
except ImportError as exc:
    if "attempted relative import with no known parent package" in str(exc):
        import main_hpc_lc_patch as _patch   # patches main_hpc in-place
    else:
        raise ImportError(
            "Unable to import the JAX backend for metacountregressor. "
            "Install package dependencies including 'jax' and 'jaxlib'."
        ) from exc

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import gc
import math
from dataclasses import replace
from functools import partial
import io
import warnings
from contextlib import redirect_stdout
from jaxopt import LBFGS
from scipy.optimize import (
    differential_evolution as scipy_differential_evolution,
    minimize as scipy_minimize,
)
from typing import Optional, Dict, List, Union, Any
from pathlib import Path

try:
    from .family_search import (
        CMFFamilySearchProblem,
        DurationSearchProblem,
        LinearSearchProblem,
        UnifiedCMFSearchProblem,
    )

    from .main_hpc import (
        StructureEvaluator,
        CountModel,
        build_base_index,
        build_datasets,
        generate_master_halton,
        generate_master_draws,
        fit_em,
        run_nsga,
        MultiStartSA,
        AdaptiveDE,
        DynamicHarmony,
        populate_allowed_roles,
        populate_allowed_distributions,
        decode_best_solution,
        decode_distribution,
        refit_and_print,
        save_run_summary_to_txt,
        check_identification,
        _is_oom_error,
    )
    from .main_hpc_lc_patch import (
        ModelSpec,
        build_param_index,
        build_model_from_manual_spec,
        mixed_model_loglik,
        mixed_model_loglik_reg,
        print_summary,
        _seed_classes_from_clusters,
        _tobit_ols_init,
    )
except ImportError:
    from family_search import (
        CMFFamilySearchProblem,
        DurationSearchProblem,
        LinearSearchProblem,
        UnifiedCMFSearchProblem,
    )

    from main_hpc import (
        StructureEvaluator,
        CountModel,
        build_base_index,
        build_datasets,
        generate_master_halton,
        generate_master_draws,
        fit_em,
        run_nsga,
        MultiStartSA,
        AdaptiveDE,
        DynamicHarmony,
        populate_allowed_roles,
        populate_allowed_distributions,
        decode_best_solution,
        decode_distribution,
        refit_and_print,
        save_run_summary_to_txt,
        check_identification,
        _is_oom_error,
    )
    from main_hpc_lc_patch import (
        ModelSpec,
        build_param_index,
        build_model_from_manual_spec,
        mixed_model_loglik,
        mixed_model_loglik_reg,
        print_summary,
        _seed_classes_from_clusters,
        _tobit_ols_init,
    )
try:
    from .output_config import SearchOutputConfig, save_search_result
except ImportError:
    from output_config import SearchOutputConfig, save_search_result

__all__ = ["StructureEvaluatorLC", "ExperimentBuilder"]


# ═══════════════════════════════════════════════════════════════════════
# UPDATED ROLE_PROBS  (add slots for roles 7 and 8)
# Applied directly to Solvers_METAJAX.ROLE_PROBS as well.
# ═══════════════════════════════════════════════════════════════════════

try:
    from . import Solvers_METAJAX as _solvers  # type: ignore[attr-defined]
except ImportError:
    import Solvers_METAJAX as _solvers

ROLE_PROBS = np.array([
    0.38,   # 0 – Excluded
    0.14,   # 1 – Fixed
    0.17,   # 2 – Random Independent
    0.16,   # 3 – Random Correlated
    0.00,   # 4 – Grouped
    0.00,   # 5 – Heterogeneity in means
    0.05,   # 6 – Zero Inflation
    0.05,   # 7 – Membership only
    0.05,   # 8 – Membership + fixed outcome
])
ROLE_PROBS = ROLE_PROBS / ROLE_PROBS.sum()
_solvers.ROLE_PROBS = ROLE_PROBS   # patch the module-level constant


ROLE_GUIDE = """
ROLE CODES
──────────
  0  Excluded          Variable not in the model.
  1  Fixed             Same coefficient across all sites/individuals.
  2  Random Indep.     Site-specific coefficient, independent draws.
  3  Random Corr.      Site-specific coefficient, jointly estimated covariance.
  4  Grouped           Coefficient shared within a group (e.g. road class).
  5  Heterogeneity     Variable that explains variation in random-param MEANS.
  6  Zero Inflation    Enters the zero-inflation probability equation.
  7  Membership only   Enters the CLASS-PROBABILITY equation only.
                       The variable shifts which latent class an individual
                       belongs to, but has no direct effect on the outcome.
                       Ignored (→ excluded) when latent_classes = 1.
  8  Membership+Fixed  Enters BOTH the class-probability equation AND the
                       outcome equation as a fixed covariate.
                       Each class gets its own outcome coefficient for this
                       variable (class-specific), AND the variable influences
                       class membership.
                       Collapsed to Fixed (role 1) when latent_classes = 1.

DISTRIBUTION CODES (roles 2, 3, 4)
────────────────────────────────────
  normal     Symmetric, unbounded.     Good default.
  lognormal  Positive-only.            Use when effect must be one-signed.
  triangular Bounded, symmetric.       Use for bounded/fractional variables.
  uniform    Flat.                     Rarely preferred.

LATENT CLASSES
──────────────
  1 class  Standard mixed model.
  2 classes Two sub-populations with separate parameter vectors.
  3+ classes Richer heterogeneity; BIC penalises extra params heavily.
  The GA selects the number of classes automatically via BIC.
"""


# ═══════════════════════════════════════════════════════════════════════
# StructureEvaluatorLC
# ═══════════════════════════════════════════════════════════════════════

class StructureEvaluatorLC(StructureEvaluator):
    """
    Extends StructureEvaluator with:
      • latent-class gene in the decision vector
      • roles 7 (membership-only) and 8 (membership + fixed outcome)
      • warm-started LC estimation in fitness()
    """

    def __init__(self, *args, max_latent_classes: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_latent_classes = max(1, int(max_latent_classes))
        # Populated after every fitness() call; read by BanditGuidedSA
        # to avoid a second fit call for identifiability diagnostics.
        self._last_fit_cache: Optional[dict] = None

    # ── build_spec ──────────────────────────────────────────────────

    def build_spec(self, decision) -> Optional[dict]:
        """
        Decode a decision vector into a manual_spec dict.

        Decision layout (dimension = 3·D + 2 for LC, 2·D + 1 for single-class):
          decision[:D]       = role codes  (0–8)
          decision[D:2D]     = dist codes
          decision[2D]       = dispersion bit
          decision[2D+1]     = latent_class_code  (0→1 class, 1→2, …)
          decision[2D+2:3D+2]= class_mask  (per-class variable assignments)
              0 = both classes   1 = class 1 only   2 = class 2 only
              Only effective when max_latent_classes > 1.
        """
        D              = len(self.vars)
        roles          = decision[:D]
        dists          = decision[D : 2*D]
        dispersion_bit = int(decision[2*D])
        # LC gene only present when max_latent_classes > 1 (dim = 2*D+2).
        if self.max_latent_classes > 1 and len(decision) > 2*D + 1:
            lc_code = int(decision[2*D + 1])
        else:
            lc_code = 0
        # Per-class masks (new)
        if self.max_latent_classes > 1 and len(decision) > 2*D + 2:
            class_mask = [int(x) for x in decision[2*D + 2 : 3*D + 2]]
        else:
            class_mask = [0] * D

        use_nb          = dispersion_bit % 2 == 1
        latent_classes  = lc_code % self.max_latent_classes + 1
        # Effective LC count for STRUCTURAL decisions (membership roles,
        # class masks, per-class variable lists).  The fitness function may
        # override the actual number of fitted classes later, but we must
        # always build the spec as if latent classes are present so that
        # membership (roles 7,8) and class-specific masks are not lost.
        struct_lc       = max(latent_classes, self.max_latent_classes) if self.max_latent_classes > 1 else 1

        fixed      = []
        rdm_ind    = []
        rdm_cor    = []
        grouped    = []
        hetero     = []
        zi         = []
        membership = []
        # Per-class variable lists (same categories, partitioned by class_mask)
        class_fixed   = [[] for _ in range(struct_lc)]
        class_rdm_ind = [[] for _ in range(struct_lc)]
        class_rdm_cor = [[] for _ in range(struct_lc)]

        for i, var in enumerate(self.vars):
            role = int(roles[i])
            if role not in self.allowed_roles.get(var, [0]):
                return None

            if var in self._unidentifiable and role != 0:
                return None

            cm = class_mask[i] if struct_lc > 1 else 0
            # Map class_mask to per-class inclusion
            in_class = []
            if struct_lc <= 1:
                in_class = [True]
            else:
                in_class = [
                    cm in (0, 1),   # class 1
                    cm in (0, 2),   # class 2
                ][:struct_lc]

            if role == 0:
                pass  # excluded

            elif role == 1:
                fixed.append(var)
                for c, inc in enumerate(in_class):
                    if inc:
                        class_fixed[c].append(var)

            elif role == 2:
                dist = decode_distribution(
                    dists[i], self.allowed_distributions.get(var, ["normal"])
                )
                rdm_ind.append(f"{var}:{dist}")
                for c, inc in enumerate(in_class):
                    if inc:
                        class_rdm_ind[c].append(f"{var}:{dist}")

            elif role == 3:
                dist = decode_distribution(
                    dists[i], self.allowed_distributions.get(var, ["normal"])
                )
                rdm_cor.append(f"{var}:{dist}")
                for c, inc in enumerate(in_class):
                    if inc:
                        class_rdm_cor[c].append(f"{var}:{dist}")

            elif role == 4:
                dist = decode_distribution(
                    dists[i], self.allowed_distributions.get(var, ["normal"])
                )
                grouped.append(f"{var}:{dist}")

            elif role == 5:
                hetero.append(var)

            elif role == 6:
                zi.append(var)

            elif role == 7:
                if struct_lc > 1:
                    membership.append(var)

            elif role == 8:
                if struct_lc > 1:
                    membership.append(var)
                fixed.append(var)
                for c, inc in enumerate(in_class):
                    if inc:
                        class_fixed[c].append(var)

        # ── Force at least one membership variable for LC models ──────
        # If no variable has been assigned a membership role (7 or 8),
        # the logit equation collapses to a constant intercept and classes
        # cannot be distinguished by covariate profiles.  Force-assign
        # one variable that is already in the outcome equation to also
        # enter the membership equation (effectively promoting it to role 8).
        if struct_lc > 1 and len(membership) == 0:
            eligible_mem = [
                v for v in self.vars
                if v in fixed
                and self.allowed_roles.get(v, [0]) != [0]
                and v not in self._unidentifiable
            ]
            if not eligible_mem:
                eligible_mem = [
                    v for v in self.vars
                    if self.allowed_roles.get(v, [0]) != [0]
                    and v not in self._unidentifiable
                ]
            if eligible_mem:
                seed = abs(hash(tuple(int(x) for x in decision))) % (2 ** 32)
                rng = np.random.default_rng(seed)
                mem_var = eligible_mem[int(rng.integers(len(eligible_mem)))]
                membership.append(mem_var)
                if mem_var not in fixed:
                    fixed.append(mem_var)
                    for c in range(struct_lc):
                        class_fixed[c].append(mem_var)

        # Single correlated var → demote to independent
        if len(rdm_cor) == 1:
            rdm_ind.extend(rdm_cor)
            rdm_cor = []
            for c in range(struct_lc):
                if len(class_rdm_cor[c]) == 1:
                    class_rdm_ind[c].extend(class_rdm_cor[c])
                    class_rdm_cor[c] = []

        # ── Force structural uniqueness across classes ────────────────
        # Every LC spec MUST differentiate the classes — identical
        # variable sets collapse the EM into a single-class solution.
        # When classes share all variables, deterministically move
        # variables so that each class has at least one unique variable.
        # The RNG is seeded from the decision vector so the same decision
        # always decodes to the same forced spec.
        if struct_lc > 1:
            class_var_sets = [
                frozenset(class_fixed[c])
                | frozenset(t.split(":")[0] for t in class_rdm_ind[c])
                | frozenset(t.split(":")[0] for t in class_rdm_cor[c])
                for c in range(struct_lc)
            ]
            all_vars = set.union(*class_var_sets) if class_var_sets else set()
            seed = abs(hash(tuple(int(x) for x in decision))) % (2 ** 32)
            rng = np.random.default_rng(seed)

            if class_var_sets[0] and all(s == class_var_sets[0] for s in class_var_sets[1:]):
                # All classes identical — force at least 2 differences if possible
                eligible = sorted(class_var_sets[0])
                n_diff = min(2, len(eligible))
                for d in range(n_diff):
                    avail = sorted(set(eligible) & class_var_sets[0])
                    if not avail:
                        break
                    var = avail[int(rng.integers(len(avail)))]
                    drop_class = (d + int(rng.integers(struct_lc))) % struct_lc
                    class_fixed[drop_class] = [v for v in class_fixed[drop_class] if v != var]
                    class_rdm_ind[drop_class] = [
                        t for t in class_rdm_ind[drop_class] if t.split(":")[0] != var
                    ]
                    class_rdm_cor[drop_class] = [
                        t for t in class_rdm_cor[drop_class] if t.split(":")[0] != var
                    ]
                    class_var_sets[drop_class] = class_var_sets[drop_class] - {var}

        spec = {
            "fixed_terms":      fixed,
            "rdm_terms":        rdm_ind,
            "rdm_cor_terms":    rdm_cor,
            "grouped_terms":    grouped,
            "hetro_in_means":   hetero,
            "zi_terms":         zi,
            "membership_terms": membership,
            "dispersion":       1 if use_nb else 0,
            "latent_classes":   latent_classes,
        }
        # Per-class terms: only include when classes actually differ
        if struct_lc > 1:
            spec["class_fixed"]   = class_fixed
            spec["class_rdm_ind"] = class_rdm_ind
            spec["class_rdm_cor"] = class_rdm_cor

        return spec

    # ── structural_signature ────────────────────────────────────────

    def structural_signature(self, spec_dict) -> Optional[tuple]:
        if spec_dict is None:
            return None

        total_random = (
            len(spec_dict["rdm_terms"]) +
            len(spec_dict["rdm_cor_terms"]) +
            len(spec_dict["grouped_terms"])
        )
        hetero_eff = (
            tuple(sorted(spec_dict["hetro_in_means"]))
            if total_random > 0 else ()
        )

        return (
            tuple(sorted(spec_dict["fixed_terms"])),
            tuple(sorted(spec_dict["rdm_terms"])),
            tuple(sorted(spec_dict["rdm_cor_terms"])),
            tuple(sorted(spec_dict["grouped_terms"])),
            hetero_eff,
            tuple(sorted(spec_dict["zi_terms"])),
            tuple(sorted(spec_dict.get("membership_terms", []))),
            tuple(
                tuple(sorted(cf)) for cf in spec_dict.get("class_fixed", [])
            ),
            spec_dict["dispersion"],
            spec_dict.get("latent_classes", 1),
        )

    # ── build_data ──────────────────────────────────────────────────

    def build_data(self, df, spec_dict, master_halton):
        """
        Extends parent build_data to pass membership_cols through to
        build_jax_data (handled by main_hpc_lc_patch).
        """
        data_tmp, spec = build_model_from_manual_spec(
            df=df,
            manual_spec=spec_dict,
            id_col=self.id_col,
            y_col=self.y_col,
            offset_col=self.offset_col,
            draw_method=getattr(self, 'draw_method', 'sobol'),
            R=self.R,
        )

        var_index = {v: i for i, v in enumerate(self.vars)}

        ind_idx = [var_index[v] for v in spec.random_ind_names]
        cor_idx = [var_index[v] for v in spec.random_cor_names]
        g_idx   = [var_index[v] for v in spec.grouped_names]

        draws_ind = master_halton[:, ind_idx, :] if spec.Kr_ind > 0 else None
        draws_cor = master_halton[:, cor_idx, :] if spec.Kr_cor > 0 else None
        draws_g   = None

        if spec.Kg > 0:
            if self.group_id_col is None:
                raise ValueError("Grouped effects require group_id_col")
            G = df[self.group_id_col].nunique()
            mh_g    = generate_master_draws(G, len(self.vars), self.R,
                                             seed=999, draw_method=getattr(self, 'draw_method', 'halton'))
            draws_g = mh_g[:, g_idx, :]

        # Rebuild with correct draws (membership_cols flows through spec_dict)
        data, spec = build_model_from_manual_spec(
            df=df,
            manual_spec=spec_dict,
            id_col=self.id_col,
            y_col=self.y_col,
            offset_col=self.offset_col,
            draws_ind=draws_ind,
            draws_cor=draws_cor,
            draws_g=draws_g,
            draw_method=getattr(self, 'draw_method', 'sobol'),
            R=self.R,
        )

        return data, spec

    # ── fitness ─────────────────────────────────────────────────────

    def fitness(self, decision) -> Union[float, np.ndarray]:
        """
        Evaluate a decision vector.

        For LC models (C > 1):
          1. Fit single-class (C=1) model for warm initialisation.
          2. Perturb θ₁ to seed each class.
          3. Run EM (≤30 steps) then MLE polish.

        Adaptive identifiability guard
        --------------------------------
        After every successful single-class fit the raw parameter vector is
        screened for near-zero coefficients (|estimate| < _id_zero_tol).
        Any active variable (role != 0) whose coefficient is essentially zero
        is recorded as a failure in the role memory so that sample_allowed_role()
        progressively avoids proposing that (variable, role) pair again.

        For membership variables specifically, the gamma params in the
        warm start are initialised to zero (constant class probs), which
        is the natural neutral starting point.
        """
        key = tuple(decision.tolist())
        if key in self.cache:
            return self.cache[key]

        spec_dict = self.build_spec(decision)
        if spec_dict is None:
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

        sig = self.structural_signature(spec_dict)
        if sig is None:
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

        if sig in self.structure_cache:
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

        self.structure_cache.clear()
        self.structure_cache.add(sig)

        C = spec_dict.get("latent_classes", 1)

        try:
            data_train, spec = self.build_data(
                self.df_train, spec_dict, self.master_halton_train
            )
            if spec_dict.get("model") is not None:
                spec = replace(spec, model=spec_dict["model"])

            # ── Single-class path ──────────────────────────────────
            if C == 1:
                model = CountModel(spec, data_train)
                result_1 = model.fit()
                bic = model.bic()

                    # ── Adaptive identifiability guard + cache ─────────
                # Store the fit so BanditGuidedSA can read t-stats
                # without a second fit call.
                self._last_fit_cache = {
                    "params": np.asarray(result_1.params),
                    "spec":   spec,
                    "data":   data_train,
                    "bic":    float(bic),
                }
                self._update_role_memory_from_fit(
                    params   = self._last_fit_cache["params"],
                    spec     = spec,
                    decision = decision,
                )

            # ── Multi-class path with warm start ───────────────────
            else:
                K_mem  = spec.K_membership
                spec_1 = replace(spec, latent_classes=1)

                # Step 1 — fit single-class
                model_1   = CountModel(spec_1, data_train)
                result_1  = model_1.fit()
                theta_1   = np.array(result_1.params)
                K_base    = build_param_index(spec_1)["total_params"]

                # Pre-build spec_c and pindex for per-class sizes
                spec_c = replace(spec, latent_classes=C)
                pindex_c = build_param_index(spec_c)
                _class_K_base = list(pindex_c.get("class_K_base", [K_base] * C))

                # Step 2 — cluster-based seeding for well-separated class starts
                rng = np.random.default_rng(abs(hash(sig)) % (2**31))
                try:
                    per_class_thetas = _seed_classes_from_clusters(
                        theta_1, data_train, spec_1, C, K_base, rng,
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

                # gamma init: zeros → equal class probs, membership coeffs=0
                gamma_init = np.zeros((C - 1) * (K_mem + 1))
                init_params = np.concatenate([theta_init, gamma_init])

                # Step 3 — EM
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

                # Step 4 — MLE polish (JAX-native optimizer, regularised if l2_penalty > 0)
                polish = LBFGS(
                    fun=lambda p: mixed_model_loglik_reg(p, data_train, spec_c),
                    maxiter=500,
                )
                result_c = polish.run(jnp.array(params_em))
                params_c = np.array(result_c.params)
                ll  = -float(mixed_model_loglik(params_c, data_train, spec_c))  # unregularised LL
                n   = data_train["y"].shape[0]
                k   = len(params_c)
                bic = k * np.log(n) - 2.0 * ll

                # ── Store fit cache so BanditGuidedSA can read t-stats ──
                self._last_fit_cache = {
                    "params": params_c,
                    "spec":   spec_c,
                    "data":   data_train,
                    "bic":    float(bic),
                }
                # Adaptive guard on the LC params (per-class fixed effects)
                self._update_role_memory_from_fit(
                    params   = params_c,
                    spec     = spec_c,
                    decision = decision,
                )

                class _Model:
                    params = params_c
                model = _Model()

            # ── Single-objective return ────────────────────────────
            if self.mode == "single":
                value = float(bic)
                self.cache[key] = value
                return value

            # ── Multi-objective return (BIC + test RMSE) ──────────
            data_test, _ = self.build_data(
                self.df_test, spec_dict, self.master_halton_test
            )
            model_test        = CountModel(spec, data_test)
            model_test.params = np.array(model.params)
            preds   = model_test.predict()
            y_true  = np.array(data_test["y"]).squeeze()
            rmse    = np.sqrt(np.mean((preds - y_true) ** 2))

            value = np.array([float(bic), float(rmse)])
            self.cache[key] = value
            return value

        except Exception as e:
            print(f"  [fitness error] {e}")
            if _is_oom_error(e):
                # A GPU OOM here means the device is at/near capacity. Free
                # whatever JAX can free immediately so the *next* candidate
                # in the search isn't doomed by the same exhaustion (left
                # unhandled, this previously caused every remaining
                # generation in a 500-gen SA run to fail identically).
                print("  [fitness error] GPU out-of-memory; "
                      "clearing JAX caches before continuing search.")
                jax.clear_caches()
                gc.collect()
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12

    # ── Adaptive identifiability helper ─────────────────────────────

    _id_zero_tol: float = 1e-3   # |coef| below this → treat as zero-effect

    def _update_role_memory_from_fit(
        self,
        params:   np.ndarray,
        spec,
        decision: np.ndarray,
    ) -> None:
        """
        Inspect fixed-effect coefficients and update the sign_violation counters
        that sample_allowed_role() reads to down-weight bad (variable, role) pairs.

        A variable is flagged as 'bad' when its fixed-effect estimate is
        effectively zero (|β| < _id_zero_tol) AND it has a non-zero role —
        meaning it is wasting a degree of freedom without contributing to fit.
        This populates the _sign_visit_counts / _sign_violation_counts dicts
        that already exist in the Solvers_METAJAX bandit hook.
        """
        if not (hasattr(self, "_sign_violation_counts")
                and hasattr(self, "_sign_visit_counts")):
            # Initialise lazily so vanilla SA still works without the memory
            self._sign_violation_counts: Dict[str, int] = {}
            self._sign_visit_counts:     Dict[str, int] = {}

        try:
            fixed_names = list(getattr(spec, "fixed_names", []))
            n_vars = len(self.vars)
            D = min(n_vars, len(decision))

            # Map fixed_name → param index (skip __INTERCEPT__)
            name_to_param_idx = {
                nm: i for i, nm in enumerate(fixed_names)
                if nm != "__INTERCEPT__"
            }

            for var_idx in range(D):
                role = int(decision[var_idx])
                if role == 0:
                    continue
                var_name = self.vars[var_idx]
                pidx = name_to_param_idx.get(var_name)
                if pidx is None or pidx >= len(params):
                    continue

                coef = float(params[pidx])
                self._sign_visit_counts[var_name] = (
                    self._sign_visit_counts.get(var_name, 0) + 1
                )
                if abs(coef) < self._id_zero_tol:
                    # Record as a violation so the bandit down-weights this role
                    self._sign_violation_counts[var_name] = (
                        self._sign_violation_counts.get(var_name, 0) + 1
                    )
        except Exception:
            pass  # Never let this crash a fitness evaluation


class ForcedModelStructureEvaluatorLC(StructureEvaluatorLC):
    def __init__(self, *args, forced_model: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_model = forced_model

    def build_spec(self, decision) -> Optional[dict]:
        spec = super().build_spec(decision)
        if spec is None:
            return None
        spec["model"] = self.forced_model
        if self.forced_model in {"lognormal", "gaussian", "tobit"}:
            spec["dispersion"] = 0
        return spec


# ═══════════════════════════════════════════════════════════════════════
# SOLVER PATCH  — generate_neighbor supports tail genes (dispersion + LC)
# and the new roles 7 / 8.  Replace AdvancedSimulatedAnnealing.generate_neighbor.
# ═══════════════════════════════════════════════════════════════════════

def _generate_neighbor_patched(self, solution, T=None, max_attempts=20, min_active=2):
    """
    Extended generate_neighbor that correctly mutates ALL gene slots:

    Tail gene index  2D     → dispersion bit (flip 0↔1)
    Tail gene index  2D+1   → latent-class code (step ±1)
    Tail gene indices 2D+2 … 3D+1 → class-specific masks (cycle 0→1→2→0)
    Role genes now include 7 and 8 (sampled via ROLE_PROBS).
    """
    for _ in range(max_attempts):

        neighbor = solution.copy()

        if T is not None and self.T0 is not None:
            mut_rate = self.mutation_rate * (T / self.T0)
        else:
            mut_rate = self.mutation_rate
        mut_rate = float(np.clip(mut_rate, 0.0, 1.0))

        n_changes = np.random.randint(self.min_changes, self.max_changes + 1)
        indices   = list(np.random.choice(self.dim, size=n_changes, replace=False))

        # ── Always include at least one class_mask mutation when LC is enabled ──
        # The class_mask genes (2D+2 … 3D+1) govern per-class variable
        # assignments.  Without forced mutation here the SA spends most
        # of its budget mutating role genes and rarely explores different
        # class-specific variable configurations, leading to collapsed
        # (identical) class specifications.
        if self.evaluator.max_latent_classes > 1:
            mask_start = 2 * D + 2
            mask_end   = 3 * D + 1
            mask_indices = list(range(mask_start, min(mask_end + 1, self.dim)))
            if mask_indices and not any(i in indices for i in mask_indices):
                # Replace 1 randomly selected index with a mask gene
                replace_ix = np.random.choice(range(len(indices)))
                indices[replace_ix] = int(np.random.choice(mask_indices))
            # Also add an extra mask mutation ~80% of the time
            if len(mask_indices) > 0 and np.random.rand() < 0.8:
                extra = int(np.random.choice(mask_indices))
                if extra not in indices:
                    indices.append(extra)
        indices = np.asarray(indices, dtype=int)

        changed = False
        D       = self.dim_core          # number of role/dist pairs

        for idx in indices:
            if np.random.rand() < mut_rate:

                if idx < D:
                    # Role gene
                    neighbor[idx] = self.sample_allowed_role(idx)
                    changed = True

                elif idx < 2 * D:
                    # Distribution gene
                    neighbor[idx] = np.random.randint(0, 6)
                    changed = True

                else:
                    # Tail genes  — indices >= 2*D
                    tail_pos = idx - 2 * D

                    if tail_pos == 0:
                        # Dispersion bit: flip
                        neighbor[idx] = 1 - int(neighbor[idx])
                    elif tail_pos == 1:
                        # Latent-class code: step ±1 within bounds
                        max_code      = self.evaluator.max_latent_classes - 1
                        step          = np.random.choice([-1, 1])
                        neighbor[idx] = int(np.clip(neighbor[idx] + step,
                                                    0, max_code))
                    else:
                        # Class-specific variable masks (indices 2D+2 … 3D+1):
                        #   0 = both classes  1 = class 1 only  2 = class 2 only
                        # Cycle through the three possible values or pick random.
                        old_val = int(neighbor[idx])
                        choices = [v for v in (0, 1, 2) if v != old_val]
                        neighbor[idx] = int(np.random.choice(choices))
                    changed = True

        # Enforce min-active constraint
        active = np.sum(neighbor[:D] != 0)
        if active < min_active:
            zeros = np.where(neighbor[:D] == 0)[0]
            if len(zeros) > 0:
                activate = np.random.choice(
                    zeros, size=min_active - active, replace=False
                )
                for j in activate:
                    neighbor[j] = self.sample_allowed_role(j, force_active=True)

        neighbor = self.repair(neighbor)
        if changed and not self.is_same(neighbor, solution):
            return neighbor

    # Fallback
    neighbor     = solution.copy()
    active_count = np.sum(neighbor[:self.dim_core] != 0)

    if active_count < min_active:
        zero_idx = np.where(neighbor[:self.dim_core] == 0)[0]
        activate = np.random.choice(
            zero_idx, size=min_active - active_count, replace=False
        )
        neighbor[activate] = np.random.randint(1, 9, size=len(activate))
    else:
        idx      = np.random.randint(0, self.dim_core)
        var_name = self.evaluator.vars[idx]
        allowed  = self.evaluator.allowed_roles[var_name]
        old      = neighbor[idx]
        possible = [v for v in allowed if v != old]
        if possible:
            neighbor[idx] = np.random.choice(possible)

    # Also randomize class masks in the fallback when LC is enabled
    has_lc = getattr(self.evaluator, "max_latent_classes", 1) > 1
    if has_lc and len(neighbor) > 2 * D + 2:
        for idx in range(2 * D + 2, min(3 * D + 2, len(neighbor))):
            neighbor[idx] = np.random.randint(0, 3)

    if self.is_same(solution, neighbor):
        return self.generate_neighbor(solution, T,
                                      min_active=min_active + 1)
    return neighbor


_solvers.AdvancedSimulatedAnnealing.generate_neighbor = _generate_neighbor_patched


# ═══════════════════════════════════════════════════════════════════════
# ExperimentBuilder
# ═══════════════════════════════════════════════════════════════════════

class ExperimentBuilder:
    """
    Inspects any DataFrame and guides you through setting up an experiment.

    Quick-start
    ───────────
    from experiment_package import ExperimentBuilder
    import pandas as pd

    df      = pd.read_csv("my_data.csv")
    builder = ExperimentBuilder(df, id_col="SITE_ID", y_col="CRASHES")

    builder.describe()          # data + variable stats
    builder.suggest_config()    # auto role/dist suggestions with explanation

    evaluator = builder.build_evaluator(
        fixed_override      = {"OFFSET": [1]},   # force OFFSET → fixed
        exclude             = ["YEAR"],           # always exclude YEAR
        mode                = "single",           # BIC only
        max_latent_classes  = 2,                  # allow up to 2 LC classes
        membership_override = {"URB": [7]},       # allow URB as membership-only
        R                   = 200,
    )

    result = builder.run(evaluator, algo="sa", max_iter=3000)
    """

    _ROLE_NAMES = {
        0: "Excluded", 1: "Fixed", 2: "Rnd-Ind", 3: "Rnd-Cor",
        4: "Grouped",  5: "Hetero", 6: "ZI", 7: "Membership", 8: "Mem+Fixed",
    }

    def __init__(
        self,
        df:           pd.DataFrame,
        id_col:       str,
        y_col:        str,
        offset_col:   Optional[str] = None,
        group_id_col: Optional[str] = None,
        default_model_family: str = "count",
        default_engine: str = "jax",
    ):
        self.df           = df.copy()
        self.id_col       = id_col
        self.y_col        = y_col
        self.offset_col   = offset_col
        self.group_id_col = group_id_col
        self.default_model_family = default_model_family.lower()
        self.default_engine = default_engine.lower()
        self._evaluator: Optional[StructureEvaluatorLC] = None

        if self.default_model_family != "count":
            raise ValueError("ExperimentBuilder defaults must remain count-first. Use build_search(model_family=...) for alternative families.")
        if self.default_engine != "jax":
            raise ValueError("The primary ExperimentBuilder engine is JAX. Use default_engine='jax'.")

        self._ensure_columns_exist([id_col, y_col], "ExperimentBuilder")
        if offset_col is not None:
            self._ensure_columns_exist([offset_col], "ExperimentBuilder")
        if group_id_col is not None:
            self._ensure_columns_exist([group_id_col], "ExperimentBuilder")

        reserved = {id_col, y_col}
        if offset_col:   reserved.add(offset_col)
        if group_id_col: reserved.add(group_id_col)
        self._candidate_vars = [c for c in df.columns if c not in reserved]

    def _ensure_columns_exist(self, columns: List[str], context: str = "builder") -> None:
        missing = [column for column in columns if column not in self.df.columns]
        if missing:
            formatted = ", ".join(sorted(missing))
            raise ValueError(f"{context} received columns that are not in the dataframe: {formatted}")

    def _normalize_variables(self, variables: Optional[List[str]], exclude: Optional[List[str]] = None) -> List[str]:
        chosen = list(self._candidate_vars if variables is None else variables)
        self._ensure_columns_exist(chosen, "variables")
        filtered = [var for var in chosen if var not in set(exclude or [])]
        if not filtered:
            raise ValueError("No searchable variables remain after applying exclude.")
        return list(dict.fromkeys(filtered))

    def _normalize_override_map(self, mapping: Optional[Dict[str, list]], label: str) -> Dict[str, list]:
        normalized = mapping or {}
        self._ensure_columns_exist(list(normalized.keys()), label)
        return normalized

    @staticmethod
    def _raise_on_unused_kwargs(kwargs: Dict[str, Any], context: str) -> None:
        if not kwargs:
            return
        unused = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unexpected arguments for {context}: {unused}")

    @staticmethod
    def _continuous_de_warm_start(
        objective,
        init_params: np.ndarray,
        *,
        enabled: bool,
        maxiter: int,
        popsize: int,
        rel_span: float,
        abs_span: float,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Refine an initial parameter vector with bounded continuous DE."""
        init = np.asarray(init_params, dtype=float).reshape(-1)
        if (not enabled) or init.size == 0:
            return init, {
                "ran": False,
                "accepted": False,
                "reason": "disabled_or_empty",
            }

        safe_init = np.where(np.isfinite(init), init, 0.0)
        span = np.maximum(abs_span, np.abs(safe_init) * rel_span)
        bounds = [(c - s, c + s) for c, s in zip(safe_init, span)]

        def _obj_np(x):
            try:
                value = float(objective(jnp.array(x)))
            except Exception:
                return 1e20
            if not np.isfinite(value):
                return 1e20
            return value

        incumbent = safe_init
        incumbent_obj = _obj_np(incumbent)
        report: dict[str, Any] = {
            "ran": True,
            "accepted": False,
            "seed": int(seed),
            "maxiter": int(maxiter),
            "popsize": int(popsize),
            "start_obj": float(incumbent_obj),
            "de_obj": None,
            "delta_obj": None,
            "reason": "not_improved",
        }

        try:
            de_result = scipy_differential_evolution(
                _obj_np,
                bounds=bounds,
                maxiter=int(maxiter),
                popsize=int(popsize),
                seed=int(seed),
                polish=False,
                updating="deferred",
                workers=1,
            )
            candidate = np.asarray(de_result.x, dtype=float)
            candidate_obj = float(de_result.fun)
            report["de_obj"] = candidate_obj
            report["delta_obj"] = float(incumbent_obj - candidate_obj) if np.isfinite(candidate_obj) else None
            if np.isfinite(candidate_obj) and candidate_obj < incumbent_obj:
                report["accepted"] = True
                report["reason"] = "improved"
                return candidate, report
        except Exception as exc:
            warnings.warn(
                f"Continuous DE warm-start skipped due to: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            report["reason"] = f"exception: {exc}"

        return incumbent, report

    @staticmethod
    def _is_cmf_local_name(name: str) -> bool:
        return str(name).startswith("__cmf_local__")

    def _build_cmf_local_bounds(
        self,
        spec,
        lower_level_param_bounds,
    ) -> tuple[list[tuple[Optional[float], Optional[float]]], int]:
        lo, hi = float(lower_level_param_bounds[0]), float(lower_level_param_bounds[1])
        if lo >= hi:
            raise ValueError("lower_level_param_bounds must be (lower, upper) with lower < upper.")

        full_index = build_param_index(spec)
        total_params = int(full_index["total_params"])
        bounds: list[tuple[Optional[float], Optional[float]]] = [(None, None)] * total_params

        base_spec = replace(spec, latent_classes=1)
        base_index = build_base_index(base_spec)

        base_target_idx: list[int] = []

        fixed_rng = base_index.get("fixed")
        if fixed_rng is not None:
            fs, _ = fixed_rng
            for i, name in enumerate(getattr(base_spec, "fixed_names", ())):
                if self._is_cmf_local_name(name):
                    base_target_idx.append(fs + i)

        cor_rng = base_index.get("cor_mean")
        if cor_rng is not None:
            cs, _ = cor_rng
            for i, name in enumerate(getattr(base_spec, "random_cor_names", ())):
                if self._is_cmf_local_name(name):
                    base_target_idx.append(cs + i)

        ind_rng = base_index.get("ind_mean")
        if ind_rng is not None:
            is_, _ = ind_rng
            for i, name in enumerate(getattr(base_spec, "random_ind_names", ())):
                if self._is_cmf_local_name(name):
                    base_target_idx.append(is_ + i)

        grp_rng = base_index.get("group_mean")
        if grp_rng is not None:
            gs, _ = grp_rng
            for i, name in enumerate(getattr(base_spec, "grouped_names", ())):
                if self._is_cmf_local_name(name):
                    base_target_idx.append(gs + i)

        base_target_idx = sorted(set(base_target_idx))
        if not base_target_idx:
            return bounds, 0

        if int(getattr(spec, "latent_classes", 1)) > 1:
            k_base = int(base_index["total_params"])
            C = int(spec.latent_classes)
            for c in range(C):
                offset = c * k_base
                for idx in base_target_idx:
                    bounds[offset + idx] = (lo, hi)
            return bounds, len(base_target_idx) * C

        for idx in base_target_idx:
            bounds[idx] = (lo, hi)
        return bounds, len(base_target_idx)

    @staticmethod
    def _bounded_refit(objective, start_params, bounds, maxiter: int = 500) -> np.ndarray:
        x0 = np.asarray(start_params, dtype=float).reshape(-1)
        if x0.size == 0:
            return x0

        from jaxopt import ScipyMinimize as JaxoptMinimize

        def _obj_val(x):
            try:
                value = float(objective(jnp.array(x, dtype=jnp.float64)))
            except Exception:
                return 1e20
            return value if np.isfinite(value) else 1e20

        start_val = _obj_val(x0)

        try:
            solver = JaxoptMinimize(
                fun=objective,
                method="SLSQP",
                tol=1e-8,
                maxiter=int(maxiter),
            )
            result = solver.run(jnp.array(x0, dtype=jnp.float64), bounds=bounds)
            cand = np.asarray(result.params, dtype=float)
            cand_val = _obj_val(cand)
            if np.isfinite(cand_val) and cand_val <= start_val:
                return cand
        except Exception as exc:
            warnings.warn(
                f"Bounded lower-level refit skipped due to: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

        return x0

    @staticmethod
    def get_search_argument_guide() -> Dict[str, Dict[str, str]]:
        return {
            "shared": {
                "algo": "Metaheuristic driver. Use 'sa', 'hc', 'de', or 'hs'.",
                "R": "Number of simulation draws for JAX mixed-model estimation.",
                "max_iter": "Search iterations for the metaheuristic driver.",
                "max_latent_classes": "Upper bound on latent classes for the count-family architecture.",
            },
            "count": {
                "variables": "Candidate columns to search over.",
                "default_roles": "Allowed structural roles per variable.",
                "fixed_override": "Force or restrict roles for specific variables.",
                "membership_override": "Allow or restrict latent-class membership roles for specific variables.",
            },
            "cmf": {
                "aadt_col": "AADT column used to build the CMF elasticity term.",
                "baseline_vars": "Baseline CMF variables entering outside log(AADT).",
                "local_vars": "CMF local variables entering as var * log(AADT).",
                "cmf_driver": "Use 'jax_count' for the main JAX architecture or 'ga' for the legacy GA route.",
            },
            "linear": {
                "objective_kwargs": "Options forwarded to the linear metaheuristic objective.",
            },
            "duration": {
                "budget_col": "Budget column used by the duration helper objective.",
                "objective": "Duration objective: 'independent' or 'budget_penalty'.",
            },
        }

    @staticmethod
    def get_family_capabilities() -> Dict[str, Dict[str, bool]]:
        return {
            "count": {
                "jax_solver": True,
                "metaheuristic_search": True,
                "random_parameters": True,
                "heterogeneity_in_means": True,
                "zero_inflation": True,
                "latent_classes": True,
                "distribution_assumptions": True,
            },
            "cmf": {
                "jax_solver": True,
                "metaheuristic_search": True,
                "random_parameters": True,
                "heterogeneity_in_means": True,
                "zero_inflation": True,
                "latent_classes": True,
                "distribution_assumptions": True,
            },
            "duration": {
                "jax_solver": True,
                "metaheuristic_search": True,
                "random_parameters": True,
                "heterogeneity_in_means": True,
                "zero_inflation": True,
                "latent_classes": True,
                "distribution_assumptions": True,
            },
            "linear": {
                "jax_solver": True,
                "metaheuristic_search": True,
                "random_parameters": True,
                "heterogeneity_in_means": True,
                "zero_inflation": True,
                "latent_classes": True,
                "distribution_assumptions": True,
            },
        }

    def make_manual_spec(
        self,
        fixed_terms: Optional[List[str]] = None,
        rdm_terms: Optional[List[str]] = None,
        rdm_cor_terms: Optional[List[str]] = None,
        grouped_terms: Optional[List[str]] = None,
        hetro_in_means: Optional[List[str]] = None,
        zi_terms: Optional[List[str]] = None,
        membership_terms: Optional[List[str]] = None,
        dispersion: int = 0,
        latent_classes: int = 1,
        group_id_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        role_columns = {
            "fixed_terms": fixed_terms or [],
            "rdm_terms": rdm_terms or [],
            "rdm_cor_terms": rdm_cor_terms or [],
            "grouped_terms": grouped_terms or [],
            "hetro_in_means": hetro_in_means or [],
            "zi_terms": zi_terms or [],
            "membership_terms": membership_terms or [],
        }

        for label, terms in role_columns.items():
            stripped = [term.split(":")[0] for term in terms]
            self._ensure_columns_exist(stripped, label)

        if latent_classes < 1:
            raise ValueError("latent_classes must be at least 1.")

        if group_id_col is not None and group_id_col not in self.df.columns:
            raise ValueError(f"group_id_col '{group_id_col}' is not in the dataframe.")

        return {
            **role_columns,
            "dispersion": int(dispersion),
            "latent_classes": int(latent_classes),
            "group_id_col": group_id_col if group_id_col is not None else self.group_id_col,
        }

    def fit_manual_model(
        self,
        manual_spec: Dict[str, Any],
        model: str = "poisson",
        df: Optional[pd.DataFrame] = None,
        R: int = 200,
        print_report: bool = False,
        use_prefit_start: bool = True,
        continuous_de_warm_start: bool = True,
        de_maxiter: int = 12,
        de_popsize: int = 8,
        de_rel_span: float = 1.5,
        de_abs_span: float = 1.0,
        de_seed: int = 0,
        latent_fast_mode: bool = False,
        latent_random_start: bool = False,
        lower_level_param_bounds: Optional[tuple[float, float]] = None,
        _lc_fallback_applied: bool = False,
    ) -> Dict[str, Any]:
        de_report: dict[str, Any] = {
            "enabled": bool(continuous_de_warm_start),
            "single_class": None,
            "latent_class_seed": None,
            "latent_class_attempts": [],
        }

        df_fit = self.df if df is None else df
        data, spec = build_model_from_manual_spec(
            df=df_fit,
            manual_spec=manual_spec,
            id_col=self.id_col,
            y_col=self.y_col,
            offset_col=self.offset_col,
            R=R,
        )
        spec = replace(spec, model=model)

        # Harden latent-class estimation with warm start + EM + polish retries.
        if spec.latent_classes > 1:
            C = int(spec.latent_classes)
            K_mem = int(spec.K_membership)
            spec_1 = replace(spec, latent_classes=1)
            K_base = build_param_index(spec_1)["total_params"]
            spec_c = replace(spec, latent_classes=C)
            pindex_c = build_param_index(spec_c)
            _class_K_base_fit = list(pindex_c.get("class_K_base", [K_base] * C))

            has_random_structure = any([
                bool(manual_spec.get("rdm_terms")),
                bool(manual_spec.get("rdm_cor_terms")),
                bool(manual_spec.get("grouped_terms")),
                bool(manual_spec.get("hetro_in_means")),
            ])

            if latent_random_start:
                rng0 = np.random.default_rng(int(de_seed))
                theta_1 = rng0.normal(0.0, 0.1, int(K_base))  # single-class warm-start size
                de_report["single_class"] = {
                    "ran": False,
                    "accepted": False,
                    "reason": "latent_random_start",
                    "start_obj": None,
                    "de_obj": None,
                    "delta_obj": None,
                    "final_obj": None,
                }
            else:
                model_1 = CountModel(spec_1, data)
                try:
                    if model == "tobit":
                        # Bypass Poisson-style prefit: use OLS starting values
                        # and a direct LBFGS on the single-class Tobit likelihood.
                        _K1 = build_base_index(spec_1)["total_params"]
                        _p0, de_report_single = self._continuous_de_warm_start(
                            objective=lambda p: mixed_model_loglik(p, data, spec_1),
                            init_params=_tobit_ols_init(data, _K1),
                            enabled=continuous_de_warm_start,
                            maxiter=de_maxiter,
                            popsize=de_popsize,
                            rel_span=de_rel_span,
                            abs_span=de_abs_span,
                            seed=de_seed,
                        )
                        de_report["single_class"] = de_report_single
                        _sol = LBFGS(
                            fun=lambda p: mixed_model_loglik(p, data, spec_1),
                            maxiter=1000,
                        ).run(jnp.array(_p0))
                        result_1 = _sol
                    else:
                        result_1 = model_1.fit(
                            use_prefit=use_prefit_start,
                            use_continuous_de=continuous_de_warm_start,
                            de_maxiter=de_maxiter,
                            de_popsize=de_popsize,
                            de_rel_span=de_rel_span,
                            de_abs_span=de_abs_span,
                            de_seed=de_seed,
                        )
                        de_report["single_class"] = getattr(model_1, "last_de_report", None)
                except Exception as exc:
                    if (not _lc_fallback_applied) and has_random_structure:
                        fallback_spec = dict(manual_spec)
                        fallback_spec["rdm_terms"] = []
                        fallback_spec["rdm_cor_terms"] = []
                        fallback_spec["grouped_terms"] = []
                        fallback_spec["hetro_in_means"] = []
                        warnings.warn(
                            "Latent-class warm-start failed on random-effect structure; "
                            "retrying with fixed-only latent-class fallback.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        return self.fit_manual_model(
                            manual_spec=fallback_spec,
                            model=model,
                            df=df,
                            R=R,
                            print_report=print_report,
                            use_prefit_start=use_prefit_start,
                            continuous_de_warm_start=continuous_de_warm_start,
                            de_maxiter=de_maxiter,
                            de_popsize=de_popsize,
                            de_rel_span=de_rel_span,
                            de_abs_span=de_abs_span,
                            de_seed=de_seed,
                            latent_fast_mode=latent_fast_mode,
                            latent_random_start=latent_random_start,
                            lower_level_param_bounds=lower_level_param_bounds,
                            _lc_fallback_applied=True,
                        )
                    raise

                theta_1 = np.asarray(result_1.params)

            best_result = None
            best_value = np.inf
            last_error: Optional[Exception] = None

            # Attempt 0 uses cluster-based seeding; remaining attempts fall
            # back to decreasing noise perturbations of the single-class fit.

            # Always guarantee at least some noise for all but the last fallback
            if latent_fast_mode:
                retry_configs = [("random", 0)] if latent_random_start else [(0.02, 0)]
                em_max_iter = 10
                polish_maxiter = 200
            else:
                retry_configs = [
                    ("cluster", 0),
                    (0.05, 1),
                    (0.02, 2),
                    (0.01, 3),  # never zero noise except as a last fallback
                    ("force_min_noise", 4),  # last fallback: forcibly add small noise if all else fails
                ]
                em_max_iter = 40
                polish_maxiter = 800

            for noise_scale, seed in retry_configs:
                try:
                    rng = np.random.default_rng(seed)
                    if noise_scale == "random":
                        theta_init = np.concatenate([
                            rng.normal(0.0, 0.1, k) + 1e-3 * (i + 1)
                            for i, k in enumerate(_class_K_base_fit)
                        ])
                    elif noise_scale == "cluster":
                        try:
                            per_class = _seed_classes_from_clusters(
                                theta_1, data, spec_1, C, K_base, rng,
                                class_K_base=_class_K_base_fit,
                            )
                            theta_init = np.concatenate(per_class)
                        except Exception:
                            # fallback: add moderate noise
                            theta_init = np.concatenate([
                                theta_1[:k] + rng.normal(0.0, 0.05, k)
                                if len(theta_1) >= k
                                else np.pad(theta_1, (0, k - len(theta_1))) + rng.normal(0.0, 0.05, k)
                                for k in _class_K_base_fit
                            ])
                    elif noise_scale == "force_min_noise":
                        # forcibly guarantee different initializations
                        theta_init = np.concatenate([
                            (theta_1[:k] if len(theta_1) >= k else np.pad(theta_1, (0, k - len(theta_1))))
                            + rng.normal(0.0, 0.01, k) + 0.01 * (i+1)
                            for i, k in enumerate(_class_K_base_fit)
                        ])
                    else:
                        # always add noise, never allow all classes to be identical
                        theta_init = np.concatenate([
                            (theta_1[:k] if len(theta_1) >= k else np.pad(theta_1, (0, k - len(theta_1))))
                            + rng.normal(0.0, noise_scale, k) + 1e-4 * (i+1)
                            for i, k in enumerate(_class_K_base_fit)
                        ])

                    # Check: if all classes are still identical, forcibly add small offset
                    # (only works when all classes have same K_base)
                    if len(set(_class_K_base_fit)) == 1:
                        theta_blocks = np.split(theta_init, C)
                        if all(np.allclose(theta_blocks[0], tb) for tb in theta_blocks[1:]):
                            for i in range(1, C):
                                theta_blocks[i] = theta_blocks[i] + 1e-3 * i
                            theta_init = np.concatenate(theta_blocks)

                    gamma_init = np.zeros((C - 1) * (K_mem + 1), dtype=float)
                    init_params = np.concatenate([theta_init, gamma_init])

                    try:
                        init_obj = float(mixed_model_loglik(jnp.array(init_params), data, spec_c))
                    except Exception:
                        init_obj = None

                    try:
                        params_em = fit_em(
                            init_params=init_params,
                            data=data,
                            spec=spec_c,
                            max_iter=em_max_iter,
                            tol=1e-4,
                            verbose=False,
                        )
                    except Exception:
                        params_em = init_params

                    polish_seed, de_report_lc = self._continuous_de_warm_start(
                        objective=lambda p: mixed_model_loglik_reg(p, data, spec_c),
                        init_params=np.asarray(params_em),
                        enabled=continuous_de_warm_start,
                        maxiter=de_maxiter,
                        popsize=de_popsize,
                        rel_span=de_rel_span,
                        abs_span=de_abs_span,
                        seed=de_seed + int(seed),
                    )
                    de_report_lc["attempt"] = int(seed)
                    de_report_lc["noise_scale"] = str(noise_scale)
                    de_report_lc["init_obj"] = init_obj
                    try:
                        de_report_lc["seed_obj"] = float(mixed_model_loglik(jnp.array(polish_seed), data, spec_c))
                    except Exception:
                        de_report_lc["seed_obj"] = None
                    de_report["latent_class_attempts"].append(de_report_lc)

                    polish = LBFGS(
                        fun=lambda p: mixed_model_loglik_reg(p, data, spec_c),
                        maxiter=polish_maxiter,
                    )
                    candidate = polish.run(jnp.array(polish_seed))

                    params_np = np.asarray(candidate.params)
                    value = float(mixed_model_loglik(jnp.array(params_np), data, spec_c))
                    de_report_lc["final_obj"] = value

                    if not np.all(np.isfinite(params_np)):
                        continue
                    if not np.isfinite(value):
                        continue

                    if value < best_value:
                        best_value = value
                        best_result = candidate
                        de_report["latent_class_seed"] = de_report_lc

                except Exception as exc:
                    last_error = exc

            if best_result is None:
                if not _lc_fallback_applied:
                    if has_random_structure:
                        fallback_spec = dict(manual_spec)
                        fallback_spec["rdm_terms"] = []
                        fallback_spec["rdm_cor_terms"] = []
                        fallback_spec["grouped_terms"] = []
                        fallback_spec["hetro_in_means"] = []
                        warnings.warn(
                            "Latent-class manual fit failed on random-effect structure; "
                            "retrying with fixed-only latent-class fallback.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        return self.fit_manual_model(
                            manual_spec=fallback_spec,
                            model=model,
                            df=df,
                            R=R,
                            print_report=print_report,
                            use_prefit_start=use_prefit_start,
                            continuous_de_warm_start=continuous_de_warm_start,
                            de_maxiter=de_maxiter,
                            de_popsize=de_popsize,
                            de_rel_span=de_rel_span,
                            de_abs_span=de_abs_span,
                            de_seed=de_seed,
                            lower_level_param_bounds=lower_level_param_bounds,
                            _lc_fallback_applied=True,
                        )

                if last_error is not None:
                    raise RuntimeError(
                        "Latent-class manual fit failed after robust retries. "
                        "Try simpler random-effects structure or fewer latent classes."
                    ) from last_error
                raise RuntimeError(
                    "Latent-class manual fit failed after robust retries with non-finite objective/parameters."
                )

            result = best_result
            fitted = CountModel(spec_c, data)
            fitted.params = np.asarray(result.params)
            spec = spec_c
        else:
            fitted = CountModel(spec, data)
            if model == "tobit":
                # OLS warm-start + direct LBFGS — bypasses Poisson-style prefit.
                _K = build_base_index(spec)["total_params"]
                _p0, de_report_single = self._continuous_de_warm_start(
                    objective=lambda p: mixed_model_loglik(p, data, spec),
                    init_params=_tobit_ols_init(data, _K),
                    enabled=continuous_de_warm_start,
                    maxiter=de_maxiter,
                    popsize=de_popsize,
                    rel_span=de_rel_span,
                    abs_span=de_abs_span,
                    seed=de_seed,
                )
                de_report["single_class"] = de_report_single
                _sol = LBFGS(
                    fun=lambda p: mixed_model_loglik(p, data, spec),
                    maxiter=1500,
                ).run(jnp.array(_p0))
                result = _sol
                fitted.params = np.asarray(_sol.params)
            else:
                result = fitted.fit(
                    use_prefit=use_prefit_start,
                    use_continuous_de=continuous_de_warm_start,
                    de_maxiter=de_maxiter,
                    de_popsize=de_popsize,
                    de_rel_span=de_rel_span,
                    de_abs_span=de_abs_span,
                    de_seed=de_seed,
                )
                de_report["single_class"] = getattr(fitted, "last_de_report", None)

        if lower_level_param_bounds is not None:
            cmf_bounds, constrained = self._build_cmf_local_bounds(spec, lower_level_param_bounds)
            if constrained > 0:
                current_params = np.asarray(
                    result.params if hasattr(result, "params") else fitted.params,
                    dtype=float,
                )
                bounded_params = self._bounded_refit(
                    objective=lambda p: mixed_model_loglik(p, data, spec),
                    start_params=current_params,
                    bounds=cmf_bounds,
                    maxiter=600,
                )
                fitted.params = np.asarray(bounded_params)

                class _BoundedResult:
                    pass

                bounded_result = _BoundedResult()
                bounded_result.params = np.asarray(bounded_params)
                result = bounded_result

        objective = partial(mixed_model_loglik, data=data, spec=spec)
        param_index = build_param_index(spec)

        if print_report:
            summary = print_summary(
                result=result,
                objective=objective,
                data=data,
                spec=spec,
                param_index=param_index,
            )
        else:
            with redirect_stdout(io.StringIO()):
                summary = print_summary(
                    result=result,
                    objective=objective,
                    data=data,
                    spec=spec,
                    param_index=param_index,
                )

        return {
            "result": result,
            "data": data,
            "spec": spec,
            "manual_spec": manual_spec,
            "summary": summary,
            "param_index": param_index,
            "predictions": np.asarray(fitted.predict()).squeeze(),
            "de_warm_start_report": de_report,
        }

    def compute_latent_class_probabilities(
        self,
        fit_result: Dict[str, Any],
        true_class_col: Optional[str] = None,
    ) -> pd.DataFrame:
        spec = fit_result["spec"]
        if spec.latent_classes <= 1:
            raise ValueError("Latent class probabilities require a fit with latent_classes > 1.")

        data = fit_result["data"]
        params = np.asarray(fit_result["result"].params)
        C = spec.latent_classes
        K_mem = spec.K_membership
        base_spec = replace(spec, latent_classes=1)
        pindex = build_param_index(spec)
        class_offsets = list(pindex.get("class_offsets", [i * build_param_index(base_spec)["total_params"] for i in range(C)]))
        class_K_base  = list(pindex.get("class_K_base", [build_param_index(base_spec)["total_params"]] * C))
        total_theta = class_offsets[-1] + class_K_base[-1] if C > 0 else 0
        gamma_size = (C - 1) * (K_mem + 1)
        gamma = params[total_theta : total_theta + gamma_size].reshape(C - 1, K_mem + 1)

        n = data["y"].shape[0]
        if K_mem > 0:
            z = np.mean(np.asarray(data["Xmem"]), axis=1)
            z_full = np.concatenate([np.ones((n, 1)), z], axis=1)
        else:
            z_full = np.ones((n, 1))

        logits = z_full @ gamma.T
        logits_full = np.concatenate([np.zeros((n, 1)), logits], axis=1)
        logits_full = logits_full - logits_full.max(axis=1, keepdims=True)
        probs = np.exp(logits_full)
        probs = probs / probs.sum(axis=1, keepdims=True)

        ids = np.asarray(data.get("ids", self.df[[self.id_col]].drop_duplicates()[self.id_col].to_numpy()))
        out = pd.DataFrame({self.id_col: ids})
        for idx in range(C):
            out[f"class_{idx + 1}_prob"] = probs[:, idx]
        out = out.drop_duplicates(subset=[self.id_col]).reset_index(drop=True)

        if true_class_col is not None:
            self._ensure_columns_exist([true_class_col], "true_class_col")
            truth = self.df[[self.id_col, true_class_col]].drop_duplicates(subset=[self.id_col])
            out = out.merge(truth, on=self.id_col, how="left")
        return out

    def print_coefficients(self, fit_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Efficiently print model coefficients from a fitted model.

        After calling fit_manual_model(), pass the result to this method to
        display a clean coefficient table with estimates and standard errors.

        Parameters
        ----------
        fit_result
            Dictionary returned by fit_manual_model()

        Returns
        -------
        pd.DataFrame
            Coefficient table as a DataFrame

        Example
        -------
        best_spec = evaluator.build_spec(result_full["best_solution"])
        fit_full = builder.fit_manual_model(manual_spec=best_spec, model="nb")
        coef_table = builder.print_coefficients(fit_full)
        print(coef_table)
        """
        spec = fit_result["spec"]
        result = fit_result["result"]
        param_index = fit_result["param_index"]
        params = np.asarray(result.params)

        # Build coefficient table with fixed, random, and dispersion parameters
        coef_rows = []

        def _add_rows(index_map: Dict[str, Any], local_params: np.ndarray, class_label: Optional[str] = None) -> None:
            label_suffix = f" [{class_label}]" if class_label else ""

            # ── Fixed coefficients ─────────────────────────────────────
            if spec.Kf > 0 and "fixed" in index_map:
                fixed_names = list(spec.fixed_names)
                fixed_start, fixed_end = index_map["fixed"]
                for name, value in zip(fixed_names, local_params[fixed_start:fixed_end]):
                    coef_rows.append({
                        "Parameter": f"{name}{label_suffix}",
                        "Type": "Fixed" if class_label is None else f"Fixed ({class_label})",
                        "Estimate": value,
                    })

            # ── Random independent (means) ─────────────────────────────
            if spec.Kr_ind > 0:
                ind_names = list(spec.random_ind_names)
                if "ind_mean" in index_map:
                    mean_start, mean_end = index_map["ind_mean"]
                    for name, value in zip(ind_names, local_params[mean_start:mean_end]):
                        coef_rows.append({
                            "Parameter": f"{name} (ind. mean){label_suffix}",
                            "Type": "Random-Ind" if class_label is None else f"Random-Ind ({class_label})",
                            "Estimate": value,
                        })

                if "ind_sd" in index_map:
                    sd_start, sd_end = index_map["ind_sd"]
                    for name, value in zip(ind_names, local_params[sd_start:sd_end]):
                        coef_rows.append({
                            "Parameter": f"{name} (ind. SD){label_suffix}",
                            "Type": "Random-Ind" if class_label is None else f"Random-Ind ({class_label})",
                            "Estimate": value,
                        })

            # ── Random correlated (means) ──────────────────────────────
            if spec.Kr_cor > 0:
                cor_names = list(spec.random_cor_names)
                if "cor_mean" in index_map:
                    mean_start, mean_end = index_map["cor_mean"]
                    for name, value in zip(cor_names, local_params[mean_start:mean_end]):
                        coef_rows.append({
                            "Parameter": f"{name} (cor. mean){label_suffix}",
                            "Type": "Random-Cor" if class_label is None else f"Random-Cor ({class_label})",
                            "Estimate": value,
                        })

            # ── Grouped effects ────────────────────────────────────────
            if spec.Kg > 0:
                grouped_names = list(spec.grouped_names)
                if "group_mean" in index_map:
                    mean_start, mean_end = index_map["group_mean"]
                    for name, value in zip(grouped_names, local_params[mean_start:mean_end]):
                        coef_rows.append({
                            "Parameter": f"{name} (group mean){label_suffix}",
                            "Type": "Grouped" if class_label is None else f"Grouped ({class_label})",
                            "Estimate": value,
                        })

                if "group_sd" in index_map:
                    sd_start, sd_end = index_map["group_sd"]
                    for name, value in zip(grouped_names, local_params[sd_start:sd_end]):
                        coef_rows.append({
                            "Parameter": f"{name} (group SD){label_suffix}",
                            "Type": "Grouped" if class_label is None else f"Grouped ({class_label})",
                            "Estimate": value,
                        })

            # ── Dispersion parameter (for negative binomial) ───────────
            if spec.model == "nb" and "dispersion" in index_map:
                disp_idx = index_map["dispersion"]
                coef_rows.append({
                    "Parameter": f"Dispersion{label_suffix}",
                    "Type": "Dispersion" if class_label is None else f"Dispersion ({class_label})",
                    "Estimate": local_params[disp_idx],
                })

        if spec.latent_classes > 1 and "class_params" in param_index:
            base_spec = replace(spec, latent_classes=1)
            C = int(spec.latent_classes)
            _class_offsets = param_index.get("class_offsets", tuple(i * param_index.get("K_base", param_index["total_params"] // C) for i in range(C)))
            _class_K_base  = param_index.get("class_K_base", (param_index.get("K_base", param_index["total_params"] // C),) * C)

            for c in range(C):
                oc = _class_offsets[c]
                kc = _class_K_base[c]
                class_slice = params[oc:oc + kc]
                _base_idx_c = build_base_index(base_spec, model=param_index.get("class_models", (base_spec.model,) * C)[c])
                _add_rows(_base_idx_c, class_slice, class_label=f"Class {c + 1}")

            class_params_end = param_index["class_params"][1]
            logits_tail = params[class_params_end:]
            if logits_tail.size > 0:
                for idx, value in enumerate(logits_tail, start=1):
                    coef_rows.append({
                        "Parameter": f"Class logit gamma {idx}",
                        "Type": "Class-Logits",
                        "Estimate": value,
                    })
        else:
            _add_rows(param_index, params)

        # Build DataFrame and print
        coef_df = pd.DataFrame(coef_rows)

        print("\n" + "=" * 80)
        print(f"  MODEL COEFFICIENTS  —  {spec.model.upper()} MODEL")
        print("=" * 80 + "\n")

        if len(coef_df) > 0:
            # Group by type for better readability
            for type_name in ["Fixed", "Random-Ind", "Random-Cor", "Grouped", "Dispersion", "Class-Logits"]:
                subset = coef_df[coef_df["Type"] == type_name]
                if len(subset) > 0:
                    print(f"  {type_name.upper()} PARAMETERS:")
                    print(f"  {'-' * 76}")
                    for _, row in subset.iterrows():
                        print(f"    {row['Parameter']:30s} = {row['Estimate']:+.6f}")
                    print()

            # Print latent-class blocks when present.
            class_types = [t for t in coef_df["Type"].unique() if "Class" in str(t) and t != "Class-Logits"]
            for type_name in class_types:
                subset = coef_df[coef_df["Type"] == type_name]
                if len(subset) > 0:
                    print(f"  {str(type_name).upper()} PARAMETERS:")
                    print(f"  {'-' * 76}")
                    for _, row in subset.iterrows():
                        print(f"    {row['Parameter']:30s} = {row['Estimate']:+.6f}")
                    print()

        print("=" * 80 + "\n")

        return coef_df[["Parameter", "Type", "Estimate"]]

    # ── print_cmf_interpretation ────────────────────────────────────

    def print_cmf_interpretation(self, fit_result: Dict[str, Any], aadt_col: Optional[str] = None, aadt_median: Optional[float] = None) -> pd.DataFrame:
        """
        Print CMF (Crash Modification Factor) interpretations for fitted model coefficients.

        For each fixed coefficient, this method computes and displays:
        - The coefficient value (β)
        - The CMF for a one-unit increase: CMF = exp(β)
        - The percent change: 100 × (exp(β) - 1)
        - Plain-language interpretation

        Parameters
        ----------
        fit_result
            Dictionary returned by fit_manual_model()
        aadt_col
            Optional: Column name containing AADT values for context.
            If provided, AADT-dependent interpretations are generated.
        aadt_median
            Optional: Median AADT value for computing traffic-dependent effects.
            If not provided, will compute from aadt_col if available.

        Returns
        -------
        pd.DataFrame
            CMF interpretation table with columns:
            - Parameter: Coefficient name
            - Coefficient: Estimated value (β)
            - CMF (+1): exp(β) for one-unit increase
            - Percent Change: 100 × (exp(β) - 1)
            - Interpretation: Plain-language explanation

        Example
        -------
        cmf_table = builder.print_cmf_interpretation(
            fit_result=fit_result,
            aadt_col='AADT',
            aadt_median=23771
        )
        print(cmf_table)
        """
        import math
        
        spec = fit_result["spec"]
        result = fit_result["result"]
        param_index = fit_result["param_index"]
        params = np.asarray(result.params)

        # Compute AADT median if provided column but no explicit median
        if aadt_col is not None and aadt_col in self.df.columns and aadt_median is None:
            aadt_median = self.df[aadt_col].median()

        cmf_rows = []

        def _extract_fixed_coefs(index_map: Dict[str, Any], local_params: np.ndarray, class_label: Optional[str] = None) -> None:
            label_suffix = f" [{class_label}]" if class_label else ""

            if spec.Kf > 0 and "fixed" in index_map:
                fixed_names = list(spec.fixed_names)
                fixed_start, fixed_end = index_map["fixed"]
                
                for name, value in zip(fixed_names, local_params[fixed_start:fixed_end]):
                    try:
                        if str(name) == "__INTERCEPT__":
                            continue

                        coef_value = float(value)
                        
                        # Compute CMF for one-unit change
                        if math.isfinite(coef_value):
                            cmf_one_unit = math.exp(max(min(coef_value, 700.0), -700.0))
                            percent_change = 100.0 * (cmf_one_unit - 1.0)
                            
                            # Generate interpretation
                            lower_name = str(name).lower()
                            is_cmf_local_term = "__cmf_local__" in lower_name
                            is_log_aadt_term = "__cmf_log_aadt" in lower_name or lower_name == "aadt"

                            aadt_context = ""
                            if is_log_aadt_term:
                                interpretation = (
                                    f"{name}: traffic elasticity = {coef_value:+.4f}; "
                                    f"1% AADT change implies about {coef_value:+.2f}% crash change"
                                )
                            elif is_cmf_local_term and aadt_median is not None and aadt_median > 0:
                                try:
                                    exponent = max(min(coef_value * math.log(aadt_median), 700.0), -700.0)
                                    aadt_effect = 100.0 * (math.exp(exponent) - 1.0)
                                    aadt_context = f" (at median AADT {aadt_median:,.0f}: {aadt_effect:+.2f}%)"
                                except (ValueError, OverflowError):
                                    pass
                                interpretation = f"{name} +1 adjusts AADT-response scaling{aadt_context}"
                            else:
                                if percent_change < 0:
                                    interpretation = f"{name} +1 → {percent_change:.2f}% crashes (safer)"
                                elif percent_change > 0:
                                    interpretation = f"{name} +1 → +{percent_change:.2f}% crashes (riskier)"
                                else:
                                    interpretation = f"{name} +1 → No change (neutral)"
                            
                            cmf_rows.append({
                                "Parameter": f"{name}{label_suffix}",
                                "Type": "Fixed" if class_label is None else f"Fixed ({class_label})",
                                "Coefficient": coef_value,
                                "CMF(+1)": cmf_one_unit,
                                "Percent Change": percent_change,
                                "Interpretation": interpretation,
                            })
                    except (ValueError, OverflowError):
                        pass

        if spec.latent_classes > 1 and "class_params" in param_index:
            base_spec = replace(spec, latent_classes=1)
            C = int(spec.latent_classes)
            _class_offsets = param_index.get("class_offsets", tuple(i * param_index.get("K_base", param_index["total_params"] // C) for i in range(C)))
            _class_K_base  = param_index.get("class_K_base", (param_index.get("K_base", param_index["total_params"] // C),) * C)

            for c in range(C):
                oc = _class_offsets[c]
                kc = _class_K_base[c]
                class_slice = params[oc:oc + kc]
                _base_idx_c = build_base_index(base_spec, model=param_index.get("class_models", (base_spec.model,) * C)[c])
                _extract_fixed_coefs(_base_idx_c, class_slice, class_label=f"Class {c + 1}")
        else:
            _extract_fixed_coefs(param_index, params)

        cmf_df = pd.DataFrame(cmf_rows)

        print("\n" + "=" * 100)
        print(f"  CMF INTERPRETATIONS  —  {spec.model.upper()} MODEL")
        print("=" * 100 + "\n")

        if len(cmf_df) > 0:
            for _, row in cmf_df.iterrows():
                print(f"  {row['Parameter']:25s}")
                print(f"    Coefficient (β)     : {row['Coefficient']:+.6f}")
                print(f"    CMF for +1 unit     : {row['CMF(+1)']:.4f}")
                print(f"    Percent Change      : {row['Percent Change']:+.2f}%")
                print(f"    ➜ {row['Interpretation']}")
                print()
        else:
            print("  (No fixed coefficients found for CMF interpretation)")
            print()

        print("=" * 100)
        print("  INTERPRETATION GUIDE:")
        print("  ─────────────────────────────────────────────────────────────────────────────────────────────────────")
        print("  CMF < 1.0  (Percent Change < 0)  →  Safer treatment (crashes decrease)")
        print("  CMF = 1.0  (Percent Change = 0)  →  Neutral effect (no change)")
        print("  CMF > 1.0  (Percent Change > 0)  →  Riskier treatment (crashes increase)")
        print("=" * 100 + "\n")

        return cmf_df[["Parameter", "Type", "Coefficient", "CMF(+1)", "Percent Change", "Interpretation"]]

    # ── validate_before_fit ─────────────────────────────────────────

    def validate_before_fit(
        self,
        variables: Optional[List[str]] = None,
        aadt_col: Optional[str] = None,
        vif_threshold: float = 10.0,
        raw_aadt_threshold: float = 500.0,
    ) -> dict:
        """
        Pre-fit data quality checks. Warns on:

          1. AADT / exposure columns that appear un-logged
             (median > raw_aadt_threshold → likely raw vehicle counts).
          2. High collinearity: VIF > vif_threshold for any variable pair.
          3. Near-constant or all-zero columns (zero-variance).
          4. Outcome variable: variance-to-mean ratio (overdispersion flag).

        Returns a dict with keys 'warnings' (list[str]) and 'vif_table'
        (pd.DataFrame or None).
        """
        cols = variables or self._candidate_vars
        warnings_out: List[str] = []

        print("\n" + "=" * 70)
        print("  PRE-FIT VALIDATION")
        print("=" * 70)

        # ── 1. AADT / exposure log-transform check ──────────────────
        aadt_candidates = []
        if aadt_col is not None:
            aadt_candidates = [aadt_col]
        else:
            aadt_candidates = [c for c in cols
                               if any(kw in c.upper()
                                      for kw in ("AADT", "ADT", "VOLUME", "EXPO", "FLOW"))]

        for col in aadt_candidates:
            if col not in self.df.columns:
                continue
            med = self.df[col].median()
            mn  = self.df[col].min()
            if med > raw_aadt_threshold and mn >= 0:
                msg = (
                    f"  [WARN] '{col}': median={med:,.0f}  min={mn:.2f}  "
                    f"→ looks like RAW traffic volume (not log-transformed). "
                    f"If this is AADT, apply log({col}) before fitting."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                print(msg)
                warnings_out.append(msg)
            elif med < 0:
                print(f"  [OK]   '{col}': median={med:.3f}  (looks log-transformed)")
            else:
                print(f"  [OK]   '{col}': median={med:.3f}  "
                      f"(within reasonable range for a log-transformed or standardised variable)")

        # ── 2. Near-constant / zero-variance ────────────────────────
        for col in cols:
            if col not in self.df.columns:
                continue
            s = self.df[col]
            if s.nunique() <= 1:
                msg = f"  [WARN] '{col}': near-constant (nunique={s.nunique()}) — will be excluded automatically."
                warnings.warn(msg, UserWarning, stacklevel=2)
                print(msg)
                warnings_out.append(msg)

        # ── 3. VIF collinearity check ────────────────────────────────
        numeric_cols = [c for c in cols
                        if c in self.df.columns
                        and pd.api.types.is_numeric_dtype(self.df[c])
                        and self.df[c].nunique() > 1]

        vif_table = None
        if len(numeric_cols) >= 2:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                X = self.df[numeric_cols].dropna()
                X = (X - X.mean()) / X.std().replace(0, 1)  # standardise for VIF stability
                X_np = X.values
                vif_vals = [
                    variance_inflation_factor(X_np, i)
                    for i in range(X_np.shape[1])
                ]
                vif_table = pd.DataFrame({"variable": numeric_cols, "VIF": vif_vals})
                vif_table = vif_table.sort_values("VIF", ascending=False)

                print(f"\n  Variance Inflation Factors  (threshold={vif_threshold}):")
                print(f"  {'Variable':<25}  {'VIF':>8}")
                print("  " + "-" * 36)
                for _, row in vif_table.iterrows():
                    flag = "  ← HIGH COLLINEARITY" if row["VIF"] > vif_threshold else ""
                    print(f"  {row['variable']:<25}  {row['VIF']:>8.2f}{flag}")
                    if row["VIF"] > vif_threshold:
                        msg = (
                            f"  [WARN] '{row['variable']}': VIF={row['VIF']:.1f} > {vif_threshold} "
                            f"— consider dropping or orthogonalising this variable."
                        )
                        warnings.warn(msg, UserWarning, stacklevel=2)
                        warnings_out.append(msg)
            except ImportError:
                print("  [INFO] statsmodels not available — skipping VIF check.")
            except Exception as exc:
                print(f"  [INFO] VIF check skipped: {exc}")

        # ── 4. Outcome overdispersion ────────────────────────────────
        y = self.df[self.y_col]
        vr = y.var() / y.mean() if y.mean() > 0 else 0.0
        zero_pct = 100.0 * (y == 0).mean()
        print(f"\n  Outcome '{self.y_col}':  mean={y.mean():.3f}  "
              f"var/mean={vr:.2f}  zeros={zero_pct:.1f}%")
        if vr > 1.5:
            print("  [OK]   Overdispersed → NB2 (dispersion=1) recommended.")
        elif vr < 0.8:
            msg = (f"  [WARN] '{self.y_col}': var/mean={vr:.2f} < 1 "
                   f"— underdispersed; verify this is a count outcome.")
            warnings.warn(msg, UserWarning, stacklevel=2)
            print(msg)
            warnings_out.append(msg)
        else:
            print("  [OK]   Near-Poisson dispersion.")

        if len(warnings_out) == 0:
            print("\n  No issues found.")
        else:
            print(f"\n  {len(warnings_out)} warning(s) raised — review before fitting.")

        print("=" * 70 + "\n")

        return {"warnings": warnings_out, "vif_table": vif_table}

    # ── smoke_test ───────────────────────────────────────────────────

    def smoke_test(
        self,
        fixed_terms: Optional[List[str]] = None,
        model: str = "nb",
        latent_classes: int = 2,
        R: int = 50,
    ) -> bool:
        """
        Quick end-to-end smoke test of the fitting pipeline.

        Fits a small 2-class NB model on the first 3 fixed_terms (or
        the first 3 candidate variables) to verify the full chain:
          data prep → warm start → EM → LBFGS polish → summary

        Returns True on success, False on any exception.

        Usage::

            ok = builder.smoke_test()
            assert ok, "Fitting pipeline smoke test failed"
        """
        print("\n" + "=" * 70)
        print("  SMOKE TEST  —  fitting pipeline")
        print("=" * 70)

        terms = fixed_terms or self._candidate_vars[:3]
        terms = [t for t in terms if t in self.df.columns][:3]
        if len(terms) < 1:
            print("  [SKIP] No candidate variables available for smoke test.")
            return False

        spec = self.make_manual_spec(
            fixed_terms=terms,
            dispersion=1 if model == "nb" else 0,
            latent_classes=latent_classes,
        )

        print(f"  Spec: model={model}  classes={latent_classes}  "
              f"fixed_terms={terms}  R={R}")

        try:
            fit = self.fit_manual_model(manual_spec=spec, model=model, R=R,
                                        print_report=False)
            summary = fit.get("summary", {})
            ll  = summary.get("loglik", float("nan"))
            bic = summary.get("bic",    float("nan"))
            k   = summary.get("num_parm", "?")
            print(f"  Result : LL={ll:.2f}  k={k}  BIC={bic:.2f}")
            if not np.isfinite(bic):
                print("  [WARN] Non-finite BIC — optimizer may have diverged.")
                return False
            print("  PASS — fitting pipeline is operational.")
            print("=" * 70 + "\n")
            return True
        except Exception as exc:
            print(f"  FAIL — {exc}")
            print("=" * 70 + "\n")
            return False

    # ── fit_split_class_models ──────────────────────────────────────

    def fit_split_class_models(
        self,
        split_col: str,
        fixed_terms: List[str],
        model: str = "nb",
        dispersion: int = 1,
        membership_terms: Optional[List[str]] = None,
        R: int = 200,
        hypothesis_label: str = "split-class hypothesis",
    ) -> dict:
        """
        Parallel validation of a latent-class hypothesis via observed splits.

        Fits:
          (a) A joint LC-2 model on the full dataset  (split_col as membership)
          (b) A separate NB model on each level of split_col  (stratified fit)

        This answers: "does an observed grouping variable (e.g. urban/rural)
        explain latent class membership as well as the joint LC model?"

        If the split-class models together achieve lower joint BIC than the
        LC model (BIC_split_A + BIC_split_B < BIC_lc), the observed split is
        informationally sufficient and the LC structure may be redundant.
        Conversely, if the LC model has much lower BIC, the latent structure
        captures heterogeneity that the observed split misses.

        Parameters
        ----------
        split_col : str
            Binary or low-cardinality column defining the observed groups
            (e.g. 'URB', 'FC', 'road_class').
        fixed_terms : list[str]
            Outcome-equation variables for all models.
        model : str
            Model family — 'nb' or 'poisson'.
        dispersion : int
            1 for NB2, 0 for Poisson.
        membership_terms : list[str], optional
            Additional membership variables for the joint LC model.
            split_col is always included.
        R : int
            Halton draws.
        hypothesis_label : str
            Label printed in the report header.

        Returns
        -------
        dict with keys:
            'lc_fit'      — full LC fit result
            'split_fits'  — {level: fit_result} for each observed class
            'comparison'  — pd.DataFrame ranked by BIC
        """
        self._ensure_columns_exist([split_col], "fit_split_class_models")

        levels = sorted(self.df[split_col].dropna().unique())
        if len(levels) < 2:
            raise ValueError(f"split_col '{split_col}' must have at least 2 distinct values.")
        if len(levels) > 6:
            raise ValueError(
                f"split_col '{split_col}' has {len(levels)} levels — "
                "use a binary or low-cardinality column."
            )

        print("\n" + "=" * 72)
        print(f"  SPLIT-CLASS HYPOTHESIS VALIDATION")
        print(f"  Hypothesis : {hypothesis_label}")
        print(f"  Split col  : '{split_col}'  levels={levels}")
        print(f"  Fixed terms: {fixed_terms}")
        print("=" * 72)

        results = {}
        comparison_rows = []

        # ── (a) Joint LC model with split_col as membership ──────────
        mem_terms = list(dict.fromkeys([split_col] + list(membership_terms or [])))
        print(f"\n  [LC]   Fitting joint {len(levels)}-class model "
              f"(membership={mem_terms}) ...")
        try:
            lc_spec = self.make_manual_spec(
                fixed_terms=fixed_terms,
                membership_terms=mem_terms,
                dispersion=dispersion,
                latent_classes=len(levels),
            )
            lc_fit = self.fit_manual_model(
                manual_spec=lc_spec, model=model, R=R, print_report=False
            )
            lc_summary = lc_fit.get("summary", {})
            lc_ll  = lc_summary.get("loglik", float("nan"))
            lc_bic = lc_summary.get("bic",    float("nan"))
            lc_k   = lc_summary.get("num_parm", "?")
            print(f"         LL={lc_ll:.2f}  k={lc_k}  BIC={lc_bic:.2f}")
            results["lc_fit"] = lc_fit
            comparison_rows.append({
                "Model": f"LC-{len(levels)} joint  (membership={split_col})",
                "N": len(self.df),
                "LL": lc_ll,
                "k": lc_k,
                "BIC": lc_bic,
                "type": "lc",
            })
        except Exception as exc:
            print(f"  [LC]   FAILED: {exc}")
            results["lc_fit"] = None
            comparison_rows.append({
                "Model": f"LC-{len(levels)} joint", "N": len(self.df),
                "LL": float("nan"), "k": "?", "BIC": float("nan"), "type": "lc",
            })

        # ── (b) Stratified models per observed class ─────────────────
        split_fits = {}
        split_bic_sum = 0.0
        split_ll_sum  = 0.0

        for level in levels:
            mask = self.df[split_col] == level
            df_sub = self.df[mask].reset_index(drop=True)
            n_sub  = df_sub[self.id_col].nunique()
            label  = f"{split_col}={level}"
            print(f"\n  [SPLIT] Fitting {model.upper()} on '{label}' "
                  f"(n_ids={n_sub}) ...")
            try:
                sub_builder = ExperimentBuilder(
                    df=df_sub,
                    id_col=self.id_col,
                    y_col=self.y_col,
                    offset_col=self.offset_col,
                )
                sub_spec = sub_builder.make_manual_spec(
                    fixed_terms=fixed_terms,
                    dispersion=dispersion,
                    latent_classes=1,
                )
                sub_fit = sub_builder.fit_manual_model(
                    manual_spec=sub_spec, model=model, R=R, print_report=False
                )
                sub_summary = sub_fit.get("summary", {})
                sub_ll  = sub_summary.get("loglik", float("nan"))
                sub_bic = sub_summary.get("bic",    float("nan"))
                sub_k   = sub_summary.get("num_parm", "?")
                print(f"         LL={sub_ll:.2f}  k={sub_k}  BIC={sub_bic:.2f}")
                split_fits[level] = sub_fit
                split_bic_sum += sub_bic if np.isfinite(sub_bic) else 0.0
                split_ll_sum  += sub_ll  if np.isfinite(sub_ll)  else 0.0
                comparison_rows.append({
                    "Model": f"Single-class {model.upper()} | {label}",
                    "N": n_sub,
                    "LL": sub_ll,
                    "k": sub_k,
                    "BIC": sub_bic,
                    "type": "split",
                })
            except Exception as exc:
                print(f"  [SPLIT] '{label}' FAILED: {exc}")
                split_fits[level] = None
                comparison_rows.append({
                    "Model": f"Single-class | {label}", "N": n_sub,
                    "LL": float("nan"), "k": "?", "BIC": float("nan"), "type": "split",
                })

        results["split_fits"] = split_fits

        # ── Summary table ────────────────────────────────────────────
        print("\n\n" + "=" * 72)
        print(f"  SPLIT-CLASS COMPARISON  —  {hypothesis_label}")
        print("=" * 72)
        df_cmp = pd.DataFrame(comparison_rows)
        df_cmp = df_cmp.sort_values("BIC")
        results["comparison"] = df_cmp

        best_bic = df_cmp["BIC"].min()
        hdr = (f"  {'Model':<46}  {'N':>6}  {'LL':>10}  {'k':>4}  "
               f"{'BIC':>10}  {'dBIC':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for _, row in df_cmp.iterrows():
            dbic = row["BIC"] - best_bic if np.isfinite(row["BIC"]) else float("nan")
            ll_s  = f"{row['LL']:>10.1f}"  if np.isfinite(row["LL"])  else f"{'—':>10}"
            bic_s = f"{row['BIC']:>10.1f}" if np.isfinite(row["BIC"]) else f"{'FAILED':>10}"
            dbi_s = f"{dbic:>+8.1f}"       if np.isfinite(dbic)        else f"{'—':>8}"
            print(f"  {row['Model']:<46}  {row['N']:>6}  {ll_s}  "
                  f"{row['k']:>4}  {bic_s}  {dbi_s}")

        # ── Interpretation ────────────────────────────────────────────
        if np.isfinite(split_bic_sum) and split_bic_sum > 0:
            lc_bic_val = next(
                (r["BIC"] for r in comparison_rows if r["type"] == "lc"),
                float("nan"),
            )
            print(f"\n  Summed BIC (stratified splits) : {split_bic_sum:.1f}")
            if np.isfinite(lc_bic_val):
                print(f"  Joint LC BIC                   : {lc_bic_val:.1f}")
                delta = split_bic_sum - lc_bic_val
                if delta > 0:
                    print(f"  → LC model preferred by ΔBIC={delta:.1f}: "
                          "latent structure captures heterogeneity the observed "
                          "split misses.")
                else:
                    print(f"  → Observed split preferred by ΔBIC={-delta:.1f}: "
                          f"'{split_col}' fully explains the class structure; "
                          "LC may be redundant.")

        print("=" * 72 + "\n")
        return results

    # ── describe ────────────────────────────────────────────────────

    def describe(self):
        print("\n" + "=" * 70)
        print("  EXPERIMENT BUILDER  —  Data Summary")
        print("=" * 70)
        print(f"\n  ID column      : {self.id_col}")
        print(f"  Outcome column : {self.y_col}")
        print(f"  Offset column  : {self.offset_col or '(none)'}")
        print(f"  Group column   : {self.group_id_col or '(none)'}")
        print(f"\n  Observations   : {len(self.df):,}")
        print(f"  Unique IDs     : {self.df[self.id_col].nunique():,}")

        y = self.df[self.y_col]
        vr = y.var() / y.mean() if y.mean() > 0 else 0
        print(f"\n  Outcome ({self.y_col}):")
        print(f"    mean     = {y.mean():.4f}")
        print(f"    std      = {y.std():.4f}")
        print(f"    zeros    = {(y == 0).sum()} ({(y==0).mean()*100:.1f}%)")
        print(f"    max      = {y.max()}")
        print(f"    var/mean = {vr:.3f}  "
              f"({'overdispersed → consider NB' if vr > 1.5 else 'near-Poisson'})")

        print(f"\n  Candidate variables ({len(self._candidate_vars)}):\n")
        print(f"  {'Variable':<20} {'Type':<12} {'Unique':>7} "
              f"{'Min':>10} {'Max':>10} {'Mean':>10} {'Zeros%':>8}")
        print("  " + "-" * 80)

        for col in self._candidate_vars:
            s = self.df[col]
            print(f"  {col:<20} {self._infer_type(s):<12} {s.nunique():>7} "
                  f"{s.min():>10.3g} {s.max():>10.3g} "
                  f"{s.mean():>10.3g} {(s==0).mean()*100:>7.1f}%")

        print("\n" + ROLE_GUIDE)

    # ── suggest_config ───────────────────────────────────────────────

    def suggest_config(self,
                       max_latent_classes: int = 1) -> Dict:
        suggestions = {}

        print("\n" + "=" * 70)
        print("  SUGGESTED VARIABLE CONFIGURATION")
        if max_latent_classes > 1:
            print(f"  Latent classes: up to {max_latent_classes}  "
                  f"(roles 7 and 8 are available)")
        print("=" * 70 + "\n")

        for col in self._candidate_vars:
            s        = self.df[col]
            vtype    = self._infer_type(s)
            roles, dists, reason = self._suggest_roles_dists(
                col, s, vtype, max_latent_classes
            )
            roles_str = ", ".join(
                f"{r}={self._ROLE_NAMES[r]}" for r in roles
            )
            print(f"  {col}")
            print(f"    Roles  : [{roles_str}]")
            print(f"    Dists  : {', '.join(dists) if dists else '—'}")
            print(f"    Reason : {reason}\n")

            suggestions[col] = {"roles": roles, "dists": dists}

        return suggestions

    # ── build_evaluator ─────────────────────────────────────────────

    def build_evaluator(
        self,
        variables:           Optional[List[str]]       = None,
        fixed_override:      Optional[Dict[str, list]] = None,
        membership_override: Optional[Dict[str, list]] = None,
        exclude:             Optional[List[str]]       = None,
        mode:                str                       = "single",
        max_latent_classes:  int                       = 1,
        R:                   int                       = 200,
        default_roles:       Optional[list]            = None,
        model_family:        Optional[str]             = None,
        engine:              Optional[str]             = None,
        constraints=None,
        **family_kwargs: Any,
    ):
        """
        Build a StructureEvaluatorLC ready for the search.

        Parameters
        ----------
        variables
            Columns to search over (default: all candidate columns).
        fixed_override
            {col: [allowed_roles]} — override for specific variables.
            Example: {"EXPOSURE": [1]} forces EXPOSURE to fixed-only.
        membership_override
            {col: [allowed_roles]} — used to allow/restrict membership
            roles (7, 8).
            Example: {"URB": [0, 1, 7]} allows URB to be excluded, fixed,
            or membership-only.
        exclude
            Columns to always exclude from the search.
        mode
            "single" (BIC) or "multi" (BIC + test RMSE).
        max_latent_classes
            Maximum number of latent classes.  Set to 1 to disable LC.
        R
            Number of Halton simulation draws.
        default_roles
            Roles available to most variables.
            Defaults to [0,1,2,3,5] when max_latent_classes = 1,
            or [0,1,2,3,5,7,8] when max_latent_classes > 1.
        model_family
            Search family to build. One of: "count", "cmf", "linear", "duration".
        engine
            Execution engine. Defaults to the builder's primary engine, which is JAX.
        """
        model_family = (model_family or self.default_model_family).lower()
        engine = (engine or self.default_engine).lower()

        if engine != "jax":
            raise ValueError("Only the JAX-first engine is supported through ExperimentBuilder.")

        if model_family != "count":
            return self.build_search(
                model_family=model_family,
                variables=variables,
                exclude=exclude,
                mode=mode,
                max_latent_classes=max_latent_classes,
                R=R,
                default_roles=default_roles,
                fixed_override=fixed_override,
                membership_override=membership_override,
                engine=engine,
                **family_kwargs,
            )

        # Merge ModelConstraints (if supplied) with explicit override kwargs.
        # Explicit kwargs always win over constraint-derived defaults.
        if constraints is not None:
            _ckw = constraints.to_evaluator_kwargs()
            fixed_override = {
                **_ckw.get("fixed_override", {}),
                **(fixed_override or {}),
            }
            membership_override = {
                **_ckw.get("membership_override", {}),
                **(membership_override or {}),
            }
            _c_exclude = _ckw.get("exclude", [])
            exclude = list(dict.fromkeys(list(_c_exclude) + list(exclude or [])))
            # Distribution overrides from constraints (merged, explicit wins)
            if "dist_override" in _ckw:
                family_kwargs.setdefault("dist_override", {})
                family_kwargs["dist_override"] = {
                    **_ckw["dist_override"],
                    **family_kwargs["dist_override"],
                }

        variables = self._normalize_variables(variables, exclude)
        fixed_override = self._normalize_override_map(fixed_override, "fixed_override")
        membership_override = self._normalize_override_map(membership_override, "membership_override")

        if default_roles is None:
            if max_latent_classes > 1:
                default_roles = [0, 1, 2, 3, 5, 7, 8]
            else:
                default_roles = [0, 1, 2, 3, 5]

        # Merge overrides: membership_override takes priority for those vars
        merged_override = {**fixed_override, **membership_override}

        allowed_roles = populate_allowed_roles(
            variables, merged_override, default_roles=default_roles
        )
        allowed_dists = populate_allowed_distributions(variables, None)

        self._evaluator = StructureEvaluatorLC(
            df                    = self.df,
            id_col                = self.id_col,
            y_col                 = self.y_col,
            offset_col            = self.offset_col,
            all_variables         = variables,
            allowed_roles         = allowed_roles,
            allowed_distributions = allowed_dists,
            group_id_col          = self.group_id_col,
            mode                  = mode,
            R                     = R,
            max_latent_classes    = max_latent_classes,
        )

        D   = len(variables)
        _dim = 3 * D + 2 if max_latent_classes > 1 else 2 * D + 1
        _dim_note = f"3×{D} + 2  (roles+dists+dispersion+LC+class_masks)" if max_latent_classes > 1 else f"2×{D} + 1  (roles+dists+dispersion)"
        print(f"\n  Evaluator ready:")
        print(f"    Variables          : {D}")
        print(f"    Decision dimension : {_dim}  ({_dim_note})")
        print(f"    Max latent classes : {max_latent_classes}")
        print(f"    Mode               : {mode}")
        print(f"    Draws (R)          : {R}")
        if max_latent_classes > 1:
            print(f"\n  Membership roles 7 and 8 are active.")
            print(f"  Role 7 = membership-only  (no outcome effect)")
            print(f"  Role 8 = membership + fixed outcome (class-specific beta)")
            print(f"  LC models are warm-started from the single-class solution.\n")

        return self._evaluator

    def build_search(
        self,
        model_family: Optional[str] = None,
        variables: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        engine: Optional[str] = None,
        **kwargs,
    ):
        model_family = (model_family or self.default_model_family).lower()
        engine = (engine or self.default_engine).lower()
        explicit_variables = variables

        if engine != "jax":
            raise ValueError("Only the JAX-first engine is supported through ExperimentBuilder.")

        variables = self._normalize_variables(variables, exclude)

        if model_family == "count":
            return self.build_evaluator(
                variables=variables,
                exclude=exclude,
                model_family="count",
                engine=engine,
                **kwargs,
            )

        if model_family == "linear":
            linear_driver = str(kwargs.pop("linear_driver", "jax_hierarchical")).lower()
            objective_kwargs = kwargs.pop("objective_kwargs", {})
            if linear_driver in {"jax", "jax_hierarchical", "main"}:
                mode = kwargs.pop("mode", "single")
                max_latent_classes = kwargs.pop("max_latent_classes", 2)
                R = kwargs.pop("R", 200)
                default_roles = kwargs.pop("default_roles", None)
                fixed_override = self._normalize_override_map(kwargs.pop("fixed_override", None), "fixed_override")
                membership_override = self._normalize_override_map(kwargs.pop("membership_override", None), "membership_override")
                if default_roles is None:
                    default_roles = [0, 1, 2, 3, 4, 5, 6]
                    if max_latent_classes > 1:
                        default_roles.extend([7, 8])
                evaluator = ForcedModelStructureEvaluatorLC(
                    df=self.df,
                    id_col=self.id_col,
                    y_col=self.y_col,
                    offset_col=self.offset_col,
                    all_variables=variables,
                    allowed_roles=populate_allowed_roles(variables, {**fixed_override, **membership_override}, default_roles=default_roles),
                    allowed_distributions=populate_allowed_distributions(variables, None),
                    group_id_col=self.group_id_col,
                    mode=mode,
                    R=R,
                    max_latent_classes=max_latent_classes,
                    forced_model="gaussian",
                )
                self._raise_on_unused_kwargs(kwargs, "linear search")
                return LinearSearchProblem(
                    builder=self,
                    evaluator=evaluator,
                    metadata={"model": "gaussian", "variables": variables, "max_latent_classes": max_latent_classes},
                )
            self._raise_on_unused_kwargs(kwargs, "legacy linear search")
            return LinearSearchProblem(
                df=self.df,
                y_col=self.y_col,
                variables=variables,
                objective_kwargs=objective_kwargs,
            )

        if model_family == "duration":
            budget_col = kwargs.pop("budget_col", "B")
            if budget_col not in self.df.columns:
                raise ValueError(f"Duration search requires budget_col='{budget_col}' in the dataframe.")
            duration_driver = str(kwargs.pop("duration_driver", "jax_hierarchical")).lower()
            if duration_driver in {"jax", "jax_hierarchical", "main"}:
                mode = kwargs.pop("mode", "single")
                max_latent_classes = kwargs.pop("max_latent_classes", 2)
                R = kwargs.pop("R", 200)
                default_roles = kwargs.pop("default_roles", None)
                fixed_override = self._normalize_override_map(kwargs.pop("fixed_override", None), "fixed_override")
                membership_override = self._normalize_override_map(kwargs.pop("membership_override", None), "membership_override")
                duration_variables = list(dict.fromkeys([*variables, budget_col]))
                if default_roles is None:
                    default_roles = [0, 1, 2, 3, 4, 5, 6]
                    if max_latent_classes > 1:
                        default_roles.extend([7, 8])
                evaluator = ForcedModelStructureEvaluatorLC(
                    df=self.df,
                    id_col=self.id_col,
                    y_col=self.y_col,
                    offset_col=self.offset_col,
                    all_variables=duration_variables,
                    allowed_roles=populate_allowed_roles(duration_variables, {**fixed_override, **membership_override}, default_roles=default_roles),
                    allowed_distributions=populate_allowed_distributions(duration_variables, None),
                    group_id_col=self.group_id_col,
                    mode=mode,
                    R=R,
                    max_latent_classes=max_latent_classes,
                    forced_model="lognormal",
                )
                self._raise_on_unused_kwargs(kwargs, "duration search")
                return DurationSearchProblem(
                    builder=self,
                    evaluator=evaluator,
                    metadata={"model": "lognormal", "variables": duration_variables, "budget_col": budget_col, "max_latent_classes": max_latent_classes},
                )
            self._raise_on_unused_kwargs(kwargs, "duration search")
            return DurationSearchProblem(
                df=self.df.copy(),
                y_col=self.y_col,
                variables=variables,
                id_col=self.id_col,
                budget_col=budget_col,
            )

        if model_family == "tobit":
            # Left-censored (at 0) linear model — random-parameters and
            # latent-class variants are fully supported.
            mode               = kwargs.pop("mode", "single")
            max_latent_classes = kwargs.pop("max_latent_classes", 2)
            R                  = kwargs.pop("R", 200)
            default_roles      = kwargs.pop("default_roles", None)
            fixed_override     = self._normalize_override_map(
                kwargs.pop("fixed_override", None), "fixed_override")
            membership_override = self._normalize_override_map(
                kwargs.pop("membership_override", None), "membership_override")
            if default_roles is None:
                default_roles = [0, 1, 2, 3, 4, 5, 6]
                if max_latent_classes > 1:
                    default_roles.extend([7, 8])
            evaluator = ForcedModelStructureEvaluatorLC(
                df=self.df,
                id_col=self.id_col,
                y_col=self.y_col,
                offset_col=self.offset_col,
                all_variables=variables,
                allowed_roles=populate_allowed_roles(
                    variables,
                    {**fixed_override, **membership_override},
                    default_roles=default_roles,
                ),
                allowed_distributions=populate_allowed_distributions(variables, None),
                group_id_col=self.group_id_col,
                mode=mode,
                R=R,
                max_latent_classes=max_latent_classes,
                forced_model="tobit",
            )
            self._raise_on_unused_kwargs(kwargs, "tobit search")
            # Re-use LinearSearchProblem as the search wrapper (same SA driver)
            return LinearSearchProblem(
                builder=self,
                evaluator=evaluator,
                metadata={
                    "model": "tobit",
                    "variables": variables,
                    "max_latent_classes": max_latent_classes,
                },
            )

        if model_family == "cmf":
            try:
                from .cmf_package import CMFExperimentBuilder
            except ImportError:
                from cmf_package import CMFExperimentBuilder

            cmf_driver = str(kwargs.pop("cmf_driver", "jax_count")).lower()
            aadt_col = kwargs.pop("aadt_col", None)
            baseline_vars = kwargs.pop("baseline_vars", None)
            local_vars = kwargs.pop("local_vars", None)

            if aadt_col is None or baseline_vars is None or local_vars is None:
                raise ValueError("CMF search requires aadt_col, baseline_vars, and local_vars.")

            cmf_builder = CMFExperimentBuilder(
                df=self.df,
                y_col=self.y_col,
                aadt_col=aadt_col,
                baseline_vars=baseline_vars,
                local_vars=local_vars,
            )
            if cmf_driver in {"jax", "jax_count", "count", "main"}:
                general_builder, evaluator, metadata = cmf_builder.build_jax_count_evaluator(
                    id_col=self.id_col,
                    offset_col=self.offset_col,
                    group_id_col=self.group_id_col,
                    variables=explicit_variables,
                    fixed_override=kwargs.pop("fixed_override", None),
                    membership_override=kwargs.pop("membership_override", None),
                    exclude=exclude,
                    mode=kwargs.pop("mode", "single"),
                    max_latent_classes=kwargs.pop("max_latent_classes", 1),
                    R=kwargs.pop("R", 200),
                    default_roles=kwargs.pop("default_roles", None),
                )
                self._raise_on_unused_kwargs(kwargs, "cmf search")
                return UnifiedCMFSearchProblem(
                    builder=general_builder,
                    evaluator=evaluator,
                    metadata=metadata,
                )

            if cmf_driver not in {"ga", "legacy_ga", "metaheuristic"}:
                raise ValueError(
                    "cmf_driver must be one of: 'jax_count' (default), 'ga', 'legacy_ga', 'metaheuristic'."
                )
            kwargs.pop("mode", None)
            kwargs.pop("max_latent_classes", None)
            kwargs.pop("R", None)
            kwargs.pop("default_roles", None)
            kwargs.pop("fixed_override", None)
            kwargs.pop("membership_override", None)
            self._raise_on_unused_kwargs(kwargs, "legacy cmf search")
            return CMFFamilySearchProblem(
                builder=cmf_builder,
                id_col=self.id_col,
                offset_col=self.offset_col,
                group_id_col=self.group_id_col,
            )

        raise ValueError("model_family must be one of: count, cmf, linear, duration")

    def build_count_evaluator(self, **kwargs):
        kwargs.setdefault("model_family", "count")
        kwargs.setdefault("engine", "jax")
        return self.build_evaluator(**kwargs)

    # ── run ─────────────────────────────────────────────────────────

    def run(
        self,
        evaluator:  Optional[StructureEvaluatorLC] = None,
        algo:       str  = "sa",
        max_iter:   int  = 3000,
        n_jobs:     int  = 1,
        seed:       int  = 0,
        config_id:  int  = 0,
        output_config: Optional[SearchOutputConfig] = None,
        **algo_kwargs,
    ) -> dict:
        """
        Run the metaheuristic search.

        algo : "sa"  Simulated Annealing (recommended for single mode)
               "de"  Differential Evolution NSGA2 (multi mode)
               "hs"  Harmony Search NSGA2 (multi mode)
        """
        evaluator = evaluator or self._evaluator
        if evaluator is None:
            raise RuntimeError("Call build_evaluator() first.")

        D   = len(evaluator.vars)
        # Decision vector: [roles(D) | dists(D) | dispersion_bit | lc_code?]
        # The LC gene (index 2*D+1) is only present when max_latent_classes > 1.
        # Without it the SA never generates valid LC solutions (IndexError in build_spec).
        has_lc = getattr(evaluator, "max_latent_classes", 1) > 1
        dim    = 3 * D + 2 if has_lc else 2 * D + 1

        print(f"\n  Running {algo.upper()} | dim={dim} | max_iter={max_iter} | seed={seed}")

        if algo in ("sa", "hc"):
            defaults = dict(
                max_iter=max_iter,
                mutation_rate=0.3, step_size=1,
                min_changes=1, max_changes=3,
                n_starts=1, alpha=0.995,
            )
            defaults.update(algo_kwargs)

            solver = MultiStartSA(
                evaluator=evaluator,
                dimension=dim,
                **defaults,
            )
            solutions, scores = solver.optimize()
            solutions = np.array(solutions)
            scores    = np.array(scores)

            best_idx      = int(np.argmin(scores))
            best_solution = solutions[best_idx]
            best_score    = float(scores[best_idx])

            # Decode best
            D2  = len(evaluator.vars)
            if has_lc and len(best_solution) > 2 * D2 + 1:
                lc = int(best_solution[2*D2+1]) % evaluator.max_latent_classes + 1
            else:
                lc = 1
            n_mem_7 = sum(
                1 for i, v in enumerate(evaluator.vars)
                if int(best_solution[i]) == 7
            )
            n_mem_8 = sum(
                1 for i, v in enumerate(evaluator.vars)
                if int(best_solution[i]) == 8
            )

            print("\n  Best structure:")
            decode_best_solution(best_solution, evaluator)
            print(f"  Best BIC              : {best_score:.4f}")
            print(f"  Latent classes        : {lc}")
            print(f"  Membership-only vars  : {n_mem_7}  (role 7)")
            print(f"  Membership+fixed vars : {n_mem_8}  (role 8)")

            refit_and_print(evaluator, best_solution)
            save_run_summary_to_txt(evaluator, best_solution,
                                    algo, seed, config_id)

            result = {
                "algorithm":     algo,
                "seed":          seed,
                "solutions":     solutions,
                "scores":        scores,
                "best_solution": best_solution,
                "best_score":    best_score,
            }
            if output_config is not None:
                result["saved_to"] = str(save_search_result(result, output_config, family="count", algorithm=algo))
            return result

        elif algo in ("de", "hs"):
            de_def = dict(population_size=20, F=0.5, CR=0.7)
            hs_def = dict(population_size=20, hmcr=0.9,
                          par_min=0.1, par_max=0.9, bw_min=1, bw_max=3)

            if algo == "de":
                de_def.update(algo_kwargs)
                op  = AdaptiveDE(F=de_def["F"], CR=de_def["CR"])
                pop = de_def["population_size"]
            else:
                hs_def.update(algo_kwargs)
                op  = DynamicHarmony(**{k: v for k, v in hs_def.items()
                                        if k != "population_size"})
                pop = hs_def["population_size"]

            result = run_nsga(evaluator=evaluator, operator=op,
                              seed=seed, pop_size=pop,
                              max_iter=max_iter, n_jobs=n_jobs)
            if output_config is not None:
                result["saved_to"] = str(save_search_result(result, output_config, family="count", algorithm=algo))
            return result

        else:
            raise ValueError(f"Unknown algo '{algo}'. Choose: sa, hc, de, hs")

    def run_search(self, search_problem=None, **kwargs):
        output_config = kwargs.pop("output_config", None)
        search_problem = search_problem or self._evaluator
        if search_problem is None:
            raise RuntimeError("Call build_evaluator() or build_search() first.")

        if isinstance(search_problem, StructureEvaluatorLC):
            return self.run(evaluator=search_problem, output_config=output_config, **kwargs)

        if hasattr(search_problem, "run"):
            result = search_problem.run(**kwargs)
            if output_config is not None:
                family = result.get("family") or getattr(search_problem, "family", "search")
                algorithm = str(kwargs.get("algo", result.get("algorithm", "run")))
                result["saved_to"] = str(save_search_result(result, output_config, family=family, algorithm=algorithm))
            return result

        raise TypeError("Unsupported search problem type.")

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _infer_type(s: pd.Series) -> str:
        n = s.nunique()
        if n == 2:                                         return "binary"
        if n <= 10 and (s % 1 == 0).all():                return "ordinal"
        if (s >= 0).all() and (s % 1 == 0).all():         return "count"
        return "continuous"

    @staticmethod
    def _suggest_roles_dists(col, s, vtype, max_lc=1):
        base_roles = [0, 1, 2]
        mem_roles  = ([7, 8] if max_lc > 1 else [])

        if vtype == "binary":
            return (
                base_roles + mem_roles,
                ["normal"],
                "Binary — fixed usually sufficient. "
                + ("Membership roles available (predicts class membership)."
                   if max_lc > 1 else "")
            )
        if vtype == "ordinal":
            return (
                [0, 1, 2] + mem_roles,
                ["normal"],
                "Ordinal — fixed or random-independent."
                + (" Role 7/8: could help explain class structure."
                   if max_lc > 1 else "")
            )
        if vtype == "count":
            return (
                [0, 1, 2, 3] + mem_roles,
                ["normal", "lognormal"],
                "Count covariate — lognormal if effect is strictly positive."
            )
        return (
            [0, 1, 2, 3, 5] + mem_roles,
            ["normal", "lognormal", "triangular"],
            "Continuous — full menu. Role 8 (mem+fixed) is useful for "
            "variables that explain both class membership and outcome level."
        )


# =====================================================================
# Standalone helper functions (importable from metacountregressor)
# =====================================================================

def extract_summary(fit_result: dict) -> dict:
    """Safely extract the summary dict from a fit_manual_model result.

    Parameters
    ----------
    fit_result : dict
        Output of ``ExperimentBuilder.fit_manual_model()`` or
        ``CMFExperimentBuilder.fit_manual_cmf_model()``.

    Returns
    -------
    dict
        Keys: ``bic``, ``aic``, ``loglik``, ``num_parm``, ``n_obs``.
    """
    s = fit_result.get("summary")
    if s is not None and isinstance(s, dict) and "bic" in s:
        return s
    # Fallback: build a minimal dict from whatever is available
    return {
        "loglik":   s.get("loglik", float("nan")) if isinstance(s, dict) else float("nan"),
        "num_parm": s.get("num_parm", float("nan")) if isinstance(s, dict) else float("nan"),
        "n_obs":    s.get("n_obs", float("nan")) if isinstance(s, dict) else float("nan"),
        "aic":      s.get("aic", float("nan")) if isinstance(s, dict) else float("nan"),
        "bic":      s.get("bic", float("nan")) if isinstance(s, dict) else float("nan"),
    }


def extract_search_best(search_result: dict) -> dict:
    """Normalise search result keys from ``ExperimentBuilder.run()``.

    The ``run()`` method returns ``best_score`` and ``best_solution``.
    This helper returns a dict with canonical names so downstream code
    is resilient to future API changes.

    Parameters
    ----------
    search_result : dict
        Output of ``ExperimentBuilder.run()``.

    Returns
    -------
    dict
        Keys: ``best_bic``, ``best_decision``, ``scores``.
    """
    return {
        "best_bic":      search_result.get("best_score",
                            search_result.get("best_fitness")),
        "best_decision": search_result.get("best_solution",
                            search_result.get("best_decision")),
        "scores":        search_result.get("scores",
                            search_result.get("history")),
    }


def compare_models(fit_results: dict) -> "pd.DataFrame":
    """Build a comparison DataFrame from a dict of fit results.

    Parameters
    ----------
    fit_results : dict[str, dict]
        ``{model_name: fit_result}`` where each ``fit_result`` is the
        output of ``fit_manual_model()`` or ``fit_manual_cmf_model()``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Model``, ``BIC``, ``AIC``, ``Log-Likelihood``,
        ``Parameters``.  Sorted by BIC ascending.
    """
    import pandas as _pd  # local import to keep module import light
    rows = []
    for name, fit in fit_results.items():
        s = extract_summary(fit)
        rows.append({
            "Model":          name,
            "BIC":            s.get("bic", float("nan")),
            "AIC":            s.get("aic", float("nan")),
            "Log-Likelihood": s.get("loglik", float("nan")),
            "Parameters":     s.get("num_parm", float("nan")),
        })
    df = _pd.DataFrame(rows).sort_values("BIC")
    df.index = range(1, len(df) + 1)
    df.index.name = "Rank"
    return df
