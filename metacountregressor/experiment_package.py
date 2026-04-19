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
import jax.numpy as jnp
from dataclasses import replace
from functools import partial
import io
from contextlib import redirect_stdout
from jaxopt import LBFGS
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
        build_datasets,
        generate_master_halton,
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
    )
    from .main_hpc_lc_patch import (
        ModelSpec,
        build_param_index,
        build_model_from_manual_spec,
        mixed_model_loglik,
        print_summary,
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
        build_datasets,
        generate_master_halton,
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
    )
    from main_hpc_lc_patch import (
        ModelSpec,
        build_param_index,
        build_model_from_manual_spec,
        mixed_model_loglik,
        print_summary,
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

    # ── build_spec ──────────────────────────────────────────────────

    def build_spec(self, decision) -> Optional[dict]:
        """
        Decode a decision vector into a manual_spec dict.

        Decision layout (dimension = 2·D + 2):
          decision[:D]   = role codes  (0–8)
          decision[D:2D] = dist codes
          decision[2D]   = dispersion bit
          decision[2D+1] = latent_class_code  (0 → 1 class, 1 → 2, …)
        """
        D              = len(self.vars)
        roles          = decision[:D]
        dists          = decision[D : 2*D]
        dispersion_bit = int(decision[2*D])
        lc_code        = int(decision[2*D + 1])

        use_nb          = dispersion_bit % 2 == 1
        latent_classes  = lc_code % self.max_latent_classes + 1

        fixed      = []
        rdm_ind    = []
        rdm_cor    = []
        grouped    = []
        hetero     = []
        zi         = []
        membership = []    # NEW: variables in class-prob equation

        for i, var in enumerate(self.vars):
            role = int(roles[i])
            if role not in self.allowed_roles.get(var, [0]):
                return None

            if role == 0:
                pass  # excluded

            elif role == 1:
                fixed.append(var)

            elif role == 2:
                dist = decode_distribution(
                    dists[i], self.allowed_distributions.get(var, ["normal"])
                )
                rdm_ind.append(f"{var}:{dist}")

            elif role == 3:
                dist = decode_distribution(
                    dists[i], self.allowed_distributions.get(var, ["normal"])
                )
                rdm_cor.append(f"{var}:{dist}")

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
                # Membership-only — only matters when LC > 1
                if latent_classes > 1:
                    membership.append(var)
                # else: treated as excluded

            elif role == 8:
                # Membership + fixed outcome
                if latent_classes > 1:
                    membership.append(var)  # class-prob equation
                fixed.append(var)           # outcome equation (class-specific)

        # Single correlated var → demote to independent
        if len(rdm_cor) == 1:
            rdm_ind.extend(rdm_cor)
            rdm_cor = []

        return {
            "fixed_terms":      fixed,
            "rdm_terms":        rdm_ind,
            "rdm_cor_terms":    rdm_cor,
            "grouped_terms":    grouped,
            "hetro_in_means":   hetero,
            "zi_terms":         zi,
            "membership_terms": membership,   # NEW
            "dispersion":       1 if use_nb else 0,
            "latent_classes":   latent_classes,
        }

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
            tuple(sorted(spec_dict.get("membership_terms", []))),   # NEW
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
            mh_g    = generate_master_halton(G, len(self.vars), self.R, seed=999)
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
                model.fit()
                bic = model.bic()

            # ── Multi-class path with warm start ───────────────────
            else:
                K_mem  = spec.K_membership
                spec_1 = replace(spec, latent_classes=1)

                # Step 1 — fit single-class
                model_1   = CountModel(spec_1, data_train)
                result_1  = model_1.fit()
                theta_1   = np.array(result_1.params)
                K_base    = build_param_index(spec_1)["total_params"]

                # Step 2 — perturb to seed each class
                rng = np.random.default_rng(abs(hash(sig)) % (2**31))
                theta_init = np.concatenate([
                    theta_1 + rng.normal(0, 0.05, K_base) for _ in range(C)
                ])

                # gamma init: zeros → equal class probs, membership coeffs=0
                gamma_init = np.zeros((C - 1) * (K_mem + 1))
                init_params = np.concatenate([theta_init, gamma_init])

                spec_c = replace(spec, latent_classes=C)

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

                # Step 4 — MLE polish (JAX-native optimizer)
                polish = LBFGS(
                    fun=lambda p: mixed_model_loglik(p, data_train, spec_c),
                    maxiter=500,
                )
                result_c = polish.run(jnp.array(params_em))
                params_c = np.array(result_c.params)
                ll  = -float(result_c.state.value)
                n   = data_train["y"].shape[0]
                k   = len(params_c)
                bic = k * np.log(n) - 2.0 * ll

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
            return np.array([1e12, 1e12]) if self.mode == "multi" else 1e12


class ForcedModelStructureEvaluatorLC(StructureEvaluatorLC):
    def __init__(self, *args, forced_model: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_model = forced_model

    def build_spec(self, decision) -> Optional[dict]:
        spec = super().build_spec(decision)
        if spec is None:
            return None
        spec["model"] = self.forced_model
        if self.forced_model in {"lognormal", "gaussian"}:
            spec["dispersion"] = 0
        return spec


# ═══════════════════════════════════════════════════════════════════════
# SOLVER PATCH  — generate_neighbor supports tail genes (dispersion + LC)
# and the new roles 7 / 8.  Replace AdvancedSimulatedAnnealing.generate_neighbor.
# ═══════════════════════════════════════════════════════════════════════

def _generate_neighbor_patched(self, solution, T=None, max_attempts=20, min_active=2):
    """
    Extended generate_neighbor that correctly mutates ALL gene slots:

    Tail gene index  2D   → dispersion bit (flip 0↔1)
    Tail gene index  2D+1 → latent-class code (step ±1)
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
        indices   = np.random.choice(self.dim, size=n_changes, replace=False)

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
                    # Tail genes
                    tail_pos = idx - 2 * D

                    if tail_pos == 0:
                        # Dispersion bit: flip
                        neighbor[idx] = 1 - int(neighbor[idx])
                    else:
                        # Latent-class code: step ±1 within bounds
                        max_code      = self.evaluator.max_latent_classes - 1
                        step          = np.random.choice([-1, 1])
                        neighbor[idx] = int(np.clip(neighbor[idx] + step,
                                                    0, max_code))
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

    if self.is_same(solution, neighbor):
        return self._generate_neighbor_patched(solution, T,
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
    ) -> Dict[str, Any]:
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
        fitted = CountModel(spec, data)
        result = fitted.fit()
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
        K_base = build_param_index(base_spec)["total_params"]
        gamma_size = (C - 1) * (K_mem + 1)
        gamma = params[C * K_base : C * K_base + gamma_size].reshape(C - 1, K_mem + 1)

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
        data = fit_result["data"]

        params = np.asarray(result.params)

        # Build coefficient table with fixed, random, and dispersion parameters
        coef_rows = []

        # ── Fixed coefficients ─────────────────────────────────────
        if spec.Kf > 0:
            fixed_names = list(spec.fixed_names)
            fixed_start, fixed_end = param_index["fixed"]
            for name, value in zip(fixed_names, params[fixed_start:fixed_end]):
                coef_rows.append({"Parameter": name, "Type": "Fixed", "Estimate": value})

        # ── Random independent (means) ─────────────────────────────
        if spec.Kr_ind > 0:
            ind_names = list(spec.random_ind_names)
            if "ind_mean" in param_index:
                mean_start, mean_end = param_index["ind_mean"]
                for name, value in zip(ind_names, params[mean_start:mean_end]):
                    coef_rows.append({
                        "Parameter": f"{name} (ind. mean)",
                        "Type": "Random-Ind",
                        "Estimate": value
                    })

            # Standard deviations
            if "ind_sd" in param_index:
                sd_start, sd_end = param_index["ind_sd"]
                for name, value in zip(ind_names, params[sd_start:sd_end]):
                    coef_rows.append({
                        "Parameter": f"{name} (ind. SD)",
                        "Type": "Random-Ind",
                        "Estimate": value
                    })

        # ── Random correlated (means) ──────────────────────────────
        if spec.Kr_cor > 0:
            cor_names = list(spec.random_cor_names)
            if "cor_mean" in param_index:
                mean_start, mean_end = param_index["cor_mean"]
                for name, value in zip(cor_names, params[mean_start:mean_end]):
                    coef_rows.append({
                        "Parameter": f"{name} (cor. mean)",
                        "Type": "Random-Cor",
                        "Estimate": value
                    })

        # ── Grouped effects ────────────────────────────────────────
        if spec.Kg > 0:
            grouped_names = list(spec.grouped_names)
            if "group_mean" in param_index:
                mean_start, mean_end = param_index["group_mean"]
                for name, value in zip(grouped_names, params[mean_start:mean_end]):
                    coef_rows.append({
                        "Parameter": f"{name} (group mean)",
                        "Type": "Grouped",
                        "Estimate": value
                    })

            if "group_sd" in param_index:
                sd_start, sd_end = param_index["group_sd"]
                for name, value in zip(grouped_names, params[sd_start:sd_end]):
                    coef_rows.append({
                        "Parameter": f"{name} (group SD)",
                        "Type": "Grouped",
                        "Estimate": value
                    })

        # ── Dispersion parameter (for negative binomial) ───────────
        if spec.model == "nb":
            if "dispersion" in param_index:
                disp_idx = param_index["dispersion"]
                coef_rows.append({
                    "Parameter": "Dispersion",
                    "Type": "Dispersion",
                    "Estimate": params[disp_idx]
                })

        # Build DataFrame and print
        coef_df = pd.DataFrame(coef_rows)

        print("\n" + "=" * 80)
        print(f"  MODEL COEFFICIENTS  —  {spec.model.upper()} MODEL")
        print("=" * 80 + "\n")

        if len(coef_df) > 0:
            # Group by type for better readability
            for type_name in ["Fixed", "Random-Ind", "Random-Cor", "Grouped", "Dispersion"]:
                subset = coef_df[coef_df["Type"] == type_name]
                if len(subset) > 0:
                    print(f"  {type_name.upper()} PARAMETERS:")
                    print(f"  {'-' * 76}")
                    for _, row in subset.iterrows():
                        print(f"    {row['Parameter']:30s} = {row['Estimate']:+.6f}")
                    print()

        print("=" * 80 + "\n")

        return coef_df[["Parameter", "Type", "Estimate"]]

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
        dim = 2 * D + 2
        print(f"\n  Evaluator ready:")
        print(f"    Variables          : {D}")
        print(f"    Decision dimension : {dim}  (2×{D} + 2)")
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
        dim = 2 * D + 2

        print(f"\n  Running {algo.upper()} | dim={dim} | max_iter={max_iter} | seed={seed}")

        if algo in ("sa", "hc"):
            defaults = dict(
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
            lc  = int(best_solution[2*D2+1]) % evaluator.max_latent_classes + 1
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
