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
import main_hpc_lc_patch as _patch   # patches main_hpc in-place

import numpy as np
import pandas as pd
from dataclasses import replace
from functools import partial
from scipy.optimize import minimize
from typing import Optional, Dict, List, Union, Any

from family_search import (
    CMFFamilySearchProblem,
    DurationSearchProblem,
    LinearSearchProblem,
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

__all__ = ["StructureEvaluatorLC", "ExperimentBuilder"]


# ═══════════════════════════════════════════════════════════════════════
# UPDATED ROLE_PROBS  (add slots for roles 7 and 8)
# Applied directly to Solvers_METAJAX.ROLE_PROBS as well.
# ═══════════════════════════════════════════════════════════════════════

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

                # Step 4 — MLE polish
                result_c = minimize(
                    lambda p: float(mixed_model_loglik(p, data_train, spec_c)),
                    params_em,
                    method="L-BFGS-B",
                    options={"maxiter": 500},
                )

                ll  = -result_c.fun
                n   = data_train["y"].shape[0]
                k   = len(result_c.x)
                bic = k * np.log(n) - 2.0 * ll

                class _Model:
                    params = result_c.x
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

        reserved = {id_col, y_col}
        if offset_col:   reserved.add(offset_col)
        if group_id_col: reserved.add(group_id_col)
        self._candidate_vars = [c for c in df.columns if c not in reserved]

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

        variables    = variables or self._candidate_vars
        exclude_set  = set(exclude or [])
        variables    = [v for v in variables if v not in exclude_set]

        fixed_override      = fixed_override      or {}
        membership_override = membership_override or {}

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

        if engine != "jax":
            raise ValueError("Only the JAX-first engine is supported through ExperimentBuilder.")

        variables = variables or self._candidate_vars
        exclude_set = set(exclude or [])
        variables = [v for v in variables if v not in exclude_set]

        if model_family == "count":
            return self.build_evaluator(
                variables=variables,
                exclude=exclude,
                model_family="count",
                engine=engine,
                **kwargs,
            )

        if model_family == "linear":
            return LinearSearchProblem(
                df=self.df,
                y_col=self.y_col,
                variables=variables,
                objective_kwargs=kwargs.pop("objective_kwargs", {}),
            )

        if model_family == "duration":
            budget_col = kwargs.pop("budget_col", "B")
            if budget_col not in self.df.columns:
                raise ValueError(f"Duration search requires budget_col='{budget_col}' in the dataframe.")
            return DurationSearchProblem(
                df=self.df.copy(),
                y_col=self.y_col,
                variables=variables,
                id_col=self.id_col,
                budget_col=budget_col,
            )

        if model_family == "cmf":
            from cmf_package import CMFExperimentBuilder

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

            return {
                "algorithm":     algo,
                "seed":          seed,
                "solutions":     solutions,
                "scores":        scores,
                "best_solution": best_solution,
                "best_score":    best_score,
            }

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

            return run_nsga(evaluator=evaluator, operator=op,
                            seed=seed, pop_size=pop,
                            max_iter=max_iter, n_jobs=n_jobs)

        else:
            raise ValueError(f"Unknown algo '{algo}'. Choose: sa, hc, de, hs")

    def run_search(self, search_problem=None, **kwargs):
        search_problem = search_problem or self._evaluator
        if search_problem is None:
            raise RuntimeError("Call build_evaluator() or build_search() first.")

        if isinstance(search_problem, StructureEvaluatorLC):
            return self.run(evaluator=search_problem, **kwargs)

        if hasattr(search_problem, "run"):
            return search_problem.run(**kwargs)

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
