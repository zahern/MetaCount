from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import warnings
import numpy as np
import pandas as pd

try:
    from .cmf_package import CMFExperimentBuilder
    from .duration_main import (
        estimate_model,
        ll_independent,
        ll_with_budget_penalty,
        prepare_data,
        predict_daily_schedule,
    )
except ImportError:
    from cmf_package import CMFExperimentBuilder
    from duration_main import (
        estimate_model,
        ll_independent,
        ll_with_budget_penalty,
        prepare_data,
        predict_daily_schedule,
    )


def _run_metaheuristic(algo: str, objective_function, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        try:
            from .metaheuristics import (
                differential_evolution,
                harmony_search,
                simulated_annealing,
            )
        except ImportError:
            from metaheuristics import (
                differential_evolution,
                harmony_search,
                simulated_annealing,
            )
    algo = algo.lower()
    if algo == "hs":
        return harmony_search(objective_function, **kwargs)
    if algo == "de":
        return differential_evolution(objective_function, **kwargs)
    if algo in {"sa", "hc"}:
        return simulated_annealing(objective_function, **kwargs)
    raise ValueError(f"Unknown algorithm '{algo}'. Choose from hs, de, sa.")


class CMFMetaheuristicObjective:
    def __init__(
        self,
        df: pd.DataFrame,
        baseline_vars: list[str],
        local_vars: list[str],
        R: int = 200,
        instance_name: str = "cmf_search",
        max_time: float = 3600.0,
        max_imp: int = 500,
        hms: int = 20,
        hmcr: float = 0.9,
        par: float = 0.3,
        mpai: int = 1,
        termination_iter: int = 200,
    ):
        self.df = df
        self.baseline_vars = list(baseline_vars)
        self.local_vars = list(local_vars)
        self.R = R
        self.instance_name = instance_name
        self.is_multi = False
        self.algorithm = "sa"
        self._obj_1 = "bic"
        self._obj_2 = "bic"
        self._hms = hms
        self._hmcr = hmcr
        self._par = par
        self._mpai = mpai
        self._mpap = 0.1
        self._max_imp = max_imp
        self._max_time = max_time
        self._max_iterations_improvement = termination_iter
        self._max_characteristics = len(self.baseline_vars) + len(self.local_vars)
        self._discrete_values = [[0, 1]] * self.get_num_parameters()
        self._cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def get_num_parameters(self) -> int:
        return 2 * (len(self.baseline_vars) + len(self.local_vars)) + 2

    def get_num_discrete_values(self, i):
        return len(self._discrete_values[i])

    def get_value(self, i, j=None):
        values = self._discrete_values[i]
        if j is None:
            return int(np.random.choice(values))
        return values[j % len(values)]

    def get_index(self, i, v):
        return int(v)

    def get_indexes_of_ints(self):
        return list(range(self.get_num_parameters()))

    def get_param_num(self, dispersion=0):
        k_base = len(self.baseline_vars)
        k_loc = len(self.local_vars)
        return int(
            np.sum(self._last_vector[: k_base + k_loc])
            if hasattr(self, "_last_vector")
            else 0
        )

    def get_max_imp(self):
        return self._max_imp

    def get_max_time(self):
        return self._max_time

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def get_termination_iter(self):
        return self._max_iterations_improvement

    def _get_obj1(self):
        return self._obj_1

    def _get_obj2(self):
        return self._obj_2

    def decode_solution(self, vector):
        k_base = len(self.baseline_vars)
        k_loc = len(self.local_vars)
        offset_rf = k_base + k_loc + 2

        baseline_mask = np.asarray(vector[:k_base], dtype=bool)
        local_mask = np.asarray(vector[k_base:k_base + k_loc], dtype=bool)
        use_halton = bool(vector[k_base + k_loc])
        model = "poisson" if int(vector[k_base + k_loc + 1]) == 0 else "nb"
        rand_baseline_all = [bool(vector[offset_rf + i]) for i in range(k_base)]
        rand_local_all = [bool(vector[offset_rf + k_base + i]) for i in range(k_loc)]

        selected_baseline = [v for v, m in zip(self.baseline_vars, baseline_mask) if m]
        selected_local = [v for v, m in zip(self.local_vars, local_mask) if m]
        rand_baseline = tuple(r for r, m in zip(rand_baseline_all, baseline_mask) if m)
        rand_local = tuple(r for r, m in zip(rand_local_all, local_mask) if m)

        return {
            "selected_baseline": selected_baseline,
            "selected_local": selected_local,
            "rand_baseline": rand_baseline,
            "rand_local": rand_local,
            "rand_baseline_all": rand_baseline_all,
            "rand_local_all": rand_local_all,
            "use_halton": use_halton,
            "model": model,
        }

    def get_fitness(self, vector, multi=False, verbose=False, max_routine=3):
        from GA_CMF_AADT_JAX import evaluate_model
        vec = np.asarray(vector, dtype=int)
        key = tuple(int(v) for v in vec.tolist())
        self._last_vector = vec
        if key in self._cache:
            return self._cache[key]

        decoded = self.decode_solution(vec)
        score = evaluate_model(
            vec,
            self.df.rename(columns={"Y": "FREQ"}) if "FREQ" not in self.df.columns and "Y" in self.df.columns else self.df,
            self.baseline_vars,
            self.local_vars,
            use_halton=decoded["use_halton"],
            model=decoded["model"],
            rand_baseline_all=decoded["rand_baseline_all"],
            rand_local_all=decoded["rand_local_all"],
            R=self.R,
        )

        result = {
            "bic": float(score),
            "layout": vec.tolist(),
            "fixed_fit": None,
            "rdm_fit": None,
            "rdm_cor_fit": None,
            "zi_fit": None,
            "family": "cmf",
            **decoded,
        }
        self._cache[key] = result
        return result


@dataclass
class LinearSearchProblem:
    df: Optional[pd.DataFrame] = None
    y_col: Optional[str] = None
    variables: Optional[list[str]] = None
    objective_kwargs: Optional[dict[str, Any]] = None
    builder: Any = None
    evaluator: Any = None
    metadata: Optional[dict[str, Any]] = None

    family: str = "linear"

    def run(
        self,
        algo: str = "hs",
        initial_solutions=None,
        **algorithm_kwargs,
    ):
        if self.builder is not None and self.evaluator is not None:
            result = self.builder.run_search(self.evaluator, algo=algo, **algorithm_kwargs)
            result["family"] = "linear"
            result["driver"] = "jax_hierarchical"
            result["linear_metadata"] = self.metadata or {}
            return result

        X = self.df[self.variables].copy()
        y = self.df[[self.y_col]].copy()

        from solution import ObjectiveFunction
        objective = ObjectiveFunction(
            X,
            y,
            linear_model=True,
            **(self.objective_kwargs or {}),
        )
        return _run_metaheuristic(
            algo,
            objective,
            initial_slns=initial_solutions,
            **algorithm_kwargs,
        )


@dataclass
class DurationSearchProblem:
    df: Optional[pd.DataFrame] = None
    y_col: Optional[str] = None
    variables: Optional[list[str]] = None
    id_col: Optional[str] = None
    budget_col: Optional[str] = None
    builder: Any = None
    evaluator: Any = None
    metadata: Optional[dict[str, Any]] = None

    family: str = "duration"

    def run(
        self,
        objective: str = "budget_penalty",
        init_params: Optional[np.ndarray] = None,
        lambda_penalty: float = 10.0,
        algo: str = "sa",
        **algorithm_kwargs,
    ) -> dict[str, Any]:
        if self.builder is not None and self.evaluator is not None:
            result = self.builder.run_search(self.evaluator, algo=algo, **algorithm_kwargs)
            result["family"] = "duration"
            result["driver"] = "jax_hierarchical"
            result["duration_metadata"] = self.metadata or {}
            return result

        X, y, ids, budgets = prepare_data(
            self.df,
            feature_cols=self.variables,
            y_col=self.y_col,
            id_col=self.id_col,
            budget_col=self.budget_col,
        )

        init = np.zeros(len(self.variables) + 1) if init_params is None else np.asarray(init_params)

        if objective == "independent":
            objective_fn = lambda p: ll_independent(p, X, y)
        elif objective == "budget_penalty":
            objective_fn = partial(
                ll_with_budget_penalty,
                X=X,
                y=y,
                ids=ids,
                budgets=budgets,
                lambda_penalty=lambda_penalty,
            )
        else:
            raise ValueError("objective must be 'independent' or 'budget_penalty'")

        result = estimate_model(objective_fn, init)
        prediction_df = self.df.copy()
        prediction_df["predicted_duration"] = predict_daily_schedule(
            result.x,
            prediction_df,
            feature_cols=self.variables,
            id_col=self.id_col,
            budget_col=self.budget_col,
        )

        return {
            "result": result,
            "predictions": prediction_df,
            "objective": objective,
            "lambda_penalty": lambda_penalty,
        }


@dataclass
class CMFFamilySearchProblem:
    builder: CMFExperimentBuilder
    id_col: Optional[str] = None
    offset_col: Optional[str] = None
    group_id_col: Optional[str] = None

    family: str = "cmf"

    def run(
        self,
        algo: str = "sa",
        final_R: Optional[int] = None,
        fit_final: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        algo = algo.lower()
        R = kwargs.pop("R", 200)

        if algo in {"ga", "cmf"}:
            search_result = self.builder.run_search(R=R)
            output = {"search_result": search_result, "driver": "ga"}

            if fit_final:
                fit_result = self.builder.fit_best_model(
                    search_result,
                    final_R=final_R or kwargs.pop("final_R", 500),
                )
                output["fit_result"] = fit_result

            return output

        if algo in {"sa", "de", "hs", "hc"}:
            # Prefer the hierarchical JAX CMF path (single-class by default)
            # whenever id_col is available. The legacy CMF objective remains
            # available via legacy_cmf_objective=True for backward compatibility.
            legacy_cmf_objective = bool(kwargs.pop("legacy_cmf_objective", False))
            if self.id_col is not None and not legacy_cmf_objective:
                max_iter = kwargs.pop("max_iter", 3000)
                seed = kwargs.pop("seed", 0)

                general_builder, evaluator, metadata = self.builder.build_jax_count_evaluator(
                    id_col=self.id_col,
                    offset_col=self.offset_col,
                    group_id_col=self.group_id_col,
                    variables=kwargs.pop("variables", None),
                    fixed_override=kwargs.pop("fixed_override", None),
                    membership_override=kwargs.pop("membership_override", None),
                    exclude=kwargs.pop("exclude", None),
                    mode=kwargs.pop("mode", "single"),
                    max_latent_classes=int(kwargs.pop("max_latent_classes", 1)),
                    R=R,
                    default_roles=kwargs.pop("default_roles", None),
                    force_aadt_term=bool(kwargs.pop("force_aadt_term", True)),
                    constraints=kwargs.pop("constraints", None),
                )

                result = general_builder.run(
                    evaluator=evaluator,
                    algo=algo,
                    max_iter=max_iter,
                    seed=seed,
                    **kwargs,
                )
                result["family"] = "cmf"
                result["driver"] = "jax_hierarchical"
                result["cmf_metadata"] = metadata
                return result

            objective = CMFMetaheuristicObjective(
                df=self.builder.df.rename(columns={self.builder.y_col: "FREQ"})
                if self.builder.y_col != "FREQ"
                else self.builder.df,
                baseline_vars=self.builder.baseline_vars,
                local_vars=self.builder.local_vars,
                R=R,
                max_time=float(kwargs.pop("_max_time", 3600.0)),
                max_imp=int(kwargs.pop("_max_imp", 500)),
                hms=int(kwargs.get("_hms", 20) or 20),
                hmcr=float(kwargs.get("_hmcr", 0.9) or 0.9),
                par=float(kwargs.get("_par", 0.3) or 0.3),
                mpai=int(kwargs.get("_mpai", 1) or 1),
                termination_iter=int(kwargs.pop("WIC", 200)),
            )
            raw = _run_metaheuristic(algo, objective, **kwargs)

            best_layout = None
            if hasattr(raw, "best_harmony"):
                best_layout = raw.best_harmony
            elif hasattr(raw, "best_solutions") and raw.best_solutions:
                best_layout = raw.best_solutions[-1]

            decoded = objective.decode_solution(best_layout) if best_layout is not None else None
            output = {
                "driver": "metaheuristic",
                "algorithm": algo,
                "raw_result": raw,
                "best_solution": best_layout,
                "decoded_best": decoded,
            }

            if fit_final and decoded is not None:
                fit_result = self.builder.fit_best_model(
                    type(
                        "CMFDecodedResult",
                        (),
                        {
                            "selected_baseline": decoded["selected_baseline"],
                            "selected_local": decoded["selected_local"],
                            "rand_baseline": decoded["rand_baseline"],
                            "rand_local": decoded["rand_local"],
                            "use_halton": decoded["use_halton"],
                            "model": decoded["model"],
                        },
                    )(),
                    final_R=final_R or kwargs.pop("final_R", 500),
                )
                output["fit_result"] = fit_result

            return output

        if self.id_col is None:
            raise ValueError("id_col is required when routing CMF variables into the general latent-class search.")

        max_iter = kwargs.pop("max_iter", 3000)
        seed = kwargs.pop("seed", 0)
        general_builder, evaluator = self.builder.build_latent_class_evaluator(
            id_col=self.id_col,
            offset_col=self.offset_col,
            group_id_col=self.group_id_col,
            **kwargs,
        )
        return general_builder.run(
            evaluator=evaluator,
            algo=algo,
            max_iter=max_iter,
            seed=seed,
            **kwargs,
        )


@dataclass
class UnifiedCMFSearchProblem:
    builder: Any
    evaluator: Any
    metadata: dict[str, Any]

    family: str = "cmf"

    def run(self, **kwargs) -> dict[str, Any]:
        result = self.builder.run_search(self.evaluator, **kwargs)
        result["family"] = "cmf"
        result["driver"] = "jax_count"
        result["cmf_metadata"] = self.metadata
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Multivariate count model – ABM activity-generation variable-selection search
# ═══════════════════════════════════════════════════════════════════════════════

class MultivariateCountObjective:
    """
    Binary variable-selection objective for a jointly-fitted multivariate
    NB/Poisson copula count model (ABM activity-generation stage).

    Decision vector layout  (all binary 0/1):
        indices  0 … D-1           : covariates included for activity_cols[0]
        indices  D … 2D-1          : covariates included for activity_cols[1]
        …
        indices  (M-1)*D … M*D-1   : covariates included for activity_cols[M-1]
        index    M*D               : copula flag   (0=gaussian, 1=vine-frank)
        index    M*D+1             : marginal flag (0=nb,       1=poisson)

    Total dimension = M * D + 2.

    The ``search_copula`` and ``search_marginal`` flags control whether those
    two trailing bits are actually searched or held fixed at ``fixed_copula``
    and ``fixed_marginal`` respectively.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        activity_cols: list[str],
        covariate_cols: list[str],
        offset_col: Optional[str] = None,
        maxiter: int = 500,
        verbose: bool = False,
        search_copula: bool = False,
        search_marginal: bool = False,
        fixed_copula: str = "gaussian",
        fixed_marginal: str = "nb",
        min_vars_per_activity: int = 1,
        add_intercept: bool = True,
        # HS / SA hyper-params forwarded from MultivariateSearchProblem.run()
        max_time: float = 3600.0,
        max_imp: int = 500,
        hms: int = 20,
        hmcr: float = 0.9,
        par: float = 0.3,
        mpai: int = 1,
        termination_iter: int = 200,
    ):
        self.df = df
        self.activity_cols = list(activity_cols)
        self.covariate_cols = list(covariate_cols)
        self.offset_col = offset_col
        self.maxiter = maxiter
        self.verbose = verbose
        self.search_copula = search_copula
        self.search_marginal = search_marginal
        self.fixed_copula = fixed_copula
        self.fixed_marginal = fixed_marginal
        self.min_vars_per_activity = max(1, int(min_vars_per_activity))
        self.add_intercept = add_intercept

        self._max_time = max_time
        self._max_imp = max_imp
        self._hms = hms
        self._hmcr = hmcr
        self._par = par
        self._mpai = mpai
        self._mpap = 0.1
        self._max_iterations_improvement = termination_iter

        self.M = len(activity_cols)
        self.D = len(covariate_cols)
        # Always reserve 2 flag bits (copula + marginal) for a stable vector layout
        self._dim = self.M * self.D + 2
        self._discrete_values = [[0, 1]] * self._dim
        self._cache: dict[tuple[int, ...], dict[str, Any]] = {}

        # Legacy interface slots expected by some metaheuristic drivers
        self.is_multi = False
        self.algorithm = "sa"
        self._obj_1 = "bic"
        self._obj_2 = "bic"
        # SA driver reads this after every get_fitness() call
        self.Last_Sol: Optional[dict] = None

        # Additional attributes/methods required by the SA/HS/DE drivers
        self.instance_name = "multivariate_search"
        self.complexity_level = "multivariate"
        self.solution_analyst = None          # None  → driver uses best initial sln
        self._characteristics = self._dim     # full vector length (for neighbour slicing)
        self._max_characteristics = self._dim # no hard upper bound on active bits
        self._min_characteristics = self.M   # at least 1 covariate per activity

    # ── Dimension / alphabet ─────────────────────────────────────────────────

    def get_num_parameters(self) -> int:
        return self._dim

    def get_num_discrete_values(self, i: int) -> int:
        return 2

    def get_value(self, i: int, j=None):
        if j is None:
            return int(np.random.randint(0, 2))
        return int(j % 2)

    def get_index(self, i: int, v) -> int:
        return int(v)

    def get_indexes_of_ints(self) -> list[int]:
        return list(range(self._dim))

    # ── HS / SA accessor shims ────────────────────────────────────────────────

    def get_max_imp(self):        return self._max_imp
    def get_max_time(self):       return self._max_time
    def get_hmcr(self):           return self._hmcr
    def get_par(self):            return self._par
    def get_hms(self):            return self._hms
    def get_mpai(self):           return self._mpai
    def get_mpap(self):           return self._mpap
    def get_termination_iter(self): return self._max_iterations_improvement
    def _get_obj1(self):          return self._obj_1
    def _get_obj2(self):          return self._obj_2

    # ── SA / HS compatibility stubs ───────────────────────────────────────────
    # These mirror the contract of ObjectiveFunction in solution.py so the
    # generic metaheuristic drivers can call them without branching.

    def use_random_seed(self) -> bool:
        """Binary-variable search never requires a fixed seed."""
        return False

    def set_random_seed(self) -> None:          # pragma: no cover
        pass

    def maximize(self) -> bool:
        """We minimise BIC."""
        return False

    def nbr_routine(self, vector) -> None:
        """Called by SA driver after accepting a neighbour; no-op here."""
        pass

    def modulo_or_divisor(self, dividend: int, divisor: int) -> int:
        """Wrap-around helper used by DE/HS mutation."""
        result = dividend % divisor
        return divisor if result == 0 else result

    def get_param_num(self, dispersion: int = 0) -> int:
        """Return the number of *active* (=1) bits in the last evaluated solution."""
        if self.Last_Sol is not None:
            layout = self.Last_Sol.get("layout", [])
            return int(np.sum(layout[:self.M * self.D]))
        return self.M  # fallback: one per activity

    def reconstruct_vector(self, data_dict):
        """
        Called by SA ``_initialize`` when ``mod_init`` is supplied.
        For a pure binary search the vector IS the encoding; return it as-is.
        """
        if isinstance(data_dict, (list, np.ndarray)):
            return list(data_dict)
        if isinstance(data_dict, dict) and "layout" in data_dict:
            return list(data_dict["layout"])
        return list(data_dict) if data_dict is not None else [0] * self._dim

    def modify_initial_fit(self, data):
        """
        Called by SA ``_initialize`` when a warm-start solution is injected.
        The multivariate objective doesn't require special post-processing.
        """
        return data

    # ── Solution codec ────────────────────────────────────────────────────────

    def decode_solution(self, vector) -> dict[str, Any]:
        """Decode a binary decision vector into model specification."""
        vec = np.asarray(vector, dtype=int).ravel()
        M, D = self.M, self.D

        selected: list[list[str]] = []
        for m in range(M):
            mask = vec[m * D: (m + 1) * D].astype(bool).copy()
            # Enforce minimum inclusion
            if int(mask.sum()) < self.min_vars_per_activity:
                for k in range(min(self.min_vars_per_activity, D)):
                    mask[k] = True
            selected.append(
                [v for v, inc in zip(self.covariate_cols, mask) if inc]
            )

        copula_flag   = int(vec[M * D])
        marginal_flag = int(vec[M * D + 1])

        copula   = ("vine-frank" if (self.search_copula   and copula_flag   == 1)
                    else self.fixed_copula)
        marginal = ("poisson"   if (self.search_marginal  and marginal_flag == 1)
                    else self.fixed_marginal)

        return {
            "selected_per_activity": selected,
            "copula":       copula,
            "marginal":     marginal,
            "copula_flag":  copula_flag,
            "marginal_flag": marginal_flag,
        }

    # ── Fitness ───────────────────────────────────────────────────────────────

    def get_fitness(self, vector, multi=False, **_kwargs) -> dict[str, Any]:
        """Fit the multivariate model for the given binary decision vector."""
        vec = np.asarray(vector, dtype=int).ravel()
        key = tuple(int(v) for v in vec.tolist())

        if key in self._cache:
            # SA driver reads Last_Sol immediately after get_fitness() returns;
            # keep it updated (must be the full fitness dict) so pareto_run can
            # index it with string keys like "bic".
            self.Last_Sol = self._cache[key]
            return self._cache[key]

        decoded = self.decode_solution(vec)
        covariate_dict = {
            act: cols
            for act, cols in zip(self.activity_cols, decoded["selected_per_activity"])
        }

        bic = 1e20
        aic = 1e20
        try:
            try:
                from .multivariate_count_regressor import fit_multivariate_activity_model
            except ImportError:
                from multivariate_count_regressor import fit_multivariate_activity_model

            fit = fit_multivariate_activity_model(
                df=self.df,
                activity_cols=self.activity_cols,
                covariate_cols=covariate_dict,
                offset_col=self.offset_col,
                copula=decoded["copula"],
                marginal=decoded["marginal"],
                add_intercept=self.add_intercept,
                maxiter=self.maxiter,
                verbose=self.verbose,
            )
            bic = float(fit.bic)
            aic = float(fit.aic)
        except Exception as exc:
            if self.verbose:
                warnings.warn(
                    f"MultivariateCountObjective: fit failed for key={key}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        import json as _json
        result: dict[str, Any] = {
            "bic":           bic,
            "aic":           aic,
            "layout":        vec.tolist(),
            "family":        "multivariate",
            # Stub keys expected by the shared logger() helper in metaheuristics.py
            "fixed_fit":     None,
            "rdm_fit":       None,
            "rdm_cor_fit":   None,
            "zi_fit":        None,
            "pvalues":       None,
            # Scalar / string fields safe for a single-row pandas DataFrame
            "copula":        decoded["copula"],
            "marginal":      decoded["marginal"],
            "copula_flag":   decoded["copula_flag"],
            "marginal_flag": decoded["marginal_flag"],
            # Store as JSON string so pandas logger never sees a list-of-lists
            "selected_per_activity": _json.dumps(decoded["selected_per_activity"]),
        }
        self._cache[key] = result
        # SA driver reads Last_Sol immediately after get_fitness() returns;
        # must be the full fitness dict so pareto_run can index by string key.
        self.Last_Sol = result
        return result


@dataclass
class MultivariateSearchProblem:
    """
    Variable-selection search problem for jointly-fitted multivariate
    NB/Poisson copula count models (ABM activity-generation stage).

    Construct via::

        problem = ExperimentBuilder(...).build_search(
            model_family='multivariate',
            activity_cols=['n_work', 'n_shop', 'n_rec', 'n_eat'],
            variables=['age', 'income', 'cars', 'hhsize', ...],
        )
        result = problem.run(algo='sa')

    Or instantiate directly for full control::

        problem = MultivariateSearchProblem(
            df=df,
            activity_cols=['n_work', 'n_shop', 'n_rec'],
            covariate_cols=['age', 'income', 'cars'],
        )
        result = problem.run(algo='hs', max_imp=300, hms=15)
    """

    df: pd.DataFrame
    activity_cols: list[str]
    covariate_cols: list[str]
    offset_col: Optional[str] = None
    maxiter: int = 500
    verbose: bool = False
    search_copula: bool = False
    search_marginal: bool = False
    fixed_copula: str = "gaussian"
    fixed_marginal: str = "nb"
    min_vars_per_activity: int = 1
    add_intercept: bool = True
    metadata: Optional[dict[str, Any]] = None

    family: str = "multivariate"

    @property
    def dim(self) -> int:
        """Total dimension of the binary decision vector (M*D + 2)."""
        return len(self.activity_cols) * len(self.covariate_cols) + 2

    def _build_objective_dim(self, D: int = None, M: int = None) -> int:
        """Return the decision-vector dimension.

        Convenience method used by scripts / notebooks.  If *D* and *M* are
        supplied they override the dataclass attributes so the method can be
        called with explicit values without needing a fully-populated instance.
        """
        _D = D if D is not None else len(self.covariate_cols)
        _M = M if M is not None else len(self.activity_cols)
        return _M * _D + 2

    def run(
        self,
        algo: str = "sa",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run the variable-selection metaheuristic search.

        Parameters
        ----------
        algo : str
            'sa' (simulated annealing, default), 'hs' (harmony search),
            'de' (differential evolution), or 'hc' (hill climb).
        max_time : float
            Wall-clock budget in seconds (default 3600).
        max_imp : int
            Maximum number of improving iterations before stopping (default 500).
        hms : int
            Harmony-memory size for HS (default 20).
        hmcr : float
            Harmony-memory consideration rate for HS (default 0.9).
        par : float
            Pitch-adjustment rate for HS (default 0.3).
        termination_iter : int
            Iterations without improvement before early termination (default 200).

        Returns
        -------
        dict
            Keys: ``best_solution``, ``best_decoded``, ``best_bic``,
            ``driver``, ``family``, ``algorithm``, ``raw_result``,
            ``multivariate_metadata``.
        """
        objective = MultivariateCountObjective(
            df=self.df,
            activity_cols=self.activity_cols,
            covariate_cols=self.covariate_cols,
            offset_col=self.offset_col,
            maxiter=self.maxiter,
            verbose=self.verbose,
            search_copula=self.search_copula,
            search_marginal=self.search_marginal,
            fixed_copula=self.fixed_copula,
            fixed_marginal=self.fixed_marginal,
            min_vars_per_activity=self.min_vars_per_activity,
            add_intercept=self.add_intercept,
            max_time=float(kwargs.pop("max_time", 3600.0)),
            max_imp=int(kwargs.pop("max_imp", 500)),
            hms=int(kwargs.pop("hms", 20)),
            hmcr=float(kwargs.pop("hmcr", 0.9)),
            par=float(kwargs.pop("par", 0.3)),
            mpai=int(kwargs.pop("mpai", 1)),
            termination_iter=int(kwargs.pop("termination_iter", 200)),
        )

        raw = _run_metaheuristic(algo, objective, **kwargs)

        # ── Recover best solution from the objective cache (most reliable) ──
        # The SA driver stores the best layout on `sa.best_struct` (an instance
        # attribute), then returns a plain dict  {elapsed_time, Iteration}
        # rather than a SimulatedAnnealingResults namedtuple.  We therefore
        # first try driver-specific result attributes, then fall back to
        # scanning the cache for the layout with the minimum BIC.
        best_layout = None

        # (a) namedtuple / object attribute styles
        if hasattr(raw, "best_struct") and raw.best_struct is not None:
            best_layout = raw.best_struct
        elif hasattr(raw, "best_harmony") and raw.best_harmony is not None:
            best_layout = raw.best_harmony
        elif hasattr(raw, "best_solutions") and raw.best_solutions:
            best_layout = raw.best_solutions[-1]
        elif isinstance(raw, dict):
            best_layout = raw.get("best_solution") or raw.get("best_harmony")

        # (b) SA driver stores best_struct on the objective via Last_Sol;
        #     if still None, scan the cache for the minimum-BIC entry
        if best_layout is None and objective._cache:
            best_key = min(objective._cache, key=lambda k: objective._cache[k].get("bic", 1e20))
            best_layout = list(best_key)

        decoded = (
            objective.decode_solution(best_layout) if best_layout is not None else None
        )
        best_bic: Optional[float] = None
        if best_layout is not None:
            cached = objective._cache.get(
                tuple(int(v) for v in np.asarray(best_layout, dtype=int).tolist())
            )
            if cached is not None:
                best_bic = float(cached.get("bic", 1e20))

        return {
            "driver":              "metaheuristic",
            "algorithm":           algo,
            "family":              "multivariate",
            "raw_result":          raw,
            "best_solution":       best_layout,
            "best_decoded":        decoded,
            "best_bic":            best_bic,
            "multivariate_metadata": self.metadata or {
                "activity_cols":  self.activity_cols,
                "covariate_cols": self.covariate_cols,
            },
        }
