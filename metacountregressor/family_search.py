from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

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
    from .metaheuristics import (
        differential_evolution,
        harmony_search,
        simulated_annealing,
    )
    from .GA_CMF_AADT_JAX import evaluate_model, fit_final_model
    from .solution import ObjectiveFunction
except ImportError:
    from cmf_package import CMFExperimentBuilder
    from duration_main import (
        estimate_model,
        ll_independent,
        ll_with_budget_penalty,
        prepare_data,
        predict_daily_schedule,
    )
    from metaheuristics import (
        differential_evolution,
        harmony_search,
        simulated_annealing,
    )
    from GA_CMF_AADT_JAX import evaluate_model, fit_final_model
    from solution import ObjectiveFunction


def _run_metaheuristic(algo: str, objective_function, **kwargs):
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
    df: pd.DataFrame
    y_col: str
    variables: list[str]
    objective_kwargs: dict[str, Any]

    family: str = "linear"

    def run(
        self,
        algo: str = "hs",
        initial_solutions=None,
        **algorithm_kwargs,
    ):
        X = self.df[self.variables].copy()
        y = self.df[[self.y_col]].copy()

        objective = ObjectiveFunction(
            X,
            y,
            linear_model=True,
            **self.objective_kwargs,
        )
        return _run_metaheuristic(
            algo,
            objective,
            initial_slns=initial_solutions,
            **algorithm_kwargs,
        )


@dataclass
class DurationSearchProblem:
    df: pd.DataFrame
    y_col: str
    variables: list[str]
    id_col: str
    budget_col: str

    family: str = "duration"

    def run(
        self,
        objective: str = "budget_penalty",
        init_params: Optional[np.ndarray] = None,
        lambda_penalty: float = 10.0,
    ) -> dict[str, Any]:
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
