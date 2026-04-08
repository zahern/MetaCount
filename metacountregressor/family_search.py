from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd

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
        algo: str = "ga",
        final_R: Optional[int] = None,
        fit_final: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        algo = algo.lower()
        if algo not in {"ga", "cmf"}:
            if self.id_col is None:
                raise ValueError("id_col is required when routing CMF variables into the general search.")

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

        search_result = self.builder.run_search(R=kwargs.pop("R", 200))
        output = {"search_result": search_result}

        if fit_final:
            fit_result = self.builder.fit_best_model(
                search_result,
                final_R=final_R or kwargs.pop("final_R", 500),
            )
            output["fit_result"] = fit_result

        return output
