from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from experiment_package import ExperimentBuilder


@dataclass
class CMFSearchResult:
    selected_baseline: list[str]
    selected_local: list[str]
    rand_baseline: list[bool]
    rand_local: list[bool]
    use_halton: bool
    model: str
    fitness: float


class CMFExperimentBuilder:
    """
    Thin package-friendly wrapper around the GA CMF search code.

    It keeps the original AADT-specific CMF search available while also
    providing a bridge back into the general ExperimentBuilder API for
    latent-class and broader mixed-model searches.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_col: str,
        aadt_col: str,
        baseline_vars: list[str],
        local_vars: list[str],
    ):
        self.df = df.copy()
        self.y_col = y_col
        self.aadt_col = aadt_col
        self.baseline_vars = list(baseline_vars)
        self.local_vars = list(local_vars)
        self._validate_columns()

    def _validate_columns(self) -> None:
        required = {self.y_col, self.aadt_col, *self.baseline_vars, *self.local_vars}
        missing = [column for column in required if column not in self.df.columns]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing required CMF columns: {missing_text}")

    @staticmethod
    def _cmf_api():
        from GA_CMF_AADT_JAX import (
            build_summary_table,
            compute_se,
            fit_final_model,
            print_cmf_results,
            print_summary_table,
            run_ga,
        )

        return {
            "build_summary_table": build_summary_table,
            "compute_se": compute_se,
            "fit_final_model": fit_final_model,
            "print_cmf_results": print_cmf_results,
            "print_summary_table": print_summary_table,
            "run_ga": run_ga,
        }

    def run_search(self, R: int = 200) -> CMFSearchResult:
        api = self._cmf_api()
        result = api["run_ga"](
            self.df,
            self.baseline_vars,
            self.local_vars,
            R=R,
        )
        return CMFSearchResult(*result)

    def fit_best_model(
        self,
        search_result: CMFSearchResult,
        final_R: int = 500,
    ) -> dict[str, Any]:
        api = self._cmf_api()
        fit_result = api["fit_final_model"](
            self.df,
            search_result.selected_baseline,
            search_result.selected_local,
            search_result.rand_baseline,
            search_result.rand_local,
            model=search_result.model,
            use_halton=search_result.use_halton,
            R=final_R,
        )

        result, y_jax, AADT_jax, baseline_jax, locals_jax, draws = fit_result
        se, cov = api["compute_se"](
            result,
            y_jax,
            AADT_jax,
            baseline_jax,
            locals_jax,
            search_result.rand_baseline,
            search_result.rand_local,
            draws,
            R=final_R,
            model=search_result.model,
        )
        summary = api["build_summary_table"](
            result,
            search_result.selected_baseline,
            search_result.selected_local,
            search_result.rand_baseline,
            search_result.rand_local,
            search_result.model,
            se,
        )

        return {
            "result": result,
            "summary": summary,
            "standard_errors": se,
            "covariance": cov,
        }

    def print_report(
        self,
        search_result: CMFSearchResult,
        fit_result: dict[str, Any],
    ) -> None:
        api = self._cmf_api()
        api["print_summary_table"](fit_result["summary"])
        api["print_cmf_results"](
            fit_result["result"],
            search_result.selected_baseline,
            search_result.selected_local,
            search_result.rand_baseline,
            search_result.rand_local,
            self.df[self.aadt_col].mean(),
            model=search_result.model,
        )

    def to_experiment_builder(
        self,
        id_col: str,
        offset_col: Optional[str] = None,
        group_id_col: Optional[str] = None,
    ) -> "ExperimentBuilder":
        from experiment_package import ExperimentBuilder

        return ExperimentBuilder(
            df=self.df,
            id_col=id_col,
            y_col=self.y_col,
            offset_col=offset_col,
            group_id_col=group_id_col,
        )

    def build_latent_class_evaluator(
        self,
        id_col: str,
        offset_col: Optional[str] = None,
        group_id_col: Optional[str] = None,
        mode: str = "single",
        max_latent_classes: int = 2,
        R: int = 200,
        default_roles: Optional[list[int]] = None,
        membership_override: Optional[dict[str, list[int]]] = None,
        fixed_override: Optional[dict[str, list[int]]] = None,
        exclude: Optional[list[str]] = None,
    ):
        from experiment_package import ExperimentBuilder

        builder = self.to_experiment_builder(
            id_col=id_col,
            offset_col=offset_col,
            group_id_col=group_id_col,
        )
        variables = [*self.baseline_vars, self.aadt_col, *self.local_vars]
        deduped_variables = list(dict.fromkeys(variables))
        evaluator = builder.build_evaluator(
            variables=deduped_variables,
            fixed_override=fixed_override,
            membership_override=membership_override,
            exclude=exclude,
            mode=mode,
            max_latent_classes=max_latent_classes,
            R=R,
            default_roles=default_roles,
        )
        return builder, evaluator
