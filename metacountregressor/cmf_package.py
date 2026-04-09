from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import re
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
        try:
            from .GA_CMF_AADT_JAX import (
                build_summary_table,
                compute_se,
                fit_final_model,
                print_cmf_results,
                print_summary_table,
                run_ga,
            )
        except ImportError:
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
        try:
            from .experiment_package import ExperimentBuilder
        except ImportError:
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
        try:
            from .experiment_package import ExperimentBuilder
        except ImportError:
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

    @staticmethod
    def _safe_token(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "var"

    def _cmf_term_map(self) -> dict[str, str]:
        term_map: dict[str, str] = {self.aadt_col: "__cmf_log_aadt"}
        for var in self.local_vars:
            term_map[var] = f"__cmf_local__{self._safe_token(var)}"
        return term_map

    def build_jax_count_evaluator(
        self,
        id_col: str,
        offset_col: Optional[str] = None,
        group_id_col: Optional[str] = None,
        variables: Optional[list[str]] = None,
        fixed_override: Optional[dict[str, list[int]]] = None,
        membership_override: Optional[dict[str, list[int]]] = None,
        exclude: Optional[list[str]] = None,
        mode: str = "single",
        max_latent_classes: int = 1,
        R: int = 200,
        default_roles: Optional[list[int]] = None,
        force_aadt_term: bool = True,
    ):
        try:
            from .experiment_package import ExperimentBuilder
        except ImportError:
            from experiment_package import ExperimentBuilder

        if (self.df[self.aadt_col] <= 0).any():
            raise ValueError(
                f"CMF search requires strictly positive values in aadt_col='{self.aadt_col}'."
            )

        df = self.df.copy()
        term_map = self._cmf_term_map()
        log_aadt_col = term_map[self.aadt_col]
        df[log_aadt_col] = np.log(df[self.aadt_col].astype(float))

        interaction_cols: list[str] = []
        for var in self.local_vars:
            interaction_col = term_map[var]
            df[interaction_col] = df[var].astype(float) * df[log_aadt_col]
            interaction_cols.append(interaction_col)

        auxiliary_vars = []
        for var in variables or []:
            mapped = term_map.get(var, var)
            if mapped not in auxiliary_vars:
                auxiliary_vars.append(mapped)

        search_vars = list(dict.fromkeys([
            *self.baseline_vars,
            log_aadt_col,
            *interaction_cols,
            *auxiliary_vars,
        ]))

        translated_fixed: dict[str, list[int]] = {}
        for mapping in (fixed_override or {}).items():
            key, allowed = mapping
            translated_fixed[term_map.get(key, key)] = allowed

        translated_membership: dict[str, list[int]] = {}
        for mapping in (membership_override or {}).items():
            key, allowed = mapping
            translated_membership[term_map.get(key, key)] = allowed

        translated_exclude = [term_map.get(name, name) for name in (exclude or [])]

        if force_aadt_term:
            translated_fixed.setdefault(log_aadt_col, [1])
            translated_exclude = [name for name in translated_exclude if name != log_aadt_col]

        if default_roles is None:
            default_roles = [0, 1, 2, 3, 4, 5, 6]
            if max_latent_classes > 1:
                default_roles.extend([7, 8])

        builder = ExperimentBuilder(
            df=df,
            id_col=id_col,
            y_col=self.y_col,
            offset_col=offset_col,
            group_id_col=group_id_col,
        )
        evaluator = builder.build_count_evaluator(
            variables=search_vars,
            fixed_override=translated_fixed,
            membership_override=translated_membership,
            exclude=translated_exclude,
            mode=mode,
            max_latent_classes=max_latent_classes,
            R=R,
            default_roles=default_roles,
        )

        metadata = {
            "family": "cmf",
            "driver": "jax_count",
            "aadt_col": self.aadt_col,
            "log_aadt_col": log_aadt_col,
            "baseline_vars": list(self.baseline_vars),
            "local_vars": list(self.local_vars),
            "interaction_cols": interaction_cols,
            "term_map": term_map,
            "search_vars": search_vars,
        }
        setattr(evaluator, "cmf_metadata", metadata)
        return builder, evaluator, metadata

    def make_manual_cmf_spec(
        self,
        baseline_fixed: Optional[list[str]] = None,
        baseline_random: Optional[list[str]] = None,
        baseline_correlated: Optional[list[str]] = None,
        local_fixed: Optional[list[str]] = None,
        local_random: Optional[list[str]] = None,
        local_correlated: Optional[list[str]] = None,
        grouped_terms: Optional[list[str]] = None,
        hetro_in_means: Optional[list[str]] = None,
        zi_terms: Optional[list[str]] = None,
        membership_terms: Optional[list[str]] = None,
        dispersion: int = 0,
        latent_classes: int = 1,
    ) -> dict[str, Any]:
        term_map = self._cmf_term_map()
        fixed_terms = list(baseline_fixed or [])
        fixed_terms.append(term_map[self.aadt_col])
        fixed_terms.extend(term_map[var] for var in (local_fixed or []))

        rdm_terms = [f"{var}:normal" for var in (baseline_random or [])]
        rdm_terms.extend(f"{term_map[var]}:normal" for var in (local_random or []))

        rdm_cor_terms = [f"{var}:normal" for var in (baseline_correlated or [])]
        rdm_cor_terms.extend(f"{term_map[var]}:normal" for var in (local_correlated or []))

        return {
            "fixed_terms": list(dict.fromkeys(fixed_terms)),
            "rdm_terms": rdm_terms,
            "rdm_cor_terms": rdm_cor_terms,
            "grouped_terms": grouped_terms or [],
            "hetro_in_means": hetro_in_means or [],
            "zi_terms": zi_terms or [],
            "membership_terms": membership_terms or [],
            "dispersion": int(dispersion),
            "latent_classes": int(latent_classes),
        }

    def fit_manual_cmf_model(
        self,
        id_col: str,
        manual_spec: dict[str, Any],
        offset_col: Optional[str] = None,
        group_id_col: Optional[str] = None,
        model: str = "poisson",
        R: int = 200,
    ) -> dict[str, Any]:
        try:
            from .experiment_package import ExperimentBuilder
        except ImportError:
            from experiment_package import ExperimentBuilder

        df = self.df.copy()
        term_map = self._cmf_term_map()
        log_aadt_col = term_map[self.aadt_col]
        if log_aadt_col not in df.columns:
            df[log_aadt_col] = np.log(df[self.aadt_col].astype(float))
        for var in self.local_vars:
            interaction_col = term_map[var]
            if interaction_col not in df.columns:
                df[interaction_col] = df[var].astype(float) * df[log_aadt_col]

        manual_builder = ExperimentBuilder(
            df=df,
            id_col=id_col,
            y_col=self.y_col,
            offset_col=offset_col,
            group_id_col=group_id_col,
        )
        return manual_builder.fit_manual_model(manual_spec=manual_spec, model=model, R=R)
