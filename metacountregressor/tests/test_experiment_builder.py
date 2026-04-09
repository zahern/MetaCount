import numpy as np
import pandas as pd
from pathlib import Path

from cmf_package import CMFExperimentBuilder
from experiment_package import ExperimentBuilder
from family_search import (
    CMFFamilySearchProblem,
    CMFMetaheuristicObjective,
    DurationSearchProblem,
    LinearSearchProblem,
    UnifiedCMFSearchProblem,
)
from output_config import SearchOutputConfig, save_search_result
from sample_data import load_example16_3_raw_data, load_example_crash_data


def make_panel_df():
    return pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "Y": [0, 1, 2, 1],
            "OFFSET": np.zeros(4),
            "FC": [1, 1, 2, 2],
            "x_fixed": [0.2, 0.4, 0.6, 0.8],
            "x_rnd_ind": [1.0, 1.2, 1.4, 1.6],
            "x_rnd_cor": [2.0, 2.2, 2.4, 2.6],
            "x_rnd_cor_2": [0.3, 0.6, 0.9, 1.2],
            "x_grouped": [1, 0, 1, 0],
            "x_hetero": [4.0, 4.5, 5.0, 5.5],
            "x_zi": [0, 1, 0, 1],
            "x_member": [1, 0, 1, 0],
            "x_member_fixed": [0.5, 0.25, 0.75, 0.3],
            "AADT": [10000, 11000, 12000, 13000],
            "cmf_a": [1, 0, 1, 0],
            "cmf_b": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_build_spec_covers_heterogeneity_roles():
    df = make_panel_df()
    builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="Y",
        offset_col="OFFSET",
        group_id_col="FC",
    )
    variables = [
        "x_fixed",
        "x_rnd_ind",
        "x_rnd_cor",
        "x_rnd_cor_2",
        "x_grouped",
        "x_hetero",
        "x_zi",
        "x_member",
        "x_member_fixed",
    ]
    evaluator = builder.build_evaluator(
        variables=variables,
        mode="single",
        max_latent_classes=3,
        R=8,
        default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )

    decision = np.array(
        [
            1, 2, 3, 3, 4, 5, 6, 7, 8,
            0, 0, 1, 3, 2, 0, 0, 0, 0,
            1, 1,
        ]
    )
    spec = evaluator.build_spec(decision)

    assert spec["fixed_terms"] == ["x_fixed", "x_member_fixed"]
    assert spec["rdm_terms"] == ["x_rnd_ind:normal"]
    assert [term.split(":")[0] for term in spec["rdm_cor_terms"]] == [
        "x_rnd_cor",
        "x_rnd_cor_2",
    ]
    assert spec["grouped_terms"][0].split(":")[0] == "x_grouped"
    assert spec["hetro_in_means"] == ["x_hetero"]
    assert spec["zi_terms"] == ["x_zi"]
    assert spec["membership_terms"] == ["x_member", "x_member_fixed"]
    assert spec["dispersion"] == 1
    assert spec["latent_classes"] == 2


def test_membership_roles_collapse_when_single_class():
    df = make_panel_df()
    builder = ExperimentBuilder(df=df, id_col="ID", y_col="Y", offset_col="OFFSET")
    evaluator = builder.build_evaluator(
        variables=["x_member", "x_member_fixed"],
        mode="single",
        max_latent_classes=1,
        R=8,
        default_roles=[0, 1, 7, 8],
    )

    decision = np.array([7, 8, 0, 0, 0, 0])
    spec = evaluator.build_spec(decision)

    assert spec["membership_terms"] == []
    assert spec["fixed_terms"] == ["x_member_fixed"]
    assert spec["latent_classes"] == 1


def test_cmf_builder_bridges_to_general_latent_class_search():
    df = make_panel_df()
    builder = CMFExperimentBuilder(
        df=df,
        y_col="Y",
        aadt_col="AADT",
        baseline_vars=["cmf_a"],
        local_vars=["cmf_b"],
    )

    general_builder, evaluator = builder.build_latent_class_evaluator(
        id_col="ID",
        offset_col="OFFSET",
        max_latent_classes=2,
        R=8,
        default_roles=[0, 1, 2, 7, 8],
    )

    assert isinstance(general_builder, ExperimentBuilder)
    assert evaluator.max_latent_classes == 2
    assert evaluator.vars == ["cmf_a", "AADT", "cmf_b"]


def test_model_family_switches_return_specialized_search_problems():
    df = make_panel_df()
    builder = ExperimentBuilder(df=df, id_col="ID", y_col="Y", offset_col="OFFSET")

    linear_problem = builder.build_evaluator(
        model_family="linear",
        variables=["x_fixed", "x_rnd_ind"],
        objective_kwargs={"algorithm": "hs", "_max_time": 1},
    )
    duration_problem = builder.build_evaluator(
        model_family="duration",
        variables=["x_fixed", "x_rnd_ind"],
        budget_col="AADT",
    )
    cmf_problem = builder.build_evaluator(
        model_family="cmf",
        aadt_col="AADT",
        baseline_vars=["cmf_a"],
        local_vars=["cmf_b"],
    )
    legacy_cmf_problem = builder.build_evaluator(
        model_family="cmf",
        aadt_col="AADT",
        baseline_vars=["cmf_a"],
        local_vars=["cmf_b"],
        cmf_driver="ga",
    )

    assert isinstance(linear_problem, LinearSearchProblem)
    assert linear_problem.family == "linear"
    assert linear_problem.metadata["model"] == "gaussian"
    assert isinstance(duration_problem, DurationSearchProblem)
    assert duration_problem.family == "duration"
    assert duration_problem.metadata["model"] == "lognormal"
    assert isinstance(cmf_problem, UnifiedCMFSearchProblem)
    assert cmf_problem.family == "cmf"
    assert isinstance(legacy_cmf_problem, CMFFamilySearchProblem)
    assert legacy_cmf_problem.family == "cmf"


def test_cmf_metaheuristic_objective_decodes_solution():
    df = make_panel_df().rename(columns={"Y": "FREQ"})
    objective = CMFMetaheuristicObjective(
        df=df,
        baseline_vars=["cmf_a"],
        local_vars=["cmf_b"],
        R=8,
    )

    vector = np.array([1, 1, 1, 0, 1, 0], dtype=int)
    decoded = objective.decode_solution(vector)

    assert decoded["selected_baseline"] == ["cmf_a"]
    assert decoded["selected_local"] == ["cmf_b"]
    assert decoded["use_halton"] is True
    assert decoded["model"] == "poisson"
    assert decoded["rand_baseline"] == (True,)
    assert decoded["rand_local"] == (False,)


def test_cmf_default_driver_uses_main_jax_count_architecture():
    df = make_panel_df()
    builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="Y",
        offset_col="OFFSET",
        group_id_col="FC",
    )

    cmf_problem = builder.build_evaluator(
        model_family="cmf",
        aadt_col="AADT",
        baseline_vars=["cmf_a"],
        local_vars=["cmf_b"],
        variables=["x_hetero", "x_zi", "x_member"],
        max_latent_classes=2,
        R=8,
    )

    assert isinstance(cmf_problem, UnifiedCMFSearchProblem)
    assert cmf_problem.metadata["driver"] == "jax_count"
    assert cmf_problem.metadata["log_aadt_col"] == "__cmf_log_aadt"
    assert "__cmf_local__cmf_b" in cmf_problem.metadata["interaction_cols"]
    assert "__cmf_log_aadt" in cmf_problem.evaluator.vars
    assert "__cmf_local__cmf_b" in cmf_problem.evaluator.vars
    assert "x_hetero" in cmf_problem.evaluator.vars
    assert "x_zi" in cmf_problem.evaluator.vars
    assert "x_member" in cmf_problem.evaluator.vars


def test_manual_spec_helper_preserves_roles_and_model_fit():
    df = make_panel_df()
    builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="Y",
        offset_col="OFFSET",
        group_id_col="FC",
    )

    manual_spec = builder.make_manual_spec(
        fixed_terms=["x_fixed"],
        rdm_terms=["x_rnd_ind:normal"],
        hetro_in_means=["x_hetero"],
        dispersion=1,
        latent_classes=1,
    )
    fit = builder.fit_manual_model(manual_spec, model="nb", R=8)

    assert fit["manual_spec"]["fixed_terms"] == ["x_fixed"]
    assert fit["spec"].model == "nb"
    assert fit["spec"].latent_classes == 1
    assert fit["predictions"].shape[0] == df["ID"].nunique()


def test_family_builders_raise_on_unexpected_kwargs():
    df = make_panel_df()
    builder = ExperimentBuilder(df=df, id_col="ID", y_col="Y", offset_col="OFFSET")

    try:
        builder.build_evaluator(
            model_family="duration",
            variables=["x_fixed"],
            budget_col="AADT",
            not_a_real_arg=2,
        )
    except ValueError as exc:
        assert "Unexpected arguments for duration search" in str(exc)
    else:
        raise AssertionError("duration search should reject unexpected kwargs")


def test_cmf_manual_helpers_build_transformed_spec_and_fit():
    df = make_panel_df()
    cmf_builder = CMFExperimentBuilder(
        df=df,
        y_col="Y",
        aadt_col="AADT",
        baseline_vars=["cmf_a"],
        local_vars=["cmf_b"],
    )

    manual_spec = cmf_builder.make_manual_cmf_spec(
        baseline_fixed=["cmf_a"],
        local_fixed=["cmf_b"],
        zi_terms=["x_zi"],
        membership_terms=["x_member"],
        dispersion=1,
        latent_classes=2,
    )
    fit = cmf_builder.fit_manual_cmf_model(
        id_col="ID",
        offset_col="OFFSET",
        group_id_col="FC",
        manual_spec=manual_spec,
        model="nb",
        R=8,
    )

    assert "__cmf_log_aadt" in manual_spec["fixed_terms"]
    assert "__cmf_local__cmf_b" in manual_spec["fixed_terms"]
    assert fit["spec"].model == "nb"
    assert fit["spec"].latent_classes == 2


def test_sample_data_loader_contains_readme_columns():
    df = load_example_crash_data()
    required = {
        "ID", "FREQ", "OFFSET", "FC", "FC_ENCODED", "FC_LABEL", "AADT", "LENGTH",
        "INCLANES", "DECLANES", "WIDTH", "MIMEDSH", "MXMEDSH", "SPEED", "URB",
        "SINGLE", "DOUBLE", "TRAIN", "PEAKHR", "GRADEBR", "MIGRADE", "MXGRADE",
        "MXGRDIFF", "TANGENT", "CURVES", "MINRAD", "ACCESS", "MEDWIDTH",
        "FRICTION", "ADTLANE", "SLOPE", "INTECHAG", "AVEPRE", "AVESNOW",
    }
    assert required.issubset(df.columns)


def test_example16_3_raw_loader_preserves_source_columns():
    df = load_example16_3_raw_data()
    assert list(df.columns) == [
        "ID", "FREQ", "LENGTH", "INCLANES", "DECLANES", "WIDTH", "MIMEDSH", "MXMEDSH",
        "SPEED", "URB", "FC", "AADT", "SINGLE", "DOUBLE", "TRAIN", "PEAKHR", "GRADEBR",
        "MIGRADE", "MXGRADE", "MXGRDIFF", "TANGENT", "CURVES", "MINRAD", "ACCESS",
        "MEDWIDTH", "FRICTION", "ADTLANE", "SLOPE", "INTECHAG", "AVEPRE", "AVESNOW",
    ]


def test_output_config_saves_search_result():
    output_dir = Path("test_results_output")
    output_dir.mkdir(exist_ok=True)
    config = SearchOutputConfig(output_dir=str(output_dir), experiment_name="unit_test", search_description="count search")
    target = save_search_result({"best_score": 12.34, "family": "count"}, config, family="count", algorithm="sa")
    assert target.exists()
