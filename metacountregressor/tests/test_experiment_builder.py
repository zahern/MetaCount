import numpy as np
import pandas as pd

from cmf_package import CMFExperimentBuilder
from experiment_package import ExperimentBuilder
from family_search import CMFFamilySearchProblem, DurationSearchProblem, LinearSearchProblem


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

    assert isinstance(linear_problem, LinearSearchProblem)
    assert linear_problem.family == "linear"
    assert isinstance(duration_problem, DurationSearchProblem)
    assert duration_problem.family == "duration"
    assert isinstance(cmf_problem, CMFFamilySearchProblem)
    assert cmf_problem.family == "cmf"
