from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp

from metacountregressor.adaptive_search import IdentifiabilityChecker
from metacountregressor.main_hpc import (
    ModelSpec,
    compute_standard_errors,
    print_summary,
    random_parameter_identification_status,
)


def test_standard_errors_report_negative_curvature_without_ridge_masking():
    objective = lambda params: 0.5 * (params[0] ** 2 - params[1] ** 2)

    se, diagnostics = compute_standard_errors(
        jnp.array([1.0, 1.0]), objective, return_diagnostics=True
    )

    assert diagnostics["n_negative_eigenvalues"] == 1
    assert diagnostics["status"] == "curvature_issues"
    assert diagnostics["unreliable_indices"] == [1]
    assert np.isfinite(float(se[0]))
    assert np.isnan(float(se[1]))


def test_print_summary_does_not_turn_nan_se_into_p_one():
    spec = ModelSpec(
        Kf=1, Kr_ind=1, Kr_cor=0, Kg=0, Kh=0, model="poisson",
        fixed_names=("x",), random_ind_names=("r",),
        random_ind_dists=("normal",),
    )
    data = {"y": np.zeros((8, 1), dtype=float)}
    result = SimpleNamespace(params=np.array([0.5, 0.0, -7.0], dtype=float))
    se = np.array([0.1, np.nan, 0.01], dtype=float)

    df = print_summary(
        result, lambda p: 0.0, data, spec, None, se=se, return_df=True
    )

    assert df is not None
    row = df[df["Parameter"] == "mean(r)"].iloc[0]
    assert np.isnan(float(row["Std.Err"]))
    assert np.isnan(float(row["z-value"]))
    assert np.isnan(float(row["p-value"]))


def test_random_parameter_identification_flags_boundary_and_nonstationary():
    report = {
        "has_random_parameters": True,
        "stationary": False,
        "boundary_parameters": ["sd(r)"],
        "divergent_mean_parameters": [],
        "max_mean_scale_coupling": 0.2,
        "random_hessian_condition": 1e16,
    }
    identified, reason = random_parameter_identification_status(report)

    assert not identified
    assert "boundary" in reason
    assert "not stationary" in reason
    assert "Hessian" in reason


def test_random_parameter_identification_accepts_clean_fit():
    report = {
        "has_random_parameters": True,
        "stationary": True,
        "boundary_parameters": [],
        "divergent_mean_parameters": [],
        "max_mean_scale_coupling": 0.2,
        "random_hessian_condition": 100.0,
    }
    identified, reason = random_parameter_identification_status(report)

    assert identified
    assert reason == ""


def test_checker_labels_contiguous_random_mean_and_sd_blocks():
    spec = ModelSpec(
        Kf=1,
        Kr_ind=1,
        Kr_cor=0,
        Kg=0,
        Kh=0,
        model="nb",
        fixed_names=("x",),
        random_ind_names=("RSDSDG10",),
    )
    objective = lambda params: 0.5 * jnp.sum(params * params)

    result = IdentifiabilityChecker().check(
        np.array([1.0, 2.0, 3.0, 0.25]), objective, spec
    )

    assert [item.name for item in result["diagnostics"]] == [
        "x",
        "RSDSDG10:mean",
        "RSDSDG10:sd",
        "dispersion",
    ]
    assert result["se_diagnostics"]["status"] == "ok"
    assert result["globally_ok"]
