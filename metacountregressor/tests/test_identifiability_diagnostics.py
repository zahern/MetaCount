import numpy as np
import jax.numpy as jnp

from metacountregressor.adaptive_search import IdentifiabilityChecker
from metacountregressor.main_hpc import ModelSpec, compute_standard_errors


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
