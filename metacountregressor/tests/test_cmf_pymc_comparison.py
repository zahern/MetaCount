import numpy as np
import pytest

try:
    from compare_cmf_pymc import (
        print_cmf_pymc_comparison,
        run_cmf_pymc_comparison,
    )
except ImportError:
    from metacountregressor.compare_cmf_pymc import (
        print_cmf_pymc_comparison,
        run_cmf_pymc_comparison,
    )


def test_cmf_and_pymc_fit_are_reported_side_by_side(capsys):
    pytest.importorskip("pymc")

    result = run_cmf_pymc_comparison(
        jax_R=8,
        draws=20,
        tune=20,
        chains=1,
        cores=1,
        seed=42,
    )
    print_cmf_pymc_comparison(result)

    coefficients = result["coefficients"]
    metrics = result["metrics"]
    assert len(result["jax_predictions"]) == 275
    assert len(result["pymc_predictions"]) == 275
    assert np.isfinite(coefficients.iloc[:4, 1:5].to_numpy()).all()
    assert np.isfinite(metrics.iloc[:2, 1:3].to_numpy()).all()
    output = capsys.readouterr().out
    assert "JAX CMF NB2 MLE" in output
    assert "PyMC CMF NBL mean" in output
