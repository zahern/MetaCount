"""Regression tests for the iteration-level L-BFGS objective trace."""
import numpy as np
import jax.numpy as jnp
from jaxopt import LBFGS

from metacountregressor.main_hpc import run_lbfgs_with_trace


def test_lbfgs_trace_matches_solver_run_and_records_every_iteration():
    def objective(x):
        return jnp.sum((x - 2.0) ** 2) + 0.1 * jnp.sum(x ** 4)

    init = jnp.array([-3.0, 1.0, 0.5])
    solver = LBFGS(
        fun=objective,
        maxiter=80,
        tol=1e-8,
        history_size=10,
    )

    plain = solver.run(init)
    traced, trace = run_lbfgs_with_trace(solver, init)

    assert np.allclose(
        np.asarray(plain.params), np.asarray(traced.params), atol=1e-6
    )
    assert abs(float(plain.state.value) - float(traced.state.value)) < 1e-6
    assert len(trace) == int(traced.state.iter_num) + 1
    assert trace[0]["iteration"] == 0
    assert trace[-1]["iteration"] == int(traced.state.iter_num)
    assert trace[-1]["objective"] <= trace[0]["objective"] + 1e-12
    assert all(np.isfinite(point["objective"]) for point in trace)
    assert all(np.isfinite(point["gradient_norm"]) for point in trace[1:])
