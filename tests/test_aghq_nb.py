"""Regression tests for the adaptive NB2 quadrature path."""

import numpy as np
import jax.numpy as jnp

from metacountregressor.GA_CMF_AADT_JAX import (
    _aghq_loglik,
    _ghq_draws,
    _mixed_ll,
)


def test_aghq_nb_matches_high_order_fixed_quadrature():
    """NB2 AGHQ must use an NB2 score when locating the conditional mode."""
    y = jnp.array([12.0])
    aadt = jnp.array([1000.0])
    baseline = jnp.array([[1.0]])
    local = jnp.zeros((1, 0))
    params = jnp.array([-5.0, 0.0, 0.5, 0.0, -0.5])

    draws, log_weights = _ghq_draws(1, 1, points=25)
    fixed_nll = float(
        _mixed_ll(
            params,
            y,
            aadt,
            baseline,
            local,
            (True,),
            (),
            draws,
            "nb",
            None,
            log_weights,
        )
    )
    adaptive_nll = float(
        _aghq_loglik(
            params,
            y,
            aadt,
            baseline,
            local,
            (True,),
            (),
            "nb",
            None,
            points=25,
        )
    )

    assert np.isfinite(adaptive_nll)
    assert abs(adaptive_nll - fixed_nll) < 1e-5
