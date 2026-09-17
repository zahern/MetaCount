"""Regression tests for the bivariate + vine copula families.

Covers the Joe and Gumbel parameterisations added alongside the existing
Frank / Gaussian / Clayton(Kimeldorf) families:

  * registry wiring,
  * the independence limit (theta == 1  ->  C(u,v) = u*v),
  * validity as a 2-increasing copula (non-negative rectangle cell mass),
  * end-to-end bivariate NB copula fits recovering positive dependence,
  * multivariate vine fits (vine-joe / vine-gumbel) producing a finite
    implied correlation matrix.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from metacountregressor import bivariate_copula as bc
from metacountregressor import multivariate_count_regressor as mv


def test_registry_has_joe_and_gumbel():
    assert "joe" in bc.COPULA_LOGS
    assert "gumbel" in bc.COPULA_LOGS


@pytest.mark.parametrize("fn", [bc.joe_logcdf, bc.gumbel_logcdf])
def test_independence_limit(fn):
    u = jnp.array([0.15, 0.4, 0.75])
    v = jnp.array([0.25, 0.6, 0.85])
    log_c = fn(u, v, jnp.array(1.0))            # theta == 1 -> independence
    expected = jnp.log(u) + jnp.log(v)
    assert float(jnp.max(jnp.abs(log_c - expected))) < 1e-8
    assert bool(jnp.all(log_c <= 1e-6))         # valid log-CDF (C <= 1)


@pytest.mark.parametrize("fn", [bc.joe_logcdf, bc.gumbel_logcdf])
def test_two_increasing(fn):
    """Every rectangle has non-negative C-measure (valid copula)."""
    g = np.linspace(0.05, 0.95, 10)
    theta = jnp.array(3.0)
    C = lambda a, b: float(jnp.exp(fn(jnp.array(a), jnp.array(b), theta)))
    worst = min(
        C(g[i + 1], g[j + 1]) - C(g[i], g[j + 1]) - C(g[i + 1], g[j]) + C(g[i], g[j])
        for i in range(len(g) - 1) for j in range(len(g) - 1)
    )
    assert worst >= -1e-9


@pytest.mark.parametrize("copula", ["joe", "gumbel"])
def test_bivariate_fit_recovers_positive_dependence(copula):
    rng = np.random.default_rng(1)
    n = 1200
    z = rng.gamma(2.0, 0.5, n)                    # shared frailty -> positive dep
    x = np.c_[np.ones(n), rng.standard_normal(n)]
    y1 = rng.poisson(np.exp(0.3 + 0.4 * x[:, 1]) * z)
    y2 = rng.poisson(np.exp(0.2 + 0.3 * x[:, 1]) * z)
    off = np.zeros(n)
    fit = bc.fit_copula_bivariate_nb(y1 * 1.0, y2 * 1.0, x, x, off, off,
                                     copula=copula, method="scipy-lbfgsb",
                                     n_iter=200)
    assert fit.converged
    assert np.isfinite(fit.loglik)
    assert fit.rho > 1.0                          # theta > 1  => positive dep
    assert np.isfinite(fit.se_rho) and fit.se_rho > 0.0


@pytest.mark.parametrize("copula", ["vine-joe", "vine-gumbel", "vine-frank"])
def test_multivariate_vine_fit(copula):
    rng = np.random.default_rng(3)
    n = 900
    z = rng.gamma(2.0, 0.5, n)
    x = np.c_[np.ones(n), rng.standard_normal(n)]
    betas = [np.array([0.2, 0.3]), np.array([0.1, 0.2]), np.array([0.3, -0.1])]
    Y = np.column_stack([rng.poisson(np.exp(x @ b) * z) for b in betas])
    m = mv.MultivariateCountRegressor(activity_names=["a", "b", "c"],
                                      copula=copula, marginal="nb",
                                      maxiter=120, verbose=False)
    fit = m.fit([x, x, x], Y, feature_names=[["const", "x"]] * 3)
    R = np.asarray(fit.correlation)
    assert R.shape == (3, 3)
    assert np.all(np.isfinite(R))
    off = R[np.triu_indices(3, 1)]
    assert np.all(off > 0.0)                      # positive dependence recovered


def test_unknown_copula_rejected():
    with pytest.raises(ValueError):
        mv.MultivariateCountRegressor(activity_names=["a", "b"],
                                      copula="vine-banana")
