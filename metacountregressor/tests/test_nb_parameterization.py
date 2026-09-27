import numpy as np
import pytest
from scipy.stats import nbinom

import jax
import jax.numpy as jnp

from metacountregressor.main_hpc import (
    ModelSpec,
    mixed_model_loglik,
    nb2_loglik,
)
from metacountregressor.nb_parameterization import nb2_logpmf


def _count_data(y):
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1, 1)
    n_obs = y.shape[0]
    return {
        "Xf": jnp.ones((n_obs, 1, 1), dtype=jnp.float64),
        "Xzi": jnp.ones((n_obs, 1, 1), dtype=jnp.float64),
        "y": y,
        "mask": jnp.ones_like(y),
        "offset": jnp.zeros_like(y),
        "draws_ind": jnp.zeros((n_obs, 0, 1), dtype=jnp.float64),
        "draws_cor": jnp.zeros((n_obs, 0, 1), dtype=jnp.float64),
        "draws_g": jnp.zeros((1, 0, 1), dtype=jnp.float64),
        "group_ids": jnp.zeros(n_obs, dtype=jnp.int32),
    }


def test_nb2_log_alpha_matches_scipy_and_has_finite_gradient():
    y = jnp.array([0.0, 1.0, 4.0, 12.0])
    eta = jnp.log(jnp.array([0.7, 2.0, 5.0, 11.0]))
    log_alpha = jnp.log(0.8)

    actual = np.asarray(nb2_logpmf(y, eta, log_alpha))
    mu = np.exp(np.asarray(eta))
    expected = nbinom.logpmf(np.asarray(y), 1.0 / 0.8, 1.0 / (1.0 + 0.8 * mu))
    gradient = jax.grad(lambda value: jnp.sum(nb2_logpmf(y, eta, value)))(log_alpha)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)
    assert np.isfinite(float(gradient))


def test_canonical_zi_and_latent_class_paths_use_log_alpha_nb2():
    data = _count_data([0.0, 1.0, 4.0])
    alpha = 0.8
    log_alpha = np.log(alpha)
    eta = np.log(2.0)

    ordinary_spec = ModelSpec(
        Kf=1, Kr_ind=0, Kr_cor=0, Kg=0, Kh=0, model="nb"
    )
    ordinary_params = jnp.array([eta, log_alpha])
    ordinary = np.asarray(
        mixed_model_loglik(ordinary_params, data, ordinary_spec, indivi=True)
    )
    expected_ordinary = np.asarray(
        nb2_loglik(data["y"], jnp.full_like(data["y"], eta), log_alpha)
    ).ravel()
    np.testing.assert_allclose(ordinary, expected_ordinary, atol=1e-10)

    zi_spec = ModelSpec(
        Kf=1, Kr_ind=0, Kr_cor=0, Kg=0, Kh=0,
        Kzi=1, zero_inflated=True, model="nb",
    )
    zi_params = jnp.array([eta, -1.25, log_alpha])
    zi_nll = float(mixed_model_loglik(zi_params, data, zi_spec))
    pi = 1.0 / (1.0 + np.exp(1.25))
    f0 = np.exp(expected_ordinary)
    expected_zi = np.where(
        np.asarray(data["y"]).ravel() == 0.0,
        np.log(pi + (1.0 - pi) * f0 + 1e-12),
        np.log(1.0 - pi + 1e-12) + expected_ordinary,
    )
    np.testing.assert_allclose(zi_nll, -expected_zi.sum(), atol=1e-10)

    latent_spec = ModelSpec(
        Kf=1, Kr_ind=0, Kr_cor=0, Kg=0, Kh=0,
        model="nb", latent_classes=2,
    )
    latent_params = jnp.array([eta, log_alpha, eta + 0.1, log_alpha, 0.2])
    latent_nll = mixed_model_loglik(latent_params, data, latent_spec)
    assert np.isfinite(float(latent_nll))


def test_numba_nb2_uses_the_same_log_alpha_coordinate():
    numba = pytest.importorskip("numba")
    del numba
    from metacountregressor.numba_rp import _nb2_alpha

    assert _nb2_alpha(0.0) == pytest.approx(1.0)
    assert _nb2_alpha(np.log(0.8)) == pytest.approx(0.8)
