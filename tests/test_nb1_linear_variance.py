"""Regression tests: the linear-variance negative binomial (``model="nb1"``).

The engine has always carried an ``nb1_loglik`` for the negative binomial with
``Var = mu * (1 + alpha)``, but nothing dispatched to it: ``model`` accepted only
``poisson``, ``nb`` (NB2, ``Var = mu + alpha * mu**2``) and the continuous
families.  These tests pin the behaviour now that ``"nb1"`` is selectable:

  * the implemented pmf really has a linear variance, measured from the pmf
    itself rather than restated from the algebra;
  * it collapses onto the Poisson as alpha -> 0, which is the nesting any
    dispersion parameter must satisfy;
  * it agrees with an independent NumPy implementation at a fixed parameter
    vector, so the two cannot drift apart silently;
  * selecting it allocates exactly one dispersion parameter, and the historical
    ``dispersion`` flag still selects NB2.

The last two matter most: the dispersion is the last parameter in the vector, so
getting the index wrong silently shifts every coefficient by one slot.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

import metacountregressor.experiment_package as _ep  # noqa: F401  (patches main_hpc)
from metacountregressor.main_hpc import (
    build_base_index,
    build_model_from_manual_spec,
    nb1_loglik,
)

TERMS = ["EXPOSE", "LOWPRE"]


def _reference_nb1(y, mu, log_alpha):
    """Independent NB1 log-pmf: r = mu/alpha, p = 1/(1+alpha)."""
    a = np.exp(log_alpha)
    r = mu / a
    p = 1.0 / (1.0 + a)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(p) + y * np.log(1.0 - p))


def _frame(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ID": np.arange(n),
        "FREQ": rng.poisson(4.0, n).astype(float),
        "EXPOSE": rng.uniform(0.1, 3.0, n),
        "LOWPRE": rng.integers(0, 2, n).astype(float),
    })


def _build(model_key, dispersion=1):
    spec = {"fixed_terms": TERMS, "rdm_terms": [], "rdm_cor_terms": [],
            "grouped_terms": [], "hetro_in_means": [], "hetro_in_variances": [],
            "zi_terms": [], "dispersion": dispersion}
    if model_key is not None:
        spec["model"] = model_key
    return build_model_from_manual_spec(
        _frame(), spec, id_col="ID", y_col="FREQ", offset_col=None,
        standardize=False, R=1)


# ── the likelihood itself ───────────────────────────────────────────────────

def test_nb1_variance_is_linear_in_the_mean():
    # Sum the pmf over a wide support and measure the moments it implies.  A
    # NB2 implementation would give Var/mu = 1 + alpha*mu here and fail.
    mu, log_alpha = 4.0, np.log(0.5)
    y = np.arange(0, 500, dtype=float)
    logp = _reference_nb1(y, mu, log_alpha)
    # the module's own pmf must reproduce that, one observation at a time
    got = np.asarray(jnp.stack([
        nb1_loglik(jnp.asarray([v]), jnp.asarray(mu), jnp.asarray(log_alpha))[0]
        for v in y
    ]), dtype=float)
    assert np.allclose(got, logp, atol=1e-6)

    p = np.exp(logp)
    p = p / p.sum()
    mean = float((y * p).sum())
    var = float(((y - mean) ** 2 * p).sum())
    assert mean == pytest.approx(mu, abs=1e-6)
    assert var / mean == pytest.approx(1.0 + np.exp(log_alpha), rel=1e-6)


@pytest.mark.parametrize("alpha", [1e-4, 1e-6, 1e-8])
def test_nb1_nests_poisson_as_alpha_vanishes(alpha):
    y = np.array([0.0, 1.0, 5.0, 12.0])
    mu = np.array([2.0, 3.0, 7.0, 11.0])
    poisson = np.sum(y * np.log(mu) - mu - gammaln(y + 1))
    got = float(jnp.sum(nb1_loglik(jnp.asarray(y), jnp.asarray(mu),
                                   jnp.asarray(np.log(alpha)))))
    assert got == pytest.approx(poisson, abs=1e-4)


def test_nb1_matches_an_independent_implementation():
    y = np.array([0.0, 2.0, 9.0, 30.0, 120.0])
    mu = np.array([1.5, 4.0, 12.0, 40.0, 90.0])
    for log_alpha in (np.log(0.05), np.log(0.6), np.log(4.0)):
        got = np.asarray(jnp.stack([
            nb1_loglik(jnp.asarray([v]), jnp.asarray(m), jnp.asarray(log_alpha))[0]
            for v, m in zip(y, mu)
        ]), dtype=float)
        assert np.allclose(got, _reference_nb1(y, mu, log_alpha), atol=1e-6)


# ── dispatch and parameter indexing ─────────────────────────────────────────

def test_nb1_is_selectable_and_allocates_one_dispersion():
    _, spec = _build("nb1")
    assert spec.model == "nb1"
    idx = build_base_index(spec, model="nb1")
    assert "dispersion" in idx
    # intercept + two terms + dispersion
    assert idx["total_params"] == len(TERMS) + 2
    assert idx["dispersion"] == idx["total_params"] - 1


def test_historical_dispersion_flag_still_selects_nb2():
    _, spec = _build(None, dispersion=1)
    assert spec.model == "nb"
    _, spec0 = _build(None, dispersion=0)
    assert spec0.model == "poisson"


def test_explicit_model_key_overrides_the_dispersion_flag():
    # dispersion=0 would otherwise mean Poisson; the explicit key wins so a
    # caller cannot silently get the wrong family.
    _, spec = _build("nb1", dispersion=0)
    assert spec.model == "nb1"
