"""Regression tests: NB1 through the two-tier (CMF) estimator in GA_CMF_AADT_JAX.

``model='nb1'`` was wired into the flat estimator first (``main_hpc``).  The
two-tier form used by the hierarchical-CMF search keeps its own parameter
layout and its own JAX likelihoods, so the family has to be carried across
separately or a spec that asks for NB1 silently gets NB2.  These tests pin the
crossing:

  * the shared pmf agrees with the flat estimator's ``nb1_loglik`` and with an
    independent NumPy implementation, and nests the Poisson as alpha -> 0;
  * ``count_params`` and ``_param_labels`` allocate exactly one dispersion, in
    the last slot, for NB1 as for NB2;
  * ``_fixed_ll`` and ``_mixed_ll`` evaluate the NB1 likelihood rather than
    falling through to the NB2 branch.

The last one matters most: NB1 and NB2 share a parameter layout, so a
mis-dispatch is not a crash or a shape error — it is a plausible number from
the wrong variance function.
"""

import numpy as np
import pytest
from scipy.special import gammaln

from metacountregressor.GA_CMF_AADT_JAX import (
    _fixed_ll,
    _mixed_ll,
    _param_labels,
    count_params,
    generate_draws,
)
from metacountregressor.main_hpc import nb1_loglik
from metacountregressor.nb_parameterization import nb1_logpmf

Y = np.array([0.0, 1.0, 3.0, 17.0, 90.0])
MU = np.array([0.7, 2.5, 6.0, 20.0, 55.0])


def _reference_nb1(y, mu, alpha):
    r"""Independent NB1 log-pmf: ``r = mu/alpha``, ``p = 1/(1+alpha)``."""
    r = mu / alpha
    p = 1.0 / (1.0 + alpha)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(p) + y * np.log(1.0 - p))


@pytest.mark.parametrize("alpha", [0.01, 0.4, 11.0])
def test_nb1_pmf_matches_independent_reference_and_flat_estimator(alpha):
    log_alpha = float(np.log(alpha))
    got = np.asarray(nb1_logpmf(Y, np.log(MU), log_alpha), dtype=float)
    assert np.allclose(got, _reference_nb1(Y, MU, alpha), atol=1e-8)
    # The flat estimator carries its own copy of the same arithmetic.
    flat = np.asarray(nb1_loglik(Y, MU, log_alpha), dtype=float)
    assert np.allclose(got, flat, atol=1e-8)


def test_nb1_nests_poisson_as_alpha_vanishes():
    poisson = float(np.sum(Y * np.log(MU) - MU - gammaln(Y + 1)))
    got = float(np.sum(nb1_logpmf(Y, np.log(MU), np.log(1e-8))))
    assert got == pytest.approx(poisson, abs=1e-3)


def test_nb1_allocates_one_dispersion_in_the_last_slot():
    rand_b, rand_l = (False, True), (False,)
    for model in ("nb", "nb1", "poisson"):
        k = count_params(rand_b, rand_l, model)
        labels = _param_labels(["A", "B"], ["C"], rand_b, rand_l, model)
        assert len(labels) == k
    assert count_params(rand_b, rand_l, "nb1") == count_params(rand_b, rand_l, "nb")
    assert _param_labels(["A", "B"], ["C"], rand_b, rand_l, "nb1")[-1] == "log_alpha"
    assert _param_labels(["A", "B"], ["C"], rand_b, rand_l, "nb")[-1] == "log_theta"
    # Neither family adds a dispersion to the Poisson layout.
    assert count_params(rand_b, rand_l, "nb1") == count_params(rand_b, rand_l, "poisson") + 1


def _design():
    rng = np.random.default_rng(3)
    n = 40
    base = rng.uniform(0.0, 3.0, (n, 2))
    loc = rng.uniform(0.0, 2.0, (n, 1))
    y = rng.poisson(6.0, n).astype(float)
    aadt = rng.uniform(500.0, 40000.0, n)
    return y, aadt, base, loc


def test_fixed_ll_evaluates_nb1_not_nb2():
    y, aadt, base, loc = _design()
    params = np.array([0.5, 0.2, -0.1, 0.9, 0.05, np.log(4.0)])
    got = float(_fixed_ll(params, y, aadt, base, loc, (False, False),
                          (False,), "nb1", None))
    log_mu = (params[0] + base @ params[1:3]
              + (params[3] + loc[:, 0] * params[4]) * np.log(aadt))
    want = -float(np.sum(nb1_logpmf(y, log_mu, params[5])))
    assert got == pytest.approx(want, rel=1e-10)

    nb2 = float(_fixed_ll(params, y, aadt, base, loc, (False, False),
                          (False,), "nb", None))
    assert abs(nb2 - got) > 1e-3, "nb1 and nb2 must not evaluate the same function"


def test_mixed_ll_uses_the_nb1_pmf_under_draws():
    # One random baseline term, so the simulation branch actually runs.  The
    # expected value is built by hand from the NB1 pmf; a fall-through to the
    # NB2 branch would still return a finite number here, which is the point.
    y, aadt, base, loc = _design()
    # alpha0, alpha[A], sigma_alpha[A], alpha[B], beta0, beta[C], log_alpha
    params = np.array([0.5, 0.2, 0.3, -0.1, 0.9, 0.05, np.log(4.0)])
    rand_b, rand_l = (True, False), (False,)
    R = 24
    draws = generate_draws(len(y), R, 1, use_halton=True, seed=7)
    assert draws is not None

    u = np.asarray(draws)[:, :, 0]                       # (R, N)
    x0 = base[:, 0][None, :]                             # (1, N)
    log_a = (params[0] + (params[1] + params[2] * u) * x0
             + params[3] * base[:, 1][None, :])
    b_exp = params[4] + params[5] * loc[:, 0]
    log_mu = log_a + b_exp[None, :] * np.log(aadt)[None, :]
    log_pmf = np.asarray(nb1_logpmf(y[None, :], log_mu, params[6]), dtype=float)
    want = -float(np.sum(
        np.logaddexp.reduce(log_pmf, axis=0) - np.log(R)))

    got = float(_mixed_ll(params, y, aadt, base, loc, rand_b, rand_l,
                          draws, "nb1", None, None))
    assert got == pytest.approx(want, rel=1e-9)

    nb2 = float(_mixed_ll(params, y, aadt, base, loc, rand_b, rand_l,
                          draws, "nb", None, None))
    assert abs(nb2 - got) > 1e-3, "nb1 and nb2 must not share the mixing likelihood"


def test_unknown_family_raises_rather_than_defaulting():
    y, aadt, base, loc = _design()
    params = np.array([0.5, 0.2, -0.1, 0.9, 0.05, np.log(4.0)])
    with pytest.raises(ValueError, match="unknown count family"):
        _fixed_ll(params, y, aadt, base, loc, (False, False), (False,),
                  "nb3", None)
