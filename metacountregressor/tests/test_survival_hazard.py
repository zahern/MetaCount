"""Weibull proportional-hazards support: likelihood, estimation, reporting.

The PH likelihood is checked against brute-force numerical differentiation of
S(t) -- the highest-risk part, since a plausible-looking slip (using log H
instead of log f for events) still returns plausible numbers.

Estimation is checked two ways that do not share code with the
implementation:
  * an independent Cox Breslow partial-likelihood fit (different formula), and
  * recovery of parameters on data simulated from known truth.
Finally PH and AFT are cross-checked against each other, since the two are
reparameterisations of the same Weibull family:
    k_PH = 1 / sigma_AFT        beta_PH = -beta_AFT / sigma_AFT
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from metacountregressor.survival_models import (
    AFTFitter,
    AFT_FAMILIES,
    PH_FAMILIES,
    _ph_weibull_ll,
)


def _simulate(n=800, seed=11, beta=(0.55, -0.40), shape=1.40, alpha=-1.20, censor=6.0):
    """Right-censored Weibull-PH data by inverse-CDF sampling.

    H(t) = exp(alpha) t**shape exp(x'b) = E with E ~ Exp(1),
    so t = exp((log E - alpha - x'b) / shape).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lp = beta[0] * x1 + beta[1] * x2
    e = rng.exponential(size=n)
    t = np.exp((np.log(e) - alpha - lp) / shape)
    event = (t <= censor).astype(int)
    duration = np.minimum(t, censor)
    df = pd.DataFrame({"dur": duration, "evt": event, "x1": x1, "x2": x2})
    return df, np.array(beta), shape, event


# --------------------------------------------------------------------------
# likelihood
# --------------------------------------------------------------------------

def test_ph_likelihood_matches_numeric_derivative_of_survival():
    """log f must be built as log h + log S, not log H."""
    rng = np.random.default_rng(3)
    n, shape_raw = 7, 0.7
    y = rng.uniform(0.5, 20.0, (n, 1, 1))
    event = (rng.uniform(size=(n, 1, 1)) > 0.35).astype(float)
    eta = rng.normal(0, 1.5, (n, 1, 1))
    k = float(np.log1p(np.exp(shape_raw)))

    def H(t, e):
        return np.exp(e) * np.power(np.maximum(t, 1e-12), k)

    def S(t, e):
        return np.exp(-H(t, e))

    eps = 1e-6
    ref = 0.0
    for i in range(n):
        ti, ei, ev = float(y[i, 0, 0]), float(eta[i, 0, 0]), event[i, 0, 0]
        if ev > 0:
            f_i = (S(ti - eps, ei) - S(ti + eps, ei)) / (2 * eps)   # f = -dS/dt
            ref += np.log(max(f_i, 1e-300))
        else:
            ref += np.log(S(ti, ei))

    got = float(np.sum(np.array(_ph_weibull_ll(y, event, eta, shape_raw))))
    assert got == pytest.approx(ref, abs=1e-6)


def test_ph_censored_branch_is_minus_cumulative_hazard():
    y = np.array([[2.0], [2.0]])
    event = np.array([[0.0], [0.0]])          # fully censored
    eta = np.array([[0.3], [-0.4]])
    shape_raw = 0.0                            # k = softplus(0) = log 2
    k = float(np.log1p(np.exp(shape_raw)))
    got = np.array(_ph_weibull_ll(y, event, eta, shape_raw)).ravel()
    expected = -np.exp(eta.ravel() + k * np.log(2.0))
    assert got == pytest.approx(expected, rel=1e-6)


def test_family_sets_are_disjoint():
    assert PH_FAMILIES and AFT_FAMILIES
    assert not (PH_FAMILIES & AFT_FAMILIES)
    assert "weibull_ph" in PH_FAMILIES
    assert "weibull" in AFT_FAMILIES


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------

def _cox_breeslow(df, cols=("x1", "x2")):
    """Independent Cox partial-likelihood MLE (Breslow ties)."""
    x = df[list(cols)].to_numpy()
    dur = df["dur"].to_numpy()
    ev = df["evt"].to_numpy()
    risk_order = np.argsort(dur)[::-1]
    cum_eta_sorted_x = x[risk_order]
    pos = {v: i for i, v in enumerate(risk_order)}
    idx_ev = np.where(ev == 1)[0]

    def neg(b):
        eta = x @ b
        cum = np.cumsum(np.exp(eta[risk_order]))
        ll = 0.0
        for i in idx_ev:
            j = np.searchsorted(-dur[risk_order], -dur[i], side="left")
            ll += eta[i] - np.log(cum[j])
        return -ll

    return minimize(neg, np.zeros(len(cols)), method="BFGS",
                    options={"gtol": 1e-10, "maxiter": 2000}).x


def test_ph_recovers_simulated_parameters():
    df, beta_true, shape_true, _ = _simulate()
    fit = AFTFitter(family="weibull_ph", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    est = fit.summary_.loc[["x1", "x2"], "coef"].to_numpy()
    assert est == pytest.approx(beta_true, abs=0.12)
    k_hat = float(np.log1p(np.exp(fit.summary_.loc["sigma", "coef"])))
    assert k_hat == pytest.approx(shape_true, abs=0.15)


def test_ph_agrees_with_independent_cox_partial_likelihood():
    df, _, _, _ = _simulate()
    fit = AFTFitter(family="weibull_ph", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    est = fit.summary_.loc[["x1", "x2"], "coef"].to_numpy()
    assert est == pytest.approx(_cox_breeslow(df), abs=5e-3)


def test_ph_and_aft_are_consistent_reparameterisations():
    df, _, _, _ = _simulate()
    ph = AFTFitter(family="weibull_ph", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    aft = AFTFitter(family="weibull", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    sigma = float(np.log1p(np.exp(aft.summary_.loc["sigma", "coef"])))
    k_ph = float(np.log1p(np.exp(ph.summary_.loc["sigma", "coef"])))
    assert k_ph == pytest.approx(1.0 / sigma, abs=5e-3)
    b_aft = aft.summary_.loc[["x1", "x2"], "coef"].to_numpy()
    b_ph = ph.summary_.loc[["x1", "x2"], "coef"].to_numpy()
    assert b_ph == pytest.approx(-b_aft / sigma, abs=8e-3)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def test_ph_reports_hazard_ratio_and_aft_reports_time_ratio():
    df, _, _, _ = _simulate()
    ph = AFTFitter(family="weibull_ph", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    aft = AFTFitter(family="weibull", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    assert "hazard_ratio" in ph.summary_.columns
    assert "time_ratio" in aft.summary_.columns
    b = ph.summary_.loc["x1", "coef"]
    assert ph.summary_.loc["x1", "hazard_ratio"] == pytest.approx(np.exp(b), rel=1e-9)
    # the shape parameter is not a covariate effect and gets no ratio
    assert np.isnan(ph.summary_.loc["sigma", "hazard_ratio"])


def test_survival_time_predictors_are_blocked_under_ph():
    df, _, _, _ = _simulate(n=200)
    ph = AFTFitter(family="weibull_ph", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    with pytest.raises(NotImplementedError):
        ph._model.predict_median()
    with pytest.raises(NotImplementedError):
        ph._model.predict_mean()
    hz = ph._model.predict_hazard(1.0)
    assert hz.shape == (200,)


def test_hazard_predictor_blocked_under_aft():
    df, _, _, _ = _simulate(n=200)
    aft = AFTFitter(family="weibull", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    with pytest.raises(NotImplementedError):
        aft._model.predict_hazard(1.0)
    # AFT survival-time predictors still work
    assert aft._model.predict_median().shape == (200,)


def test_unknown_family_still_raises():
    with pytest.raises(ValueError):
        AFTFitter(family="not_a_family")
