"""Random parameters in the CMF form: mean vs median, spread and identification.

The hierarchical CMF is

    CMF_i = exp(alpha0 + sum alpha_k x_k) * AADT_i ** (beta0 + sum beta_k z_k)

with a random parameter entering as ``alpha_k + sigma_k u_k`` (upper tier) or
``beta_k + sigma_k v_k`` (lower tier).  Because the coefficient is random, the
CMF is a distribution, and the number that exp(alpha) gives is its **median**,
not its mean.  The mean is exp(alpha) * E[exp(sigma U)] - the lognormal
correction - which is what an aggregate before/after or benefit-cost
calculation needs.  These tests pin that distinction, the distribution-aware
share of sites with a protective effect, and the convergence/identification
report the fit returns.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from metacountregressor import GA_CMF_AADT_JAX as G
from metacountregressor.cmf_agds_search import (
    boundary_lrt_pvalue,
    refit_and_report,
    sigma_boundary_lrt,
)
from metacountregressor.cmf_catalogue import (
    build_cmf_catalogue,
    mixing_cdf,
    mixing_mgf,
)


# ── the mixing-distribution expectations ────────────────────────────────────

def test_normal_mgf_is_the_lognormal_correction():
    for s in (0.05, 0.3, 1.0):
        assert mixing_mgf(s, "normal") == pytest.approx(np.exp(s * s / 2))
    assert mixing_mgf(0.0, "normal") == 1.0


def test_bounded_mgf_matches_closed_forms():
    s = 0.3
    a = np.sqrt(3.0)
    assert mixing_mgf(s, "uniform") == pytest.approx(
        np.sinh(a * s) / (a * s), rel=1e-9)
    b = np.sqrt(6.0)
    assert mixing_mgf(s, "triangular") == pytest.approx(
        4.0 * np.sinh(b * s / 2.0) ** 2 / (b * b * s * s), rel=1e-6)
    # Bounded support: the correction is below the normal one.
    assert mixing_mgf(s, "uniform") < mixing_mgf(s, "normal")
    assert mixing_mgf(s, "triangular") < mixing_mgf(s, "normal")


def test_lognormal_mean_is_undefined():
    # The MGF of a lognormal diverges, so there is no finite mean CMF.  A
    # numerical expectation would just be an artefact of truncating the tail.
    assert np.isnan(mixing_mgf(0.3, "lognormal"))


def test_mixing_cdf_is_distribution_aware():
    for d in ("normal", "uniform", "triangular"):
        assert mixing_cdf(0.0, d) == pytest.approx(0.5, abs=1e-9)
        assert mixing_cdf(-10.0, d) == pytest.approx(0.0, abs=1e-12)
        assert mixing_cdf(10.0, d) == pytest.approx(1.0, abs=1e-12)
    # The standardised lognormal is right-skewed, so its median is negative and
    # more than half the mass sits below zero.
    assert mixing_cdf(0.0, "lognormal") > 0.5
    assert 0.0 < mixing_cdf(0.0, "lognormal") < 1.0


# ── the catalogue carries both numbers ──────────────────────────────────────

def _data(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"AADT": np.linspace(1000, 20000, n),
                         "FREQ": rng.poisson(3.0, n).astype(float),
                         "X": rng.normal(size=n),
                         "Z": rng.normal(size=n)})


def _cat(sigma, dist="normal", tier="upper"):
    """One random term (upper X or lower Z) with a prescribed sigma/dist."""
    df = _data()
    upper = [True] if tier == "upper" else [False]
    lb = ["X"]
    sl = ["Z"] if tier == "lower" else []
    lr = [tier == "lower"]
    labels = G._param_labels(lb, sl, upper, lr, "poisson")
    params = np.zeros(len(labels))
    se = np.full(len(labels), 0.05)
    params[labels.index("alpha[X]")] = 0.3
    params[labels.index("beta0")] = 0.9
    if tier == "lower":
        params[labels.index("beta[Z]")] = -0.4
    params[labels.index("sigma_alpha[X]" if tier == "upper"
                        else "sigma_beta[Z]")] = sigma
    cat = build_cmf_catalogue(params, se, df, lb, sl, upper, lr,
                              dists=[dist] if sigma is not None else None)
    return cat[cat["tier"] == tier].iloc[0]


def test_upper_mean_cmf_is_median_times_mgf():
    row = _cat(0.3)
    assert row["tier"] == "upper"
    median = np.exp(0.3)
    assert row["cmf"] == pytest.approx(median, rel=1e-12)
    assert row["cmf_median"] == pytest.approx(median, rel=1e-12)
    assert row["cmf_mean"] == pytest.approx(median * np.exp(0.3 ** 2 / 2))
    assert row["cmf_mean"] > row["cmf"]
    assert row["mixing"] == "normal"
    assert row["share_protective"] == pytest.approx(norm.cdf(-0.3 / 0.3))


def test_lower_mean_cmf_uses_the_doubling_basis():
    # Lower-tier CMF is per AADT doubling: 2**(beta + sigma*v), so the mean is
    # 2**beta * E[exp(sigma*ln2*v)].
    row = _cat(0.25, tier="lower")
    assert row["tier"] == "lower"
    median = 2.0 ** -0.4
    assert row["cmf"] == pytest.approx(median, rel=1e-12)
    assert row["cmf_mean"] == pytest.approx(
        median * np.exp((0.25 * np.log(2.0)) ** 2 / 2))
    assert row["share_protective"] == pytest.approx(
        norm.cdf(0.4 / 0.25))          # beta < 0, protective at most sites


def test_mixing_distribution_reaches_mean_and_share():
    normal = _cat(0.3, dist="normal")
    uniform = _cat(0.3, dist="uniform")
    assert uniform["mixing"] == "uniform"
    assert uniform["cmf_mean"] < normal["cmf_mean"]
    assert uniform["share_protective"] != pytest.approx(
        normal["share_protective"])

    lognormal = _cat(0.3, dist="lognormal")
    assert np.isnan(lognormal["cmf_mean"])       # no finite mean
    assert np.isfinite(lognormal["cmf"])         # median still defined
    assert 0.0 < lognormal["share_protective"] < 1.0


def test_fixed_terms_have_no_mean_cmf():
    df = _data()
    labels = G._param_labels(["X"], ["Z"], [False], [False], "poisson")
    cat = build_cmf_catalogue(np.zeros(len(labels)), np.full(len(labels), 0.05),
                              df, ["X"], ["Z"], [False], [False])
    assert cat["cmf_mean"].isna().all()
    assert (cat["mixing"] == "").all()


# ── the fit reports whether it actually converged ───────────────────────────

def test_boundary_lrt_pvalue_is_the_mixture():
    # 50:50 mixture of chi2_0 and chi2_1: half the naive chi2_1 tail.
    assert boundary_lrt_pvalue(0.0) == 0.5
    assert boundary_lrt_pvalue(-1.0) == 0.5
    assert boundary_lrt_pvalue(np.nan) == 0.5
    assert boundary_lrt_pvalue(3.841458820694124) == pytest.approx(0.025, rel=1e-6)
    assert boundary_lrt_pvalue(2.705543454095404) == pytest.approx(0.05, rel=1e-6)
    assert boundary_lrt_pvalue(10.0) < boundary_lrt_pvalue(5.0)


def test_boundary_lrt_separates_random_from_fixed_term():
    # One genuinely random term (sigma 0.4) and one that is only flagged random:
    # the formal test should reject sigma = 0 for the first and not the second.
    rng = np.random.default_rng(5)
    n = 600
    aadt = rng.uniform(1000, 20000, n)
    x, w = rng.normal(size=(2, n))
    eta = (-8.0 + (0.30 + 0.40 * rng.normal(size=n)) * x
           + 0.20 * w + 0.90 * np.log(aadt))
    df = pd.DataFrame({"FREQ": rng.poisson(np.exp(eta)).astype(float),
                       "AADT": aadt, "X": x, "W": w, "LENGTH": 1.0})
    assert df["FREQ"].sum() > 50       # enough information to test sigma
    spec = {"upper": ["X", "W"], "lower": [],
            "rand_upper": [True, True], "rand_lower": [],
            "dist_upper": ["normal", "normal"], "dist_lower": [],
            "model": "poisson", "rho": 0.0}
    full = refit_and_report(df, spec, R=100)
    tests = sigma_boundary_lrt(df, spec, R=100, ll_full=full.ll,
                               fit=full.fit_parts)
    assert set(tests) == {("upper", "X"), ("upper", "W")}
    assert all(t["common_draws"] for t in tests.values())
    assert tests[("upper", "X")]["lr"] > 0
    assert tests[("upper", "X")]["p"] < 0.05
    # Nested models on the same draws: the restricted fit cannot beat the full
    # one, so the LR stays non-negative rather than carrying simulation noise.
    assert tests[("upper", "W")]["lr"] >= -1e-6
    assert tests[("upper", "W")]["p"] > 0.2
    assert tests[("upper", "W")]["lr"] < tests[("upper", "X")]["lr"]


def test_refit_and_report_attaches_lrt_columns():
    rng = np.random.default_rng(9)
    n = 400
    aadt = rng.uniform(1000, 20000, n)
    x = rng.normal(size=n)
    eta = -8.0 + (0.3 + 0.4 * rng.normal(size=n)) * x + 0.9 * np.log(aadt)
    df = pd.DataFrame({"FREQ": rng.poisson(np.exp(eta)).astype(float),
                       "AADT": aadt, "X": x, "LENGTH": 1.0})
    spec = {"upper": ["X"], "lower": [], "rand_upper": [True],
            "rand_lower": [], "dist_upper": ["normal"], "dist_lower": [],
            "model": "poisson", "rho": 0.0}
    rep = refit_and_report(df, spec, R=100, lrt=True)
    row = rep.catalogue[rep.catalogue.term == "X"].iloc[0]
    assert np.isfinite(row["lrt"]) and 0 <= row["lrt_p"] <= 0.5
    assert ("upper", "X") in rep.lrt
    assert rep.lrt[("upper", "X")]["common_draws"] is True
    assert set(rep.fit_parts) >= {"y", "A", "B", "L", "draws", "params",
                                  "offset"}


def test_identification_report_reports_stationarity():
    rng = np.random.default_rng(3)
    n = 700
    aadt = rng.uniform(1000, 20000, n)
    x, z = rng.normal(size=(2, n))
    mu = np.exp(-8.0 + (0.3 + 0.4 * rng.normal(size=n)) * x
                + (0.9 + 0.15 * rng.normal(size=n)) * z * np.log(aadt))
    df = pd.DataFrame({"FREQ": rng.poisson(mu).astype(float),
                       "AADT": aadt, "X": x, "Z": z})
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        df, ["X"], ["Z"], [True], [True], model="poisson", R=200)
    rep = G.identification_report(res, y, A, B, L, ["X"], ["Z"], [True],
                                  [True], draws, "poisson", off)
    assert rep["converged"] is True
    assert np.isfinite(rep["gradient_inf_norm"])
    assert rep["gradient_inf_norm"] < 1e-3
    assert rep["hessian_positive_definite"] is True
    assert set(rep) >= {"gradient_inf_norm", "converged",
                        "hessian_positive_definite", "identified"}
