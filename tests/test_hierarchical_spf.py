"""Tests for the Chen & Persaud (2014) hierarchical SPF module.

The paper's Toronto/Edmonton data are not public, so the models are checked
by simulating data with the paper's design and published estimates
(Table 9, Edmonton hierarchical model) and recovering them, and by checking
the likelihood structure of the random-parameter extensions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

from metacountregressor.hierarchical_spf import (
    HierarchicalSPF, HierarchicalSPFSpec, fit_hierarchical_spf)

T9 = {"ln_alpha0": -10.6091, "alpha[LT]": -0.04553, "alpha[RT]": 0.104,
      "b0[AADT_MAJ]": 0.9238, "beta[AADT_MAJ|AREA]": -0.1324,
      "b0[AADT_MIN]": 0.3399, "beta[AADT_MIN|AREA]": 0.1881,
      "dispersion_alpha": 0.5236}


def _edmonton_like(seed=3, n_sites=499, years=6):
    rng = np.random.default_rng(seed)

    def lognorm(mean, sd, n, lo, hi):
        s2 = np.log(1 + (sd / mean) ** 2)
        return np.clip(rng.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), n), lo, hi)

    maj = lognorm(24674, 10849, n_sites, 4720, 70331)
    mnr = lognorm(9634, 7004, n_sites, 102, 34926)
    p_lt = np.array([93, 74, 158, 45, 129], float)
    p_rt = np.array([167, 95, 138, 20, 79], float)
    lt = rng.choice(5, n_sites, p=p_lt / p_lt.sum())
    rt = rng.choice(5, n_sites, p=p_rt / p_rt.sum())
    area = (rng.random(n_sites) < 251 / 515).astype(float)
    i = np.repeat(np.arange(n_sites), years)
    df = pd.DataFrame(dict(SITE=i, AADT_MAJ=maj[i], AADT_MIN=mnr[i],
                           LT=lt[i].astype(float), RT=rt[i].astype(float), AREA=area[i]))
    lmu = (T9["ln_alpha0"] + T9["alpha[LT]"] * df.LT + T9["alpha[RT]"] * df.RT
           + T9["b0[AADT_MAJ]"] * np.exp(T9["beta[AADT_MAJ|AREA]"] * df.AREA) * np.log(df.AADT_MAJ)
           + T9["b0[AADT_MIN]"] * np.exp(T9["beta[AADT_MIN|AREA]"] * df.AREA) * np.log(df.AADT_MIN))
    a = T9["dispersion_alpha"]
    df["Y"] = rng.poisson(rng.gamma(1 / a, a * np.exp(lmu))).astype(float)
    return df


def _spec(**kw):
    base = dict(aadt_cols=["AADT_MAJ", "AADT_MIN"], a_terms=["LT", "RT"],
                b_terms={"AADT_MAJ": ["AREA"], "AADT_MIN": ["AREA"]},
                exponent_form="exp", site_col="SITE")
    base.update(kw)
    return HierarchicalSPFSpec(**base)


def test_recovers_paper_table9_structure():
    df = _edmonton_like()
    fit = fit_hierarchical_spf(df, "Y", _spec())
    s = fit.summary().set_index("parameter")
    assert s.attrs["converged"] and s.attrs["max_abs_grad"] < 1e-4
    for name, truth in T9.items():
        est, se = s.loc[name, "estimate"], s.loc[name, "std_err"]
        assert abs(est - truth) < 3.5 * se, (name, est, truth, se)


def test_nb2_likelihood_matches_closed_form():
    df = _edmonton_like(n_sites=60)
    m = HierarchicalSPF(_spec())
    m._prepare(df, "Y")
    p = np.array([T9["ln_alpha0"], T9["alpha[LT]"], T9["alpha[RT]"],
                  np.log(T9["b0[AADT_MAJ]"]), T9["beta[AADT_MAJ|AREA]"],
                  np.log(T9["b0[AADT_MIN]"]), T9["beta[AADT_MIN|AREA]"],
                  np.log(T9["dispersion_alpha"])])
    lmu = (p[0] + p[1] * df.LT + p[2] * df.RT
           + np.exp(p[3] + p[4] * df.AREA) * np.log(df.AADT_MAJ)
           + np.exp(p[5] + p[6] * df.AREA) * np.log(df.AADT_MIN)).to_numpy()
    mu, a, y = np.exp(lmu), T9["dispersion_alpha"], df.Y.to_numpy()
    ll = (gammaln(y + 1 / a) - gammaln(1 / a) - gammaln(y + 1)
          + (1 / a) * np.log(1 / (1 + a * mu)) + y * np.log(a * mu / (1 + a * mu))).sum()
    assert -float(m.negloglik(p)) == pytest.approx(ll, rel=1e-10)


def test_random_parameter_with_tiny_sd_matches_fixed_model():
    df = _edmonton_like(n_sites=80)
    df["DIST"] = df.SITE % 10
    fixed = HierarchicalSPF(_spec())
    fixed._prepare(df, "Y")
    p_fixed = np.array([T9["ln_alpha0"], T9["alpha[LT]"], T9["alpha[RT]"],
                        np.log(T9["b0[AADT_MAJ]"]), T9["beta[AADT_MAJ|AREA]"],
                        np.log(T9["b0[AADT_MIN]"]), T9["beta[AADT_MIN|AREA]"],
                        np.log(T9["dispersion_alpha"])])
    ll_fixed = float(fixed.negloglik(p_fixed))
    for rand, extra in (({"alpha[LT]": "site"}, 1),
                        ({"b0[AADT_MAJ]": "group"}, 1),
                        ({"b0[AADT_MAJ]": "both"}, 2)):
        m = HierarchicalSPF(_spec(random=rand, group_col="DIST"), R=20, inner_draws=10)
        m._prepare(df, "Y")
        p = np.concatenate([p_fixed[:-1], np.full(extra, np.log(1e-8)), p_fixed[-1:]])
        assert float(m.negloglik(p)) == pytest.approx(ll_fixed, rel=1e-8), rand


def test_site_random_parameters_are_constant_within_a_site():
    """Panel structure: a site's years share one draw, so duplicating a
    site's rows changes the likelihood differently from adding a new site."""
    df = _edmonton_like(n_sites=40)
    m = HierarchicalSPF(_spec(random={"alpha[LT]": "site"}), R=30)
    m._prepare(df, "Y")
    # integrand for each site is a product over its years inside the
    # simulation average -> per-site likelihood != product of per-row ones
    p = np.concatenate([[T9["ln_alpha0"], T9["alpha[LT]"], T9["alpha[RT]"],
                         np.log(T9["b0[AADT_MAJ]"]), T9["beta[AADT_MAJ|AREA]"],
                         np.log(T9["b0[AADT_MIN]"]), T9["beta[AADT_MIN|AREA]"]],
                        [np.log(0.5)], [np.log(T9["dispersion_alpha"])]])
    ll_panel = float(m.negloglik(p))
    m_rows = HierarchicalSPF(_spec(random={"alpha[LT]": "site"}, site_col=None), R=30)
    m_rows._prepare(df, "Y")
    assert not np.isclose(ll_panel, float(m_rows.negloglik(p)), rtol=1e-6)


def test_spec_validation():
    df = _edmonton_like(n_sites=20)
    with pytest.raises(ValueError, match="not a model parameter"):
        HierarchicalSPF(_spec(random={"alpha[XX]": "site"}))._prepare(df, "Y")
    with pytest.raises(ValueError, match="group_col"):
        HierarchicalSPF(_spec(random={"alpha[LT]": "group"}))._prepare(df, "Y")
    df2 = df.copy()
    df2.loc[0, "AADT_MAJ"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        HierarchicalSPF(_spec())._prepare(df2, "Y")


def _one_aadt(n_sites=300, years=5, seed=11):
    rng = np.random.default_rng(seed)
    aadt = np.exp(rng.normal(8.8, 0.7, n_sites))
    lanes = rng.integers(2, 5, n_sites).astype(float)
    urb = (rng.random(n_sites) < 0.5).astype(float)
    ln_len = np.log(rng.uniform(0.2, 2.0, n_sites))
    s = np.repeat(np.arange(n_sites), years)
    df = pd.DataFrame(dict(SITE=s, DIST=s % 15, AADT=aadt[s], LANES=lanes[s],
                           URBAN=urb[s], LNLEN=ln_len[s]))
    lmu = -6.0 - 0.15 * df.LANES + 0.75 * np.exp(0.12 * df.URBAN) * np.log(df.AADT) + df.LNLEN
    df["Y"] = rng.poisson(rng.gamma(1 / 0.3, 0.3 * np.exp(lmu))).astype(float)
    return df


@pytest.mark.parametrize("random,correlated", [
    ({}, False),
    ({"alpha[LANES]": "site", "b0[AADT]": "site"}, True),
    ({"b0[AADT]": "group"}, False),
    ({"b0[AADT]": "both", "alpha[LANES]": "site"}, False),
])
def test_single_aadt_supports_every_flavour(random, correlated):
    df = _one_aadt()
    spec = HierarchicalSPFSpec(aadt_cols=["AADT"], a_terms=["LANES"],
                               b_terms={"AADT": ["URBAN"]}, exponent_form="exp",
                               site_col="SITE", group_col="DIST", offset_col="LNLEN",
                               random=random, correlated=correlated)
    fit = fit_hierarchical_spf(df, "Y", spec, R=40, inner_draws=16, maxiter=300)
    s = fit.summary().set_index("parameter")
    assert np.isfinite(s.attrs["loglik"])
    assert s.loc["b0[AADT]", "estimate"] == pytest.approx(0.75, abs=0.15)
    assert s.loc["alpha[LANES]", "estimate"] == pytest.approx(-0.15, abs=0.1)
    if correlated:
        assert fit.site_covariance().shape == (2, 2)
