"""Regression tests: when does a random-parameter sigma count as zero?

A selected random term is only evidence of heterogeneity if its SD can be
separated from zero.  ``sigma_is_flat`` decides that with two criteria:

  * absolute - ``|sigma| < SIGMA_FLAT_TOL`` (the mixing distribution collapsed
    onto the mean), and
  * relative - ``sigma / SE(sigma) < SIGMA_FLAT_Z`` (the point estimate is not
    distinguishable from zero given its own precision).

The engine's identification report and the CMF catalogue must agree, so these
tests pin the rule itself, the cases with no usable standard error, and the
catalogue columns that carry the verdict.
"""

import numpy as np
import pandas as pd
import pytest

from metacountregressor import GA_CMF_AADT_JAX as G
from metacountregressor.cmf_catalogue import build_cmf_catalogue


# ── the rule on its own ─────────────────────────────────────────────────────

def test_absolute_tolerance_flags_collapsed_sigma():
    # A sigma on the boundary is flat whatever its precision, including when no
    # standard error is supplied at all.
    assert G.sigma_is_flat(0.0, None)
    assert G.sigma_is_flat(3e-4, 1e-3)          # 0.3 SE but below the floor
    assert G.sigma_is_flat(G.SIGMA_FLAT_TOL / 2, 1e-12)
    assert G.sigma_is_flat(-3e-4, 1e-3)         # magnitude, not sign


def test_standard_error_rule_flags_imprecise_sigma():
    # Point estimate far above the absolute floor, but the same size as its own
    # SE: zero is not ruled out, so there is no identified heterogeneity.
    assert G.sigma_is_flat(0.09, 0.09)
    assert G.sigma_is_flat(0.09, 0.05)          # 1.8 SE, inside 1.96
    assert G.sigma_is_flat(0.09, 0.12, z=1.0)   # 0.75 SE at a 68 % cutoff


def test_standard_error_rule_keeps_identified_sigma():
    # Washington's URB: 0.0864 (0.0113), 7.6 SE from zero - real heterogeneity.
    assert not G.sigma_is_flat(0.0864, 0.0113)
    assert not G.sigma_is_flat(0.5, 0.1)
    assert not G.sigma_is_flat(1.0, 1e-6)


def test_standard_error_boundary_is_strict():
    se = 0.01
    at_limit = G.SIGMA_FLAT_Z * se
    assert not G.sigma_is_flat(at_limit, se)
    assert G.sigma_is_flat(at_limit * (1 - 1e-9), se)
    # A shorter z makes the test stricter about precision.
    assert G.sigma_is_flat(1.5 * se, se, z=2.5)
    assert not G.sigma_is_flat(1.5 * se, se, z=1.0)


@pytest.mark.parametrize("se", [None, np.nan, np.inf, 0.0, -0.01])
def test_unusable_standard_error_falls_back_to_absolute_rule(se):
    # Without a usable SE the relative test cannot fire, but the absolute test
    # still applies: an identified-looking sigma must not be flagged, and a
    # collapsed one must not slip through.
    assert not G.sigma_is_flat(0.2, se)
    assert G.sigma_is_flat(1e-4, se)


def test_non_finite_sigma_is_flat():
    assert G.sigma_is_flat(np.nan, 0.01)
    assert G.sigma_is_flat(np.inf, 0.01)


# ── the catalogue columns that carry the verdict ────────────────────────────

def _data(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"AADT": np.linspace(1000, 20000, n),
                         "FREQ": rng.poisson(3.0, n).astype(float),
                         "X": rng.normal(size=n),
                         "Z": rng.normal(size=n)})


def _catalogue(sigma, sigma_se, z=1.96):
    """Upper-tier random X with a prescribed sigma and sigma SE."""
    df = _data()
    labels = G._param_labels(["X"], ["Z"], [True], [False], "poisson")
    params = np.zeros(len(labels))
    se = np.zeros(len(labels))
    params[labels.index("alpha[X]")] = 0.3
    se[labels.index("alpha[X]")] = 0.05
    params[labels.index("beta0")] = 0.9
    se[labels.index("beta0")] = 0.02
    se[labels.index("beta[Z]")] = 0.03
    params[labels.index("sigma_alpha[X]")] = sigma
    se[labels.index("sigma_alpha[X]")] = sigma_se
    return build_cmf_catalogue(params, se, df, ["X"], ["Z"], [True], [False],
                               region="test", z=z)


def test_catalogue_matches_rule_for_each_regime():
    # Identified: the mean and the spread are both reported.
    row = _catalogue(0.0864, 0.0113).iloc[0]
    assert not row["sigma_collapsed"] and not row["sigma_flat_abs"]
    assert row["sigma_z"] == pytest.approx(0.0864 / 0.0113)
    assert np.isfinite(row["share_protective"])
    assert row["cmf_spread_lo"] < row["cmf"] < row["cmf_spread_hi"]

    # Imprecise: absolute floor not tripped, but sigma is only ~1 SE from zero.
    row = _catalogue(0.09, 0.09).iloc[0]
    assert row["sigma_collapsed"] and not row["sigma_flat_abs"]
    assert np.isnan(row["share_protective"])

    # Collapsed: both criteria fire.
    row = _catalogue(3e-4, 0.03).iloc[0]
    assert row["sigma_collapsed"] and row["sigma_flat_abs"]
    assert np.isnan(row["share_protective"])


def test_catalogue_flag_agrees_with_sigma_is_flat():
    for sigma, sigma_se in [(0.0864, 0.0113), (0.09, 0.09), (0.0003, 0.03),
                            (0.2, 0.01), (0.09, np.nan), (0.0005, np.nan)]:
        row = _catalogue(sigma, sigma_se).iloc[0]
        assert row["sigma_collapsed"] == G.sigma_is_flat(sigma, sigma_se), (
            sigma, sigma_se)
        # A flagged term reports no spread claim at all.
        if row["sigma_collapsed"]:
            assert np.isnan(row["share_protective"])


def test_catalogue_passes_its_z_through():
    # sigma at 1.5 SE is flagged at the default 1.96 but not at z = 1.
    assert _catalogue(0.15, 0.10).iloc[0]["sigma_collapsed"]
    assert not _catalogue(0.15, 0.10, z=1.0).iloc[0]["sigma_collapsed"]


def test_fixed_terms_never_flagged():
    df = _data()
    labels = G._param_labels(["X"], ["Z"], [False], [False], "poisson")
    params = np.zeros(len(labels))
    se = np.full(len(labels), 0.05)
    cat = build_cmf_catalogue(params, se, df, ["X"], ["Z"], [False], [False])
    assert not cat["sigma_collapsed"].any()
    assert np.isnan(cat["sigma_z"]).all()


# ── the two consumers must not disagree ─────────────────────────────────────

def test_identification_report_uses_the_same_rule():
    rng = np.random.default_rng(1)
    n = 800
    aadt = rng.uniform(1000, 20000, n)
    x, z = rng.normal(size=(2, n))
    mu = np.exp(-8.0 + 0.3 * x + (0.9 + 0.2 * z) * np.log(aadt))
    df = pd.DataFrame({"FREQ": rng.poisson(mu).astype(float),
                       "AADT": aadt, "X": x, "Z": z})
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        df, ["X"], ["Z"], [True], [False], model="poisson", R=50)
    rep = G.identification_report(res, y, A, B, L, ["X"], ["Z"], [True],
                                  [False], draws, "poisson", off)
    labels = G._param_labels(["X"], ["Z"], [True], [False], "poisson")
    expected = [l for l in labels if l.startswith("sigma_")
                and G.sigma_is_flat(res.params[labels.index(l)],
                                    rep["sigma_se"].get(l))]
    assert rep["flat_sigmas"] == expected
    assert set(rep["sigma_se"]) == {l for l in labels if l.startswith("sigma_")}
