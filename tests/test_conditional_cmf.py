"""The helper-conditional CMFunction: a structural random term translated.

A random coefficient whose mean and SD move with observed covariates is not one
CMF.  ``cmf_catalogue.HelperConditionalCMF`` turns it into

    upper: CMF(w)      = exp((coef + sum delta_z w_z) * dx)
    lower: CMF(AADT,w) = AADT ** ((coef + sum delta_z w_z) * dz)

with sigma(w) = exp(logsig + sum omega_z w_z) driving the site spread, and a
delta-method interval from the covariance of [coef, delta].  These tests pin the
reduction to the ordinary catalogue CMF, the helper dependence, the sigma shift,
the interval, and the mapping out of a fitted hierarchical parameter vector.
"""

import numpy as np
import pandas as pd
import pytest

from metacountregressor.cmf_catalogue import (
    HelperConditionalCMF,
    build_cmf_catalogue,
    conditional_cmf_from_fit,
    helper_level_cases,
)


def _toy_df(n=24):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ID": np.repeat(np.arange(n // 2), 2),
        "Y": rng.poisson(2.0, n).astype(float),
        "X1": rng.normal(size=n),
        "X2": rng.normal(size=n),
        "X3": rng.normal(size=n),
        "Z1": rng.normal(size=n),
        "W1": rng.normal(size=n),
    })


# --------------------------------------------------------------------------
# The CMFunction itself
# --------------------------------------------------------------------------

def test_no_helper_reduces_to_catalogue_cmf():
    """With no helpers the object is the ordinary CMF, not a near-copy."""
    lower = HelperConditionalCMF("URB", "lower", 0.1, changes=1.0)
    np.testing.assert_allclose(lower.cmf(2.0), 2.0 ** 0.1)
    assert lower.aadt_free is False
    upper = HelperConditionalCMF("WIDTH", "upper", -0.05, changes=1.0)
    assert upper.cmf(50_000) == pytest.approx(np.exp(-0.05))
    assert upper.aadt_free is True
    # no helper, no sigma shift, no spread invented
    assert np.isnan(upper.spread(50_000)[0])


def test_mean_helper_shifts_log_cmf_linearly():
    c = HelperConditionalCMF("URB", "lower", 0.02, delta={"SLOPE": -0.003})
    aadt = 30_000.0
    w0, w1 = 1.0, 3.0
    l0, l1 = c.log_cmf(aadt, {"SLOPE": w0}), c.log_cmf(aadt, {"SLOPE": w1})
    slope = (l1 - l0) / (w1 - w0)
    assert slope == pytest.approx(-0.003 * np.log(aadt))
    assert c.coef_at({"SLOPE": w1}) == pytest.approx(0.02 - 0.003 * w1)
    # a pure delta shift on the coefficient is AADT^delta per unit of w
    np.testing.assert_allclose(c.cmf(aadt, {"SLOPE": w1}),
                               aadt ** (0.02 - 0.003 * w1))


def test_variance_helper_moves_sigma_and_the_spread():
    logsig = np.log(0.2)
    c = HelperConditionalCMF("URB", "lower", 0.0, sigma=0.2, logsig=logsig,
                             omega={"CURVES": 0.5})
    assert c.sigma_at({"CURVES": 0.0}) == pytest.approx(0.2)
    assert c.sigma_at({"CURVES": 2.0}) == pytest.approx(np.exp(logsig + 1.0))
    aadt, w = 20_000.0, {"CURVES": 2.0}
    sig = c.sigma_at(w)
    lo, hi = c.spread(aadt, w)
    np.testing.assert_allclose([lo, hi],
                               [aadt ** (0.0 - sig), aadt ** (0.0 + sig)])
    # a wider sigma must widen the band, and the band is not the Wald interval
    assert c.spread(aadt, {"CURVES": -2.0})[1] < hi


def test_collapsed_sigma_reports_no_spread_band():
    """sigma indistinguishable from zero => no heterogeneity to draw a band."""
    c = HelperConditionalCMF("URB", "lower", 0.0, sigma=1e-4)
    lo, hi = c.spread(20_000.0)
    assert np.isnan(lo) and np.isnan(hi)


def test_delta_method_interval_uses_coef_and_loading_covariance():
    # Cov of (coef, SLOPE loading): coef variance 0.01, loading variance 0.0025,
    # covariance 0.001.  At w=0 the interval is the coefficient's Wald interval.
    cov = np.array([[0.01, 0.001], [0.001, 0.0025]])
    c = HelperConditionalCMF("URB", "lower", 0.0, delta={"SLOPE": 0.0},
                             cov=cov, param_index=[0, 1], z=1.96)
    aadt = 25_000.0
    scale = np.log(aadt)
    lo, hi = c.interval(aadt, {"SLOPE": 0.0})
    se0 = scale * np.sqrt(0.01)
    assert lo == pytest.approx(np.exp(-1.96 * se0))
    assert hi == pytest.approx(np.exp(1.96 * se0))
    # moving the helper widens the interval: g = scale*[1, w]
    w = 4.0
    se_w = abs(scale) * np.sqrt(0.01 + 2 * 0.001 * w + 0.0025 * w ** 2)
    lo_w, _ = c.interval(aadt, {"SLOPE": w})
    assert lo_w == pytest.approx(np.exp(-1.96 * se_w))
    assert lo_w < lo
    # without a covariance the interval is NaN, not a false zero-width band
    bare = HelperConditionalCMF("URB", "lower", 0.0, delta={"SLOPE": 0.0})
    assert np.isnan(bare.interval(aadt, {"SLOPE": 0.0})[0])


def test_expression_names_aadt_helper_and_sigma():
    c = HelperConditionalCMF("URB", "lower", 0.02, delta={"SLOPE": -0.002},
                             sigma=0.3, logsig=np.log(0.3),
                             omega={"CURVES": 0.4})
    expr = c.expression()
    assert "AADT" in expr and "SLOPE" in expr and "CURVES" in expr
    assert "sigma" in expr
    upper = HelperConditionalCMF("WIDTH", "upper", -0.03,
                                 delta={"MIMEDSH": 0.004})
    assert "exp(" in upper.expression() and "AADT" not in upper.expression()


def test_surface_is_aadt_free_for_upper_and_aadt_dependent_for_lower():
    upper = HelperConditionalCMF("WIDTH", "upper", -0.03,
                                 delta={"MIMEDSH": 0.01})
    up = upper.surface([10_000, 60_000], [("w=0", {"MIMEDSH": 0.0}),
                                          ("w=1", {"MIMEDSH": 1.0})])
    assert np.isnan(up["aadt"]).all()
    assert up["cmf"].nunique() == 2          # depends on w, not on AADT

    lower = HelperConditionalCMF("URB", "lower", 0.02)
    lo = lower.surface([10_000, 60_000], [("w=0", {})])
    assert lo["cmf"].nunique() == 2
    np.testing.assert_allclose(sorted(lo["cmf"]),
                               [10_000 ** 0.02, 60_000 ** 0.02])


def test_helper_level_cases_follow_the_data():
    cases = helper_level_cases({"SLOPE": np.arange(1.0, 11.0)})
    labels = [lab for lab, _ in cases]
    assert "SLOPE mean" in labels
    vals = {lab: w["SLOPE"] for lab, w in cases}
    assert vals["SLOPE mean"] == pytest.approx(5.5)
    assert min(vals.values()) >= 1.0 and max(vals.values()) <= 10.0
    extra = helper_level_cases({"SLOPE": [1.0, 2.0]},
                               extra={"SLOPE = 0": {"SLOPE": 0.0}})
    assert extra[-1] == ("SLOPE = 0", {"SLOPE": 0.0})


# --------------------------------------------------------------------------
# Reading the mapping out of a fitted hierarchical parameter vector
# --------------------------------------------------------------------------

def _random_model_spec(term="X2", mean_helper="Z1", var_helper="W1"):
    return {
        "fixed_terms": ["X1"],
        "rdm_terms": [f"{term}:normal"],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": [mean_helper] if mean_helper else [],
        "hetro_in_variances": [var_helper] if var_helper else [],
        "zi_terms": [],
        "dispersion": 1,
        "latent_classes": 1,
    }


def test_conditional_cmf_from_fit_reads_the_right_parameters():
    from metacountregressor import main_hpc as hpc

    df = _toy_df()
    data, spec = hpc.build_model_from_manual_spec(
        df, _random_model_spec(), id_col="ID", y_col="Y", R=8)
    idx = hpc.build_base_index(spec, model="nb")
    p = np.zeros(idx["total_params"])
    p[idx["ind_mean"][0]] = 0.25
    p[idx["ind_sd"][0]] = np.log(0.3)
    p[idx["hetro"][0]] = 0.07
    p[idx["hetro_var"][0]] = -0.4

    cond = conditional_cmf_from_fit(p, spec, idx, "X2", "upper",
                                    changes=2.0,
                                    cov=np.eye(idx["total_params"]))
    assert cond.coef == pytest.approx(0.25)
    assert cond.sigma == pytest.approx(0.3, rel=1e-6)
    assert cond.delta == {"Z1": pytest.approx(0.07)}
    assert cond.omega == {"W1": pytest.approx(-0.4)}
    # CMF at the fitted coefficient, scaled by the requested change
    np.testing.assert_allclose(cond.cmf(1.0), np.exp(0.25 * 2.0))
    # the conditional CMF must equal the same arithmetic done by hand
    np.testing.assert_allclose(
        cond.cmf(None, {"Z1": 2.0}), np.exp((0.25 + 0.07 * 2.0) * 2.0))
    assert cond.sigma_at({"W1": 1.0}) == pytest.approx(np.exp(np.log(0.3) - 0.4))


def test_conditional_cmf_from_fit_requires_a_random_term():
    from metacountregressor import main_hpc as hpc

    df = _toy_df()
    spec_dict = _random_model_spec()
    data, spec = hpc.build_model_from_manual_spec(
        df, spec_dict, id_col="ID", y_col="Y", R=8)
    idx = hpc.build_base_index(spec, model="nb")
    p = np.zeros(idx["total_params"])
    with pytest.raises(ValueError, match="not a random term"):
        conditional_cmf_from_fit(p, spec, idx, "X1", "upper")
    # a non-normal correlated term is refused: the engine maps that draw as
    # exp(mean + L u), so its SD is not a shift on the coefficient
    data_c, spec_c = hpc.build_model_from_manual_spec(
        df, {**spec_dict, "rdm_terms": [], "rdm_cor_terms": ["X2:lognormal"]},
        id_col="ID", y_col="Y", R=8)
    idx_c = hpc.build_base_index(spec_c, model="nb")
    with pytest.raises(ValueError, match="mixing distribution"):
        conditional_cmf_from_fit(np.zeros(idx_c["total_params"]), spec_c, idx_c,
                                 "X2", "upper")


# --------------------------------------------------------------------------
# Correlated random terms: the marginal SD is the Cholesky row norm
# --------------------------------------------------------------------------

def _correlated_model(spec_dict=None):
    from metacountregressor import main_hpc as hpc

    df = _toy_df()
    spec = spec_dict or {
        "fixed_terms": ["X1"],
        "rdm_terms": [],
        "rdm_cor_terms": ["X2:normal", "X3:normal"],
        "grouped_terms": [],
        "hetro_in_means": [],
        "hetro_in_variances": [],
        "zi_terms": [],
        "dispersion": 1,
        "latent_classes": 1,
    }
    data, s = hpc.build_model_from_manual_spec(df, spec, id_col="ID", y_col="Y",
                                               R=8)
    idx = hpc.build_base_index(s, model="nb")
    return s, idx


def test_correlated_marginal_sd_is_the_cholesky_row_norm():
    s, idx = _correlated_model()
    assert s.Kr_cor == 2
    p = np.zeros(idx["total_params"])
    cs = idx["chol"][0]
    p[idx["cor_mean"][0]] = 0.2
    p[idx["cor_mean"][0] + 1] = -0.1
    # tril order: L00, L10, L11 with L00/L11 stored as log-SDs
    p[cs], p[cs + 1], p[cs + 2] = np.log(0.5), 0.3, np.log(0.4)

    first = conditional_cmf_from_fit(p, s, idx, "X2", "upper")
    assert first.coef == pytest.approx(0.2)
    assert first.sigma == pytest.approx(0.5)         # row 0 is diagonal only
    assert first.sd_floor == 0.0

    second = conditional_cmf_from_fit(p, s, idx, "X3", "upper")
    assert second.coef == pytest.approx(-0.1)
    # sqrt(0.3^2 + 0.4^2) — the off-diagonal pair is part of the marginal SD
    assert second.sigma == pytest.approx(np.hypot(0.3, 0.4))
    assert second.sd_floor == pytest.approx(0.3)
    # the spread band uses that marginal, not the diagonal alone
    lo, hi = second.spread(None)
    np.testing.assert_allclose([lo, hi],
                               np.exp([-0.1 - np.hypot(0.3, 0.4),
                                       -0.1 + np.hypot(0.3, 0.4)]))


def test_correlated_variance_helper_shifts_only_the_diagonal():
    spec = {
        "fixed_terms": ["X1"],
        "rdm_terms": [],
        "rdm_cor_terms": ["X2:normal", "X3:normal"],
        "grouped_terms": [],
        "hetro_in_means": [],
        "hetro_in_variances": ["W1"],
        "zi_terms": [],
        "dispersion": 1,
        "latent_classes": 1,
    }
    s, idx = _correlated_model(spec)
    p = np.zeros(idx["total_params"])
    cs = idx["chol"][0]
    p[cs], p[cs + 1], p[cs + 2] = np.log(0.4), 0.3, np.log(0.4)
    # hetro_var layout is (helper, random) in random order: W1 x (X2, X3)
    p[idx["hetro_var"][0] + 1] = -0.5      # loading on X3

    cond = conditional_cmf_from_fit(p, s, idx, "X3", "upper")
    assert cond.omega == {"W1": pytest.approx(-0.5)}
    assert cond.sd_floor == pytest.approx(0.3)
    # sigma(w)^2 = sd_floor^2 + exp(2 (log 0.4 + omega w))
    for w in (0.0, 1.5):
        diag = 0.4 * np.exp(-0.5 * w)
        assert cond.sigma_at({"W1": w}) == pytest.approx(
            np.hypot(0.3, diag))
    # the off-diagonal part does not decay, so sigma(w) has a floor
    assert cond.sigma_at({"W1": 50.0}) == pytest.approx(0.3, abs=1e-6)
    assert "sqrt(" in cond.expression()


def test_derived_sigma_matches_the_engine_draws():
    """The row norm must be what the likelihood's draws actually do.

    Reading the Cholesky layout from ``index`` is only right if it matches the
    engine's own construction, so this simulates the draws the fit integrates
    over and compares their sample SD with the derived marginal.
    """
    from metacountregressor import main_hpc as hpc

    s, idx = _correlated_model()
    p = np.zeros(idx["total_params"])
    cs = idx["chol"][0]
    p[cs], p[cs + 1], p[cs + 2] = np.log(0.5), 0.3, np.log(0.4)
    p[idx["cor_mean"][0] + 1] = 0.1
    cond = conditional_cmf_from_fit(p, s, idx, "X3", "upper")

    draws = hpc.generate_halton_normal(1, 2, 4000, seed=7)
    beta = np.asarray(hpc.random_correlated(
        hpc.jnp.asarray(p[idx["cor_mean"][0]:idx["cor_mean"][1]]),
        hpc.jnp.asarray(p[cs:cs + 3]), hpc.jnp.asarray(draws), 2,
        hpc.jnp.asarray([0, 0])))
    np.testing.assert_allclose(np.std(beta[0, 1, :]), cond.sigma, rtol=0.05)
    np.testing.assert_allclose(np.mean(beta[0, 1, :]), cond.coef, atol=0.05)
    # and the CMF at +/-1 sigma is the band the spread reports
    np.testing.assert_allclose(cond.spread(None),
                               np.exp(cond.coef + np.array([-1, 1]) * cond.sigma))


def test_correlated_mean_helper_shifts_the_coefficient():
    spec = {
        "fixed_terms": ["X1"],
        "rdm_terms": [],
        "rdm_cor_terms": ["X2:normal", "X3:normal"],
        "grouped_terms": [],
        "hetro_in_means": ["Z1"],
        "hetro_in_variances": [],
        "zi_terms": [],
        "dispersion": 1,
        "latent_classes": 1,
    }
    s, idx = _correlated_model(spec)
    p = np.zeros(idx["total_params"])
    p[idx["cor_mean"][0] + 1] = 0.25
    p[idx["hetro"][0] + 1] = 0.06           # (Z1, X3) in random order
    cond = conditional_cmf_from_fit(p, s, idx, "X3", "upper")
    assert cond.delta == {"Z1": pytest.approx(0.06)}
    assert cond.coef_at({"Z1": 3.0}) == pytest.approx(0.25 + 0.18)
    assert cond.sigma == pytest.approx(1.0)   # all log-SDs zero



def test_term_map_translates_the_cmf_name_to_the_engine_column():
    from metacountregressor import main_hpc as hpc

    df = _toy_df().assign(__cmf_local__URB=lambda d: d["X2"])
    spec_dict = _random_model_spec(term="__cmf_local__URB",
                                   mean_helper=None, var_helper=None)
    data, spec = hpc.build_model_from_manual_spec(
        df, spec_dict, id_col="ID", y_col="Y", R=8)
    idx = hpc.build_base_index(spec, model="nb")
    p = np.zeros(idx["total_params"])
    p[idx["ind_mean"][0]] = 0.12
    p[idx["ind_sd"][0]] = np.log(0.15)
    cond = conditional_cmf_from_fit(p, spec, idx, "URB", "lower",
                                    term_map={"URB": "__cmf_local__URB"})
    assert cond.term == "URB" and cond.tier == "lower"
    np.testing.assert_allclose(cond.cmf(40_000.0), 40_000.0 ** 0.12)


# --------------------------------------------------------------------------
# The catalogue carries the function, not just a number
# --------------------------------------------------------------------------

def test_catalogue_marks_and_quotes_conditional_rows():
    data = _toy_df()
    data["AADT"] = np.linspace(5_000.0, 60_000.0, len(data))
    labels_len = 3                      # alpha0, alpha[X1], log_beta0
    params = np.array([-5.0, 0.1, 0.2])
    se = np.full(labels_len, 0.05)
    cond = HelperConditionalCMF("X1", "upper", 0.1, delta={"Z1": 0.01})
    cat = build_cmf_catalogue(params, se, data, ["X1"], [], [False], [],
                              model="poisson", conditional={"X1": cond})
    row = cat[cat["term"] == "X1"].iloc[0]
    assert bool(row["helper_conditional"]) is True
    assert "Z1" in row["cmf_expression"] and "exp(" in row["cmf_expression"]
    # a term without one is untouched by the feature
    assert cat.attrs["conditional"] == {"X1": cond}
    plain = build_cmf_catalogue(params, se, data, ["X1"], [], [False], [],
                                model="poisson")
    assert not bool(plain["helper_conditional"].any())
    assert (plain["cmf_expression"] == "").all()
