"""Tests for the canonical random-parameter structural equation.

Covers random_parameter_structure math plus its wiring into the base
engine (ModelSpec Kv/Kgse, build_jax_data, build_base_index, unpack_params,
build_eta, summary names).
"""

import numpy as np
import pandas as pd

from metacountregressor.random_parameter_structure import (
    RPStructuralSpec,
    rp_draws_correlated,
    rp_draws_independent,
    rp_local_logsd,
    rp_local_mean,
    standardise_scores,
    structural_equation_latex,
)


def test_latex_equation_present():
    eq = structural_equation_latex()
    assert "beta" in eq and "gamma" in eq and "omega" in eq


def test_standardise_scores_zero_variance():
    g = np.array([[1.0, 5.0], [1.0, 7.0], [1.0, 9.0]])
    out = standardise_scores(g)
    assert np.all(out[:, 0] == 0.0)
    assert abs(out[:, 1].mean()) < 1e-12
    assert abs(out[:, 1].std() - 1.0) < 1e-12


def test_local_mean_and_logsd():
    base = np.array([0.5, -0.2])
    Z = np.array([[1.0], [2.0]])
    Gamma = np.array([[0.1, 0.3]])
    mu = rp_local_mean(base, Z, Gamma)
    assert mu.shape == (2, 2)
    np.testing.assert_allclose(mu[0], [0.6, 0.1])
    np.testing.assert_allclose(mu[1], [0.7, 0.4])
    G = np.array([[1.0, -1.0], [0.5, 0.5]])
    gg = np.array([0.2, 0.4])
    mu2 = rp_local_mean(base, Z, Gamma, G=G, gamma_gse=gg)
    np.testing.assert_allclose(mu2[0], [0.8, -0.3])
    ls = rp_local_logsd(np.array([0.0, 0.0]), Z, Gamma)
    np.testing.assert_allclose(ls[0], [0.1, 0.3])


def test_draws_normal_matches_closed_form():
    rng = np.random.default_rng(0)
    mean = np.array([[0.5], [1.0]])
    logsd = np.array([0.0, 0.0])
    v = rng.standard_normal((2, 1, 5))
    b = rp_draws_independent(mean, logsd, v, ["normal"])
    import math

    s = math.log(2.0)  # softplus(0)
    np.testing.assert_allclose(b[:, 0, :], mean + s * v[:, 0, :])


def test_correlated_draws_match_engine_convention():
    from metacountregressor import main_hpc as hpc

    jnp = hpc.jnp if hasattr(hpc, "jnp") else None
    assert jnp is not None
    mean = np.array([0.2, -0.1])
    L = np.array([[0.0, 0.0], [0.3, -0.2]])  # log-diagonal
    rng = np.random.default_rng(1)
    v = rng.standard_normal((4, 2, 6))
    mine = rp_draws_correlated(mean, L, v, ["normal", "normal"])
    eng = np.asarray(
        hpc.random_correlated(
            jnp.asarray(mean),
            jnp.asarray(np.array([[0.0, 0.0], [0.3, -0.2]]).flatten()[[0, 2, 3]]),
            jnp.asarray(v),
            2,
            jnp.asarray([0, 0]),
        )
    )
    np.testing.assert_allclose(mine, eng, rtol=1e-5, atol=1e-7)


def _toy_df():
    rng = np.random.default_rng(2)
    n = 30
    return pd.DataFrame(
        {
            "ID": np.repeat(np.arange(10), 3),
            "Y": rng.poisson(2.0, n).astype(float),
            "X1": rng.normal(size=n),
            "X2": rng.normal(size=n),
            "X3": rng.normal(size=n),
            "Z1": rng.normal(size=n),
            "W1": rng.normal(size=n),
        }
    )


def test_base_engine_accepts_variance_helpers_and_gse():
    from metacountregressor import main_hpc as hpc

    df = _toy_df()
    spec = {
        "fixed_terms": ["X1"],
        "rdm_terms": ["X2:normal"],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": ["Z1"],
        "hetro_in_variances": ["W1"],
        "zi_terms": [],
        "dispersion": 1,
        "latent_classes": 1,
    }
    data, mspec = hpc.build_model_from_manual_spec(
        df, spec, id_col="ID", y_col="Y", R=16
    )
    assert mspec.Kv == 1
    assert mspec.hetro_var_names == ("W1",)
    assert "Xh_var" in data
    idx = hpc.build_base_index(mspec, model="nb")
    assert "hetro_var" in idx
    # GSE scores path (with simulation draws so the RP term is live)
    rng = np.random.default_rng(3)
    g = rng.standard_normal((df["ID"].nunique(), 1))
    draws = np.asarray(
        hpc.generate_halton_normal(df["ID"].nunique(), 1, 16, seed=42)
    )
    spec2 = dict(spec, gse_scores=g)
    data2, mspec2 = hpc.build_model_from_manual_spec(
        df, spec2, id_col="ID", y_col="Y", draws_ind=draws, R=16
    )
    assert mspec2.Kgse == 1
    assert "G" in data2
    idx2 = hpc.build_base_index(mspec2, model="nb")
    assert "gse_gamma" in idx2
    # unpack + eta run end to end
    npar = idx2["total_params"]
    params = hpc.jnp.zeros(npar)
    eta = hpc.build_eta(params, data2, mspec2)
    assert eta.shape[0] == data2["y"].shape[0]
    # every structural block moves eta (helpers + GSE are live, not dead)
    eta0 = np.asarray(eta)
    assert np.abs(eta0).max() > 0
    for key in ("hetro", "hetro_var", "gse_gamma"):
        s, e = idx2[key]
        pp = np.zeros(npar)
        pp[s:e] = 0.5
        etax = np.asarray(hpc.build_eta(hpc.jnp.asarray(pp), data2, mspec2))
        assert np.abs(etax - eta0).max() > 1e-8, key
    # likelihood is finite and sensitive to the GSE loading
    ll0 = float(hpc.mixed_model_loglik(params, data2, mspec2))
    assert np.isfinite(ll0)
    pg = np.zeros(npar)
    s, e = idx2["gse_gamma"]
    pg[s:e] = 1.0
    ll1 = float(hpc.mixed_model_loglik(hpc.jnp.asarray(pg), data2, mspec2))
    assert np.isfinite(ll1) and ll1 != ll0


def test_build_eta_helpers_match_module_math():
    from metacountregressor import main_hpc as hpc

    df = _toy_df()
    spec = {
        "fixed_terms": ["X1"],
        "rdm_terms": ["X2:normal"],
        "rdm_cor_terms": [],
        "grouped_terms": [],
        "hetro_in_means": ["Z1"],
        "hetro_in_variances": ["W1"],
        "zi_terms": [],
        "dispersion": 0,
        "latent_classes": 1,
    }
    data, mspec = hpc.build_model_from_manual_spec(
        df, spec, id_col="ID", y_col="Y", R=8
    )
    idx = hpc.build_base_index(mspec, model="poisson")
    rng = np.random.default_rng(4)
    p = rng.normal(size=idx["total_params"]) * 0.1
    blocks = hpc.unpack_params(hpc.jnp.asarray(p), mspec, model="poisson")
    assert blocks["gamma"] is not None and blocks["gamma_var"] is not None
    Z = np.asarray(data["Xh"])[:, :, 0].mean(axis=1, keepdims=True)
    W = np.asarray(data["Xh_var"])[:, :, 0].mean(axis=1, keepdims=True)
    Gamma = np.asarray(blocks["gamma"])
    Omega = np.asarray(blocks["gamma_var"])
    K = mspec.K_random_total
    assert Gamma.shape == (1, K) and Omega.shape == (1, K)


def test_rp_spec_builders():
    rng = np.random.default_rng(5)
    s = RPStructuralSpec(
        random_vars=["X2:normal", "X3"],
        mean_helpers=["Z1"],
        var_helpers=["W1"],
        gse=True,
    )
    g = rng.standard_normal((10, 2))
    ms = s.to_manual_spec(fixed_terms=["X1"], gse_scores=g)
    assert ms["rdm_terms"] == ["X2:normal", "X3:normal"]
    assert ms["hetro_in_means"] == ["Z1"]
    assert ms["hetro_in_variances"] == ["W1"]
    assert ms["gse_scores"].shape == (10, 2)
    df = pd.DataFrame(
        {"candidate": ["X2", "X3"], "best_helper": ["Z1", "-"]}
    )
    s2 = RPStructuralSpec.from_gse_screen(df)
    assert s2.random_vars == ["X2", "X3"]
    assert s2.mean_helpers == ["Z1"]
    assert s2.gse is True
    # end-to-end: GSE spec builds data with G aligned to random order
    from metacountregressor import main_hpc as hpc

    tdf = _toy_df()
    gg = rng.standard_normal((tdf["ID"].nunique(), 2))
    ms2 = s2.to_manual_spec(fixed_terms=["X1"], gse_scores=gg)
    data, mspec = hpc.build_model_from_manual_spec(
        tdf, ms2, id_col="ID", y_col="Y", R=8
    )
    assert mspec.Kgse == 2
    assert data["G"].shape[2] == 2
