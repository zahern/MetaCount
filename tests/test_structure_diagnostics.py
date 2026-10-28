"""Structure tests: where should random parameters attach in the hierarchy?"""

import numpy as np
import pandas as pd

from metacountregressor.structure_diagnostics import structure_diagnostics


def _simulate(sigma_upper=0.0, sigma_lower=0.0, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    aadt = rng.uniform(1000, 20000, n)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    a = -8.0 + (0.3 + sigma_upper * rng.normal(size=n)) * x
    e = 0.9 + (0.2 + sigma_lower * rng.normal(size=n)) * z
    mu = np.exp(a + e * np.log(aadt))
    return pd.DataFrame({"FREQ": rng.poisson(mu).astype(float),
                         "AADT": aadt, "X": x, "Z": z})


def test_identification_report_flags_flat_sigma():
    import jax.numpy as jnp
    from metacountregressor import GA_CMF_AADT_JAX as G
    df = _simulate()
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        df, ["X"], ["Z"], [False], [False], model="poisson", R=50)
    rep = G.identification_report(res, y, A, B, L, ["X"], ["Z"], [False],
                                  [False], draws, "poisson", off)
    assert rep["identified"] and np.isfinite(rep["condition_number"])
    res_r, y, A, B, L, draws, _, off = G.fit_final_model(
        df, ["X"], ["Z"], [True], [False], model="poisson", R=50)
    rep = G.identification_report(res_r, y, A, B, L, ["X"], ["Z"], [True],
                                  [False], draws, "poisson", off)
    assert "condition_number" in rep and "flat_sigmas" in rep


def test_cmf_catalogue_recovers_upper_effect():
    from metacountregressor import GA_CMF_AADT_JAX as G
    from metacountregressor.cmf_catalogue import build_cmf_catalogue
    df = _simulate(n=3000)
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        df, ["X"], ["Z"], [False], [False], model="poisson", R=50)
    se, _ = G.compute_se(res, y, A, B, L, [False], [False], draws, 50,
                         "poisson", off)
    cat = build_cmf_catalogue(res.params, se, df, ["X"], ["Z"], [False], [False])
    assert set(cat["tier"]) == {"upper", "lower"}
    up = cat[cat.term == "X"].iloc[0]
    assert abs(np.log(up.cmf) - 0.3) < 0.1
    assert up.cmf_lo < up.cmf < up.cmf_hi


def test_derive_cmf_matches_truth_and_combines():
    from metacountregressor import GA_CMF_AADT_JAX as G
    from metacountregressor.cmf_catalogue import derive_cmf
    df = _simulate(n=3000)
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        df, ["X"], ["Z"], [False], [False], model="poisson", R=50)
    se, _ = G.compute_se(res, y, A, B, L, [False], [False], draws, 50,
                         "poisson", off)
    cov = G.observed_covariance(res, y, A, B, L, [False], [False], draws,
                                "poisson", off)
    assert np.allclose(np.sqrt(np.diag(cov)), se, rtol=0.2)
    args = (res.params, cov, ["X"], ["Z"], [False], [False])
    args = (res.params, cov, ["X"], ["Z"], [False], [False])
    one = derive_cmf(*args, changes={"X": 1})
    assert abs(np.log(one["cmf"]) - 0.3) < 0.1
    assert one["cmf_lo"] < one["cmf"] < one["cmf_hi"]
    both = derive_cmf(*args, changes={"X": 1, "Z": 1}, aadt=5000)
    assert both["cmf"] > one["cmf"]


def test_homogeneous_data_recommends_fixed():
    df = _simulate()
    out = structure_diagnostics(df, ["X"], ["Z"], R=100)
    assert set(out["recommendation"]) == {"fixed"}
    assert out["p_value"].between(0, 0.5).all()


def test_upper_heterogeneity_attaches_to_upper_tier():
    df = _simulate(sigma_upper=0.6)
    out = structure_diagnostics(df, ["X"], ["Z"], R=100)
    rec = out.drop_duplicates("term").set_index("term")["recommendation"]
    assert rec["X"] == "upper"
