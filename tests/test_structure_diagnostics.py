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
