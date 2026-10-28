import numpy as np
import pandas as pd

from metacountregressor import GA_CMF_AADT_JAX as G
from metacountregressor.cmf_agds_search import (
    CMFAGDSObjective, run_agds_search, refit_and_report)


def _data(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    aadt = rng.uniform(1000, 20000, n)
    x, z, w = rng.normal(size=(3, n))
    mu = np.exp(-8.0 + 0.3 * x + (0.9 + 0.2 * z) * np.log(aadt))
    return pd.DataFrame({"FREQ": rng.poisson(mu), "AADT": aadt,
                         "X": x, "Z": z, "W": w})


def test_shape_draws_unit_variance():
    import jax.numpy as jnp
    d = jnp.array(np.random.default_rng(0).normal(size=(4000, 3, 3)))
    out = np.asarray(G.shape_draws(
        d, ["uniform", "triangular", "lognormal"], rho=0.4))
    assert np.allclose(out.reshape(-1, 3).std(0), 1.0, atol=0.1)
    assert np.allclose(out.reshape(-1, 3).mean(0), 0.0, atol=0.1)


def test_objective_decode_and_fitness():
    df = _data()
    obj = CMFAGDSObjective(df, ["X", "Z", "W"], R=30, n_hops=0)
    layout = [1, 0, 0,  1, 2, 1,  0, 0, 0,  1, 0]
    sp = obj.spec(layout)
    assert sp["upper"] == ["X"] and sp["lower"] == [] and sp["model"] == "poisson"
    f = obj.get_fitness(layout)["bic"]
    assert np.isfinite(f) and f < 1e11
    assert obj.get_fitness([0] * 3 + [1] * 3 + [0] * 3 + [1, 0])["bic"] == 1e12


def test_agds_search_prefers_true_terms_and_reports():
    df = _data()
    out = run_agds_search(df, ["X", "Z", "W"], seeds=[0], pop_size=6,
                          generations=3, R=30, n_hops=0,
                          allow_models=("poisson",), allow_dists=("normal",))
    best = out[0]["spec"]
    assert "X" in best["upper"] + best["lower"]
    rep = refit_and_report(df, best, R=40)
    assert len(rep.catalogue) == len(best["upper"]) + len(best["lower"])
    assert np.isfinite(rep.mean_aadt_elasticity)
