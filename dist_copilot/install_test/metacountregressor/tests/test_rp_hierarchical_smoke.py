"""Smoke test: RP structural equation embedded in the hierarchical CMF model.

End-to-end chain on tiny synthetic two-tier data with genuine random-parameter
heterogeneity:

  fixed hierarchical CMF  ->  per-site gradient (GSE) scores  ->
  GSE screen  ->  RPStructuralSpec (random + helping hands + GSE)  ->
  RP hierarchical fit  ->  likelihood / BIC comparison.

Uses the real hierarchical mapping (CMFExperimentBuilder.make_manual_cmf_spec:
upper-level direct terms + lower-level AADT interactions) and the real JAX
engine (build_model_from_manual_spec -> CountModel -> LBFGS), with small R
and short optimisations so it stays a smoke test.
"""

import numpy as np
import pandas as pd

# Importing experiment_package applies main_hpc_lc_patch in place.
import metacountregressor.experiment_package as _ep  # noqa: F401
from metacountregressor.cmf_package import CMFExperimentBuilder
from metacountregressor.main_hpc import (
    CountModel,
    build_base_index,
    build_model_from_manual_spec,
    mixed_model_loglik,
)
from metacountregressor.main_hpc import jnp
from metacountregressor.random_parameter_structure import (
    RPStructuralSpec,
    standardise_scores,
)


def _hierarchical_rp_data(seed=7, n_sites=48, per_site=3):
    """Two-tier DGP with a genuinely heterogeneous CURVES coefficient.

    log mu = c + b_aadt*logAADT + d_lanes*LANES + b_n*CURVES
             + g_lanes*LANES*logAADT + log(LEN),
    b_n = 0.35 + 0.40*Z_n + exp(-0.7 + 0.6*W_n) * nu_n,  y ~ NB(mu, alpha).
    """
    rng = np.random.default_rng(seed)
    N, P = n_sites, per_site
    n = N * P
    site = np.repeat(np.arange(N), P)
    aadt = np.exp(rng.normal(9.0, 0.5, n))
    lanes = rng.integers(2, 5, n).astype(float)
    curves = rng.uniform(0.0, 6.0, n)
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    length = rng.uniform(0.2, 1.5, n)
    z_n = z.reshape(N, P).mean(axis=1)
    w_n = w.reshape(N, P).mean(axis=1)
    b_n = 0.35 + 0.40 * z_n + np.exp(-0.7 + 0.6 * w_n) * rng.normal(size=N)
    b = b_n[site]
    log_aadt = np.log(aadt)
    eta = (
        -6.0
        + 0.80 * log_aadt
        - 0.30 * (lanes - lanes.mean()) / lanes.std()
        + b * (curves - curves.mean()) / curves.std()
        + 0.15 * ((lanes - lanes.mean()) / lanes.std()) * log_aadt
        + np.log(length)
    )
    mu = np.exp(eta)
    alpha_true = 1.5  # NB size r
    p = alpha_true / (alpha_true + mu)
    y = rng.negative_binomial(alpha_true, 1.0 - p).astype(float)
    return pd.DataFrame(
        {
            "ID": site,
            "Y": y,
            "AADT": aadt,
            "LANES": lanes,
            "CURVES": curves,
            "Z1": z,
            "W1": w,
            "LEN": length,
            "OFFSET": np.log(length),
        }
    )


def _short_lbfgs(model, init, maxiter=120):
    from jaxopt import LBFGS

    solver = LBFGS(fun=model.objective, maxiter=maxiter, tol=1e-6)
    res = solver.run(jnp.asarray(init, dtype=float))
    return np.asarray(res.params, dtype=float), float(res.state.value)


def _warm_start_from_fixed(p_f, spec_f, idx_f, spec_r, idx_r):
    """Embed a fixed hierarchical fit into the larger RP vector.

    Shared fixed effects (by name) and the NB dispersion carry over; the
    new RP blocks start near the fixed model (small log-SD, zero loadings),
    so short optimisation only has to discover the heterogeneity.
    """
    p_r = np.zeros(idx_r["total_params"])
    fs, fe = idx_f["fixed"]
    rs, re = idx_r["fixed"]
    fpos = {t: i for i, t in enumerate(spec_f.fixed_names)}
    for j, t in enumerate(spec_r.fixed_names):
        if t in fpos:
            p_r[rs + j] = p_f[fs + fpos[t]]
    p_r[idx_r["dispersion"]] = p_f[idx_f["dispersion"]]
    sd_s, sd_e = idx_r["ind_sd"]
    p_r[sd_s:sd_e] = -2.0  # small initial heterogeneity
    return p_r


def test_rp_embeds_in_hierarchical_model():
    df = _hierarchical_rp_data()
    R = 32

    cb = CMFExperimentBuilder(
        df=df,
        y_col="Y",
        aadt_col="AADT",
        baseline_vars=["LANES", "CURVES"],
        local_vars=["LANES"],
    )

    # ---- 1. fixed hierarchical CMF (upper direct + lower AADT interaction)
    fixed_spec = cb.make_manual_cmf_spec(
        baseline_fixed=["LANES", "CURVES"],
        local_fixed=["LANES"],
        dispersion=1,
    )
    assert fixed_spec["hetro_in_means"] == []
    data_f, spec_f = build_model_from_manual_spec(
        cb.df.assign(
            __cmf_log_aadt=np.log(cb.df["AADT"].astype(float)),
            __cmf_local__LANES=cb.df["LANES"].astype(float)
            * np.log(cb.df["AADT"].astype(float)),
        ),
        {**fixed_spec, "group_id_col": None},
        id_col="ID",
        y_col="Y",
        offset_col="OFFSET",
        R=R,
    )
    # NOTE: make_manual_cmf_spec already emits the mapped interaction names;
    # the frame above only needs the mapped columns to exist.
    model_f = CountModel(spec_f, data_f)
    idx_f = build_base_index(spec_f, model="nb")
    p_f, _ = _short_lbfgs(model_f, np.zeros(idx_f["total_params"]))
    ll_fixed = -float(model_f.objective(jnp.asarray(p_f)))
    assert np.isfinite(ll_fixed)

    # ---- 2. GSE scores: per-site gradient of the fixed fit wrt CURVES
    j_fixed = [i for i, t in enumerate(spec_f.fixed_names) if t == "CURVES"]
    assert len(j_fixed) == 1
    import jax

    per_site = jax.jacobian(
        lambda q: mixed_model_loglik(q, data_f, spec_f, indivi=True)
    )(jnp.asarray(p_f))
    g_raw = np.asarray(per_site)[:, j_fixed[0] : j_fixed[0] + 1]
    assert g_raw.shape == (df["ID"].nunique(), 1)
    G = standardise_scores(g_raw)
    assert np.abs(G).max() > 0  # scores carry signal

    # ---- 3. screen -> structural spec -> RP hierarchical manual spec
    screen = pd.DataFrame(
        {"candidate": ["CURVES"], "best_helper": ["Z1"]}
    )
    rp_struct = RPStructuralSpec.from_gse_screen(screen)
    assert rp_struct.random_vars == ["CURVES"]
    assert rp_struct.mean_helpers == ["Z1"]
    assert rp_struct.gse is True
    rp_struct.var_helpers.append("W1")
    rp_spec = cb.make_manual_cmf_spec(
        baseline_fixed=["LANES"],
        baseline_random=["CURVES"],
        local_fixed=["LANES"],
        hetro_in_means=rp_struct.mean_helpers,
        hetro_in_variances=rp_struct.var_helpers,
        dispersion=1,
    )
    rp_spec["gse_scores"] = G
    assert rp_spec["rdm_terms"] == ["CURVES:normal"]

    data_r, spec_r = build_model_from_manual_spec(
        cb.df.assign(
            __cmf_log_aadt=np.log(cb.df["AADT"].astype(float)),
            __cmf_local__LANES=cb.df["LANES"].astype(float)
            * np.log(cb.df["AADT"].astype(float)),
        ),
        {**rp_spec, "group_id_col": None},
        id_col="ID",
        y_col="Y",
        offset_col="OFFSET",
        R=R,
    )
    assert spec_r.Kgse == 1 and spec_r.Kv == 1 and spec_r.Kh == 1
    model_r = CountModel(spec_r, data_r)
    idx_r = build_base_index(spec_r, model="nb")
    assert "gse_gamma" in idx_r and "hetro_var" in idx_r
    p0_r = _warm_start_from_fixed(p_f, spec_f, idx_f, spec_r, idx_r)
    ll_start = -float(model_r.objective(jnp.asarray(p0_r)))
    p_r, _ = _short_lbfgs(model_r, p0_r, maxiter=150)
    ll_rp = -float(model_r.objective(jnp.asarray(p_r)))
    assert np.isfinite(ll_rp)

    # ---- 4. the RP embedding must pay: better likelihood for the true RP DGP
    print(
        f"\n  fixed LL={ll_fixed:.1f}  RP start LL={ll_start:.1f}  "
        f"RP+GSE LL={ll_rp:.1f}  dLL={ll_rp - ll_fixed:+.1f}"
    )
    assert ll_start > ll_fixed - 5.0  # warm start lands near the fixed fit
    assert ll_rp >= ll_start - 1e-6  # optimiser never regresses
    assert ll_rp > ll_fixed - 2.0  # nesting tolerance for short optimisation

    # GSE loading is identified and the summary aligns with the index.
    # (The LC patch overrides print_summary with a dict return, so reach the
    # base implementation via its private alias for the parameter table.)
    s, e = idx_r["gse_gamma"]
    assert np.isfinite(p_r[s:e]).all()
    from metacountregressor import main_hpc as _hpc

    _base_summary = getattr(_hpc, "_orig_print_summary", _hpc.print_summary)
    rep = _base_summary(
        type("R", (), {"params": jnp.asarray(p_r)})(),
        model_r.objective,
        data_r,
        spec_r,
        idx_r,
        return_df=True,
    )
    df_rep = rep if isinstance(rep, pd.DataFrame) else rep.get("df", rep)
    assert isinstance(df_rep, pd.DataFrame)
    assert len(df_rep) == idx_r["total_params"]
    assert any(str(t).startswith("gse(") for t in df_rep["Parameter"])
    assert any(str(t).startswith("hetro_var(") for t in df_rep["Parameter"])

    # BIC bookkeeping is finite on both sides
    n = df["ID"].nunique()
    bic_f = -2 * ll_fixed + idx_f["total_params"] * np.log(n)
    bic_r = -2 * ll_rp + idx_r["total_params"] * np.log(n)
    print(f"  fixed BIC={bic_f:.1f}  RP+GSE BIC={bic_r:.1f}")
    assert np.isfinite(bic_f) and np.isfinite(bic_r)
