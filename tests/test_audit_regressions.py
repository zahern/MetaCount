"""Regression tests for the model-structure audit.

Covers the defects fixed in the audit so they cannot silently return:

* pavement functional forms (all 12 fit; forms 5/9 build interactions;
  data-driven shape search; GP operators), hazard likelihood consistency,
* SA perturbation operators honour allowed roles / bans,
* latent-class count likelihood reproduces the published NLOGIT values for
  Washington, Karlaftis & Mannering, Example 16-3 (LC-NB and LC-NB with a
  class-probability function) at the published estimates,
* random-parameter columns are not centred (RPNB at the book estimates),
* triangular draws are triangular, correlated heterogeneity in variances is
  observation-specific, grouped coefficients are integrated per group, and
  nested (group + site) random coefficients validate correctly.

Run from the repository root so ``import metacountregressor`` resolves to
the real package (not the stale nested copy under metacountregressor/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metacountregressor.experiment_package as _ep  # noqa: F401  (patches main_hpc)
import metacountregressor.main_hpc_lc_patch as P
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Example 16-3 data (variables exactly as in the NLOGIT command file)
# ---------------------------------------------------------------------------
X163 = ["EXPOSE", "LOWPRE", "GBRPM", "INTPM", "CPM", "HISNOW", "FRICTION"]


def _ex163():
    from metacountregressor import load_example16_3_raw_data
    df = load_example16_3_raw_data()
    df["EXPOSE"] = df.AADT * df.LENGTH * 365 / 1e8
    df["LOWPRE"] = (df.AVEPRE <= 1.5).astype(float)
    df["GBRPM"] = df.GRADEBR / df.LENGTH
    df["INTPM"] = df.INTECHAG / df.LENGTH
    df["CPM"] = df.CURVES / df.LENGTH
    df["HISNOW"] = (df.AVESNOW > 1).astype(float)
    return df


def _manual(**kw):
    base = dict(fixed_terms=[], rdm_terms=[], rdm_cor_terms=[], grouped_terms=[],
                hetro_in_means=[], zi_terms=[], membership_terms=[],
                dispersion=0, latent_classes=1)
    base.update(kw)
    return base


def _to_pkg_scale(df, data, raw_by_name):
    """Map raw-unit coefficients {name: value} (incl. 'const') to the
    package's internal (scaled / centred) parameterisation."""
    sc = data["scaler"]
    const = raw_by_name["const"]
    out = {}
    for name, b in raw_by_name.items():
        if name == "const":
            continue
        mu, s = sc.get(name, (0.0, 1.0))
        out[name] = b * s
        const += b * mu
    out["const"] = const
    return out


# NLOGIT estimates (book PDF, Example 16-3 extended). NLOGIT's LC "Alpha" is
# theta = 1/alpha; the package uses log(alpha).
_LC_BOOK = {
    1: dict(const=1.77652, EXPOSE=2.82737, LOWPRE=-.33465, GBRPM=.08621,
            INTPM=.33500, CPM=-.08092, HISNOW=.06144, FRICTION=.00125, theta=4.34981),
    2: dict(const=6.79519, EXPOSE=1.34809, LOWPRE=-.81472, GBRPM=-.18550,
            INTPM=-.55301, CPM=-.49793, HISNOW=-.81487, FRICTION=-.07137, theta=1.05338),
}
_LCM_BOOK = {
    1: dict(const=1.03907, EXPOSE=6.90031, LOWPRE=-.18132, GBRPM=.13900,
            INTPM=.24185, CPM=-.06283, HISNOW=.14752, FRICTION=-.00130, theta=4.29362),
    2: dict(const=3.89765, EXPOSE=1.43017, LOWPRE=-.08827, GBRPM=-.22511,
            INTPM=-.09078, CPM=-.11575, HISNOW=-.21533, FRICTION=-.01622, theta=1.31509),
}


def _lc_params(df, data, book, gamma):
    p = []
    for c in (1, 2):
        b = dict(book[c])
        theta = b.pop("theta")
        m = _to_pkg_scale(df, data, b)
        p += [m["const"]] + [m[v] for v in X163] + [np.log(1.0 / theta)]
    return jnp.array(p + list(gamma))


def test_lc_nb_likelihood_reproduces_nlogit_ex16_3():
    df = _ex163()
    data, spec = P.build_model_from_manual_spec(
        df=df, manual_spec=_manual(fixed_terms=X163, dispersion=1, latent_classes=2),
        id_col="ID", y_col="FREQ", offset_col=None, R=10)
    spec = P.replace(spec, model="nb")
    # package logit is class 2 vs class 1 (reference)
    gamma = [np.log(.23412 / .76588)]
    ll = -float(P.mixed_model_loglik(_lc_params(df, data, _LC_BOOK, gamma), data, spec))
    assert ll == pytest.approx(-940.18855, abs=2e-3)


def test_lc_nb_class_probability_function_reproduces_nlogit_ex16_3():
    df = _ex163()
    data, spec = P.build_model_from_manual_spec(
        df=df, manual_spec=_manual(fixed_terms=X163, dispersion=1, latent_classes=2,
                                   membership_terms=["EXPOSE"]),
        id_col="ID", y_col="FREQ", offset_col=None, R=10)
    spec = P.replace(spec, model="nb")
    mu, s = data["scaler"]["EXPOSE"]
    g0, g1 = 3.84194, -13.2268          # NLOGIT logit of class 1 vs class 2
    gamma = [-(g0 + g1 * mu), -(g1 * s)]
    ll = -float(P.mixed_model_loglik(_lc_params(df, data, _LCM_BOOK, gamma), data, spec))
    assert ll == pytest.approx(-933.83707, abs=2e-3)


def test_random_parameter_columns_are_scaled_not_centred():
    df = _ex163()
    rd = ["EXPOSE", "INTPM", "CPM"]
    data, spec = P.build_model_from_manual_spec(
        df=df, manual_spec=_manual(fixed_terms=["LOWPRE", "GBRPM", "HISNOW", "FRICTION"],
                                   rdm_terms=[f"{v}:normal" for v in rd], dispersion=1),
        id_col="ID", y_col="FREQ", offset_col=None, R=1000)
    spec = P.replace(spec, model="nb")
    for v in rd:
        assert data["scaler"][v][0] == 0.0
    fixed = dict(const=2.88190, LOWPRE=-.41734, GBRPM=-.05080, HISNOW=-.05048, FRICTION=-.01281)
    m = _to_pkg_scale(df, data, fixed)
    sd = {v: data["scaler"][v][1] for v in rd}
    means = dict(EXPOSE=2.50802, INTPM=.16909, CPM=-.17775)
    sds = dict(EXPOSE=.83830, INTPM=.51164, CPM=.03133)
    p = ([m["const"]] + [m[v] for v in ["LOWPRE", "GBRPM", "HISNOW", "FRICTION"]]
         + [means[v] * sd[v] for v in rd] + [np.log(sds[v] * sd[v]) for v in rd]
         + [np.log(1 / 2.96187)])
    ll = -float(P.mixed_model_loglik(jnp.array(p), data, spec))
    # NLOGIT (200 Halton draws): -962.65751. Centring the random columns gave -966.8.
    assert ll == pytest.approx(-962.66, abs=0.75)


# ---------------------------------------------------------------------------
# Random-parameter draws and heterogeneity
# ---------------------------------------------------------------------------
def test_triangular_draws_are_triangular_not_uniform():
    from metacountregressor.main_hpc import transform_draws
    z = np.random.default_rng(0).normal(size=(1, 1, 200_000))
    tri = np.asarray(transform_draws(jnp.array(z), jnp.zeros(1), jnp.zeros(1), jnp.array([2])))
    uni = np.asarray(transform_draws(jnp.array(z), jnp.zeros(1), jnp.zeros(1), jnp.array([3])))
    # triangular on [-1, 1] has variance 1/6; uniform 1/3
    assert np.var(tri) == pytest.approx(1 / 6, rel=0.02)
    assert np.var(uni) == pytest.approx(1 / 3, rel=0.02)


def test_correlated_heterogeneity_in_variances_is_observation_specific():
    from metacountregressor.main_hpc import random_correlated
    rng = np.random.default_rng(1)
    draws = jnp.array(rng.normal(size=(2, 2, 20_000)))
    chol = jnp.array([0.0, 0.0, 0.0])            # log-diag 0 -> L = I
    scale = jnp.array([[0.0, 0.0], [np.log(3.0), np.log(3.0)]])
    beta = np.asarray(random_correlated(jnp.zeros((2, 2)), chol, draws, 2,
                                        jnp.array([0, 0]), row_log_scale=scale))
    assert beta[0].std(axis=-1) == pytest.approx([1.0, 1.0], rel=0.03)
    assert beta[1].std(axis=-1) == pytest.approx([3.0, 3.0], rel=0.03)


def test_grouped_coefficients_are_integrated_at_group_level():
    rng = np.random.default_rng(3)
    rows = []
    for g in range(12):
        u = rng.normal()
        for s in range(5):
            x = rng.normal()
            for t in range(4):
                rows.append(dict(SITE=g * 5 + s, CORR=g, X=x,
                                 Y=rng.poisson(np.exp(0.5 + (0.4 + 0.5 * u) * x))))
    df = pd.DataFrame(rows)
    ms = _manual(grouped_terms=["X:normal"], group_id_col="CORR")
    data, spec = P.build_model_from_manual_spec(df=df, manual_spec=ms, id_col="SITE",
                                                y_col="Y", offset_col=None, R=50)
    spec = P.replace(spec, model="poisson")
    p = jnp.array([0.4, 0.2, -0.5])
    total = -float(P.mixed_model_loglik(p, data, spec))
    per_site = np.asarray(P.mixed_model_loglik(p, data, spec, indivi=True)).sum()
    # Group-level integration differs from summing per-site marginals.
    assert not np.isclose(total, per_site, atol=1e-3)
    # ...and equals the explicit group-level simulator.
    from metacountregressor.main_hpc import build_eta, ensure_3d, poisson_loglik
    import jax
    eta = build_eta(p, data, spec)
    ll = poisson_loglik(ensure_3d(data["y"]), jnp.exp(jnp.clip(eta, -25, 25)))
    ll_nr = jnp.sum(ll * ensure_3d(data["mask"]), axis=1)
    g = jax.ops.segment_sum(ll_nr, data["group_ids"], num_segments=data["draws_g"].shape[0])
    expect = float(jnp.sum(jax.scipy.special.logsumexp(g, axis=-1) - jnp.log(g.shape[-1])))
    assert total == pytest.approx(expect, rel=1e-6)


def test_nested_random_terms_validation():
    df = pd.DataFrame(dict(ID=range(20), G=np.repeat(range(4), 5), X=np.linspace(0, 1, 20),
                           Y=np.arange(20) % 3))
    with pytest.raises(ValueError, match="grouped_terms"):
        P._validate_role_exclusivity(_manual(nested_rdm_terms=["X:normal"]))
    with pytest.raises(ValueError, match="normal"):
        P._validate_role_exclusivity(_manual(grouped_terms=["X:normal"],
                                             nested_rdm_terms=["X:lognormal"]))
    P._validate_role_exclusivity(_manual(grouped_terms=["X:normal"],
                                         nested_rdm_terms=["X:normal"]))
    data, spec = P.build_model_from_manual_spec(
        df=df, manual_spec=_manual(grouped_terms=["X:normal"], nested_rdm_terms=["X:normal"],
                                   group_id_col="G"),
        id_col="ID", y_col="Y", offset_col=None, R=8)
    assert spec.ind_zero_mean == (True,)


# ---------------------------------------------------------------------------
# Pavement functional forms / hazard
# ---------------------------------------------------------------------------
def _pave_data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    age = rng.uniform(1, 30, n)
    aadt = rng.uniform(200, 8000, n)
    return rng, age, aadt, np.column_stack([age, aadt])


def test_every_pavement_form_fits():
    from metacountregressor.pavement_forms import fit_form, N_FORMS
    from metacountregressor.pavement_gp import tree_to_lambda
    rng, age, aadt, X = _pave_data()
    y = 2 - 0.3 * np.log(age) + rng.normal(0, .05, len(age))
    gp = tree_to_lambda(("log", ("var", 0)), 2)
    for f in range(N_FORMS):
        out = fit_form(X, y, ["age", "aadt"], f, np.array([True, True]),
                       gp_callable=gp if f == 11 else None)
        assert np.isfinite(out["rss"]) and out["design"].shape[0] == len(y)


def test_interaction_forms_add_interaction_columns():
    from metacountregressor.pavement_forms import fit_form
    rng, age, aadt, X = _pave_data()
    y = 2 - 0.3 * np.log(age) + 0.1 * np.log(age) * np.log(aadt) + rng.normal(0, .05, len(age))
    m = np.array([True, True])
    r0 = fit_form(X, y, ["age", "aadt"], 0, m)
    for f in (5, 9):
        rf = fit_form(X, y, ["age", "aadt"], f, m)
        assert rf["design"].shape[1] == r0["design"].shape[1] + 1
        assert rf["rss"] < 0.7 * r0["rss"]


@pytest.mark.parametrize("form,truth,key", [
    (7, {"sigma0": 15.0, "sigma1": 2.0}, "sigma0"),
    (8, {"kappa1": 12.0, "kappa2": 22.0}, "kappa1"),
    (10, {"alpha": 0.4}, "alpha"),
])
def test_shape_forms_recover_parameters_on_real_age_scale(form, truth, key):
    from metacountregressor.pavement_forms import fit_form
    rng, age, aadt, X = _pave_data()
    if form == 7:
        y = 4 - 2 / (1 + np.exp(-(age - 15) / 2))
    elif form == 8:
        y = 4 - 0.1 * np.maximum(age - 12, 0) - 0.2 * np.maximum(age - 22, 0)
    else:
        y = 4 - 2 * (1 - np.exp(-0.4 * age))
    y = y + rng.normal(0, .05, len(age))
    out = fit_form(X, y, ["age", "aadt"], form, np.array([True, True]))
    assert out["shape"][key] == pytest.approx(truth[key], rel=0.1)


def test_gp_operators():
    from metacountregressor.pavement_gp import tree_to_lambda
    X = np.ones((3, 2))
    assert tree_to_lambda(("div", ("const", 6.0), ("const", 3.0)), 2)(X, None)[0, 0] == pytest.approx(2.0)
    assert tree_to_lambda(("pow", ("const", 2.0), ("const", 3.0)), 2)(X, None)[0, 0] == pytest.approx(8.0)


@pytest.mark.parametrize("family", ["exponential", "weibull", "gompertz", "loglogistic"])
def test_hazard_equals_minus_dlogS_dt(family):
    from metacountregressor.pavement_hazard import _hazard_from, _survival_from, _neg_log_likelihood
    Xd = np.array([[0.3, -0.2]])
    t = np.array([2.5])
    params = (np.array([0.7, 0.5, -0.3]) if family == "exponential"
              else np.array([0.7, 0.4, 0.5, -0.3]))
    eps = 1e-6
    num = -(np.log(_survival_from(params, family, t + eps, Xd))
            - np.log(_survival_from(params, family, t - eps, Xd))) / (2 * eps)
    h = _hazard_from(params, family, t, Xd)
    assert h[0] == pytest.approx(num[0], rel=1e-5)
    ll = -_neg_log_likelihood(params, family, t, np.array([1]), Xd)
    assert ll == pytest.approx(np.log(h[0]) + np.log(_survival_from(params, family, t, Xd)[0]), abs=1e-6)


# ---------------------------------------------------------------------------
# SA perturbation operators honour allowed roles and bans
# ---------------------------------------------------------------------------
def test_neighbour_generator_respects_allowed_roles_and_bans():
    from metacountregressor import ExperimentBuilder, load_example16_3_model_data, ModelConstraints
    from metacountregressor.Solvers_METAJAX import AdvancedSimulatedAnnealing
    df = load_example16_3_model_data()
    b = ExperimentBuilder(df=df, id_col="ID", y_col="FREQ", offset_col="OFFSET")
    vars_ = ["URB", "ACCESS", "GRADEBR", "CURVES", "LENGTH", "FC_ENCODED", "SPEED", "AVESNOW"]
    ev = b.build_evaluator(variables=vars_, constraints=ModelConstraints().force_fixed("URB"),
                           max_latent_classes=3, R=10, fixed_override={"ACCESS": [0, 1]},
                           membership_override={"FC_ENCODED": [0, 7]})
    ev._banned_variables = {"AVESNOW"}
    D = len(ev.vars)
    sa = AdvancedSimulatedAnnealing(ev, dimension=4 * D + 2, max_iter=1)
    sa.T0 = 1.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    for trial in range(600):
        sol = np.zeros(4 * D + 2, int)
        for j in range(D):
            sol[j] = sa.sample_allowed_role(j)
        sol[D:2 * D] = rng.integers(0, 5, D)
        sol[2 * D + 1] = rng.integers(0, 3)
        if trial % 3 == 0:
            sol[:D] = 0
        nb = sa.generate_neighbor(sol, T=rng.uniform(0.01, 1.0))
        assert not np.array_equal(nb, sol)
        for j, v in enumerate(ev.vars):
            assert int(nb[j]) in ev.allowed_roles[v], (v, int(nb[j]))
        assert nb[ev.vars.index("AVESNOW")] == 0
