"""Perturbation tests for the survival / hazard models.

Each test perturbs the data or the specification and asserts the model
responds the way the mathematics says it must.  Several of these would FAIL
if the censoring indicator were not actually reaching the likelihood, or if
the PH parameterisation were wrong -- so they are integration proofs, not
just unit tests.

  1. shuffled covariate loses its effect (and BIC worsens)
  2. flipping the event flag changes the log-likelihood at fixed params
     (direct proof data["event"] is read)
  3. randomly censoring half the events preserves the signal but grows SEs
  4. multiplying all durations by c leaves k and beta invariant and shifts
     the intercept by exactly k*log(c)  (Weibull-PH identity)
  5. a larger true effect yields a larger estimated hazard ratio
  6. adding a truly predictive covariate improves BIC
  7. the search objective (evaluator fitness) worsens on shuffled data,
     deterministically, without running the stochastic SA
"""

import numpy as np
import pandas as pd
import pytest

from metacountregressor.survival_models import (
    AFTFitter,
    survival_mixed_model_loglik,
)


def _sim(n=400, seed=11, beta=(0.55, -0.40), shape=1.40, alpha=-1.20, censor=6.0):
    """Right-censored Weibull-PH by inverse-CDF sampling."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    lp = beta[0] * x1 + beta[1] * x2
    e = rng.exponential(size=n)
    t = np.exp((np.log(e) - alpha - lp) / shape)
    event = (t <= censor).astype(int)
    return pd.DataFrame({
        "dur": np.minimum(t, censor), "evt": event, "x1": x1, "x2": x2,
    })


def _fit(df, cols=("x1", "x2")):
    return AFTFitter(family="weibull_ph", n_draws=1).fit(
        df, "dur", "evt", list(cols))


# --------------------------------------------------------------------------
# 1. shuffled covariate
# --------------------------------------------------------------------------

def test_shuffled_covariate_loses_its_effect_and_bic_worsens():
    df = _sim()
    base = _fit(df)
    b0 = float(base.summary_.loc["x1", "coef"])
    bic0 = float(base.bic())
    p0 = float(base.summary_.loc["x1", "pvalue"])

    sh = df.copy()
    sh["x1"] = np.random.default_rng(99).permutation(sh["x1"].to_numpy())
    sfl = _fit(sh)
    b1 = float(sfl.summary_.loc["x1", "coef"])
    bic1 = float(sfl.bic())
    p1 = float(sfl.summary_.loc["x1", "pvalue"])

    print(f"\n  x1 coef {b0:+.4f} (p={p0:.2g}) -> shuffled {b1:+.4f} (p={p1:.2g})")
    print(f"  BIC {bic0:.2f} -> {bic1:.2f} (want worse)")
    assert abs(b1) < 0.35 * abs(b0), "shuffled covariate kept its effect"
    assert p1 > 0.05, "shuffled covariate still significant"
    assert bic1 > bic0, "BIC did not worsen on shuffled data"


# --------------------------------------------------------------------------
# 2. the event flag must reach the likelihood
# --------------------------------------------------------------------------

def test_flipping_event_flags_changes_loglik_at_fixed_params():
    """Same params, same durations, different event flags.

    If data["event"] were not plumbed through, the two log-likelihoods
    would be IDENTICAL.  They must differ by a large margin.
    """
    df = _sim()
    fit = _fit(df)
    p = np.asarray(fit._model.params)
    data, spec = fit._data, fit._spec

    from functools import partial
    obj = partial(survival_mixed_model_loglik, data=data, spec=spec,
                  family="weibull_ph")
    ll_true = -float(obj(p))

    flipped = dict(data)
    flipped["event"] = 1.0 - np.asarray(data["event"])
    obj_f = partial(survival_mixed_model_loglik, data=flipped, spec=spec,
                    family="weibull_ph")
    ll_flip = -float(obj_f(p))

    print(f"\n  LL(events as observed) = {ll_true:.3f}")
    print(f"  LL(events flipped)      = {ll_flip:.3f}")
    assert abs(ll_true - ll_flip) > 50.0, (
        "flipping the event flag barely moved the likelihood -- "
        "the censoring indicator is not reaching it"
    )


# --------------------------------------------------------------------------
# 3. partial censoring: signal preserved, uncertainty grows
# --------------------------------------------------------------------------

def test_randomly_censoring_half_the_events_grows_standard_errors():
    df = _sim()
    base = _fit(df)
    b0 = float(base.summary_.loc["x1", "coef"])
    se0 = float(base.summary_.loc["x1", "stderr"])

    rng = np.random.default_rng(7)
    cens = df.copy()
    ev_idx = cens.index[cens["evt"] == 1].to_numpy()
    drop = rng.choice(ev_idx, size=len(ev_idx) // 2, replace=False)
    cens.loc[drop, "evt"] = 0
    assert cens["evt"].sum() < df["evt"].sum()

    cfit = _fit(cens)
    b1 = float(cfit.summary_.loc["x1", "coef"])
    se1 = float(cfit.summary_.loc["x1", "stderr"])

    print(f"\n  coef {b0:+.4f} -> {b1:+.4f}   (want similar)")
    print(f"  se   {se0:.4f} -> {se1:.4f}   (want larger)")
    assert abs(b1 - b0) < 0.15, "signal moved too far under extra censoring"
    assert se1 > se0, "standard error did not grow under heavier censoring"


# --------------------------------------------------------------------------
# 4. duration scaling identity
# --------------------------------------------------------------------------

def test_scaling_durations_shifts_only_the_intercept():
    """t -> c*t must leave (k, beta) invariant and send
    alpha -> alpha - k*log(c)."""
    df = _sim()
    base = _fit(df)
    a0 = float(base.summary_.loc["__INTERCEPT__", "coef"])
    k0 = float(np.log1p(np.exp(base.summary_.loc["sigma", "coef"])))
    b0 = base.summary_.loc[["x1", "x2"], "coef"].to_numpy()

    c = 10.0
    scl = df.copy()
    scl["dur"] = scl["dur"] * c
    sfit = _fit(scl)
    a1 = float(sfit.summary_.loc["__INTERCEPT__", "coef"])
    k1 = float(np.log1p(np.exp(sfit.summary_.loc["sigma", "coef"])))
    b1 = sfit.summary_.loc[["x1", "x2"], "coef"].to_numpy()

    print(f"\n  k {k0:.4f} -> {k1:.4f} (want equal)")
    print(f"  beta {b0} -> {b1} (want equal)")
    print(f"  alpha {a0:.4f} -> {a1:.4f} (want shift {-(k0 * np.log(c)):.4f})")
    assert k1 == pytest.approx(k0, abs=2e-2)
    assert b1 == pytest.approx(b0, abs=2e-2)
    assert (a1 - a0) == pytest.approx(-k0 * np.log(c), abs=5e-2)


# --------------------------------------------------------------------------
# 5. stronger truth -> larger hazard ratio
# --------------------------------------------------------------------------

def test_larger_true_effect_gives_larger_hazard_ratio():
    weak = _fit(_sim(beta=(0.25, -0.40)))
    strong = _fit(_sim(beta=(0.80, -0.40)))
    hr_w = float(weak.summary_.loc["x1", "coef"])
    hr_s = float(strong.summary_.loc["x1", "coef"])
    print(f"\n  true 0.25 -> est {hr_w:+.4f};  true 0.80 -> est {hr_s:+.4f}")
    assert 0.10 < hr_w < 0.45
    assert 0.60 < hr_s < 1.05
    assert hr_s > 1.7 * hr_w


# --------------------------------------------------------------------------
# 6. a truly predictive covariate improves BIC
# --------------------------------------------------------------------------

def test_adding_a_true_covariate_improves_bic():
    df = _sim()
    full = AFTFitter(family="weibull_ph", n_draws=1).fit(df, "dur", "evt", ["x1", "x2"])
    drop = AFTFitter(family="weibull_ph", n_draws=1).fit(df, "dur", "evt", ["x1"])
    print(f"\n  BIC(x1,x2)={full.bic():.2f}  BIC(x1)={drop.bic():.2f}  (want lower)")
    assert full.bic() < drop.bic() - 5.0


# --------------------------------------------------------------------------
# 7. the search objective itself responds, deterministically
# --------------------------------------------------------------------------

def test_search_objective_worsens_on_shuffled_covariate():
    """Evaluator fitness on a fixed decision vector, original vs shuffled.

    No SA is run, so this is fully deterministic and still proves the
    survival search objective -- the thing the metaheuristic optimises --
    tracks the signal.
    """
    from metacountregressor.experiment_package import ExperimentBuilder

    df = _sim(n=240, seed=5)
    cols = ["x1", "x2"]
    df["ID"] = [f"i{i}" for i in range(len(df))]
    df["grp"] = 0

    b = ExperimentBuilder(df, "ID", "dur", group_id_col="grp", event_col="evt")
    ev = b.build_search(variables=cols, model_family="survival",
                        family="weibull_ph", R=6, default_roles=[0, 1])

    import numpy as _np

    D = len(ev.evaluator.vars)
    vec = _np.zeros(2 * D + 1)
    vec[0] = 1  # x1 fixed
    vec[1] = 1  # x2 fixed
    spec = ev.evaluator.build_spec(vec)
    assert spec is not None and spec["model"] == "weibull_ph"

    s_orig = float(ev.evaluator.fitness(vec))

    b2df = df.copy()
    b2df["x1"] = np.random.default_rng(99).permutation(b2df["x1"].to_numpy())
    b2 = ExperimentBuilder(b2df, "ID", "dur", group_id_col="grp", event_col="evt")
    ev2 = b2.build_search(variables=cols, model_family="survival",
                          family="weibull_ph", R=6, default_roles=[0, 1])
    s_shuf = float(ev2.evaluator.fitness(vec))

    print(f"\n  fitness(original)={s_orig:.2f}  fitness(shuffled)={s_shuf:.2f}  (want worse)")
    assert np.isfinite(s_orig) and np.isfinite(s_shuf)
    assert s_shuf > s_orig, "search objective did not worsen on shuffled data"
