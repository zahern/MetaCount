"""build_search(model_family="survival") -- the hazard head's structure search.

Proves the survival family reaches the same metaheuristic SA the count family
uses, that a structure is actually selected, and that the refit returns named
hazard/time ratios off the fit object.  Also pins the two guards: a survival
search without event_col must fail loudly rather than silently dropping the
censoring indicator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from metacountregressor.experiment_package import ExperimentBuilder


def _surv_frame(n=240, seed=5):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    k, alpha = 1.3, -1.0
    e = rng.exponential(size=n)
    t = np.exp((np.log(e) - alpha - (0.5 * x1 - 0.3 * x2)) / k)
    cen = 6.0
    return pd.DataFrame({
        "ID": [f"i{i}" for i in range(n)],
        "grp": rng.integers(0, 3, n),
        "dur": np.minimum(t, cen),
        "evt": (t <= cen).astype(int),
        "x1": x1,
        "x2": x2,
    })


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_survival_search_builds_an_evaluator_and_declares_its_ratio():
    df = _surv_frame()
    b = ExperimentBuilder(df, "ID", "dur", group_id_col="grp", event_col="evt")
    ev = b.build_search(variables=["x1", "x2"], model_family="survival",
                        family="weibull_ph", R=6, default_roles=[0, 1, 2, 3])
    assert ev.metadata["model"] == "weibull_ph"
    assert ev.metadata["ratio"] == "hazard_ratio"
    assert ev.metadata["event_col"] == "evt"
    # it must own a real evaluator, not be a bare shell
    assert hasattr(ev, "evaluator") and hasattr(ev.evaluator, "build_spec")


def test_aft_survival_search_declares_time_ratio():
    df = _surv_frame()
    b = ExperimentBuilder(df, "ID", "dur", group_id_col="grp", event_col="evt")
    ev = b.build_search(variables=["x1", "x2"], model_family="survival",
                        family="weibull", R=6, default_roles=[0, 1])
    assert ev.metadata["ratio"] == "time_ratio"


def test_survival_search_selects_a_structure():
    df = _surv_frame()
    b = ExperimentBuilder(df, "ID", "dur", group_id_col="grp", event_col="evt")
    ev = b.build_search(variables=["x1", "x2"], model_family="survival",
                        family="weibull_ph", R=6, default_roles=[0, 1, 2, 3])
    res = b.run(ev, algo="sa", max_iter=3, seed=3)
    spec = res.get("best_spec") or res.get("model_spec")
    assert spec is not None
    assert spec["model"] == "weibull_ph"
    assert spec["dispersion"] == 0
    assert not spec.get("zi_terms")
    assert np.isfinite(float(res["best_score"]))


def test_survival_refit_returns_named_ratios_with_standard_errors():
    df = _surv_frame()
    b = ExperimentBuilder(df, "ID", "dur", group_id_col="grp", event_col="evt")
    ev = b.build_search(variables=["x1", "x2"], model_family="survival",
                        family="weibull_ph", R=6, default_roles=[0, 1, 2, 3])
    res = b.run(ev, algo="sa", max_iter=3, seed=3)
    spec = res.get("best_spec") or res.get("model_spec")
    fit = b.fit_manual_model(manual_spec=spec, model="weibull_ph", R=8)
    assert fit["survival_family"] == "weibull_ph"
    assert fit["ratio_kind"] == "hazard_ratio"
    # named coefficients must come off the fit object, not a scraped TXT
    assert fit["coef_source" if "coef_source" in fit else "params"]
    assert len(fit["params"]) > 0
    assert "x1" in fit["params"] or "x2" in fit["params"]
    assert fit["bse"].get("x1") is not None or fit["bse"].get("x2") is not None
    assert "hazard_ratio" in fit["coef_table"].columns
    assert isinstance(fit["summary"], dict) and "bic" in fit["summary"]


def test_survival_search_requires_event_col():
    df = _surv_frame()
    b = ExperimentBuilder(df, "ID", "dur", group_id_col="grp")  # no event_col
    with pytest.raises(ValueError, match="event_col"):
        b.build_search(variables=["x1", "x2"], model_family="survival", R=6)


def test_survival_search_rejects_unknown_family():
    df = _surv_frame()
    b = ExperimentBuilder(df, "ID", "dur", group_id_col="grp", event_col="evt")
    with pytest.raises(ValueError):
        b.build_search(variables=["x1", "x2"], model_family="survival",
                       family="not_a_family", R=6)


def test_count_family_still_works_and_is_unaffected():
    """The survival branch must not disturb the count path."""
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "ID": [f"i{i}" for i in range(n)],
        "grp": rng.integers(0, 2, n),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "Y": rng.poisson(3, n),
    })
    b = ExperimentBuilder(df, "ID", "Y", group_id_col="grp")
    # the SA needs >=2 active variables, so offer two
    ev = b.build_search(variables=["x1", "x2"], model_family="count", R=6,
                        max_latent_classes=1, default_roles=[0, 1, 2])
    assert not hasattr(ev, "metadata")  # count returns a bare evaluator
    assert hasattr(ev, "build_spec")
    res = b.run(ev, algo="sa", max_iter=3, seed=1)
    assert np.isfinite(float(res["best_score"]))
    spec = res.get("best_spec") or res.get("model_spec")
    # the count path carries a dispersion bit rather than a model name
    model = spec.get("model") or ("nb" if spec.get("dispersion") else "poisson")
    assert model in ("poisson", "nb")
    fit = b.fit_manual_model(manual_spec=spec, model=model, R=8)
    # named-coefficient plumbing still works on the count path
    named = fit["coefficient_names"]
    assert any("x1" in nm or "x2" in nm for nm in named)
    assert fit["survival_family"] is None
