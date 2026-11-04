"""Terms taken not to scale with AADT may only sit in the upper tier.

The lower tier exists to let a covariate change the AADT elasticity:
``exp(alpha0 + sum alpha_k x_k) * AADT**(beta0 + sum beta_k x_k)``.  Putting a
term there is a claim that its effect grows with traffic, which is not plausible
for fixed geometry or a roadside feature — so those terms belong in the upper
tier, where the coefficient is a plain CMF.

These tests pin the mechanism (the tier gene is fixed, and a hand-written layout
cannot bypass it), the loud failure when a recorded specification violates the
declaration, and the asymmetry of the data screen: it forces upper only when the
data *prefers* upper, because on a small crash count a failed-to-detect
interaction is not evidence of absence.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import metacountregressor.cmf_agds_search as S
from metacountregressor.cmf_agds_search import (
    CMFAGDSObjective,
    refit_and_report,
    tier_screen,
)


def _frame(n=30):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"GEO": (rng.random(n) < 0.5).astype(float),
                         "TRAFFIC": rng.uniform(0.0, 1.0, n)})


def test_the_tier_gene_is_fixed_for_an_upper_only_term():
    obj = CMFAGDSObjective(_frame(), ["GEO", "TRAFFIC"], max_terms=2,
                           min_lower=0, R=4, upper_only=["GEO"])
    K = obj.K
    assert obj._discrete_values[K] == [1]          # GEO: upper only
    assert obj._discrete_values[K + 1] == [1, 2]   # TRAFFIC: either tier


def test_a_hand_written_layout_cannot_put_an_upper_only_term_in_the_lower_tier():
    obj = CMFAGDSObjective(_frame(), ["GEO", "TRAFFIC"], max_terms=2,
                           min_lower=0, R=4, upper_only=["GEO"])
    # inc, tier (GEO asked for 2 = lower, TRAFFIC for 2 = lower), dist, model, rho
    layout = [1, 1,  2, 2,  0, 0,  1, 0]
    spec = obj.spec(layout)
    assert spec["upper"] == ["GEO"]      # coerced back to upper
    assert spec["lower"] == ["TRAFFIC"]


def test_an_unknown_upper_only_name_is_rejected():
    with pytest.raises(ValueError, match="not among the candidates"):
        CMFAGDSObjective(_frame(), ["GEO"], max_terms=1, min_lower=0, R=4,
                         upper_only=["NOT_A_CANDIDATE"])


def test_refit_refuses_an_upper_only_term_in_the_lower_tier():
    spec = {"upper": [], "lower": ["GEO"], "rand_upper": [], "rand_lower": [False],
            "dist_upper": [], "dist_lower": ["normal"], "model": "poisson",
            "rho": 0.0}
    with pytest.raises(ValueError, match="upper-only"):
        refit_and_report(_frame(), spec, R=2, upper_only=["GEO"])


def _fake_refit(bics):
    def _fit(data, spec, R=20, integration="halton", lrt=False, **kwargs):
        name = (spec["upper"] or spec["lower"])[0]
        where = "upper" if spec["upper"] else "lower"
        value = bics[name][where]
        if value is None:
            raise RuntimeError("no fit")
        return SimpleNamespace(bic=float(value))
    return _fit


def test_the_screen_forces_upper_only_when_upper_is_clearly_better(monkeypatch):
    monkeypatch.setattr(S, "refit_and_report", _fake_refit({
        "GEO": {"upper": 100.0, "lower": 130.0},      # upper better by 30
        "TRAFFIC": {"upper": 110.0, "lower": 111.0},  # a wash
    }))
    out = tier_screen(_frame(), ["GEO", "TRAFFIC"], R=4)
    assert out["GEO"]["verdict"] == "upper"
    assert out["GEO"]["delta"] == pytest.approx(-30.0)
    # A near-tie stays eligible for either tier: low power must not be read as
    # evidence that the effect is AADT-independent.
    assert out["TRAFFIC"]["verdict"] == "either"


def test_a_failed_screen_leaves_the_term_flexible(monkeypatch):
    monkeypatch.setattr(S, "refit_and_report", _fake_refit({
        "GEO": {"upper": None, "lower": 120.0},
    }))
    out = tier_screen(_frame(), ["GEO"], R=4)
    assert out["GEO"]["verdict"] == "failed"
    assert not np.isfinite(out["GEO"]["delta"])
