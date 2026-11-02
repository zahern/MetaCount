"""Nested candidate terms must not be co-selected in the CMF search.

If every row with ``A = 1`` also has ``B = 1``, then ``A``'s effect is already
inside ``B``'s: the two indicators span the same rows and the coefficient on
``A`` has nothing of its own left to explain.  A search that is free to pick
both will happily do so — the pair often fits marginally better than either
alone — and report a CMF for a term that is not separately identified.

The exclusions are derived from the design rather than named in a list, so a
renamed or recomputed covariate cannot carry a stale exclusion with it.  These
tests pin the derivation (nesting, duplicates, and the cases that must *not*
trigger), the violation test across tiers, and that the objective rejects a
violating layout before spending a fit on it.
"""

import numpy as np
import pandas as pd
import pytest

from metacountregressor.cmf_agds_search import (
    CMFAGDSObjective,
    derive_forbidden_pairs,
    find_violation,
    normalise_forbidden,
)


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    curve = (rng.random(n) < 0.6).astype(float)          # broad: "any curve"
    curve50 = np.where(rng.random(n) < 0.2, 1.0, 0.0) * curve   # a strict subset
    width = rng.uniform(2.0, 12.0, n)                    # continuous
    lanes = (rng.random(n) < 0.5).astype(float)
    always = np.ones(n)                                  # carries no information
    same = curve.copy()                                  # exact duplicate
    return pd.DataFrame({"Curve": curve, "Curve50": curve50, "WIDTH": width,
                         "Lanes": lanes, "ALWAYS": always, "DUP": same})


def test_nesting_is_found_and_oriented():
    pairs = derive_forbidden_pairs(_frame(), ["Curve", "Curve50", "WIDTH", "Lanes"])
    nested = [p for p in pairs if p["reason"] == "nested"]
    assert len(nested) == 1
    # Curve50 is nested in Curve: every Curve50 row also has Curve.
    assert (nested[0]["a"], nested[0]["b"]) == ("Curve50", "Curve")
    assert "nested in" in nested[0]["detail"]


def test_continuous_and_always_on_columns_never_nest():
    # A positive continuous column would swallow every indicator under a naive
    # "subset of the positive support" test; a column that is always 1 likewise.
    pairs = derive_forbidden_pairs(_frame(), ["WIDTH", "Curve", "Lanes", "ALWAYS"])
    assert [p for p in pairs if p["reason"] == "nested"] == []


def test_a_duplicated_indicator_is_reported_as_a_duplicate():
    pairs = derive_forbidden_pairs(_frame(), ["Curve", "DUP"])
    dup = [p for p in pairs if p["reason"] == "duplicate"]
    assert len(dup) == 1
    assert {dup[0]["a"], dup[0]["b"]} == {"Curve", "DUP"}


def test_disjoint_indicators_are_left_alone():
    # Curve and Lanes overlap without containing each other: both are legitimate.
    pairs = derive_forbidden_pairs(_frame(), ["Curve", "Lanes"])
    assert pairs == []


def test_correlation_rule_is_opt_in():
    df = _frame()
    df["NEAR"] = df["WIDTH"] * 2.0 + 1e-9
    assert derive_forbidden_pairs(df, ["WIDTH", "NEAR"]) == []
    pairs = derive_forbidden_pairs(df, ["WIDTH", "NEAR"], corr_threshold=0.95)
    assert [p["reason"] for p in pairs] == ["correlated"]


def test_a_violating_pair_is_found_across_tiers():
    # Unidentifiable is unidentifiable whichever tier each term sits in.
    pairs = [{"a": "Curve50", "b": "Curve", "reason": "nested", "detail": ""}]
    spec = {"upper": ["Curve"], "lower": ["Curve50"]}
    assert find_violation(spec, pairs) == ("Curve", "Curve50")
    assert find_violation({"upper": ["Curve"], "lower": ["Lanes"]}, pairs) is None
    assert find_violation(spec, []) is None


def test_pairs_may_be_given_as_dicts_or_tuples():
    assert normalise_forbidden([{"a": "A", "b": "B"}]) == [frozenset({"A", "B"})]
    assert normalise_forbidden([("A", "B")]) == [frozenset({"A", "B"})]
    with pytest.raises(ValueError):
        normalise_forbidden([("A",)])


def test_the_objective_rejects_a_violating_layout_before_fitting(monkeypatch):
    from metacountregressor.cmf_agds_search import G

    # get_fitness swallows exceptions from the fit (a failed trial is just a
    # bad candidate), so the stub records calls rather than raising.
    calls = []

    def _record(*a, **k):
        calls.append(1)
        raise RuntimeError("not a real fit")

    monkeypatch.setattr(G, "evaluate_model", _record)
    df = _frame()
    pairs = derive_forbidden_pairs(df, ["Curve", "Curve50"])
    assert pairs, "fixture must contain a nested pair"
    obj = CMFAGDSObjective(df, ["Curve", "Curve50"], max_terms=4, min_lower=0,
                           R=8, forbid_together=pairs)

    # Layout is inc[K] + tier[K] + dist[K] + model + rho for K=2.  Both terms
    # in, one per tier: the violating layout.
    layout = [1, 1,  1, 2,  0, 0,  1, 0]
    spec = obj.spec(layout)
    assert set(spec["upper"] + spec["lower"]) == {"Curve", "Curve50"}
    assert find_violation(spec, pairs) is not None
    assert obj.get_fitness(layout)["bic"] == 1e12
    assert calls == [], "the guard must fire before any fit is attempted"
    # ...and a single-term layout is still evaluated, i.e. the guard did not
    # simply break every candidate.
    obj.get_fitness([1, 0,  1, 1,  0, 0,  1, 0])
    assert calls == [1]
