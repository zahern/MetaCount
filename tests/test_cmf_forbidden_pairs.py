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
    correlation_report,
    derive_forbidden_pairs,
    derive_indicator_pairs,
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


def _speed_frame():
    """A continuous speed limit with dummies that encode it.

    ``HSP`` is a threshold on ``SP``; ``MSP`` is an interior band.  ``RSHS`` is
    the trap: it *looks* like a speed dummy but is a conjunction with a road
    type, so it is not reproducible from ``SP`` alone.
    """
    sp = np.array([40, 50, 60, 70, 80, 90, 100, 110] * 5, float)
    rural = np.tile([0.0, 1.0, 0.0, 1.0, 0.0], 8)[:sp.size]
    return pd.DataFrame({
        "SP": sp,
        "HSP": (sp >= 100).astype(float),
        "MSP": ((sp >= 50) & (sp < 100)).astype(float),
        "Rural": rural,
        "RSHS": ((rural > 0) & (sp >= 100)).astype(float),
    })


def test_a_threshold_dummy_is_matched_to_its_continuous_parent():
    df = _speed_frame()
    pairs = derive_indicator_pairs(df, ["SP", "HSP", "MSP"])
    hsp = [p for p in pairs if p["a"] == "HSP"]
    assert len(hsp) == 1
    # a is the redundant term, b its parent - same convention as nesting.
    assert hsp[0]["b"] == "SP"
    assert hsp[0]["reason"] == "encoded"
    assert "SP >= 100" in hsp[0]["detail"]


def test_an_interior_band_dummy_is_matched_too():
    # MSP is not a tail, so a threshold-only test would miss it.
    pairs = derive_indicator_pairs(_speed_frame(), ["SP", "MSP"])
    assert len(pairs) == 1 and pairs[0]["a"] == "MSP"
    assert "50 <= SP <= 90" in pairs[0]["detail"]


def test_a_conjunction_dummy_is_not_treated_as_a_re_encoding():
    # RSHS = rural & high speed: high speed does not make RSHS=1, so the pair
    # is a genuine interaction and must stay selectable.
    df = _speed_frame()
    assert [p for p in derive_indicator_pairs(df, ["SP", "RSHS"])] == []
    # ...and it is not swallowed by the binary-binary rules either.
    assert [p for p in derive_forbidden_pairs(df, ["SP", "RSHS", "HSP"])
            if {p["a"], p["b"]} == {"SP", "RSHS"}] == []


def test_a_block_on_a_few_levels_is_ignored_as_coincidence():
    # Four lanes values offer only a handful of possible blocks, so a dummy on
    # one of them is as likely to be chance as design.
    lanes = np.array([1.0, 2.0, 3.0, 4.0] * 10)
    df = pd.DataFrame({"Nlanes": lanes, "LanesOne": (lanes == 1).astype(float)})
    assert derive_indicator_pairs(df, ["Nlanes", "LanesOne"]) == []


def test_a_tie_straddling_the_cut_rules_out_re_encoding():
    # b alternates within every value of x, so b is not a function of x: it is
    # a correlate, not a re-encoding, and only the opt-in rule should touch it.
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0,
                  5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0])
    b = np.array([0.0, 1.0] * 8)
    df = pd.DataFrame({"X": x, "B": b})
    assert derive_indicator_pairs(df, ["X", "B"]) == []
    assert [p["reason"] for p in
            derive_forbidden_pairs(df, ["X", "B"], corr_threshold=0.0)] == ["correlated"]


def test_the_dummy_and_its_parent_cannot_both_be_selected():
    df = _speed_frame()
    pairs = derive_forbidden_pairs(df, ["SP", "HSP", "MSP", "Rural"])
    encoded = {frozenset((p["a"], p["b"])) for p in pairs
               if p["reason"] == "encoded"}
    assert frozenset(("HSP", "SP")) in encoded
    assert find_violation({"upper": ["HSP"], "lower": ["SP"]}, pairs) is not None
    assert find_violation({"upper": ["HSP"], "lower": ["Rural"]}, pairs) is None


def test_correlation_is_reported_worst_first_without_forbidding():
    rng = np.random.default_rng(3)
    n = 60
    width = rng.uniform(2.0, 12.0, n)
    df = pd.DataFrame({"WIDTH": width,
                       "NEAR": width * 2.0 + 1e-9,
                       "MID": width * 0.9 + rng.normal(0, 1.0, n),
                       "NOISE": rng.normal(0, 1.0, n)})
    report = correlation_report(df, list(df.columns), threshold=0.8)
    assert report[0]["r"] > 0.999 and {report[0]["a"], report[0]["b"]} == {"WIDTH", "NEAR"}
    assert all(abs(p["r"]) >= 0.8 for p in report)
    assert "NOISE" not in {p["a"] for p in report} | {p["b"] for p in report}
    # Reporting is not forbidding: the default derivation leaves them selectable.
    assert derive_forbidden_pairs(df, ["WIDTH", "NEAR"]) == []


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
