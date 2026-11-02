"""Stacking CMFs: the multiplication caveats, enforced rather than footnoted.

HSM/CMF-Clearinghouse practice combines countermeasures at a location by
multiplying their factors.  That presumes the treatments act independently,
which is tenable for two and unreliable beyond, and a large combined reduction
is the symptom of overlapping effects being counted twice.  ``combine_cmfs``
returns the product together with those review flags, and accepts the
covariance of the log CMFs so a model-based stack does not have to assume
independence at all.
"""

import numpy as np
import pandas as pd
import pytest

from metacountregressor import GA_CMF_AADT_JAX as G
from metacountregressor.cmf_catalogue import (
    COMBINE_REDUCTION_WARN,
    COMBINE_TERMS_WARN,
    build_cmf_catalogue,
    combine_cmfs,
    derive_cmf,
)


def test_two_cmfs_multiply():
    out = combine_cmfs([0.82, 0.89])
    assert out["cmf"] == pytest.approx(0.82 * 0.89)
    assert out["reduction"] == pytest.approx(1 - 0.82 * 0.89)
    assert out["n_terms"] == 2
    assert out["scrutinise"] is False
    assert out["notes"] == []
    # No variances supplied: the product is exact but has no interval.
    assert np.isnan(out["se_log_cmf"]) and np.isnan(out["cmf_lo"])


def test_three_or_more_flags_independence_caveat():
    out = combine_cmfs([0.9, 0.9, 0.9])
    assert out["n_terms"] == 3 == COMBINE_TERMS_WARN
    assert out["scrutinise"] is True
    assert any("independent effects" in n for n in out["notes"])


def test_large_reduction_is_flagged():
    # 0.5 * 0.8 = 0.40 -> a 60 % reduction, exactly the review threshold.
    out = combine_cmfs([0.5, 0.8])
    assert out["cmf"] == pytest.approx(0.40, abs=1e-12)
    assert out["reduction"] >= COMBINE_REDUCTION_WARN
    assert out["scrutinise"] is True
    assert any("60%" in n for n in out["notes"])
    # Just under the threshold, with only two factors: nothing to review.
    under = combine_cmfs([0.5, 0.81])
    assert under["scrutinise"] is False and under["notes"] == []


def test_variance_sources_are_distinguished():
    # Independent variances: Var(log) is the sum of the squares.
    ind = combine_cmfs([0.8, 0.9], se_log=[0.05, 0.05])
    assert ind["assumed_independence"] is True
    assert ind["se_log_cmf"] == pytest.approx(np.hypot(0.05, 0.05))
    assert any("independent" in n for n in ind["notes"])

    # The same fit's covariance carries the positive correlation, so the
    # interval is wider than the independence assumption would claim.
    cov = np.array([[0.05 ** 2, 0.8 * 0.05 ** 2],
                    [0.8 * 0.05 ** 2, 0.05 ** 2]])
    cor = combine_cmfs([0.8, 0.9], cov=cov)
    assert cor["assumed_independence"] is False
    assert cor["se_log_cmf"] > ind["se_log_cmf"]
    assert cor["cmf_hi"] > ind["cmf_hi"]
    assert not any("independent" in n for n in cor["notes"])
    # The point estimate is unaffected by how the variance is obtained.
    assert cor["cmf"] == pytest.approx(ind["cmf"])

    # Negative covariance (redundant treatments) narrows it instead.
    neg = combine_cmfs([0.8, 0.9], cov=np.array([[0.0025, -0.002],
                                                [-0.002, 0.0025]]))
    assert neg["se_log_cmf"] < ind["se_log_cmf"]


def test_same_mechanism_is_flagged_as_double_counting():
    out = combine_cmfs([0.8, 0.85], mechanisms=["rear-end", "rear-end"])
    assert out["scrutinise"] is True
    assert any("same crash mechanism" in n and "rear-end" in n
               for n in out["notes"])
    distinct = combine_cmfs([0.8, 0.85], mechanisms=["rear-end", "pedestrian"])
    assert not any("same crash mechanism" in n for n in distinct["notes"])


@pytest.mark.parametrize("cmfs", [[], [0.0], [-0.5], [0.8, np.nan],
                                  [0.8, np.inf]])
def test_invalid_cmfs_raise(cmfs):
    with pytest.raises(ValueError):
        combine_cmfs(cmfs)


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        combine_cmfs([0.8, 0.9], se_log=[0.05])          # one SE for two CMFs
    with pytest.raises(ValueError):
        combine_cmfs([0.8, 0.9], cov=np.eye(3))          # wrong size
    with pytest.raises(ValueError):
        combine_cmfs([0.8, 0.9], mechanisms=["rear-end"])  # wrong length


# ── the model's own stack ───────────────────────────────────────────────────

def _synthetic(vars_, alphas, se=0.01):
    """Params/cov for an upper-tier-only fit, no estimation needed."""
    labels = G._param_labels(list(vars_), [], [False] * len(vars_), [],
                             "poisson")
    params = np.zeros(len(labels))
    for v, a in zip(vars_, alphas):
        params[labels.index(f"alpha[{v}]")] = np.log(a)
    cov = np.eye(len(labels)) * se ** 2
    return params, cov, labels


def test_derive_cmf_single_change_is_not_flagged():
    params, cov, _ = _synthetic(["A"], [0.8])
    out = derive_cmf(params, cov, ["A"], [], [False], [False],
                     changes={"A": 1})
    assert out["cmf"] == pytest.approx(0.8, rel=1e-9)
    assert out["n_terms"] == 1
    assert out["scrutinise"] is False and out["notes"] == []


def test_derive_cmf_three_terms_flags_and_respects_covariance():
    params, cov, labels = _synthetic(["A", "B", "C"], [0.9, 0.9, 0.9])
    sel = (["A", "B", "C"], [], [False] * 3, [False])
    one = derive_cmf(params, cov, *sel, changes={"A": 1})
    assert one["scrutinise"] is False

    three = derive_cmf(params, cov, *sel, changes={"A": 1, "B": 1, "C": 1})
    assert three["n_terms"] == 3
    assert three["cmf"] == pytest.approx(0.9 ** 3, rel=1e-9)
    assert three["scrutinise"] is True
    assert any("independent effects" in n for n in three["notes"])
    # Delta method: the three independent log-MCF variances sum in quadrature.
    assert three["se_log_cmf"] == pytest.approx(np.sqrt(3) * 0.01, rel=1e-9)
    assert three["se_log_cmf"] > one["se_log_cmf"]

    # A positive covariance between the coefficients widens the model's
    # interval relative to the diagonal-only covariance.
    cor = cov.copy()
    i, j = labels.index("alpha[A]"), labels.index("alpha[B]")
    cor[i, j] = cor[j, i] = 0.8 * 0.01 ** 2
    wider = derive_cmf(params, cor, ["A", "B", "C"], [], [False] * 3, [False],
                       changes={"A": 1, "B": 1, "C": 1})
    assert wider["se_log_cmf"] > three["se_log_cmf"]


def test_derive_cmf_large_reduction_flagged():
    params, cov, _ = _synthetic(["A", "B"], [0.6, 0.6])
    out = derive_cmf(params, cov, ["A", "B"], [], [False, False], [False],
                     changes={"A": 1, "B": 1})
    assert out["cmf"] == pytest.approx(0.36, rel=1e-9)
    assert out["reduction"] >= COMBINE_REDUCTION_WARN
    assert out["scrutinise"] is True
    assert any("60%" in n for n in out["notes"])


# ── the catalogue is the input to a stack, so its terms must be stackable ───

def test_catalogue_rows_combine_through_the_helper():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "AADT": np.linspace(1000, 20000, n),
        "FREQ": rng.poisson(3.0, n).astype(float),
        "X": rng.normal(size=n), "Z": rng.normal(size=n)})
    labels = G._param_labels(["X"], ["Z"], [False], [False], "poisson")
    params = np.zeros(len(labels))
    se = np.full(len(labels), 0.02)
    params[labels.index("alpha[X]")] = np.log(0.85)
    params[labels.index("beta[Z]")] = np.log(0.9)
    cat = build_cmf_catalogue(params, se, df, ["X"], ["Z"], [False], [False])
    x = cat[cat.term == "X"].iloc[0]
    z = cat[cat.term == "Z"].iloc[0]
    # Upper-tier CMFs are stackable directly; lower-tier ones act through AADT,
    # so a naive product would be the wrong operation for them.
    assert x["tier"] == "upper" and z["tier"] == "lower"
    out = combine_cmfs([x["cmf"]], se_log=[x["se"]])
    assert out["cmf"] == pytest.approx(0.85, rel=1e-6)
    assert out["n_terms"] == 1
