"""Does a term's effect vary with AADT?  The nested version of that question.

``tier_screen`` asks where a term fits better; this screen asks whether it varies
at all, by nesting the upper-only fit inside one where the same term also enters
the lower tier.  The extra parameter is exactly the AADT interaction, so one
likelihood-ratio test answers it.

The distinction these tests protect is ``varying`` vs ``weak``: with a large n
the BIC penalty per parameter (~log n, i.e. 7.5 on 1,848 rows) is far stricter
than p < 0.05, so a term can be nominally significant and still not earn its
place.  Collapsing those two into "significant" would overstate the evidence.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import metacountregressor.cmf_agds_search as S
from metacountregressor.cmf_agds_search import aadt_variation_screen


def _frame(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "BIG": rng.uniform(0, 1, n),
        "SMALL": rng.uniform(0, 1, n),
        "FLAT": rng.uniform(0, 1, n),
        "RARE": np.r_[np.ones(4), np.zeros(n - 4)],   # 4 positives: too few
        "CONST": np.full(n, 1.0),
    })


def _fake(bics, lls):
    """refit_and_report stand-in keyed on the variable and whether it is nested."""
    def _fit(data, spec, R=20, integration="halton", lrt=False, **kwargs):
        name = (spec["upper"] or spec["lower"])[0]
        nested = bool(spec["lower"])
        if lls[name].get("raise"):
            raise RuntimeError("no fit")
        return SimpleNamespace(ll=lls[name]["both" if nested else "upper"],
                               bic=bics[name]["both" if nested else "upper"])
    return _fit


def test_a_significant_interaction_that_earns_its_bic_is_varying(monkeypatch):
    monkeypatch.setattr(S, "refit_and_report", _fake(
        bics={"BIG": {"upper": 100.0, "both": 90.0}},
        lls={"BIG": {"upper": -100.0, "both": -95.0}}))   # LR = 10
    out = aadt_variation_screen(_frame(), ["BIG"], R=4)
    assert out["BIG"]["verdict"] == "varying"
    assert out["BIG"]["lr"] == pytest.approx(10.0)
    assert out["BIG"]["p"] < 0.05
    assert out["BIG"]["delta_bic"] < 0


def test_a_significant_but_bic_negative_interaction_is_only_weak(monkeypatch):
    # LR = 5 clears p < 0.05 (3.84) but not the BIC penalty: the term is
    # nominally significant and still not worth a parameter.
    monkeypatch.setattr(S, "refit_and_report", _fake(
        bics={"SMALL": {"upper": 100.0, "both": 101.0}},
        lls={"SMALL": {"upper": -100.0, "both": -97.5}}))
    out = aadt_variation_screen(_frame(), ["SMALL"], R=4)
    assert out["SMALL"]["p"] < 0.05
    assert out["SMALL"]["verdict"] == "weak"


def test_an_insignificant_interaction_is_none(monkeypatch):
    monkeypatch.setattr(S, "refit_and_report", _fake(
        bics={"FLAT": {"upper": 100.0, "both": 107.0}},
        lls={"FLAT": {"upper": -100.0, "both": -99.95}}))   # LR = 0.1
    out = aadt_variation_screen(_frame(), ["FLAT"], R=4)
    assert out["FLAT"]["verdict"] == "none"


def test_a_term_with_too_few_minority_rows_is_not_tested(monkeypatch):
    # RSLS on Queensland has 2 rows: a likelihood ratio there measures nothing,
    # and reporting it as "none" would read as evidence of no variation.
    called = []
    monkeypatch.setattr(S, "refit_and_report",
                        lambda *a, **k: called.append(1) or None)
    out = aadt_variation_screen(_frame(), ["RARE", "CONST"], R=4)
    assert out["RARE"]["verdict"] == "unsupported"
    assert "minority" in out["RARE"]["reason"]
    assert out["CONST"]["verdict"] == "unsupported"
    assert called == [], "the screen must not spend a fit on these"


def test_a_failed_fit_is_reported_rather_than_guessed(monkeypatch):
    monkeypatch.setattr(S, "refit_and_report", _fake(
        bics={"BIG": {"upper": 0.0, "both": 0.0}},
        lls={"BIG": {"raise": True}}))
    out = aadt_variation_screen(_frame(), ["BIG"], R=4)
    assert out["BIG"]["verdict"] == "failed"
