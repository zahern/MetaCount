"""The cross-family rule: which published CMFs survive a change of count family.

A CMF is reported as a property of the road, but it is a ratio of conditional
means from *one* fitted model.  Refitting the same specification under a
different variance function — NB2 vs NB1 vs Poisson — reweights the observations,
and a weakly identified term can move enough to cross 1.0 or lose significance.
`family_robust_flags` decides which terms survive that, and the deliverable's
dashboards mark the ones that do not.

These tests pin the three ways a CMF can fail: it moves, it flips sign, or it
changes its significance verdict.  They also pin the failure that would be worst
in practice — a comparison that silently reports "robust" for a term it never
actually compared.
"""

import numpy as np
import pytest

from metacountregressor.cmf_catalogue import FAMILY_LOG_TOL, family_robust_flags


def _fit(**terms):
    """``_fit(WIDTH=dict(cmf=0.98, t=-3.0))`` -> one family's catalogue."""
    return {k: {"cmf": v["cmf"], "t": v.get("t")} for k, v in terms.items()}


def test_identical_families_are_robust():
    a = _fit(WIDTH=dict(cmf=0.981, t=-3.75), URB=dict(cmf=0.994, t=-2.29))
    flags = family_robust_flags({"nb": a, "nb1": a.copy()}, "nb")
    assert flags == {"WIDTH": True, "URB": True}


def test_a_large_move_is_not_robust():
    base = _fit(MIMEDSH=dict(cmf=0.910, t=-2.52))
    alt = _fit(MIMEDSH=dict(cmf=1.009, t=+0.19))
    assert family_robust_flags({"nb": base, "nb1": alt}, "nb") == {"MIMEDSH": False}


def test_the_tolerance_judges_size_not_direction():
    # 0.5x the tolerance is the same CMF; 1.5x it is not, whichever way it
    # moved.  The boundary itself is not asserted to the last bit: comparing
    # log(exp(tol)) to tol is a float lottery, and >= is the documented side.
    base = _fit(X=dict(cmf=1.0, t=2.5))
    under = _fit(X=dict(cmf=float(np.exp(FAMILY_LOG_TOL * 0.5)), t=2.5))
    over_up = _fit(X=dict(cmf=float(np.exp(FAMILY_LOG_TOL * 1.5)), t=2.5))
    over_dn = _fit(X=dict(cmf=float(np.exp(-FAMILY_LOG_TOL * 1.5)), t=2.5))
    assert family_robust_flags({"nb": base, "nb1": under}, "nb") == {"X": True}
    assert family_robust_flags({"nb": base, "nb1": over_up}, "nb") == {"X": False}
    assert family_robust_flags({"nb": base, "nb1": over_dn}, "nb") == {"X": False}
    assert FAMILY_LOG_TOL == 0.02


def test_a_sign_flip_is_not_robust_even_when_the_size_barely_moves():
    # 0.999 -> 1.001 is 0.2% on the log scale, well inside the tolerance, but
    # the term changes direction: that is a different claim about the road.
    base = _fit(X=dict(cmf=0.999, t=-0.5))
    alt = _fit(X=dict(cmf=1.001, t=+0.5))
    assert family_robust_flags({"nb": base, "nb1": alt}, "nb") == {"X": False}


def test_losing_significance_is_not_robust():
    base = _fit(URB=dict(cmf=0.9815, t=-2.29))
    alt = _fit(URB=dict(cmf=0.9948, t=-0.73))          # same sign, now inside z
    assert family_robust_flags({"nb": base, "nb1": alt}, "nb") == {"URB": False}


def test_gaining_significance_is_also_a_verdict_change():
    base = _fit(X=dict(cmf=0.999, t=0.5))
    alt = _fit(X=dict(cmf=0.998, t=2.5))
    assert family_robust_flags({"nb": base, "nb1": alt}, "nb") == {"X": False}


def test_missing_t_falls_back_to_magnitude_only():
    # A Poisson fit has no dispersion but still has a t-ratio; a hand-built
    # catalogue may carry none at all, and that must not crash the screen.
    base = _fit(X=dict(cmf=0.99))
    alt = _fit(X=dict(cmf=0.99, t=None))
    assert family_robust_flags({"nb": base, "nb1": alt}, "nb") == {"X": True}
    nan_alt = _fit(X=dict(cmf=0.99, t=float("nan")))
    assert family_robust_flags({"nb": base, "nb1": nan_alt}, "nb") == {"X": True}
    moved = _fit(X=dict(cmf=1.05, t=None))
    assert family_robust_flags({"nb": base, "nb1": moved}, "nb") == {"X": False}


def test_robust_requires_every_alternative():
    # Two of three families agree; the third flips the term.  The published
    # claim has to hold against all of them, not against the ones that agree.
    base = _fit(X=dict(cmf=0.98, t=-3.0))
    agree = _fit(X=dict(cmf=0.981, t=-3.1))
    flip = _fit(X=dict(cmf=1.02, t=+0.4))
    flags = family_robust_flags({"poisson": base, "nb": agree, "nb1": flip},
                                "poisson")
    assert flags == {"X": False}
    assert family_robust_flags({"poisson": base, "nb": agree}, "poisson") == {"X": True}


def test_a_term_missing_from_an_alternative_raises():
    # Silence here would report "robust" for a term nobody compared.
    base = _fit(X=dict(cmf=0.98, t=-3.0), Y=dict(cmf=1.01, t=2.0))
    partial = _fit(X=dict(cmf=0.98, t=-3.0))
    with pytest.raises(KeyError, match="do not cover the same terms"):
        family_robust_flags({"nb": base, "nb1": partial}, "nb")


def test_an_unknown_family_raises():
    base = _fit(X=dict(cmf=0.98, t=-3.0))
    with pytest.raises(KeyError, match="baseline family"):
        family_robust_flags({"nb": base}, "poisson")
    with pytest.raises(KeyError, match="alternative family"):
        family_robust_flags({"nb": base}, "nb", alternatives=["nb1"])
