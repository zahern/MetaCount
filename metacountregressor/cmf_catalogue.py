"""Turn a fitted two-tier CMF model into a transferable catalogue table."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .GA_CMF_AADT_JAX import SIGMA_FLAT_TOL, sigma_is_flat, _param_labels

# Quadrature nodes for expectations over a standardised mixing distribution.
# Hermite-E nodes integrate against the standard normal density; weights are
# normalised to sum to one, so a mean/variance/CDF is a weighted average.
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(100)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def _standard_mixing(dist, z):
    """Map standard normal variates to a mixing distribution, as ``shape_draws``.

    Every marginal is mean 0 / variance 1, so ``sigma`` is the SD and the
    standardised variate ``u`` has these moments whichever distribution is
    chosen.  Must stay in step with ``GA_CMF_AADT_JAX.shape_draws``.
    """
    if dist in (None, "normal"):
        return z
    if dist == "uniform":
        return np.sqrt(3.0) * (2.0 * norm.cdf(z) - 1.0)
    if dist == "triangular":
        p = norm.cdf(z)
        t = np.where(p < 0.5, np.sqrt(2.0 * p) - 1.0,
                     1.0 - np.sqrt(2.0 * (1.0 - p)))
        return t * np.sqrt(6.0)
    if dist == "lognormal":
        s = 0.5
        return ((np.exp(s * z) - np.exp(s * s / 2.0))
                / np.sqrt((np.exp(s * s) - 1.0) * np.exp(s * s)))
    raise ValueError(f"Unknown distribution '{dist}'")


def mixing_mgf(scale, dist="normal"):
    """E[exp(scale * U)] for a standardised mixing variate U.

    This is the factor that separates the **mean** CMF from the CMF at the
    mean coefficient: a random coefficient c = coef + sigma*U gives
    E[exp(c)] = exp(coef) * mgf(sigma), not exp(coef).  For a normal mixing
    distribution the factor is exp(sigma^2 / 2) (the lognormal correction);
    the bounded distributions give a smaller factor.

    ``lognormal`` mixing returns NaN: its MGF does not exist for a positive
    scale, because the right tail is too heavy (the expectation is the sum of
    moments of a lognormal, which diverges).  A lognormal random coefficient
    therefore has a finite median and a well-defined distribution of CMFs, but
    **no finite mean CMF** - reporting one would be an artefact of truncating
    the tail.
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale == 0.0:
        return 1.0
    if dist == "lognormal":
        return float("nan")
    u = _standard_mixing(dist, _GH_NODES)
    return float(np.sum(_GH_WEIGHTS * np.exp(scale * u)))


def mixing_cdf(x, dist="normal"):
    """P(U < x) for a standardised mixing variate U.

    The share of sites whose CMF falls below 1 is this CDF at -coef/sigma, so
    it must use the fitted mixing distribution rather than a normal CDF: a
    uniform or triangular mixing distribution has bounded support, and a
    lognormal one is skewed.
    """
    x = float(x)
    if not np.isfinite(x):
        return np.nan
    if dist in (None, "normal"):
        return float(norm.cdf(x))
    if dist == "uniform":
        return float(np.clip((x + np.sqrt(3.0)) / (2.0 * np.sqrt(3.0)), 0.0, 1.0))
    if dist == "triangular":
        a = np.sqrt(6.0)
        if x <= -a:
            return 0.0
        if x >= a:
            return 1.0
        if x < 0:
            return float((x + a) ** 2 / (2.0 * a * a))
        return float(1.0 - (a - x) ** 2 / (2.0 * a * a))
    if dist == "lognormal":
        s = 0.5
        m = np.exp(s * s / 2.0)
        k = np.sqrt((np.exp(s * s) - 1.0) * np.exp(s * s))
        arg = m + k * x
        if arg <= 0:
            return 0.0
        return float(norm.cdf(np.log(arg) / s))
    raise ValueError(f"Unknown distribution '{dist}'")


def _readable_basis(span, coef):
    """Power of ten that makes a wide-range count effect readable.

    A per-unit CMF for a variable spanning hundreds of units is not
    interpretable: a per-unit elasticity of 0.0028 over 1-1732 units compounds
    to a 133-fold CMF, which is a linear-in-exposure proxy rather than a
    published-style CMF.  Report such terms per 100 or per 1000 units instead.
    Only applies when the full-span extrapolation exceeds one decade, so
    narrow-range and dummy terms keep their per-unit reading.
    """
    if not np.isfinite(span) or span < 100 or coef == 0 or not np.isfinite(coef):
        return 1
    if abs(coef * span) < np.log(10):
        return 1
    want = 0.3 / abs(coef)          # basis putting the CMF near 1.35
    mag = 10 ** int(np.round(np.log10(max(want, 1.0))))
    return int(min(mag, 10 ** int(np.floor(np.log10(span)))))


def build_cmf_catalogue(params, se, data, selected_baseline, selected_local,
                        rand_baseline, rand_local, model="poisson",
                        region=None, z=1.96, dists=None, conditional=None):
    """One row per term.

    Upper tier: CMF = exp(alpha * basis) with a Wald interval on alpha; the
    basis is one unit, except for wide-range count variables where a power of
    ten (``basis_scale``) is used so the CMF is readable (``cmf_basis`` says
    which, and ``coef`` stays per unit with ``coef_per_basis`` alongside).
    Lower tier: the term shifts the AADT elasticity, so the CMF depends on
    traffic; ``cmf_aadt_doubling`` is 2**beta, the multiplicative effect of a
    one-unit change in the variable when AADT doubles.
    Random terms also report the population spread exp(alpha +/- sigma) and the
    share of sites where the effect is protective (CMF < 1).  A random
    coefficient is c = coef + sigma*U, so the reported ``cmf`` is the
    **median** (typical-site) CMF exp(coef) while ``cmf_mean`` is the mean of
    the CMF distribution, E[exp(c)] = exp(coef) * mgf(sigma); the two differ by
    the variance correction and it is the mean that an aggregate before/after
    or benefit-cost calculation needs.  ``dists`` gives the mixing distribution
    of each random term in draw order (baseline terms, then local terms); it
    drives both the mean and ``share_protective``, which is P(U < -coef/sigma)
    for that distribution rather than a normal CDF.
    ``sigma_collapsed`` marks a selected random term whose sigma cannot be
    separated from zero (``sigma_is_flat``: below ``SIGMA_FLAT_TOL``, or smaller
    than ``z`` standard errors): the sign of the mean is uniform across sites, so
    the spread is not heterogeneity and ``share_protective`` is left empty
    rather than reported as 0 or 1.  ``sigma_se`` and ``sigma_z`` (= sigma / SE)
    are reported so a reader can check that judgement, and ``sigma_flat_abs``
    says whether the estimate collapsed onto the mean rather than merely being
    imprecise.
    ``min``/``max`` give the applicability range seen in the estimation data.
    ``conditional`` maps a term to a ``HelperConditionalCMF`` for a structural
    fit whose coefficient mean or SD moves with a helper covariate; that row
    then carries ``cmf_expression`` (the CMFunction in AADT and the helper) and
    ``helper_conditional`` = True, and the objects are returned in
    ``out.attrs["conditional"]`` for a dashboard to evaluate.  A row without one
    is unchanged, so helper-free catalogues keep the single-number form.
    """
    conditional = conditional or {}
    params = np.asarray(params, dtype=float)
    se = np.asarray(se, dtype=float)
    labels = _param_labels(selected_baseline, selected_local,
                           rand_baseline, rand_local, model)
    idx = {l: i for i, l in enumerate(labels)}
    # Mixing distribution per random term, in draw order: baseline terms first,
    # then local terms (the order shape_draws uses in the likelihood).
    dist_map = {}
    if dists:
        _d = iter(dists)
        for v, r in zip(selected_baseline, rand_baseline):
            if r:
                dist_map[("upper", v)] = next(_d, "normal")
        for v, r in zip(selected_local, rand_local):
            if r:
                dist_map[("lower", v)] = next(_d, "normal")
    rows = []
    for tier, vars_, rand in (("upper", selected_baseline, rand_baseline),
                              ("lower", selected_local, rand_local)):
        key = "alpha" if tier == "upper" else "beta"
        for v, is_rand in zip(vars_, rand):
            i = idx[f"{key}[{v}]"]
            b, s = params[i], se[i]
            sig = abs(params[idx[f"sigma_{key}[{v}]"]]) if is_rand else 0.0
            sig_se = float(se[idx[f"sigma_{key}[{v}]"]]) if is_rand else np.nan
            # sigma is a boundary parameter, so "is there heterogeneity?" is a
            # Wald question against sigma's own SE, not only an absolute cutoff:
            # a sigma of 0.09 judged to 0.09 is no more evidence of site-level
            # variation than a collapsed one.
            sig_z = (sig / sig_se if is_rand and np.isfinite(sig_se) and sig_se > 0
                     else np.nan)
            collapsed = bool(is_rand and sigma_is_flat(sig, sig_se, z=z))
            flat_abs = bool(is_rand and sig < SIGMA_FLAT_TOL)
            dist = (dist_map.get((tier, v), "normal") if is_rand else "")
            col = data[v] if v in data.columns else None
            span = float(col.max() - col.min()) if col is not None else np.nan
            basis = _readable_basis(span, b) if tier == "upper" else 1
            basis_factor = basis if tier == "upper" else 1.0
            if tier == "upper":
                cmf_f = lambda e: float(np.exp(basis_factor * e))
            else:
                cmf_f = lambda e: float(2.0 ** e)
            rows.append({
                "term": v, "tier": tier, "region": region,
                "coef": b, "se": s,
                "coef_per_basis": b * basis_factor, "se_per_basis": s * basis_factor,
                "basis_scale": basis,
                "cmf": cmf_f(b),
                "cmf_median": cmf_f(b),
                # Mean of the CMF distribution: the coefficient is random with
                # SD sigma, so E[CMF] = CMF(median) * E[exp(sigma * U)] with
                # U standardised on the mixing distribution and the effect
                # scaled by the reporting basis (ln 2 for a doubling).
                "cmf_mean": (cmf_f(b) * mixing_mgf(
                    sig * (basis_factor if tier == "upper" else np.log(2.0)),
                    dist) if is_rand else np.nan),
                "mixing": dist,
                "cmf_lo": cmf_f(b - z * s),
                "cmf_hi": cmf_f(b + z * s),
                "cmf_basis": ("AADT doubling" if tier == "lower" else
                              "per unit" if basis == 1 else f"per {basis} units"),
                "random": bool(is_rand), "sigma": sig,
                "sigma_se": sig_se, "sigma_z": sig_z,
                "sigma_collapsed": collapsed, "sigma_flat_abs": flat_abs,
                "cmf_spread_lo": (cmf_f(b - sig) if is_rand else np.nan),
                "cmf_spread_hi": (cmf_f(b + sig) if is_rand else np.nan),
                "share_protective": (np.nan if (not is_rand or collapsed) else
                                     mixing_cdf(-b / sig, dist) if sig > 0 else
                                     float(b < 0)),
                "significant": bool(s > 0 and abs(b / s) > z),
                "min": float(col.min()) if col is not None else np.nan,
                "max": float(col.max()) if col is not None else np.nan,
            })
    out = pd.DataFrame(rows)
    # Published CMFs are almost always within about 0.3-3; flag anything beyond
    # on the reported basis for a coding/sparsity check.
    span = (out["max"] - out["min"]).replace(0, np.nan)
    # Size the effect over the full observed range, per unit, whatever basis the
    # row is reported on; a wide count range legitimately compounds, so that is
    # recorded separately rather than called implausible.
    out["cmf_over_range"] = np.where(out["tier"] == "upper",
                                     np.exp(out["coef"] * span), np.nan)
    out["wide_range_extrapolation"] = (out["tier"] == "upper") & (
        out["cmf_over_range"] > 10)
    out["implausible"] = (out["tier"] == "upper") & (
        (out["cmf"] > 3) | (out["cmf"] < 1 / 3))
    # A helper-conditional term is a function, not a number: say so on the row
    # and quote the CMFunction, so a structural random term is translated rather
    # than described.  Terms without one keep an empty expression.
    out["helper_conditional"] = out["term"].isin(list(conditional))
    out["cmf_expression"] = out["term"].map(
        lambda t: conditional[t].expression() if t in conditional else "")
    out.attrs["aadt_range"] = (float(data["AADT"].min()), float(data["AADT"].max()))
    out.attrs["conditional"] = dict(conditional)
    return out


# HSM / CMF-Clearinghouse practice stacks countermeasures by multiplying their
# factors, which assumes the treatments act independently.  That is tenable for
# two factors and unreliable beyond, and a large combined reduction is a signal
# that the product has double-counted overlapping effects.  Both thresholds are
# enforced here rather than left to a footnote: they are what a reviewer asks
# about after seeing the number.
COMBINE_TERMS_WARN = 3          # multiplication assumes independence of effects
COMBINE_REDUCTION_WARN = 0.60   # a total reduction this large needs scrutiny


def _combination_notes(n_terms, cmf, assumed_independence, mechanisms=None):
    """Review notes for a stacked CMF (shared by ``combine_cmfs``/``derive_cmf``)."""
    notes = []
    if n_terms >= COMBINE_TERMS_WARN:
        notes.append(
            f"{n_terms} countermeasures combined by multiplication: the product "
            "assumes independent effects, which is unreliable at this many - "
            "review the total reduction critically.")
    if assumed_independence:
        notes.append(
            "Variances were summed as if the CMFs were independent. If they "
            "come from one fitted model, pass that model's covariance instead.")
    if mechanisms is not None:
        seen, dups = set(), []
        for m in mechanisms:
            if m is None:
                continue
            if m in seen and m not in dups:
                dups.append(m)
            seen.add(m)
        if dups:
            notes.append(
                "Two factors address the same crash mechanism "
                f"({', '.join(map(str, dups))}): multiplying them overstates "
                "the benefit - omit one or estimate the joint effect.")
    reduction = 1.0 - cmf
    if reduction >= COMBINE_REDUCTION_WARN:
        notes.append(
            f"Combined reduction of {reduction:.0%} is at or beyond the "
            f"{COMBINE_REDUCTION_WARN:.0%} that warrants careful scrutiny.")
    return notes


def combine_cmfs(cmfs, se_log=None, cov=None, mechanisms=None, z=1.96):
    """Stack several CMFs into one factor, with the practice's caveats.

    Countermeasures at one location are combined by multiplying their factors;
    ``log CMF`` add, so the interval comes from the covariance of the log CMFs.
    Pass ``cov`` when the CMFs come from one fitted model - the hierarchical
    CMF model estimates the covariances, so independence need not be assumed.
    With only ``se_log`` the variances are summed (independence assumed) and
    that assumption is reported.  ``mechanisms`` may label each factor's crash
    mechanism so two factors treating the same one can be flagged as likely
    double counting.

    Returns the product, a log-normal interval (when variances are supplied),
    the total ``reduction`` (1 - CMF), and ``notes``: multi-factor products
    (>= COMBINE_TERMS_WARN), reductions >= COMBINE_REDUCTION_WARN, assumed
    independence and duplicated mechanisms.  ``scrutinise`` is True when any
    of the three countermeasure-level triggers fired.
    """
    c = np.asarray(cmfs, dtype=float).ravel()
    if c.size == 0:
        raise ValueError("combine_cmfs needs at least one CMF")
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("CMFs must be finite and positive")
    n = c.size
    if mechanisms is not None and len(mechanisms) != n:
        raise ValueError("mechanisms must have one entry per CMF")
    assumed_independence = False
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (n, n):
            raise ValueError("cov must be a square matrix matching the CMFs")
        v = float(np.ones(n) @ cov @ np.ones(n))
    elif se_log is not None:
        s = np.asarray(se_log, dtype=float).ravel()
        if s.size != n:
            raise ValueError("se_log must have one entry per CMF")
        if not np.all(np.isfinite(s)):
            raise ValueError("se_log must be finite")
        v = float(np.sum(s ** 2))
        assumed_independence = True
    else:
        v = np.nan
    log_cmf = float(np.sum(np.log(c)))
    cmf = float(np.exp(log_cmf))
    se = float(np.sqrt(max(v, 0.0))) if np.isfinite(v) else np.nan
    notes = _combination_notes(n, cmf, assumed_independence, mechanisms)
    return {
        "cmf": cmf,
        "cmf_lo": float(np.exp(log_cmf - z * se)) if np.isfinite(se) else np.nan,
        "cmf_hi": float(np.exp(log_cmf + z * se)) if np.isfinite(se) else np.nan,
        "se_log_cmf": se,
        "reduction": 1.0 - cmf,
        "n_terms": int(n),
        "assumed_independence": assumed_independence,
        "scrutinise": bool(
            n >= COMBINE_TERMS_WARN
            or (1.0 - cmf) >= COMBINE_REDUCTION_WARN
            or any("same crash mechanism" in t for t in notes)),
        "notes": notes,
    }


FAMILY_LOG_TOL = 0.02   # a CMF that moves more than 2% on the log scale is not "the same CMF"


def family_robust_flags(fits, baseline, alternatives=None, log_tol=FAMILY_LOG_TOL,
                        z=1.96):
    """Which CMFs survive a change of count family?

    A published CMF is a ratio of conditional means, so it looks like a
    property of the road.  It is not: refitting the same specification under a
    different variance function (NB2 vs NB1 vs Poisson) reweights the
    observations, and a weakly identified coefficient can move enough to change
    the CMF's sign or significance.  This decides, term by term, whether that
    happened.

    Parameters
    ----------
    fits : mapping
        ``{family: {term: {"cmf": value, "t": value}}}`` — the CMF of each term
        and the coefficient's t-ratio, as estimated under each family.
    baseline : str
        The family being published.  Alternatives are judged against it.
    alternatives : iterable of str, optional
        Families to compare; defaults to every key of ``fits`` except
        ``baseline``.
    log_tol : float
        Tolerance on ``|log(cmf_alt) - log(cmf_base)|``.
    z : float
        Significance threshold for the ``t`` verdict (1.96 = 5%).

    Returns
    -------
    dict
        ``{term: bool}``, True where the CMF is robust to every alternative.

    A term is robust only if, against *each* alternative, its CMF moves by less
    than ``log_tol``, the coefficient keeps its sign, and the significance
    verdict is unchanged.  ``t`` may be ``None``/NaN for either side, in which
    case only the magnitude is judged.

    A term present in the baseline but missing from an alternative raises
    ``KeyError``: the comparison is only meaningful over the same term set, and
    silently calling a missing term robust would turn a broken comparison into a
    clean bill of health.
    """
    if baseline not in fits:
        raise KeyError(f"baseline family {baseline!r} is not among {sorted(fits)}")
    alts = list(alternatives) if alternatives is not None else \
        [f for f in fits if f != baseline]
    for fam in alts:
        if fam not in fits:
            raise KeyError(f"alternative family {fam!r} is not among {sorted(fits)}")

    def _t(entry):
        val = entry.get("t")
        if val is None:
            return None
        val = float(val)
        return val if np.isfinite(val) else None

    out: dict[str, bool] = {}
    for term, base in fits[baseline].items():
        robust = True
        for fam in alts:
            if term not in fits[fam]:
                raise KeyError(
                    f"term {term!r} is fitted under {baseline!r} but not under "
                    f"{fam!r}; the two catalogues do not cover the same terms")
            alt = fits[fam][term]
            if abs(np.log(float(alt["cmf"]) / float(base["cmf"]))) >= log_tol:
                robust = False
                break
            t_b, t_a = _t(base), _t(alt)
            if t_b is not None and t_a is not None:
                if np.sign(t_b) != np.sign(t_a):
                    robust = False
                    break
                if (abs(t_b) > z) != (abs(t_a) > z):
                    robust = False
                    break
        out[term] = robust
    return out


def benchmark_against_reference(catalogue, reference, term_map=None):
    """Compare estimated CMFs with published values (e.g. CMF Clearinghouse).

    ``reference`` needs columns ``term`` (or the keys of ``term_map`` mapped to
    catalogue terms), ``cmf`` and optionally ``cmf_se`` or ``cmf_lo``/``cmf_hi``
    (published interval).  Only per-unit upper-tier terms are comparable
    directly.  Returns agreement flags: interval overlap and same direction.
    """
    ref = reference.copy()
    if term_map:
        ref["term"] = ref["term"].map(term_map).fillna(ref["term"])
    cat = catalogue[catalogue.tier == "upper"]
    m = cat.merge(ref, on="term", suffixes=("", "_ref"))
    if "cmf_lo_ref" not in m.columns:
        se = m["cmf_se"] if "cmf_se" in m.columns else 0.0
        m["cmf_lo_ref"] = m["cmf_ref"] - 1.96 * se
        m["cmf_hi_ref"] = m["cmf_ref"] + 1.96 * se
    m["same_direction"] = (m["cmf"] < 1) == (m["cmf_ref"] < 1)
    m["intervals_overlap"] = (m["cmf_lo"] <= m["cmf_hi_ref"]) & (m["cmf_hi"] >= m["cmf_lo_ref"])
    m["ratio_to_ref"] = m["cmf"] / m["cmf_ref"]
    return m[["term", "cmf", "cmf_lo", "cmf_hi", "cmf_ref", "cmf_lo_ref",
              "cmf_hi_ref", "same_direction", "intervals_overlap", "ratio_to_ref"]]


def derive_cmf(params, cov, selected_baseline, selected_local,
               rand_baseline, rand_local, model="poisson",
               changes=None, aadt=None, z=1.96):
    """CMF for a specific treatment/scenario with a delta-method interval.

    ``changes`` maps variable -> change in its value (e.g. {"X": 1} for a unit
    increase, or {"Shoulder": 1} for dummy 0 -> 1).  Upper-tier terms scale
    crashes by exp(alpha * dx); lower-tier terms scale them by
    AADT ** (beta * dz), so ``aadt`` is required if any are changed.
    log CMF = g' theta, so se(log CMF) = sqrt(g' Cov g); the interval is
    exp(log CMF +/- z * se), which is asymmetric and always positive (this is
    the form HSM/Clearinghouse CMFs are reported in).  Several changes combine
    multiplicatively (their log effects add), with covariances respected, so the
    independence assumption behind multiplying published CMFs is not needed
    here.  The stacked-total caveats still apply and are returned as ``notes``:
    a combination of COMBINE_TERMS_WARN or more changes, or one implying a
    reduction of COMBINE_REDUCTION_WARN or more, sets ``scrutinise``.
    """
    params = np.asarray(params, dtype=float)
    cov = np.asarray(cov, dtype=float)
    labels = _param_labels(selected_baseline, selected_local,
                           rand_baseline, rand_local, model)
    idx = {l: i for i, l in enumerate(labels)}
    g = np.zeros(len(labels))
    for v, dx in (changes or {}).items():
        if v in selected_baseline:
            g[idx[f"alpha[{v}]"]] += dx
        elif v in selected_local:
            if aadt is None:
                raise ValueError("aadt is required for lower-tier changes")
            g[idx[f"beta[{v}]"]] += dx * np.log(aadt)
        else:
            raise KeyError(f"{v} is not in the fitted model")
    log_cmf = float(g @ params)
    se = float(np.sqrt(max(g @ cov @ g, 0.0)))
    cmf = float(np.exp(log_cmf))
    n_terms = sum(1 for dx in (changes or {}).values() if dx)
    return {"cmf": cmf,
            "cmf_lo": float(np.exp(log_cmf - z * se)),
            "cmf_hi": float(np.exp(log_cmf + z * se)),
            "se_log_cmf": se,
            "cmf_se": cmf * se,
            "significant": bool(se > 0 and abs(log_cmf) / se > z),
            "reduction": 1.0 - cmf,
            "n_terms": int(n_terms),
            "scrutinise": bool(n_terms >= COMBINE_TERMS_WARN
                               or (1.0 - cmf) >= COMBINE_REDUCTION_WARN),
            "notes": _combination_notes(n_terms, cmf, False)}


def _num(x, dp=4):
    """Compact number for an expression string (no trailing zeros)."""
    x = float(x)
    if not np.isfinite(x):
        return "nan"
    s = f"{x:.{dp}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


class HelperConditionalCMF:
    """A random term translated into a CMFunction of AADT *and* the helper.

    The structural equation (``random_parameter_structure``) lets a random
    coefficient move with observed covariates:

        c_k(w) = coef_k + sum_z delta_kz * w_z            (mean shift)
        sigma_k(w) = exp(logsig_k + sum_z omega_kz * w_z)  (log-scale shift)

    A random coefficient is not one CMF, so a helper-conditional term is not one
    number either: the two-tier CMF map turns it into

        upper tier   CMF(dx, w)     = exp(c_k(w) * dx)
        lower tier   CMF(dz, w, A)  = AADT ** (c_k(w) * dz)

    with the site spread at ``w`` being the same map at ``c_k(w) +/- sigma_k(w)``.
    ``delta`` empty and ``omega`` empty reduces exactly to the ordinary catalogue
    CMF, so a fixed-effect or helper-free random term needs no separate path.

    Two bands are reported and they answer different questions.  ``cmf_lo``/
    ``cmf_hi`` is the delta-method interval for the CMF **at a given helper
    level** (uncertainty in the coefficient and in the loadings, so it widens as
    ``w`` moves away from zero).  ``cmf_spread_lo``/``cmf_spread_hi`` is the
    site-to-site heterogeneity band ``c_k(w) +/- sigma_k(w)``, which is a
    statement about the road population, not about estimation error.
    """

    def __init__(self, term, tier, coef, delta=None, sigma=0.0, logsig=None,
                 omega=None, changes=1.0, dist="normal", z=1.96,
                 cov=None, param_index=None, label=None, sd_floor=0.0,
                 sigma_se=None):
        if tier not in ("upper", "lower"):
            raise ValueError("tier must be 'upper' or 'lower'")
        self.term = str(term)
        self.label = str(label or term)
        self.tier = tier
        self.coef = float(coef)
        self.delta = {str(k): float(v) for k, v in (delta or {}).items()}
        self.omega = {str(k): float(v) for k, v in (omega or {}).items()}
        self.sigma = abs(float(sigma))
        self.logsig = None if logsig is None else float(logsig)
        # Correlated random terms keep part of their variance off the diagonal of
        # the Cholesky factor, where it does not scale with any helper.  It adds
        # to sigma(w) in quadrature rather than through the log-scale shift.
        self.sd_floor = abs(float(sd_floor))
        # SE of the fitted sigma, so the ``sigma_is_flat`` screen can be applied:
        # an SD judged to its own SE is no more evidence of heterogeneity than a
        # collapsed one, and no spread band should be drawn for it at any w.
        self.sigma_se = (None if sigma_se is None or not np.isfinite(sigma_se)
                         else float(sigma_se))
        self.changes = float(changes)
        self.dist = dist
        self.z = float(z)
        self._names = list(self.delta)
        # Covariance of [coef, delta...] only: omega enters the heterogeneity
        # band, not the interval around the conditional mean.
        self._cov = None
        if cov is not None and param_index is not None:
            cov = np.asarray(cov, dtype=float)
            idx = np.asarray(param_index, dtype=int)
            if cov.shape[0] <= int(idx.max()):
                raise ValueError("cov is smaller than param_index")
            self._cov = cov[np.ix_(idx, idx)]

    @property
    def collapsed(self):
        """True when sigma cannot be separated from zero by either criterion."""
        return bool(sigma_is_flat(self.sigma, self.sigma_se))

    # -- coefficient and dispersion at a helper level ---------------------

    def omega_names(self):
        """Helper names that shift sigma (``hetro_in_variances``)."""
        return list(self.omega)

    def helper_names(self):
        """Every helper the CMF depends on, mean shift first."""
        return list(self.delta) + [n for n in self.omega if n not in self.delta]

    def _w_vec(self, w, names):
        """Helper values as a vector in ``names`` order.

        Accepts a scalar (applied to every helper), a mapping, or None (all
        helpers at 0, i.e. the model's intercept).
        """
        if w is None:
            return np.zeros(len(names), dtype=float)
        if isinstance(w, dict):
            return np.array([float(w.get(n, 0.0)) for n in names], dtype=float)
        return np.full(len(names), float(w), dtype=float)

    def coef_at(self, w=None):
        """Mean coefficient at helper values ``w`` (``delta`` shift applied)."""
        return self.coef + float(np.dot(self._w_vec(w, self._names), list(self.delta.values())))

    def sigma_at(self, w=None):
        """Site-to-site SD at ``w``; the log-scale shift makes it w-dependent.

        ``sigma(w)^2 = sd_floor^2 + exp(2 (logsig + omega'w))``.  For an
        independent random term ``sd_floor`` is 0 and this is the log-scale
        shift alone; for a correlated one the off-diagonal Cholesky entries
        contribute the SD that no helper can move.
        """
        if not self.omega:
            return self.sigma
        base = self.logsig
        if base is None:
            base = np.log(self.sigma) if self.sigma > 0 else -np.inf
        shift = float(np.dot(self._w_vec(w, list(self.omega)), list(self.omega.values())))
        diag = float(np.exp(base + shift))
        return float(np.sqrt(self.sd_floor ** 2 + diag ** 2))

    # -- the CMFunction --------------------------------------------------

    def _scale(self, aadt=None):
        """Factor multiplying the coefficient to give log CMF."""
        if self.tier == "lower":
            if aadt is None:
                raise ValueError(f"CMF[{self.term}] is lower-tier: AADT is needed")
            return self.changes * np.log(np.asarray(aadt, dtype=float))
        return self.changes

    def log_cmf(self, aadt=None, w=None):
        """log CMF (vectorised over AADT; one helper level per call)."""
        return self.coef_at(w) * self._scale(aadt)

    def cmf(self, aadt=None, w=None):
        """CMF at a helper level, as a function of AADT (lower tier only)."""
        return np.exp(self.log_cmf(aadt, w))

    def interval(self, aadt=None, w=None, z=None):
        """Delta-method interval for the conditional CMF (uses ``cov``).

        d(log CMF)/d(coef, delta) = scale * [1, w], so the variance is the
        quadratic form in the fitted covariance.  With no covariance the bounds
        are NaN rather than silently zero-width.
        """
        z = self.z if z is None else float(z)
        log_cmf = self.log_cmf(aadt, w)
        if self._cov is None:
            nan = np.full(np.shape(log_cmf), np.nan)
            return nan, nan
        scale = self._scale(aadt)
        g = np.concatenate([[1.0], self._w_vec(w, self._names)])
        var = float(g @ self._cov @ g)
        se = np.abs(scale) * np.sqrt(max(var, 0.0))
        return np.exp(log_cmf - z * se), np.exp(log_cmf + z * se)

    def spread(self, aadt=None, w=None):
        """CMF at coefficient ``+/- sigma(w)``: the site-level spread.

        Empty (NaN) when the fitted sigma is indistinguishable from zero at
        ``w``: the coefficient then has one sign at every site, so a spread
        band would invent heterogeneity that was not estimated.
        """
        sig = self.sigma_at(w)
        if self.collapsed or sigma_is_flat(sig, None):
            nan = np.full(np.shape(self._scale(aadt)), np.nan)
            return nan, nan
        scale = self._scale(aadt)
        c = self.coef_at(w)
        return np.exp((c - sig) * scale), np.exp((c + sig) * scale)

    @property
    def aadt_free(self):
        """True when the CMF does not vary with AADT (upper tier)."""
        return self.tier == "upper"

    def expression(self):
        """The CMFunction as a readable formula, for a table or dashboard."""
        head = _num(self.coef)
        for k, v in self.delta.items():
            head += f" {'+' if v >= 0 else '-'} {_num(abs(v))}*{k}"
        helpers = ", ".join(self.helper_names())
        if self.tier == "lower":
            f = f"CMF[{self.term}](AADT{', ' + helpers if helpers else ''}) = " \
                f"AADT^(({head})*{_num(self.changes)})"
        else:
            f = f"CMF[{self.term}]({helpers or 'all AADT'}) = " \
                f"exp(({head})*{_num(self.changes)})"
        if self.omega:
            # A loading fitted to (numerically) zero is not a term in the formula:
            # printing "+ 0*CURVES" reads as a modelled effect that is absent.
            o = ""
            for k, v in self.omega.items():
                if abs(v) < 1e-8:
                    continue
                o += f" {'+' if v >= 0 else '-'} {_num(abs(v))}*{k}"
            base = self.logsig if self.logsig is not None else np.log(self.sigma)
            if not o:
                flat = float(np.sqrt(self.sd_floor ** 2 + np.exp(2 * base)))
                f += f"; sigma = {_num(flat)}"
                return f
            if self.sd_floor > 0:
                f += (f"; sigma = sqrt({_num(self.sd_floor ** 2, 5)} + "
                      f"exp({_num(base)}{o})^2)")
            else:
                f += f"; sigma = exp({_num(base)}{o})"
        elif self.sd_floor > 0:
            f += f"; sigma = sqrt({_num(self.sd_floor ** 2, 5)} + " \
                 f"{_num(self.sigma ** 2, 5)})"
        return f

    def surface(self, aadt_values, w_cases, z=None):
        """Grid of the CMFunction: one row per (helper level, AADT).

        ``w_cases`` is a sequence of ``(label, w)`` with ``w`` a scalar or a
        mapping of helper values (see ``helper_level_cases``).  Upper-tier terms
        are AADT-free by construction, so their rows carry ``aadt = NaN`` rather
        than pretending to vary with traffic.
        """
        z = self.z if z is None else float(z)
        rows = []
        for w_label, w in w_cases:
            w = dict(w) if not isinstance(w, dict) else w
            aadts = [np.nan] if self.aadt_free else list(np.atleast_1d(aadt_values))
            sig = self.sigma_at(w)
            for a in aadts:
                aa = None if self.aadt_free else a
                lo, hi = self.interval(aa, w, z=z)
                slo, shi = self.spread(aa, w)
                rows.append({
                    "term": self.term, "tier": self.tier,
                    "helper_level": w_label,
                    **({"helper_value": _num(w[self._names[0]], 3)}
                       if len(self._names) == 1 else {}),
                    "aadt": a,
                    "coef": self.coef_at(w),
                    "sigma": sig,
                    "cmf": float(np.exp(self.log_cmf(aa, w))),
                    "cmf_lo": float(lo), "cmf_hi": float(hi),
                    "cmf_spread_lo": float(slo), "cmf_spread_hi": float(shi),
                    "sigma_collapsed_at_w": bool(self.collapsed or
                                                sigma_is_flat(sig, None)),
                })
        return pd.DataFrame(rows)


def helper_level_cases(data, quantiles=(0.1, 0.5, 0.9), extra=None):
    """Helper levels to evaluate a conditional CMF at.

    Uses the observed distribution rather than an arbitrary grid: ``p10`` /
    ``p50`` / ``p90`` of the helper plus its mean, so the quoted CMFs sit inside
    the range the model was estimated on.  ``data`` is a mapping of helper name
    to values; ``extra`` adds named cases as ``{label: {helper: value}}``.
    """
    cases = []
    for name, col in dict(data or {}).items():
        vals = np.asarray(col, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        cases.append((f"{name} mean", {name: float(vals.mean())}))
        for q in quantiles:
            cases.append((f"{name} p{int(q * 100)}", {name: float(np.quantile(vals, q))}))
    for label, w in (extra or {}).items():
        cases.append((label, dict(w)))
    return cases


def cholesky_sd_se(params, index, cov, K, k):
    """Delta-method SE of the row-norm SD of correlated random term ``k``.

    ``sd_k = ||L_k,:||`` with ``L_kk = exp(ld_k)``, so ``d sd / d L_kj = L_kj/sd``
    for the entries below the diagonal and ``d sd / d ld_k = L_kk^2 / sd``.  The
    SD is a boundary quantity like any other sigma, so it needs its own SE before
    it can be called heterogeneity.
    """
    params = np.asarray(params, dtype=float)
    cov = np.asarray(cov, dtype=float)
    cs = index["chol"][0]
    L = np.zeros((K, K))
    L[np.tril_indices(K)] = params[cs:cs + K * (K + 1) // 2]
    diag = np.clip(np.diag(L).copy(), -12.0, 6.0)
    L[np.diag_indices(K)] = np.exp(diag)
    sd = float(np.linalg.norm(L[k]))
    if sd <= 0:
        return sd, np.nan
    pos = {tuple(t): cs + i for i, t in enumerate(zip(*np.tril_indices(K)))}
    ids = [pos[(k, j)] for j in range(k)] + [pos[(k, k)]]
    g = np.array([L[k, j] / sd for j in range(k)] + [L[k, k] ** 2 / sd])
    sub = cov[np.ix_(ids, ids)]
    return sd, float(np.sqrt(max(g @ sub @ g, 0.0)))


def conditional_cmf_from_fit(params, spec, index, term, tier, changes=1.0,
                             dist="normal", z=1.96, cov=None, term_map=None,
                             label=None):
    """Translate one random term of a hierarchical (``main_hpc``) fit.

    Reads the random coefficient, its ``hetro`` (mean) and ``hetro_var``
    (log-scale) loadings out of ``params`` using the fit's own ``index``, so the
    CMFunction cannot drift from the parameter layout the likelihood used.

    ``index`` comes from ``main_hpc.build_base_index``; ``term`` is the name in
    the CMF catalogue (``URB``) and ``term_map`` its engine column
    (``__cmf_local__URB``) for lower-tier terms.  ``changes`` is the change in
    the variable the CMF is for: ``dz`` for a lower-tier term (1 = a dummy), ``dx``
    for an upper-tier one.

    An **independent** random term carries its SD in ``ind_sd``.  A **correlated**
    one carries it in the Cholesky factor instead: the diagonal is a log-SD and
    the entries below it are free, so the marginal SD of term ``k`` is
    ``sqrt(sum_{j<k} L_kj^2 + exp(ld_k)^2)`` — the row norm.  The off-diagonal
    part is passed as ``sd_floor`` so that a variance helper, which shifts only
    the diagonal, still gives ``sigma(w)^2 = sd_floor^2 + exp(2(ld_k + omega'w))``.
    Correlated terms are translated under normal mixing only: the engine maps a
    non-normal correlated draw as ``exp(mean + L u)`` rather than ``mean + L u``,
    so its SD cannot be read as a shift on the coefficient.

    Helpers enter ``build_eta`` averaged over the panel (``jnp.mean(..., axis=1)``),
    so ``w`` is the panel mean of the helper.  With one row per site — as in the
    CMF deliverable — that is just the site's value.
    """
    params = np.asarray(params, dtype=float)
    engine = (term_map or {}).get(term, term)
    ind_names = list(spec.random_ind_names)
    cor_names = list(spec.random_cor_names)
    if engine not in ind_names + cor_names:
        raise ValueError(
            f"{term!r} (engine column {engine!r}) is not a random term of this fit "
            f"(random: {ind_names + cor_names}). A fixed coefficient has no "
            "heterogeneity to condition on.")
    Ktot = spec.K_random_total
    sd_floor, dist, se_sigma = 0.0, "normal", None
    if engine in cor_names:
        k = cor_names.index(engine)
        dists = list(getattr(spec, "random_cor_dists", ()) or ())
        dist = (dists[k] if k < len(dists) else "normal") or "normal"
        if dist != "normal":
            raise ValueError(
                f"{term!r} has a {dist!r} correlated mixing distribution: the engine "
                "maps that draw as exp(mean + L u) rather than mean + L u, so its "
                "marginal SD is not a shift the CMF basis can use. Translate it "
                "under normal mixing, or report the CMF at the median draw instead.")
        K_cor = len(cor_names)
        cs, ce = index["chol"]
        L = np.zeros((K_cor, K_cor))
        L[np.tril_indices(K_cor)] = params[cs:ce]
        diag = np.clip(np.diag(L).copy(), -12.0, 6.0)   # as _positive_scale
        L[np.diag_indices(K_cor)] = np.exp(diag)
        sd_floor = float(np.linalg.norm(L[k, :k]))
        sigma = float(np.linalg.norm(L[k]))
        logsig = float(diag[k])
        coef_idx = index["cor_mean"][0] + k
        rng_pos = k                             # engine order: correlated first
        if cov is not None:
            se_sigma = cholesky_sd_se(params, index, cov, K_cor, k)[1]
    else:
        r_ind = ind_names.index(engine)
        rng_pos = len(cor_names) + r_ind
        coef_idx = index["ind_mean"][0] + r_ind
        ls_idx = index["ind_sd"][0] + r_ind
        logsig = float(params[ls_idx])
        sigma = float(np.exp(logsig))
        if cov is not None:
            se_sigma = float(sigma * np.sqrt(
                max(float(np.asarray(cov)[ls_idx, ls_idx]), 0.0)))
    delta, delta_idx = {}, []
    if len(spec.hetro_names) and "hetro" in index:
        hs = index["hetro"][0]
        for zi, nm in enumerate(spec.hetro_names):
            j = hs + zi * Ktot + rng_pos      # helper-major, as build_eta reshapes
            delta[str(nm)] = float(params[j])
            delta_idx.append(j)
    omega = {}
    if len(spec.hetro_var_names) and "hetro_var" in index:
        vs = index["hetro_var"][0]
        for zi, nm in enumerate(spec.hetro_var_names):
            omega[str(nm)] = float(params[vs + zi * Ktot + rng_pos])
    return HelperConditionalCMF(
        term=term, tier=tier, coef=float(params[coef_idx]), delta=delta,
        sigma=sigma, logsig=logsig, omega=omega, changes=changes, dist=dist,
        z=z, cov=cov, param_index=[coef_idx] + delta_idx, label=label,
        sd_floor=sd_floor, sigma_se=se_sigma)


def compare_regions(catalogues):
    """Transferability: spread of each term's coefficient across regions."""
    allc = pd.concat(catalogues, ignore_index=True)
    g = allc.groupby(["term", "tier"])
    return g.agg(n_regions=("coef", "size"), mean_coef=("coef", "mean"),
                 between_sd=("coef", "std"),
                 mean_se=("se", "mean")).reset_index().assign(
        heterogeneity_ratio=lambda d: d["between_sd"] / d["mean_se"])
