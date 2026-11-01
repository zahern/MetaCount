"""Turn a fitted two-tier CMF model into a transferable catalogue table."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .GA_CMF_AADT_JAX import SIGMA_FLAT_TOL, sigma_is_flat, _param_labels


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
                        region=None, z=1.96):
    """One row per term.

    Upper tier: CMF = exp(alpha * basis) with a Wald interval on alpha; the
    basis is one unit, except for wide-range count variables where a power of
    ten (``basis_scale``) is used so the CMF is readable (``cmf_basis`` says
    which, and ``coef`` stays per unit with ``coef_per_basis`` alongside).
    Lower tier: the term shifts the AADT elasticity, so the CMF depends on
    traffic; ``cmf_aadt_doubling`` is 2**beta, the multiplicative effect of a
    one-unit change in the variable when AADT doubles.
    Random terms also report the population spread exp(alpha +/- sigma) and the
    share of sites where the effect is protective (CMF < 1).
    ``sigma_collapsed`` marks a selected random term whose sigma cannot be
    separated from zero (``sigma_is_flat``: below ``SIGMA_FLAT_TOL``, or smaller
    than ``z`` standard errors): the sign of the mean is uniform across sites, so
    the spread is not heterogeneity and ``share_protective`` is left empty
    rather than reported as 0 or 1.  ``sigma_se`` and ``sigma_z`` (= sigma / SE)
    are reported so a reader can check that judgement, and ``sigma_flat_abs``
    says whether the estimate collapsed onto the mean rather than merely being
    imprecise.
    ``min``/``max`` give the applicability range seen in the estimation data.
    """
    params = np.asarray(params, dtype=float)
    se = np.asarray(se, dtype=float)
    labels = _param_labels(selected_baseline, selected_local,
                           rand_baseline, rand_local, model)
    idx = {l: i for i, l in enumerate(labels)}
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
                                     float(norm.cdf(-b / sig)) if sig > 0 else
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
    out.attrs["aadt_range"] = (float(data["AADT"].min()), float(data["AADT"].max()))
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
    multiplicatively (their log effects add), with covariances respected.
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
    return {"cmf": float(np.exp(log_cmf)),
            "cmf_lo": float(np.exp(log_cmf - z * se)),
            "cmf_hi": float(np.exp(log_cmf + z * se)),
            "se_log_cmf": se,
            "cmf_se": float(np.exp(log_cmf) * se),
            "significant": bool(se > 0 and abs(log_cmf) / se > z)}


def compare_regions(catalogues):
    """Transferability: spread of each term's coefficient across regions."""
    allc = pd.concat(catalogues, ignore_index=True)
    g = allc.groupby(["term", "tier"])
    return g.agg(n_regions=("coef", "size"), mean_coef=("coef", "mean"),
                 between_sd=("coef", "std"),
                 mean_se=("se", "mean")).reset_index().assign(
        heterogeneity_ratio=lambda d: d["between_sd"] / d["mean_se"])
