"""Turn a fitted two-tier CMF model into a transferable catalogue table."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .GA_CMF_AADT_JAX import _param_labels


def build_cmf_catalogue(params, se, data, selected_baseline, selected_local,
                        rand_baseline, rand_local, model="poisson",
                        region=None, z=1.96):
    """One row per term.

    Upper tier: CMF per unit = exp(alpha) with a Wald interval on alpha.
    Lower tier: the term shifts the AADT elasticity, so the CMF depends on
    traffic; ``cmf_aadt_doubling`` is 2**beta, the multiplicative effect of a
    one-unit change in the variable when AADT doubles.
    Random terms also report the population spread exp(alpha +/- sigma) and the
    share of sites where the effect is protective (CMF < 1).
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
            base = 1.0 if tier == "upper" else 2.0
            col = data[v] if v in data.columns else None
            rows.append({
                "term": v, "tier": tier, "region": region,
                "coef": b, "se": s,
                "cmf": base ** b if tier == "lower" else np.exp(b),
                "cmf_lo": (base ** (b - z * s)) if tier == "lower" else np.exp(b - z * s),
                "cmf_hi": (base ** (b + z * s)) if tier == "lower" else np.exp(b + z * s),
                "cmf_basis": "AADT doubling" if tier == "lower" else "per unit",
                "random": bool(is_rand), "sigma": sig,
                "cmf_spread_lo": (np.exp(b - sig) if tier == "upper" else 2.0 ** (b - sig)) if is_rand else np.nan,
                "cmf_spread_hi": (np.exp(b + sig) if tier == "upper" else 2.0 ** (b + sig)) if is_rand else np.nan,
                "share_protective": float(norm.cdf(-b / sig)) if sig > 0 else float(b < 0),
                "significant": bool(s > 0 and abs(b / s) > z),
                "min": float(col.min()) if col is not None else np.nan,
                "max": float(col.max()) if col is not None else np.nan,
            })
    out = pd.DataFrame(rows)
    # Published CMFs are almost always within about 0.3-3; flag anything beyond
    # for a coding/sparsity check, and size the effect over the observed range.
    span = (out["max"] - out["min"]).replace(0, np.nan)
    out["cmf_over_range"] = np.where(out["tier"] == "upper",
                                     np.exp(out["coef"] * span), np.nan)
    out["implausible"] = (out["tier"] == "upper") & (
        (out["cmf"] > 3) | (out["cmf"] < 1 / 3) | (out["cmf_over_range"] > 10))
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
