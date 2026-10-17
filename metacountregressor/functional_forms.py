"""
functional_forms.py
================================================================================
Continuous functional-form discovery for count models.

How a count model represents a term
-----------------------------------
In metacountregressor a covariate enters a model as a *plain dataframe column*:
``build_spec`` maps a decision vector to term names, ``parse_manual_spec``
strips the ``col:dist`` distribution suffix, and the JAX design builder reads
the column.  Role (fixed / random / grouped / ...) and distribution are what
the structure search chooses; the *shape* of the effect -- the functional form
-- is whatever the column already is.

So the way to make the search discover a functional form is to give it a
**basis**: materialise candidate transforms of each continuous variable as
their own columns, offer them all as search candidates, and let the existing
BIC-guided role/distribution search choose which shape wins.  That is exactly
what this module does.  It changes nothing in the likelihood, ``ModelSpec`` or
the solver decision-vector layout, so an existing run is unchanged until the
expanded columns are actually offered as candidates.

Basis
-----
1-D transforms (:data:`TRANSFORMS`): ``log1p``, ``log``, ``sqrt``, ``square``,
``cube``, ``inv`` (protected), ``arcsinh``, ``exp_clip``; plus a piecewise-
linear **hinge** basis (spline-like kinks at quantiles) for monotone-but-non-
linear effects, and **interaction** products for second-order structure.

A fast Poisson-IRLS **BIC screen** (:func:`poisson_bic_screen`) ranks the forms
for one variable so a caller can prune the basis before the full search.

Typical use
-----------
    from metacountregressor import expand_forms
    df2, new_cols = expand_forms(df, ["log_flow_z", "curvature_z"],
                                 include_hinge=True, include_interactions=True)
    variables = base_variables + new_cols          # offer them to the search
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "TRANSFORMS",
    "apply_transform",
    "expand_forms",
    "expand_interactions",
    "poisson_bic_screen",
    "discover_best_form",
    "DEFAULT_FORMS",
]

_EPS = 1e-9
DEFAULT_FORMS = ("log1p", "sqrt", "square", "arcsinh")


def _clean(name: str) -> str:
    """Make a column-safe token out of a form/variable fragment."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_")


def _as_float(values) -> np.ndarray:
    return np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"),
                      dtype=float)


def apply_transform(values, name: str, hinge_knot: float | None = None,
                    center: float | None = None, scale: float | None = None):
    """Apply a named 1-D transform, returning a finite-valued float array.

    ``hinge_knot`` is required for the ``hinge`` form; ``center``/``scale`` are
    optional z-scoring applied *after* the transform (recommended for squared
    and inverse forms so the search sees a comparable scale).
    """
    x = _as_float(values)
    if center is None:
        center = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
    name = str(name).lower()
    with np.errstate(all="ignore"):
        if name in ("no", "none", "identity", "id"):
            f = x
        elif name == "log":
            f = np.log(np.clip(x, _EPS, None))
        elif name == "log1p":
            f = np.log1p(np.clip(x, 0.0, None)) if np.nanmin(x) >= 0 \
                else np.sign(x) * np.log1p(np.abs(x))
        elif name == "sqrt":
            f = np.sign(x) * np.sqrt(np.abs(x))
        elif name == "square":
            f = x * x
        elif name == "cube":
            f = x ** 3
        elif name == "inv":
            f = 1.0 / np.where(np.abs(x) < 1e-6,
                               np.sign(x) * 1e-6 + (1e-6 if x == 0 else 0.0), x)
        elif name == "arcsinh":
            f = np.arcsinh(x)
        elif name == "exp_clip":
            f = np.exp(np.clip(x, -30.0, 30.0))
        elif name == "hinge":
            if hinge_knot is None:
                raise ValueError("hinge transform needs hinge_knot")
            f = np.maximum(x - float(hinge_knot), 0.0)
        else:
            raise ValueError(f"unknown transform {name!r}")
    if scale is None:
        sd = float(np.nanstd(f)) if np.isfinite(f).any() else 0.0
        scale = sd if sd > _EPS else 1.0
    out = (f - center) / scale
    return np.where(np.isfinite(out), out, center)


TRANSFORMS = {
    "log": lambda x: apply_transform(x, "log"),
    "log1p": lambda x: apply_transform(x, "log1p"),
    "sqrt": lambda x: apply_transform(x, "sqrt"),
    "square": lambda x: apply_transform(x, "square"),
    "cube": lambda x: apply_transform(x, "cube"),
    "inv": lambda x: apply_transform(x, "inv"),
    "arcsinh": lambda x: apply_transform(x, "arcsinh"),
    "exp_clip": lambda x: apply_transform(x, "exp_clip"),
}


# =====================================================================
# DataFrame expansion
# =====================================================================

def expand_forms(df, variables, forms=DEFAULT_FORMS, include_hinge=False,
                 hinge_knots=(0.25, 0.5, 0.75), include_interactions=False,
                 max_pair_interactions=None, standardise=True) -> tuple:
    """Materialise transformed candidate columns for ``variables``.

    Returns ``(df_expanded, new_columns)`` where ``df_expanded`` is a copy of
    ``df`` with one new column per (variable, form) -- named
    ``<var>__<form>`` (and ``<var>__hinge<q>`` for hinges) -- and
    ``new_columns`` is the list of those names, ready to append to the search's
    candidate variable list.  The original columns are never modified, so the
    identity form remains available and the search can reject every transform.
    """
    out = df.copy()
    new_cols: list[str] = []
    for var in variables:
        if var not in df.columns:
            continue
        vtoken = _clean(var)
        base = _as_float(df[var])
        for form in forms:
            if str(form).lower() in ("no", "none", "identity", "id"):
                continue
            col = f"{vtoken}__{_clean(form)}"
            if col in out.columns or col in new_cols:
                continue
            vals = apply_transform(base, form) if standardise \
                else apply_transform(base, form, center=0.0, scale=1.0)
            out[col] = vals.astype(float)
            new_cols.append(col)
        if include_hinge:
            for q in hinge_knots:
                knot = float(np.nanquantile(base, q))
                col = f"{vtoken}__hinge{_clean(f'{q:g}')}"
                if col in out.columns or col in new_cols:
                    continue
                vals = apply_transform(base, "hinge", hinge_knot=knot) \
                    if standardise else apply_transform(
                        base, "hinge", hinge_knot=knot, center=0.0, scale=1.0)
                out[col] = vals.astype(float)
                new_cols.append(col)
    if include_interactions:
        out, ix_cols = expand_interactions(
            out, variables, max_pairs=max_pair_interactions,
            standardise=standardise)
        new_cols.extend(c for c in ix_cols if c not in new_cols)
    return out, new_cols


def expand_interactions(df, variables, max_pairs=None, standardise=True) -> tuple:
    """Add pairwise product columns for ``variables`` (lower triangle).

    Standardised products capture second-order interaction structure the
    additive structure search cannot otherwise express.  ``max_pairs`` caps the
    number added (cheapest-first by variable order) to bound the candidate
    explosion.
    """
    out = df.copy()
    cols: list[str] = []
    pairs = [(a, b) for i, a in enumerate(variables)
             for b in variables[i + 1:] if a in df.columns and b in df.columns]
    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]
    for a, b in pairs:
        col = f"{_clean(a)}__x__{_clean(b)}"
        if col in out.columns or col in cols:
            continue
        x = apply_transform(df[a], "identity") if standardise \
            else _as_float(df[a])
        y = apply_transform(df[b], "identity") if standardise \
            else _as_float(df[b])
        out[col] = (x * y).astype(float)
        cols.append(col)
    return out, cols


# =====================================================================
# Poisson-BIC screen (fast standalone form ranking)
# =====================================================================

def _poisson_irls(X, y, max_iter=60, tol=1e-9):
    """Poisson log-link MLE by IRLS.  Returns (beta, deviance, converged)."""
    n, k = X.shape
    if y.sum() <= 0:
        return np.zeros(k), np.inf, False
    beta = np.zeros(k)
    beta[0] = np.log(max(float(y.mean()), 1e-6))
    for _ in range(int(max_iter)):
        eta = X @ beta
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        w = np.maximum(mu, 1e-9)
        z = eta + (y - mu) / w
        sw = np.sqrt(w)
        WX = X * sw[:, None]
        Wz = z * sw
        try:
            beta_new, *_ = np.linalg.lstsq(WX, Wz, rcond=None)
        except np.linalg.LinAlgError:
            return beta, np.inf, False
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = np.exp(np.clip(eta, -30.0, 30.0))
    dev = 2.0 * np.sum(
        y * np.log(np.maximum(y, 1e-12) / np.maximum(mu, 1e-12)) - (y - mu))
    return beta, float(max(dev, 0.0)), np.isfinite(dev)


def poisson_bic_screen(df, var, y, forms=None, include_hinge=True,
                       hinge_knots=(0.25, 0.5, 0.75)) -> dict:
    """Rank candidate forms of ``var`` for a Poisson count ``y`` by BIC.

    Each form is fitted as ``log mu = b0 + b1 * f(x)`` (2 params) plus an
    intercept-only baseline; BIC = deviance + k*ln(n).  Returns
    ``{form: {"bic": ..., "beta": (b0, b1)}}`` sorted best (lowest) first.
    This is a *screen* -- the full structure search still makes the final call.
    """
    yv = _as_float(y)
    base = _as_float(df[var])
    ok = np.isfinite(yv) & np.isfinite(base)
    yv, base = yv[ok], base[ok]
    n = len(yv)
    table: dict[str, dict] = {}
    if n < 8:
        return table
    n_param = 2
    cand = {f: (base, f, None, n_param) for f in (forms or DEFAULT_FORMS)}
    if include_hinge:
        # A hinge's knot is chosen from the data, so charge it 3 params
        # (intercept + slope + knot), not 2 -- otherwise a free knot on pure
        # noise can buy an unlucky BIC improvement.
        for q in hinge_knots:
            label = f"hinge{_clean(f'{q:g}')}"
            cand[label] = (base, "hinge",
                           float(np.nanquantile(base, q)), 3)
    # intercept-only baseline
    _, dev0, _ = _poisson_irls(np.ones((n, 1)), yv)
    table["intercept"] = {"bic": float(dev0 + 1 * np.log(n)), "beta": (0.0, 0.0)}
    for form, (xb, tf_name, knot, k_par) in cand.items():
        f = apply_transform(xb, tf_name, hinge_knot=knot)
        m = np.isfinite(f)
        if m.sum() < 8 or np.std(f[m]) <= _EPS:
            continue
        X = np.column_stack([np.ones(m.sum()), f[m]])
        beta, dev, conv = _poisson_irls(X, yv[m])
        if not conv or not np.isfinite(dev):
            continue
        bic = float(dev + k_par * np.log(max(m.sum(), 2)))
        table[form] = {"bic": bic, "beta": (float(beta[0]), float(beta[1]))}
    return dict(sorted(table.items(), key=lambda kv: kv[1]["bic"]))


def discover_best_form(df, var, y, min_delta=6.0, **kwargs) -> tuple:
    """Return ``(best_form, screen_table)`` from :func:`poisson_bic_screen`.

    ``best_form`` is ``None`` when the intercept-only model wins, or when no
    form beats it by at least ``min_delta`` BIC (a margin that stops a pure
    noise variable from being handed a spurious shape).
    """
    table = poisson_bic_screen(df, var, y, **kwargs)
    if not table:
        return None, table
    best = next(iter(table))
    if best == "intercept" or len(table) < 2:
        return None, table
    # Prefer a real form only if it beats the intercept baseline by a margin.
    if table[best]["bic"] >= table.get("intercept", {}).get("bic", np.inf) \
            - float(min_delta):
        return None, table
    return best, table
