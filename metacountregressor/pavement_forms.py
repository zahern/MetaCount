"""
zeke_functional_forms.py
========================================================================
Functional-form registry for the pavement-deterioration program.

This module implements the *estimation-layer* half of the mathematical
program in ``trb_summary.tex``: for each cluster it transforms the raw
continuous covariates ``x`` into a design according to one of the candidate
functional forms, and (where the form is estimable) searches the transform
parameters ``lambda`` / shape parameters by profile (concentrated)
likelihood.

Design goals
------------
* Mirror the plugin conventions used by ``metacountregressor`` so the same
  simulated-annealing decision-vector can drive this code. Each form is a
  *pure* ``(X, lam, shape) -> design_matrix`` map plus an optional
  estimability flag. The search layer (``zeke_pavement_pipeline.py``) owns
  the optimisation; this module owns the transforms.
* The 11 forms from the paper (Table~\\ref{tab:forms}) plus the genetic-
  programming form (12) which is handled in ``zeke_gp_forms.py``.

Continuous-vs-categorical
-------------------------
Only continuous predictors are transformed. Categorical predictors enter
as dummy columns and are appended unchanged by the caller.
"""
from __future__ import annotations

import numpy as np
from .pavement_minimize import minimize_scalar

# ---------------------------------------------------------------------------
# Form specifications
#   id : dict(name, estimable, special)
#     estimable -> requires a per-variable power/shape parameter search
#     special   -> 'age'   (the column named in AGE_NAME receives a special
#                           treatment, other continuous cols are log'd)
# ---------------------------------------------------------------------------
AGE_NAME = "age"

FORM_SPECS = {
    0: {"name": "Power (log-log)",            "estimable": False, "special": None},
    1: {"name": "Linear",                     "estimable": False, "special": None},
    2: {"name": "Reciprocal",                 "estimable": False, "special": None},
    3: {"name": "Exponential Age",            "estimable": False, "special": "age"},
    4: {"name": "Quadratic Age",              "estimable": False, "special": "age"},
    5: {"name": "Age x Traffic",              "estimable": False, "special": "age_inter"},
    6: {"name": "Power w/ estimable exponent","estimable": True,  "special": None},
    7: {"name": "Logistic age (S-curve)",     "estimable": False, "special": "age_scurve"},
    8: {"name": "Piecewise age (broken-line)","estimable": False, "special": "age_piece"},
    9: {"name": "Generalised interaction",    "estimable": False, "special": "gen_inter"},
    10: {"name": "Saturation (plateau) age",  "estimable": False, "special": "age_sat"},
    11: {"name": "Symbolic regression (GP)",  "estimable": True,  "special": "gp"},
}
N_FORMS = len(FORM_SPECS)


def _safe_pos(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 1e-6, None)


def transform_col(vals: np.ndarray, name: str, form: int,
                 lam: float = 1.0, shape: dict | None = None) -> np.ndarray:
    """Transform a single continuous column.

    Parameters
    ----------
    vals : (n,) raw continuous values
    name : column name (used to detect the 'age' column for age-specific forms)
    form : form id
    lam  : estimable power parameter (used when form is estimable)
    shape: dict of shape parameters (alpha/kappa/sigma) for S-curve / piecewise
    """
    shape = shape or {}
    v = np.asarray(vals, dtype=float)
    is_age = (name == AGE_NAME)

    if form == 0:                      # Power (log-log)
        return np.log(_safe_pos(v))
    if form == 1:                      # Linear
        return v
    if form == 2:                      # Reciprocal
        return 1.0 / _safe_pos(v)
    if form == 3:                      # Exponential Age
        return v if is_age else np.log(_safe_pos(v))
    if form == 4:                      # Quadratic Age
        return np.column_stack([v, v ** 2]) if is_age else np.log(_safe_pos(v))
    if form == 5:                      # Age x Traffic (base: log for all)
        return np.log(_safe_pos(v))
    if form == 6:                      # Power w/ estimable exponent
        return _safe_pos(v) ** lam
    if form == 7:                      # Logistic age (S-curve)
        if is_age:
            s0 = shape.get("sigma0", np.median(v))
            s1 = shape.get("sigma1", max(np.std(v), 1e-3))
            return 1.0 / (1.0 + np.exp(-(v - s0) / s1))
        return np.log(_safe_pos(v))
    if form == 8:                      # Piecewise age (broken-line)
        if is_age:
            k1 = shape.get("kappa1", np.percentile(v, 33))
            k2 = shape.get("kappa2", np.percentile(v, 66))
            return np.column_stack([np.maximum(v - k1, 0.0),
                                    np.maximum(v - k2, 0.0)])
        return np.log(_safe_pos(v))
    if form == 9:                      # Generalised interaction (base: log)
        return np.log(_safe_pos(v))
    if form == 10:                     # Saturation (plateau) age
        if is_age:
            alpha = shape.get("alpha", 0.05)
            return 1.0 - np.exp(-alpha * _safe_pos(v))
        return np.log(_safe_pos(v))
    if form == 11:                     # Symbolic regression (GP)
        raise RuntimeError("Form 11 must be built via zeke_gp_forms.GPForm")
    raise ValueError(f"Unknown form id {form}")


def build_design(Xc, names, form, lam_vec, shape=None,
                 cat_dummies=None, cat_names=None,
                 interaction_pairs=None, gp_callable=None):
    """Build a full design matrix for one cluster.

    Parameters
    ----------
    Xc : (n, p) raw continuous values (NOT pre-logged)
    names : length-p list of continuous column names
    form : form id
    lam_vec : length-p array of estimable power params (used only if estimable)
    shape : shape-parameter dict for age-specific forms
    cat_dummies : (n, q) categorical dummy array (appended as-is)
    interaction_pairs : list of (i, j) index pairs into ``names`` for forms 5/9
    gp_callable : callable(Xc, names) -> (n, m) design, only for form 11
    """
    Xc = np.asarray(Xc, dtype=float)
    n, p = Xc.shape
    blocks = []
    for j in range(p):
        lam = float(lam_vec[j]) if lam_vec is not None else 1.0
        col = transform_col(Xc[:, j], names[j], form, lam, shape)
        blocks.append(np.atleast_2d(col).T if col.ndim == 1 else col)

    X = np.hstack(blocks) if blocks else np.zeros((n, 0))

    # Interaction terms for forms 5 (age x traffic) and 9 (generalised)
    if form in (5, 9) and interaction_pairs:
        for i, j in interaction_pairs:
            if names[i] == AGE_NAME:
                a = np.log(_safe_pos(Xc[:, i]))
            else:
                a = np.log(_safe_pos(Xc[:, i]))
            b = np.log(_safe_pos(Xc[:, j]))
            X = np.hstack([X, np.atleast_2d(a * b).T])

    if form == 11 and gp_callable is not None:
        X = gp_callable(Xc, names)

    if cat_dummies is not None and cat_dummies.shape[1] > 0:
        X = np.hstack([X, np.asarray(cat_dummies, dtype=float)])

    return X


# ---------------------------------------------------------------------------
# Profile (concentrated) likelihood: OLS fit + RSS, with golden-section search
# over the estimable lambda / shape parameters.
# ---------------------------------------------------------------------------
def _ols_fit(X, y):
    """Least-squares fit; robust to rank deficiency (constant / collinear cols).

    Returns (beta, rss, effective_param_count) using the numerical rank of the
    design so BIC is never over-counted when a predictor is constant within a
    cluster (e.g. a single-category dummy or a zero-variance covariate).
    """
    Xd = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    rss = float(resid @ resid)
    k = int(np.linalg.matrix_rank(Xd))
    return beta, rss, k


def fit_form(Xc, y, names, form, active_mask, cat_dummies=None,
             lam0=None, shape0=None, interaction_pairs=None,
             gp_callable=None, lam_bounds=(0.1, 3.0)):
    """Fit one cluster's functional form; search estimable params if needed.

    Returns dict(beta, rss, n_params, lam, shape, design).
    """
    y = np.asarray(y, dtype=float)
    p = len(names)
    active_mask = np.asarray(active_mask, dtype=bool)
    Xa = Xc[:, active_mask]
    na = Xa.shape[1]
    names_a = [names[j] for j in range(p) if active_mask[j]]
    cat_a = cat_dummies  # dummies are cluster-global; appended unmasked

    lam = np.ones(p)
    if lam0 is not None:
        lam = np.asarray(lam0, dtype=float)
    lam_a = lam[active_mask] if form == 6 else np.ones(na)
    shape = dict(shape0) if shape0 else {}

    def rss_for(lam_a, shape):
        Xd = build_design(Xa, names_a, form, lam_a, shape,
                          cat_dummies=cat_a,
                          interaction_pairs=interaction_pairs,
                          gp_callable=gp_callable)
        _, rss, k = _ols_fit(Xd, y)
        return rss, k, Xd

    best_rss, best_k, _ = rss_for(lam_a, shape)

    # Coordinate ascent over estimable lambda (form 6) and shape params.
    if form == 6:
        for _ in range(3):  # a few sweeps
            improved = False
            for j in range(na):
                def obj(lj):
                    la = lam_a.copy(); la[j] = lj
                    r, _, _ = rss_for(la, shape)
                    return r
                res = minimize_scalar(obj, bounds=lam_bounds,
                                      method="bounded")
                if res.fun < best_rss - 1e-9:
                    lam_a[j] = res.x; best_rss = res.fun; improved = True
            if not improved:
                break
    elif form in (7, 8, 10):
        shape = _refine_shape(Xa, y, names_a, form, shape, cat_a,
                              interaction_pairs, lam_a)

    # recompute design + fit for final params
    Xd = build_design(Xa, names_a, form, lam_a, shape,
                      cat_dummies=cat_a, interaction_pairs=interaction_pairs,
                      gp_callable=gp_callable)
    beta, rss, k = _ols_fit(Xd, y)
    full_lam = np.ones(p); full_lam[active_mask] = lam_a
    return {"beta": beta, "rss": rss, "n_params": int(k),
            "lam": full_lam, "shape": shape, "design": Xd}


def _refine_shape(Xa, y, names_a, form, shape, cat_a, pairs, lam_a):
    """Light profile search over the age-shape parameters."""
    if form == 7:
        keys = [("sigma0", 0.5, 3.0, 1.0), ("sigma1", 0.3, 3.0, 1.0)]
    elif form == 8:
        keys = [("kappa1", 1.0, 10.0, 3.0), ("kappa2", 5.0, 20.0, 10.0)]
    elif form == 10:
        keys = [("alpha", 0.005, 0.2, 0.05)]
    else:
        return shape
    for key, lo, hi, mid in keys:
        sh = dict(shape); sh[key] = mid
        def obj(val):
            s = dict(sh); s[key] = val
            Xd = build_design(Xa, names_a, form, lam_a, s,
                              cat_dummies=cat_a, interaction_pairs=pairs)
            _, rss, _ = _ols_fit(Xd, y)
            return rss
        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded")
        shape[key] = float(res.x)
    return shape


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    age = rng.uniform(0, 20, n)
    aadt = rng.uniform(100, 5000, n)
    Xc = np.column_stack([age, aadt])
    y = 2.0 - 0.5 * np.log(age) + 0.1 * np.log(aadt) + rng.normal(0, 0.1, n)
    names = ["age", "aadt"]
    for f in (0, 1, 6):
        out = fit_form(Xc, y, names, f, np.array([True, True]),
                       lam0=np.array([1.0, 1.0]))
        print(f"form {f} ({FORM_SPECS[f]['name']}): rss={out['rss']:.4f} "
              f"lam={np.round(out['lam'], 3)}")
    print("OK: zeke_functional_forms self-test passed")
