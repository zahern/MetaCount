"""pavement_bayes.py
====================================================================
Export a searched pavement structure to an Approximate Bayesian
Computation (ABC) bundle, run rejection ABC on the exported design, and
compare posterior coefficients and predictive outcomes against the
search point estimates.

Workflow
--------
1. ``export_search_result(result, evaluator, out_dir)`` re-evaluates the
   selected genome and writes ``spec.json`` plus one CSV per cluster with
   the exact design matrix, response, segment id and time used by the
   search (``cluster_k.csv`` for training; ``cluster_k_test.csv`` when a
   hold-out panel is available).
2. ``run_abc(export_dir, ...)`` runs rejection ABC with optional
   local-linear regression adjustment for every cluster using the same
   generative family (OLS / AR(1) / AR(2) / random walk / near-unit-root).
   Posterior draw files ``abc_posterior_cluster_k.csv`` and
   ``abc_summary_cluster_k.json`` are written next to the export.
3. ``compare_coefficients`` and ``compare_outcomes`` tabulate posterior
   means/CIs against the point estimates and posterior-predictive
   one-step errors against the point-model errors.

The module only needs numpy/pandas/scipy, so it runs even when PyMC is
not installed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from .pavement_forms import build_design
    from .temporal_models import _stationary_ar2_covariance
except ImportError:  # flat import (script inside package dir)
    from pavement_forms import build_design
    from temporal_models import _stationary_ar2_covariance

__all__ = [
    "export_search_result",
    "export_structure",
    "run_abc",
    "compare_coefficients",
    "compare_outcomes",
]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def _structure_with_fitted_form(structure, fit, k, K):
    lam = np.asarray(structure["lam"], float).copy()
    shape = list(structure.get("shape", [None] * K))
    if fit is not None:
        if fit.get("lam") is not None:
            lam[k] = np.asarray(fit["lam"], float)
        if fit.get("shape") is not None:
            shape[k] = fit["shape"]
    out = dict(structure)
    out["lam"], out["shape"] = lam, shape
    return out


def _cluster_design(base, frame, st_k, k, level_frame):
    """Design matrix matching the search construction for cluster ``k``."""
    mask = np.asarray(st_k["var_mask"][k], bool)
    names = list(base.cont_vars)
    names_a = [names[j] for j in range(len(names)) if mask[j]]
    lam_full = np.asarray(st_k["lam"][k], float)
    form_id = int(st_k["form"][k])
    lam_a = lam_full[mask] if form_id == 6 else np.ones(int(mask.sum()))
    shape = st_k["shape"][k] if "shape" in st_k else None
    Xc = frame[names].to_numpy(float)
    num = build_design(Xc[:, mask], names_a, form_id, lam_a, shape=shape)
    blocks = []
    for c in base.cat_vars:
        if c not in level_frame.columns or c not in frame.columns:
            continue
        levels = sorted(level_frame[c].astype(str).unique())
        if len(levels) < 2:
            continue
        b = pd.get_dummies(pd.Categorical(
            frame[c].astype(str), categories=levels)).to_numpy(float)
        blocks.append(b[:, 1:])
    return np.hstack([num] + blocks)


def _point_parameters(eval_result, k, model):
    """Normalised point estimates for cluster ``k`` of an EvalResult."""
    rec = (getattr(eval_result, "temporal", None) or {}).get(k)
    fit = (rec or {}).get("fit")
    if fit is None:
        fit = (eval_result.fits or {}).get(k)
    if fit is None:
        return None
    beta = np.asarray(fit.get("coefficients", []), dtype=float)
    resid = np.asarray(fit.get("residuals", []), dtype=float)
    rss = float(fit.get("rss", np.nan))
    n = int(fit.get("n", resid.size if resid.size else 0))
    sigma = None
    for key in ("sigma_ar1", "sigma_ar2", "sigma_rw", "sigma_nur"):
        if fit.get(key) is not None:
            sigma = float(fit[key])
            break
    if sigma is None:
        dof = max(n - max(int(fit.get("n_params_total", len(beta))), 1), 1)
        if np.isfinite(rss):
            sigma = float(np.sqrt(max(rss / dof, 1e-12)))
        elif resid.size:
            sigma = float(np.std(resid))
    params = {
        "beta": beta.tolist(),
        "sigma": sigma,
        "rho": fit.get("rho_ar1", fit.get("rho_nur")),
        "phi1": fit.get("rho_ar2_1"),
        "phi2": fit.get("rho_ar2_2"),
        "drift": fit.get("drift_rw"),
        "mu": fit.get("drift_nur"),
        "model": str(model),
        "n_params": fit.get("n_params_total"),
    }
    return params


def export_structure(structure, eval_result, evaluator, out_dir,
                     test_df=None) -> dict:
    """Write an ABC-ready export bundle for one evaluated structure.

    Parameters
    ----------
    structure : decision dict (decoded genome)
    eval_result : EvalResult from ``PavementDeteriorationEvaluator.evaluate``
    evaluator : PavementMultiObjectiveEvaluator (or an object with ``base``)
    out_dir : directory for ``spec.json`` and ``cluster_*.csv``
    test_df : optional hold-out frame (defaults to ``evaluator.test_df``)

    Returns the spec dict that was written.
    """
    base = getattr(evaluator, "base", evaluator)
    membership = np.asarray(structure["membership"], int)
    K = int(structure["K"])
    id_c, psi_c, t_c = base.id_col, base.psi_col, base.time_col
    if test_df is None:
        test_df = getattr(evaluator, "test_df", None)
    os.makedirs(out_dir, exist_ok=True)

    spec = {
        "id_col": id_c,
        "time_col": t_c,
        "response": psi_c,
        "cont_vars": list(base.cont_vars),
        "cat_vars": list(base.cat_vars),
        "K": K,
        "temporal_models": list(structure.get("temporal_models",
                                              ["ols"] * K)),
        "clusters": [],
    }

    for k in range(K):
        fit_k = (eval_result.fits or {}).get(k)
        st_k = _structure_with_fitted_form(structure, fit_k, k, K)
        segs = base.seg_ids[membership == k]
        d_tr = base.df[base.df[id_c].isin(segs)].copy()
        d_tr = d_tr.sort_values([id_c, t_c], kind="mergesort").reset_index(drop=True)
        X_tr = _cluster_design(base, d_tr, st_k, k, d_tr)
        cols = [f"x{i}" for i in range(X_tr.shape[1])]
        out_tr = pd.DataFrame({
            id_c: d_tr[id_c].to_numpy(),
            t_c: d_tr[t_c].to_numpy(),
            psi_c: d_tr[psi_c].to_numpy(float),
        })
        for i, c in enumerate(cols):
            out_tr[c] = X_tr[:, i]
        tr_file = f"cluster_{k + 1}.csv"
        out_tr.to_csv(os.path.join(out_dir, tr_file), index=False)

        te_file = None
        if test_df is not None and len(test_df):
            d_te = test_df[test_df[id_c].isin(segs)].copy()
            if len(d_te):
                d_te = d_te.sort_values([id_c, t_c],
                                        kind="mergesort").reset_index(drop=True)
                X_te = _cluster_design(base, d_te, st_k, k, d_tr)
                out_te = pd.DataFrame({
                    id_c: d_te[id_c].to_numpy(),
                    t_c: d_te[t_c].to_numpy(),
                    psi_c: d_te[psi_c].to_numpy(float),
                })
                for i, c in enumerate(cols):
                    out_te[c] = X_te[:, i]
                te_file = f"cluster_{k + 1}_test.csv"
                out_te.to_csv(os.path.join(out_dir, te_file), index=False)

        model = spec["temporal_models"][k]
        spec["clusters"].append({
            "cluster": k + 1,
            "temporal_model": model,
            "form": int(structure["form"][k]),
            "mask": [bool(v) for v in np.asarray(structure["var_mask"][k])],
            "x_columns": cols,
            "n_rows": int(len(out_tr)),
            "point_parameters": _point_parameters(eval_result, k, model),
            "file": tr_file,
            "test_file": te_file,
        })

    with open(os.path.join(out_dir, "spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    return spec


def export_search_result(result, evaluator, out_dir, index=0,
                         test_df=None) -> dict:
    """Export the selected model of a multi-objective pavement search.

    ``result`` may be the dict returned by
    ``run_pavement_multiobjective_search`` (a Pareto ``front`` is expected),
    a decoded structure dict, or a genome vector.
    """
    if isinstance(result, dict) and "front" in result:
        structure = result["front"][int(index)]
    elif isinstance(result, dict) and "K" in result:
        structure = result
    else:
        structure = evaluator.decode(np.asarray(result, int))
    res = evaluator.base.evaluate(structure)
    return export_structure(structure, res, evaluator, out_dir,
                            test_df=test_df)


def load_export(export_dir):
    """Load ``spec.json`` plus the per-cluster frames written by the export."""
    with open(os.path.join(export_dir, "spec.json"), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    frames, test_frames = {}, {}
    for cl in spec["clusters"]:
        k = cl["cluster"]
        frames[k] = pd.read_csv(os.path.join(export_dir, cl["file"]))
        if cl.get("test_file"):
            test_frames[k] = pd.read_csv(
                os.path.join(export_dir, cl["test_file"]))
    return spec, frames, test_frames


# ---------------------------------------------------------------------------
# ABC engine
# ---------------------------------------------------------------------------
_MODELS = ("ols", "ar1", "ar2", "random_walk", "nur")


def _segment_slices(seg_ids):
    order = np.unique(seg_ids)
    return [np.where(seg_ids == s)[0] for s in order]


def _ols(X, y):
    Xd = np.hstack([np.ones((len(y), 1)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    return beta, resid


def _summaries(model, X, y, seg_ids):
    """Summary statistics used by the ABC distance, per error structure."""
    slices = _segment_slices(seg_ids)
    if model == "random_walk":
        rows_d, rows_x = [], []
        for idx in slices:
            if len(idx) < 2:
                continue
            rows_d.append(np.diff(y[idx]))
            rows_x.append(X[idx[1:]] - X[idx[:-1]])
        if not rows_d:
            return np.full(X.shape[1] + 2, np.nan)
        dy = np.concatenate(rows_d)
        DX = np.concatenate(rows_x, axis=0)
        beta, resid = _ols(DX, dy)
        stats = [beta, np.std(resid)]
    else:
        beta, resid = _ols(X, y)
        ac = []
        for lag in (1, 2):
            num, den = 0.0, 0.0
            for idx in slices:
                r = resid[idx]
                if len(r) > lag:
                    num += float(np.sum(r[lag:] * r[:-lag]))
                den += float(np.sum(r ** 2))
            ac.append(num / den if den > 0 else 0.0)
        stats = [beta, np.std(resid)]
        if model in ("ar1", "nur"):
            stats.append(ac[0])
        elif model == "ar2":
            stats.extend(ac)
    return np.concatenate([np.atleast_1d(s).ravel() for s in stats])


def _draw_prior(model, m, y, rng, prior_sd):
    sd = float(prior_sd)
    if model == "random_walk":
        return {
            "drift": rng.normal(0.0, sd),
            "beta": rng.normal(0.0, sd, size=m),
            "sigma": abs(rng.normal(0.0, sd)),
        }
    params = {
        "beta": rng.normal(0.0, sd, size=m + 1),
        "sigma": abs(rng.normal(0.0, sd)),
    }
    if model == "ar1":
        params["rho"] = rng.uniform(-0.99, 0.99)
    elif model == "ar2":
        for _ in range(100):
            p1 = rng.uniform(-0.99, 0.99)
            p2 = rng.uniform(-0.99, 0.99)
            if abs(p2) < 1 and p1 + p2 < 1 and p2 - p1 < 1:
                break
        params["phi1"], params["phi2"] = p1, p2
    elif model == "nur":
        params["rho"] = rng.uniform(0.85, 0.999)
        params["mu"] = rng.normal(float(np.mean(y)), sd)
    return params


def _simulate(model, X, y, seg_ids, params, rng):
    """Simulate a response under the exported model family."""
    slices = _segment_slices(seg_ids)
    if model == "random_walk":
        out = np.empty(len(y))
        drift = float(params["drift"])
        b = np.asarray(params["beta"], float)
        sigma = float(params["sigma"])
        for idx in slices:
            out[idx[0]] = rng.normal(0.0, sigma)
            for t in range(1, len(idx)):
                dx = X[idx[t]] - X[idx[t - 1]]
                out[idx[t]] = (out[idx[t - 1]] + drift + float(dx @ b)
                               + rng.normal(0.0, sigma))
        return out

    beta = np.asarray(params["beta"], float)
    mu_x = np.hstack([np.ones((len(y), 1)), X]) @ beta
    sigma = float(params["sigma"])
    out = np.empty(len(y))
    if model == "ols":
        out[:] = mu_x + rng.normal(0.0, sigma, size=len(y))
        return out
    if model == "ar1":
        rho = float(params["rho"])
        for idx in slices:
            u0 = rng.normal(0.0, sigma / np.sqrt(max(1.0 - rho ** 2, 1e-6)))
            out[idx[0]] = mu_x[idx[0]] + u0
            for t in range(1, len(idx)):
                out[idx[t]] = mu_x[idx[t]] + rho * (out[idx[t - 1]]
                                                    - mu_x[idx[t - 1]]) \
                    + rng.normal(0.0, sigma)
        return out
    if model == "ar2":
        phi1, phi2 = float(params["phi1"]), float(params["phi2"])
        cov = _stationary_ar2_covariance(phi1, phi2, sigma ** 2, 2)
        for idx in slices:
            u = np.asarray(rng.multivariate_normal(np.zeros(2), cov))
            out[idx[0]] = mu_x[idx[0]] + u[0]
            if len(idx) > 1:
                out[idx[1]] = mu_x[idx[1]] + u[1]
            for t in range(2, len(idx)):
                out[idx[t]] = mu_x[idx[t]] + phi1 * (out[idx[t - 1]]
                                                     - mu_x[idx[t - 1]]) \
                    + phi2 * (out[idx[t - 2]] - mu_x[idx[t - 2]]) \
                    + rng.normal(0.0, sigma)
        return out
    rho = float(params["rho"])
    mu = float(params["mu"])
    for idx in slices:
        out[idx[0]] = mu + mu_x[idx[0]] + rng.normal(
            0.0, sigma / np.sqrt(max(1.0 - rho ** 2, 1e-6)))
        for t in range(1, len(idx)):
            out[idx[t]] = rho * out[idx[t - 1]] + (1.0 - rho) * mu \
                + mu_x[idx[t]] + rng.normal(0.0, sigma)
    return out


def _prior_columns(model):
    if model == "random_walk":
        return ["drift", "beta", "sigma"]
    cols = ["beta", "sigma"]
    if model == "ar1":
        cols.append("rho")
    elif model == "ar2":
        cols.extend(["phi1", "phi2"])
    elif model == "nur":
        cols.extend(["rho", "mu"])
    return cols


def _flatten_params(model, params):
    if model == "random_walk":
        return np.concatenate([[params["drift"]], params["beta"],
                               [params["sigma"]]])
    cols = list(params["beta"]) + [params["sigma"]]
    if model == "ar1":
        cols.append(params["rho"])
    elif model == "ar2":
        cols.extend([params["phi1"], params["phi2"]])
    elif model == "nur":
        cols.extend([params["rho"], params["mu"]])
    return np.asarray(cols, dtype=float)


def _posterior_names(model, m):
    names = (["drift"] if model == "random_walk" else ["beta_0"])
    names += [f"x{i}" for i in range(m)]
    names.append("sigma")
    if model == "ar1":
        names.append("rho")
    elif model == "ar2":
        names.extend(["phi1", "phi2"])
    elif model == "nur":
        names.extend(["rho", "mu"])
    return names


def run_abc(export_dir, out_dir=None, n_sims: int = 3000,
            n_posterior: int = 300, prior_sd: Optional[float] = None,
            adjustment: str = "linear", seed: int = 0,
            clusters=None) -> dict:
    """Rejection ABC (optional local-linear adjustment) on an export bundle.

    ``prior_sd`` is the scale of the coefficient/drift/mu priors. When None
    (default) it is set to ``5 * std(y)`` so the prior comfortably covers
    the observed response scale; pass a value to override.
    """
    spec, frames, _ = load_export(export_dir)
    out_dir = out_dir or export_dir
    rng = np.random.default_rng(seed)
    posteriors, summaries = {}, {}

    for cl in spec["clusters"]:
        k = cl["cluster"]
        if clusters is not None and k not in clusters:
            continue
        model = str(cl["temporal_model"])
        if model not in _MODELS:
            raise ValueError(f"unknown temporal model {model!r}")
        frame = frames[k]
        id_c, t_c, psi_c = spec["id_col"], spec["time_col"], spec["response"]
        seg_ids = frame[id_c].to_numpy()
        y = frame[psi_c].to_numpy(float)
        xcols = cl["x_columns"]
        X = frame[xcols].to_numpy(float)
        m = X.shape[1]

        sd_k = (float(prior_sd) if prior_sd is not None
                else 5.0 * max(float(np.std(y)), 1.0))
        s_obs = _summaries(model, X, y, seg_ids)
        sims, thetas = [], []
        for i in range(int(n_sims)):
            params = _draw_prior(model, m, y, rng, sd_k)
            y_sim = _simulate(model, X, y, seg_ids, params, rng)
            s = _summaries(model, X, y_sim, seg_ids)
            if not np.all(np.isfinite(s)):
                continue
            sims.append(s)
            thetas.append(_flatten_params(model, params))
        if not sims:
            raise RuntimeError(f"cluster {k}: no valid ABC simulations")
        S = np.asarray(sims, float)
        T = np.asarray(thetas, float)
        scale = S.std(axis=0)
        scale[scale < 1e-12] = 1.0
        S_std = (S - S.mean(axis=0)) / scale
        s_obs_std = (s_obs - S.mean(axis=0)) / scale
        dist = np.sqrt(np.sum((S_std - s_obs_std) ** 2, axis=1))
        n_acc = min(int(n_posterior), len(dist))
        acc = np.argsort(dist)[:n_acc]
        S_acc, T_acc = S_std[acc], T[acc]

        if adjustment == "linear":
            Sb = S_acc - S_acc.mean(axis=0)
            Tb = T_acc - T_acc.mean(axis=0)
            A = np.hstack([np.ones((len(Sb), 1)), Sb])
            coef, *_ = np.linalg.lstsq(A, T_acc, rcond=None)
            T_post = T_acc + (s_obs_std - S_acc.mean(axis=0)) @ coef[1:]
        else:
            T_post = T_acc

        names = _posterior_names(model, m)
        post = pd.DataFrame(T_post, columns=names)
        post.insert(0, "cluster", k)
        post.to_csv(os.path.join(out_dir, f"abc_posterior_cluster_{k}.csv"),
                    index=False)
        summary = {
            "cluster": k,
            "temporal_model": model,
            "n_sims": int(n_sims),
            "n_posterior": int(n_acc),
            "adjustment": adjustment,
            "prior_sd": sd_k,
            "parameters": {},
        }
        for name in names:
            vals = post[name].to_numpy(float)
            summary["parameters"][name] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals)),
                "q025": float(np.percentile(vals, 2.5)),
                "q50": float(np.percentile(vals, 50)),
                "q975": float(np.percentile(vals, 97.5)),
            }
        with open(os.path.join(out_dir, f"abc_summary_cluster_{k}.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        posteriors[k] = post
        summaries[k] = summary

    result = {"spec": spec, "posterior": posteriors, "summary": summaries,
              "out_dir": out_dir}
    with open(os.path.join(out_dir, "abc_run.json"), "w",
              encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in summaries.items()}, fh, indent=2)
    return result


# ---------------------------------------------------------------------------
# predictions and comparison
# ---------------------------------------------------------------------------
def _point_predict(frame, model, params, id_c, time_col, psi_c):
    """One-step expected predictions under a parameter set."""
    seg_ids = frame[id_c].to_numpy()
    y = frame[psi_c].to_numpy(float)
    X = frame[[c for c in frame.columns
               if c.startswith("x")]].to_numpy(float)
    slices = _segment_slices(seg_ids)
    if model == "random_walk":
        b = np.asarray(params["beta"], float)
        drift = float(params["drift"])
        pred = np.empty(len(y))
        for idx in slices:
            pred[idx[0]] = drift + float(X[idx[0]] @ b)
            for t in range(1, len(idx)):
                dx = X[idx[t]] - X[idx[t - 1]]
                pred[idx[t]] = y[idx[t - 1]] + drift + float(dx @ b)
        return pred
    Xd = np.hstack([np.ones((len(y), 1)), X])
    beta = np.asarray(params["beta"], float)
    Xb = Xd @ beta
    pred = np.empty(len(y))
    if model == "ols":
        return Xb
    if model in ("ar1", "ar2"):
        phi1 = float(params.get("rho", params.get("phi1", 0.0)))
        phi2 = float(params.get("phi2", 0.0)) if model == "ar2" else 0.0
        for idx in slices:
            pred[idx[0]] = Xb[idx[0]]
            for t in range(1, len(idx)):
                p = t - 1
                val = Xb[idx[t]] + phi1 * (y[idx[p]] - Xb[idx[p]])
                if model == "ar2" and t >= 2:
                    val += phi2 * (y[idx[t - 2]] - Xb[idx[t - 2]])
                pred[idx[t]] = val
        return pred
    rho = float(params["rho"])
    mu = float(params["mu"])
    for idx in slices:
        pred[idx[0]] = Xb[idx[0]]
        for t in range(1, len(idx)):
            pred[idx[t]] = rho * y[idx[t - 1]] + (1.0 - rho) * mu + Xb[idx[t]]
    return pred


def _point_params_from_spec(cl):
    pt = cl.get("point_parameters") or {}
    beta = np.asarray(pt.get("beta", []), float)
    model = str(cl["temporal_model"])
    out = {"beta": beta if model == "random_walk" else beta,
           "sigma": pt.get("sigma")}
    if model == "ar1":
        out["rho"] = pt.get("rho")
    elif model == "ar2":
        out["phi1"] = pt.get("phi1")
        out["phi2"] = pt.get("phi2")
    elif model == "nur":
        out["rho"] = pt.get("rho")
        out["mu"] = pt.get("mu")
    elif model == "random_walk":
        out["drift"] = pt.get("drift")
        out["beta"] = beta[1:] if beta.size else beta
    return out


def compare_coefficients(export_dir, out_csv=None) -> pd.DataFrame:
    """Posterior coefficient/parameter intervals vs search point estimates."""
    spec, _, _ = load_export(export_dir)
    rows = []
    for cl in spec["clusters"]:
        k = cl["cluster"]
        post_file = os.path.join(export_dir, f"abc_posterior_cluster_{k}.csv")
        if not os.path.exists(post_file):
            continue
        post = pd.read_csv(post_file)
        pt = _point_params_from_spec(cl)
        model = str(cl["temporal_model"])
        for name, values in post.items():
            if name == "cluster":
                continue
            if name.startswith("x"):
                j = int(name[1:])
                point = (pt["beta"][j] if model == "random_walk"
                         else pt["beta"][j + 1] if j + 1 < len(pt["beta"])
                         else np.nan)
            elif name == "beta_0":
                point = pt["beta"][0] if len(pt["beta"]) else np.nan
            elif name == "drift":
                point = pt.get("drift", np.nan)
            else:
                point = pt.get(name, np.nan)
            vals = values.to_numpy(float)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({
                "cluster": k,
                "parameter": name,
                "point_estimate": point,
                "posterior_mean": float(np.mean(vals)),
                "posterior_sd": float(np.std(vals)),
                "ci_2.5": float(lo),
                "ci_97.5": float(hi),
                "inside_95ci": bool(lo <= point <= hi)
                if point is not None and np.isfinite(point) else None,
                "z_point_vs_post": (float((point - np.mean(vals))
                                          / max(np.std(vals), 1e-12))
                                    if point is not None
                                    and np.isfinite(point) else None),
            })
    df = pd.DataFrame(rows)
    out_csv = out_csv or os.path.join(export_dir, "coeff_comparison.csv")
    df.to_csv(out_csv, index=False)
    return df


def compare_outcomes(export_dir, out_csv=None, n_draws: int = 200,
                     seed: int = 0) -> pd.DataFrame:
    """One-step predictive error: point model vs ABC posterior predictive."""
    spec, frames, test_frames = load_export(export_dir)
    id_c, t_c, psi_c = spec["id_col"], spec["time_col"], spec["response"]
    rng = np.random.default_rng(seed)
    rows = []
    for cl in spec["clusters"]:
        k = cl["cluster"]
        post_file = os.path.join(export_dir, f"abc_posterior_cluster_{k}.csv")
        if not os.path.exists(post_file):
            continue
        post = pd.read_csv(post_file)
        model = str(cl["temporal_model"])
        pt = _point_params_from_spec(cl)
        splits = [("train", frames[k])]
        if k in test_frames:
            splits.append(("test", test_frames[k]))
        for split, frame in splits:
            frame = frame.sort_values([id_c, t_c],
                                      kind="mergesort").reset_index(drop=True)
            y = frame[psi_c].to_numpy(float)
            pred_pt = _point_predict(frame, model, pt, id_c, t_c, psi_c)
            draws = post.sample(min(int(n_draws), len(post)),
                                random_state=int(rng.integers(1 << 31)))
            preds = []
            for _, row in draws.iterrows():
                params = _params_from_draw(model, row, cl)
                preds.append(_point_predict(frame, model, params,
                                            id_c, t_c, psi_c))
            preds = np.asarray(preds, float)
            pred_mean = preds.mean(axis=0)
            rmse_draws = np.sqrt(np.mean((preds - y) ** 2, axis=1))
            rows.append({
                "cluster": k,
                "temporal_model": model,
                "split": split,
                "n": int(len(y)),
                "rmse_point": float(np.sqrt(np.mean((pred_pt - y) ** 2))),
                "mae_point": float(np.mean(np.abs(pred_pt - y))),
                "rmse_post_mean": float(np.sqrt(np.mean((pred_mean - y) ** 2))),
                "mae_post_mean": float(np.mean(np.abs(pred_mean - y))),
                "corr_point_postmean": float(np.corrcoef(pred_pt,
                                                         pred_mean)[0, 1])
                if len(y) > 1 else np.nan,
                "rmse_draw_mean": float(np.mean(rmse_draws)),
                "rmse_draw_median": float(np.median(rmse_draws)),
                "rmse_draw_2.5": float(np.percentile(rmse_draws, 2.5)),
                "rmse_draw_97.5": float(np.percentile(rmse_draws, 97.5)),
            })
    df = pd.DataFrame(rows)
    out_csv = out_csv or os.path.join(export_dir, "outcome_comparison.csv")
    df.to_csv(out_csv, index=False)
    return df


def _params_from_draw(model, row, cl):
    m = len(cl["x_columns"])
    beta_cols = (["x" + str(i) for i in range(m)])
    if model == "random_walk":
        return {
            "drift": float(row["drift"]),
            "beta": row[beta_cols].to_numpy(float),
            "sigma": float(row["sigma"]),
        }
    beta = np.concatenate([[float(row["beta_0"])],
                           row[beta_cols].to_numpy(float)])
    out = {"beta": beta, "sigma": float(row["sigma"])}
    if model == "ar1":
        out["rho"] = float(row["rho"])
    elif model == "ar2":
        out["phi1"] = float(row["phi1"])
        out["phi2"] = float(row["phi2"])
    elif model == "nur":
        out["rho"] = float(row["rho"])
        out["mu"] = float(row["mu"])
    return out
